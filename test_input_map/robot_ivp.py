import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree


# -----------------------------
# Load inputs
# -----------------------------
def load_spline(project_root: Path):
    p = project_root / "data" / "centerline_spline.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Step 2)")
    return np.load(p).astype(np.float32), p


def load_dt(project_root: Path):
    """
    Distance transform map (computed on the inflated corridor mask in width_and_corridor.py):
    dt[y,x] = distance (px) to nearest corridor boundary.
    """
    p = project_root / "data" / "dt_halfwidth.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Step 3: width_and_corridor.py)")
    return np.load(p).astype(np.float32), p


def load_mask(project_root: Path, use_inflated: bool = True):
    """
    Loads the corridor mask for visualization.
    If use_inflated=True, it will prefer route_mask_inflated.png (wider corridor),
    falling back to route_mask.png if the inflated one doesn't exist.
    """
    if use_inflated:
        p = project_root / "data" / "route_mask_inflated.png"
        if not p.exists():
            p = project_root / "data" / "route_mask.png"
    else:
        p = project_root / "data" / "route_mask.png"

    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Step 1 / Step 3)")

    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read {p}")
    return (mask > 0), p


def load_corridor_params(project_root: Path):
    p = project_root / "data" / "corridor_params.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Step 3)")
    params = json.loads(p.read_text(encoding="utf-8"))
    return params, p


# -----------------------------
# IVP model
# xdot = v (with saturation)
# vdot = kp (T(t) - x) - kd v + wall_force(x)
# -----------------------------
def rk4_step(
    x,
    v,
    t_idx,
    dt,
    spline_pts,
    kp,
    kd,
    vmax,
    target_idx_fn,
    dt_map=None,
    robot_radius=0.0,
    k_wall=0.0,
    wall_margin=5.0,
):
    def f(xi, vi, ti):
        # Target point on spline (time-varying waypoint)
        T = spline_pts[target_idx_fn(ti)]

        # Base tracking + damping acceleration
        a = kp * (T - xi) - kd * vi

        # -----------------------------
        # NEW: Active wall force using Distance Transform
        # -----------------------------
        if dt_map is not None and k_wall > 0.0:
            h, w = dt_map.shape
            # clamp to safe interior for central differences
            px = int(np.clip(round(float(xi[0])), 1, w - 2))
            py = int(np.clip(round(float(xi[1])), 1, h - 2))

            dist_to_wall = float(dt_map[py, px])
            thresh = float(robot_radius + wall_margin)

            if dist_to_wall < thresh:
                # central differences (0.5 factor)
                gx = 0.5 * (float(dt_map[py, px + 1]) - float(dt_map[py, px - 1]))
                gy = 0.5 * (float(dt_map[py + 1, px]) - float(dt_map[py - 1, px]))

                grad = np.array([gx, gy], dtype=np.float32)
                grad /= (np.linalg.norm(grad) + 1e-9)  # normalize for stability

                pen = thresh - dist_to_wall  # penetration depth into unsafe band
                a += float(k_wall) * pen * grad

        # Velocity saturation on xdot (as in your original)
        speed = np.linalg.norm(vi) + 1e-9
        scale = min(1.0, vmax / speed)
        xdot = vi * scale
        vdot = a
        return xdot, vdot

    k1x, k1v = f(x, v, t_idx)
    k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t_idx + 0.5)
    k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t_idx + 0.5)
    k4x, k4v = f(x + dt * k3x, v + dt * k3v, t_idx + 1.0)

    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_new, v_new


# -----------------------------
# Constraint check:
# dist_to_centerline(x) + r <= w_half_eff
# -----------------------------
def build_centerline_tree(spline_pts):
    return cKDTree(spline_pts)


def corridor_margin(x, tree, spline_pts, w_half_eff, r):
    dist, idx = tree.query(x, k=1)
    margin = (w_half_eff - r) - dist  # >=0 safe
    return float(margin), float(dist), int(idx)


# -----------------------------
# Target schedule along spline
# -----------------------------
def make_target_index_fn(M, frames, mode="ease"):
    if mode == "linear":
        def fn(ti):
            alpha = np.clip(ti / (frames - 1), 0.0, 1.0)
            return int(alpha * (M - 1))
        return fn

    def fn(ti):
        a = np.clip(ti / (frames - 1), 0.0, 1.0)
        a = a * a * (3 - 2 * a)  # smoothstep
        return int(a * (M - 1))

    return fn


# -----------------------------
# Main simulation + animation
# -----------------------------
def main():
    project_root = Path(__file__).resolve().parents[1]

    spline_pts, _ = load_spline(project_root)
    dt_map, _ = load_dt(project_root)  # NEW: DT for wall forces
    route_bin, mask_used = load_mask(project_root, use_inflated=True)
    print("Visualizing corridor mask:", mask_used)
    cparams, _ = load_corridor_params(project_root)

    w_half = float(cparams["w_half_px"])
    r = float(cparams["robot_radius_px"])

    # =============================
    # Controller tuning
    # =============================
    kp = 5.0
    kd = 3.0
    vmax = 40.0

    # NEW: Wall force settings
    USE_WALL_FORCE = True
    k_wall = 120.0         # start modest; increase if you still graze walls
    wall_margin = 5.0      # your "+5" buffer band

    # =========================================
    # Safety buffer (switch on/off)
    # =========================================
    USE_EPS_BUFFER = True
    eps = 0.3  # pixels
    w_half_eff = w_half - eps if USE_EPS_BUFFER else w_half

    if w_half_eff <= r:
        raise ValueError(
            f"Effective corridor too small: w_half_eff={w_half_eff:.2f} <= r={r:.2f}. "
            f"Reduce eps or robot radius."
        )

    # Simulation timeline
    frames = 900
    dt = 0.05

    target_idx_fn = make_target_index_fn(len(spline_pts), frames, mode="ease")

    # Initial state
    x = spline_pts[0].copy()
    v = np.zeros(2, dtype=np.float32)

    tree = build_centerline_tree(spline_pts)

    traj = np.zeros((frames, 2), dtype=np.float32)
    margins = np.zeros((frames,), dtype=np.float32)
    target_traj = np.zeros((frames, 2), dtype=np.float32)

    for k in range(frames):
        traj[k] = x
        margin, dist, idx = corridor_margin(x, tree, spline_pts, w_half_eff, r)
        margins[k] = margin
        target_traj[k] = spline_pts[target_idx_fn(k)]

        x, v = rk4_step(
            x, v, k, dt,
            spline_pts, kp, kd, vmax, target_idx_fn,
            dt_map=dt_map if USE_WALL_FORCE else None,
            robot_radius=r,
            k_wall=k_wall if USE_WALL_FORCE else 0.0,
            wall_margin=wall_margin
        )

    # Report constraint results
    max_violation = float(np.max(np.maximum(0.0, -margins)))
    viol_rate = float(np.mean(margins < 0.0) * 100.0)

    print("\n=== Task 1 (IVP) constraint report ===")
    print(f"w_half = {w_half:.2f} px, robot_radius = {r:.2f} px")
    if USE_EPS_BUFFER:
        print(f"Using effective half-width w_half_eff = {w_half_eff:.2f} px (eps={eps})")
    else:
        print("Using raw half-width (no eps buffer)")
    print(f"kp={kp}, kd={kd}, vmax={vmax}, dt={dt}, frames={frames}")
    if USE_WALL_FORCE:
        print(f"Wall force ON: k_wall={k_wall}, wall_margin={wall_margin}")
    else:
        print("Wall force OFF")
    print(f"Max violation (px): {max_violation:.3f}   (0.0 is ideal)")
    print(f"Violation rate (% frames): {viol_rate:.2f}%")

    # PASS/FAIL metric
    if max_violation == 0.0 and viol_rate == 0.0:
        print("PASS ✅ Robot stayed inside corridor for all frames.")
    else:
        print("NOT PASS YET ⚠️ Minor boundary violations occurred.")
        print("Tips: reduce vmax, increase kd, lower kp, or increase eps slightly.")

    # Animation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(route_bin, cmap="gray", alpha=0.25)
    ax.plot(spline_pts[:, 0], spline_pts[:, 1], linewidth=2)
    ax.set_title("Task 1: Single robot following spline inside corridor (IVP + RK4)")
    ax.axis("off")

    robot_patch = plt.Circle((traj[0, 0], traj[0, 1]), radius=r, fill=False, linewidth=2)
    ax.add_patch(robot_patch)
    target_dot = ax.scatter([target_traj[0, 0]], [target_traj[0, 1]], s=60)

    trace, = ax.plot([], [], linewidth=2)

    def update(frame):
        xx = traj[frame]
        robot_patch.center = (float(xx[0]), float(xx[1]))
        txy = target_traj[frame]
        target_dot.set_offsets([txy[0], txy[1]])
        trace.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])

        # Visual hint if violation
        robot_patch.set_linestyle("--" if margins[frame] < 0 else "-")
        return robot_patch, target_dot, trace

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()

    # Optional save:
    # out = project_root / "outputs" / "task1_robot_ivp.mp4"
    # out.parent.mkdir(exist_ok=True)
    # anim.save(str(out), fps=30, dpi=150)
    # print("Saved:", out)

    if "inflate_px" in cparams:
        print(f"Corridor inflation (inflate_px) = {cparams['inflate_px']} px")


if __name__ == "__main__":
    main()
