import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Load inputs
def load_spline(project_root: Path):
    p = project_root / "data" / "centerline_spline.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Step 2)")
    return np.load(p).astype(np.float32), p


def load_dt(project_root: Path):
    p = project_root / "data" / "dt_halfwidth.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Step 3: width_and_corridor.py)")
    return np.load(p).astype(np.float32), p


def load_mask(project_root: Path, use_inflated: bool = True):
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


# RK4 Solver
def rk4_step(
        x, v, t_idx, dt,
        spline_pts, kp, kd, vmax, target_idx_fn,
        dt_map=None, robot_radius=0.0, k_wall=0.0, wall_margin=5.0
):

    def get_derivatives(xi, vi, ti):
        # 1. Target Attraction
        T = spline_pts[target_idx_fn(ti)]
        a = kp * (T - xi) - kd * vi

        # 2. Wall Repulsion (Active Force)
        if dt_map is not None and k_wall > 0.0:
            h, w = dt_map.shape
            # Clamp to valid image coordinates
            px = int(np.clip(round(float(xi[0])), 1, w - 2))
            py = int(np.clip(round(float(xi[1])), 1, h - 2))

            dist_to_wall = float(dt_map[py, px])
            thresh = float(robot_radius + wall_margin)

            if dist_to_wall < thresh:
                # Finite Central Differences
                gx = 0.5 * (float(dt_map[py, px + 1]) - float(dt_map[py, px - 1]))
                gy = 0.5 * (float(dt_map[py + 1, px]) - float(dt_map[py - 1, px]))
                grad = np.array([gx, gy], dtype=np.float32)

                # MAGNITUDE GATING
                # Only normalize if gradient is significant to avoid noise/singularity
                gnorm = np.linalg.norm(grad)
                if gnorm > 1e-3:
                    grad /= gnorm
                    pen = thresh - dist_to_wall
                    a += float(k_wall) * pen * grad

        # Physics: dX/dt = V, dV/dt = A
        return vi, a

    # RK4 Integration Steps
    k1x, k1v = get_derivatives(x, v, t_idx)
    k2x, k2v = get_derivatives(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t_idx + 0.5)
    k3x, k3v = get_derivatives(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t_idx + 0.5)
    k4x, k4v = get_derivatives(x + dt * k3x, v + dt * k3v, t_idx + 1.0)

    # Update State
    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    # STATE VELOCITY SATURATION
    # Clamping the speed, not acceleration
    speed = np.linalg.norm(v_new)
    if speed > vmax:
        v_new = v_new * (vmax / speed)

    return x_new, v_new


# Target schedule
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



def main():
    project_root = Path(__file__).resolve().parents[1]

    spline_pts, _ = load_spline(project_root)
    dt_map, _ = load_dt(project_root)
    route_bin, mask_used = load_mask(project_root, use_inflated=True)
    print("Visualizing corridor mask:", mask_used)
    cparams, _ = load_corridor_params(project_root)

    w_half = float(cparams["w_half_px"])
    r = float(cparams["robot_radius_px"])

    # Tuning
    kp = 5.0
    kd = 3.0
    vmax = 40.0
    k_wall = 120.0
    wall_margin = 5.0

    frames = 900
    dt = 0.05

    target_idx_fn = make_target_index_fn(len(spline_pts), frames, mode="ease")

    x = spline_pts[0].copy()
    v = np.zeros(2, dtype=np.float32)

    traj = np.zeros((frames, 2), dtype=np.float32)
    margins = np.zeros((frames,), dtype=np.float32)
    target_traj = np.zeros((frames, 2), dtype=np.float32)

    print(f"\nRunning Task 1 ")

    for k in range(frames):
        traj[k] = x

        # Check actual distance to wall using DT map
        h_map, w_map = dt_map.shape
        cx = int(np.clip(round(x[0]), 0, w_map - 1))
        cy = int(np.clip(round(x[1]), 0, h_map - 1))
        dist_to_wall = float(dt_map[cy, cx])

        # Margin > 0 means safe, Margin < 0 means collision
        # This is more accurate than checking distance to spline
        margins[k] = dist_to_wall - r

        target_traj[k] = spline_pts[target_idx_fn(k)]

        # RK4 Step
        x, v = rk4_step(
            x, v, k, dt,
            spline_pts, kp, kd, vmax, target_idx_fn,
            dt_map=dt_map, robot_radius=r, k_wall=k_wall, wall_margin=wall_margin
        )

    # Report results
    min_margin = float(np.min(margins))
    viol_rate = float(np.mean(margins < 0.0) * 100.0)

    print("\n=== Task 1 Validation Report ===")
    print(f"Robot Radius: {r:.2f} px")
    print(f"Min Distance to Wall: {float(np.min(margins) + r):.2f} px")
    print(f"Worst Margin (dist - r): {min_margin:.3f} px (Negative = Collision)")
    print(f"Violation Rate: {viol_rate:.2f}%")

    if viol_rate == 0.0:
        print("PASS Robot stayed inside corridor for all frames.")
    else:
        print("FAIL Boundary violations detected.")

    # Animation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(route_bin, cmap="gray", alpha=0.25)
    ax.plot(spline_pts[:, 0], spline_pts[:, 1], linewidth=2)
    ax.set_title("Task 1: Single Robot IVP")
    ax.axis("off")

    robot_patch = plt.Circle((traj[0, 0], traj[0, 1]), radius=r, fill=False, linewidth=2, color='red')
    ax.add_patch(robot_patch)
    target_dot = ax.scatter([target_traj[0, 0]], [target_traj[0, 1]], s=60, c='blue')
    trace, = ax.plot([], [], 'r-', linewidth=1)

    def update(frame):
        xx = traj[frame]
        robot_patch.center = (float(xx[0]), float(xx[1]))
        txy = target_traj[frame]
        target_dot.set_offsets([txy[0], txy[1]])
        trace.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])

        # Visual feedback for collision
        if margins[frame] < 0:
            robot_patch.set_color('orange')
        else:
            robot_patch.set_color('red')
        return robot_patch, target_dot, trace

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()