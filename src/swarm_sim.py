from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist


# Load inputs
def load_spline(root: Path):
    p = root / "data" / "centerline_spline.npy"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    return np.load(p).astype(np.float32)


def load_dt(root: Path):
    p = root / "data" / "dt_halfwidth.npy"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    return np.load(p).astype(np.float32)


def load_init(root: Path):
    p = root / "data" / "task2_init.npy"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    return np.load(p, allow_pickle=True).item()


def load_mask(root: Path):
    p = root / "data" / "route_mask_inflated.png"
    if not p.exists(): p = root / "data" / "route_mask.png"
    return (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0), p


# Geometry helpers
def build_normals(centerline: np.ndarray):
    M = len(centerline)
    normals = np.zeros_like(centerline, dtype=np.float32)
    for i in range(M):
        i0 = max(0, i - 1)
        i1 = min(M - 1, i + 1)
        t = centerline[i1] - centerline[i0]
        t = t / (np.linalg.norm(t) + 1e-9)
        normals[i] = np.array([-t[1], t[0]], dtype=np.float32)
    return normals


def build_lanes(centerline: np.ndarray, lane_offset: float):
    normals = build_normals(centerline)
    laneA = (centerline + lane_offset * normals).astype(np.float32)
    laneB = (centerline - lane_offset * normals).astype(np.float32)
    return laneA, laneB


def arclength(pts: np.ndarray):
    d = pts[1:] - pts[:-1]
    ds = np.sqrt((d ** 2).sum(axis=1)).astype(np.float32)
    s = np.zeros((len(pts),), dtype=np.float32)
    s[1:] = np.cumsum(ds)
    return s


def project_local(xy: np.ndarray, lane: np.ndarray, idx_hint: int, win: int = 40):
    M = len(lane)
    lo = max(0, idx_hint - win)
    hi = min(M, idx_hint + win + 1)
    seg = lane[lo:hi]
    d2 = np.sum((seg - xy[None, :]) ** 2, axis=1)
    k = int(np.argmin(d2))
    return lane[lo + k], lo + k


# Forces
def pairwise_repulsion(X: np.ndarray, d_safe: float, k_rep: float):
    N = X.shape[0]
    F = np.zeros_like(X, dtype=np.float32)
    eps = 1e-6
    for i in range(N):
        for j in range(i + 1, N):
            dvec = X[i] - X[j]
            dist = float(np.sqrt(dvec[0] ** 2 + dvec[1] ** 2) + eps)
            if dist >= d_safe: continue
            s = (d_safe - dist) / d_safe
            mag = k_rep * (s ** 2) / dist
            push = (mag * dvec).astype(np.float32)
            F[i] += push
            F[j] -= push
    return F


def wall_force_from_dt(xy: np.ndarray, dt_map: np.ndarray, r: float, k_wall: float, margin: float):
    h, w = dt_map.shape
    x = int(np.clip(round(float(xy[0])), 1, w - 2))
    y = int(np.clip(round(float(xy[1])), 1, h - 2))
    d = float(dt_map[y, x])
    thresh = float(r + margin)
    if d >= thresh: return np.zeros(2, dtype=np.float32)

    gx = 0.5 * (float(dt_map[y, x + 1]) - float(dt_map[y, x - 1]))
    gy = 0.5 * (float(dt_map[y + 1, x]) - float(dt_map[y - 1, x]))
    g = np.array([gx, gy], dtype=np.float32)
    gnorm = np.linalg.norm(g)
    if gnorm > 1e-3:
        g /= gnorm
        pen = thresh - d
        return (k_wall * pen * g).astype(np.float32)
    return np.zeros(2, dtype=np.float32)


def pack_spread_pattern(j: int) -> int:
    if j == 0: return 0
    k = (j + 1) // 2
    return k if (j % 2 == 1) else -k


# RK4 Integrator
def rk4_step(X, V, t, dt, accel_fn):
    A1 = accel_fn(X, V, t)
    k1x, k1v = V, A1

    X2 = X + 0.5 * dt * k1x
    V2 = V + 0.5 * dt * k1v
    A2 = accel_fn(X2, V2, t + 0.5 * dt)
    k2x, k2v = V2, A2

    X3 = X + 0.5 * dt * k2x
    V3 = V + 0.5 * dt * k2v
    A3 = accel_fn(X3, V3, t + 0.5 * dt)
    k3x, k3v = V3, A3

    X4 = X + dt * k3x
    V4 = V + dt * k3v
    A4 = accel_fn(X4, V4, t + dt)
    k4x, k4v = V4, A4

    Xn = X + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Vn = V + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Xn, Vn


def main():
    root = Path(__file__).resolve().parents[1]
    spline = load_spline(root)
    dt_map = load_dt(root)
    init = load_init(root)
    route_bin, mask_path = load_mask(root)

    X0 = init["X0"].astype(np.float32)
    V0 = init["V0"].astype(np.float32)
    N_A, N_B = int(init["N_A"]), int(init["N_B"])
    r, d_safe = float(init["r"]), float(init["d_safe"])
    N = N_A + N_B

    # --- initial position check ---
    D0 = cdist(X0, X0)
    np.fill_diagonal(D0, np.inf)
    min_start_dist = np.min(D0)
    print(f"\n[INIT CHECK] Min Pair Distance at T=0: {min_start_dist:.2f} px")
    if min_start_dist < 2.0 * r:
        print("Robots are spawning on top of each other.")
        print("run 'src/task2_swarm_init.py' again before running this.")

    # tuning
    spacing_px = 3.0 * r
    pack_spread_px = 2.0
    v_leader = 18.0
    lane_offset = float(max(6.0, 0.8 * d_safe))

    kp, kd = 35.0, 12.0
    vmax = 25.0
    dt = 0.05

    k_lane = 140.0
    k_rep = 3200.0
    k_wall = 180.0
    wall_margin = 4.0

    # forced cohesion off
    k_coh = 0.0
    k_align = 0.55

    laneA, laneB = build_lanes(spline, lane_offset)
    sA, sB = arclength(laneA), arclength(laneB)

    frames = int(np.ceil((sA[-1] / max(1e-6, v_leader) + 4.0) / dt))

    hints0 = np.zeros((N,), dtype=np.int32)
    for i in range(N):
        lane = laneA if i < N_A else laneB
        d2 = np.sum((lane - X0[i][None, :]) ** 2, axis=1)
        hints0[i] = int(np.argmin(d2))

    s0 = float(sA[hints0[0]])

    print(f"\nRunning Task 2 (RK4, Lanes)...")
    print(f"Mask: {mask_path.name}")
    print(f"Spacing: {spacing_px:.2f}px (Diameter is {2 * r:.2f}px)")
    print(f"Cohesion: {k_coh} (Forced 0.0)")

    traj = np.zeros((frames, N, 2), dtype=np.float32)
    min_pair = np.zeros((frames,), dtype=np.float32)
    any_wall_contact = np.zeros((frames,), dtype=bool)
    any_hard_collision = np.zeros((frames,), dtype=bool)
    any_unsafe = np.zeros((frames,), dtype=bool)  # Re-added for metrics

    X, V = X0.copy(), V0.copy()
    hints_main = hints0.copy()

    def accel_fn(Xc, Vc, t):
        local_hints = hints_main.copy()

        master_s = float(np.clip(s0 + v_leader * t, 0.0, float(sA[-1])))
        alpha = float(np.clip(master_s / max(1e-6, float(sA[-1])), 0.0, 1.0))

        targets = np.zeros_like(Xc)
        P_rails = np.zeros_like(Xc)

        # Averages for Alignment (Cohesion force is 0, so cA/cB unused but safe to keep)
        cA = Xc[:N_A].mean(axis=0)
        cB = Xc[N_A:].mean(axis=0)
        vA = Vc[:N_A].mean(axis=0)
        vB = Vc[N_A:].mean(axis=0)

        for i in range(N):
            if i < N_A:
                lane, s = laneA, sA
                target_s = alpha * float(s[-1]) - float(i) * spacing_px
                target_s = float(np.clip(target_s, 0.0, float(s[-1])))
                t_idx = int(np.clip(np.searchsorted(s, target_s, side="left"), 1, len(lane) - 2))

                tvec = lane[t_idx + 1] - lane[t_idx - 1]
                tvec /= (np.linalg.norm(tvec) + 1e-9)
                nvec = np.array([-tvec[1], tvec[0]])

                targets[i] = lane[t_idx] + (pack_spread_pattern(i) * pack_spread_px) * nvec
            else:
                lane, s = laneB, sB
                j = i - N_A
                target_s = (1.0 - alpha) * float(s[-1]) + float(j) * spacing_px
                target_s = float(np.clip(target_s, 0.0, float(s[-1])))
                t_idx = int(np.clip(np.searchsorted(s, target_s, side="left"), 1, len(lane) - 2))

                tvec = lane[t_idx + 1] - lane[t_idx - 1]
                tvec /= (np.linalg.norm(tvec) + 1e-9)
                nvec = np.array([-tvec[1], tvec[0]])

                targets[i] = lane[t_idx] + (pack_spread_pattern(j) * pack_spread_px) * nvec

            P_rails[i], local_hints[i] = project_local(Xc[i], lane, local_hints[i], win=45)

        A = kp * (targets - Xc) - kd * Vc + k_lane * (P_rails - Xc)
        A[:N_A] += k_coh * (cA - Xc[:N_A]) + k_align * (vA - Vc[:N_A])
        A[N_A:] += k_coh * (cB - Xc[N_A:]) + k_align * (vB - Vc[N_A:])

        A += pairwise_repulsion(Xc, d_safe, k_rep)
        for ii in range(N):
            A[ii] += wall_force_from_dt(Xc[ii], dt_map, r, k_wall, wall_margin)
        return A.astype(np.float32)

    for k in range(frames):
        t = k * dt
        traj[k] = X

        for i in range(N):
            lane = laneA if i < N_A else laneB
            _, hints_main[i] = project_local(X[i], lane, hints_main[i], win=45)

        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        m = float(np.min(D))
        min_pair[k] = m
        any_hard_collision[k] = (m < (2.0 * r))
        any_unsafe[k] = (m < d_safe)

        h_m, w_m = dt_map.shape
        cx = np.clip(X[:, 0].astype(int), 0, w_m - 1)
        cy = np.clip(X[:, 1].astype(int), 0, h_m - 1)
        any_wall_contact[k] = bool(np.any(dt_map[cy, cx] < r))

        X, V = rk4_step(X, V, t, dt, accel_fn)

        speeds = np.linalg.norm(V, axis=1) + 1e-9
        if np.any(speeds > vmax):
            scale = np.minimum(1.0, vmax / speeds)
            V = V * scale[:, None]

    print("\n=== TASK 2 NUMERICAL VALIDATION ===")
    print(f"Min pair distance: {float(np.min(min_pair)):.2f} px")
    print(f"Interaction Rate (< {d_safe:.2f}px): {float(np.mean(any_unsafe) * 100.0):.2f}%")
    print(f"Hard Collision Rate (< {2.0 * r:.2f}px): {float(np.mean(any_hard_collision) * 100.0):.2f}%")
    print(f"Wall Contact Rate: {float(np.mean(any_wall_contact) * 100.0):.2f}%")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(route_bin, cmap="gray", alpha=0.25)
    ax.plot(laneA[:, 0], laneA[:, 1], linewidth=2)
    ax.plot(laneB[:, 0], laneB[:, 1], linewidth=2)

    robots = [plt.Circle((0, 0), r, fill=True, ec='k') for _ in range(N)]
    for i, rob in enumerate(robots):
        rob.set_facecolor('cyan' if i < N_A else 'orange')
        ax.add_patch(rob)

    def update(frame):
        for i in range(N):
            robots[i].center = traj[frame, i]
        return robots

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()