from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist


# -----------------------------
# Load inputs
# -----------------------------
def load_spline(root: Path):
    p = root / "data" / "centerline_spline.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run centerline_and_spline.py")
    return np.load(p).astype(np.float32)


def load_dt(root: Path):
    p = root / "data" / "dt_halfwidth.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run width_and_corridor.py")
    return np.load(p).astype(np.float32)


def load_init(root: Path):
    p = root / "data" / "task2_init.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run task2_swarm_init.py first.")
    return np.load(p, allow_pickle=True).item()


def load_mask(root: Path):
    p = root / "data" / "route_mask_inflated.png"
    if not p.exists():
        p = root / "data" / "route_mask.png"
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {p}")
    return (img > 0), p


# -----------------------------
# Geometry helpers
# -----------------------------
def build_normals(centerline: np.ndarray):
    M = len(centerline)
    normals = np.zeros_like(centerline, dtype=np.float32)
    tangents = np.zeros_like(centerline, dtype=np.float32)
    for i in range(M):
        i0 = max(0, i - 1)
        i1 = min(M - 1, i + 1)
        t = centerline[i1] - centerline[i0]
        t = t / (np.linalg.norm(t) + 1e-9)
        tangents[i] = t
        normals[i] = np.array([-t[1], t[0]], dtype=np.float32)
    return tangents, normals


def build_lanes(centerline: np.ndarray, lane_offset: float):
    _, normals = build_normals(centerline)
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
    idx = lo + k
    return lane[idx], idx


# -----------------------------
# Forces
# -----------------------------
def pairwise_repulsion(X: np.ndarray, d_safe: float, k_rep: float):
    """
    Repulsion when distance < d_safe.
    """
    N = X.shape[0]
    F = np.zeros_like(X, dtype=np.float32)
    eps = 1e-6
    for i in range(N):
        for j in range(i + 1, N):
            dvec = X[i] - X[j]
            dist = float(np.sqrt(dvec[0] ** 2 + dvec[1] ** 2) + eps)
            if dist >= d_safe:
                continue
            s = (d_safe - dist) / d_safe
            mag = k_rep * (s ** 2) / dist
            push = (mag * dvec).astype(np.float32)
            F[i] += push
            F[j] -= push
    return F


def wall_force_from_dt(xy: np.ndarray, dt_map: np.ndarray, r: float, k_wall: float, margin: float):
    """
    If dt(x) < r + margin: push inward along normalized gradient of dt (finite differences).
    """
    h, w = dt_map.shape
    x = int(np.clip(round(float(xy[0])), 1, w - 2))
    y = int(np.clip(round(float(xy[1])), 1, h - 2))

    d = float(dt_map[y, x])
    thresh = float(r + margin)
    if d >= thresh:
        return np.zeros(2, dtype=np.float32)

    gx = 0.5 * (float(dt_map[y, x + 1]) - float(dt_map[y, x - 1]))
    gy = 0.5 * (float(dt_map[y + 1, x]) - float(dt_map[y - 1, x]))
    g = np.array([gx, gy], dtype=np.float32)
    g /= (np.linalg.norm(g) + 1e-9)

    pen = thresh - d
    return (k_wall * pen * g).astype(np.float32)


def pack_spread_pattern(j: int) -> int:
    """
    0, +1, -1, +2, -2, ...
    """
    if j == 0:
        return 0
    k = (j + 1) // 2
    return k if (j % 2 == 1) else -k


# -----------------------------
# RK4 Integrator for swarm state
# -----------------------------
def rk4_step(X, V, t, dt, accel_fn):
    # k1
    A1 = accel_fn(X, V, t)
    k1x = V
    k1v = A1

    # k2
    X2 = X + 0.5 * dt * k1x
    V2 = V + 0.5 * dt * k1v
    A2 = accel_fn(X2, V2, t + 0.5 * dt)
    k2x = V2
    k2v = A2

    # k3
    X3 = X + 0.5 * dt * k2x
    V3 = V + 0.5 * dt * k2v
    A3 = accel_fn(X3, V3, t + 0.5 * dt)
    k3x = V3
    k3v = A3

    # k4
    X4 = X + dt * k3x
    V4 = V + dt * k3v
    A4 = accel_fn(X4, V4, t + dt)
    k4x = V4
    k4v = A4

    Xn = X + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Vn = V + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Xn, Vn


# -----------------------------
# Main simulation
# -----------------------------
def main():
    root = Path(__file__).resolve().parents[1]

    spline = load_spline(root)
    dt_map = load_dt(root)
    init = load_init(root)
    route_bin, mask_path = load_mask(root)

    # Initial state
    X0 = init["X0"].astype(np.float32)
    V0 = init["V0"].astype(np.float32)
    N_A = int(init["N_A"])
    N_B = int(init["N_B"])
    r = float(init["r"])
    d_safe = float(init["d_safe"])
    N = N_A + N_B

    # -----------------------------
    # Tunable settings (keep your current config)
    # -----------------------------
    pack_spread_px = 2.0
    v_leader = 18.0

    lane_offset = float(max(6.0, 0.8 * d_safe))

    kp = 35.0
    kd = 12.0
    vmax = 25.0
    dt = 0.05

    k_lane = 140.0
    k_rep = 3200.0
    k_wall = 180.0
    wall_margin = 4.0

    k_coh = 0.25
    k_align = 0.55

    # -----------------------------
    # Build lanes + arc-length
    # -----------------------------
    laneA, laneB = build_lanes(spline, lane_offset)
    sA = arclength(laneA)
    sB = arclength(laneB)

    # auto frames so both sides reach ends, then settle a bit
    T_traverse = float(sA[-1] / max(1e-6, v_leader))
    T_settle = 4.0
    frames = int(np.ceil((T_traverse + T_settle) / dt))

    # projection hints
    hints0 = np.zeros((N,), dtype=np.int32)
    for i in range(N):
        lane = laneA if i < N_A else laneB
        d2 = np.sum((lane - X0[i][None, :]) ** 2, axis=1)
        hints0[i] = int(np.argmin(d2))

    # "master progress" starts based on leader's projection
    s0 = float(sA[hints0[0]])

    # -----------------------------
    # CRITICAL FIX: convoy spacing to avoid within-group overlap
    # - keep the same smooth motion and forces
    # - but robots in the same group do NOT target the same arc-length point
    # -----------------------------
    spacing_px = float(max(12,11))  # >= 2r ensures no overlap

    print("\nRunning Task 2 (RK4, tight-pack + normalized progress) [NO OVERLAP FIX]...")
    print(f"Mask used: {mask_path.name}")
    print(f"N_A={N_A}, N_B={N_B}, r={r:.2f}, d_safe={d_safe:.2f}")
    print(f"lane_offset={lane_offset:.2f}, pack_spread={pack_spread_px:.2f}px, v_leader={v_leader:.2f}px/s")
    print(f"spacing_px={spacing_px:.2f} (convoy spacing)")
    print(f"kp={kp}, kd={kd}, vmax={vmax}, dt={dt}, frames={frames}")
    print(f"k_coh={k_coh}, k_align={k_align}, k_lane={k_lane}, k_rep={k_rep}, k_wall={k_wall}")

    # Storage + metrics
    traj = np.zeros((frames, N, 2), dtype=np.float32)
    min_pair = np.zeros((frames,), dtype=np.float32)
    any_wall_violation = np.zeros((frames,), dtype=bool)
    any_unsafe = np.zeros((frames,), dtype=bool)

    # HARD collision metric (this is what "accident" really means)
    any_hard_collision = np.zeros((frames,), dtype=bool)

    # current state
    X = X0.copy()
    V = V0.copy()
    hints = hints0.copy()

    def accel_fn(Xc: np.ndarray, Vc: np.ndarray, t: float) -> np.ndarray:
        """
        Computes acceleration for all robots at continuous time t.
        Uses normalized progress alpha(t) so RK4 sub-steps are consistent.
        """
        nonlocal hints

        # master arc-length progress (laneA units)
        master_s = s0 + v_leader * t
        master_s = float(np.clip(master_s, 0.0, float(sA[-1])))

        alpha = master_s / max(1e-6, float(sA[-1]))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        targets = np.zeros_like(Xc, dtype=np.float32)
        P_rails = np.zeros_like(Xc, dtype=np.float32)

        cA = Xc[:N_A].mean(axis=0)
        cB = Xc[N_A:].mean(axis=0)
        vA = Vc[:N_A].mean(axis=0)
        vB = Vc[N_A:].mean(axis=0)

        # targets + rails
        for i in range(N):
            if i < N_A:
                lane = laneA
                s = sA

                # Group A: leader goes forward, followers stay behind by spacing_px
                base_s = alpha * float(s[-1])
                target_s = base_s - float(i) * spacing_px
                target_s = float(np.clip(target_s, 0.0, float(s[-1])))

                t_idx = int(np.searchsorted(s, target_s, side="left"))
                t_idx = int(np.clip(t_idx, 1, len(lane) - 2))

                tvec = lane[t_idx + 1] - lane[t_idx - 1]
                tvec = tvec / (np.linalg.norm(tvec) + 1e-9)
                nvec = np.array([-tvec[1], tvec[0]], dtype=np.float32)

                kspread = pack_spread_pattern(i)
                targets[i] = lane[t_idx] + (kspread * pack_spread_px) * nvec

            else:
                lane = laneB
                s = sB

                # Group B: leader goes backward (from end to start)
                j = i - N_A
                base_s = (1.0 - alpha) * float(s[-1])
                target_s = base_s + float(j) * spacing_px
                target_s = float(np.clip(target_s, 0.0, float(s[-1])))

                t_idx = int(np.searchsorted(s, target_s, side="left"))
                t_idx = int(np.clip(t_idx, 1, len(lane) - 2))

                tvec = lane[t_idx + 1] - lane[t_idx - 1]
                tvec = tvec / (np.linalg.norm(tvec) + 1e-9)
                nvec = np.array([-tvec[1], tvec[0]], dtype=np.float32)

                kspread = pack_spread_pattern(j)
                targets[i] = lane[t_idx] + (kspread * pack_spread_px) * nvec

            # rail projection (local)
            P, hints[i] = project_local(Xc[i], lane, hints[i], win=45)
            P_rails[i] = P

        # base accel
        A = kp * (targets - Xc) - kd * Vc + k_lane * (P_rails - Xc)

        # cohesion + alignment per group (kept exactly as you had it)
        A[:N_A] += k_coh * (cA - Xc[:N_A]) + k_align * (vA - Vc[:N_A])
        A[N_A:] += k_coh * (cB - Xc[N_A:]) + k_align * (vB - Vc[N_A:])

        # repulsion + wall (kept exactly as you had it)
        A += pairwise_repulsion(Xc, d_safe=d_safe, k_rep=k_rep)
        for ii in range(N):
            A[ii] += wall_force_from_dt(Xc[ii], dt_map, r=r, k_wall=k_wall, margin=wall_margin)

        return A

    # time loop
    for k in range(frames):
        t = k * dt
        traj[k] = X

        # metrics
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        m = float(np.min(D))
        min_pair[k] = m
        any_unsafe[k] = (m < d_safe)
        any_hard_collision[k] = (m < (2.0 * r))

        h_m, w_m = dt_map.shape
        cx = np.clip(X[:, 0].astype(int), 0, w_m - 1)
        cy = np.clip(X[:, 1].astype(int), 0, h_m - 1)
        any_wall_violation[k] = bool(np.any(dt_map[cy, cx] < r))

        # RK4 integrate
        X, V = rk4_step(X, V, t, dt, accel_fn)

        # speed cap (keep same behavior)
        speeds = np.linalg.norm(V, axis=1) + 1e-9
        scale = np.minimum(1.0, vmax / speeds)
        V = V * scale[:, None]

    # report
    min_dist = float(np.min(min_pair))
    unsafe_rate = float(np.mean(any_unsafe) * 100.0)
    hard_rate = float(np.mean(any_hard_collision) * 100.0)
    wall_rate = float(np.mean(any_wall_violation) * 100.0)

    print("\n=== TASK 2 NUMERICAL VALIDATION ===")
    print(f"Min pair distance: {min_dist:.2f} px")
    print(f"% frames with d < d_safe ({d_safe:.2f}): {unsafe_rate:.2f}%   [interaction]")
    print(f"% frames with d < 2r ({2.0*r:.2f}): {hard_rate:.2f}%   [TRUE collision]")
    print(f"Wall-violation frames (%): {wall_rate:.2f}%")

    # Animation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(route_bin, cmap="gray", alpha=0.25)
    ax.plot(laneA[:, 0], laneA[:, 1], linewidth=2)
    ax.plot(laneB[:, 0], laneB[:, 1], linewidth=2)
    ax.set_title("Task 2: Swarm (RK4, convoy spacing + tight lateral pack)")
    ax.axis("off")

    robots = []
    for i in range(N):
        circ = plt.Circle((traj[0, i, 0], traj[0, i, 1]), radius=r, fill=True)
        robots.append(circ)
        ax.add_patch(circ)

    for i in range(N):
        if i < N_A:
            robots[i].set_facecolor((0.2, 0.5, 1.0, 0.9))
        else:
            robots[i].set_facecolor((1.0, 0.55, 0.15, 0.9))
        robots[i].set_edgecolor((0.0, 0.0, 0.0, 1.0))
        robots[i].set_linewidth(1.0)

    def update(frame):
        for i in range(N):
            robots[i].center = (float(traj[frame, i, 0]), float(traj[frame, i, 1]))
        return robots

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
