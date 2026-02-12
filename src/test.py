import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist


# --- Utility Functions ---
def load_spline(root): return np.load(root / "data/centerline_spline.npy").astype(np.float32), None


def load_dt(root): return np.load(root / "data/dt_halfwidth.npy").astype(np.float32), None


def load_params(root): return json.loads((root / "data/corridor_params.json").read_text()), None


def load_init(root): return np.load(root / "data/task2_init.npy", allow_pickle=True).item(), None


def load_mask(root): return (cv2.imread(str(root / "data/route_mask.png"), 0) > 0), None


def build_lanes(spline, offset):
    M = len(spline)
    normals = np.zeros_like(spline)
    for i in range(M):
        i0, i1 = max(0, i - 1), min(M - 1, i + 1)
        t = spline[i1] - spline[i0]
        t /= (np.linalg.norm(t) + 1e-9)
        normals[i] = np.array([-t[1], t[0]])
    return (spline + offset * normals), (spline - offset * normals)


# ------------------------------------------------------------
# Core Physics Engine
# ------------------------------------------------------------
def get_acceleration(Xi, Vi, targets, P_rails, kp, kd, k_lane, d_safe, k_rep, dt_map, r, k_wall):
    N = Xi.shape[0]
    # Path tracking + Rail pull
    A = kp * (targets - Xi) - kd * Vi + k_lane * (P_rails - Xi)
    # Collision avoidance
    dists = cdist(Xi, Xi)
    mask = (dists < d_safe) & (dists > 0.01)
    for i in range(N):
        if np.any(mask[i]):
            diff = Xi[i] - Xi[mask[i]]
            d = dists[i, mask[i]][:, None]
            A[i] += np.sum(k_rep * ((d_safe - d) / d_safe) ** 2 * (diff / d), axis=0)
    # Wall avoidance
    h, w = dt_map.shape
    for i in range(N):
        x, y = int(np.clip(Xi[i, 0], 1, w - 2)), int(np.clip(Xi[i, 1], 1, h - 2))
        d_val = dt_map[y, x]
        if d_val < (r + 4.0):
            gx = (dt_map[y, x + 1] - dt_map[y, x - 1])
            gy = (dt_map[y + 1, x] - dt_map[y - 1, x])
            grad = np.array([gx, gy]) / (np.hypot(gx, gy) + 1e-9)
            A[i] += k_wall * (r + 4.0 - d_val) * grad
    return A


def main():
    root = Path(__file__).resolve().parents[1]
    spline, _ = load_spline(root)
    dt_map, _ = load_dt(root)
    init, _ = load_init(root)
    route_bin, _ = load_mask(root)

    X, V = init["X0"].astype(float), init["V0"].astype(float)
    N_A, N_B, r, d_safe = init["N_A"], init["N_B"], init["r"], init["d_safe"]
    N, M = len(X), len(spline)

    # --- SETTINGS ---
    lane_offset = 10.0
    idx_spacing = 25
    kp, kd = 50.0, 15.0
    vmax = 25.0
    k_lane, k_rep, k_wall = 100.0, 1800.0, 200.0
    dt, frames = 0.05, 1100

    laneA, laneB = build_lanes(spline, lane_offset)

    hints = np.array([np.argmin(np.sum(((laneA if i < N_A else laneB) - X[i]) ** 2, axis=1)) for i in range(N)])
    master_progress = float(hints[0])

    # --- Metrics Setup ---
    traj = np.zeros((frames, N, 2))
    min_pair = np.zeros(frames)
    any_wall_violation = np.zeros(frames, dtype=bool)
    any_unsafe = np.zeros(frames, dtype=bool)

    print(f"Running Swarm (5 vs 5)...")

    for k in range(frames):
        traj[k] = X.copy()

        # 1. Update Validity Metrics for this frame
        dist_matrix = cdist(X, X)
        np.fill_diagonal(dist_matrix, np.inf)
        m_dist = np.min(dist_matrix)
        min_pair[k] = m_dist
        any_unsafe[k] = m_dist < d_safe

        h_m, w_m = dt_map.shape
        # Convert coordinates to pixel indices for wall check
        check_x = np.clip(X[:, 0].astype(int), 0, w_m - 1)
        check_y = np.clip(X[:, 1].astype(int), 0, h_m - 1)
        any_wall_violation[k] = np.any(dt_map[check_y, check_x] < r)

        # 2. Physics & Navigation
        master_progress += vmax * dt * 0.8
        targets = np.zeros_like(X)
        P_rails = np.zeros_like(X)

        for i in range(N):
            lane = laneA if i < N_A else laneB
            if i < N_A:
                t_idx = int(master_progress - (i * idx_spacing))
            else:
                prog_from_start = master_progress - hints[0]
                t_idx = int((M - 1 - hints[0]) - prog_from_start + (i - N_A) * idx_spacing)

            targets[i] = lane[np.clip(t_idx, 0, M - 1)]

            h, win = hints[i], 50
            seg = lane[max(0, h - win):min(M, h + win)]
            hints[i] = max(0, h - win) + np.argmin(np.sum((seg - X[i]) ** 2, axis=1))
            P_rails[i] = lane[hints[i]]

        A = get_acceleration(X, V, targets, P_rails, kp, kd, k_lane, d_safe, k_rep, dt_map, r, k_wall)
        V += A * dt
        sp = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        V *= np.minimum(1.0, vmax / sp)
        X += V * dt

    # --- 3. Final Constraint Report ---
    wall_viol_rate = float(any_wall_violation.mean() * 100.0)
    unsafe_rate = float(any_unsafe.mean() * 100.0)
    min_dist_overall = float(min_pair.min())

    print("\n=== Task 2 Final Constraint Report ===")
    print(f"Corridor Violation Rate: {wall_viol_rate:.2f}%")
    print(f"Collision Risk Rate (Dist < d_safe): {unsafe_rate:.2f}%")
    print(f"Minimum Distance recorded: {min_dist_overall:.2f} px (Target: > {d_safe:.2f})")

    if wall_viol_rate == 0 and unsafe_rate == 0:
        print("RESULT: PASS ✅ All safety constraints met.")
    else:
        print("RESULT: FAIL ⚠️ Safety violations detected.")

    # --- 4. Visualization ---
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
    ax.imshow(route_bin, cmap="gray", alpha=0.1)

    ax.plot(laneA[:, 0], laneA[:, 1], color='white', alpha=0.15, linewidth=1)
    ax.plot(laneB[:, 0], laneB[:, 1], color='white', alpha=0.15, linewidth=1)

    colors = ['#00FFFF'] * N_A + ['#FF851B'] * N_B
    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=40, c=colors, edgecolors='black', linewidth=0.5)

    ax.set_title("Task 2: Swarm Convoy Passage (5 vs 5)", color='white')
    ax.axis("off")

    def update(frame):
        scat.set_offsets(traj[frame])
        return (scat,)

    ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()