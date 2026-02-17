import numpy as np
import cv2
import matplotlib
# Useing TkAgg for better compatibility with interactive point picking in PyCharm/Windows
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


# RK4 in "frame-time"
def rk4_step_frame_time(X, V, k, accel_frame_fn):
    """
    RK4 step in 'frame index' time (step size = 1 frame).
    Integrates dX/dk = V, dV/dk = A_frame
    """
    h = 1.0

    # K1
    A1 = accel_frame_fn(X, V, k)
    k1x = V
    k1v = A1

    # K2
    X2 = X + 0.5 * h * k1x
    V2 = V + 0.5 * h * k1v
    A2 = accel_frame_fn(X2, V2, k + 0.5)
    k2x = V2
    k2v = A2

    # K3
    X3 = X + 0.5 * h * k2x
    V3 = V + 0.5 * h * k2v
    A3 = accel_frame_fn(X3, V3, k + 0.5)
    k3x = V3
    k3v = A3

    # K4
    X4 = X + h * k3x
    V4 = V + h * k3v
    A4 = accel_frame_fn(X4, V4, k + 1.0)
    k4x = V4
    k4v = A4

    Xn = X + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Vn = V + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Xn, Vn


def safe_unit(v):
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)



# Point Picking Helpers
def pick_points_from_video(video_path: Path, save_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read the first frame of the video.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame_rgb)
    ax.set_title(
        "Click 4 points in ORDER:\n1. R1 Start -> 2. R1 Goal -> 3. R2 Start -> 4. R2 Goal\n(Close window when done)")
    ax.axis("off")

    print("Waiting for 4 clicks in the plot window...")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)

    if len(pts) != 4:
        raise RuntimeError(f"Expected 4 clicks, got {len(pts)}")

    pts = np.array(pts, dtype=np.float32)
    start_positions = np.array([pts[0], pts[2]], dtype=np.float32)
    goals = np.array([pts[1], pts[3]], dtype=np.float32)

    np.save(save_path, {"start_positions": start_positions, "goals": goals}, allow_pickle=True)
    print(f"Saved Task 3 points to: {save_path}")
    return start_positions, goals


def load_or_pick_points(video_path: Path, points_path: Path, force_repick: bool = False):
    if (not force_repick) and points_path.exists():
        try:
            data = np.load(points_path, allow_pickle=True).item()
            print(f"Loaded Task 3 points from: {points_path}")
            return data["start_positions"], data["goals"]
        except Exception as e:
            print(f"Error loading points, re-picking... {e}")

    return pick_points_from_video(video_path, points_path)



def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_path = data_dir / "pedestrian_trajectories.npy"
    video_path = data_dir / "pedestrians.mp4"
    points_path = data_dir / "task3_points.npy"

    if not data_path.exists():
        print("Error: Run pedestrian_detector.py first!")
        return

    peds_data = np.load(data_path, allow_pickle=True)

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dt_sec = 1.0 / fps
    frames_count = len(peds_data)

    # Point Picking
    start_positions, goals = load_or_pick_points(video_path, points_path, force_repick=False)

    # Physics Parameters
    kp_goal, kd_phys = 18.0, 6.0
    goal_tol, vmax = 14.0, 120.0
    k_rep_ped, k_rep_rob = 6500.0, 9000.0
    k_hard_ped, k_hard_rob = 22000.0, 26000.0
    d_safe, r_robot, r_ped = 80.0, 12.0, 10.0
    hard_safe_rp = r_robot + r_ped
    hard_safe_rr = 2.0 * r_robot

    X = start_positions.copy()
    V = np.zeros((2, 2), dtype=np.float32)
    trajectories = np.zeros((frames_count, 2, 2), dtype=np.float32)

    # Safety Metrics Storage
    min_robot_ped = np.full((frames_count,), np.inf)
    min_robot_robot = np.full((frames_count,), np.inf)
    below_dsafe_rp = np.zeros((frames_count,), dtype=bool)
    below_hard_rp = np.zeros((frames_count,), dtype=bool)
    below_dsafe_rr = np.zeros((frames_count,), dtype=bool)
    below_hard_rr = np.zeros((frames_count,), dtype=bool)

    def accel_frame_fn(Xc, Vc, k):
        frame_idx = int(np.clip(round(float(k)), 0, frames_count - 1))
        current_peds = np.array(peds_data[frame_idx], dtype=np.float32)
        A = np.zeros((2, 2), dtype=np.float32)

        for i in range(2):
            # Goal logic
            gvec = goals[i] - Xc[i]
            dist_goal = np.linalg.norm(gvec)
            a_goal = -kd_phys * Vc[i] if dist_goal <= goal_tol else kp_goal * safe_unit(gvec)

            # Repulsion
            a_rep = np.zeros(2, dtype=np.float32)
            if current_peds.size > 0:
                diffs = Xc[i] - current_peds
                dists = np.linalg.norm(diffs, axis=1) + 1e-9
                # Pedestrian repulsion (Soft Zone + Hard Barrier)
                for d_limit, k_val in [(d_safe, k_rep_ped), (hard_safe_rp, k_hard_ped)]:
                    mask = dists < d_limit
                    if np.any(mask):
                        d, u = dists[mask], diffs[mask] / dists[mask, None]
                        wgt = ((d_limit - d) / d_limit) ** 2
                        a_rep += (k_val * (wgt[:, None] * u)).sum(axis=0)

            # Robot-Robot repulsion
            vec_rr = Xc[i] - Xc[1 - i]
            dist_rr = np.linalg.norm(vec_rr) + 1e-9
            for d_limit, k_val in [(d_safe, k_rep_rob), (hard_safe_rr, k_hard_rob)]:
                if dist_rr < d_limit:
                    a_rep += k_val * ((d_limit - dist_rr) / d_limit) ** 2 * (vec_rr / dist_rr)

            A[i] = (a_goal + a_rep - kd_phys * Vc[i]) * dt_sec
        return A

    print("Simulating physics...")
    for k in range(frames_count):
        trajectories[k] = X

        # --- Calculate Safety Metrics ---
        # Robot-Robot Dist
        rr_dist = np.linalg.norm(X[0] - X[1])
        min_robot_robot[k] = rr_dist
        below_dsafe_rr[k] = (rr_dist < d_safe)
        below_hard_rr[k] = (rr_dist < hard_safe_rr)

        # Robot-Ped Dist
        current_peds = np.array(peds_data[k], dtype=np.float32)
        if current_peds.size > 0:
            d_mat = np.linalg.norm(X[:, None, :] - current_peds[None, :, :], axis=2)
            rp_dist = np.min(d_mat)
            min_robot_ped[k] = rp_dist
            below_dsafe_rp[k] = (rp_dist < d_safe)
            below_hard_rp[k] = (rp_dist < hard_safe_rp)

        # --- Integrate ---
        X, V = rk4_step_frame_time(X, V, float(k), accel_frame_fn)

        # Speed cap & Video bounds
        speed = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        V *= np.minimum(1.0, (vmax * dt_sec) / speed)
        X[:, 0] = np.clip(X[:, 0], 0, w - 1)
        X[:, 1] = np.clip(X[:, 1], 0, h - 1)

    # Defense Logs
    print("\n=== TASK 3 NUMERICAL VALIDATION ===")
    print(f"Frames: {frames_count}, dt={dt_sec:.4f}s, fps={fps:.2f}")
    print(f"Robot radius r_robot = {r_robot:.2f} px")
    print(f"Ped radius (assumed) r_ped = {r_ped:.2f} px")
    print(f"Hard robot–ped safety threshold: {hard_safe_rp:.2f} px")
    print(f"Hard robot–robot overlap threshold: {hard_safe_rr:.2f} px")
    print(f"Interaction radius d_safe: {d_safe:.2f} px")

    print("\nRobot–Pedestrian distances:")
    print(f"  Min distance: {np.min(min_robot_ped):.2f} px")
    print(f"  % frames in soft zone (< d_safe): {np.mean(below_dsafe_rp) * 100.0:.2f}%")
    print(f"  % frames in hard zone (< collision): {np.mean(below_hard_rp) * 100.0:.2f}%")

    print("\nRobot–Robot distances:")
    print(f"  Min distance: {np.min(min_robot_robot):.2f} px")
    print(f"  % frames in soft zone (< d_safe): {np.mean(below_dsafe_rr) * 100.0:.2f}%")
    print(f"  % frames in hard zone (< collision): {np.mean(below_hard_rr) * 100.0:.2f}%")

    if np.any(below_hard_rp) or np.any(below_hard_rr):
        print("\nSTATUS: WARNING Hard-threshold collision occurred.")
    else:
        print("\nSTATUS: SUCCESS No hard-threshold collisions recorded.")

    # Animation Setup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    im = ax.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))

    ped_scat = ax.scatter([], [], s=20, c='yellow')
    # Init with 2 dummy points to satisfy color consistency
    rob_scat = ax.scatter([0, 0], [0, 0], s=100, c=['cyan', 'magenta'], edgecolors='white')
    ax.scatter(goals[:, 0], goals[:, 1], s=150, marker="X", c=['cyan', 'magenta'])

    def update(k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame = cap.read()
        if ret:
            im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        peds = np.array(peds_data[k], dtype=np.float32)
        if peds.size > 0:
            ped_scat.set_offsets(peds)
        else:
            ped_scat.set_offsets(np.empty((0, 2)))

        rob_scat.set_offsets(trajectories[k])
        return [im, ped_scat, rob_scat]

    print("\nStarting animation...")
    anim = FuncAnimation(fig, update, frames=frames_count, interval=1000 / fps, blit=True)
    plt.show()
    cap.release()


if __name__ == "__main__":
    main()