import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def rk4_step_frame_time(X, V, k, accel_frame_fn):
    """
    RK4 step in 'frame-time' (step size = 1 frame).
    Dynamics:
      dX/dk = V
      dV/dk = A_frame  (A_frame already includes dt_sec scaling)
    """
    h = 1.0  # one frame

    A1 = accel_frame_fn(X, V, k)
    k1x = V
    k1v = A1

    X2 = X + 0.5 * h * k1x
    V2 = V + 0.5 * h * k1v
    A2 = accel_frame_fn(X2, V2, k + 0.5)
    k2x = V2
    k2v = A2

    X3 = X + 0.5 * h * k2x
    V3 = V + 0.5 * h * k2v
    A3 = accel_frame_fn(X3, V3, k + 0.5)
    k3x = V3
    k3v = A3

    X4 = X + h * k3x
    V4 = V + h * k3v
    A4 = accel_frame_fn(X4, V4, k + 1.0)
    k4x = V4
    k4v = A4

    Xn = X + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Vn = V + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Xn, Vn


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "pedestrian_trajectories.npy"
    video_path = project_root / "data" / "pedestrians.mp4"

    if not data_path.exists():
        print("Error: Run pedestrian_detector.py first!")
        return

    peds_data = np.load(data_path, allow_pickle=True)
    cap = cv2.VideoCapture(str(video_path))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    dt_sec = 1.0 / fps

    # Two robots, opposite directions
    start_positions = np.array([
        [w // 2 - 100, h - 50],
        [w // 2 + 100, 50]
    ], dtype=float)

    goals = np.array([
        [w // 2 - 50, 50],
        [w // 2 + 50, h - 50]
    ], dtype=float)

    # Parameters
    kp = 12.0
    kd = 5.0
    vmax = 10.0

    k_rep_ped = 6000.0
    k_rep_rob = 8000.0
    d_safe = 80.0

    robot_r = 12.0
    ped_r_assumed = 10.0
    hard_safe_rp = robot_r + ped_r_assumed
    hard_safe_rr = 2.0 * robot_r

    X = start_positions.copy()
    V = np.zeros((2, 2), dtype=float)

    frames_count = len(peds_data)
    trajectories = [[] for _ in range(2)]

    # Safety metrics (reporting only)
    min_robot_ped = np.full((frames_count,), np.inf, dtype=float)
    min_robot_robot = np.full((frames_count,), np.inf, dtype=float)

    below_dsafe_rp = np.zeros((frames_count,), dtype=bool)
    below_hard_rp = np.zeros((frames_count,), dtype=bool)

    below_dsafe_rr = np.zeros((frames_count,), dtype=bool)
    below_hard_rr = np.zeros((frames_count,), dtype=bool)

    def accel_frame_fn(Xc, Vc, k):
        """
        Returns dV/dk (frame-based), so we multiply physical acceleration by dt_sec.
        This keeps your old behavior consistent, but uses RK4 instead of Euler.
        """
        frame_idx = int(np.clip(round(k), 0, frames_count - 1))
        current_peds = np.array(peds_data[frame_idx], dtype=float)

        A = np.zeros((2, 2), dtype=float)

        for i in range(2):
            # a) goal pull
            dist_to_goal = np.linalg.norm(goals[i] - Xc[i])
            if dist_to_goal > 5:
                unit_to_goal = (goals[i] - Xc[i]) / (dist_to_goal + 1e-9)
                f_target = kp * unit_to_goal
            else:
                f_target = -kd * Vc[i]

            # b) pedestrian repulsion
            f_rep_peds = np.zeros(2, dtype=float)
            if len(current_peds) > 0:
                diffs = Xc[i] - current_peds
                dists = np.linalg.norm(diffs, axis=1)
                for j in range(len(current_peds)):
                    if dists[j] < d_safe:
                        f_rep_peds += k_rep_ped * ((d_safe - dists[j]) / d_safe) ** 2 * (
                            diffs[j] / (dists[j] + 1e-9)
                        )

            # c) robot-robot repulsion
            other = 1 - i
            vec = Xc[i] - Xc[other]
            dist = np.linalg.norm(vec)
            f_rep_rob = np.zeros(2, dtype=float)
            if dist < d_safe:
                f_rep_rob = k_rep_rob * ((d_safe - dist) / d_safe) ** 2 * (vec / (dist + 1e-9))

            # physical acceleration (per second^2-ish in this toy model)
            a_phys = f_target + f_rep_peds + f_rep_rob - kd * Vc[i]

            # convert to per-frame derivative of V (dV/dk)
            A[i] = a_phys * dt_sec

        return A

    print("Calculating physics (Task 3, RK4)...")
    for k in range(frames_count):
        trajectories[0].append(X[0].copy())
        trajectories[1].append(X[1].copy())

        # Metrics for reporting (based on current X)
        rr = float(np.linalg.norm(X[0] - X[1]))
        min_robot_robot[k] = rr
        below_dsafe_rr[k] = (rr < d_safe)
        below_hard_rr[k] = (rr < hard_safe_rr)

        current_peds = np.array(peds_data[k], dtype=float)
        if len(current_peds) > 0:
            dmat = np.linalg.norm(X[:, None, :] - current_peds[None, :, :], axis=2)
            rp = float(np.min(dmat))
        else:
            rp = float("inf")

        min_robot_ped[k] = rp
        below_dsafe_rp[k] = (rp < d_safe)
        below_hard_rp[k] = (rp < hard_safe_rp)

        # RK4 step in frame-time
        X, V = rk4_step_frame_time(X, V, k, accel_frame_fn)

        # speed cap (unchanged intent)
        speeds = np.linalg.norm(V, axis=1) + 1e-9
        scale = np.minimum(1.0, vmax / speeds)
        V = V * scale[:, None]

    trajectories = np.array(trajectories)

    # ---- print validation ----
    print("\n=== TASK 3 NUMERICAL VALIDATION ===")
    print(f"Frames: {frames_count}, dt=1/{fps:.2f}={dt_sec:.4f}s, fps={fps:.2f}")
    print(f"Robot radius r_robot = {robot_r:.2f} px")
    print(f"Ped radius (assumed for reporting) r_ped = {ped_r_assumed:.2f} px")
    print(f"Hard robot–ped safety threshold (r_robot + r_ped): {hard_safe_rp:.2f} px")
    print(f"Hard robot–robot overlap threshold (2*r_robot): {hard_safe_rr:.2f} px")
    print(f"Interaction radius d_safe: {d_safe:.2f} px")

    finite_rp = min_robot_ped[np.isfinite(min_robot_ped)]
    if finite_rp.size == 0:
        print("\nRobot–Pedestrian distances: no pedestrian detections present.")
    else:
        print("\nRobot–Pedestrian distances:")
        print(f"  Min distance (px): {float(np.min(finite_rp)):.2f}")
        print(f"  % frames with dist < d_safe: {float(np.mean(below_dsafe_rp) * 100.0):.2f}%")
        print(f"  % frames with dist < (r_robot + r_ped): {float(np.mean(below_hard_rp) * 100.0):.2f}%")

    print("\nRobot–Robot distances:")
    print(f"  Min distance (px): {float(np.min(min_robot_robot)):.2f}")
    print(f"  % frames with dist < d_safe: {float(np.mean(below_dsafe_rr) * 100.0):.2f}%")
    print(f"  % frames with dist < (2*r_robot): {float(np.mean(below_hard_rr) * 100.0):.2f}%")

    if finite_rp.size > 0 and (np.mean(below_hard_rp) == 0.0) and (np.mean(below_hard_rr) == 0.0):
        print("\nSTATUS: SUCCESS ✅ No hard-threshold collisions recorded (robot–ped and robot–robot).")
    else:
        print("\nSTATUS: REVIEW ⚠️ Near-miss/overlap detected OR no detections to validate against.")

    # ---- animation overlay ----
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))

    rob0 = plt.Circle((0, 0), robot_r, color='cyan', ec='black', label='Robot A')
    rob1 = plt.Circle((0, 0), robot_r, color='magenta', ec='black', label='Robot B')
    ax.add_patch(rob0)
    ax.add_patch(rob1)

    ped_scat = ax.scatter([], [], c='lime', s=20, edgecolors='black', label='Pedestrians')

    ax.set_title("Task 3: Bidirectional Navigation (RK4, discrete obstacle repulsion)")
    ax.legend(loc='upper right')
    ax.axis('off')

    def update(frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if ok:
            im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        rob0.center = trajectories[0, frame_idx]
        rob1.center = trajectories[1, frame_idx]

        peds = np.array(peds_data[frame_idx])
        if len(peds) > 0:
            ped_scat.set_offsets(peds)
        else:
            ped_scat.set_offsets(np.zeros((0, 2)))

        return im, rob0, rob1, ped_scat

    ani = FuncAnimation(fig, update, frames=frames_count, interval=1000 / fps, blit=True)
    plt.show()
    cap.release()


if __name__ == "__main__":
    main()
