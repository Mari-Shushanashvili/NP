import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


# ------------------------------------------------------------
# 1. Setup and Data Loading
# ------------------------------------------------------------
def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "pedestrian_trajectories.npy"
    video_path = project_root / "data" / "pedestrians.mp4"

    if not data_path.exists():
        print("Error: Run pedestrian_detector.py first!")
        return

    peds_data = np.load(data_path, allow_pickle=True)
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 1e-6:
        fps = 25.0  # fallback (rare codec issue)
    dt = 1.0 / fps

    # ------------------------------------------------------------
    # 2. Define Two Robots (Direction A and Direction B)
    # ------------------------------------------------------------
    # Robot 0: Bottom -> Top
    # Robot 1: Top -> Bottom
    start_positions = np.array([
        [w // 2 - 100, h - 50],  # Robot 0 start
        [w // 2 + 100, 50]       # Robot 1 start
    ], dtype=float)

    goals = np.array([
        [w // 2 - 50, 50],       # Robot 0 goal
        [w // 2 + 50, h - 50]    # Robot 1 goal
    ], dtype=float)

    # ------------------------------------------------------------
    # 3. Parameters (Numerical Programming Requirements)
    # ------------------------------------------------------------
    kp = 12.0          # Goal pull
    kd = 5.0           # Damping
    vmax = 10.0        # Velocity Saturation
    k_rep_ped = 6000.0 # Repulsion from people
    k_rep_rob = 8000.0 # Repulsion from other robot
    d_safe = 80.0      # Detection / interaction radius
    robot_r = 12.0     # Robot radius for drawing + reporting

    # Reporting-only assumption (detections are points, so we pick a conservative radius)
    ped_r_assumed = 10.0
    hard_safe_rp = robot_r + ped_r_assumed   # robot–ped hard threshold (reporting)
    hard_safe_rr = 2.0 * robot_r             # robot–robot overlap threshold

    # Initial states for both robots
    X = start_positions.copy()
    V = np.zeros((2, 2))

    frames_count = len(peds_data)
    trajectories = [[] for _ in range(2)]

    # ------------------------------------------------------------
    # 3.5 Safety metrics storage (NEW)
    # ------------------------------------------------------------
    min_robot_ped = np.full((frames_count,), np.inf, dtype=float)  # min over both robots vs all peds
    min_robot_robot = np.full((frames_count,), np.inf, dtype=float)

    below_dsafe_rp = np.zeros((frames_count,), dtype=bool)
    below_hard_rp = np.zeros((frames_count,), dtype=bool)

    below_dsafe_rr = np.zeros((frames_count,), dtype=bool)
    below_hard_rr = np.zeros((frames_count,), dtype=bool)

    # ------------------------------------------------------------
    # 4. Simulation Loop (Pre-calculating paths for the animation)
    # ------------------------------------------------------------
    print("Calculating physics for both directions...")
    for k in range(frames_count):
        current_peds = np.array(peds_data[k], dtype=float)

        new_accels = np.zeros((2, 2), dtype=float)

        for i in range(2):  # For each robot
            # a) Target Force (Pull to goal)
            dist_to_goal = np.linalg.norm(goals[i] - X[i])
            if dist_to_goal > 5:
                unit_to_goal = (goals[i] - X[i]) / (dist_to_goal + 1e-9)
                f_target = kp * unit_to_goal
            else:
                f_target = -kd * V[i]  # Brake at goal

            # b) Pedestrian Repulsion (Dynamic Obstacles)
            f_rep_peds = np.zeros(2, dtype=float)
            if len(current_peds) > 0:
                diffs = X[i] - current_peds
                dists = np.linalg.norm(diffs, axis=1)
                for j in range(len(current_peds)):
                    if dists[j] < d_safe:
                        # Inverse square law style (scaled by distance-to-safe)
                        f_rep_peds += k_rep_ped * ((d_safe - dists[j]) / d_safe) ** 2 * (diffs[j] / (dists[j] + 1e-9))

            # c) Robot-Robot Avoidance
            f_rep_robots = np.zeros(2, dtype=float)
            other_robot = 1 - i
            vec_to_rob = X[i] - X[other_robot]
            dist_to_rob = np.linalg.norm(vec_to_rob)
            if dist_to_rob < d_safe:
                f_rep_robots = k_rep_rob * ((d_safe - dist_to_rob) / d_safe) ** 2 * (vec_to_rob / (dist_to_rob + 1e-9))

            # Governing Equation: v_dot = forces - kd*v
            new_accels[i] = f_target + f_rep_peds + f_rep_robots - kd * V[i]

        # Integrate and Saturate (UNCHANGED behavior)
        for i in range(2):
            V[i] += new_accels[i] * dt
            speed = np.linalg.norm(V[i])
            if speed > vmax:
                V[i] = (V[i] / (speed + 1e-9)) * vmax

            # NOTE: keeping your original integration style for position (no extra dt factor)
            X[i] += V[i]
            trajectories[i].append(X[i].copy())

        # ------------------------------------------------------------
        # 4.5 Safety metrics update (NEW) — does NOT affect motion
        # ------------------------------------------------------------
        # Robot-robot distance (2 robots)
        rr = float(np.linalg.norm(X[0] - X[1]))
        min_robot_robot[k] = rr
        below_dsafe_rr[k] = (rr < d_safe)
        below_hard_rr[k] = (rr < hard_safe_rr)

        # Robot-ped distance
        if len(current_peds) > 0:
            # distances shape (2, P)
            dmat = np.linalg.norm(X[:, None, :] - current_peds[None, :, :], axis=2)
            rp = float(np.min(dmat))
        else:
            rp = float("inf")

        min_robot_ped[k] = rp
        below_dsafe_rp[k] = (rp < d_safe)
        below_hard_rp[k] = (rp < hard_safe_rp)

    trajectories = np.array(trajectories)

    # ------------------------------------------------------------
    # 4.9 Print Numerical Validation (NEW)
    # ------------------------------------------------------------
    print("\n=== TASK 3 NUMERICAL VALIDATION ===")
    print(f"Frames: {frames_count}, dt={dt:.4f}s, fps={fps:.2f}")
    print(f"Robot radius r_robot = {robot_r:.2f} px")
    print(f"Ped radius (assumed for reporting) r_ped = {ped_r_assumed:.2f} px")
    print(f"Hard robot–ped safety threshold (r_robot + r_ped): {hard_safe_rp:.2f} px")
    print(f"Hard robot–robot overlap threshold (2*r_robot): {hard_safe_rr:.2f} px")
    print(f"Interaction radius d_safe: {d_safe:.2f} px")

    # Robot–Ped
    finite_rp = min_robot_ped[np.isfinite(min_robot_ped)]
    if finite_rp.size == 0:
        print("\nRobot–Pedestrian distances: no pedestrian detections present.")
    else:
        print("\nRobot–Pedestrian distances:")
        print(f"  Min distance (px): {float(np.min(finite_rp)):.2f}")
        print(f"  % frames with dist < d_safe: {float(np.mean(below_dsafe_rp) * 100.0):.2f}%")
        print(f"  % frames with dist < (r_robot + r_ped): {float(np.mean(below_hard_rp) * 100.0):.2f}%")

    # Robot–Robot
    print("\nRobot–Robot distances:")
    print(f"  Min distance (px): {float(np.min(min_robot_robot)):.2f}")
    print(f"  % frames with dist < d_safe: {float(np.mean(below_dsafe_rr) * 100.0):.2f}%")
    print(f"  % frames with dist < (2*r_robot): {float(np.mean(below_hard_rr) * 100.0):.2f}%")

    # Simple status line (report-friendly)
    if finite_rp.size > 0 and (np.mean(below_hard_rp) == 0.0) and (np.mean(below_hard_rr) == 0.0):
        print("\nSTATUS: SUCCESS ✅ No hard-threshold collisions recorded (robot–ped and robot–robot).")
    else:
        print("\nSTATUS: REVIEW ⚠️ Some hard-threshold near-misses/overlaps detected OR no detections to validate against.")

    # ------------------------------------------------------------
    # 5. Live Animation Overlay
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Placeholder for the video frame
    im = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))

    # Visual elements
    rob0_circ = plt.Circle((0, 0), robot_r, color='cyan', label='Robot A (Up)', ec='black')
    rob1_circ = plt.Circle((0, 0), robot_r, color='magenta', label='Robot B (Down)', ec='black')
    ax.add_patch(rob0_circ)
    ax.add_patch(rob1_circ)

    ped_scat = ax.scatter([], [], c='lime', s=20, edgecolors='black', label='Pedestrians')

    ax.set_title("KIU Task 3: Bidirectional Navigation in Pedestrian Flow")
    ax.legend(loc='upper right')
    ax.axis('off')

    def update(frame_idx):
        # Update video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, img = cap.read()
        if success:
            im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Update Robot positions
        rob0_circ.center = trajectories[0, frame_idx]
        rob1_circ.center = trajectories[1, frame_idx]

        # Update Pedestrian dots
        peds = np.array(peds_data[frame_idx])
        if len(peds) > 0:
            ped_scat.set_offsets(peds)

        return im, rob0_circ, rob1_circ, ped_scat

    ani = FuncAnimation(fig, update, frames=frames_count, interval=1000 / fps, blit=True)
    plt.show()
    cap.release()


if __name__ == "__main__":
    main()

