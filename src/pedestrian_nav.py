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

    # ------------------------------------------------------------
    # 2. Define Two Robots (Direction A and Direction B)
    # ------------------------------------------------------------
    # Robot 0: Bottom -> Top
    # Robot 1: Top -> Bottom
    start_positions = np.array([
        [w // 2 - 100, h - 50],  # Robot 0 start
        [w // 2 + 100, 50]  # Robot 1 start
    ], dtype=float)

    goals = np.array([
        [w // 2 - 50, 50],  # Robot 0 goal
        [w // 2 + 50, h - 50]  # Robot 1 goal
    ], dtype=float)

    # ------------------------------------------------------------
    # 3. Parameters (Numerical Programming Requirements)
    # ------------------------------------------------------------
    kp = 12.0  # Goal pull
    kd = 5.0  # Damping
    vmax = 10.0  # Velocity Saturation (Page 5)
    k_rep_ped = 6000.0  # Repulsion from people
    k_rep_rob = 8000.0  # Repulsion from other robot
    d_safe = 80.0  # Detection radius
    robot_r = 12.0
    dt = 1.0 / fps

    # Initial states for both robots
    X = start_positions.copy()
    V = np.zeros((2, 2))

    frames_count = len(peds_data)
    trajectories = [[] for _ in range(2)]

    # ------------------------------------------------------------
    # 4. Simulation Loop (Pre-calculating paths for the animation)
    # ------------------------------------------------------------
    print("Calculating physics for both directions...")
    for k in range(frames_count):
        current_peds = np.array(peds_data[k])

        new_accels = np.zeros((2, 2))

        for i in range(2):  # For each robot
            # a) Target Force (Pull to goal)
            dist_to_goal = np.linalg.norm(goals[i] - X[i])
            if dist_to_goal > 5:
                unit_to_goal = (goals[i] - X[i]) / (dist_to_goal + 1e-9)
                f_target = kp * unit_to_goal
            else:
                f_target = -kd * V[i]  # Brake at goal

            # b) Pedestrian Repulsion (Dynamic Obstacles)
            f_rep_peds = np.zeros(2)
            if len(current_peds) > 0:
                diffs = X[i] - current_peds
                dists = np.linalg.norm(diffs, axis=1)
                for j in range(len(current_peds)):
                    if dists[j] < d_safe:
                        # Inverse square law (Page 8)
                        f_rep_peds += k_rep_ped * ((d_safe - dists[j]) / d_safe) ** 2 * (diffs[j] / (dists[j] + 1e-9))

            # c) Robot-Robot Avoidance
            f_rep_robots = np.zeros(2)
            other_robot = 1 - i
            vec_to_rob = X[i] - X[other_robot]
            dist_to_rob = np.linalg.norm(vec_to_rob)
            if dist_to_rob < d_safe:
                f_rep_robots = k_rep_rob * ((d_safe - dist_to_rob) / d_safe) ** 2 * (vec_to_rob / (dist_to_rob + 1e-9))

            # Governing Equation: v_dot = f_target + f_rep - kd*v
            new_accels[i] = f_target + f_rep_peds + f_rep_robots - kd * V[i]

        # Integrate and Saturate
        for i in range(2):
            V[i] += new_accels[i] * dt
            # Velocity Saturation (Page 7)
            speed = np.linalg.norm(V[i])
            if speed > vmax:
                V[i] = (V[i] / speed) * vmax
            X[i] += V[i]
            trajectories[i].append(X[i].copy())

    trajectories = np.array(trajectories)

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