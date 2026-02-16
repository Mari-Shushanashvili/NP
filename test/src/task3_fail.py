import numpy as np
import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def rk4_step_frame_time(X, V, k, accel_frame_fn):
    h = 1.0
    A1 = accel_frame_fn(X, V, k)
    k1x, k1v = V, A1
    X2, V2 = X + 0.5 * h * k1x, V + 0.5 * h * k1v
    A2 = accel_frame_fn(X2, V2, k + 0.5)
    k2x, k2v = V2, A2
    X3, V3 = X + 0.5 * h * k2x, V + 0.5 * h * k2v
    A3 = accel_frame_fn(X3, V3, k + 0.5)
    k3x, k3v = V3, A3
    X4, V4 = X + h * k3x, V + h * k3v
    A4 = accel_frame_fn(X4, V4, k + 1.0)
    k4x, k4v = V4, A4
    return X + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x), V + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)


def safe_unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "pedestrian_trajectories.npy"
    video_path = project_root / "data" / "pedestrians.mp4"
    peds_data = np.load(data_path, allow_pickle=True)
    cap = cv2.VideoCapture(str(video_path))
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25.0
    dt_sec = 1.0 / fps

    # --- LIMITATION PARAMETERS ---
    # We increase d_safe and k_rep massively to show the "Freeze" limitation
    kp_goal = 5.0  # Lower pull toward goal
    kd_phys = 10.0  # Higher damping
    d_safe = 150.0  # HUGE Safety Radius (The Limitation)
    k_rep_ped = 5e5  # Massive Repulsion
    vmax = 80.0

    # Initialize 1 Robot in the middle-bottom, goal at middle-top
    X = np.array([[w // 2, h - 100]], dtype=np.float32)
    V = np.zeros((1, 2), dtype=np.float32)
    goal = np.array([[w // 2, 100]], dtype=np.float32)
    trajectories = []

    def accel_frame_fn(Xc, Vc, k):
        idx = int(np.clip(round(float(k)), 0, len(peds_data) - 1))
        current_peds = np.array(peds_data[idx], dtype=np.float32)

        # Goal Force
        gvec = goal[0] - Xc[0]
        A = kp_goal * safe_unit(gvec)

        # Repulsion (The cause of the freeze)
        if current_peds.size > 0:
            diffs = Xc[0] - current_peds
            dists = np.linalg.norm(diffs, axis=1)
            mask = dists < d_safe
            if np.any(mask):
                # Using your exact weighting logic but with the "Fail" parameters
                wgt = ((d_safe - dists[mask]) / d_safe) ** 2
                A += (k_rep_ped * (wgt[:, None] * (diffs[mask] / dists[mask, None]))).sum(axis=0)

        return (A - kd_phys * Vc[0]) * dt_sec

    print("Simulating Limitation (Local Minimum)...")
    for k in range(len(peds_data)):
        trajectories.append(X.copy())
        X, V = rk4_step_frame_time(X, V, float(k), accel_frame_fn)
        # Cap speed
        speed = np.linalg.norm(V)
        if speed > vmax * dt_sec: V *= (vmax * dt_sec / speed)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ret, frame0 = cap.read()
    im = ax.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))

    # Add a visual "Panic Zone" to show the limitation
    panic_zone = plt.Circle(trajectories[0][0], d_safe, color='red', fill=True, alpha=0.15, label='Paranoid Zone')
    rob_circ = plt.Circle(trajectories[0][0], 15, color='cyan', label='Robot')
    ax.add_patch(panic_zone)
    ax.add_patch(rob_circ)

    ped_scat = ax.scatter([], [], s=20, c='yellow')
    ax.scatter(goal[0, 0], goal[0, 1], marker='X', s=200, c='lime', label='Goal')
    ax.legend(loc='upper right')

    def update(k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame = cap.read()
        if ret: im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        curr_x = trajectories[k][0]
        rob_circ.center = panic_zone.center = (curr_x[0], curr_x[1])
        peds = np.array(peds_data[k])
        if peds.size > 0: ped_scat.set_offsets(peds)
        return im, rob_circ, panic_zone, ped_scat

    ani = FuncAnimation(fig, update, frames=len(trajectories), interval=1000 / fps, blit=True)
    plt.title(f"Limitation: d_safe={d_safe}px causes Local Minimum (Freeze)")
    plt.show()


if __name__ == "__main__": main()


"""
=== LIMITATION DOCUMENTATION: TASK 3 (PEDESTRIAN NAVIGATION) ===

WHAT IS THE LIMITATION?
The primary limitation of this Artificial Potential Field (APF) approach is the "Local Minimum" 
or "Freezing Behavior." By increasing the 'd_safe' parameter (e.g., to 150px), the robot's 
repulsive field becomes too large. 

WHY DOES IT OCCUR?
In a dense environment (like the car-adjacent area in the video), the combined repulsive forces 
from all pedestrians within that 150px radius create a 'force wall.' Mathematically, the sum 
of these repulsive vectors becomes equal and opposite to the goal-seeking vector. 

RESULT:
The robot does not "crash," but it stops moving entirely. This proves that while the RK4 
integration is stable, the navigation logic requires high-precision tuning of the interaction 
radius to balance safety against progress.
"""
