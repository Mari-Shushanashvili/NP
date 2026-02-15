import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


# RK4 STEP
def rk4_step_frame_time(X, V, k, accel_fn):
    h = 1.0
    k1v = accel_fn(X, V, k)
    k2v = accel_fn(X + 0.5 * h * V, V + 0.5 * h * k1v, k + 0.5)
    k3v = accel_fn(X + 0.5 * h * V, V + 0.5 * h * k2v, k + 0.5)
    k4v = accel_fn(X + h * V, V + h * k3v, k + 1.0)
    return X + h * V, V + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)


def main():
    root = Path(__file__).resolve().parents[1]
    peds_data = np.load(root / "data" / "pedestrian_trajectories.npy", allow_pickle=True)
    video_path = root / "data" / "pedestrians.mp4"
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- FAILURE PARAMETERS ---
    d_safe = 300.0  # HUGE SAFETY ZONE
    k_rep = 80000.0  # HUGE FEAR
    kp = 3.0  # Weak Goal Pull

    print("Task 3 Failure: Watch the robot freeze when peds enter the giant red circle.")

    start = np.array([[w // 2 - 50, h - 50]], dtype=float)
    goal = np.array([[w // 2 - 50, 50]], dtype=float)  # Robot wants to go UP

    X = start.copy()
    V = np.zeros_like(X)
    dt_sec = 0.033

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))

    rob_dot = plt.Circle((0, 0), radius=12, color='cyan', label='Robot')
    safety_circle = plt.Circle((0, 0), radius=d_safe, color='red', fill=False, linestyle='--', label='Paranoid Zone')
    ped_scat = ax.scatter([], [], c='lime', s=30, label='Peds')

    ax.add_patch(rob_dot)
    ax.add_patch(safety_circle)
    ax.legend(loc="upper right")
    ax.set_title(f"FAIL: d_safe={d_safe} (Local Minima Freeze)")
    ax.axis('off')

    def accel_fn(Xc, Vc, k):
        idx = int(np.clip(k, 0, len(peds_data) - 1))
        curr_peds = np.array(peds_data[idx])

        # Goal Force
        A = kp * (goal - Xc)

        if len(curr_peds) > 0:
            diff = Xc - curr_peds
            dist = np.linalg.norm(diff, axis=1, keepdims=True)

            # Massive repulsion if inside paranoid zone
            mask = dist < d_safe
            if np.any(mask):
                # Repulsion explodes as 1/r^2
                force = k_rep * (1.0 / (dist[mask] ** 2 + 1.0)) * (diff[mask] / dist[mask])
                A += np.sum(force, axis=0)

        return (A - 5.0 * Vc) * dt_sec

    def update(frame):
        nonlocal X, V
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        if ret:
            im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        X, V = rk4_step_frame_time(X, V, frame, accel_fn)

        rob_dot.center = (X[0, 0], X[0, 1])
        safety_circle.center = (X[0, 0], X[0, 1])

        curr_peds = peds_data[frame]
        if len(curr_peds) > 0:
            ped_scat.set_offsets(curr_peds)
        else:
            ped_scat.set_offsets(np.zeros((0, 2)))

        return im, rob_dot, safety_circle, ped_scat

    ani = FuncAnimation(fig, update, frames=len(peds_data), interval=30, blit=True)
    plt.show()
    cap.release()


if __name__ == "__main__":
    main()