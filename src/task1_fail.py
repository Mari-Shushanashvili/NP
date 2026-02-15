import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- SETUP ---
def load_spline(project_root: Path):
    p = project_root / "data" / "centerline_spline.npy"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    return np.load(p).astype(np.float32)


def load_dt(project_root: Path):
    p = project_root / "data" / "dt_halfwidth.npy"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    return np.load(p).astype(np.float32)


def load_mask(project_root: Path):
    p = project_root / "data" / "route_mask_inflated.png"
    if not p.exists(): p = project_root / "data" / "route_mask.png"
    return (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0)


# RK4 STEP (Same as your robot_ivp.py)
def rk4_step(x, v, t_idx, dt, spline_pts, kp, kd, vmax, target_idx_fn, dt_map, r, k_wall, wall_margin):
    def f(xi, vi, ti):
        # Simple target selection
        idx = target_idx_fn(ti)
        idx = int(np.clip(idx, 0, len(spline_pts) - 1))
        T = spline_pts[idx]

        a = kp * (T - xi) - kd * vi

        # Wall force (Finite Diff)
        h, w = dt_map.shape
        px = int(np.clip(round(float(xi[0])), 1, w - 2))
        py = int(np.clip(round(float(xi[1])), 1, h - 2))
        dist = float(dt_map[py, px])
        thresh = float(r + wall_margin)

        if dist < thresh:
            gx = 0.5 * (float(dt_map[py, px + 1]) - float(dt_map[py, px - 1]))
            gy = 0.5 * (float(dt_map[py + 1, px]) - float(dt_map[py - 1, px]))
            grad = np.array([gx, gy], dtype=np.float32)
            grad /= (np.linalg.norm(grad) + 1e-9)
            a += float(k_wall) * (thresh - dist) * grad

        speed = np.linalg.norm(vi) + 1e-9
        scale = min(1.0, vmax / speed)
        return vi * scale, a

    k1x, k1v = f(x, v, t_idx)
    k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t_idx + 0.5)
    k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t_idx + 0.5)
    k4x, k4v = f(x + dt * k3x, v + dt * k3v, t_idx + 1.0)
    return x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x), v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)


def main():
    root = Path(__file__).resolve().parents[1]
    spline = load_spline(root)
    dt_map = load_dt(root)
    mask = load_mask(root)

    # --- FAILURE PARAMETERS ---
    vmax = 200.0  # TOO FAST
    kp = 2.0  # TOO WEAK
    k_wall = 50.0  # TOO WEAK
    dt = 0.05
    frames = 400

    print("Task 1 Failure: Watch the red dot fly off the track.")

    # Init state
    x = spline[0].copy()
    v = np.zeros(2)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mask, cmap="gray")
    ax.plot(spline[:, 0], spline[:, 1], 'c--', alpha=0.3)

    robot_dot, = ax.plot([], [], 'ro', markersize=10, label="Robot")
    trail, = ax.plot([], [], 'r-', linewidth=1, alpha=0.5)
    ax.set_title(f"FAIL: Speed {vmax} > Wall Force Limit")
    ax.legend()

    path_x, path_y = [], []

    def update(frame):
        nonlocal x, v
        # Target moves fast
        target_idx = min(len(spline) - 1, int((frame / frames) * len(spline) * 1.5))

        x, v = rk4_step(x, v, frame, dt, spline, kp, 1.0, vmax, lambda t: target_idx, dt_map, 5.0, k_wall, 5.0)

        path_x.append(x[0])
        path_y.append(x[1])

        robot_dot.set_data([x[0]], [x[1]])
        trail.set_data(path_x, path_y)
        return robot_dot, trail

    ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()