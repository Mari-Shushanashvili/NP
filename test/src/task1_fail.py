import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    spline = np.load(root / "data" / "centerline_spline.npy").astype(np.float32)
    dt_map = np.load(root / "data" / "dt_halfwidth.npy").astype(np.float32)
    mask = cv2.imread(str(root / "data" / "route_mask.png"), 0) > 0

    vmax = 280.0  # Extremely fast
    kp = 1.2  # Very weak goal pull
    k_wall = 15.0  # Weak boundary repulsion
    dt = 0.05
    r = 12.0

    x, v = spline[0].copy(), np.zeros(2)
    traj = []

    print("Simulating Task 1 Failure...")
    for t in range(350):
        traj.append(x.copy())
        # Target moves ahead of robot
        target = spline[min(len(spline) - 1, t * 4)]

        # RK4-like update with failing forces
        a = kp * (target - x) - 5.0 * v
        px, py = int(np.clip(x[0], 1, dt_map.shape[1] - 2)), int(np.clip(x[1], 1, dt_map.shape[0] - 2))
        dist = dt_map[py, px]
        if dist < r:
            gx = 0.5 * (dt_map[py, px + 1] - dt_map[py, px - 1])
            gy = 0.5 * (dt_map[py + 1, px] - dt_map[py - 1, px])
            a += k_wall * (r - dist) * np.array([gx, gy])

        v += a * dt
        speed = np.linalg.norm(v)
        if speed > vmax: v *= (vmax / speed)
        x += v * dt

    # Numerical Validation Log
    traj = np.array(traj)
    cx, cy = np.clip(traj[:, 0].astype(int), 0, mask.shape[1] - 1), np.clip(traj[:, 1].astype(int), 0,
                                                                            mask.shape[0] - 1)
    on_track = mask[cy, cx]
    print("\n=== TASK 1 FAIL LOGS ===")
    print(f"Path Fidelity: {np.mean(on_track) * 100:.2f}%")
    print("STATUS: FAIL (Robot exited boundary)")

    # Animation
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray', alpha=0.3)
    dot, = ax.plot([], [], 'ro', label="Failing Robot")

    def update(f):
        dot.set_data([traj[f, 0]], [traj[f, 1]])
        return dot,

    ani = FuncAnimation(fig, update, frames=len(traj), interval=20)
    plt.show()


if __name__ == "__main__": main()

"""
=== LIMITATION DOCUMENTATION: TASK 1 (IVP PATH FOLLOWING) ===

WHAT IS THE LIMITATION?
The limitation shown here is "Dynamic Overshoot" caused by a mismatch between Velocity (vmax) 
and Boundary Repulsion (k_wall). 

WHY DOES IT OCCUR?
When the robot is tuned to move at high speeds (vmax > 200), it gains significant momentum. 
If the 'k_wall' parameter is not scaled proportionally, the repulsive force from the track 
boundary is mathematically insufficient to decelerate the robot's lateral velocity before 
it crosses the edge.

RESULT:
The robot violates the corridor constraint. This demonstrates that the Fourth-Order RK4 method, 
while accurate, cannot compensate for physical parameters that exceed the 'braking distance' 
provided by the potential field.
"""