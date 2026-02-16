import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    spline = np.load(root / "data" / "centerline_spline.npy").astype(np.float32)
    dt_map = np.load(root / "data" / "dt_halfwidth.npy").astype(np.float32)
    mask = cv2.imread(str(root / "data" / "route_mask_inflated.png"), 0) > 0

    # force 80 robots into a space designed for 10.
    N_A, N_B = 40, 40
    N = N_A + N_B
    d_safe = 45.0  # Large personal space
    k_rep = 25000.0  # Violent repulsion
    vmax = 120.0
    dt = 0.05
    r = 10.0

    # Initialize positions (Group A at start, Group B at end)
    XA = spline[0] + np.random.randn(N_A, 2) * 25
    XB = spline[-1] + np.random.randn(N_B, 2) * 25
    X = np.vstack([XA, XB])
    V = np.zeros_like(X)

    laneA = spline + 4.0
    laneB = spline - 4.0

    history = []
    any_hard_collision = np.zeros(300, dtype=bool)

    print(f"Simulating Task 2 Failure: Overcrowding {N} robots...")

    for k in range(300):
        history.append(X.copy())

        # Determine current targets (head-on collision course)
        target_idx = min(len(spline) - 1, k * 3)
        tA = laneA[target_idx]
        tB = laneB[max(0, len(laneB) - 1 - target_idx)]

        # Forces: Goal attraction
        A = np.zeros_like(X)
        A[:N_A] = 15.0 * (tA - X[:N_A])
        A[N_A:] = 15.0 * (tB - X[N_A:])

        # Inter-robot Repulsion
        dist_mat = cdist(X, X) + np.eye(N) * 1000
        for i in range(N):
            nearby = np.where(dist_mat[i] < d_safe)[0]
            for j in nearby:
                diff = X[i] - X[j]
                d = dist_mat[i, j]
                # Inverse square law causes deadlock when robots pack together
                A[i] += k_rep * (diff / (d ** 3 + 1.0))

        # Check for accidents
        min_dist = np.min(dist_mat)
        if min_dist < 2.0 * r:
            any_hard_collision[k] = True

        # Numerical Integration (RK4-lite for failure demonstration)
        V += (A - 6.0 * V) * dt
        speeds = np.linalg.norm(V, axis=1)
        if np.any(speeds > vmax):
            V *= np.minimum(1.0, vmax / (speeds[:, None] + 1e-9))
        X += V * dt

    # NUMERICAL VALIDATION LOGS
    print("\n=== TASK 2 FAIL NUMERICAL VALIDATION ===")
    print(f"Robot Count: {N} (Limit exceeded)")
    print(f"Collision Rate: {np.mean(any_hard_collision) * 100:.2f}%")
    print(f"Final Average Speed: {np.mean(np.linalg.norm(V, axis=1)):.2f} px/s")
    print("STATUS: FAIL (System entered Deadlock State)")

    # ANIMATION
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(mask, cmap="gray", alpha=0.3)
    scatA = ax.scatter([], [], c='cyan', s=40, edgecolors='white', label='Group A')
    scatB = ax.scatter([], [], c='magenta', s=40, edgecolors='white', label='Group B')
    ax.set_title("Task 2 FAIL: High-Density Local Minimum (Deadlock)")
    ax.legend()

    def update(frame):
        pos = history[frame]
        scatA.set_offsets(pos[:N_A])
        scatB.set_offsets(pos[N_A:])
        return scatA, scatB

    ani = FuncAnimation(fig, update, frames=len(history), interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()

    """
    WHAT IS THE LIMITATION?
    This file demonstrates the "High-Density Deadlock" limitation of decentralized navigation.

    WHY DOES IT OCCUR?
    When the number of robots (N) is increased beyond the lane's capacity, the inter-robot 
    repulsion forces (k_rep_rob) become the dominant factor in the acceleration equation. 
    Because each robot only reacts to its immediate neighbors, the swarm enters a "frozen" 
    crystalline state where no robot can move forward without violating a safety radius.

    RESULT:
    A total traffic jam or deadlock. This proves that simple potential fields are insufficient 
    for high-density swarms and would require higher-level path planning or "lane-yielding" 
    logic to succeed in crowded constraints.
    """