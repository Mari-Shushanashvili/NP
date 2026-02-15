from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist


def main():
    root = Path(__file__).resolve().parents[1]
    spline = np.load(root / "data" / "centerline_spline.npy").astype(np.float32)
    mask = cv2.imread(str(root / "data" / "route_mask_inflated.png"), 0) > 0

    # --- FAILURE PARAMETERS ---
    N_A = 25  # CROWD
    N_B = 25  # CROWD
    N = N_A + N_B
    frames = 300

    print("Task 2 Failure: Watch the traffic jam/deadlock in the center.")

    # Initialize in messy piles near ends
    XA = spline[0] + np.random.randn(N_A, 2) * 15
    XB = spline[-1] + np.random.randn(N_B, 2) * 15
    X = np.vstack([XA, XB])
    V = np.zeros_like(X)

    # Create single lanes (head-on collision course)
    normals = np.zeros_like(spline)  # Dummy normals for simple targets
    laneA = spline + 5.0  # Slight offset
    laneB = spline - 5.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(mask, cmap="gray", alpha=0.5)

    scatA = ax.scatter([], [], c='cyan', s=50, edgecolors='k', label='Group A')
    scatB = ax.scatter([], [], c='orange', s=50, edgecolors='k', label='Group B')
    ax.set_title(f"FAIL: N={N} (Deadlock / Overcrowding)")
    ax.legend()
    ax.axis('off')

    def update(frame):
        nonlocal X, V

        A = np.zeros_like(X)

        # Simple targets
        tA = laneA[min(len(laneA) - 1, frame * 3)]
        tB = laneB[max(0, len(laneB) - 1 - frame * 3)]

        A[:N_A] += 2.0 * (tA - X[:N_A])
        A[N_A:] += 2.0 * (tB - X[N_A:])

        # PANIC REPULSION
        D = cdist(X, X) + np.eye(N) * 1e5
        # Interaction radius 15px
        cols = np.where(D < 15.0)
        for i, j in zip(cols[0], cols[1]):
            diff = X[i] - X[j]
            dist = D[i, j]
            # Massive push
            A[i] += 5000.0 * (diff / (dist ** 2 + 0.1))

        # Damping
        A -= 5.0 * V

        # Euler Step
        V += A * 0.05
        spd = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        V = V * np.minimum(1.0, 15.0 / spd)
        X += V * 0.05

        scatA.set_offsets(X[:N_A])
        scatB.set_offsets(X[N_A:])
        return scatA, scatB

    ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()