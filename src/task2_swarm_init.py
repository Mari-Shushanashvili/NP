from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Geometry Helpers
def build_normals(centerline: np.ndarray):
    """
    Returns (tangents, normals) to match the calling convention.
    """
    M = len(centerline)
    tangents = np.zeros_like(centerline, dtype=np.float32)
    normals = np.zeros_like(centerline, dtype=np.float32)

    for i in range(M):
        i0 = max(0, i - 1)
        i1 = min(M - 1, i + 1)
        t = centerline[i1] - centerline[i0]
        t = t / (np.linalg.norm(t) + 1e-9)

        tangents[i] = t
        normals[i] = np.array([-t[1], t[0]], dtype=np.float32)

    return tangents, normals


def dt_at(dt_map: np.ndarray, xy: np.ndarray) -> float:
    h, w = dt_map.shape
    x = int(np.clip(round(float(xy[0])), 0, w - 1))
    y = int(np.clip(round(float(xy[1])), 0, h - 1))
    return float(dt_map[y, x])


def spawn_linear_convoy(
        spline: np.ndarray,
        dt_map: np.ndarray,
        idx_start: int,
        n_robots: int,
        lane_offset: float,
        r: float,
        spacing: float,
        direction: int  # +1 forward, -1 backward
):
    """
    Places robots in a line along the lane.
    """
    placed = []
    _, normals = build_normals(spline)

    for i in range(n_robots):
        # Calculate index step based on spacing (approximate)
        # We step backward from the start point to form a line behind the leader
        step = i * int(spacing)

        # Safe index
        idx = idx_start - (direction * step)
        idx = int(np.clip(idx, 0, len(spline) - 1))

        # Calculate Lane Position
        pos = spline[idx] + lane_offset * normals[idx]

        # Sanity check wall
        if dt_at(dt_map, pos) < r:
            print(f"Warning: Robot {i} got too close to wall.")

        placed.append(pos)

    return np.array(placed, dtype=np.float32)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    spline = np.load(data_dir / "centerline_spline.npy").astype(np.float32)
    dt_map = np.load(data_dir / "dt_halfwidth.npy").astype(np.float32)

    N_A, N_B = 5, 5
    r = 3.5
    d_safe = 8.0

    # Use the same logic as sim: lane_offset depends on d_safe
    lane_offset_mag = float(max(6.0, 0.8 * d_safe))

    # Initial Spacing (make it generous to prevent start collision)
    init_spacing_idx = 15  # ~15 pixels between robots along spline

    # Indices
    M = len(spline)
    idx_A = 20  # Start A slightly in
    idx_B = M - 21  # Start B slightly in

    print("Generating Clean Linear Convoy Init...")

    # Spawn Group A
    XA = spawn_linear_convoy(
        spline, dt_map, idx_A, N_A,
        lane_offset=lane_offset_mag, r=r, spacing=init_spacing_idx, direction=1
    )

    # Spawn Group B
    XB = spawn_linear_convoy(
        spline, dt_map, idx_B, N_B,
        lane_offset=-lane_offset_mag, r=r, spacing=init_spacing_idx, direction=-1
    )

    X0 = np.vstack([XA, XB]).astype(np.float32)
    V0 = np.zeros_like(X0, dtype=np.float32)

    out = data_dir / "task2_init.npy"
    np.save(out, {"X0": X0, "V0": V0, "N_A": N_A, "N_B": N_B, "d_safe": float(d_safe), "r": float(r)})
    print(f"Saved: {out}")

    # Debug
    mask_path = data_dir / "route_mask_inflated.png"
    if mask_path.exists():
        bg = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(10, 5))
        plt.imshow(bg, cmap='gray')
        plt.scatter(XA[:, 0], XA[:, 1], c='cyan', label='A')
        plt.scatter(XB[:, 0], XB[:, 1], c='orange', label='B')
        plt.legend()
        plt.title("Init Positions")
        plt.show()


if __name__ == "__main__":
    main()