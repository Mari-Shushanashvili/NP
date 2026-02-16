import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


# Utilities
def load_config(project_root: Path):
    cfg_path = project_root / "data" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    A = tuple(cfg["A"])  # (x,y)
    B = tuple(cfg["B"])
    return cfg, A, B


def load_route_mask(project_root: Path):
    mask_path = project_root / "data" / "route_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read {mask_path}")
    # binary boolean
    return (mask > 0), mask_path


def nearest_true_pixel(bin_img: np.ndarray, xy):
    """
    Find the closest True pixel in a boolean image to a given (x,y).
    Returns (x,y) int.
    """
    x, y = xy
    ys, xs = np.where(bin_img)
    if len(xs) == 0:
        raise ValueError("Binary image contains no True pixels.")
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    k = int(np.argmin(d2))
    return int(xs[k]), int(ys[k])


def bfs_shortest_path_8n(skel: np.ndarray, start_xy, goal_xy):
    """
    BFS on 8-neighborhood for skeleton pixels (boolean array).
    Returns ordered path as list of (x,y) from start to goal.
    """
    h, w = skel.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    # Parent pointers: store predecessor for each visited pixel
    # Use -1 to mean "unvisited"
    parent_x = -np.ones((h, w), dtype=np.int32)
    parent_y = -np.ones((h, w), dtype=np.int32)

    from collections import deque
    q = deque()
    q.append((sx, sy))
    parent_x[sy, sx] = sx
    parent_y[sy, sx] = sy

    nbrs = [(-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)]

    found = False
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            found = True
            break

        for dx, dy in nbrs:
            xn, yn = x + dx, y + dy
            if 0 <= xn < w and 0 <= yn < h:
                if not skel[yn, xn]:
                    continue
                if parent_x[yn, xn] != -1:
                    continue
                parent_x[yn, xn] = x
                parent_y[yn, xn] = y
                q.append((xn, yn))

    if not found:
        raise RuntimeError("No skeleton path found between start and goal.")

    # Reconstruct
    path = []
    x, y = gx, gy
    while True:
        path.append((x, y))
        px, py = parent_x[y, x], parent_y[y, x]
        if (px, py) == (x, y):
            break
        x, y = int(px), int(py)

    path.reverse()
    return path


def downsample_path(path_xy, step=3):
    """
    Keep every 'step'-th point to reduce noise/jaggies.
    """
    if step <= 1:
        return path_xy
    return path_xy[::step] + ([path_xy[-1]] if path_xy[-1] != path_xy[::step][-1] else [])


def fit_spline_from_pixels(path_xy, smooth=3.0, num_samples=1200):
    """
    Fit parametric spline x(u), y(u) through ordered pixels.
    Returns dense spline samples as (num_samples, 2) array.
    """
    pts = np.array(path_xy, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    # Parameterize by cumulative arclength
    dx = np.diff(x)
    dy = np.diff(y)
    s = np.hstack([0.0, np.cumsum(np.sqrt(dx*dx + dy*dy))])
    if s[-1] == 0:
        raise ValueError("Path length is zero; cannot fit spline.")
    u = s / s[-1]

    # Fit spline
    tck, _ = splprep([x, y], u=u, s=smooth)
    uu = np.linspace(0.0, 1.0, num_samples)
    xs, ys = splev(uu, tck)
    spline_pts = np.stack([xs, ys], axis=1)
    return spline_pts


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg, A, B = load_config(project_root)
    route_bin, mask_path = load_route_mask(project_root)

    # 1) Skeletonize (centerline, 1px wide)
    skel = skeletonize(route_bin)  # boolean

    # 2) Snap A and B to nearest skeleton pixels (robust)
    A_s = nearest_true_pixel(skel, A)
    B_s = nearest_true_pixel(skel, B)

    # 3) Shortest path along skeleton pixels
    path = bfs_shortest_path_8n(skel, A_s, B_s)

    # 4) Downsample to reduce zigzag
    path_ds = downsample_path(path, step=4)

    # 5) Fit spline + dense sampling
    spline_pts = fit_spline_from_pixels(path_ds, smooth=2.0, num_samples=1200)

    # Save outputs
    out_centerline = project_root / "data" / "centerline_pixels.npy"
    out_spline = project_root / "data" / "centerline_spline.npy"
    np.save(out_centerline, np.array(path, dtype=np.int32))
    np.save(out_spline, spline_pts.astype(np.float32))

    print("Saved:", out_centerline)
    print("Saved:", out_spline)
    print("A_skeleton:", A_s, "B_skeleton:", B_s, "Path length (pixels):", len(path))

    # Debug visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(route_bin, cmap="gray")
    ax.set_title("Route mask + skeleton path (red) + spline (cyan)")
    ax.axis("off")

    # Plot path (red)
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, linewidth=2)

    # Plot spline (cyan)
    ax.plot(spline_pts[:, 0], spline_pts[:, 1], linewidth=2)

    # Mark A and B (original + snapped)
    ax.scatter([A[0], B[0]], [A[1], B[1]], s=60, marker="o")
    ax.scatter([A_s[0], B_s[0]], [A_s[1], B_s[1]], s=60, marker="x")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
