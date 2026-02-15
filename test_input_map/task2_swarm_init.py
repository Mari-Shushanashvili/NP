import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


def build_normals(centerline: np.ndarray):
    M = len(centerline)
    normals = np.zeros_like(centerline, dtype=np.float32)
    tangents = np.zeros_like(centerline, dtype=np.float32)
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


def safe_dt(dt_map: np.ndarray, xy: np.ndarray, r: float, eps: float = 0.0) -> bool:
    return dt_at(dt_map, xy) >= float(r + eps)


def build_edge_hugging_lanes(centerline: np.ndarray, dt_map: np.ndarray, r: float,
                             edge_margin: float = 2.5, safety: float = 1.5):
    _, normals = build_normals(centerline)
    M = len(centerline)
    offsets = np.zeros((M,), dtype=np.float32)

    for i in range(M):
        w_half_local = dt_at(dt_map, centerline[i])
        max_offset = w_half_local - (r + edge_margin)
        offsets[i] = max(0.0, min(max_offset, max_offset - safety))

    laneA = (centerline + offsets[:, None] * normals).astype(np.float32)
    laneB = (centerline - offsets[:, None] * normals).astype(np.float32)
    return laneA, laneB, offsets


def tangent_normal(spline: np.ndarray, idx: int):
    M = len(spline)
    i0 = max(0, idx - 1)
    i1 = min(M - 1, idx + 1)
    t = spline[i1] - spline[i0]
    t = t / (np.linalg.norm(t) + 1e-9)
    n = np.array([-t[1], t[0]], dtype=np.float32)
    return t.astype(np.float32), n.astype(np.float32)


def spawn_cluster_near_endpoint(
    spline: np.ndarray,
    dt_map: np.ndarray,
    idx_center: int,
    n_robots: int,
    lane_sign: float,
    lane_offset_at_idx: float,
    r: float,
    d_spawn: float,
    seed: int,
    direction: int,
    eps_safe: float = 0.0,
):
    rng = np.random.default_rng(seed)
    placed = []

    row_steps = list(range(0, 30, 3))  # tight near endpoint
    lateral_mult = [0, 1, -1, 2, -2, 3, -3]
    tangential_mult = [0, 1, -1, 2, -2]

    for step in row_steps:
        idx = int(np.clip(idx_center + direction * step, 0, len(spline) - 1))
        base = spline[idx]
        t, n = tangent_normal(spline, idx)

        lane_center = base + lane_sign * lane_offset_at_idx * n

        for tm in tangential_mult:
            tang = (tm * 0.55 * d_spawn) + float(rng.uniform(-0.05 * d_spawn, 0.05 * d_spawn))
            for lm in lateral_mult:
                lat = (lm * d_spawn)
                cand = lane_center + lat * n + tang * t

                if not safe_dt(dt_map, cand, r, eps=eps_safe):
                    continue

                if placed:
                    arr = np.array(placed, dtype=np.float32)
                    dist = np.sqrt(((arr - cand) ** 2).sum(axis=1))
                    if np.any(dist < d_spawn):
                        continue

                placed.append(cand.astype(np.float32))
                if len(placed) >= n_robots:
                    return np.array(placed, dtype=np.float32)

    raise RuntimeError("Could not place robots near endpoint (try lower d_spawn or larger search range).")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    spline = np.load(data_dir / "centerline_spline.npy").astype(np.float32)
    dt_map = np.load(data_dir / "dt_halfwidth.npy").astype(np.float32)

    mask_path = data_dir / "route_mask_inflated.png"
    if not mask_path.exists():
        mask_path = data_dir / "route_mask.png"
    bg = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    route_bin = (bg > 0)

    # Settings matching swarm_sim
    N_A, N_B = 5, 5
    r = 3.5
    buffer_px = 1.0
    d_safe = 2.0 * r + buffer_px
    d_spawn = d_safe

    edge_margin = 2.5
    safety = 1.5
    laneA, laneB, offsets = build_edge_hugging_lanes(spline, dt_map, r=r, edge_margin=edge_margin, safety=safety)

    M = len(spline)
    idx_A = 10
    idx_B = M - 11

    # Use local offset at the endpoint indices so init matches sim lanes
    offA = float(offsets[idx_A])
    offB = float(offsets[idx_B])

    XA = spawn_cluster_near_endpoint(
        spline=spline, dt_map=dt_map, idx_center=idx_A, n_robots=N_A,
        lane_sign=+1.0, lane_offset_at_idx=offA, r=r, d_spawn=d_spawn,
        seed=1, direction=+1, eps_safe=0.0
    )
    XB = spawn_cluster_near_endpoint(
        spline=spline, dt_map=dt_map, idx_center=idx_B, n_robots=N_B,
        lane_sign=-1.0, lane_offset_at_idx=offB, r=r, d_spawn=d_spawn,
        seed=2, direction=-1, eps_safe=0.0
    )

    X0 = np.vstack([XA, XB]).astype(np.float32)
    V0 = np.zeros_like(X0, dtype=np.float32)

    out = data_dir / "task2_init.npy"
    np.save(out, {"X0": X0, "V0": V0, "N_A": N_A, "N_B": N_B, "d_safe": float(d_safe), "r": float(r)})
    print(f"Saved init state: {out}")

    # Plot with matching styles
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(route_bin, cmap="gray", alpha=0.25)
    ax.plot(laneA[:, 0], laneA[:, 1], linewidth=2)
    ax.plot(laneB[:, 0], laneB[:, 1], linewidth=2)

    ax.scatter(XA[:, 0], XA[:, 1], s=70, c=[(0.2, 0.5, 1.0)], edgecolors="black", label="Group A start (A)")
    ax.scatter(XB[:, 0], XB[:, 1], s=70, c=[(1.0, 0.55, 0.15)], edgecolors="black", label="Group B start (B)")

    ax.set_title("Task 2 init: lanes + tight start packs (matches simulation)")
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
