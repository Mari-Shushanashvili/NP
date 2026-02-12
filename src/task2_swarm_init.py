# import json
# from pathlib import Path
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
#
# # ------------------------------------------------------------
# # Step 0: Load artifacts
# # ------------------------------------------------------------
# def load_artifacts(project_root: Path):
#     spline_path = project_root / "data" / "centerline_spline.npy"
#     params_path = project_root / "data" / "corridor_params.json"
#     dt_path = project_root / "data" / "dt_halfwidth.npy"
#
#     if not spline_path.exists():
#         raise FileNotFoundError(f"Missing {spline_path} (run Task 1 Step 2)")
#     if not params_path.exists():
#         raise FileNotFoundError(f"Missing {params_path} (run Task 1 Step 3)")
#     if not dt_path.exists():
#         raise FileNotFoundError(f"Missing {dt_path} (run Task 1 Step 3)")
#
#     spline = np.load(spline_path).astype(np.float32)
#     params = json.loads(params_path.read_text(encoding="utf-8"))
#     dt = np.load(dt_path).astype(np.float32)
#
#     mask_vis_path = project_root / "data" / "route_mask_inflated.png"
#     if not mask_vis_path.exists():
#         mask_vis_path = project_root / "data" / "route_mask.png"
#
#     mask_vis = cv2.imread(str(mask_vis_path), cv2.IMREAD_GRAYSCALE)
#     if mask_vis is None:
#         raise FileNotFoundError(f"Could not read {mask_vis_path}")
#
#     route_bin = (mask_vis > 0)
#     return spline, params, dt, route_bin, mask_vis_path
#
#
# # ------------------------------------------------------------
# # Geometry helpers
# # ------------------------------------------------------------
# def tangent_and_normal(spline: np.ndarray, idx: int):
#     M = len(spline)
#     i0 = max(0, idx - 1)
#     i1 = min(M - 1, idx + 1)
#     t = spline[i1] - spline[i0]
#     nrm = np.linalg.norm(t) + 1e-9
#     t = t / nrm
#     n = np.array([-t[1], t[0]], dtype=np.float32)
#     return t, n
#
#
# def safe_at(dt: np.ndarray, xy, r, eps=0.0):
#     h, w = dt.shape
#     x = int(np.clip(round(float(xy[0])), 0, w - 1))
#     y = int(np.clip(round(float(xy[1])), 0, h - 1))
#     return dt[y, x] >= (r + eps)
#
#
# def min_pair_distance(X: np.ndarray) -> float:
#     N = X.shape[0]
#     dmin = float("inf")
#     for i in range(N):
#         d = X[i] - X[i + 1:]
#         if d.shape[0] == 0:
#             continue
#         rr = np.sqrt((d[:, 0] ** 2 + d[:, 1] ** 2))
#         m = float(rr.min())
#         if m < dmin:
#             dmin = m
#     return dmin if np.isfinite(dmin) else 0.0
#
#
# # ------------------------------------------------------------
# # Step 1: Lane-based init (endpoint-tight version)
# # ------------------------------------------------------------
# def spawn_group_near_endpoint_lanes(
#     spline: np.ndarray,
#     dt: np.ndarray,
#     r: float,
#     idx_center: int,
#     n_robots: int,
#     d_safe: float,
#     seed: int,
#     direction: int,
#     eps_safe: float = 0.0,
# ):
#     """
#     Deterministic lane-based placement with correct search direction.
#
#     direction:
#       +1 for A-side (start): search forward along spline (increasing index)
#       -1 for B-side (end)  : search backward along spline (decreasing index)
#
#     UPDATED: keeps robots CLOSE to endpoints by limiting row_steps.
#     """
#     rng = np.random.default_rng(seed)
#     placed = []
#
#     # ------------------------------------------------------------
#     # UPDATED: keep robots very close to the endpoint.
#     # Only a few rows near idx_center. No shuffle -> fills closest rows first.
#     # If you want even tighter, reduce 40 to 20.
#     # ------------------------------------------------------------
#     row_steps = list(range(0, 40, 4))  # 0,4,8,...,36
#
#     # Lane order across the corridor normal (center, then alternating sides)
#     lane_order = [0]
#     for k in range(1, 120):
#         lane_order.extend([k, -k])
#
#     for step in row_steps:
#         j = int(np.clip(idx_center + direction * step, 0, len(spline) - 1))
#         base = spline[j]
#         t, n = tangent_and_normal(spline, j)
#
#         # ------------------------------------------------------------
#         # UPDATED: smaller tangential jitter keeps cluster near endpoint
#         # ------------------------------------------------------------
#         tangential_jitter = float(rng.uniform(-0.15 * d_safe, 0.15 * d_safe))
#
#         for lk in lane_order:
#             cand = base + (lk * d_safe) * n + tangential_jitter * t
#
#             if not safe_at(dt, cand, r, eps=eps_safe):
#                 continue
#
#             if placed:
#                 arr = np.array(placed, dtype=np.float32)
#                 dist = np.sqrt(((arr - cand) ** 2).sum(axis=1))
#                 if np.any(dist < d_safe):
#                     continue
#
#             placed.append(cand.astype(np.float32))
#             if len(placed) >= n_robots:
#                 return np.array(placed, dtype=np.float32)
#
#     raise RuntimeError(
#         f"Could not place {n_robots} robots near idx={idx_center} (direction={direction}). "
#         f"Placed {len(placed)}. "
#         f"Try: reduce N, reduce buffer, reduce r, or increase row_steps range slightly."
#     )
#
#
# def main():
#     project_root = Path(__file__).resolve().parents[1]
#     spline, params, dt, route_bin, mask_vis_path = load_artifacts(project_root)
#
#     w_half = float(params["w_half_px"])
#
#     # ------------------------------------------------------------
#     # Task 2 override: smaller robots for multi-agent feasibility
#     # ------------------------------------------------------------
#     r = 3.0
#
#     # ------------------------------------------------------------
#     # UPDATED: 10 robots per side (as you requested)
#     # ------------------------------------------------------------
#     N_A = 10
#     N_B = 10
#
#     # Safety distance (no overlap + small buffer)
#     buffer_px = 1.0
#     d_safe = 2.0 * r + buffer_px
#
#     eps_safe = 0.0
#
#     print("\n=== Task 2 Step 0 (settings) ===")
#     print(f"Using mask for visualization: {mask_vis_path.name}")
#     if "inflate_px" in params:
#         print(f"corridor inflate_px: {params['inflate_px']}")
#     print(f"w_half = {w_half:.2f}, robot_radius r = {r:.2f}")
#     print(f"N_A = {N_A}, N_B = {N_B}, total = {N_A + N_B}")
#     print(f"d_safe = 2r + buffer = {d_safe:.2f} (buffer={buffer_px})")
#
#     # Endpoint indices (slightly inside)
#     idx_A = 10
#     idx_B = len(spline) - 11
#
#     # A-side searches forward (+1), B-side searches backward (-1)
#     XA = spawn_group_near_endpoint_lanes(
#         spline=spline, dt=dt, r=r,
#         idx_center=idx_A, n_robots=N_A,
#         d_safe=d_safe, seed=1, direction=+1, eps_safe=eps_safe
#     )
#
#     XB = spawn_group_near_endpoint_lanes(
#         spline=spline, dt=dt, r=r,
#         idx_center=idx_B, n_robots=N_B,
#         d_safe=d_safe, seed=2, direction=-1, eps_safe=eps_safe
#     )
#
#     X0 = np.vstack([XA, XB]).astype(np.float32)
#     V0 = np.zeros_like(X0, dtype=np.float32)
#
#     dmin = min_pair_distance(X0)
#
#     print("\n=== Task 2 Step 1 (init check) ===")
#     print(f"Min pair distance at t=0: {dmin:.2f} px (need >= {d_safe:.2f})")
#     if dmin < d_safe:
#         print("WARNING: Some robots are too close.")
#     else:
#         print("PASS ✅ Initial configuration has no overlaps.")
#
#     # Save init state for reproducibility
#     out = project_root / "data" / "task2_init.npy"
#     np.save(out, {"X0": X0, "V0": V0, "N_A": N_A, "N_B": N_B, "d_safe": d_safe, "r": r})
#     print("Saved init state:", out)
#
#     # Debug plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.imshow(route_bin, cmap="gray", alpha=0.25)
#     ax.plot(spline[:, 0], spline[:, 1], linewidth=2)
#
#     ax.scatter(XA[:, 0], XA[:, 1], s=70, label="Group A (A→B)")
#     ax.scatter(XB[:, 0], XB[:, 1], s=70, label="Group B (B→A)")
#
#     ax.set_title("Task 2 Step 1: Initial robot positions (tight to endpoints)")
#     ax.axis("off")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()

import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    spline = np.load(data_dir / "centerline_spline.npy").astype(np.float32)
    params = json.loads((data_dir / "corridor_params.json").read_text())
    dt = np.load(data_dir / "dt_halfwidth.npy").astype(np.float32)
    bg = cv2.imread(str(data_dir / "route_mask_inflated.png"), 0)

    # UPDATED: Set exactly 5 per side as requested
    N_A, N_B = 5, 5
    r = 3.5
    d_safe = 4.0 * r  # Spaced out for convoy stability
    M = len(spline)

    def spawn(start_idx, direction):
        pts = []
        # Place robots in a neat line on their respective lane
        for step in range(0, 150, 15):
            idx = int(np.clip(start_idx + direction * step, 0, M - 1))
            base = spline[idx]
            # Lane offset (A is +10, B is -10)
            i0, i1 = max(0, idx - 1), min(M - 1, idx + 1)
            t = (spline[i1] - spline[i0]) / (np.linalg.norm(spline[i1] - spline[i0]) + 1e-9)
            n = np.array([-t[1], t[0]])
            cand = base + (direction * 10.0) * n
            pts.append(cand)
            if len(pts) >= N_A: break
        return np.array(pts)

    XA = spawn(10, 1)
    XB = spawn(M - 11, -1)

    X0 = np.vstack([XA, XB]).astype(np.float32)
    V0 = np.zeros_like(X0)

    # Save so swarm_sim.py can load the correct positions
    np.save(data_dir / "task2_init.npy", {"X0": X0, "V0": V0, "N_A": N_A, "N_B": N_B, "d_safe": d_safe, "r": r})

    print(f"Initialized {N_A + N_B} robots (5 per side).")
    plt.imshow(bg, cmap='gray', alpha=0.3)
    plt.scatter(XA[:, 0], XA[:, 1], c='cyan')
    plt.scatter(XB[:, 0], XB[:, 1], c='orange')
    plt.show()


if __name__ == "__main__":
    main()