# import json
# from pathlib import Path
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
#
# # ============================================================
# # Performance toggles
# # ============================================================
# DEBUG_HEAVY_METRICS = False
# METRICS_EVERY = 10
#
#
# # ------------------------------------------------------------
# # Load artifacts
# # ------------------------------------------------------------
# def load_spline(project_root: Path):
#     p = project_root / "data" / "centerline_spline.npy"
#     if not p.exists():
#         raise FileNotFoundError(f"Missing {p} (run Task 1 Step 2)")
#     return np.load(p).astype(np.float32), p
#
#
# def load_dt(project_root: Path):
#     p = project_root / "data" / "dt_halfwidth.npy"
#     if not p.exists():
#         raise FileNotFoundError(f"Missing {p} (run Task 1 Step 3)")
#     return np.load(p).astype(np.float32), p
#
#
# def load_corridor_params(project_root: Path):
#     p = project_root / "data" / "corridor_params.json"
#     if not p.exists():
#         raise FileNotFoundError(f"Missing {p} (run Task 1 Step 3)")
#     return json.loads(p.read_text(encoding="utf-8")), p
#
#
# def load_mask_for_vis(project_root: Path):
#     p = project_root / "data" / "route_mask_inflated.png"
#     if not p.exists():
#         p = project_root / "data" / "route_mask.png"
#     img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Could not read {p}")
#     return (img > 0), p
#
#
# # ------------------------------------------------------------
# # Lane building from centerline (normal offset)
# # ------------------------------------------------------------
# def build_lanes_from_centerline(spline: np.ndarray, lane_offset: float):
#     M = len(spline)
#     normals = np.zeros_like(spline, dtype=np.float32)
#
#     for i in range(M):
#         i0 = max(0, i - 1)
#         i1 = min(M - 1, i + 1)
#         t = spline[i1] - spline[i0]
#         t = t / (np.linalg.norm(t) + 1e-9)
#         n = np.array([-t[1], t[0]], dtype=np.float32)
#         normals[i] = n
#
#     laneA = spline + lane_offset * normals
#     laneB = spline - lane_offset * normals
#     return laneA.astype(np.float32), laneB.astype(np.float32)
#
#
# # ------------------------------------------------------------
# # LOCAL projection to lane
# # ------------------------------------------------------------
# def project_to_lane_local(xy: np.ndarray, lane_pts: np.ndarray, idx_hint: int, win: int = 30):
#     M = lane_pts.shape[0]
#     lo = max(0, idx_hint - win)
#     hi = min(M, idx_hint + win + 1)
#
#     seg = lane_pts[lo:hi]
#     d2 = np.sum((seg - xy[None, :]) ** 2, axis=1)
#     k = int(np.argmin(d2))
#     idx_best = lo + k
#     return lane_pts[idx_best].astype(np.float32), idx_best
#
#
# # ------------------------------------------------------------
# # Geometry: curvature + arclength
# # ------------------------------------------------------------
# def compute_ds_and_arclen(pts: np.ndarray):
#     d = pts[1:] - pts[:-1]
#     ds = np.sqrt((d ** 2).sum(axis=1)).astype(np.float32)
#     s = np.zeros((len(pts),), dtype=np.float32)
#     s[1:] = np.cumsum(ds)
#     return ds, s
#
#
# def curvature_discrete(pts: np.ndarray):
#     M = len(pts)
#     kappa = np.zeros((M,), dtype=np.float32)
#
#     for i in range(1, M - 1):
#         p0 = pts[i - 1]
#         p1 = pts[i]
#         p2 = pts[i + 1]
#
#         a = float(np.linalg.norm(p1 - p0) + 1e-9)
#         b = float(np.linalg.norm(p2 - p1) + 1e-9)
#         c = float(np.linalg.norm(p2 - p0) + 1e-9)
#
#         v1 = p1 - p0
#         v2 = p2 - p0
#         area2 = float(abs(v1[0] * v2[1] - v1[1] * v2[0]))
#
#         kappa[i] = (2.0 * area2) / (a * b * c + 1e-9)
#
#     if M >= 2:
#         kappa[0] = kappa[1]
#         kappa[-1] = kappa[-2]
#     return kappa
#
#
# # ------------------------------------------------------------
# # Swarm init: circle formation
# # ------------------------------------------------------------
# def circle_formation(center_xy: np.ndarray, n: int, radius: float, phase: float = 0.0):
#     if n <= 0:
#         return np.zeros((0, 2), dtype=np.float32)
#     if n == 1:
#         return center_xy[None, :].astype(np.float32)
#
#     angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
#     angles = angles + phase
#     pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * float(radius)
#     return (center_xy[None, :] + pts).astype(np.float32)
#
#
# # ------------------------------------------------------------
# # Forces
# # ------------------------------------------------------------
# def pairwise_repulsion(X: np.ndarray, d_safe: float, k_rep: float):
#     N = X.shape[0]
#     F = np.zeros_like(X, dtype=np.float32)
#     eps = 1e-6
#
#     for i in range(N):
#         for j in range(i + 1, N):
#             dvec = X[i] - X[j]
#             dist = float(np.sqrt(dvec[0] ** 2 + dvec[1] ** 2) + eps)
#             if dist >= d_safe:
#                 continue
#             s = (d_safe - dist) / d_safe
#             mag = k_rep * (s ** 2) / dist
#             push = (mag * dvec).astype(np.float32)
#             F[i] += push
#             F[j] -= push
#
#     return F
#
#
# def wall_force_from_dt(xy: np.ndarray, dt: np.ndarray, r: float, k_wall: float, margin: float):
#     h, w = dt.shape
#     x = int(np.clip(round(float(xy[0])), 1, w - 2))
#     y = int(np.clip(round(float(xy[1])), 1, h - 2))
#
#     d = float(dt[y, x])
#     if d >= (r + margin):
#         return np.zeros(2, dtype=np.float32)
#
#     gx = float(dt[y, x + 1] - dt[y, x - 1]) * 0.5
#     gy = float(dt[y + 1, x] - dt[y - 1, x]) * 0.5
#     g = np.array([gx, gy], dtype=np.float32)
#     g = g / (float(np.linalg.norm(g)) + 1e-9)
#
#     pen = (r + margin) - d
#     return (k_wall * pen * g).astype(np.float32)
#
#
# # ------------------------------------------------------------
# # Pure pursuit helper
# # ------------------------------------------------------------
# def lookahead_index_from_distance(s_arclen: np.ndarray, idx0: int, L: float, direction: int):
#     M = len(s_arclen)
#     s0 = float(s_arclen[idx0])
#
#     if direction == +1:
#         target_s = s0 + L
#         idx = int(np.searchsorted(s_arclen, target_s, side="left"))
#         return int(np.clip(idx, 0, M - 1))
#     else:
#         target_s = s0 - L
#         idx = int(np.searchsorted(s_arclen, target_s, side="right") - 1)
#         return int(np.clip(idx, 0, M - 1))
#
#
# # ------------------------------------------------------------
# # RK4 step with: adaptive lookahead, cohesion, alignment,
# # AND a "formation spring" term to prevent later spreading.
# # ------------------------------------------------------------
# def rk4_step_swarm(
#     X, V, dt,
#     laneA, laneB, sA, sB, kappaA, kappaB,
#     N_A,
#     kp, kd,
#     vmax_base, alpha_curv,
#     lookahead_L, lookahead_L_min, lookahead_curv_gain,
#     d_safe, k_rep,
#     dt_map, r, k_wall, wall_margin,
#     k_lane, lane_idx, proj_win,
#     k_coh, k_align,
#     k_form, form_radius
# ):
#     def compute_targets_limits_and_projection(Xi, Vi, lane_idx_local):
#         N = Xi.shape[0]
#         T = np.zeros_like(Xi, dtype=np.float32)
#         Pproj = np.zeros_like(Xi, dtype=np.float32)
#         vlim = np.zeros((N,), dtype=np.float32)
#
#         for i in range(N):
#             if i < N_A:
#                 lane = laneA
#                 s = sA
#                 kappa = kappaA
#                 direction = +1
#             else:
#                 lane = laneB
#                 s = sB
#                 kappa = kappaB
#                 direction = -1
#
#             P, idx_proj = project_to_lane_local(Xi[i], lane, int(lane_idx_local[i]), win=proj_win)
#             lane_idx_local[i] = idx_proj
#             Pproj[i] = P
#
#             curv_here = float(abs(kappa[idx_proj]))
#             L_eff = float(lookahead_L / (1.0 + lookahead_curv_gain * curv_here))
#             L_eff = float(np.clip(L_eff, lookahead_L_min, lookahead_L))
#             idx_t = lookahead_index_from_distance(s, idx_proj, L_eff, direction)
#             T[i] = lane[idx_t]
#
#             vlim[i] = float(vmax_base / (1.0 + alpha_curv * curv_here))
#
#         return T, vlim, Pproj, lane_idx_local
#
#     def xdot(Vi, vlim):
#         out = Vi.copy()
#         for i in range(out.shape[0]):
#             sp = float(np.linalg.norm(out[i]) + 1e-9)
#             vmax_i = float(vlim[i])
#             if sp > vmax_i:
#                 out[i] *= (vmax_i / sp)
#         return out
#
#     def accel(Xi, Vi, lane_idx_local):
#         T, vlim, Pproj, lane_idx_local = compute_targets_limits_and_projection(Xi, Vi, lane_idx_local)
#
#         A = kp * (T - Xi) - kd * Vi
#
#         # Group stats
#         cA = Xi[:N_A].mean(axis=0)
#         cB = Xi[N_A:].mean(axis=0)
#         vA = Vi[:N_A].mean(axis=0)
#         vB = Vi[N_A:].mean(axis=0)
#
#         # Cohesion + alignment
#         A[:N_A] += k_coh * (cA - Xi[:N_A]) + k_align * (vA - Vi[:N_A])
#         A[N_A:] += k_coh * (cB - Xi[N_A:]) + k_align * (vB - Vi[N_A:])
#
#         # Formation spring: keep each robot near a small ring around the centroid
#         # (prevents later spreading / stretching)
#         def formation_spring(Xg, cg):
#             d = Xg - cg[None, :]
#             dist = np.linalg.norm(d, axis=1) + 1e-9
#             # radial error relative to desired cluster radius
#             err = (dist - form_radius)
#             # pull inward/outward along radial direction
#             return (-k_form * err[:, None] * (d / dist[:, None])).astype(np.float32)
#
#         A[:N_A] += formation_spring(Xi[:N_A], cA)
#         A[N_A:] += formation_spring(Xi[N_A:], cB)
#
#         # Repulsion
#         A += pairwise_repulsion(Xi, d_safe=d_safe, k_rep=k_rep)
#
#         # Wall + rail
#         for i in range(Xi.shape[0]):
#             A[i] += wall_force_from_dt(Xi[i], dt_map, r=r, k_wall=k_wall, margin=wall_margin)
#             A[i] += k_lane * (Pproj[i] - Xi[i])
#
#         return A, vlim, lane_idx_local
#
#     lane_idx1 = lane_idx.copy()
#     k1v, vlim1, lane_idx1 = accel(X, V, lane_idx1)
#     k1x = xdot(V, vlim1)
#
#     lane_idx2 = lane_idx.copy()
#     X2 = X + 0.5 * dt * k1x
#     V2 = V + 0.5 * dt * k1v
#     k2v, vlim2, lane_idx2 = accel(X2, V2, lane_idx2)
#     k2x = xdot(V2, vlim2)
#
#     lane_idx3 = lane_idx.copy()
#     X3 = X + 0.5 * dt * k2x
#     V3 = V + 0.5 * dt * k2v
#     k3v, vlim3, lane_idx3 = accel(X3, V3, lane_idx3)
#     k3x = xdot(V3, vlim3)
#
#     lane_idx4 = lane_idx.copy()
#     X4 = X + dt * k3x
#     V4 = V + dt * k3v
#     k4v, vlim4, lane_idx4 = accel(X4, V4, lane_idx4)
#     k4x = xdot(V4, vlim4)
#
#     X_new = X + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
#     V_new = V + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
#
#     lane_idx[:] = lane_idx4[:]
#     return X_new.astype(np.float32), V_new.astype(np.float32), lane_idx
#
#
# # ------------------------------------------------------------
# # Optional heavy metrics
# # ------------------------------------------------------------
# def compute_dt_values(X: np.ndarray, dt_map: np.ndarray):
#     h, w = dt_map.shape
#     dvals = np.zeros((X.shape[0],), dtype=np.float32)
#     for i in range(X.shape[0]):
#         x = int(np.clip(round(float(X[i, 0])), 0, w - 1))
#         y = int(np.clip(round(float(X[i, 1])), 0, h - 1))
#         dvals[i] = dt_map[y, x]
#     return dvals
#
#
# def min_pair_distance(X: np.ndarray) -> float:
#     m = float("inf")
#     N = X.shape[0]
#     for i in range(N):
#         for j in range(i + 1, N):
#             dv = X[i] - X[j]
#             d = float(np.sqrt(dv[0] ** 2 + dv[1] ** 2))
#             if d < m:
#                 m = d
#     return m if np.isfinite(m) else 0.0
#
#
# # ------------------------------------------------------------
# # Main
# # ------------------------------------------------------------
# def main():
#     project_root = Path(__file__).resolve().parents[1]
#
#     spline, _ = load_spline(project_root)
#     dt_map, _ = load_dt(project_root)
#     params, _ = load_corridor_params(project_root)
#     route_bin, mask_used = load_mask_for_vis(project_root)
#
#     # ------------------------------------------------------------
#     # Force 5 robots per side (user request)
#     # ------------------------------------------------------------
#     N_A = 5
#     N_B = 5
#     N = N_A + N_B
#
#     # Use your corridor/robot settings from params/init equivalents
#     # We keep your old defaults, but you can tune if needed:
#     d_safe = 14.0
#     r = 4.0
#
#     w_half = float(params["w_half_px"])
#     eps = 0.3
#     w_half_eff = w_half - eps
#
#     # faster sim
#     dt = 0.06  # slightly bigger than before for speed (still stable with our damping)
#
#     # lanes
#     lane_offset_max = 0.9 * (w_half_eff - r)
#     lane_offset = min(0.75 * d_safe, lane_offset_max)
#     laneA, laneB = build_lanes_from_centerline(spline, lane_offset)
#
#     # direction fix: if spline runs right->left, reverse lanes
#     global_dir = spline[-1] - spline[0]
#     if float(global_dir[0]) < 0.0:
#         laneA = laneA[::-1].copy()
#         laneB = laneB[::-1].copy()
#
#     _, sA = compute_ds_and_arclen(laneA)
#     _, sB = compute_ds_and_arclen(laneB)
#     kappaA = curvature_discrete(laneA)
#     kappaB = curvature_discrete(laneB)
#
#     # start circles at endpoints
#     centerA = laneA[0].copy()
#     centerB = laneB[-1].copy()
#
#     max_rad = max(1.0, (w_half_eff - r - 2.0))
#     radius = min(0.75 * d_safe, 0.7 * max_rad)
#
#     XA0 = circle_formation(centerA, N_A, radius=radius, phase=0.0)
#     XB0 = circle_formation(centerB, N_B, radius=radius, phase=np.pi / 10.0)
#     X = np.vstack([XA0, XB0]).astype(np.float32)
#     V = np.zeros_like(X, dtype=np.float32)
#
#     # controller (faster + stickier)
#     kp = 12.0
#     kd = 6.0            # a bit less damping so they move quicker
#     k_coh = 1.4         # stronger cohesion
#     k_align = 0.55      # stronger alignment reduces worm/stretch
#     k_form = 2.2        # formation spring gain (new)
#     form_radius = 0.55 * radius  # keep group compact during whole route
#
#     vmax_base = 10.0    # faster
#     alpha_curv = 40.0   # still slows in bends
#     lookahead_L = 22.0
#     lookahead_L_min = 8.0
#     lookahead_curv_gain = 40.0
#
#     k_rep = 900.0       # slightly less repulsion so cohesion wins
#     k_lane = 95.0       # stronger rail -> stays on lane
#     proj_win = 20
#     k_wall = 120.0
#     wall_margin = 4.0
#
#     lane_idx = np.zeros((N,), dtype=np.int32)
#     lane_idx[:N_A] = 0
#     lane_idx[N_A:] = len(laneB) - 1
#
#     # frames upper bound + early stop
#     path_len = float(sA[-1])
#     v_eff = 0.65 * vmax_base
#     frames = int(path_len / (max(v_eff, 1e-3) * dt)) + 250
#     frames = max(frames, 800)
#
#     traj = np.zeros((frames, N, 2), dtype=np.float32)
#
#     for k in range(frames):
#         traj[k] = X
#
#         X, V, lane_idx = rk4_step_swarm(
#             X, V, dt,
#             laneA=laneA, laneB=laneB,
#             sA=sA, sB=sB,
#             kappaA=kappaA, kappaB=kappaB,
#             N_A=N_A,
#             kp=kp, kd=kd,
#             vmax_base=vmax_base, alpha_curv=alpha_curv,
#             lookahead_L=lookahead_L,
#             lookahead_L_min=lookahead_L_min,
#             lookahead_curv_gain=lookahead_curv_gain,
#             d_safe=d_safe, k_rep=k_rep,
#             dt_map=dt_map, r=r, k_wall=k_wall, wall_margin=wall_margin,
#             k_lane=k_lane, lane_idx=lane_idx, proj_win=proj_win,
#             k_coh=k_coh, k_align=k_align,
#             k_form=k_form, form_radius=form_radius
#         )
#
#         doneA = bool(np.all(lane_idx[:N_A] >= (len(laneA) - 5)))
#         doneB = bool(np.all(lane_idx[N_A:] <= 4))
#         slow = bool(np.linalg.norm(V, axis=1).mean() < 0.25)
#         if doneA and doneB and slow:
#             traj = traj[:k + 1]
#             break
#
#     # ------------------------------------------------------------
#     # Animation visuals
#     # ------------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(12, 6))
#     fig.patch.set_facecolor("black")
#     ax.set_facecolor("black")
#
#     ax.imshow(route_bin, cmap="gray", alpha=0.12)
#     ax.plot(laneA[:, 0], laneA[:, 1], linewidth=1.4, color="white", alpha=0.9)
#     ax.plot(laneB[:, 0], laneB[:, 1], linewidth=1.4, color="white", alpha=0.9)
#
#     ax.set_title("Task 2: Two-lane swarm (Adaptive Pure Pursuit + Formation Lock)", color="white")
#     ax.axis("off")
#
#     # 5 dots per side, outlined black border
#     dot_size = 28
#     colorA = "#00FFFF"   # cyan
#     colorB = "#FF4D00"   # orange-red
#
#     scatA = ax.scatter(
#         traj[0, :N_A, 0], traj[0, :N_A, 1],
#         s=dot_size, c=colorA, edgecolors="black", linewidths=0.9, label="Group A (A→B)"
#     )
#     scatB = ax.scatter(
#         traj[0, N_A:, 0], traj[0, N_A:, 1],
#         s=dot_size, c=colorB, edgecolors="black", linewidths=0.9, label="Group B (B→A)"
#     )
#
#     leg = ax.legend(loc="upper right")
#     for text in leg.get_texts():
#         text.set_color("white")
#
#     info_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, color="white")
#
#     def update(frame):
#         XA = traj[frame, :N_A]
#         XB = traj[frame, N_A:]
#         scatA.set_offsets(XA)
#         scatB.set_offsets(XB)
#         info_text.set_text(f"frame={frame}/{len(traj)-1}")
#         return scatA, scatB, info_text
#
#     anim = FuncAnimation(fig, update, frames=len(traj), interval=18, blit=True)
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
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist


# --- Utility Functions (Keep your existing load functions) ---
def load_spline(root): return np.load(root / "data/centerline_spline.npy").astype(np.float32), None


def load_dt(root): return np.load(root / "data/dt_halfwidth.npy").astype(np.float32), None


def load_params(root): return json.loads((root / "data/corridor_params.json").read_text()), None


def load_init(root): return np.load(root / "data/task2_init.npy", allow_pickle=True).item(), None


def load_mask(root): return (cv2.imread(str(root / "data/route_mask.png"), 0) > 0), None


def build_lanes(spline, offset):
    M = len(spline)
    normals = np.zeros_like(spline)
    for i in range(M):
        i0, i1 = max(0, i - 1), min(M - 1, i + 1)
        t = spline[i1] - spline[i0]
        t /= (np.linalg.norm(t) + 1e-9)
        normals[i] = np.array([-t[1], t[0]])
    return (spline + offset * normals), (spline - offset * normals)


# ------------------------------------------------------------
# Core Physics Engine
# ------------------------------------------------------------
def get_acceleration(Xi, Vi, targets, P_rails, kp, kd, k_lane, d_safe, k_rep, dt_map, r, k_wall):
    N = Xi.shape[0]
    # Path tracking + Rail pull
    A = kp * (targets - Xi) - kd * Vi + k_lane * (P_rails - Xi)
    # Collision avoidance
    dists = cdist(Xi, Xi)
    mask = (dists < d_safe) & (dists > 0.01)
    for i in range(N):
        if np.any(mask[i]):
            diff = Xi[i] - Xi[mask[i]]
            d = dists[i, mask[i]][:, None]
            A[i] += np.sum(k_rep * ((d_safe - d) / d_safe) ** 2 * (diff / d), axis=0)
    # Wall avoidance
    h, w = dt_map.shape
    for i in range(N):
        x, y = int(np.clip(Xi[i, 0], 1, w - 2)), int(np.clip(Xi[i, 1], 1, h - 2))
        d_val = dt_map[y, x]
        if d_val < (r + 4.0):
            gx = (dt_map[y, x + 1] - dt_map[y, x - 1])
            gy = (dt_map[y + 1, x] - dt_map[y - 1, x])
            grad = np.array([gx, gy]) / (np.hypot(gx, gy) + 1e-9)
            A[i] += k_wall * (r + 4.0 - d_val) * grad
    return A


def main():
    root = Path(__file__).resolve().parents[1]
    spline, _ = load_spline(root)
    dt_map, _ = load_dt(root)
    init, _ = load_init(root)
    route_bin, _ = load_mask(root)

    X, V = init["X0"].astype(float), init["V0"].astype(float)
    N_A, N_B, r, d_safe = init["N_A"], init["N_B"], init["r"], init["d_safe"]
    N, M = len(X), len(spline)

    # --- SETTINGS ---
    lane_offset = 10.0
    idx_spacing = 25  # Space between robots in the "train"
    kp, kd = 50.0, 15.0
    vmax = 25.0
    k_lane, k_rep, k_wall = 100.0, 1800.0, 200.0
    dt, frames = 0.05, 1100

    laneA, laneB = build_lanes(spline, lane_offset)

    # Track the current progress of the leaders
    hints = np.array([np.argmin(np.sum(((laneA if i < N_A else laneB) - X[i]) ** 2, axis=1)) for i in range(N)])
    master_progress = float(hints[0])  # Progress in indices

    traj = np.zeros((frames, N, 2))

    print(f"Running Swarm (5 vs 5)...")

    for k in range(frames):
        traj[k] = X

        # Advance the "Virtual Leader" to pull the whole convoy
        master_progress += vmax * dt * 0.8

        targets = np.zeros_like(X)
        P_rails = np.zeros_like(X)

        for i in range(N):
            lane = laneA if i < N_A else laneB
            # Group A moves Forward (0 -> M), Group B moves Backward (M -> 0)
            if i < N_A:
                t_idx = int(master_progress - (i * idx_spacing))
            else:
                prog_from_start = master_progress - hints[0]
                t_idx = int((M - 1 - hints[0]) - prog_from_start + (i - N_A) * idx_spacing)

            targets[i] = lane[np.clip(t_idx, 0, M - 1)]

            # Rail Projection
            h, win = hints[i], 50
            seg = lane[max(0, h - win):min(M, h + win)]
            hints[i] = max(0, h - win) + np.argmin(np.sum((seg - X[i]) ** 2, axis=1))
            P_rails[i] = lane[hints[i]]

        # RK4 / Physics Step
        A = get_acceleration(X, V, targets, P_rails, kp, kd, k_lane, d_safe, k_rep, dt_map, r, k_wall)
        V += A * dt
        # Speed saturation (PDF Page 5)
        sp = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        V *= np.minimum(1.0, vmax / sp)
        X += V * dt

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
    ax.imshow(route_bin, cmap="gray", alpha=0.1)

    # Slim tracks
    ax.plot(laneA[:, 0], laneA[:, 1], color='white', alpha=0.15, linewidth=1)
    ax.plot(laneB[:, 0], laneB[:, 1], color='white', alpha=0.15, linewidth=1)

    colors = ['#00FFFF'] * N_A + ['#FF851B'] * N_B  # Cyan and Orange
    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=40, c=colors, edgecolors='black', linewidth=0.5)

    ax.set_title("Task 2: Swarm Convoy Passage (5 vs 5)", color='white')
    ax.axis("off")

    def update(frame):
        scat.set_offsets(traj[frame])
        return (scat,)


    ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()