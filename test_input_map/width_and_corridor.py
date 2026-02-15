import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[1]

    mask_path = project_root / "data" / "route_mask.png"
    spline_path = project_root / "data" / "centerline_spline.npy"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing {mask_path}. Run Step 1 first.")
    if not spline_path.exists():
        raise FileNotFoundError(f"Missing {spline_path}. Run Step 2 first.")

    # ------------------------------------------------------------
    # Load corridor mask (white corridor on black)
    # ------------------------------------------------------------
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    route_bin = (mask > 0).astype(np.uint8)  # 0/1

    # ============================================================
    # NEW (for Task 2 readiness): Inflate the corridor width
    # ============================================================
    # We "dilate" the corridor mask so the white tube becomes wider in pixels.
    # This is useful when you want more room for multiple robots passing.
    #
    # Interpretation for report: this dilation represents an uncertainty/safety margin
    # due to finite pixel resolution / extraction noise from the screenshot.
    #
    # Tune:
    #   inflate_px = 0  -> no widening (original corridor)
    #   inflate_px = 3  -> small widening
    #   inflate_px = 5  -> moderate widening (recommended start)
    #   inflate_px = 8  -> large widening
    inflate_px = 10  # <---- YOU CAN CHANGE THIS VALUE

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * inflate_px + 1, 2 * inflate_px + 1)
    )

    # route_bin_inflated is the widened corridor mask
    route_bin_inflated = cv2.dilate(route_bin, kernel, iterations=1)

    # Optional: save inflated mask for debugging / slides
    inflated_mask_path = project_root / "data" / "route_mask_inflated.png"
    cv2.imwrite(str(inflated_mask_path), (route_bin_inflated * 255).astype(np.uint8))
    print("Saved inflated corridor mask:", inflated_mask_path)

    # ------------------------------------------------------------
    # Distance transform (computed on the *inflated* corridor)
    # dt[y,x] = distance to nearest boundary of the inflated corridor
    # ------------------------------------------------------------
    dt = cv2.distanceTransform(route_bin_inflated * 255, cv2.DIST_L2, 5).astype(np.float32)

    # Save DT for reuse
    dt_out = project_root / "data" / "dt_halfwidth.npy"
    np.save(dt_out, dt)
    print("Saved distance transform:", dt_out)

    # ------------------------------------------------------------
    # Load spline and sample DT along it -> estimate half-width
    # ------------------------------------------------------------
    spline = np.load(spline_path).astype(np.float32)
    h, w = route_bin.shape

    xs = np.clip(np.round(spline[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(spline[:, 1]).astype(int), 0, h - 1)
    dt_samples = dt[ys, xs]

    # Robust half-width estimate from DT values on centerline
    w_half = float(np.percentile(dt_samples, 30))

    # ------------------------------------------------------------
    # Robot radius definition
    # (keep same as before; widening corridor will automatically make safe region bigger)
    # ------------------------------------------------------------
    robot_radius = float(max(3.0, min(0.25 * w_half, 12.0)))

    # Safe margin for center positions
    safe_margin = w_half - robot_radius

    # ------------------------------------------------------------
    # Save corridor parameters (include inflation info for reproducibility)
    # ------------------------------------------------------------
    params = {
        "w_half_px": w_half,
        "robot_radius_px": robot_radius,
        "safe_margin_px": safe_margin,
        "inflate_px": inflate_px,
        "width_method": "distance_transform_on_INFLATED_mask + percentile30_along_spline",
        "constraint": "dist_to_centerline(x) + robot_radius <= w_half",
        "notes": "Corridor widened by dilation to provide extra margin/passing space."
    }

    params_path = project_root / "data" / "corridor_params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print("Saved corridor params:", params_path)

    print("\n=== Corridor summary (copy into report) ===")
    print(f"inflate_px           = {inflate_px} px (mask dilation)")
    print(f"Half-width w_half     = {w_half:.2f} px")
    print(f"Robot radius r        = {robot_radius:.2f} px")
    print(f"Safe margin (w_half-r)= {safe_margin:.2f} px")
    print("Constraint: dist_to_centerline + r <= w_half\n")

    # ------------------------------------------------------------
    # Debug visualization: show safe region for robot centers
    # Safe centers: dt >= robot_radius
    # ------------------------------------------------------------
    safe_center_region = (dt >= robot_radius).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show the inflated corridor lightly
    ax.imshow(route_bin_inflated, cmap="gray", alpha=0.35)

    # Show the safe region brighter
    ax.imshow(safe_center_region, cmap="gray", alpha=0.65)

    # Overlay spline
    ax.plot(spline[:, 0], spline[:, 1], linewidth=2)

    ax.set_title("Inflated corridor (light) and safe region for robot centers (bright)")
    ax.axis("off")
    plt.tight_layout()

    out_img = project_root / "data" / "corridor_debug.png"
    fig.savefig(out_img, dpi=200)
    print("Saved debug overlay:", out_img)
    plt.show()


if __name__ == "__main__":
    main()
