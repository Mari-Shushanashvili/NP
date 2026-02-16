import json
from pathlib import Path

import numpy as np
import cv2


def load_image_rgb_float01(path: Path) -> np.ndarray:
    """
    Load image with cv2 and convert to RGB float [0..1].
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    mask: uint8 {0,255} or {0,1}
    """
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8)
    else:
        mask_u8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return (mask_u8 * 255).astype(np.uint8)

    # label 0 is background
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest).astype(np.uint8) * 255
    return out


def make_blue_mask(rgb_float01: np.ndarray,
                   hsv_lower=(95, 80, 80),
                   hsv_upper=(140, 255, 255)) -> np.ndarray:
    """
    Threshold in HSV to capture Google Maps-like blue route.
    hsv ranges: H [0..179], S [0..255], V [0..255] in OpenCV.
    """
    rgb_u8 = np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)  # uint8 {0,255}
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    close gaps then remove noise.
    """
    mask = mask.copy()

    # Close small gaps along the route
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Remove small speckles
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    return mask


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "data" / "config.json"
    out_mask_path = project_root / "data" / "route_mask.png"
    out_model_path = project_root / "data" / "route_model.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    img_path = project_root / cfg["image"]

    rgb = load_image_rgb_float01(img_path)

    # --- HSV thresholds (tuned for Google Maps blue) ---
    hsv_lower = (95, 70, 60)
    hsv_upper = (145, 255, 255)

    # 1) raw threshold
    mask = make_blue_mask(rgb, hsv_lower=hsv_lower, hsv_upper=hsv_upper)

    # 2) cleanup
    mask = clean_mask(mask)

    # 3) keep largest component (your route)
    mask = largest_connected_component(mask)

    # Save mask image
    cv2.imwrite(str(out_mask_path), mask)
    print(f"Saved route mask -> {out_mask_path}")

    # Save model metadata (for reproducibility)
    route_pixels = int((mask > 0).sum())
    model = {
        "strategy": "blue_hsv_threshold",
        "hsv_lower": list(hsv_lower),
        "hsv_upper": list(hsv_upper),
        "route_pixels": route_pixels,
        "notes": "Mask extracted via HSV thresholding for Google-Maps style blue route. "
                 "Then morphology cleanup + largest connected component."
    }
    out_model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved route model -> {out_model_path}")


if __name__ == "__main__":
    main()
