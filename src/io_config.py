import json
from pathlib import Path
import matplotlib.image as mpimg


def load_config(config_path: str | Path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    # Basic validation
    for key in ["image", "A", "B"]:
        if key not in cfg:
            raise ValueError(f"Missing key '{key}' in config")

    if not (isinstance(cfg["A"], list) and len(cfg["A"]) == 2):
        raise ValueError("cfg['A'] must be [x, y]")
    if not (isinstance(cfg["B"], list) and len(cfg["B"]) == 2):
        raise ValueError("cfg['B'] must be [x, y]")

    return cfg


def load_image_from_config(cfg, project_root: str | Path | None = None):
    """
    Loads the image specified in cfg["image"].
    """
    img_rel = Path(cfg["image"])
    img_path = img_rel if project_root is None else Path(project_root) / img_rel

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = mpimg.imread(str(img_path))
    return img, img_path


if __name__ == "__main__":
    # Quick sanity check
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "data" / "config.json")
    img, img_path = load_image_from_config(cfg, project_root=project_root)
    print("Loaded:", img_path, "shape:", img.shape, "A:", cfg["A"], "B:", cfg["B"])
