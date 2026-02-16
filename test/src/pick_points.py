import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    project_root = Path(__file__).resolve().parents[1]
    img_path = project_root / "data" / "map.png"
    out_path = project_root / "data" / "config.json"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = mpimg.imread(str(img_path))

    print("INSTRUCTIONS:")
    print("1) Click once on START point A.")
    print("2) Click once on END point B.")
    print("Close the window after the two clicks are registered.\n")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title("Click A (start), then B (end)")
    ax.axis("off")

    clicks = []

    def on_click(event):
        # Ignore clicks outside the image axes
        if event.inaxes != ax:
            return

        x, y = event.xdata, event.ydata
        clicks.append((x, y))

        label = "A" if len(clicks) == 1 else "B" if len(clicks) == 2 else f"P{len(clicks)}"
        ax.plot([x], [y], marker="o")
        ax.text(x + 5, y + 5, label, fontsize=12)
        fig.canvas.draw()

        print(f"Registered {label} at (x={x:.1f}, y={y:.1f})")

        # After two clicks, auto-save and disconnect
        if len(clicks) == 2:
            A = [int(round(clicks[0][0])), int(round(clicks[0][1]))]
            B = [int(round(clicks[1][0])), int(round(clicks[1][1]))]

            cfg = {
                "image": "data/map.png",
                "A": A,
                "B": B,
                "sample_window": 120,
                "ring_inner": 10,
                "ring_outer": 55,
            }

            out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"\nSaved config to: {out_path}")
            print("A =", A, "B =", B)
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
