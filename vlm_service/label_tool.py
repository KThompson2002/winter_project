"""
Simple OpenCV labeling tool for recorded frames.

Usage:
    python label_tool.py --data_dir /tmp/pipeline_eval

Controls:
    - Click and drag to draw a bounding box
    - Press 'n' to skip (no label)
    - Press 'r' to redo the box
    - Press ENTER to confirm and move to the next frame
    - Press 'q' to quit (progress is saved)

You will be prompted in the terminal for the target_label text.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

drawing = False
ix, iy = 0, 0
box = None


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, box
    img = param["img"]
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        display = img.copy()
        cv2.rectangle(display, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("label", display)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        display = img.copy()
        cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("label", display)


def main():
    global box
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels_path = data_dir / "labels.json"

    with open(labels_path) as f:
        labels = json.load(f)

    for name in sorted(labels.keys()):
        entry = labels[name]
        if entry.get("target_label", ""):
            print(f"Skipping {name} (already labeled)")
            continue

        npz_path = data_dir / f"{name}.npz"
        if not npz_path.exists():
            continue

        data = np.load(str(npz_path))
        rgb = data["rgb"]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        box = None
        param = {"img": bgr.copy()}
        cv2.imshow("label", bgr)
        cv2.setMouseCallback("label", mouse_cb, param)

        print(f"\n--- {name} ---")
        print("Draw a box, then press ENTER. 'n' to skip, 'r' to redo, 'q' to quit.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                with open(labels_path, "w") as f:
                    json.dump(labels, f, indent=2)
                cv2.destroyAllWindows()
                print("Saved and quit.")
                return
            elif key == ord("n"):
                print(f"Skipped {name}")
                break
            elif key == ord("r"):
                box = None
                cv2.imshow("label", bgr)
                print("Reset. Draw again.")
            elif key in (13, 10):  # ENTER
                target = input("  target_label: ").strip()
                if not target:
                    print("  No label entered, skipping.")
                    break
                entry["target_label"] = target
                entry["bbox_xyxy"] = list(box) if box else []
                with open(labels_path, "w") as f:
                    json.dump(labels, f, indent=2)
                print(f"  Saved: label={target} box={entry['bbox_xyxy']}")
                break

    cv2.destroyAllWindows()
    print("Done labeling.")


if __name__ == "__main__":
    main()
