"""
Convert any YOLO segmentation-format lines (polygon, >5 values) to
bounding-box format (5 values) across the entire dataset.

Run once before training:
    python grape_yolo/fix_labels.py
"""

import glob
import os

DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "UAV_unhealthy_dataset",
)


def polygon_to_bbox(parts: list[str]) -> str:
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_center = (min(xs) + max(xs)) / 2
    y_center = (min(ys) + max(ys)) / 2
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def fix_file(path: str) -> int:
    with open(path) as f:
        lines = f.readlines()

    fixed = []
    converted = 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) > 5:
            fixed.append(polygon_to_bbox(parts))
            converted += 1
        else:
            fixed.append(line if line.endswith("\n") else line + "\n")

    if converted:
        with open(path, "w") as f:
            f.writelines(fixed)

    return converted


def main():
    pattern = os.path.join(DATASET_ROOT, "**", "labels", "*.txt")
    total_converted = 0
    total_files = 0

    for path in glob.glob(pattern, recursive=True):
        n = fix_file(path)
        if n:
            print(f"  {path}: {n} segment line(s) converted")
            total_files += 1
            total_converted += n

    if total_converted:
        print(f"\nDone. Converted {total_converted} line(s) in {total_files} file(s).")
        print("Delete the .cache files so Ultralytics rescans the labels:")
        for cache in glob.glob(os.path.join(DATASET_ROOT, "**", "labels.cache"), recursive=True):
            os.remove(cache)
            print(f"  Removed {cache}")
    else:
        print("No segmentation lines found — dataset is clean.")


if __name__ == "__main__":
    main()
