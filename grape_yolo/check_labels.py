"""
check_labels.py
Diagnose YOLO-seg label quality: instance counts, polygon sizes, fragmentation.

Usage:
    python check_labels.py --root ../data/vine_seg_yolo
    python check_labels.py --root ../data/vine_seg_yolo --split train
"""

import argparse
from pathlib import Path


def poly_area_norm(coords):
    """Shoelace area of a normalized polygon (fraction of image area)."""
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    return 0.5 * abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                         for i in range(n)))


def analyze(label_dir: Path, split_name: str):
    files = sorted(label_dir.glob("*.txt"))
    if not files:
        print(f"  [{split_name}] no label files found in {label_dir}")
        return

    counts, pts, areas = [], [], []
    empty_files = 0

    for f in files:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        counts.append(len(lines))
        if not lines:
            empty_files += 1
        for l in lines:
            parts = l.split()
            coords = [float(v) for v in parts[1:]]
            pts.append(len(coords) // 2)
            areas.append(poly_area_norm(coords))

    n_poly = len(pts)
    if n_poly == 0:
        print(f"  [{split_name}] {len(files)} files but 0 polygons!")
        return

    frag_pts = sum(1 for p in pts if p < 10)
    frag_area = sum(1 for a in areas if a < 0.0005)

    print(f"\n  --- {split_name} ---")
    print(f"  label files            : {len(files)}  (empty: {empty_files})")
    print(f"  total polygons         : {n_poly}")
    print(f"  instances per image    : mean {sum(counts)/len(counts):.1f}   "
          f"min {min(counts)}   max {max(counts)}")
    print(f"  points per polygon     : mean {sum(pts)/n_poly:.1f}   "
          f"min {min(pts)}   max {max(pts)}")
    print(f"  polygon area (of image): mean {sum(areas)/n_poly*100:.2f}%   "
          f"min {min(areas)*100:.3f}%   max {max(areas)*100:.2f}%")
    print(f"  fragments (<10 points) : {frag_pts} / {n_poly}  "
          f"({frag_pts/n_poly*100:.0f}%)")
    print(f"  slivers (<0.05% area)  : {frag_area} / {n_poly}  "
          f"({frag_area/n_poly*100:.0f}%)")

    # verdict
    mean_inst = sum(counts) / len(counts)
    frag_pct = frag_pts / n_poly * 100
    print("  verdict                : ", end="")
    if mean_inst > 10 or frag_pct > 25:
        print("FRAGMENTED - multipolygon bug likely still present")
    elif mean_inst > 8:
        print("borderline - more instances/image than expected, inspect visually")
    else:
        print("looks healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="YOLO-seg dataset root")
    ap.add_argument("--split", help="only check one split (train/valid/test)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"Dataset: {root}")

    splits = [args.split] if args.split else ["train", "valid", "test"]
    for s in splits:
        d = root / s / "labels"
        if d.is_dir():
            analyze(d, s)
        else:
            print(f"\n  [{s}] no labels dir at {d}")

    print("\nExpected for your vineyard frames: ~2-6 instances/image, "
          "few fragments, polygons with 20+ points.")


if __name__ == "__main__":
    main()