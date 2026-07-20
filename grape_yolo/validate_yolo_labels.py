"""
validate_yolo_labels.py  (v2 - robust path discovery)
-----------------------------------------------------
Diagnose (and optionally repair) YOLO detection label problems that cause
Ultralytics training crashes like:

  RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) ...
  RuntimeError: shape mismatch: value tensor of shape [X] cannot be broadcast ...

v2 change: instead of trusting the (often broken) relative paths inside
data.yaml, it scans the dataset ROOT folder for every 'labels' directory.
Point --data at data.yaml OR --root at the dataset folder directly.

Usage:
  python validate_yolo_labels.py --data ../data/UAV_dataset/data.yaml
  python validate_yolo_labels.py --root ../data/UAV_dataset
  python validate_yolo_labels.py --root ../data/UAV_dataset --fix --backup

Checks per label file:
  1. Wrong column count (segmentation/polygon lines, truncated lines)
  2. Class ids outside [0, nc-1]  (nc taken from data.yaml if given)
  3. Coordinates outside [0, 1], NaN/inf, zero/negative width or height
  4. Duplicate lines
  5. Non-numeric garbage
Also deletes stale Ultralytics *.cache files (always), since a stale cache
alone reproduces these crashes.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def get_nc(data_yaml):
    if data_yaml is None:
        return None
    if yaml is None:
        print("PyYAML not installed; skipping class-id checks. pip install pyyaml")
        return None
    cfg = yaml.safe_load(data_yaml.read_text())
    if "nc" in cfg:
        return int(cfg["nc"])
    names = cfg.get("names")
    if isinstance(names, (list, dict)):
        return len(names)
    return None


def looks_like_yolo(txt):
    try:
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                return False
            int(float(parts[0]))
            [float(p) for p in parts[1:]]
            return True
    except Exception:
        return False
    return False


def find_label_dirs(root):
    dirs = []
    for d in root.rglob("labels"):
        if d.is_dir() and any(d.glob("*.txt")):
            dirs.append(d)
    if not dirs:
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            txts = list(d.glob("*.txt"))
            if len(txts) >= 2 and looks_like_yolo(txts[0]):
                dirs.append(d)
    return sorted(set(dirs))


def print_tree(root, max_depth=3):
    print(f"\nFolder tree under {root} (depth {max_depth}):")
    root = root.resolve()
    for p in sorted(root.rglob("*")):
        try:
            depth = len(p.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        if p.is_dir():
            n_txt = len(list(p.glob("*.txt")))
            tag = f"   [{n_txt} .txt]" if n_txt else ""
            print(f"{'  ' * (depth - 1)}{p.name}/{tag}")


def polygon_to_bbox(vals):
    xs, ys = vals[0::2], vals[1::2]
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return [(x0 + x1) / 2, (y0 + y1) / 2, w, h]


def clip01(v):
    return min(1.0, max(0.0, v))


def process_file(path, nc, fix, stats):
    kept, seen = [], set()
    file_bad = False

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            cls = int(float(tokens[0]))
            vals = [float(t) for t in tokens[1:]]
        except (ValueError, IndexError):
            stats["non_numeric"] += 1; file_bad = True; continue

        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
            stats["nan_inf"] += 1; file_bad = True; continue

        if nc is not None and not (0 <= cls < nc):
            stats["bad_class"] += 1; file_bad = True; continue
        if nc is None:
            stats["max_class_seen"] = max(stats["max_class_seen"], cls)

        if len(vals) == 4:
            box = vals
        elif len(vals) > 4 and len(vals) % 2 == 0:
            stats["polygon"] += 1; file_bad = True
            box = polygon_to_bbox(vals)
            if box is None:
                stats["dropped_polygon"] += 1; continue
        else:
            stats["wrong_cols"] += 1; file_bad = True; continue

        xc, yc, w, h = box
        if not all(0 <= v <= 1 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            stats["bad_coords"] += 1; file_bad = True
            xc, yc = clip01(xc), clip01(yc)
            w = min(w, 2 * xc, 2 * (1 - xc))
            h = min(h, 2 * yc, 2 * (1 - yc))
            if w <= 1e-6 or h <= 1e-6:
                stats["dropped_degenerate"] += 1; continue

        out = f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        if out in seen:
            stats["duplicates"] += 1; file_bad = True; continue
        seen.add(out); kept.append(out)

    if file_bad:
        stats["files_with_issues"] += 1
        if fix:
            path.write_text("\n".join(kept) + ("\n" if kept else ""))
            stats["files_fixed"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to data.yaml (used for nc + root)")
    ap.add_argument("--root", help="dataset root folder to scan (overrides --data root)")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    if not args.data and not args.root:
        print("Provide --data path/to/data.yaml or --root path/to/dataset")
        sys.exit(1)

    data_yaml = Path(args.data).resolve() if args.data else None
    nc = get_nc(data_yaml)
    root = Path(args.root).resolve() if args.root else data_yaml.parent

    print(f"Dataset root: {root}")
    print(f"nc = {nc if nc is not None else 'unknown (class checks skipped)'}\n")

    for cache in root.rglob("*.cache"):
        print(f"Deleting stale cache: {cache}")
        cache.unlink()

    label_dirs = find_label_dirs(root)
    if not label_dirs:
        print("No label directories found under the root.")
        print_tree(root)
        print("\n-> Send me this tree and I'll point the script at the right place.")
        sys.exit(1)

    print("Label dirs found:")
    for d in label_dirs:
        print(f"  {d}  ({len(list(d.glob('*.txt')))} files)")
    print()

    stats = {k: 0 for k in (
        "files_total", "files_with_issues", "files_fixed", "polygon",
        "dropped_polygon", "wrong_cols", "bad_class", "bad_coords",
        "dropped_degenerate", "duplicates", "non_numeric", "nan_inf")}
    stats["max_class_seen"] = -1

    for d in label_dirs:
        if args.fix and args.backup:
            bkp = d.parent / (d.name + "_backup")
            if not bkp.exists():
                shutil.copytree(d, bkp)
                print(f"Backup created: {bkp}")
        for f in sorted(d.glob("*.txt")):
            stats["files_total"] += 1
            process_file(f, nc, args.fix, stats)

    print("\n===== SUMMARY =====")
    for k, v in stats.items():
        print(f"{k:22s}: {v}")
    if nc is None and stats["max_class_seen"] >= 0:
        print(f"\nHighest class id seen: {stats['max_class_seen']} "
              f"-> your data.yaml nc should be {stats['max_class_seen'] + 1}")

    if stats["files_with_issues"] and not args.fix:
        print("\nIssues found. Re-run with --fix --backup to repair, then retrain.")
    elif args.fix:
        print("\nLabels repaired and caches cleared. Retrain now.")
    else:
        print("\nLabels look clean. Deleted caches were likely the cause; retrain.")


if __name__ == "__main__":
    main()