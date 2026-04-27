#!/usr/bin/env python3
"""
KestrelEye — Dataset Builder
Merges multiple grape disease datasets into a single YOLO-format dataset.

Usage:
    python scripts/build_dataset.py --config scripts/dataset_sources.yaml
    python scripts/build_dataset.py --config scripts/dataset_sources.yaml --dry-run
    python scripts/build_dataset.py --config scripts/dataset_sources.yaml --fix-only

Steps performed automatically:
    1. Load all enabled sources from config
    2. Fix mixed detect+segment labels (strips polygon points, keeps bbox only)
    3. Remap source class IDs → unified taxonomy
    4. Convert PlantVillage-style folder datasets (no labels → full-image bbox)
    5. Deduplicate by image content hash (MD5)
    6. Cap classes to max_per_class to avoid severe imbalance
    7. Stratified train/valid/test split (or reuse pre-split sources)
    8. Write output dataset with prefixed filenames to avoid collisions
    9. Generate data.yaml and print statistics
"""

import argparse
import hashlib
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

# ─── Unified taxonomy ─────────────────────────────────────────────────────────

UNIFIED_CLASSES = {
    0: "healthy_leaf",
    1: "flavescence_doree_leaf",
    2: "esca_leaf",
    3: "downy_mildew_leaf",
    4: "powdery_mildew_leaf",
    5: "black_rot_leaf",
    6: "healthy_bunch",
    7: "damaged_bunch",
    8: "shoot",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

log = logging.getLogger("build_dataset")


# ─── Label utilities ──────────────────────────────────────────────────────────

def is_segment_line(tokens: list[str]) -> bool:
    """A YOLO polygon line has more than 5 tokens."""
    return len(tokens) > 5


def segment_to_bbox(tokens: list[str]) -> list[str]:
    """Convert a polygon annotation to its axis-aligned bounding box."""
    class_id = tokens[0]
    coords = list(map(float, tokens[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    # clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return [class_id, f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]


def read_label(label_path: Path) -> tuple[list[list[str]], int]:
    """
    Read a YOLO label file.
    Returns (annotations, n_segments_converted).
    Polygon annotations are converted to their axis-aligned bounding box.
    """
    if not label_path.exists():
        return [], 0
    lines = label_path.read_text(encoding="utf-8").splitlines()
    result = []
    n_segments = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if is_segment_line(tokens):
            tokens = segment_to_bbox(tokens)
            n_segments += 1
        result.append(tokens)
    return result, n_segments


def remap_labels(
    annotations: list[list[str]],
    class_map: dict[int, int | None],
) -> list[list[str]]:
    """
    Apply source→unified class remapping.
    Annotations whose source class is not in class_map are discarded.
    Annotations mapped to None are discarded.
    """
    remapped = []
    for tokens in annotations:
        src_cls = int(tokens[0])
        if src_cls not in class_map:
            continue
        dst_cls = class_map[src_cls]
        if dst_cls is None:
            continue
        remapped.append([str(dst_cls)] + tokens[1:])
    return remapped


def write_label(label_path: Path, annotations: list[list[str]]) -> None:
    label_path.write_text(
        "\n".join(" ".join(t) for t in annotations),
        encoding="utf-8",
    )


# ─── Image utilities ──────────────────────────────────────────────────────────

def image_hash(image_path: Path) -> str:
    """MD5 of first 64 KB — fast enough for deduplication."""
    h = hashlib.md5()
    with image_path.open("rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def find_label_for_image(image_path: Path) -> Path:
    """Return the expected label path for an image (labels/ sibling dir)."""
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def full_image_bbox(class_id: int) -> list[str]:
    """Generate a full-image bounding box annotation (for classification datasets)."""
    return [str(class_id), "0.500000", "0.500000", "1.000000", "1.000000"]


# ─── Source loaders ───────────────────────────────────────────────────────────

def collect_from_presplit_source(
    source_path: Path,
    source_name: str,
    class_map: dict[int, int | None],
    stats: dict,
) -> dict[str, list[tuple[Path, list[list[str]]]]]:
    """
    Load from a dataset that already has train/valid/test structure.
    Returns {split: [(image_path, annotations), ...]}
    """
    splits = {}
    for split in ("train", "valid", "test"):
        img_dir = source_path / split / "images"
        if not img_dir.exists():
            # try alternative name
            img_dir = source_path / split / "imgs"
        if not img_dir.exists():
            log.warning("[%s] No %s/images dir found, skipping split", source_name, split)
            continue

        items = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = find_label_for_image(img_path)
            raw_annotations, n_segs = read_label(label_path)
            remapped = remap_labels(raw_annotations, class_map)

            stats["segment_converted"] += n_segs
            stats["labels_discarded"] += len(raw_annotations) - len(remapped)

            if not remapped:
                stats["images_skipped_no_label"] += 1
                continue

            for ann in remapped:
                stats["class_counts"][int(ann[0])] += 1

            items.append((img_path, remapped))
            stats["images_loaded"] += 1

        splits[split] = items
        log.info("[%s] %s: %d images loaded", source_name, split, len(items))
    return splits


def collect_from_flat_source(
    source_path: Path,
    source_name: str,
    class_map: dict[int, int | None],
    stats: dict,
) -> list[tuple[Path, list[list[str]]]]:
    """
    Load from a flat dataset (images/ + labels/ side by side, or just images/).
    Returns list of (image_path, annotations).
    """
    # Try standard flat layout: images/ and labels/ as siblings
    img_dirs = [
        source_path / "images",
        source_path,   # images directly in root
    ]
    img_dir = next((d for d in img_dirs if d.exists()), None)
    if img_dir is None:
        log.error("[%s] Could not find images directory", source_name)
        return []

    items = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = find_label_for_image(img_path)
        raw_annotations, n_segs = read_label(label_path)
        remapped = remap_labels(raw_annotations, class_map)

        stats["segment_converted"] += n_segs
        stats["labels_discarded"] += len(raw_annotations) - len(remapped)

        if not remapped:
            stats["images_skipped_no_label"] += 1
            continue

        for ann in remapped:
            stats["class_counts"][int(ann[0])] += 1

        items.append((img_path, remapped))
        stats["images_loaded"] += 1

    log.info("[%s] flat: %d images loaded", source_name, len(items))
    return items


def collect_from_folder_classification_source(
    source_path: Path,
    source_name: str,
    folder_class_map: dict[str, int],
    stats: dict,
) -> list[tuple[Path, list[list[str]]]]:
    """
    Load a classification dataset where images are in class-named folders.
    Each image gets a full-image bounding box annotation.
    """
    img_dir = source_path / "images"
    if not img_dir.exists():
        img_dir = source_path

    items = []
    for class_folder in sorted(img_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        if class_folder.name not in folder_class_map:
            log.debug("[%s] Folder '%s' not in folder_class_map, skipping", source_name, class_folder.name)
            continue
        unified_cls = folder_class_map[class_folder.name]
        for img_path in sorted(class_folder.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            annotations = [full_image_bbox(unified_cls)]
            stats["class_counts"][unified_cls] += 1
            items.append((img_path, annotations))
            stats["images_loaded"] += 1

    log.info("[%s] folder-classification: %d images loaded", source_name, len(items))
    return items


# ─── Dataset writing ──────────────────────────────────────────────────────────

def split_items(
    items: list,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list]:
    """Stratified split by class of first annotation."""
    rng = random.Random(seed)
    by_class: dict[int, list] = defaultdict(list)
    for item in items:
        _, annotations = item
        if annotations:
            cls = int(annotations[0][0])
        else:
            cls = -1
        by_class[cls].append(item)

    train, valid, test = [], [], []
    for cls_items in by_class.values():
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train.extend(cls_items[:n_train])
        valid.extend(cls_items[n_train : n_train + n_valid])
        test.extend(cls_items[n_train + n_valid :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return {"train": train, "valid": valid, "test": test}


def write_split(
    items: list[tuple[Path, list[list[str]]]],
    out_dir: Path,
    split: str,
    source_prefix: str,
    seen_hashes: set[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Write images and labels to output directory. Returns (written, duplicates)."""
    img_out = out_dir / split / "images"
    lbl_out = out_dir / split / "labels"
    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    written = 0
    duplicates = 0
    ann_counts: dict[int, int] = defaultdict(int)
    for img_path, annotations in items:
        h = image_hash(img_path)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)

        stem = f"{source_prefix}__{img_path.stem}"
        dst_img = img_out / (stem + img_path.suffix.lower())
        dst_lbl = lbl_out / (stem + ".txt")

        if not dry_run:
            shutil.copy2(img_path, dst_img)
            write_label(dst_lbl, annotations)

        for ann in annotations:
            ann_counts[int(ann[0])] += 1
        written += 1

    return written, duplicates, dict(ann_counts)


# ─── Statistics ───────────────────────────────────────────────────────────────

def print_stats(
    stats: dict,
    class_counts_final: dict[int, dict[str, int]],
) -> None:
    log.info("=" * 64)
    log.info("DATASET BUILD SUMMARY")
    log.info("=" * 64)
    log.info("Images loaded across all sources : %d", stats["images_loaded"])
    log.info("Images skipped (no valid label)  : %d", stats["images_skipped_no_label"])
    log.info("Segment annotations converted    : %d", stats["segment_converted"])
    log.info("Labels discarded (unmapped class): %d", stats["labels_discarded"])
    log.info("Duplicates removed               : %d", stats["duplicates"])
    log.info("-" * 64)
    log.info("Final split counts:")
    total = 0
    for split in ("train", "valid", "test"):
        n = class_counts_final.get("_total", {}).get(split, 0)
        log.info("  %-8s: %d images", split, n)
        total += n
    log.info("  TOTAL   : %d images", total)
    log.info("-" * 64)
    log.info("Annotations per class (train split):")
    train_counts = class_counts_final.get("train", {})
    for cls_id, cls_name in UNIFIED_CLASSES.items():
        count = train_counts.get(cls_id, 0)
        bar = "█" * min(40, count // 50)
        log.info("  [%d] %-28s: %5d  %s", cls_id, cls_name, count, bar)
    log.info("=" * 64)

    # Imbalance warning
    counts = [train_counts.get(i, 0) for i in UNIFIED_CLASSES if train_counts.get(i, 0) > 0]
    if counts:
        ratio = max(counts) / min(counts)
        if ratio > 5:
            log.warning(
                "Class imbalance ratio %.1fx detected. "
                "Consider lowering max_per_class or adding more data for minority classes.",
                ratio,
            )



# ─── Main ─────────────────────────────────────────────────────────────────────

def build(config_path: Path, dry_run: bool, fix_only: bool) -> None:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    project_root = config_path.parent.parent
    out_dir = project_root / cfg["output_dir"]
    seed = cfg.get("seed", 42)
    split_cfg = cfg.get("split", {"train": 0.70, "valid": 0.20, "test": 0.10})
    max_per_class = cfg.get("max_per_class", 0)

    stats: dict = {
        "images_loaded": 0,
        "images_skipped_no_label": 0,
        "segment_converted": 0,
        "labels_discarded": 0,
        "duplicates": 0,
        "class_counts": defaultdict(int),
    }

    # Collect all items per split
    all_splits: dict[str, list] = {"train": [], "valid": [], "test": []}
    seen_hashes: set[str] = set()

    for source in cfg.get("sources", []):
        if not source.get("enabled", True):
            log.info("Skipping disabled source: %s", source["name"])
            continue

        src_name = source["name"]
        src_path = project_root / source["path"]

        if not src_path.exists():
            log.warning("[%s] Path not found: %s — skipping", src_name, src_path)
            continue

        log.info("Processing source: %s (%s)", src_name, src_path)

        # PlantVillage-style folder classification dataset
        if "folder_class_map" in source:
            items = collect_from_folder_classification_source(
                src_path, src_name, source["folder_class_map"], stats
            )
            splits = split_items(
                items, split_cfg["train"], split_cfg["valid"], seed
            )
            for split, split_items in splits.items():
                all_splits[split].extend(split_items)
            continue

        # Parse numeric class_map (keys/values may be strings in YAML)
        raw_map = source.get("class_map", {})
        class_map: dict[int, int | None] = {
            int(k): (int(v) if v is not None else None)
            for k, v in raw_map.items()
        }

        if source.get("pre_split", False):
            split_data = collect_from_presplit_source(
                src_path, src_name, class_map, stats
            )
            for split, items in split_data.items():
                all_splits[split].extend(items)
        else:
            items = collect_from_flat_source(src_path, src_name, class_map, stats)
            splits = split_items(items, split_cfg["train"], split_cfg["valid"], seed)
            for split, split_items_list in splits.items():
                all_splits[split].extend(split_items_list)

    if fix_only:
        log.info("--fix-only mode: label fixes applied in-place, no output dataset written.")
        return

    if not dry_run:
        if out_dir.exists():
            log.warning("Output dir already exists, overwriting: %s", out_dir)
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    # Apply per-class cap on training split only
    if max_per_class > 0:
        log.info("Applying max_per_class cap: %d", max_per_class)
        by_class_train: dict[int, list] = defaultdict(list)
        for item in all_splits["train"]:
            _, anns = item
            cls = int(anns[0][0]) if anns else -1
            by_class_train[cls].append(item)

        rng = random.Random(seed)
        capped_train = []
        for cls_id, cls_items in by_class_train.items():
            if cls_id >= 0 and len(cls_items) > max_per_class:
                rng.shuffle(cls_items)
                capped_train.extend(cls_items[:max_per_class])
                log.info(
                    "  Class %d (%s): capped %d → %d",
                    cls_id,
                    UNIFIED_CLASSES.get(cls_id, "?"),
                    len(cls_items),
                    max_per_class,
                )
            else:
                capped_train.extend(cls_items)
        all_splits["train"] = capped_train

    # Write splits
    final_counts: dict[str, dict] = {"_total": {}}
    for split, items in all_splits.items():
        written, dupes, ann_counts = write_split(
            items, out_dir, split, "ke", seen_hashes, dry_run
        )
        stats["duplicates"] += dupes
        final_counts["_total"][split] = written
        final_counts[split] = ann_counts
        log.info("%s: %d images written, %d duplicates skipped", split, written, dupes)

    # Write data.yaml
    if not dry_run:
        data_yaml = {
            "path": str(out_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(UNIFIED_CLASSES),
            "names": list(UNIFIED_CLASSES.values()),
        }
        yaml_path = out_dir / "data.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        log.info("data.yaml written to: %s", yaml_path)

    print_stats(stats, final_counts)

    if not dry_run:
        log.info("")
        log.info("Dataset ready. To train:")
        log.info(
            "  python train.py --data %s/data.yaml --device mps --size s --batch 8 --epochs 150",
            out_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="KestrelEye Dataset Builder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/dataset_sources.yaml"),
        help="Path to dataset sources YAML config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate build without writing files",
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Only fix segment labels in-place, do not build merged dataset",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  --  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        raise SystemExit(1)

    build(args.config, dry_run=args.dry_run, fix_only=args.fix_only)


if __name__ == "__main__":
    main()
