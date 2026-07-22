"""
coco_to_yolo_seg.py
Convert a Roboflow COCO segmentation export into YOLO11-seg format.
Polygons are preserved as segmentation labels (NOT collapsed to boxes).

Usage:
    python coco_to_yolo_seg.py --src ../data/segmentation_dataset_UAV --out ../data/vine_seg_yolo
"""

import argparse
import json
import shutil
from pathlib import Path


def convert_split(split_dir, out_root, split_name, class_remap):
    coco_json = split_dir / "_annotations.coco.json"
    if not coco_json.exists():
        print(f"  [skip] no _annotations.coco.json in {split_dir}")
        return set()

    data = json.loads(coco_json.read_text())
    images = {img["id"]: img for img in data["images"]}

    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    img_out = out_root / split_name / "images"
    lbl_out = out_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    classes_used = set()
    n_imgs = 0

    for img_id, img in images.items():
        fname = img["file_name"]
        w, h = img["width"], img["height"]
        src_img = split_dir / fname
        if not src_img.exists():
            print(f"  [warn] image file missing: {src_img}")
            continue

        shutil.copy2(src_img, img_out / fname)
        n_imgs += 1

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cls = class_remap[ann["category_id"]]
            classes_used.add(cls)

            segs = ann.get("segmentation", [])
            if not segs:
                continue

            # COCO multipolygon: one annotation can hold several parts.
            # Keep only the LARGEST part -> one instance per annotation.
            valid = [p for p in segs if len(p) >= 6]
            if not valid:
                continue

            def poly_area(p):
                xs, ys = p[0::2], p[1::2]
                return 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] -
                                     xs[(i + 1) % len(xs)] * ys[i]
                                     for i in range(len(xs))))

            poly = max(valid, key=poly_area)

            # drop degenerate slivers (tiny area relative to image)
            if poly_area(poly) < 0.0005 * w * h:
                continue

            norm = []
            for i in range(0, len(poly), 2):
                x = min(max(poly[i] / w, 0.0), 1.0)
                y = min(max(poly[i + 1] / h, 0.0), 1.0)
                norm.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(f"{cls} " + " ".join(norm))

        label_path = lbl_out / (Path(fname).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"  {split_name}: {n_imgs} images, classes used: {sorted(classes_used)}")
    return classes_used


def build_class_remap(src):
    """
    Roboflow COCO exports add a dummy id-0 supercategory named after the project.
    We keep only real object categories. If a single real class remains, everything
    maps to class 0.
    """
    for split in ("train", "valid", "test"):
        cj = src / split / "_annotations.coco.json"
        if cj.exists():
            data = json.loads(cj.read_text())
            cats = sorted(data["categories"], key=lambda c: c["id"])

            # Drop the Roboflow placeholder: the id-0 supercategory whose name
            # matches the project (it has no real annotations pointing only to it).
            # Heuristic: keep categories that are NOT the id-0 container.
            real = [c for c in cats if c["id"] != 0]

            # Fallback: if that removed everything, keep all.
            if not real:
                real = cats

            names = [c["name"] for c in real]
            remap = {c["id"]: idx for idx, c in enumerate(real)}
            return remap, names
    raise FileNotFoundError("No _annotations.coco.json found in train/valid/test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Roboflow COCO export root")
    ap.add_argument("--out", required=True, help="output YOLO-seg dataset root")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    remap, names = build_class_remap(src)
    print(f"Classes: {names}  (remap {remap})\n")

    for split in ("train", "valid", "test"):
        sd = src / split
        if sd.is_dir():
            convert_split(sd, out, split, remap)

    yaml_text = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    (out / "data.yaml").write_text(yaml_text)
    print(f"\nWrote {out / 'data.yaml'}")
    print("Done. Train with a *-seg model, e.g. yolo11s-seg.pt")


if __name__ == "__main__":
    main()