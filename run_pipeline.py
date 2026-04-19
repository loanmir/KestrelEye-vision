#!/usr/bin/env python3
"""
run_pipeline.py
----------------
Offline inference pipeline for KestrelEye (Opzione A).

Takes a folder of images (from microSD or the grape dataset), runs
YOLOv8, and writes three files per run:

    data/output/<run_name>/
    |-- detections.csv       one row per detection (or blank row per clean image)
    |-- detections.json      same data as JSON array (positive detections only)
    |-- summary.txt          human-readable counts and confidence stats
    |-- annotated/           JPEGs with bounding boxes drawn
    +-- logs/<run_name>.log  full run log

Quick start
-----------
    source .venv/bin/activate

    # Test with pretrained COCO weights on the validation split
    python run_pipeline.py \
        --input data/grape_disease/valid/images/

    # After training: use your fine-tuned weights
    python run_pipeline.py \
        --input   data/grape_disease/valid/images/ \
        --weights runs/train/kestreleye_grape/weights/best.pt \
        --conf    0.50 \
        --name    val_v1

    # On real drone images (no GPS filtering needed in this pipeline)
    python run_pipeline.py \
        --input   data/raw/flight_20260615/ \
        --weights runs/train/kestreleye_grape/weights/best.pt \
        --name    flight_20260615

    # CPU-only (no Apple Silicon / NVIDIA GPU)
    python run_pipeline.py --input data/raw/ --device cpu
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.loader import get_model, get_model_info
from src.pipeline.inference import collect_images, run_inference
from src.pipeline.export import (
    build_dataframe,
    export_csv,
    export_json,
    export_summary_txt,
    print_summary,
)
from src.utils.image import filter_valid, annotate_batch
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KestrelEye -- offline vineyard disease detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True, type=Path,
                   help="Folder of images to analyse")
    p.add_argument("--output-dir", default="data/output", type=Path,
                   help="Root folder for all outputs")
    p.add_argument("--name",       default=None,
                   help="Run name (subfolder under output-dir). "
                        "Defaults to timestamp.")
    p.add_argument("--weights",    default=None, type=Path,
                   help="Path to .pt weights. Omit to use pretrained COCO.")
    p.add_argument("--size",       default="n", choices=["n","s","m","l","x"],
                   help="Backbone size (only used when --weights is not given)")
    p.add_argument("--conf",       default=0.45, type=float,
                   help="Confidence threshold")
    p.add_argument("--iou",        default=0.50, type=float,
                   help="NMS IoU threshold")
    p.add_argument("--imgsz",      default=640,  type=int,
                   help="Inference image size")
    p.add_argument("--device",     default="mps",
                   help="'mps' | 'cpu' | '0' (CUDA)")
    p.add_argument("--no-annotate", action="store_true",
                   help="Skip saving annotated images (faster, less disk use)")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level    = args.log_level,
        log_dir  = args.output_dir / "logs",
        run_name = run_name,
    )

    import logging
    log = logging.getLogger(__name__)

    log.info("=" * 56)
    log.info("  KestrelEye -- Inference Pipeline")
    log.info("  Run   : %s", run_name)
    log.info("  Input : %s", args.input)
    log.info("=" * 56)

    # 1. Collect + validate images
    log.info("[1/4] Collecting images ...")
    all_images   = collect_images(args.input)
    valid_images = filter_valid(all_images)
    log.info("  Valid: %d / %d", len(valid_images), len(all_images))

    if not valid_images:
        log.error("No valid images found in %s -- aborting.", args.input)
        sys.exit(1)

    # 2. Load model
    log.info("[2/4] Loading model ...")
    model = get_model(weights=args.weights, size=args.size)
    info  = get_model_info(model)
    log.info("  Classes (%d): %s", info["num_classes"], info["class_names"])

    # 3. Inference
    log.info("[3/4] Running inference on %d images ...", len(valid_images))
    results = run_inference(
        model   = model,
        images  = valid_images,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device,
    )

    # 4. Export results
    log.info("[4/4] Exporting results ...")

    df = build_dataframe(results)

    export_csv(df,         run_dir / "detections.csv")
    export_json(df,        run_dir / "detections.json")
    export_summary_txt(df, run_dir / "summary.txt")

    if not args.no_annotate:
        annotate_batch(results, run_dir / "annotated")

    print_summary(df)
    log.info("All outputs saved in: %s", run_dir)
    log.info("Open results folder : open %s", run_dir)


if __name__ == "__main__":
    main()