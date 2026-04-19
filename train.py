#!/usr/bin/env python3
"""
train.py
--------
Train YOLOv8 on the grape disease dataset.

Quick start
-----------
    source .venv/bin/activate

    # Default: nano backbone, 50 epochs, Apple Silicon GPU
    python train.py --data data/grape_disease/data.yaml

    # Custom run
    python train.py \
        --data    data/grape_disease/data.yaml \
        --size    s \
        --epochs  100 \
        --batch   8 \
        --device  cpu \
        --name    kestreleye_v2

    # Resume an interrupted run
    python train.py \
        --data   data/grape_disease/data.yaml \
        --resume runs/train/kestreleye_grape/weights/last.pt

Expected dataset layout
-----------------------
    data/grape_disease/
    |-- data.yaml
    |-- train/
    |   |-- images/   (.jpg / .png)
    |   +-- labels/   (.txt, YOLO format)
    |-- valid/
    |   |-- images/
    |   +-- labels/
    +-- test/          (optional)
        |-- images/
        +-- labels/

Outputs  (inside runs/train/<name>/)
-------------------------------------
    weights/best.pt       best checkpoint by val mAP50
    weights/last.pt       final epoch checkpoint
    results.csv           per-epoch metrics
    confusion_matrix.png
    PR_curve.png
    val_batch*.jpg        validation sample visualisations
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.loader import get_model
from src.models.trainer import Trainer, TrainConfig
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KestrelEye -- train YOLOv8 on grape disease dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",     required=True, type=Path,
                   help="Path to data.yaml")
    p.add_argument("--size",     default="n", choices=["n","s","m","l","x"],
                   help="Backbone size ('n' = nano, fastest)")
    p.add_argument("--weights",  default=None, type=Path,
                   help="Start from a local .pt file instead of pretrained")
    p.add_argument("--resume",   default=None, type=Path,
                   help="Resume from last.pt of an interrupted run")
    p.add_argument("--epochs",   default=50,    type=int)
    p.add_argument("--imgsz",    default=640,   type=int)
    p.add_argument("--batch",    default=16,    type=int)
    p.add_argument("--lr0",      default=0.01,  type=float)
    p.add_argument("--patience", default=20,    type=int,
                   help="Early-stopping patience (0 = disabled)")
    p.add_argument("--device",   default="mps",
                   help="'mps' | 'cpu' | '0' (CUDA)")
    p.add_argument("--project",  default="runs/train")
    p.add_argument("--name",     default="kestreleye_grape")
    p.add_argument("--exist-ok", action="store_true",
                   help="Overwrite existing run with the same name")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level,
                  log_dir="data/output/logs",
                  run_name=args.name)

    import logging
    log = logging.getLogger(__name__)
    log.info("KestrelEye -- Training Pipeline")

    # Load model
    if args.resume:
        log.info("Resuming from: %s", args.resume)
        model = get_model(weights=args.resume)
    elif args.weights:
        log.info("Custom weights: %s", args.weights)
        model = get_model(weights=args.weights)
    else:
        log.info("Pretrained backbone: yolov8%s.pt", args.size)
        model = get_model(size=args.size)

    config = TrainConfig(
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        lr0       = args.lr0,
        patience  = args.patience,
        device    = args.device,
        project   = args.project,
        name      = args.name,
        exist_ok  = args.exist_ok,
    )

    trainer = Trainer(model=model, data_yaml=args.data, config=config)
    trainer.train()

    best = trainer.best_weights
    if best:
        log.info("Best weights saved to: %s", best)
        log.info(
            "Next step -- run inference:\n"
            "  python run_pipeline.py "
            "--input data/raw/<flight>/ "
            "--weights %s", best
        )


if __name__ == "__main__":
    main()