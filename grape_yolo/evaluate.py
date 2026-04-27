"""
Validate a trained YOLOv8 model on the validation split.

Usage:
    python evaluate.py --weights grape_project/exp/weights/best.pt
"""

import argparse

from ultralytics import YOLO


def evaluate(weights_path: str):
    model = YOLO(weights_path)
    metrics = model.val()
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="grape_project/exp/weights/best.pt",
        help="Path to trained .pt weights",
    )
    args = parser.parse_args()
    evaluate(args.weights)