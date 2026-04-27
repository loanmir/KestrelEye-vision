"""
Export a trained YOLOv8 model to ONNX (or other) format.

Usage:
    python export.py --weights grape_project/exp/weights/best.pt --format onnx
"""

import argparse

from ultralytics import YOLO


def export(weights_path: str, fmt: str = "onnx"):
    model = YOLO(weights_path)
    return model.export(format=fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="grape_project/exp/weights/best.pt",
        help="Path to trained .pt weights",
    )
    parser.add_argument(
        "--format",
        default="onnx",
        help="Export format: onnx, torchscript, coreml, tflite, ...",
    )
    args = parser.parse_args()
    export(args.weights, args.format)