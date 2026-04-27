"""
Run inference with a trained YOLOv8 model and display the predictions.

Usage:
    python predict.py --weights grape_project/exp/weights/best.pt --source path/to/images
"""

import argparse
from glob import glob

from PIL import Image
from ultralytics import YOLO

from config import IMG_VAL, CONF_THRESHOLD, DEVICE


def predict(weights_path: str, source: str, conf: float = CONF_THRESHOLD):
    model = YOLO(weights_path)
    results = model.predict(source=source, conf=conf, save=True, device=DEVICE)
    return results


def show_predictions(predict_dir: str = "runs/detect/predict", n: int = 5):
    """Open the first N saved prediction images."""
    preds = glob(f"{predict_dir}/*.jpg")
    for img in preds[:n]:
        Image.open(img).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="grape_project/exp/weights/best.pt",
        help="Path to trained .pt weights",
    )
    parser.add_argument(
        "--source",
        default=IMG_VAL,
        help="Image, directory, or video path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help="Confidence threshold",
    )
    args = parser.parse_args()
    predict(args.weights, args.source, args.conf)