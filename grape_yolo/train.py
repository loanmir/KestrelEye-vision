"""
Train YOLOv8 on the grape leaf disease dataset.

Usage:
    python train.py
"""

from ultralytics import YOLO

from config import (
    YAML_PATH,
    MODEL_WEIGHTS,
    EPOCHS,
    IMG_SIZE,
    BATCH_SIZE,
    PROJECT,
    RUN_NAME,
    DEVICE,
    WORKERS,
)


def train():
    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=RUN_NAME,
        device=DEVICE,
        workers=WORKERS,
    )
    return results


if __name__ == "__main__":
    train()