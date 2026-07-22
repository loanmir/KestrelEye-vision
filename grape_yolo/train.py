"""
Train YOLOv8 on the grape leaf disease dataset.

Usage:
    python train.py
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    #OPTIMIZER,
    #LR0,
    #LRF,
    #WEIGHT_DECAY,
    PATIENCE,
    #CLOSE_MOSAIC,
    #FLIPUD,
    #DEGREES,
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
        #amp=False,
        #optimizer=OPTIMIZER,
        #lr0=LR0,
        #lrf=LRF,
        #weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        #close_mosaic=CLOSE_MOSAIC,
        #flipud=FLIPUD,
        #degrees=DEGREES,
    )
    return results


if __name__ == "__main__":
    train()