"""
src/models/loader.py
--------------------
Single point of truth for loading a YOLO model instance.

Three modes
-----------
1. Pretrained backbone  -> get_model()            no weights arg
2. Local checkpoint     -> get_model(weights=...)  path to any .pt file
3. Named KE checkpoint  -> get_finetuned_model("v1")

Every other module always receives a plain YOLO object from here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)

VALID_SIZES = {"n", "s", "m", "l", "x"}

# Fine-tuned weights live here (gitignored -- never committed)
WEIGHTS_DIR = Path(__file__).parent / "weights"


def get_model(
    weights: str | Path | None = None,
    size: str = "n",
    task: str = "detect",
) -> YOLO:
    """
    Load and return a YOLOv8 model.

    Parameters
    ----------
    weights : str | Path | None
        - None  -> pretrained COCO backbone (yolov8n.pt, downloaded once)
        - Path  -> any local .pt checkpoint
    size : str
        Backbone size: "n" | "s" | "m" | "l" | "x". Used only when
        weights=None.
    task : str
        "detect" (default) for object detection.

    Raises
    ------
    ValueError        If size is not in VALID_SIZES.
    FileNotFoundError If a weights path is given but the file is missing.
    """
    if size not in VALID_SIZES:
        raise ValueError(
            f"Invalid backbone size '{size}'. "
            f"Choose from: {sorted(VALID_SIZES)}"
        )

    if weights is not None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights}\n"
                "Run train.py first, or place a .pt file in src/models/weights/"
            )
        logger.info("Loading weights: %s", weights)
        return YOLO(str(weights), task=task)

    backbone = f"yolov8{size}.pt"
    logger.info("Loading pretrained backbone: %s", backbone)
    return YOLO(backbone, task=task)


def get_finetuned_model(version: str = "v1") -> YOLO:
    """
    Load a named KestrelEye checkpoint from src/models/weights/.

    Expected filename: kestreleye_<version>.pt
    """
    path = WEIGHTS_DIR / f"kestreleye_{version}.pt"
    return get_model(weights=path)


def get_model_info(model: YOLO) -> dict:
    """Return a summary dict -- useful for logging at pipeline start."""
    return {
        "task":        model.task,
        "num_classes": len(model.names) if model.names else "unknown",
        "class_names": model.names,
    }