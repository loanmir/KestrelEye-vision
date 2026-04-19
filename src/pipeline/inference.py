"""
src/pipeline/inference.py
--------------------------
Runs YOLOv8 batch inference on a folder of images and returns raw results.

This module handles only the inference step.
Result storage -> src/pipeline/export.py
Image utils   -> src/utils/image.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(source: str | Path) -> list[Path]:
    """
    Recursively collect all supported image files under source.

    Parameters
    ----------
    source : str | Path
        A directory or a single image file.

    Returns
    -------
    list[Path]
        Sorted list of image paths.

    Raises
    ------
    FileNotFoundError  If source does not exist.
    ValueError         If source is a directory with no supported images,
                       or a file with an unsupported extension.
    """
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {source.suffix}")
        return [source]

    images = sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(
            f"No images found under {source}.\n"
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    logger.info("Found %d images in %s", len(images), source)
    return images


def run_inference(
    model: YOLO,
    images: list[Path],
    conf: float = 0.45,
    iou: float = 0.50,
    imgsz: int = 640,
    device: str = "mps",
    verbose: bool = False,
) -> list[Results]:
    """
    Run YOLOv8 inference on a list of image paths.

    Parameters
    ----------
    model : YOLO
        Loaded model from src.models.loader.get_model().
    images : list[Path]
        Output of collect_images().
    conf : float
        Minimum confidence threshold to keep a detection.
    iou : float
        IoU threshold for Non-Maximum Suppression.
    imgsz : int
        Input resolution. Must match the resolution used during training.
    device : str
        "mps" (Apple Silicon), "cpu", or "0" (CUDA).
    verbose : bool
        Print per-image stats. Keep False in normal use.

    Returns
    -------
    list[Results]
        One Results object per input image. Each exposes:
            result.path        -- str, source image path
            result.boxes       -- Boxes object (may be empty)
            result.boxes.cls   -- tensor of class IDs
            result.boxes.conf  -- tensor of confidence scores
            result.boxes.xyxy  -- tensor of [x1, y1, x2, y2] pixel coords
            result.names       -- dict {class_id: class_name}
    """
    if not images:
        logger.warning("run_inference called with empty image list.")
        return []

    logger.info(
        "Inference: %d images  [conf=%.2f  iou=%.2f  imgsz=%d  device=%s]",
        len(images), conf, iou, imgsz, device,
    )

    results: list[Results] = list(
        model.predict(
            source=[str(p) for p in images],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,    # generator mode: avoids loading all results into RAM
            save=False,     # we save ourselves in export.py
            verbose=verbose,
        )
    )

    n = sum(len(r.boxes) for r in results)
    logger.info("Inference complete -- %d detections across %d images", n, len(results))
    return results


def quick_predict(
    model: YOLO,
    image_path: str | Path,
    conf: float = 0.45,
    device: str = "mps",
) -> Results:
    """
    Single-image prediction. Useful for debugging in a notebook or REPL.

    Example
    -------
    >>> from src.models.loader import get_model
    >>> from src.pipeline.inference import quick_predict
    >>> model  = get_model(weights="runs/train/kestreleye_grape/weights/best.pt")
    >>> result = quick_predict(model, "data/raw/DJI_0001.JPG")
    >>> for box in result.boxes:
    ...     print(result.names[int(box.cls)], f"{float(box.conf):.2%}")
    """
    return run_inference(
        model=model,
        images=[Path(image_path)],
        conf=conf,
        device=device,
        verbose=True,
    )[0]