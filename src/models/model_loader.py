"""Utilities for loading and configuring YOLO models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass(slots=True)
class ModelConfig:
	"""Configuration for YOLO model loading and inference defaults."""

	model_path: str = "yolov8n.pt"
	confidence: float = 0.25
	iou: float = 0.45
	image_size: int = 640
	device: str | None = None


def load_yolo_model(model_path: str = "yolov8n.pt") -> YOLO:
	"""Load a YOLO model from a local path or model name.

	Args:
		model_path: Local path to a .pt/.onnx model, or a YOLO model alias.

	Returns:
		A ready-to-use Ultralytics YOLO model instance.

	Raises:
		FileNotFoundError: If a local file path is provided but does not exist.
	"""

	path = Path(model_path)
	looks_like_local_file = path.suffix in {".pt", ".onnx", ".engine", ".torchscript"}
	if looks_like_local_file and not path.exists():
		raise FileNotFoundError(f"Model file not found: {path}")

	return YOLO(model_path)


def create_model_from_config(config: ModelConfig) -> YOLO:
	"""Convenience wrapper to build a model from a configuration object."""

	return load_yolo_model(config.model_path)
