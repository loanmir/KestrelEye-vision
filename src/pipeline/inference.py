"""Inference pipeline for YOLO object detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.models.model_loader import ModelConfig, create_model_from_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class Detection:
	"""Normalized prediction output for one bounding box."""

	image_path: str
	class_id: int
	class_name: str
	confidence: float
	x1: float
	y1: float
	x2: float
	y2: float
	width: int
	height: int

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


def _is_image(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _extract_detections(result: Any, image_path: Path, image_size: tuple[int, int]) -> list[Detection]:
	detections: list[Detection] = []
	raw_names = result.names or {}
	if isinstance(raw_names, dict):
		names_map = raw_names
	elif isinstance(raw_names, list):
		names_map = {idx: name for idx, name in enumerate(raw_names)}
	else:
		names_map = {}
	width, height = image_size

	if result.boxes is None:
		return detections

	for box in result.boxes:
		cls = int(box.cls[0].item())
		conf = float(box.conf[0].item())
		x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
		detections.append(
			Detection(
				image_path=str(image_path),
				class_id=cls,
			class_name=str(names_map.get(cls, str(cls))),
				confidence=conf,
				x1=x1,
				y1=y1,
				x2=x2,
				y2=y2,
				width=width,
				height=height,
			)
		)

	return detections


def run_image_inference(
	image_path: str | Path,
	config: ModelConfig | None = None,
	save_visualization: bool = False,
	output_dir: str | Path | None = None,
) -> list[Detection]:
	"""Run YOLO object detection on one image.

	Args:
		image_path: Path to input image.
		config: Optional model configuration. Uses defaults if omitted.
		save_visualization: If True, saves YOLO-rendered image output.
		output_dir: Directory for rendered output when save_visualization is True.

	Returns:
		List of detections found in the image.
	"""

	config = config or ModelConfig()
	image_path = Path(image_path)

	if not _is_image(image_path):
		raise FileNotFoundError(f"Image not found or unsupported format: {image_path}")

	model = create_model_from_config(config)

	predict_kwargs: dict[str, Any] = {
		"conf": config.confidence,
		"iou": config.iou,
		"imgsz": config.image_size,
		"verbose": False,
	}
	if config.device:
		predict_kwargs["device"] = config.device

	if save_visualization:
		output_dir = Path(output_dir) if output_dir else Path("output") / "raw"
		output_dir.mkdir(parents=True, exist_ok=True)
		predict_kwargs["save"] = True
		predict_kwargs["project"] = str(output_dir)
		predict_kwargs["name"] = "predictions"
		predict_kwargs["exist_ok"] = True

	results = model.predict(str(image_path), **predict_kwargs)

	with Image.open(image_path) as img:
		image_size = img.size

	if not results:
		return []

	return _extract_detections(results[0], image_path, image_size)


def run_directory_inference(
	image_dir: str | Path,
	config: ModelConfig | None = None,
	recursive: bool = True,
) -> list[Detection]:
	"""Run YOLO object detection on all images inside a directory."""

	image_dir = Path(image_dir)
	if not image_dir.exists() or not image_dir.is_dir():
		raise NotADirectoryError(f"Invalid image directory: {image_dir}")

	glob_pattern = "**/*" if recursive else "*"
	image_paths = sorted(path for path in image_dir.glob(glob_pattern) if _is_image(path))

	all_detections: list[Detection] = []
	for image_path in image_paths:
		all_detections.extend(run_image_inference(image_path, config=config))

	return all_detections
