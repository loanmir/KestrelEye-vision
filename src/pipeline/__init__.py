"""Pipeline exports for inference, training, and output serialization."""

from __future__ import annotations

from src.pipeline.export import (
	detections_to_dataframe,
	export_detections_csv,
	export_detections_json,
)
from src.pipeline.inference import Detection, run_directory_inference, run_image_inference

__all__ = [
	"Detection",
	"run_image_inference",
	"run_directory_inference",
	"detections_to_dataframe",
	"export_detections_csv",
	"export_detections_json",
	"TrainConfig",
	"train_model",
	"validate_model",
	"predict_split_images",
	"run_simple_pipeline",
]


def __getattr__(name: str):
	if name in {
		"TrainConfig",
		"train_model",
		"validate_model",
		"predict_split_images",
		"run_simple_pipeline",
	}:
		from src.pipeline import train as train_module

		return getattr(train_module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
