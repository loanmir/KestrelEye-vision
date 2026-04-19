"""KestrelEye vision package."""

from __future__ import annotations

from src.models.model_loader import ModelConfig, create_model_from_config, load_yolo_model
from src.pipeline import (
	Detection,
	detections_to_dataframe,
	export_detections_csv,
	export_detections_json,
	run_directory_inference,
	run_image_inference,
)

__all__ = [
	"ModelConfig",
	"load_yolo_model",
	"create_model_from_config",
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
		from src import pipeline as pipeline_module

		return getattr(pipeline_module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
