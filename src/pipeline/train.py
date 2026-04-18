"""Simple YOLO train/val/test pipeline utilities."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO
import yaml


@dataclass(slots=True)
class TrainConfig:
    """Configuration for YOLO training."""

    data_yaml: str = "data/grape_dataset/data.yaml"
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    project: str = "data/output/raw"
    name: str = "train"
    patience: int = 20
    workers: int = 8


def _default_split_dir(data_yaml: str | Path, split: str) -> Path:
    dataset_root = Path(data_yaml).resolve().parent
    return dataset_root / split / "images"


def _resolve_dataset_root(data_yaml: Path, configured_root: str | None) -> Path:
    if not configured_root:
        return data_yaml.parent.resolve()

    root = Path(configured_root)
    if root.is_absolute():
        return root.resolve()

    # Try both YAML-relative and CWD-relative resolution, then pick an existing one.
    yaml_relative = (data_yaml.parent / root).resolve()
    cwd_relative = (Path.cwd() / root).resolve()

    if yaml_relative.exists():
        return yaml_relative
    if cwd_relative.exists():
        return cwd_relative

    return yaml_relative


def _resolve_split_images_dir(data_yaml: str | Path, split_key: str) -> Path:
    yaml_path = Path(data_yaml).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    dataset_root = _resolve_dataset_root(yaml_path, data.get("path"))
    split_value = data.get(split_key)
    if not split_value:
        raise ValueError(f"Missing '{split_key}' entry in dataset yaml: {yaml_path}")

    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path

    return (dataset_root / split_path).resolve()


def _build_resolved_data_yaml(data_yaml: str | Path) -> Path:
    """Build a temporary dataset yaml with absolute 'path' for Ultralytics."""

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    dataset_root = _resolve_dataset_root(yaml_path, data.get("path"))
    data["path"] = str(dataset_root)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="resolved_dataset_",
        delete=False,
        encoding="utf-8",
    )
    with temp_file:
        yaml.safe_dump(data, temp_file, sort_keys=False)

    return Path(temp_file.name)


def train_model(config: TrainConfig) -> dict[str, str]:
    """Train YOLO on the configured dataset and return key output paths."""

    data_yaml = Path(config.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    resolved_data_yaml = _build_resolved_data_yaml(data_yaml)

    model = YOLO(config.model)
    train_kwargs: dict[str, Any] = {
        "data": str(resolved_data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "project": config.project,
        "name": config.name,
        "patience": config.patience,
        "workers": config.workers,
    }
    if config.device:
        train_kwargs["device"] = config.device

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)

    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    return {
        "train_run_dir": str(save_dir),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
    }


def validate_model(
    weights_path: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
) -> Any:
    """Run YOLO validation on the requested dataset split (val or test)."""

    weights_path = Path(weights_path)
    data_yaml = Path(data_yaml)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
    if split not in {"val", "test"}:
        raise ValueError("split must be either 'val' or 'test'")

    model = YOLO(str(weights_path))
    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
    }
    if device:
        val_kwargs["device"] = device

    return model.val(**val_kwargs)


def predict_split_images(
    weights_path: str | Path,
    source_dir: str | Path,
    output_root: str | Path,
    run_name: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
) -> Path:
    """Run prediction on a folder and save rendered prediction images."""

    weights_path = Path(weights_path)
    source_dir = Path(source_dir)
    output_root = Path(output_root)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(f"Invalid source directory: {source_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    predict_kwargs: dict[str, Any] = {
        "source": str(source_dir),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "save": True,
        "project": str(output_root),
        "name": run_name,
        "exist_ok": True,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    model.predict(**predict_kwargs)
    return output_root / run_name


def run_simple_pipeline(
    config: TrainConfig,
    output_root: str | Path = "data/output/raw",
    valid_images_dir: str | Path | None = None,
    test_images_dir: str | Path | None = None,
) -> dict[str, str]:
    """Train model, validate on val/test, and save rendered predictions for val/test."""

    resolved_data_yaml = _build_resolved_data_yaml(config.data_yaml)

    training_paths = train_model(config)
    best_weights = training_paths["best_weights"]

    validate_model(
        weights_path=best_weights,
        data_yaml=resolved_data_yaml,
        split="val",
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
    )
    validate_model(
        weights_path=best_weights,
        data_yaml=resolved_data_yaml,
        split="test",
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
    )

    val_source = (
        Path(valid_images_dir)
        if valid_images_dir
        else _resolve_split_images_dir(resolved_data_yaml, "val")
    )
    test_source = (
        Path(test_images_dir)
        if test_images_dir
        else _resolve_split_images_dir(resolved_data_yaml, "test")
    )

    if not val_source.exists():
        raise FileNotFoundError(f"Validation images folder not found: {val_source}")
    if not test_source.exists():
        raise FileNotFoundError(f"Test images folder not found: {test_source}")

    val_pred_dir = predict_split_images(
        weights_path=best_weights,
        source_dir=val_source,
        output_root=output_root,
        run_name="predictions_val",
        imgsz=config.imgsz,
        device=config.device,
    )
    test_pred_dir = predict_split_images(
        weights_path=best_weights,
        source_dir=test_source,
        output_root=output_root,
        run_name="predictions_test",
        imgsz=config.imgsz,
        device=config.device,
    )

    return {
        **training_paths,
        "predictions_val_dir": str(val_pred_dir),
        "predictions_test_dir": str(test_pred_dir),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO on grape dataset")
    parser.add_argument("--data", default="data/grape_dataset/data.yaml", help="Path to dataset yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or path to .pt weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="data/output/raw")
    parser.add_argument("--name", default="train")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = TrainConfig(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    outputs = run_simple_pipeline(config=config, output_root=args.project)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()