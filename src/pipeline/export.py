"""Export helpers for inference results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.pipeline.inference import Detection


def detections_to_dataframe(detections: Iterable[Detection]) -> pd.DataFrame:
	"""Convert detections into a tabular representation."""

	rows = [d.to_dict() for d in detections]
	return pd.DataFrame(rows)


def export_detections_csv(detections: Iterable[Detection], output_path: str | Path) -> Path:
	"""Export detections to CSV file and return the resolved output path."""

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df = detections_to_dataframe(detections)
	df.to_csv(output_path, index=False)
	return output_path


def export_detections_json(detections: Iterable[Detection], output_path: str | Path) -> Path:
	"""Export detections to JSON file and return the resolved output path."""

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	payload = [d.to_dict() for d in detections]
	output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return output_path
