"""
src/pipeline/export.py
-----------------------
Turns raw inference results into a pandas DataFrame and saves it to disk.

DataFrame schema  (one row = one detection)
-------------------------------------------
filename      str    DJI_0042.JPG
pathology     str    black_rot
confidence    float  0.7821
severity      str    alta | media | bassa
bbox_x1       float  pixel x of top-left corner
bbox_y1       float  pixel y of top-left corner
bbox_x2       float  pixel x of bottom-right corner
bbox_y2       float  pixel y of bottom-right corner
image_w       int    original image width  (px)
image_h       int    original image height (px)

Images where the model finds NO detections still contribute one row
with pathology=None so the coverage log accounts for every analysed image.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pandas as pd
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

# Confidence thresholds for the human-readable severity label
SEVERITY_HIGH   = 0.80
SEVERITY_MEDIUM = 0.60


def _severity(conf: float) -> str:
    if conf >= SEVERITY_HIGH:
        return "alta"
    if conf >= SEVERITY_MEDIUM:
        return "media"
    return "bassa"


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(results: list[Results]) -> pd.DataFrame:
    """
    Build the master detection DataFrame from inference results.

    Parameters
    ----------
    results : list[Results]
        Output of src.pipeline.inference.run_inference().

    Returns
    -------
    pd.DataFrame
        All detections (and blank rows for images with no detections).
        Sorted by filename ascending.
    """
    records: list[dict] = []

    for result in results:
        filename = Path(result.path).name
        h, w     = result.orig_shape   # (height, width)

        if len(result.boxes) == 0:
            records.append({
                "filename":   filename,
                "pathology":  None,
                "confidence": None,
                "severity":   None,
                "bbox_x1":    None,
                "bbox_y1":    None,
                "bbox_x2":    None,
                "bbox_y2":    None,
                "image_w":    w,
                "image_h":    h,
            })
        else:
            for box in result.boxes:
                conf     = float(box.conf)
                cls_name = result.names[int(box.cls)]
                x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]

                records.append({
                    "filename":   filename,
                    "pathology":  cls_name,
                    "confidence": round(conf, 4),
                    "severity":   _severity(conf),
                    "bbox_x1":    x1,
                    "bbox_y1":    y1,
                    "bbox_x2":    x2,
                    "bbox_y2":    y2,
                    "image_w":    w,
                    "image_h":    h,
                })

    df = pd.DataFrame(records).sort_values("filename").reset_index(drop=True)

    n_det = df["pathology"].notna().sum()
    logger.info(
        "DataFrame built: %d detections in %d images",
        n_det, df["filename"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save the DataFrame as a UTF-8 CSV.

    Importable into QGIS, Google Sheets, and Excel.

    Returns
    -------
    Path  Resolved path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("CSV  -> %s  (%d rows)", output_path, len(df))
    return output_path


def export_json(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save detections as a JSON array (one object per detection).

    Useful for downstream processing or a simple REST API response.

    Returns
    -------
    Path  Resolved path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positives = df.dropna(subset=["pathology"])
    records   = positives.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info("JSON -> %s  (%d detections)", output_path, len(records))
    return output_path


def export_summary_txt(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Write a plain-text summary report.

    Contains per-class counts, severity distribution, and confidence stats.
    Easy to read in a terminal or attach to an email.

    Returns
    -------
    Path  Resolved path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 52)
    lines.append("  KestrelEye -- Detection Summary")
    lines.append("=" * 52)

    total_images = df["filename"].nunique()
    total_det    = df["pathology"].notna().sum()
    hit_images   = df.loc[df["pathology"].notna(), "filename"].nunique()

    lines.append(f"  Images analysed   : {total_images}")
    lines.append(f"  Images with hits  : {hit_images}")
    lines.append(f"  Total detections  : {total_det}")

    if total_det > 0:
        lines.append("")
        lines.append("  By pathology:")
        for cls, n in df["pathology"].value_counts().items():
            pct = n / total_det * 100
            lines.append(f"    {cls:<26} {n:>4}  ({pct:.1f}%)")

        lines.append("")
        lines.append("  By severity:")
        for sev, n in df["severity"].value_counts().items():
            lines.append(f"    {sev:<26} {n:>4}")

        conf = df["confidence"].dropna()
        if not conf.empty:
            lines.append("")
            lines.append(f"  Mean confidence   : {conf.mean():.3f}")
            lines.append(f"  Min  confidence   : {conf.min():.3f}")
            lines.append(f"  Max  confidence   : {conf.max():.3f}")

    lines.append("=" * 52)
    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    logger.info("TXT  -> %s", output_path)
    return output_path


def print_summary(df: pd.DataFrame) -> None:
    """Print the summary to stdout (mirrors export_summary_txt)."""
    total_images = df["filename"].nunique()
    total_det    = df["pathology"].notna().sum()
    hit_images   = df.loc[df["pathology"].notna(), "filename"].nunique()

    print("\n" + "=" * 52)
    print("  KestrelEye -- Detection Summary")
    print("=" * 52)
    print(f"  Images analysed   : {total_images}")
    print(f"  Images with hits  : {hit_images}")
    print(f"  Total detections  : {total_det}")

    if total_det > 0:
        print("\n  By pathology:")
        for cls, n in df["pathology"].value_counts().items():
            print(f"    {cls:<26} {n:>4}  ({n/total_det*100:.1f}%)")
        print("\n  By severity:")
        for sev, n in df["severity"].value_counts().items():
            print(f"    {sev:<26} {n:>4}")
        conf = df["confidence"].dropna()
        if not conf.empty:
            print(f"\n  Mean confidence   : {conf.mean():.3f}")
            print(f"  Min  confidence   : {conf.min():.3f}")
            print(f"  Max  confidence   : {conf.max():.3f}")

    print("=" * 52 + "\n")