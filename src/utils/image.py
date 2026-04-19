"""
src/utils/image.py
-------------------
Image validation and bounding-box annotation helpers.

Used by run_pipeline.py to:
  1. Filter corrupted files before inference.
  2. Draw bounding boxes + labels on detections and save annotated JPEGs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Colour per pathology class (edit freely to match your dataset's class names)
PATHOLOGY_COLORS: dict[str, str] = {
    "black_rot":          "#E74C3C",
    "esca":               "#8E44AD",
    "leaf_blight":        "#E67E22",
    "powdery_mildew":     "#F1C40F",
    "downy_mildew":       "#3498DB",
    "botrytis":           "#9B59B6",
    "healthy":            "#2ECC71",
    "default":            "#95A5A6",
}

_FONT: ImageFont.ImageFont | None = None


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            _FONT = ImageFont.load_default()
    return _FONT


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_image(path: str | Path) -> bool:
    """
    Return True if the file is readable and not corrupted.

    Uses PIL verify() which is fast but does not decode pixel data.
    A second open() is needed after verify() because PIL invalidates
    the file handle after calling it.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as exc:
        logger.warning("Corrupted / unreadable: %s -- %s", path, exc)
        return False


def filter_valid(image_paths: list[Path]) -> list[Path]:
    """
    Return only the images that pass validate_image().

    Logs a warning for every rejected file.
    """
    valid = [p for p in image_paths if validate_image(p)]
    n_bad = len(image_paths) - len(valid)
    if n_bad:
        logger.warning("Filtered out %d corrupted image(s).", n_bad)
    return valid


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_image(
    image_path: str | Path,
    result,                        # ultralytics.engine.results.Results
    output_dir: str | Path,
    box_thickness: int = 3,
    font_size: int = 14,
) -> Path:
    """
    Draw bounding boxes on one image and save it to output_dir.

    Only called when result.boxes is non-empty.

    Returns
    -------
    Path  Path to the saved annotated JPEG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)

    for box in result.boxes:
        cls_name = result.names[int(box.cls)]
        conf     = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = PATHOLOGY_COLORS.get(cls_name, PATHOLOGY_COLORS["default"])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

        label = f"{cls_name}  {conf:.0%}"
        try:
            lx1, ly1, lx2, ly2 = draw.textbbox(
                (x1, y1 - font_size - 4), label, font=font
            )
        except AttributeError:
            lx1, ly1 = x1, y1 - font_size - 4
            lx2, ly2 = lx1 + len(label) * 7, ly1 + font_size + 2

        draw.rectangle([lx1, ly1, lx2, ly2], fill=color)
        draw.text((lx1 + 2, ly1 + 1), label, fill="white", font=font)

    out = output_dir / f"{Path(image_path).stem}_annotated.jpg"
    img.save(out, quality=92, optimize=True)
    return out


def annotate_batch(
    results,               # list[Results]
    output_dir: str | Path,
) -> list[Path]:
    """
    Annotate and save every image that has at least one detection.

    Images with zero detections are skipped.

    Returns
    -------
    list[Path]  Paths of all saved annotated images.
    """
    saved = []
    for result in results:
        if len(result.boxes) == 0:
            continue
        out = annotate_image(result.path, result, output_dir)
        saved.append(out)
    logger.info("Annotated %d image(s) -> %s", len(saved), output_dir)
    return saved