"""
Run inference with a trained YOLO11-seg model on images or video.

Usage:
    python predict.py --source path/to/images
    python predict.py --source path/to/photo.jpg --crops
    python predict.py --source path/to/clip.mp4 --track
"""

import argparse
from glob import glob
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from config import CONF_THRESHOLD

# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = "weights/best.pt"   # <-- your trained YOLO11-seg model
DEFAULT_IMGSZ   = 1024                          # match training; use 640 for video
DEFAULT_PROJECT = "runs/segment_crop"               # parent dir for saved results
DEFAULT_NAME    = "predict"                     # subfolder name (auto-incremented unless --overwrite)
VIDEO_EXTS      = {".mp4", ".mov", ".avi", ".mkv"}


def pick_device():
    """Inference device: CUDA > MPS > CPU. Safe on both Windows and Mac."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_crops(result, crops_dir, pad=10, gain=1.02):
    """Save each detected instance as its own JPEG, skipping degenerate (zero-area) boxes.

    Ultralytics' built-in save_crop=True raises "cannot write empty image as JPEG" and aborts
    the whole predict() call when a box collapses to zero width/height after clipping (common
    with tiny or edge-of-frame detections from an undertrained model).
    """
    if result.boxes is None or len(result.boxes) == 0:
        return
    im = result.orig_img  # BGR np.ndarray
    h, w = im.shape[:2]
    stem = Path(result.path).stem
    for i, (x1, y1, x2, y2) in enumerate(result.boxes.xyxy.cpu().tolist()):
        bw, bh = (x2 - x1) * gain + pad, (y2 - y1) * gain + pad
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx1, cx2 = int(max(cx - bw / 2, 0)), int(min(cx + bw / 2, w))
        cy1, cy2 = int(max(cy - bh / 2, 0)), int(min(cy + bh / 2, h))
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # degenerate box after clipping, skip instead of crashing
        label = result.names[int(result.boxes.cls[i])]
        out_dir = Path(crops_dir) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_{i}.jpg"), im[cy1:cy2, cx1:cx2])


def predict(weights, source, conf, imgsz, save_crops=False, track=False,
            project=DEFAULT_PROJECT, name=DEFAULT_NAME, overwrite=False):
    model = YOLO(weights)
    device = pick_device()
    src = Path(source)
    is_video = src.suffix.lower() in VIDEO_EXTS

    print(f"model  : {weights}")
    print(f"source : {source}  ({'video' if is_video else 'image(s)'})")
    print(f"device : {device}   imgsz: {imgsz}   conf: {conf}\n")

    kwargs = dict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=True,
        project=project,
        name=name,
        exist_ok=overwrite,   # False (default) auto-increments predict, predict2, ... instead of overwriting
        verbose=False,
    )
    # NOTE: we save crops ourselves (via _save_crops) instead of passing save_crop=True to
    # ultralytics — its built-in save_crop raises and aborts the whole run on zero-area boxes.

    if is_video:
        kwargs["stream"] = True        # frame-by-frame, avoids loading whole video in RAM
        gen = model.track(tracker="bytetrack.yaml", persist=True, **kwargs) if track \
              else model.predict(**kwargs)
        n = 0
        save_dir = None
        for r in gen:                  # must consume the generator for it to run
            save_dir = save_dir or r.save_dir
            if save_crops:
                _save_crops(r, Path(save_dir) / "crops")
            n += 1
            if n % 50 == 0:
                print(f"  {n} frames processed")
        print(f"\nDone: {n} frames.")
        print(f"Results saved to {save_dir}/")
        return None

    results = model.predict(**kwargs)
    total = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"{len(results)} image(s), {total} vine instances detected.")
    if results:
        if save_crops:
            for r in results:
                _save_crops(r, Path(results[0].save_dir) / "crops")
        print(f"Results saved to {results[0].save_dir}/")
    return results


def show_predictions(predict_dir=f"{DEFAULT_PROJECT}/{DEFAULT_NAME}", n=5):
    """Open the first N saved prediction images in the default viewer."""
    from PIL import Image
    preds = sorted(glob(f"{predict_dir}/*.jpg")) + sorted(glob(f"{predict_dir}/*.png"))
    if not preds:
        print(f"No prediction images found in {predict_dir}")
        return
    for img in preds[:n]:
        Image.open(img).show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--source", required=True, help="image, folder, or video path")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--crops", action="store_true",
                   help="also save each detected vine as a cropped image")
    p.add_argument("--track", action="store_true",
                   help="video only: assign persistent IDs across frames (ByteTrack)")
    p.add_argument("--show", type=int, default=0,
                   help="open the first N result images after running")
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help="parent directory for saved results")
    p.add_argument("--name", default=DEFAULT_NAME,
                   help="subfolder name for this run's results")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing --name folder instead of auto-incrementing it")
    args = p.parse_args()

    results = predict(args.weights, args.source, args.conf, args.imgsz,
                       save_crops=args.crops, track=args.track,
                       project=args.project, name=args.name, overwrite=args.overwrite)

    if args.show:
        save_dir = results[0].save_dir if results else Path(args.project) / args.name
        show_predictions(predict_dir=str(save_dir), n=args.show)