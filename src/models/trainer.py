"""
src/models/trainer.py
---------------------
Wraps the Ultralytics training loop for KestrelEye.

Usage
-----
    from src.models.loader import get_model
    from src.models.trainer import Trainer, TrainConfig

    model   = get_model(size="n")
    trainer = Trainer(model, data_yaml="data/grape_disease/data.yaml")
    results = trainer.train()
    print(trainer.best_weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """
    All hyper-parameters for one training run.

    Sensible Fase-0 defaults are set so Trainer.train() works with zero args.

    Attributes
    ----------
    epochs : int
        Full passes over the training set. 50 is a good start for ~1 000
        images; increase to 100-150 for larger datasets.
    imgsz : int
        Square input resolution fed to the model. 640 is the YOLOv8 default.
    batch : int
        Images per gradient step. 16 fits in 8 GB RAM on an M-series Mac.
        Drop to 8 if you see out-of-memory errors.
    lr0 : float
        Initial learning rate.
    lrf : float
        Final LR as a fraction of lr0 (cosine annealing schedule).
    momentum : float
        SGD momentum.
    weight_decay : float
        L2 regularisation.
    warmup_epochs : float
        Epochs to ramp LR from 0 to lr0.
    patience : int
        Early-stopping: halt if val mAP50 does not improve for this many
        consecutive epochs. Set 0 to disable.
    device : str
        "mps"  -> Apple Silicon GPU  (fastest on M1/M2/M3)
        "cpu"  -> always works, slower
        "0"    -> CUDA GPU (Jetson / NVIDIA workstation)
    project : str
        Parent folder for all run artefacts.
    name : str
        Subfolder inside project (Ultralytics auto-increments if it exists).
    exist_ok : bool
        Overwrite an existing run with the same name when True.
    pretrained : bool
        Start from COCO pretrained weights. Always True for Fase 0 --
        random init on a small dataset diverges.
    cache : bool
        Cache images in RAM. Speeds up training on small datasets that fit
        in memory; leave False for large ones.
    workers : int
        DataLoader worker processes. 0 = main thread only (safest on Mac).
    save_period : int
        Save a checkpoint every N epochs. -1 = save only best + last.
    """

    epochs: int         = 50
    imgsz: int          = 640
    batch: int          = 16
    lr0: float          = 0.01
    lrf: float          = 0.01
    momentum: float     = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    patience: int       = 20
    device: str         = "mps"
    project: str        = "runs/train"
    name: str           = "kestreleye_grape"
    exist_ok: bool      = False
    pretrained: bool    = True
    cache: bool         = False
    workers: int        = 0
    save_period: int    = -1


class Trainer:
    """
    Encapsulates a YOLOv8 training run for KestrelEye.

    Parameters
    ----------
    model : YOLO
        Model returned by src.models.loader.get_model().
    data_yaml : str | Path
        Path to the dataset YAML (Roboflow / YOLO format).
        Must contain 'train', 'val', 'nc', and 'names' keys.
    config : TrainConfig | None
        Hyper-parameters. Uses class defaults when None.
    """

    def __init__(
        self,
        model: YOLO,
        data_yaml: str | Path,
        config: TrainConfig | None = None,
    ) -> None:
        self.model     = model
        self.data_yaml = Path(data_yaml).resolve()
        self.config    = config or TrainConfig()
        self._results  = None
        self._effective_data_yaml = self.data_yaml

        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {self.data_yaml}\n"
                "Expected layout:\n"
                "  data/grape_disease/\n"
                "    data.yaml\n"
                "    train/images/  train/labels/\n"
                "    valid/images/  valid/labels/\n"
                "    test/images/   test/labels/  (optional)"
            )

        self._effective_data_yaml = self._prepare_data_yaml_for_ultralytics()

    def _prepare_data_yaml_for_ultralytics(self) -> Path:
        """
        Return a YAML path with an absolute dataset root to avoid Ultralytics
        resolving relative paths against its global DATASETS_DIR setting.
        """
        data = yaml_load(self.data_yaml)
        raw_path = data.get("path")

        if raw_path is None:
            return self.data_yaml

        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            return self.data_yaml

        candidates = [
            (Path.cwd() / path_obj).resolve(),
            (self.data_yaml.parent / path_obj).resolve(),
            self.data_yaml.parent.resolve(),
        ]
        resolved_root = next((p for p in candidates if p.exists()), candidates[0])

        data["path"] = str(resolved_root)

        generated_yaml = self.data_yaml.parent / f"{self.data_yaml.stem}.resolved.yaml"
        yaml_save(generated_yaml, data)
        logger.info("Resolved dataset root to: %s", resolved_root)
        logger.info("Using generated dataset YAML: %s", generated_yaml)
        return generated_yaml

    def train(self, **overrides) -> object:
        """
        Launch training.

        Any TrainConfig field can be overridden at call time:
            trainer.train(epochs=100, batch=8, device="cpu")

        Returns the Ultralytics results object. Key attributes:
            results.results_dict  -- dict of final metrics
            results.save_dir      -- Path to run artefacts folder
        """
        cfg = {**self.config.__dict__, **overrides}

        logger.info("=" * 56)
        logger.info("KestrelEye -- training run starting")
        logger.info("  data    : %s", self._effective_data_yaml)
        logger.info("  epochs  : %s", cfg["epochs"])
        logger.info("  imgsz   : %s", cfg["imgsz"])
        logger.info("  batch   : %s", cfg["batch"])
        logger.info("  device  : %s", cfg["device"])
        logger.info("  project : %s/%s", cfg["project"], cfg["name"])
        logger.info("=" * 56)

        self._results = self.model.train(
            data=str(self._effective_data_yaml),
            **cfg,
        )

        self._log_summary()
        return self._results

    def _log_summary(self) -> None:
        if self._results is None:
            return
        try:
            logger.info("=" * 56)
            logger.info("Training complete -- final metrics:")
            for k, v in self._results.results_dict.items():
                logger.info("  %-30s %.4f", k, v)
            logger.info("  Saved to: %s", self._results.save_dir)
            logger.info("=" * 56)
        except Exception:
            logger.warning("Could not parse training results dict.")

    @property
    def best_weights(self) -> Path | None:
        """Path to best.pt, or None if training has not run yet."""
        if self._results is None:
            return None
        p = Path(self._results.save_dir) / "weights" / "best.pt"
        return p if p.exists() else None

    @property
    def last_weights(self) -> Path | None:
        """Path to last.pt (most recent checkpoint)."""
        if self._results is None:
            return None
        p = Path(self._results.save_dir) / "weights" / "last.pt"
        return p if p.exists() else None