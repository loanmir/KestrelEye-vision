"""
Central configuration for the grape leaf disease detection project.
Edit paths and hyperparameters here, then import from any other script.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_PATH = os.path.join(PROJECT_ROOT, "data", "kaggle_dataset")

IMG_TRAIN = os.path.join(BASE_PATH, "images", "train")
IMG_VAL   = os.path.join(BASE_PATH, "images", "val")

LBL_TRAIN = os.path.join(BASE_PATH, "labels", "train")
LBL_VAL   = os.path.join(BASE_PATH, "labels", "val")

# Path to the data.yaml consumed by Ultralytics
YAML_PATH = os.path.join(BASE_PATH, "data.yaml")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Grape__BlackRot",
    "Grape__Esca",
    "Grape__Healthy",
    "Grape__LeafBlight",
]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"   # base checkpoint
EPOCHS        = 30
IMG_SIZE      = 640
BATCH_SIZE    = 16
PROJECT       = "grape_project"
RUN_NAME      = "exp"
DEVICE        = "mps"
WORKERS       = 8 

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.25