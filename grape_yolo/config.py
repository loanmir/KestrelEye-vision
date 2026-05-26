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
BASE_PATH = os.path.join(PROJECT_ROOT, "data", "UAV_dataset")

IMG_TRAIN = os.path.join(BASE_PATH, "train", "images")
IMG_VAL   = os.path.join(BASE_PATH, "valid", "images")

LBL_TRAIN = os.path.join(BASE_PATH, "train", "labels")
LBL_VAL   = os.path.join(BASE_PATH, "valid", "labels")

# Path to the data.yaml consumed by Ultralytics
YAML_PATH = os.path.join(BASE_PATH, "data.yaml")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
"""CLASS_NAMES = [
    "Grape__BlackRot",
    "Grape__Esca",
    "Grape__Healthy",
    "Grape__LeafBlight",
]"""

CLASS_NAMES = ["unhealthy"]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"   # base checkpoint
EPOCHS        = 30
IMG_SIZE      = 640
BATCH_SIZE    = 16
PROJECT       = "grape_UAV_project"
RUN_NAME      = "exp"
DEVICE        = "cpu"
WORKERS       = 8 

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.25