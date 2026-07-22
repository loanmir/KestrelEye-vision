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
BASE_PATH = os.path.join(PROJECT_ROOT, "data", "vine_seg_yolo") 

IMG_TRAIN = os.path.join(BASE_PATH, "train", "images")
IMG_VAL   = os.path.join(BASE_PATH, "valid", "images")
LBL_TRAIN = os.path.join(BASE_PATH, "train", "labels")
LBL_VAL   = os.path.join(BASE_PATH, "valid", "labels")

# Path to the data.yaml consumed by Ultralytics
YAML_PATH = os.path.join(BASE_PATH, "data.yaml")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

CLASS_NAMES = ["vine"]                                            

# Training hyperparameters
MODEL_WEIGHTS = "yolo11s-seg.pt"   # <-- SEGMENTATION model (the -seg suffix is essential)
EPOCHS        = 100
IMG_SIZE      = 1024               # aerial vines are small; higher res helps. drop to 640 if OOM
BATCH_SIZE    = 16                 # lower to 8/4 if you hit memory limits
PROJECT       = "vine_seg_project"
RUN_NAME      = "seg_exp"
DEVICE        = 0                  # 0 = first GPU (Colab). use "cpu" locally, "mps" on the Mac
WORKERS       = 8
PATIENCE      = 30