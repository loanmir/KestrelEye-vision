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
MODEL_WEIGHTS = "yolov8s.pt"   # medium model — good balance for small UAV objects
EPOCHS        = 15
IMG_SIZE      = 640
BATCH_SIZE    = 16              # reduced because 1280px uses more memory
PROJECT       = "grape_UAV_project"
RUN_NAME      = "exp"
DEVICE        = "cpu"          # Apple Silicon GPU
WORKERS       = 8              # MPS can choke with too many workers

# Optimizer & learning rate
#OPTIMIZER     = "AdamW"        # more stable than SGD on small single-class datasets
#LR0           = 0.001          # AdamW works best at lower initial LR
#LRF           = 0.01           # final LR = LR0 * LRF = 1e-5

# Regularisation & scheduling
#WEIGHT_DECAY  = 0.0005
PATIENCE      = 20             # early stopping — stops if val mAP stalls for 20 epochs
#CLOSE_MOSAIC  = 10             # disable mosaic in last N epochs for stable fine-tuning

# Augmentation (UAV-specific)
#FLIPUD        = 0.5            # vertical flip — drones have no fixed orientation
#DEGREES       = 10             # mild rotation to handle drone heading variation

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.25