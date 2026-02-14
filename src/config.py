import os
from pathlib import Path

# 1. Get the absolute path of the project root
# This works no matter where you run the script from
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent  # Go up two levels (src -> root)

# 2. Define Key Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# 3. Define Model Paths (Single Source of Truth)
MODEL_PATHS = {
    "yolo": MODELS_DIR / "yolov8n.pt",
    "mobilenet": MODELS_DIR / "mobilenet_v3.pth",
    "resnet": MODELS_DIR / "resnet18.pth",
    # The compiled C++ executable path
    "rce_cpp_exe": SRC_DIR / "cpp_engine" / "build" / "rce_engine"
}

# 4. Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR / "logs", exist_ok=True)
os.makedirs(RESULTS_DIR / "plots", exist_ok=True)