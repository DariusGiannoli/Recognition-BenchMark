import torch
import sys
from pathlib import Path
from ultralytics import YOLO
from torchvision import models

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

# NOW we can import from src.config
from src.config import MODELS_DIR, MODEL_PATHS

print(f"⬇️ DOWNLOADING MODELS TO: {MODELS_DIR}\n")

# --- 1. YOLOv8 ---
print("1️⃣ Downloading YOLOv8 Nano...")
# Ultralytics downloads to current dir by default, so we move it
model = YOLO('yolov8n.pt') 
src_path = Path('yolov8n.pt')
dest_path = MODEL_PATHS['yolo']

if src_path.exists():
    src_path.rename(dest_path)
    print(f"✅ Moved to {dest_path}")
else:
    print(f"⚠️ Check if {dest_path} already exists.")

# --- 2. MOBILENET ---
print("\n2️⃣ Downloading MobileNetV3...")
mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
torch.save(mobilenet.state_dict(), MODEL_PATHS['mobilenet'])
print(f"✅ Saved to {MODEL_PATHS['mobilenet']}")

# --- 3. RESNET ---
print("\n3️⃣ Downloading ResNet-18...")
resnet = models.resnet18(weights='DEFAULT')
torch.save(resnet.state_dict(), MODEL_PATHS['resnet'])
print(f"✅ Saved to {MODEL_PATHS['resnet']}")

print("\n🎉 ALL SYSTEMS GO.")