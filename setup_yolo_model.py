"""
Download YOLOv8n to project root, back it up to backup/, and keep root copy
as the pretrained model for training and comparison.
"""
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "yolov8n.pt"
# Ultralytics assets release URL (update version if needed)
DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
BACKUP_DIR = ROOT / "backup"


def main():
    root_model = ROOT / MODEL_NAME
    backup_model = BACKUP_DIR / MODEL_NAME

    BACKUP_DIR.mkdir(exist_ok=True)

    if not root_model.exists():
        print(f"Downloading {MODEL_NAME} to root...")
        try:
            urllib.request.urlretrieve(DOWNLOAD_URL, root_model)
            print(f"  -> {root_model}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Fallback: run 'python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"' from project root to cache the model, then copy it here.")
            return 1
    else:
        print(f"Root model exists: {root_model}")

    shutil.copy2(root_model, backup_model)
    print(f"Backup copied to: {backup_model}")

    print("\nSetup done. You can now:")
    print("  - Train (uses root pretrained):  yolo train model=yolov8n.pt data=bdd100k_yolo/data.yaml epochs=100 imgsz=640")
    print("  - Compare on video:               python compare_models_video.py your_video.mov")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
