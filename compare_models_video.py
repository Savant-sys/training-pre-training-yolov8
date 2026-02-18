"""
Run a sample video through pretrained YOLOv8 (root yolov8n.pt) and your trained
model (runs/detect/train/weights/best.pt), and save two output videos to compare.
"""
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    raise SystemExit(1)

# Project root: pretrained model lives here (run setup_yolo_model.py to download + backup)
ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED = ROOT / "yolov8n.pt"
DEFAULT_TRAINED = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"


def run_video(model_path: str, video_path: str, project: str, name: str):
    """Run detection on video; saves to project/name/."""
    model = YOLO(model_path)
    model.predict(
        source=video_path,
        save=True,
        project=project,
        name=name,
        exist_ok=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained vs your trained YOLO on a video")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--pretrained", type=str, default=str(DEFAULT_PRETRAINED), help="Pretrained model in project root (default: root yolov8n.pt)")
    parser.add_argument("--trained", type=str, default=None, help="Your trained weights (default: runs/detect/train/weights/best.pt)")
    parser.add_argument("--out-dir", type=str, default="comparison_output", help="Folder for output videos")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trained_weights = args.trained
    if trained_weights is None:
        candidate = DEFAULT_TRAINED
        if candidate.exists():
            trained_weights = str(candidate)
        else:
            print("Trained model not found at runs/detect/train/weights/best.pt")
            print("Train first, or pass --trained path/to/best.pt")
            return 1
    if not Path(trained_weights).exists():
        print(f"Trained model not found: {trained_weights}")
        return 1

    print("Running PRETRAINED model:", args.pretrained)
    run_video(args.pretrained, str(video_path), str(out_dir), "pretrained")

    print("Running YOUR TRAINED model:", trained_weights)
    run_video(trained_weights, str(video_path), str(out_dir), "trained")

    print("\nDone. Output videos are in:")
    print("  -", out_dir.resolve() / "pretrained")
    print("  -", out_dir.resolve() / "trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
