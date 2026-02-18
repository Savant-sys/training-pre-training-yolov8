"""
Run a sample video through pretrained YOLOv8 (root yolov8n.pt) and your trained
model (runs/detect/train/weights/best.pt), and save two output videos to compare.
Outputs MP4 files (smaller and widely supported).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    raise SystemExit(1)

# Project root: pretrained model lives here (run setup_yolo_model.py to download + backup)
ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED = ROOT / "yolov8n.pt"
DEFAULT_TRAINED = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"


def _get_video_info(video_path: str):
    """Get fps and (width, height) from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, (w, h)


def run_video(model_path: str, video_path: str, project: str, name: str):
    """Run detection on video; saves as MP4 to project/name/."""
    model = YOLO(model_path)
    fps, (w, h) = _get_video_info(video_path)

    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_name = Path(video_path).stem + ".mp4"
    out_path = save_dir / out_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        for result in model.predict(
            source=video_path,
            stream=True,
            save=False,
        ):
            frame = result.plot()  # BGR numpy array
            if frame is not None:
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
    finally:
        writer.release()

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained vs your trained YOLO on a video")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--pretrained", type=str, default=str(DEFAULT_PRETRAINED), help="Pretrained model in project root (default: yolov8n.pt)")
    parser.add_argument("--trained", type=str, default=None, help="Your trained weights (default: runs/detect/train/weights/best.pt)")
    parser.add_argument("--out-dir", type=str, default="comparison_output", help="Folder for output videos")
    parser.add_argument("--pretrained-only", action="store_true", help="Run only pretrained model (no trained weights needed)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trained_weights = args.trained
    if not args.pretrained_only:
        if trained_weights is None:
            candidate = DEFAULT_TRAINED
            if candidate.exists():
                trained_weights = str(candidate)
            else:
                print("Trained model not found at runs/detect/train/weights/best.pt")
                print("Train first, or pass --trained path/to/best.pt")
                print("Or use --pretrained-only to run only the pretrained model.")
                return 1
        if not Path(trained_weights).exists():
            print(f"Trained model not found: {trained_weights}")
            return 1

    out_paths = []
    print("Running PRETRAINED model:", args.pretrained)
    save_dir = run_video(args.pretrained, str(video_path), str(out_dir), "pretrained")
    out_paths.append(("pretrained", save_dir))

    if not args.pretrained_only:
        print("Running YOUR TRAINED model:", trained_weights)
        save_dir = run_video(trained_weights, str(video_path), str(out_dir), "trained")
        out_paths.append(("trained", save_dir))

    print("\nDone. Output videos are in:")
    for label, path in out_paths:
        print("  -", Path(path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
