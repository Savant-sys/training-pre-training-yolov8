# YOLO Training Project (Pretrained → Custom)

Train a YOLO object detection model on your own dataset (e.g. BDD100K), then compare pretrained vs trained on video. Supports continuing training on a second dataset from your best weights.

---

## What This Project Does

1. **Setup** – Download a pretrained YOLO model (e.g. YOLOv8n) and back it up.
2. **Train** – Fine-tune the model on your dataset (e.g. BDD100K driving classes).
3. **Compare** – Run a video through both the pretrained and your trained model and save two MP4s side by side.
4. **Continue training** – Use your `best.pt` as the starting point for a second dataset (no restart from scratch).

---

## Prerequisites

- **Python 3.8+** with `pip`
- **GPU** (recommended; CPU is very slow)
- **Enough RAM** – 16GB+ recommended; 48GB+ for large datasets like BDD100K with default-style training

### Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

(`ultralytics` includes PyTorch; if you need a specific CUDA version, install PyTorch first from [pytorch.org](https://pytorch.org), then `pip install ultralytics`.)

---

## Step-by-Step

### Step 1: Get the project and go to the project folder

```bash
cd path\to\training-pre-training-yolov8
```

---

### Step 2: Download the pretrained model (first time only)

This downloads a pretrained model to the project root and backs it up to `backup/`.

**Option A – Script (downloads YOLOv8n):**

```bash
python setup_yolo_model.py
```

You should see:
- `yolov8n.pt` in the project root
- A copy in `backup/yolov8n.pt`

**Option B – Use another model (e.g. YOLO26n):**

If you use a model name like `yolo26n.pt` in training or comparison, Ultralytics will download it on first use. No extra step needed.

---

### Step 3: Prepare your dataset in YOLO format

Your data must be in YOLO format:

- **Images** in folders (e.g. `bdd100k_yolo/images/train/`, `bdd100k_yolo/images/val/`).
- **Labels** – one `.txt` per image, same name as the image, in `labels/train/` and `labels/val/`. Each line: `class_id x_center y_center width height` (normalized 0–1).

Example layout:

```
bdd100k_yolo/
  data.yaml          # dataset config (path, train/val, nc, names)
  images/
    train/           # training images
    val/             # validation images
  labels/
    train/           # .txt per image
    val/
```

The project includes `bdd100k_yolo/data.yaml` for the BDD100K driving dataset (10 classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign). Replace paths and class names in `data.yaml` if you use a different dataset.

---

### Step 4: Train the model

From the **project root**:

**Windows (batch file):**

```bash
train_bdd100k.bat
```

**Or run the command directly (Windows / Linux / Mac):**

```bash
yolo detect train model=yolov8n.pt data=bdd100k_yolo/data.yaml epochs=100 imgsz=640 batch=64 workers=4 cache=False amp=True
```

- **model** – Pretrained weights (e.g. `yolov8n.pt` or `yolo26n.pt`).
- **data** – Your dataset YAML (e.g. `bdd100k_yolo/data.yaml`).
- **epochs** – e.g. 100 for first training.
- **imgsz** – Input size, typically 640.
- **batch** – Increase if you have enough GPU memory (e.g. 64 on a strong GPU).
- **workers** – Lower (e.g. 4) if you run out of RAM.

Training saves to:

- **Best weights:** `runs/detect/train/weights/best.pt`
- **Last epoch:** `runs/detect/train/weights/last.pt`
- Logs and plots in `runs/detect/train/`

Training can take hours (e.g. 100 epochs on BDD100K). If RAM hits 100% and GPU is low, reduce `workers` (e.g. to 2 or 4).

---

### Step 5: Compare pretrained vs trained on a video

Run a video through both the pretrained model and your trained model; two MP4s are saved.

```bash
python compare_models_video.py path\to\your_video.mov
```

- **With trained model:**  
  Uses `yolov8n.pt` (or your default pretrained) and `runs/detect/train/weights/best.pt`.  
  Outputs:
  - `comparison_output/pretrained/your_video.mp4`
  - `comparison_output/trained/your_video.mp4`

- **Pretrained only** (if you don’t have `best.pt` yet):

  ```bash
  python compare_models_video.py path\to\your_video.mov --pretrained-only
  ```

- **Custom paths:**

  ```bash
  python compare_models_video.py video.mov --pretrained yolov8n.pt --trained runs/detect/train/weights/best.pt --out-dir my_comparison
  ```

---

### Step 6 (Optional): Train on a second dataset from your current best

You can keep improving from your current weights instead of starting over:

1. Copy your current best (optional but recommended):

   ```bash
   copy runs\detect\train\weights\best.pt backup\best_after_dataset1.pt
   ```

2. Train on the second dataset, **starting from** `best.pt`:

   ```bash
   yolo detect train model=runs/detect/train/weights/best.pt data=path/to/second_dataset.yaml project=runs/detect name=train_dataset2 epochs=100 imgsz=640 batch=64 workers=4
   ```

   New run is saved under `runs/detect/train_dataset2/`. Training continues from your current weights; it does not restart from the original pretrained model.

---

## Project layout (main items)

```
training-pre-training-yolov8/
  README.md                 # This file
  setup_yolo_model.py       # Download pretrained model + backup
  train_bdd100k.bat         # Windows: run training (tuned for RAM/GPU)
  train_bdd100k.sh          # Bash: same training
  compare_models_video.py   # Compare pretrained vs trained on video → MP4
  bdd100k_yolo/
    data.yaml               # Dataset config (path, classes)
    images/train, val
    labels/train, val
  runs/detect/train/        # Training outputs
    weights/best.pt         # Best weights (use for inference or next training)
    weights/last.pt
  comparison_output/        # compare_models_video.py output
    pretrained/             # Video from pretrained model
    trained/                # Video from your best.pt
  backup/                   # Backup of pretrained model (from setup)
  .gitignore                # Ignores large files (models, data, videos)
```

---

## Tips

- **RAM full, GPU underused:** Lower `workers` (e.g. 4 or 2) and keep `batch` as high as your GPU allows.
- **Training from scratch:** Use the architecture YAML instead of `.pt`, e.g. `model=yolov8n.yaml`. Needs more epochs and data; use only if you have a good reason.
- **Different pretrained model:** e.g. `yolo26n.pt` – use it in `model=...` and in `compare_models_video.py` via `--pretrained yolo26n.pt`.
- **Large files (models, datasets, videos):** The repo’s `.gitignore` is set up so you can push only code to GitHub and keep big files local.

---

## Quick reference

| Goal                    | Command / action |
|-------------------------|-------------------|
| Download pretrained     | `python setup_yolo_model.py` |
| Train (first time)      | `train_bdd100k.bat` or `yolo detect train model=yolov8n.pt data=bdd100k_yolo/data.yaml epochs=100 imgsz=640 batch=64 workers=4` |
| Compare on video        | `python compare_models_video.py your_video.mov` |
| Compare pretrained only | `python compare_models_video.py your_video.mov --pretrained-only` |
| Train on 2nd dataset    | `yolo detect train model=runs/detect/train/weights/best.pt data=second_data.yaml project=runs/detect name=train_dataset2 epochs=100 ...` |
