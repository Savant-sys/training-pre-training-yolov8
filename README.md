# YOLO Training Project (Pretrained → Custom)

Train a YOLO object detection model on your own dataset (e.g. BDD100K), then compare pretrained vs trained on video. Supports continuing training on a second dataset from your best weights.

---

## What This Project Does

1. **Setup dataset** – Convert BDD100K-style label JSON to YOLO `.txt` format, then arrange images into `bdd100k_yolo/`.
2. **Setup YOLO model** – Download a pretrained YOLO model (e.g. YOLOv8n) and back it up.
3. **Train** – Fine-tune the model on your dataset (e.g. BDD100K driving classes).
4. **Compare** – Run a video through both the pretrained and your trained model and save two MP4s side by side.
5. **Continue training** – Use your `best.pt` as the starting point for a second dataset (no restart from scratch).

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

### Step 1: Open the project folder

```bash
cd path\to\training-pre-training-yolov8
```

---

### Step 2: Setup dataset (convert JSON → YOLO .txt, then arrange images)

Your raw BDD100K data has **images** and **label JSON** (one big JSON or per-image JSONs). YOLO needs **one `.txt` per image** with lines like `class_id x_center y_center width height` (normalized 0–1). This step converts the labels and sets up the folder structure.

**2a. Convert label JSON to YOLO .txt**

From the **project root**, run:

```bash
python setup_dataset/convert_labels_to_yolo.py
```

- The script looks for BDD100K under the folder in the `BDD_ROOT` environment variable, or `downloads/` if not set.
- Expected layout:
  - **Images:** `bdd100k_images_100k/100k/train/` and `.../val/`
  - **Labels (one of):**
    - Per-image JSON: `bdd100k_labels/100k/train/` and `.../val/` (e.g. `0000f77c-6257be58.json`)
    - Or one big JSON: `bdd100k_labels/det_20/bdd100k_labels_images_train.json` and `..._val.json`
- It writes **YOLO-format .txt** files to `bdd100k_yolo/labels/train/` and `bdd100k_yolo/labels/val/` (one `.txt` per image, same stem as the image filename).
- Classes are mapped to the 10 BDD100K detection classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign.

If your data is elsewhere, set the root before running:

```bash
set BDD_ROOT=C:\path\to\your\bdd100k_folder
python setup_dataset/convert_labels_to_yolo.py
```

**2b. Copy/link images into the YOLO dataset folder**

```bash
python setup_dataset/setup_dataset.py
```

- This fills `bdd100k_yolo/images/train/` and `bdd100k_yolo/images/val/` from your BDD100K images (copy on Windows; can use symlinks on Linux/Mac).
- Uses the same `BDD_ROOT` (or `downloads/`). Image filenames must match the label stems from step 2a.

After 2a and 2b you have:

```
bdd100k_yolo/
  images/train/   images/val/
  labels/train/   labels/val/   (from convert_labels_to_yolo.py)
  data.yaml       (already in repo)
```

**Using a different dataset (already in YOLO format):** If you have images and `.txt` labels in YOLO format elsewhere, create a `data.yaml` (path, train/val dirs, `nc`, `names`) and skip steps 2a–2b. Then continue from Step 3.

---

### Step 3: Setup YOLO model (download pretrained weights)

Download a pretrained model to the project root and back it up to `backup/`.

```bash
python setup_yolo_model.py
```

You should see:
- `yolov8n.pt` in the project root
- A copy in `backup/yolov8n.pt`

To use another model (e.g. YOLO26n), use `model=yolo26n.pt` when training or comparing; Ultralytics will download it on first use.

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
  setup_yolo_model.py       # Step 3: download pretrained model + backup
  setup_dataset/
    convert_labels_to_yolo.py   # Step 2a: JSON → YOLO .txt labels
    setup_dataset.py            # Step 2b: copy/link images into bdd100k_yolo
  train_bdd100k.bat         # Windows: run training (tuned for RAM/GPU)
  train_bdd100k.sh          # Bash: same training
  compare_models_video.py   # Compare pretrained vs trained on video → MP4
  bdd100k_yolo/
    data.yaml               # Dataset config (path, classes)
    images/train, val       # from setup_dataset.py
    labels/train, val       # from convert_labels_to_yolo.py
  runs/detect/train/        # Training outputs
    weights/best.pt         # Best weights (use for inference or next training)
    weights/last.pt
  comparison_output/        # compare_models_video.py output
    pretrained/             # Video from pretrained model
    trained/                # Video from your best.pt
  backup/                   # Backup of pretrained model (from setup_yolo_model.py)
  .gitignore                # Ignores large files (models, data, videos)
```

---

## Tips

- **RAM full, GPU underused:** Lower `workers` (e.g. 4 or 2) and keep `batch` as high as your GPU allows.
- **Training from scratch:** Use the architecture YAML instead of `.pt`, e.g. `model=yolov8n.yaml`. Needs more epochs and data; use only if you have a good reason.
- **Different pretrained model:** e.g. `yolo26n.pt` – use it in `model=...` and in `compare_models_video.py` via `--pretrained yolo26n.pt`.
- **Large files (models, datasets, videos):** The repo’s `.gitignore` is set up so you can push only code to GitHub and keep big files local.

---

## Quick reference (order to run)

| Step | Goal                    | Command / action |
|------|-------------------------|-------------------|
| 2a   | Convert JSON → YOLO .txt| `python setup_dataset/convert_labels_to_yolo.py` |
| 2b   | Setup dataset images    | `python setup_dataset/setup_dataset.py` |
| 3    | Download pretrained     | `python setup_yolo_model.py` |
| 4    | Train (first time)      | `train_bdd100k.bat` or `yolo detect train model=yolov8n.pt data=bdd100k_yolo/data.yaml epochs=100 imgsz=640 batch=64 workers=4` |
| 5    | Compare on video        | `python compare_models_video.py your_video.mov` |
| 5    | Compare pretrained only | `python compare_models_video.py your_video.mov --pretrained-only` |
| 6    | Train on 2nd dataset    | `yolo detect train model=runs/detect/train/weights/best.pt data=second_data.yaml project=runs/detect name=train_dataset2 epochs=100 ...` |
