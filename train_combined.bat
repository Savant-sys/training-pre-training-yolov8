@echo off
cd /d "%~dp0"

echo ========================================
echo Training Combined Detection Model
echo ========================================
echo.

if not exist "datasets\combined_yolo\data.yaml" (
    echo Building combined dataset...
    py "datasets\setup_dataset\build_combined_dataset.py"
    if errorlevel 1 (
        echo ERROR: Failed to build combined dataset
        pause
        exit /b 1
    )
    echo.
)

echo Starting training...
echo Model: yolov8n.pt
echo Dataset: datasets/combined_yolo/data.yaml
echo.

yolo detect train model=yolov8n.pt data=datasets/combined_yolo/data.yaml project=runs/detect name=train_combined epochs=100 imgsz=640 batch=64 workers=4 cache=False amp=True

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo Training completed!
pause
