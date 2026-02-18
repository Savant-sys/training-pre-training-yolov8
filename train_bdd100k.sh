#!/usr/bin/env bash
# BDD100K training tuned for 48GB RAM + RTX 5090
# Fix: fewer workers = less RAM; larger batch = more GPU use

yolo detect train \
  model=yolov8n.pt \
  data=bdd100k_yolo/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=64 \
  workers=4 \
  cache=False \
  amp=True
