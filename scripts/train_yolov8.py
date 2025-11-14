from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on AgroPest-12.")
    parser.add_argument("--data", type=str, default="agropest.yaml", help="Path to YOLO data YAML.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained checkpoint or model config.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--project", type=str, default="experiments/yolov8", help="Directory to save runs.")
    parser.add_argument("--name", type=str, default="agropest", help="Name of this training run.")
    parser.add_argument("--device", type=str, default="", help="CUDA device string ('' for auto).")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwrite of existing run directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-format predictions during validation.")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16) training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    train_args: Dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "device": args.device or None,
        "patience": args.patience,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "save_json": args.save_json,
        "half": args.half,
    }

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model.train(**train_args)


if __name__ == "__main__":
    main()


