from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径，确保可以导入 src 模块
# 获取脚本所在目录的父目录（项目根目录）
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.training.faster_rcnn_trainer import FasterRCNNTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on AgroPest-12.")
    parser.add_argument("--config", type=str, default="", help="Optional path to JSON config overriding defaults.")
    parser.add_argument("--output-dir", type=str, default="", help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=0, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size.")
    parser.add_argument("--device", type=str, default="", help="Override compute device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainerConfig()
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = TrainerConfig(**{**cfg.__dict__, **data})

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device

    trainer = FasterRCNNTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()


