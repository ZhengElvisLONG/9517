"""训练 RT-DETR 模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ultralytics import RTDETR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-DETR on AgroPest-12.")
    parser.add_argument(
        "--data",
        type=str,
        default="agropest.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rtdetr-l.pt",
        help="RT-DETR model size (rtdetr-l.pt or rtdetr-x.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device string.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="experiments/rt_detr",
        help="Project directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="agropest_rtdetr",
        help="Experiment name.",
    )
    
    args = parser.parse_args()
    
    # 创建模型
    model = RTDETR(args.model)
    
    # 训练
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
    )
    
    print(f"\n✓ Training complete. Results saved to {args.project}/{args.name}")


if __name__ == "__main__":
    main()

