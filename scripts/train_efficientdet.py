"""训练 EfficientDet 模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from torch.utils.data import DataLoader

from src.datasets.agropest import AgroPestDataset, detection_collate_fn
from src.models.efficientdet import build_efficientdet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientDet on AgroPest-12.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional path to JSON config overriding defaults.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/efficientdet",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--compound-coef",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
        help="EfficientDet compound coefficient (0-7).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string.",
    )
    
    args = parser.parse_args()
    
    # 加载配置（如果有）
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
    
    # 创建数据集
    train_dataset = AgroPestDataset(data_root=args.data_root, split="train")
    val_dataset = AgroPestDataset(data_root=args.data_root, split="valid")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=4,
    )
    
    # 创建模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_efficientdet(
        num_classes=train_dataset.num_classes + 1,  # +1 for background
        compound_coef=args.compound_coef,
        pretrained=True,
    )
    model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 训练循环（简化版，实际应该使用完整的训练器）
    print("Starting training...")
    print("Note: This is a simplified training script. For full training, use a proper trainer class.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 这里应该实现完整的训练循环
    # 由于 EfficientDet 的训练比较复杂，建议使用专门的训练框架
    print("\n⚠️  Warning: Full EfficientDet training not implemented in this script.")
    print("Please use a dedicated EfficientDet training framework or implement a full trainer.")
    print(f"Model created and saved to {output_dir}")
    
    # 保存模型
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "num_classes": train_dataset.num_classes,
                "compound_coef": args.compound_coef,
            },
        },
        output_dir / "efficientdet_init.pth",
    )


if __name__ == "__main__":
    main()

