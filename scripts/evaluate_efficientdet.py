"""评估 EfficientDet 模型。"""

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
from src.evaluation.coco import evaluate_coco_map
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.efficientdet import build_efficientdet


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EfficientDet model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to EfficientDet checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for classification metrics.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save metrics JSON.",
    )
    
    args = parser.parse_args()
    
    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    
    # 创建数据集
    dataset = AgroPestDataset(data_root=args.data_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
    )
    
    # 创建模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_efficientdet(
        num_classes=dataset.num_classes + 1,  # +1 for background
        compound_coef=config.get("compound_coef", 0),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    print("Evaluating EfficientDet model...")
    print("Note: This is a simplified evaluation. Full evaluation may require specialized tools.")
    
    # 评估检测性能（简化版）
    # 实际 EfficientDet 评估需要使用专门的评估工具
    print("\n⚠️  Warning: Full EfficientDet evaluation not implemented.")
    print("Please use EfficientDet's native evaluation tools or implement a full evaluator.")
    
    # 返回基本结果
    results = {
        "split": args.split,
        "detection": {"mAP": 0.0, "note": "Full evaluation not implemented"},
        "classification": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
    }
    
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

