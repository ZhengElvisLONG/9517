"""评估 RT-DETR 模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ultralytics import RTDETR

from src.datasets.agropest import AgroPestDataset
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR model.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to RT-DETR weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="agropest.yaml",
        help="Path to dataset YAML file.",
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
        choices=["train", "val", "test", "valid"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device string.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save metrics JSON.",
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model = RTDETR(args.weights)
    
    # 评估检测性能（使用 Ultralytics 内置评估）
    detection_results = model.val(
        data=args.data,
        split=args.split if args.split != "valid" else "val",
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device or None,
    )
    
    detection_summary = {}
    if hasattr(detection_results, "results_dict"):
        detection_summary = detection_results.results_dict
    elif hasattr(detection_results, "metrics"):
        detection_summary = getattr(detection_results, "metrics", {})
    
    # 评估分类性能（需要自定义实现）
    dataset = AgroPestDataset(data_root=args.data_root, split=args.split if args.split != "val" else "valid")
    
    y_true = []
    y_pred = []
    y_prob = []
    
    from tqdm import tqdm
    import numpy as np
    import torch
    
    for image_tensor, target in tqdm(dataset, desc="Evaluating classification"):
        image_path = target["image_path"]
        results = model.predict(
            source=image_path,
            conf=args.conf,
            verbose=False,
            imgsz=args.imgsz,
        )
        result = results[0]
        boxes = result.boxes
        
        pred_boxes = boxes.xyxy.cpu() if boxes is not None else torch.zeros((0, 4))
        pred_labels = boxes.cls.to(torch.int64).cpu().tolist() if boxes is not None else []
        pred_scores = boxes.conf.cpu().tolist() if boxes is not None else []
        
        gt_boxes = target["boxes"]
        matches = greedy_match_iou(gt_boxes, pred_boxes, iou_threshold=args.iou_threshold)
        
        for gt_idx, pred_idx in matches:
            gt_label = int(target["labels"][gt_idx])
            predicted_label = int(pred_labels[pred_idx])
            if predicted_label >= dataset.num_classes:
                continue
            score_vec = np.zeros(dataset.num_classes, dtype=np.float32)
            score_vec[predicted_label] = float(pred_scores[pred_idx])
            
            y_true.append(gt_label)
            y_pred.append(predicted_label)
            y_prob.append(score_vec)
    
    classification_metrics = (
        compute_classification_metrics(y_true, y_pred, y_prob=y_prob, average="macro")
        if y_true
        else ClassificationMetrics(0.0, 0.0, 0.0, 0.0, None)
    )
    
    results = {
        "split": args.split,
        "detection": detection_summary,
        "classification": classification_metrics.to_dict(),
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

