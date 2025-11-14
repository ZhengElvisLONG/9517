"""评估 HOG+SVM 基线模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from tqdm import tqdm

from src.datasets.agropest import AgroPestDataset
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.hog_svm import HOGSVMDetector


def evaluate_hog_svm(
    model_path: Path,
    dataset: AgroPestDataset,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict:
    """
    评估 HOG+SVM 模型。
    
    Args:
        model_path: 模型路径
        dataset: 测试数据集
        iou_threshold: IoU 阈值
        score_threshold: 置信度阈值
    
    Returns:
        评估结果字典
    """
    # 加载模型
    detector = HOGSVMDetector.load(model_path)
    
    y_true = []
    y_pred = []
    y_prob = []
    
    all_detections = []
    all_targets = []
    
    print("Evaluating HOG+SVM model...")
    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        image, target = dataset[idx]
        
        # 转换为 numpy 数组（BGR 格式）
        if isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            img_array = np.array(image)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 检测
        detections = detector.detect(img_array, score_threshold=score_threshold)
        
        # 转换为标准格式
        gt_boxes = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()
        
        pred_boxes = np.array([[d[0], d[1], d[0] + d[2], d[1] + d[3]] for d in detections])
        pred_scores = np.array([d[4] for d in detections])
        pred_labels = np.array([d[5] for d in detections])
        
        # 匹配预测和真值
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            matches = greedy_match_iou(
                torch.from_numpy(gt_boxes),
                torch.from_numpy(pred_boxes),
                iou_threshold=iou_threshold,
            )
            
            for gt_idx, pred_idx in matches:
                gt_label = int(gt_labels[gt_idx])
                pred_label = int(pred_labels[pred_idx])
                
                score_vec = [0.0] * dataset.num_classes
                if pred_label < dataset.num_classes:
                    score_vec[pred_label] = float(pred_scores[pred_idx])
                
                y_true.append(gt_label)
                y_pred.append(pred_label)
                y_prob.append(score_vec)
        
        all_detections.append({
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        })
        all_targets.append({
            "boxes": gt_boxes,
            "labels": gt_labels,
        })
    
    # 计算分类指标
    if y_true:
        classification_metrics = compute_classification_metrics(
            y_true, y_pred, y_prob=y_prob, average="macro"
        )
    else:
        classification_metrics = ClassificationMetrics(0.0, 0.0, 0.0, 0.0, None)
    
    # 简化的检测指标（HOG+SVM 不直接输出 mAP，需要额外计算）
    # 这里返回基本的匹配统计
    num_matched = len(y_true)
    num_gt = sum(len(t["labels"]) for t in all_targets)
    num_pred = sum(len(d["labels"]) for d in all_detections)
    
    detection_metrics = {
        "num_ground_truth": num_gt,
        "num_predictions": num_pred,
        "num_matched": num_matched,
        "precision": num_matched / num_pred if num_pred > 0 else 0.0,
        "recall": num_matched / num_gt if num_gt > 0 else 0.0,
    }
    
    return {
        "split": "test",
        "detection": detection_metrics,
        "classification": classification_metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HOG+SVM model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained HOG+SVM model.",
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
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Detection score threshold.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save metrics JSON.",
    )
    
    args = parser.parse_args()
    
    # 创建数据集
    dataset = AgroPestDataset(data_root=args.data_root, split=args.split)
    
    # 评估
    results = evaluate_hog_svm(
        model_path=Path(args.model),
        dataset=dataset,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    
    # 保存结果
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    # 打印结果
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

