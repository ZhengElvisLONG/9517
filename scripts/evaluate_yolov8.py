from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# 自动添加项目根目录到 Python 路径，确保可以导入 src 模块
# 获取脚本所在目录的父目录（项目根目录）
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.datasets.agropest import AgroPestDataset
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 detector on AgroPest-12.")
    parser.add_argument("--weights", type=str, required=True, help="Path or alias to trained YOLOv8 weights.")
    parser.add_argument("--data", type=str, default="agropest.yaml", help="Path to dataset YAML description.")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory containing train/valid/test folders.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "valid"], help="Dataset split to evaluate.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching predictions to ground truth.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation.")
    parser.add_argument("--device", type=str, default="", help="Device string recognised by Ultralytics.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save per-image prediction/GT summary.")
    return parser.parse_args()


def evaluate_classification(
    model: YOLO,
    dataset: AgroPestDataset,
    conf: float,
    iou_threshold: float,
    imgsz: int,
) -> ClassificationMetrics:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    for image_tensor, target in tqdm(dataset, desc="Evaluating classification", unit="image"):
        image_path = target["image_path"]

        results = model.predict(
            source=image_path,
            conf=conf,
            verbose=False,
            imgsz=imgsz,
        )
        result = results[0]
        boxes = result.boxes

        pred_boxes = boxes.xyxy.cpu() if boxes is not None else torch.zeros((0, 4))
        pred_labels = boxes.cls.to(torch.int64).cpu().tolist() if boxes is not None else []
        pred_scores = boxes.conf.cpu().tolist() if boxes is not None else []

        gt_boxes = target["boxes"]
        matches = greedy_match_iou(gt_boxes, pred_boxes, iou_threshold=iou_threshold)

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

    if not y_true:
        raise RuntimeError("No matched predictions found to compute classification metrics.")

    return compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, average="macro")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data_root = data_cfg.get("path", args.data_root)
    data_root = Path(data_root)
    if not data_root.is_absolute():
        data_root = (data_path.parent / data_root).resolve()
    split_arg = args.split.lower()
    if split_arg == "valid":
        yolo_split = "val"
        dataset_split = "valid"
    else:
        yolo_split = split_arg
        dataset_split = "valid" if split_arg == "val" else split_arg

    dataset = AgroPestDataset(data_root=data_root, split=dataset_split, transforms=None, preload_images=False)

    weights_path = Path(args.weights)
    weight_source = str(weights_path) if weights_path.exists() else args.weights
    model = YOLO(weight_source)
    detection_metrics = model.val(
        data=str(data_path),
        split=yolo_split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device or None,
        save_json=bool(args.save_json),
    )

    detection_summary: Dict[str, Any] = {}
    if hasattr(detection_metrics, "results_dict"):
        detection_summary = detection_metrics.results_dict
    elif hasattr(detection_metrics, "metrics"):
        detection_summary = getattr(detection_metrics, "metrics", {})

    classification_metrics = evaluate_classification(
        model,
        dataset,
        conf=args.conf,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz,
    )

    payload = {
        "split": dataset_split,
        "conf": args.conf,
        "iou_threshold": args.iou_threshold,
        "detection": detection_summary,
        "classification": classification_metrics.to_dict(),
    }

    print(json.dumps(payload, indent=2))

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()


