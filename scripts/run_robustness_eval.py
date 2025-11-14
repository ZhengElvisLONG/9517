"""运行鲁棒性评估：在失真测试集上评估模型性能。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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
from src.models.faster_rcnn import build_faster_rcnn
from ultralytics import YOLO


def evaluate_faster_rcnn_on_corrupted(
    checkpoint_path: Path,
    corrupted_images_dir: Path,
    labels_dir: Path,
    data_root: Path,
    device: torch.device,
    batch_size: int = 4,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    在失真测试集上评估 Faster R-CNN。
    
    Args:
        checkpoint_path: Faster R-CNN 检查点路径
        corrupted_images_dir: 失真图像目录
        labels_dir: 标签目录（使用原始测试集标签）
        data_root: 数据集根目录
        device: 计算设备
        batch_size: 批次大小
        iou_threshold: IoU 阈值
    
    Returns:
        包含检测和分类指标的字典
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    
    # 创建临时数据集（使用失真图像目录）
    # 我们需要创建一个自定义数据集，因为标准数据集期望特定的目录结构
    from src.datasets.agropest import AgroPestDataset
    
    # 创建一个临时数据集，使用失真图像目录
    # 由于 AgroPestDataset 期望 data_root/split/images 结构，我们需要临时修改
    # 或者创建一个自定义评估函数
    
    # 简化方案：直接使用原始测试集结构，但替换图像目录
    # 这里我们需要一个更灵活的方法
    
    # 临时方案：复制标签到临时目录，然后使用标准数据集
    import shutil
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_data_root = Path(tmpdir)
        tmp_test_dir = tmp_data_root / "test"
        tmp_test_dir.mkdir(parents=True)
        
        # 创建符号链接或复制图像和标签
        tmp_images_dir = tmp_test_dir / "images"
        tmp_labels_dir = tmp_test_dir / "labels"
        tmp_images_dir.mkdir()
        tmp_labels_dir.mkdir()
        
        # 复制失真图像
        for img_file in corrupted_images_dir.glob("*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img_file, tmp_images_dir / img_file.name)
        
        # 复制标签（从原始测试集）
        for label_file in labels_dir.glob("*.txt"):
            shutil.copy2(label_file, tmp_labels_dir / label_file.name)
        
        # 创建数据集
        dataset = AgroPestDataset(data_root=tmp_data_root, split="test")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=detection_collate_fn,
        )
        
        # 加载模型
        model = build_faster_rcnn(
            num_classes=dataset.num_classes,
            backbone=cfg.get("backbone", "resnet50"),
            pretrained=False,
            trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        
        # 评估检测性能
        detection_metrics = evaluate_coco_map(model, dataloader, device=device)
        
        # 评估分类性能
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                for target, output in zip(targets, outputs):
                    matches = greedy_match_iou(
                        target["boxes"].to(device),
                        output["boxes"].to(device),
                        iou_threshold=iou_threshold,
                    )
                    for gt_idx, pred_idx in matches:
                        gt_label = int(target["labels"][gt_idx])
                        pred_label = int(output["labels"][pred_idx].item())
                        score_vector = [0.0] * dataset.num_classes
                        score_vector[pred_label] = float(output["scores"][pred_idx].item())
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                        y_prob.append(score_vector)
        
        classification_metrics = (
            compute_classification_metrics(y_true, y_pred, y_prob=y_prob, average="macro")
            if y_true
            else ClassificationMetrics(0.0, 0.0, 0.0, 0.0, None)
        )
    
    return {
        "detection": detection_metrics,
        "classification": classification_metrics.to_dict(),
    }


def evaluate_yolov8_on_corrupted(
    weights_path: Path,
    corrupted_images_dir: Path,
    labels_dir: Path,
    data_root: Path,
    device: str,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
) -> Dict[str, float]:
    """
    在失真测试集上评估 YOLOv8。
    
    Args:
        weights_path: YOLOv8 权重路径
        corrupted_images_dir: 失真图像目录
        labels_dir: 标签目录
        data_root: 数据集根目录
        device: 设备字符串
        conf: 置信度阈值
        iou_threshold: IoU 阈值
        imgsz: 图像大小
    
    Returns:
        包含检测和分类指标的字典
    """
    import shutil
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_data_root = Path(tmpdir)
        tmp_test_dir = tmp_data_root / "test"
        tmp_test_dir.mkdir(parents=True)
        
        tmp_images_dir = tmp_test_dir / "images"
        tmp_labels_dir = tmp_test_dir / "labels"
        tmp_images_dir.mkdir()
        tmp_labels_dir.mkdir()
        
        # 复制失真图像和标签
        for img_file in corrupted_images_dir.glob("*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img_file, tmp_images_dir / img_file.name)
        
        for label_file in labels_dir.glob("*.txt"):
            shutil.copy2(label_file, tmp_labels_dir / label_file.name)
        
        # 创建数据集
        dataset = AgroPestDataset(data_root=tmp_data_root, split="test")
        
        # 加载模型
        model = YOLO(str(weights_path))
        
        # 评估检测性能（使用 YOLO 内置评估）
        # 需要创建临时 YAML 文件
        import yaml
        tmp_yaml = tmp_data_root / "data.yaml"
        with tmp_yaml.open("w") as f:
            yaml.dump(
                {
                    "path": str(tmp_data_root),
                    "train": "train/images",
                    "val": "test/images",
                    "test": "test/images",
                    "nc": dataset.num_classes,
                    "names": dataset.class_names,
                },
                f,
            )
        
        detection_results = model.val(
            data=str(tmp_yaml),
            split="test",
            imgsz=imgsz,
            conf=conf,
            device=device or None,
        )
        
        detection_summary = {}
        if hasattr(detection_results, "results_dict"):
            detection_summary = detection_results.results_dict
        elif hasattr(detection_results, "metrics"):
            detection_summary = getattr(detection_results, "metrics", {})
        
        # 评估分类性能
        y_true, y_pred, y_prob = [], [], []
        from tqdm import tqdm
        import numpy as np
        
        for image_tensor, target in tqdm(dataset, desc="Evaluating classification"):
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
        
        classification_metrics = (
            compute_classification_metrics(y_true, y_pred, y_prob=y_prob, average="macro")
            if y_true
            else ClassificationMetrics(0.0, 0.0, 0.0, 0.0, None)
        )
    
    return {
        "detection": detection_summary,
        "classification": classification_metrics.to_dict(),
    }


def run_robustness_evaluation(
    corrupted_testset_dir: Path,
    baseline_results: Dict,
    frcnn_checkpoint: Path | None = None,
    yolov8_weights: Path | None = None,
    data_root: Path = Path("data"),
    labels_dir: Path | None = None,
    output_dir: Path = Path("experiments/robustness"),
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict[str, Dict]:
    """
    运行完整的鲁棒性评估。
    
    Args:
        corrupted_testset_dir: 失真测试集根目录
        baseline_results: 原始测试集的基线结果（用于计算下降值）
        frcnn_checkpoint: Faster R-CNN 检查点路径
        yolov8_weights: YOLOv8 权重路径
        data_root: 数据集根目录
        labels_dir: 标签目录（如果为 None，使用 data_root/test/labels）
        output_dir: 输出目录
        device: 设备字符串
        batch_size: 批次大小
    
    Returns:
        包含所有评估结果的字典
    """
    if labels_dir is None:
        labels_dir = data_root / "test" / "labels"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有失真类型和级别
    corruption_dirs = [
        d for d in corrupted_testset_dir.iterdir() if d.is_dir() and "_level" in d.name
    ]
    
    results = {}
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    
    for corruption_dir in corruption_dirs:
        corruption_name = corruption_dir.name
        print(f"\nEvaluating {corruption_name}...")
        
        corruption_type, level_str = corruption_name.rsplit("_level", 1)
        level = int(level_str)
        
        result_entry = {
            "corruption_type": corruption_type,
            "level": level,
            "faster_rcnn": {},
            "yolov8": {},
        }
        
        # 评估 Faster R-CNN
        if frcnn_checkpoint and frcnn_checkpoint.exists():
            print(f"  Evaluating Faster R-CNN...")
            try:
                frcnn_metrics = evaluate_faster_rcnn_on_corrupted(
                    checkpoint_path=frcnn_checkpoint,
                    corrupted_images_dir=corruption_dir,
                    labels_dir=labels_dir,
                    data_root=data_root,
                    device=device_torch,
                    batch_size=batch_size,
                )
                result_entry["faster_rcnn"] = frcnn_metrics
                
                # 计算下降值
                # 基线结果可能是直接的结果字典，或者包含在 "faster_rcnn" 键下
                if "faster_rcnn" in baseline_results:
                    baseline_map = baseline_results["faster_rcnn"].get("detection", {}).get("mAP", 0.0)
                elif "detection" in baseline_results:
                    baseline_map = baseline_results["detection"].get("mAP", 0.0)
                else:
                    baseline_map = 0.0
                
                current_map = frcnn_metrics.get("detection", {}).get("mAP", 0.0)
                result_entry["faster_rcnn"]["delta_map"] = baseline_map - current_map
            except Exception as e:
                print(f"  Error evaluating Faster R-CNN: {e}")
        
        # 评估 YOLOv8
        if yolov8_weights and yolov8_weights.exists():
            print(f"  Evaluating YOLOv8...")
            try:
                yolo_metrics = evaluate_yolov8_on_corrupted(
                    weights_path=yolov8_weights,
                    corrupted_images_dir=corruption_dir,
                    labels_dir=labels_dir,
                    data_root=data_root,
                    device=device,
                )
                result_entry["yolov8"] = yolo_metrics
                
                # 计算下降值
                # 基线结果可能是直接的结果字典，或者包含在 "yolov8" 键下
                if "yolov8" in baseline_results:
                    baseline_map = baseline_results["yolov8"].get("detection", {}).get("mAP@0.5", 0.0)
                elif "detection" in baseline_results:
                    # YOLO 结果可能使用不同的键名
                    baseline_map = baseline_results["detection"].get("mAP@0.5", baseline_results["detection"].get("metrics/mAP50(B)", 0.0))
                else:
                    baseline_map = 0.0
                
                current_map = yolo_metrics.get("detection", {}).get("mAP@0.5", yolo_metrics.get("detection", {}).get("metrics/mAP50(B)", 0.0))
                result_entry["yolov8"]["delta_map"] = baseline_map - current_map
            except Exception as e:
                print(f"  Error evaluating YOLOv8: {e}")
        
        results[corruption_name] = result_entry
    
    # 保存结果
    results_path = output_dir / "robustness_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Robustness evaluation complete. Results saved to {results_path}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness evaluation on corrupted test set.")
    parser.add_argument(
        "--corrupted-testset-dir",
        type=str,
        default="test_corrupted",
        help="Directory containing corrupted test sets.",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        required=True,
        help="Path to baseline results JSON (from original test set evaluation).",
    )
    parser.add_argument(
        "--frcnn-checkpoint",
        type=str,
        default="",
        help="Path to Faster R-CNN checkpoint.",
    )
    parser.add_argument(
        "--yolov8-weights",
        type=str,
        default="",
        help="Path to YOLOv8 weights.",
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
        default="experiments/robustness",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    
    args = parser.parse_args()
    
    # 加载基线结果
    baseline_path = Path(args.baseline_results)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {baseline_path}")
    
    with baseline_path.open("r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    
    frcnn_checkpoint = Path(args.frcnn_checkpoint) if args.frcnn_checkpoint else None
    yolov8_weights = Path(args.yolov8_weights) if args.yolov8_weights else None
    
    run_robustness_evaluation(
        corrupted_testset_dir=Path(args.corrupted_testset_dir),
        baseline_results=baseline_results,
        frcnn_checkpoint=frcnn_checkpoint,
        yolov8_weights=yolov8_weights,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

