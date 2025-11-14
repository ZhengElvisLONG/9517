"""统一的模型评估接口，支持所有模型类型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from torch.utils.data import DataLoader

from src.datasets.agropest import AgroPestDataset, detection_collate_fn
from src.evaluation.coco import evaluate_coco_map
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.hog_svm import HOGSVMDetector


ModelType = Literal["faster_rcnn", "yolov8", "hog_svm", "efficientdet", "rt_detr", "rtdetr"]


class UnifiedEvaluator:
    """
    统一的模型评估器，支持多种模型类型。
    
    支持的模型：
    - faster_rcnn: Faster R-CNN
    - yolov8: YOLOv8
    - hog_svm: HOG+SVM 基线
    - efficientdet: EfficientDet
    - rtdetr: RT-DETR
    """
    
    def __init__(
        self,
        model_type: ModelType,
        model_path: Path,
        data_root: Path = Path("data"),
        split: str = "test",
        device: str = "cuda",
        batch_size: int = 4,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ):
        """
        初始化评估器。
        
        Args:
            model_type: 模型类型
            model_path: 模型文件路径
            data_root: 数据集根目录
            split: 数据集划分（train/valid/test）
            device: 计算设备
            batch_size: 批次大小
            iou_threshold: IoU 阈值
            conf_threshold: 置信度阈值
        """
        self.model_type = model_type
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.split = split
        self.device = device
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        
        # 创建数据集
        self.dataset = AgroPestDataset(data_root=self.data_root, split=self.split)
        
        # 加载模型
        self.model = self._load_model()
    
    def _load_model(self) -> Any:
        """加载指定类型的模型。"""
        if self.model_type == "faster_rcnn":
            return self._load_faster_rcnn()
        elif self.model_type == "yolov8":
            return self._load_yolov8()
        elif self.model_type == "hog_svm":
            return self._load_hog_svm()
        elif self.model_type == "efficientdet":
            return self._load_efficientdet()
        elif self.model_type in ("rt_detr", "rtdetr"):
            return self._load_rt_detr()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_faster_rcnn(self) -> torch.nn.Module:
        """加载 Faster R-CNN 模型。"""
        from src.models.faster_rcnn import build_faster_rcnn
        
        checkpoint = torch.load(self.model_path, map_location="cpu")
        config = checkpoint.get("config", {})
        
        model = build_faster_rcnn(
            num_classes=self.dataset.num_classes,
            backbone=config.get("backbone", "resnet50"),
            pretrained=False,
            trainable_backbone_layers=config.get("trainable_backbone_layers", 3),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(torch.device(self.device if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model
    
    def _load_yolov8(self) -> Any:
        """加载 YOLOv8 模型。"""
        from ultralytics import YOLO
        
        return YOLO(str(self.model_path))
    
    def _load_hog_svm(self) -> HOGSVMDetector:
        """加载 HOG+SVM 模型。"""
        return HOGSVMDetector.load(self.model_path)
    
    def _load_efficientdet(self) -> torch.nn.Module:
        """加载 EfficientDet 模型。"""
        from src.models.efficientdet import build_efficientdet
        
        checkpoint = torch.load(self.model_path, map_location="cpu")
        config = checkpoint.get("config", {})
        
        model = build_efficientdet(
            num_classes=self.dataset.num_classes + 1,  # +1 for background
            compound_coef=config.get("compound_coef", 0),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(torch.device(self.device if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model
    
    def _load_rt_detr(self) -> Any:
        """加载 RT-DETR 模型。"""
        from ultralytics import RTDETR
        
        return RTDETR(str(self.model_path))
    
    def evaluate(self) -> Dict[str, Any]:
        """
        执行评估，返回统一的评估结果格式。
        
        Returns:
            包含检测和分类指标的字典
        """
        if self.model_type == "faster_rcnn":
            return self._evaluate_faster_rcnn()
        elif self.model_type == "yolov8":
            return self._evaluate_yolov8()
        elif self.model_type == "hog_svm":
            return self._evaluate_hog_svm()
        elif self.model_type == "efficientdet":
            return self._evaluate_efficientdet()
        elif self.model_type in ("rt_detr", "rtdetr"):
            return self._evaluate_rt_detr()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _evaluate_faster_rcnn(self) -> Dict[str, Any]:
        """评估 Faster R-CNN。"""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=detection_collate_fn,
        )
        
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        
        # 检测指标
        detection_metrics = evaluate_coco_map(self.model, dataloader, device=device)
        
        # 分类指标
        y_true, y_pred, y_prob = [], [], []
        self.model.eval()
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                outputs = self.model(images)
                
                for target, output in zip(targets, outputs):
                    gt_boxes = target["boxes"].to(device)
                    pred_boxes = output["boxes"].to(device)
                    matches = greedy_match_iou(gt_boxes, pred_boxes, iou_threshold=self.iou_threshold)
                    
                    for gt_idx, pred_idx in matches:
                        gt_label = int(target["labels"][gt_idx])
                        pred_label = int(output["labels"][pred_idx].item())
                        score_vec = [0.0] * self.dataset.num_classes
                        score_vec[pred_label] = float(output["scores"][pred_idx].item())
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                        y_prob.append(score_vec)
        
        classification_metrics = (
            compute_classification_metrics(y_true, y_pred, y_prob=y_prob, average="macro")
            if y_true
            else ClassificationMetrics(0.0, 0.0, 0.0, 0.0, None)
        )
        
        return {
            "model_type": "faster_rcnn",
            "split": self.split,
            "detection": detection_metrics,
            "classification": classification_metrics.to_dict(),
        }
    
    def _evaluate_yolov8(self) -> Dict[str, Any]:
        """评估 YOLOv8。"""
        import yaml
        from tqdm import tqdm
        import numpy as np
        
        # 检测指标（使用 YOLO 内置评估）
        data_yaml = self.data_root / "data.yaml"
        if not data_yaml.exists():
            data_yaml = Path("agropest.yaml")
        
        yolo_split = "val" if self.split == "valid" else self.split
        detection_results = self.model.val(
            data=str(data_yaml),
            split=yolo_split,
            conf=self.conf_threshold,
            device=self.device or None,
        )
        
        detection_summary = {}
        if hasattr(detection_results, "results_dict"):
            detection_summary = detection_results.results_dict
        elif hasattr(detection_results, "metrics"):
            detection_summary = getattr(detection_results, "metrics", {})
        
        # 分类指标
        y_true, y_pred, y_prob = [], [], []
        for image_tensor, target in tqdm(self.dataset, desc="Evaluating classification"):
            image_path = target["image_path"]
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            
            pred_boxes = boxes.xyxy.cpu() if boxes is not None else torch.zeros((0, 4))
            pred_labels = boxes.cls.to(torch.int64).cpu().tolist() if boxes is not None else []
            pred_scores = boxes.conf.cpu().tolist() if boxes is not None else []
            
            gt_boxes = target["boxes"]
            matches = greedy_match_iou(gt_boxes, pred_boxes, iou_threshold=self.iou_threshold)
            
            for gt_idx, pred_idx in matches:
                gt_label = int(target["labels"][gt_idx])
                predicted_label = int(pred_labels[pred_idx])
                if predicted_label >= self.dataset.num_classes:
                    continue
                score_vec = np.zeros(self.dataset.num_classes, dtype=np.float32)
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
            "model_type": "yolov8",
            "split": self.split,
            "detection": detection_summary,
            "classification": classification_metrics.to_dict(),
        }
    
    def _evaluate_hog_svm(self) -> Dict[str, Any]:
        """评估 HOG+SVM。"""
        import cv2
        import numpy as np
        from tqdm import tqdm
        
        y_true, y_pred, y_prob = [], [], []
        all_detections = []
        all_targets = []
        
        for idx in tqdm(range(len(self.dataset)), desc="Evaluating HOG+SVM"):
            image, target = self.dataset[idx]
            
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
            detections = self.model.detect(img_array, score_threshold=self.conf_threshold)
            
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
                    iou_threshold=self.iou_threshold,
                )
                
                for gt_idx, pred_idx in matches:
                    gt_label = int(gt_labels[gt_idx])
                    pred_label = int(pred_labels[pred_idx])
                    
                    score_vec = [0.0] * self.dataset.num_classes
                    if pred_label < self.dataset.num_classes:
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
        
        # 简化的检测指标
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
            "model_type": "hog_svm",
            "split": self.split,
            "detection": detection_metrics,
            "classification": classification_metrics.to_dict(),
        }
    
    def _evaluate_efficientdet(self) -> Dict[str, Any]:
        """评估 EfficientDet。"""
        # EfficientDet 评估需要专门的实现
        # 这里返回基本结构
        return {
            "model_type": "efficientdet",
            "split": self.split,
            "detection": {"mAP": 0.0, "note": "Full evaluation not implemented"},
            "classification": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
        }
    
    def _evaluate_rt_detr(self) -> Dict[str, Any]:
        """评估 RT-DETR。"""
        import yaml
        from tqdm import tqdm
        import numpy as np
        
        # 检测指标
        data_yaml = self.data_root / "data.yaml"
        if not data_yaml.exists():
            data_yaml = Path("agropest.yaml")
        
        yolo_split = "val" if self.split == "valid" else self.split
        detection_results = self.model.val(
            data=str(data_yaml),
            split=yolo_split,
            conf=self.conf_threshold,
            device=self.device or None,
        )
        
        detection_summary = {}
        if hasattr(detection_results, "results_dict"):
            detection_summary = detection_results.results_dict
        elif hasattr(detection_results, "metrics"):
            detection_summary = getattr(detection_results, "metrics", {})
        
        # 分类指标
        y_true, y_pred, y_prob = [], [], []
        for image_tensor, target in tqdm(self.dataset, desc="Evaluating classification"):
            image_path = target["image_path"]
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            
            pred_boxes = boxes.xyxy.cpu() if boxes is not None else torch.zeros((0, 4))
            pred_labels = boxes.cls.to(torch.int64).cpu().tolist() if boxes is not None else []
            pred_scores = boxes.conf.cpu().tolist() if boxes is not None else []
            
            gt_boxes = target["boxes"]
            matches = greedy_match_iou(gt_boxes, pred_boxes, iou_threshold=self.iou_threshold)
            
            for gt_idx, pred_idx in matches:
                gt_label = int(target["labels"][gt_idx])
                predicted_label = int(pred_labels[pred_idx])
                if predicted_label >= self.dataset.num_classes:
                    continue
                score_vec = np.zeros(self.dataset.num_classes, dtype=np.float32)
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
            "model_type": "rt_detr",
            "split": self.split,
            "detection": detection_summary,
            "classification": classification_metrics.to_dict(),
        }

