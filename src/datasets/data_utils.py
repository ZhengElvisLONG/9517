"""数据集工具函数，确保所有模型兼容。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def convert_image_format(
    image: Image.Image | np.ndarray | torch.Tensor,
    target_format: str = "rgb",
) -> np.ndarray:
    """
    转换图像格式，确保所有模型都能使用。
    
    Args:
        image: 输入图像（PIL Image, numpy array, 或 torch Tensor）
        target_format: 目标格式 ("rgb", "bgr", "tensor")
    
    Returns:
        转换后的图像（numpy array 或 torch Tensor）
    """
    # 转换为 numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    elif isinstance(image, torch.Tensor):
        img_array = image.cpu().numpy()
        # 如果是 CHW 格式，转换为 HWC
        if img_array.ndim == 3 and img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        # 归一化到 0-255
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = np.array(image)
    
    # 确保是 uint8 格式
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # 转换为目标格式
    if target_format == "rgb":
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # 检查是否是 BGR
            if img_array[0, 0, 0] > img_array[0, 0, 2]:  # 简单启发式：BGR 通常 B > R
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return img_array
    elif target_format == "bgr":
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # 检查是否是 RGB
            if img_array[0, 0, 0] < img_array[0, 0, 2]:  # 简单启发式：RGB 通常 R > B
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    elif target_format == "tensor":
        # 转换为 torch Tensor (CHW, 归一化到 0-1)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        if img_array.shape[2] == 3:
            # RGB to tensor
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = torch.from_numpy(img_array).float() / 255.0
        return img_tensor
    else:
        return img_array


def normalize_bbox_format(
    boxes: np.ndarray | torch.Tensor,
    format: str = "xyxy",
) -> np.ndarray:
    """
    标准化边界框格式。
    
    Args:
        boxes: 边界框数组
        format: 当前格式 ("xyxy", "xywh", "cxcywh")
    
    Returns:
        标准化的边界框（xyxy 格式，numpy array）
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    boxes = np.array(boxes)
    
    if format == "xyxy":
        return boxes
    elif format == "xywh":
        # (x, y, w, h) -> (x1, y1, x2, y2)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return boxes
    elif format == "cxcywh":
        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
        return boxes
    else:
        raise ValueError(f"Unsupported bbox format: {format}")


def convert_yolo_to_xyxy(
    yolo_box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    将 YOLO 格式的边界框转换为 xyxy 格式。
    
    Args:
        yolo_box: YOLO 格式边界框 (class_id, x_center, y_center, width, height) 归一化
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        xyxy 格式边界框 (x1, y1, x2, y2)
    """
    if len(yolo_box) == 5:
        _, xc, yc, w, h = yolo_box
    else:
        xc, yc, w, h = yolo_box
    
    x1 = (xc - w / 2) * image_width
    y1 = (yc - h / 2) * image_height
    x2 = (xc + w / 2) * image_width
    y2 = (yc + h / 2) * image_height
    
    return (x1, y1, x2, y2)


def convert_xyxy_to_yolo(
    xyxy_box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    将 xyxy 格式的边界框转换为 YOLO 格式。
    
    Args:
        xyxy_box: xyxy 格式边界框 (x1, y1, x2, y2)
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        YOLO 格式边界框 (x_center, y_center, width, height) 归一化
    """
    x1, y1, x2, y2 = xyxy_box
    
    xc = ((x1 + x2) / 2) / image_width
    yc = ((y1 + y2) / 2) / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    
    return (xc, yc, w, h)


def validate_dataset_structure(data_root: Path) -> Dict[str, bool]:
    """
    验证数据集结构是否完整。
    
    Args:
        data_root: 数据集根目录
    
    Returns:
        验证结果字典
    """
    results = {
        "train_images": False,
        "train_labels": False,
        "valid_images": False,
        "valid_labels": False,
        "test_images": False,
        "test_labels": False,
        "data_yaml": False,
    }
    
    # 检查各个划分
    for split in ["train", "valid", "test"]:
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"
        
        results[f"{split}_images"] = images_dir.exists() and len(list(images_dir.glob("*"))) > 0
        results[f"{split}_labels"] = labels_dir.exists() and len(list(labels_dir.glob("*.txt"))) > 0
    
    # 检查 YAML 文件
    yaml_path = data_root / "data.yaml"
    if not yaml_path.exists():
        yaml_path = Path("agropest.yaml")
    results["data_yaml"] = yaml_path.exists()
    
    return results


def get_dataset_statistics(data_root: Path) -> Dict[str, int]:
    """
    获取数据集统计信息。
    
    Args:
        data_root: 数据集根目录
    
    Returns:
        统计信息字典
    """
    stats = {}
    
    for split in ["train", "valid", "test"]:
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"
        
        if images_dir.exists():
            image_count = len(list(images_dir.glob("*")))
            stats[f"{split}_images"] = image_count
        
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob("*.txt")))
            stats[f"{split}_labels"] = label_count
            
            # 统计标注框数量
            total_boxes = 0
            for label_file in labels_dir.glob("*.txt"):
                with label_file.open("r", encoding="utf-8") as f:
                    total_boxes += len([line for line in f if line.strip()])
            stats[f"{split}_boxes"] = total_boxes
    
    return stats

