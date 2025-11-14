"""训练 HOG+SVM 基线模型。"""

from __future__ import annotations

import argparse
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
from src.models.hog_svm import HOGSVMDetector


def extract_training_samples(
    dataset: AgroPestDataset,
    window_size: tuple[int, int] = (64, 64),
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    从数据集中提取训练样本。
    
    Args:
        dataset: AgroPest 数据集
        window_size: 窗口大小
    
    Returns:
        (positive_samples, negative_samples) 正负样本列表
    """
    positive_samples = []
    negative_samples = []
    
    print("Extracting training samples from dataset...")
    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        image, target = dataset[idx]
        
        # 转换为 numpy 数组（BGR 格式，用于 OpenCV）
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            # 如果是 PIL Image 或 torch Tensor，转换
            img_array = np.array(image)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # RGB to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        height, width = img_array.shape[:2]
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        
        # 提取正样本（从标注框）
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 > x1 and y2 > y1:
                # 提取目标区域
                roi = img_array[y1:y2, x1:x2]
                # 调整大小
                roi_resized = cv2.resize(roi, window_size)
                positive_samples.append(roi_resized)
        
        # 提取负样本（随机背景区域）
        num_negative = len(boxes) * 2  # 负样本数量是正样本的 2 倍
        for _ in range(num_negative):
            # 随机选择位置，确保不与正样本重叠太多
            x = np.random.randint(0, max(1, width - window_size[0]))
            y = np.random.randint(0, max(1, height - window_size[1]))
            
            # 检查是否与正样本重叠
            overlap = False
            for box in boxes:
                bx1, by1, bx2, by2 = box.astype(int)
                if not (x + window_size[0] < bx1 or x > bx2 or y + window_size[1] < by1 or y > by2):
                    overlap = True
                    break
            
            if not overlap:
                roi = img_array[y : y + window_size[1], x : x + window_size[0]]
                negative_samples.append(roi)
    
    return positive_samples, negative_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HOG+SVM baseline model.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/hog_svm/hog_svm_model.pkl",
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Detection window size (width height).",
    )
    
    args = parser.parse_args()
    
    # 创建数据集
    dataset = AgroPestDataset(data_root=args.data_root, split=args.split)
    
    # 提取训练样本
    positive_samples, negative_samples = extract_training_samples(
        dataset,
        window_size=tuple(args.window_size),
    )
    
    print(f"\nExtracted {len(positive_samples)} positive samples and {len(negative_samples)} negative samples.")
    
    if len(positive_samples) == 0:
        raise ValueError("No positive samples extracted. Check dataset and annotations.")
    
    # 创建检测器
    detector = HOGSVMDetector(window_size=tuple(args.window_size))
    
    # 训练
    detector.train(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        class_names=dataset.class_names,
    )
    
    # 保存模型
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output_path)
    
    print(f"\n✓ Model saved to {output_path}")


if __name__ == "__main__":
    main()

