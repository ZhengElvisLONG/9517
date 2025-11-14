"""HOG + SVM 基线模型：传统目标检测方法。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class HOGSVMDetector:
    """
    HOG (Histogram of Oriented Gradients) + SVM 目标检测器。
    
    这是一个传统的目标检测基线方法，使用滑动窗口 + HOG 特征 + SVM 分类器。
    """
    
    def __init__(
        self,
        window_size: Tuple[int, int] = (64, 64),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        """
        初始化 HOG 描述符和 SVM 分类器。
        
        Args:
            window_size: 检测窗口大小 (width, height)
            block_size: HOG 块大小 (width, height)
            block_stride: HOG 块步长 (width, height)
            cell_size: HOG 单元格大小 (width, height)
            nbins: 方向直方图的 bin 数量
        """
        self.window_size = window_size
        self.hog = cv2.HOGDescriptor(
            _winSize=window_size,
            _blockSize=block_size,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=nbins,
        )
        self.svm = SVC(kernel="linear", probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.num_classes = 0
        self.class_names = []
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        从图像中提取 HOG 特征。
        
        Args:
            image: 输入图像 (BGR 格式)
        
        Returns:
            HOG 特征向量
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 调整图像大小以匹配窗口大小
        if gray.shape[:2] != self.window_size[::-1]:  # HOG 使用 (width, height)
            gray = cv2.resize(gray, self.window_size)
        
        # 计算 HOG 特征
        features = self.hog.compute(gray)
        return features.flatten()
    
    def train(
        self,
        positive_samples: List[np.ndarray],
        negative_samples: List[np.ndarray],
        class_names: List[str],
    ) -> None:
        """
        训练 SVM 分类器。
        
        Args:
            positive_samples: 正样本列表（包含目标的图像块）
            negative_samples: 负样本列表（背景图像块）
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # 提取特征
        print("Extracting HOG features from training samples...")
        positive_features = [self.extract_hog_features(img) for img in positive_samples]
        negative_features = [self.extract_hog_features(img) for img in negative_samples]
        
        # 准备训练数据
        X = np.vstack([positive_features, negative_features])
        y = np.hstack([
            np.ones(len(positive_features)),  # 正样本标签为 1
            np.zeros(len(negative_features)),  # 负样本标签为 0
        ])
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练 SVM
        print("Training SVM classifier...")
        self.svm.fit(X_scaled, y)
        self.is_trained = True
        print("Training complete.")
    
    def detect(
        self,
        image: np.ndarray,
        scale_factor: float = 1.05,
        min_neighbors: int = 3,
        window_step: int = 8,
        score_threshold: float = 0.5,
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        在图像中检测目标。
        
        Args:
            image: 输入图像 (BGR 格式)
            scale_factor: 多尺度检测的缩放因子
            min_neighbors: 非极大值抑制的最小邻居数
            window_step: 滑动窗口的步长
            score_threshold: 置信度阈值
        
        Returns:
            检测结果列表，每个元素为 (x, y, w, h, score, class_id)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        detections = []
        height, width = image.shape[:2]
        
        # 多尺度检测
        current_scale = 1.0
        while True:
            # 计算当前尺度的窗口大小
            scaled_window_w = int(self.window_size[0] * current_scale)
            scaled_window_h = int(self.window_size[1] * current_scale)
            
            # 如果窗口大于图像，停止
            if scaled_window_w > width or scaled_window_h > height:
                break
            
            # 滑动窗口检测
            for y in range(0, height - scaled_window_h + 1, window_step):
                for x in range(0, width - scaled_window_w + 1, window_step):
                    # 提取窗口
                    window = image[y : y + scaled_window_h, x : x + scaled_window_w]
                    
                    # 提取特征并预测
                    features = self.extract_hog_features(window)
                    features_scaled = self.scaler.transform([features])
                    
                    # 预测概率
                    prob = self.svm.predict_proba(features_scaled)[0]
                    score = prob[1]  # 正样本的概率
                    
                    if score >= score_threshold:
                        detections.append((
                            x,
                            y,
                            scaled_window_w,
                            scaled_window_h,
                            score,
                            0,  # 单类检测，class_id 为 0
                        ))
            
            # 更新尺度
            current_scale *= scale_factor
        
        # 非极大值抑制
        detections = self._non_max_suppression(detections, min_neighbors)
        
        return detections
    
    def _non_max_suppression(
        self,
        detections: List[Tuple[int, int, int, int, float, int]],
        min_neighbors: int,
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        非极大值抑制。
        
        Args:
            detections: 检测结果列表
            min_neighbors: 最小邻居数
        
        Returns:
            抑制后的检测结果
        """
        if not detections:
            return []
        
        # 按分数排序
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        suppressed = []
        while detections:
            # 取最高分的检测
            current = detections.pop(0)
            suppressed.append(current)
            
            # 移除与当前检测重叠的检测
            x1, y1, w1, h1 = current[0], current[1], current[2], current[3]
            remaining = []
            
            for det in detections:
                x2, y2, w2, h2 = det[0], det[1], det[2], det[3]
                
                # 计算 IoU
                intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
                    0, min(y1 + h1, y2 + h2) - max(y1, y2)
                )
                union = w1 * h1 + w2 * h2 - intersection
                iou = intersection / union if union > 0 else 0
                
                # 如果 IoU 小于阈值，保留
                if iou < 0.3:
                    remaining.append(det)
            
            detections = remaining
        
        return suppressed
    
    def save(self, path: Path) -> None:
        """保存模型。"""
        model_data = {
            "hog_params": {
                "window_size": self.window_size,
                "block_size": self.hog.getBlockSize(),
                "block_stride": self.hog.getBlockStride(),
                "cell_size": self.hog.getCellSize(),
                "nbins": self.hog.getNBins(),
            },
            "svm": self.svm,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
        with path.open("wb") as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Path) -> "HOGSVMDetector":
        """加载模型。"""
        with path.open("rb") as f:
            model_data = pickle.load(f)
        
        hog_params = model_data["hog_params"]
        detector = cls(
            window_size=tuple(hog_params["window_size"]),
            block_size=tuple(hog_params["block_size"]),
            block_stride=tuple(hog_params["block_stride"]),
            cell_size=tuple(hog_params["cell_size"]),
            nbins=hog_params["nbins"],
        )
        
        detector.svm = model_data["svm"]
        detector.scaler = model_data["scaler"]
        detector.is_trained = model_data["is_trained"]
        detector.num_classes = model_data["num_classes"]
        detector.class_names = model_data["class_names"]
        
        return detector

