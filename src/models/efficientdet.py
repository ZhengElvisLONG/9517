"""EfficientDet 模型实现。"""

from __future__ import annotations

from typing import Literal, Optional

try:
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
except ImportError:
    raise ImportError("torch and torchvision are required for EfficientDet")

# 尝试导入 efficientdet-pytorch，如果没有则使用简化实现
try:
    from efficientdet import EfficientDet
    EFFICIENTDET_AVAILABLE = True
except ImportError:
    EFFICIENTDET_AVAILABLE = False
    print("Warning: efficientdet-pytorch not installed. Using simplified implementation.")


def build_efficientdet(
    num_classes: int,
    *,
    compound_coef: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0,
    pretrained: bool = True,
) -> nn.Module:
    """
    构建 EfficientDet 模型。
    
    Args:
        num_classes: 类别数量（包括背景）
        compound_coef: 复合系数（0-7，对应 D0-D7）
        pretrained: 是否使用预训练权重
    
    Returns:
        EfficientDet 模型
    """
    if EFFICIENTDET_AVAILABLE:
        # 使用 efficientdet-pytorch 库
        model = EfficientDet(
            num_classes=num_classes,
            compound_coef=compound_coef,
            onnx_export=False,
            coefficients=None,
        )
        if pretrained:
            # 加载预训练权重（如果有）
            try:
                from efficientdet import get_efficientdet_config
                config = get_efficientdet_config(f"efficientdet-d{compound_coef}")
                # 这里可以加载预训练权重
                pass
            except Exception:
                print("Warning: Could not load pretrained weights.")
    else:
        # 简化实现：使用 EfficientNet 作为骨干网络 + 简单的检测头
        print("Using simplified EfficientDet implementation with EfficientNet backbone.")
        
        # 选择 EfficientNet 变体
        efficientnet_models = {
            0: efficientnet_b0,
            1: efficientnet_b1,
            2: efficientnet_b2,
            3: efficientnet_b3,
        }
        
        if compound_coef not in efficientnet_models:
            raise ValueError(f"compound_coef {compound_coef} not supported in simplified implementation. Use 0-3.")
        
        backbone_fn = efficientnet_models[compound_coef]
        backbone = backbone_fn(weights="DEFAULT" if pretrained else None)
        
        # 创建简化的检测模型
        model = SimplifiedEfficientDet(backbone, num_classes=num_classes)
    
    return model


class SimplifiedEfficientDet(nn.Module):
    """
    简化的 EfficientDet 实现。
    
    使用 EfficientNet 作为骨干网络，添加简单的检测头。
    """
    
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # 获取骨干网络的输出特征维度
        # EfficientNet 的最后一层是分类层，我们需要特征提取部分
        backbone_features = nn.Sequential(*list(backbone.features.children()))
        self.features = backbone_features
        
        # 简单的检测头（这里使用简化的实现）
        # 实际 EfficientDet 使用 BiFPN，这里简化处理
        feature_dim = backbone.classifier[1].in_features
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        
        # 回归头（边界框）
        self.reg_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # x, y, w, h
        )
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x: 输入图像张量 (B, C, H, W)
        
        Returns:
            检测结果字典
        """
        # 提取特征
        features = self.features(x)
        
        # 全局平均池化
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        # 分类和回归
        cls_logits = self.cls_head(features)
        bbox_preds = self.reg_head(features)
        
        # 转换为标准格式
        # 注意：这是一个简化实现，实际 EfficientDet 使用更复杂的架构
        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
        }

