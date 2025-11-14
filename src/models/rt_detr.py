"""RT-DETR (Real-Time DETR) 模型实现。"""

from __future__ import annotations

from typing import Literal, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("torch is required for RT-DETR")

# 尝试使用 Ultralytics 的 RT-DETR 实现
try:
    from ultralytics import RTDETR
    ULTRALYTICS_RTDETR_AVAILABLE = True
except ImportError:
    ULTRALYTICS_RTDETR_AVAILABLE = False
    print("Warning: Ultralytics RT-DETR not available. Using alternative implementation.")


def build_rt_detr(
    num_classes: int,
    *,
    model_size: Literal["rtdetr-l", "rtdetr-x"] = "rtdetr-l",
    pretrained: bool = True,
) -> nn.Module:
    """
    构建 RT-DETR 模型。
    
    Args:
        num_classes: 类别数量（不包括背景）
        model_size: 模型大小（"rtdetr-l" 或 "rtdetr-x"）
        pretrained: 是否使用预训练权重
    
    Returns:
        RT-DETR 模型
    """
    if ULTRALYTICS_RTDETR_AVAILABLE:
        # 使用 Ultralytics 的 RT-DETR
        # 注意：Ultralytics RT-DETR 使用字符串模型名称
        model_name = f"{model_size}.pt" if pretrained else None
        model = RTDETR(model_name) if model_name else RTDETR()
        
        # 如果类别数不是默认的，需要修改分类头
        # 注意：Ultralytics 模型可能需要特殊处理
        return model
    else:
        # 使用 PyTorch 的 DETR 作为替代
        print("Warning: Using PyTorch DETR as RT-DETR alternative.")
        try:
            from torchvision.models.detection import detr_resnet50
            model = detr_resnet50(
                pretrained=pretrained,
                num_classes=num_classes + 1,  # +1 for background
            )
            return model
        except ImportError:
            raise ImportError(
                "Neither Ultralytics RT-DETR nor PyTorch DETR available. "
                "Please install ultralytics or use torchvision >= 0.12"
            )


class SimplifiedRTDETR(nn.Module):
    """
    简化的 RT-DETR 实现（如果上述方法都不可用）。
    
    这是一个占位符实现，实际应该使用官方实现。
    """
    
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        super().__init__()
        self.num_classes = num_classes
        # 这里应该实现完整的 RT-DETR 架构
        # 包括：backbone, encoder, decoder, prediction heads
        raise NotImplementedError(
            "Simplified RT-DETR not implemented. "
            "Please install ultralytics or use torchvision DETR."
        )

