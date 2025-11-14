"""鲁棒性测试模块：失真注入与评估工具。"""

from src.robustness.corruptions import (
    add_gaussian_noise,
    apply_motion_blur,
    adjust_brightness_contrast,
    add_occlusion,
    apply_jpeg_compression,
)

__all__ = [
    "add_gaussian_noise",
    "apply_motion_blur",
    "adjust_brightness_contrast",
    "add_occlusion",
    "apply_jpeg_compression",
]

