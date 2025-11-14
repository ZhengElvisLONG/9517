"""图像失真注入函数：用于鲁棒性测试。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def add_gaussian_noise(image: Image.Image, noise_level: float) -> Image.Image:
    """
    添加高斯噪声到图像。
    
    Args:
        image: PIL Image 对象
        noise_level: 噪声标准差（通常 0.05, 0.10, 0.20）
    
    Returns:
        添加噪声后的 PIL Image
    """
    # 将 PIL 图像转换为 numpy 数组（RGB 格式，值范围 0-255）
    img_array = np.array(image, dtype=np.float32)
    
    # 生成高斯噪声（均值为 0，标准差为 noise_level * 255）
    noise = np.random.normal(0, noise_level * 255, img_array.shape).astype(np.float32)
    
    # 添加噪声并裁剪到有效范围
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # 转换回 PIL Image
    return Image.fromarray(noisy_array)


def apply_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    """
    应用运动模糊。
    
    Args:
        image: PIL Image 对象
        kernel_size: 模糊核大小（通常 7 或 15）
    
    Returns:
        模糊后的 PIL Image
    """
    # 将 PIL 图像转换为 OpenCV 格式（BGR）
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 创建运动模糊核（水平方向）
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32)
    kernel = kernel / kernel_size
    
    # 应用模糊
    blurred = cv2.filter2D(img_bgr, -1, kernel)
    
    # 转换回 RGB 并返回 PIL Image
    img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def adjust_brightness_contrast(
    image: Image.Image, brightness_factor: float, contrast_factor: float
) -> Image.Image:
    """
    调整亮度和对比度。
    
    Args:
        image: PIL Image 对象
        brightness_factor: 亮度因子（< 1.0 变暗，> 1.0 变亮，通常 0.6 或 0.4）
        contrast_factor: 对比度因子（< 1.0 降低，> 1.0 提高，通常 0.7 或 0.4）
    
    Returns:
        调整后的 PIL Image
    """
    # 将 PIL 图像转换为 numpy 数组（值范围 0-255）
    img_array = np.array(image, dtype=np.float32)
    
    # 调整亮度：乘以亮度因子
    img_array = img_array * brightness_factor
    
    # 调整对比度：使用公式 (pixel - 128) * contrast + 128
    img_array = (img_array - 128.0) * contrast_factor + 128.0
    
    # 裁剪到有效范围并转换回 uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def add_occlusion(image: Image.Image, occlusion_ratio: float) -> Image.Image:
    """
    添加随机遮挡（黑色方块）。
    
    Args:
        image: PIL Image 对象
        occlusion_ratio: 遮挡面积比例（通常 0.20 或 0.35）
    
    Returns:
        添加遮挡后的 PIL Image
    """
    # 将 PIL 图像转换为 numpy 数组
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # 计算遮挡区域的总面积
    total_area = height * width
    occlusion_area = int(total_area * occlusion_ratio)
    
    # 随机生成遮挡块的数量和大小（1-3 个块）
    num_blocks = np.random.randint(1, 4)
    area_per_block = occlusion_area // num_blocks
    
    # 为每个遮挡块随机生成位置和大小
    for _ in range(num_blocks):
        # 计算块的边长（近似正方形）
        block_size = int(np.sqrt(area_per_block))
        block_h = min(block_size, height // 2)
        block_w = min(block_size, width // 2)
        
        # 随机生成块的位置
        x = np.random.randint(0, max(1, width - block_w))
        y = np.random.randint(0, max(1, height - block_h))
        
        # 添加黑色遮挡（值为 0）
        img_array[y : y + block_h, x : x + block_w] = 0
    
    return Image.fromarray(img_array)


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """
    应用 JPEG 压缩。
    
    Args:
        image: PIL Image 对象
        quality: JPEG 质量（1-100，通常 30 或 10）
    
    Returns:
        压缩后的 PIL Image
    """
    # 将图像保存到内存缓冲区（JPEG 格式）
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    
    # 从缓冲区重新加载图像（这会应用 JPEG 压缩）
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    
    # 确保返回 RGB 模式
    if compressed_image.mode != "RGB":
        compressed_image = compressed_image.convert("RGB")
    
    return compressed_image


def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    level: int,
) -> Image.Image:
    """
    根据类型和强度级别应用失真。
    
    Args:
        image: PIL Image 对象
        corruption_type: 失真类型（"noise", "blur", "brightness", "occlusion", "jpeg"）
        level: 强度级别（1, 2, 3 对应轻度、中度、重度）
    
    Returns:
        失真后的 PIL Image
    """
    if corruption_type == "noise":
        # 噪声级别：0.05, 0.10, 0.20
        noise_levels = [0.05, 0.10, 0.20]
        return add_gaussian_noise(image, noise_levels[level - 1])
    
    elif corruption_type == "blur":
        # 模糊核大小：7, 15
        kernel_sizes = [7, 15]
        return apply_motion_blur(image, kernel_sizes[level - 1])
    
    elif corruption_type == "brightness":
        # 亮度/对比度：(0.6, 0.7), (0.4, 0.4)
        brightness_contrast_pairs = [(0.6, 0.7), (0.4, 0.4)]
        brightness, contrast = brightness_contrast_pairs[level - 1]
        return adjust_brightness_contrast(image, brightness, contrast)
    
    elif corruption_type == "occlusion":
        # 遮挡比例：0.20, 0.35
        occlusion_ratios = [0.20, 0.35]
        return add_occlusion(image, occlusion_ratios[level - 1])
    
    elif corruption_type == "jpeg":
        # JPEG 质量：30, 10
        qualities = [30, 10]
        return apply_jpeg_compression(image, qualities[level - 1])
    
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

