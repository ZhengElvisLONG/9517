"""生成失真测试集：对测试集副本应用各种失真。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tqdm import tqdm
from PIL import Image

from src.robustness.corruptions import apply_corruption


def generate_corrupted_testset(
    test_images_dir: Path,
    output_base_dir: Path,
    corruption_types: list[str],
    levels: list[int],
) -> None:
    """
    生成失真测试集。
    
    Args:
        test_images_dir: 原始测试集图像目录
        output_base_dir: 输出基础目录（将在其下创建 <type>_<level>/ 子目录）
        corruption_types: 失真类型列表（["noise", "blur", "brightness", "occlusion", "jpeg"]）
        levels: 强度级别列表（[1, 2] 或 [1, 2, 3]）
    """
    # 获取所有图像文件
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in test_images_dir.iterdir() if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No images found in {test_images_dir}")
    
    print(f"Found {len(image_files)} images in {test_images_dir}")
    
    # 为每种失真类型和每个强度级别创建输出目录并处理图像
    for corruption_type in corruption_types:
        for level in levels:
            # 创建输出目录：test_corrupted/<type>_<level>/
            output_dir = output_base_dir / f"{corruption_type}_level{level}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nProcessing {corruption_type} level {level}...")
            
            # 处理每张图像
            for img_path in tqdm(image_files, desc=f"{corruption_type}_l{level}"):
                try:
                    # 加载原始图像
                    image = Image.open(img_path).convert("RGB")
                    
                    # 应用失真
                    corrupted_image = apply_corruption(image, corruption_type, level)
                    
                    # 保存失真后的图像（保持原始文件名）
                    output_path = output_dir / img_path.name
                    corrupted_image.save(output_path, quality=95)
                    
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
                    continue
    
    print(f"\n✓ Corrupted test set generated in {output_base_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrupted test set for robustness evaluation."
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        default="data/test/images",
        help="Directory containing original test images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_corrupted",
        help="Base output directory for corrupted images.",
    )
    parser.add_argument(
        "--corruption-types",
        type=str,
        nargs="+",
        default=["noise", "blur", "brightness", "occlusion"],
        help="Corruption types to apply.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Corruption intensity levels (1=light, 2=medium, 3=heavy).",
    )
    
    args = parser.parse_args()
    
    test_images_dir = Path(args.test_images_dir)
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
    
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    generate_corrupted_testset(
        test_images_dir=test_images_dir,
        output_base_dir=output_base_dir,
        corruption_types=args.corruption_types,
        levels=args.levels,
    )


if __name__ == "__main__":
    main()

