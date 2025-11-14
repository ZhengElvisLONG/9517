"""生成统一的可视化报告，支持所有模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.datasets.data_utils import get_dataset_statistics, validate_dataset_structure
from src.visualization.model_comparison import (
    create_comparison_table,
    load_evaluation_results,
    plot_model_comparison,
    plot_radar_chart,
)


def plot_dataset_statistics(data_root: Path, output_path: Path) -> None:
    """绘制数据集统计信息。"""
    stats = get_dataset_statistics(data_root)
    
    # 准备数据
    splits = ["train", "valid", "test"]
    image_counts = [stats.get(f"{s}_images", 0) for s in splits]
    box_counts = [stats.get(f"{s}_boxes", 0) for s in splits]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图像数量
    ax1.bar(splits, image_counts, color=["#4C72B0", "#55A868", "#C44E52"])
    ax1.set_ylabel("Number of Images", fontsize=12)
    ax1.set_title("Dataset Split: Image Counts", fontsize=14, fontweight="bold")
    for i, v in enumerate(image_counts):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    
    # 标注框数量
    ax2.bar(splits, box_counts, color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.set_ylabel("Number of Bounding Boxes", fontsize=12)
    ax2.set_title("Dataset Split: Annotation Counts", fontsize=14, fontweight="bold")
    for i, v in enumerate(box_counts):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(
    results: dict,
    data_root: Path,
    output_path: Path,
) -> None:
    """生成汇总报告。"""
    report_lines = [
        "# Model Evaluation Summary Report\n",
        "## Dataset Information\n",
    ]
    
    # 数据集统计
    stats = get_dataset_statistics(data_root)
    report_lines.append(f"- Training images: {stats.get('train_images', 0)}")
    report_lines.append(f"- Training boxes: {stats.get('train_boxes', 0)}")
    report_lines.append(f"- Validation images: {stats.get('valid_images', 0)}")
    report_lines.append(f"- Validation boxes: {stats.get('valid_boxes', 0)}")
    report_lines.append(f"- Test images: {stats.get('test_images', 0)}")
    report_lines.append(f"- Test boxes: {stats.get('test_boxes', 0)}\n")
    
    report_lines.append("## Model Performance\n")
    report_lines.append("| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Accuracy |")
    report_lines.append("|-------|---------|--------------|-----------|--------|----|----------|")
    
    for model_name, result in results.items():
        detection = result.get("detection", {})
        classification = result.get("classification", {})
        
        # 标准化检测指标
        map50 = detection.get("mAP_50", detection.get("mAP@0.5", detection.get("metrics/mAP50(B)", 0.0)))
        map50_95 = detection.get("mAP@0.5:0.95", detection.get("metrics/mAP50-95(B)", 0.0))
        precision = detection.get("precision", detection.get("metrics/precision(B)", 0.0))
        recall = detection.get("recall", detection.get("metrics/recall(B)", 0.0))
        
        f1 = classification.get("f1", 0.0)
        accuracy = classification.get("accuracy", 0.0)
        
        report_lines.append(
            f"| {model_name} | {map50:.4f} | {map50_95:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} |"
        )
    
    report_lines.append("\n## Notes\n")
    report_lines.append("- mAP: Mean Average Precision")
    report_lines.append("- Precision/Recall: Detection metrics")
    report_lines.append("- F1/Accuracy: Classification metrics (on matched detections)")
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified visualization report for all models.")
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Evaluation result JSON files in format 'model_name:path/to/results.json'.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="agropest.yaml",
        help="Dataset YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/unified",
        help="Output directory for reports.",
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    data_root = Path(args.data_root)
    
    # 验证数据集结构
    print("Validating dataset structure...")
    validation = validate_dataset_structure(data_root)
    if not all(validation.values()):
        print("Warning: Some dataset components are missing:")
        for key, value in validation.items():
            if not value:
                print(f"  - {key}: Missing")
    
    # 绘制数据集统计
    print("\nGenerating dataset statistics...")
    plot_dataset_statistics(data_root, output_dir / "figures" / "dataset_statistics.png")
    
    # 如果有结果文件，生成模型对比
    if args.results:
        print("\nLoading evaluation results...")
        result_paths = {}
        for result_spec in args.results:
            if ":" in result_spec:
                model_name, path = result_spec.split(":", 1)
                result_paths[model_name] = Path(path)
            else:
                path = Path(result_spec)
                model_name = path.stem
                result_paths[model_name] = path
        
        results = load_evaluation_results(result_paths)
        
        if results:
            print(f"Generating comparison for {len(results)} models...")
            
            # 生成对比图
            plot_model_comparison(
                results,
                output_dir / "figures" / "detection_comparison.png",
                metric_type="detection",
            )
            plot_model_comparison(
                results,
                output_dir / "figures" / "classification_comparison.png",
                metric_type="classification",
            )
            plot_radar_chart(results, output_dir / "figures" / "performance_radar.png")
            
            # 生成对比表格
            create_comparison_table(
                results,
                output_dir / "tables" / "comparison_table.csv",
            )
            
            # 生成汇总报告
            generate_summary_report(
                results,
                data_root,
                output_dir / "summary_report.md",
            )
            
            print(f"\n✓ Unified visualization report generated in {output_dir}")
        else:
            print("Warning: No valid results found.")
    else:
        print("\nNo results specified. Only dataset statistics generated.")


if __name__ == "__main__":
    main()

