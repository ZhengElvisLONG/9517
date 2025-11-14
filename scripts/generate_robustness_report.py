"""生成鲁棒性测试报告：可视化 mAP 下降曲线和对比分析。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def load_robustness_results(results_path: Path) -> Dict:
    """加载鲁棒性评估结果。"""
    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_robustness_curves(
    results: Dict,
    baseline_results: Dict,
    output_dir: Path,
) -> None:
    """
    绘制鲁棒性曲线：横轴失真强度，纵轴 mAP。
    
    Args:
        results: 鲁棒性评估结果
        baseline_results: 基线结果（原始测试集）
        output_dir: 输出目录
    """
    # 组织数据：按失真类型分组
    corruption_types = {}
    
    for corruption_name, result in results.items():
        corruption_type = result["corruption_type"]
        level = result["level"]
        
        if corruption_type not in corruption_types:
            corruption_types[corruption_type] = {
                "levels": [],
                "frcnn_maps": [],
                "yolo_maps": [],
            }
        
        corruption_types[corruption_type]["levels"].append(level)
        
        # 获取 mAP 值
        frcnn_map = result.get("faster_rcnn", {}).get("detection", {}).get("mAP", 0.0)
        yolo_map = result.get("yolov8", {}).get("detection", {}).get("mAP@0.5", 0.0)
        
        corruption_types[corruption_type]["frcnn_maps"].append(frcnn_map)
        corruption_types[corruption_type]["yolo_maps"].append(yolo_map)
    
    # 获取基线 mAP（支持多种结果格式）
    if "faster_rcnn" in baseline_results:
        baseline_frcnn_map = baseline_results["faster_rcnn"].get("detection", {}).get("mAP", 0.0)
    elif "detection" in baseline_results:
        baseline_frcnn_map = baseline_results["detection"].get("mAP", 0.0)
    else:
        baseline_frcnn_map = 0.0
    
    if "yolov8" in baseline_results:
        baseline_yolo_map = baseline_results["yolov8"].get("detection", {}).get("mAP@0.5", 0.0)
    elif "detection" in baseline_results:
        baseline_yolo_map = baseline_results["detection"].get("mAP@0.5", baseline_results["detection"].get("metrics/mAP50(B)", 0.0))
    else:
        baseline_yolo_map = 0.0
    
    # 为每种失真类型创建图表
    for corruption_type, data in corruption_types.items():
        # 排序按级别
        sorted_indices = np.argsort(data["levels"])
        levels = np.array(data["levels"])[sorted_indices]
        frcnn_maps = np.array(data["frcnn_maps"])[sorted_indices]
        yolo_maps = np.array(data["yolo_maps"])[sorted_indices]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制基线（水平线）
        ax.axhline(
            y=baseline_frcnn_map,
            color="blue",
            linestyle="--",
            alpha=0.5,
            label=f"Faster R-CNN Baseline (mAP={baseline_frcnn_map:.3f})",
        )
        ax.axhline(
            y=baseline_yolo_map,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"YOLOv8 Baseline (mAP={baseline_yolo_map:.3f})",
        )
        
        # 绘制下降曲线
        ax.plot(
            levels,
            frcnn_maps,
            marker="o",
            color="blue",
            linewidth=2,
            markersize=8,
            label="Faster R-CNN",
        )
        ax.plot(
            levels,
            yolo_maps,
            marker="s",
            color="red",
            linewidth=2,
            markersize=8,
            label="YOLOv8",
        )
        
        ax.set_xlabel("Corruption Intensity Level", fontsize=12)
        ax.set_ylabel("mAP@0.5", fontsize=12)
        ax.set_title(f"Robustness Curve: {corruption_type.capitalize()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = output_dir / f"robustness_curve_{corruption_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved: {output_path}")


def create_robustness_table(
    results: Dict,
    baseline_results: Dict,
    output_dir: Path,
) -> None:
    """
    创建鲁棒性表格：每类失真下的 mAP 下降值。
    
    Args:
        results: 鲁棒性评估结果
        baseline_results: 基线结果
        output_dir: 输出目录
    """
    rows = []
    
    # 获取基线 mAP（支持多种结果格式）
    if "faster_rcnn" in baseline_results:
        baseline_frcnn_map = baseline_results["faster_rcnn"].get("detection", {}).get("mAP", 0.0)
    elif "detection" in baseline_results:
        baseline_frcnn_map = baseline_results["detection"].get("mAP", 0.0)
    else:
        baseline_frcnn_map = 0.0
    
    if "yolov8" in baseline_results:
        baseline_yolo_map = baseline_results["yolov8"].get("detection", {}).get("mAP@0.5", 0.0)
    elif "detection" in baseline_results:
        baseline_yolo_map = baseline_results["detection"].get("mAP@0.5", baseline_results["detection"].get("metrics/mAP50(B)", 0.0))
    else:
        baseline_yolo_map = 0.0
    
    for corruption_name, result in results.items():
        corruption_type = result["corruption_type"]
        level = result["level"]
        
        frcnn_map = result.get("faster_rcnn", {}).get("detection", {}).get("mAP", 0.0)
        yolo_map = result.get("yolov8", {}).get("detection", {}).get("mAP@0.5", 0.0)
        
        frcnn_delta = baseline_frcnn_map - frcnn_map
        yolo_delta = baseline_yolo_map - yolo_map
        
        rows.append(
            {
                "Corruption Type": corruption_type.capitalize(),
                "Level": level,
                "Faster R-CNN mAP": f"{frcnn_map:.4f}",
                "Faster R-CNN ΔmAP": f"{frcnn_delta:.4f}",
                "YOLOv8 mAP": f"{yolo_map:.4f}",
                "YOLOv8 ΔmAP": f"{yolo_delta:.4f}",
            }
        )
    
    df = pd.DataFrame(rows)
    
    # 保存为 CSV
    csv_path = output_dir / "robustness_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # 保存为 Markdown 表格
    md_path = output_dir / "robustness_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Robustness Evaluation Results\n\n")
        f.write(f"Baseline mAP - Faster R-CNN: {baseline_frcnn_map:.4f}\n")
        f.write(f"Baseline mAP - YOLOv8: {baseline_yolo_map:.4f}\n\n")
        f.write(df.to_markdown(index=False))
    print(f"  Saved: {md_path}")


def generate_summary_analysis(
    results: Dict,
    baseline_results: Dict,
    output_dir: Path,
) -> None:
    """
    生成总结分析报告。
    
    Args:
        results: 鲁棒性评估结果
        baseline_results: 基线结果
        output_dir: 输出目录
    """
    # 获取基线 mAP（支持多种结果格式）
    if "faster_rcnn" in baseline_results:
        baseline_frcnn_map = baseline_results["faster_rcnn"].get("detection", {}).get("mAP", 0.0)
    elif "detection" in baseline_results:
        baseline_frcnn_map = baseline_results["detection"].get("mAP", 0.0)
    else:
        baseline_frcnn_map = 0.0
    
    if "yolov8" in baseline_results:
        baseline_yolo_map = baseline_results["yolov8"].get("detection", {}).get("mAP@0.5", 0.0)
    elif "detection" in baseline_results:
        baseline_yolo_map = baseline_results["detection"].get("mAP@0.5", baseline_results["detection"].get("metrics/mAP50(B)", 0.0))
    else:
        baseline_yolo_map = 0.0
    
    # 计算每种失真的平均下降值
    corruption_summaries = {}
    
    for corruption_name, result in results.items():
        corruption_type = result["corruption_type"]
        level = result["level"]
        
        if corruption_type not in corruption_summaries:
            corruption_summaries[corruption_type] = {
                "frcnn_deltas": [],
                "yolo_deltas": [],
            }
        
        frcnn_map = result.get("faster_rcnn", {}).get("detection", {}).get("mAP", 0.0)
        yolo_map = result.get("yolov8", {}).get("detection", {}).get("mAP@0.5", 0.0)
        
        frcnn_delta = baseline_frcnn_map - frcnn_map
        yolo_delta = baseline_yolo_map - yolo_map
        
        corruption_summaries[corruption_type]["frcnn_deltas"].append(frcnn_delta)
        corruption_summaries[corruption_type]["yolo_deltas"].append(yolo_delta)
    
    # 找出最严重的失真类型
    worst_corruption_frcnn = max(
        corruption_summaries.items(),
        key=lambda x: np.mean(x[1]["frcnn_deltas"]) if x[1]["frcnn_deltas"] else 0,
    )
    worst_corruption_yolo = max(
        corruption_summaries.items(),
        key=lambda x: np.mean(x[1]["yolo_deltas"]) if x[1]["yolo_deltas"] else 0,
    )
    
    # 生成报告
    report_path = output_dir / "robustness_analysis.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Robustness Evaluation Analysis\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Baseline mAP (Faster R-CNN)**: {baseline_frcnn_map:.4f}\n")
        f.write(f"- **Baseline mAP (YOLOv8)**: {baseline_yolo_map:.4f}\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Most Damaging Corruptions\n\n")
        f.write(f"- **Faster R-CNN**: {worst_corruption_frcnn[0].capitalize()} ")
        f.write(f"(Average ΔmAP: {np.mean(worst_corruption_frcnn[1]['frcnn_deltas']):.4f})\n")
        f.write(f"- **YOLOv8**: {worst_corruption_yolo[0].capitalize()} ")
        f.write(f"(Average ΔmAP: {np.mean(worst_corruption_yolo[1]['yolo_deltas']):.4f})\n\n")
        
        f.write("### Model Comparison\n\n")
        # 计算两阶段 vs 单阶段的平均鲁棒性
        avg_frcnn_delta = np.mean(
            [
                delta
                for corr_data in corruption_summaries.values()
                for delta in corr_data["frcnn_deltas"]
            ]
        )
        avg_yolo_delta = np.mean(
            [
                delta
                for corr_data in corruption_summaries.values()
                for delta in corr_data["yolo_deltas"]
            ]
        )
        
        f.write(f"- **Faster R-CNN (Two-stage) Average ΔmAP**: {avg_frcnn_delta:.4f}\n")
        f.write(f"- **YOLOv8 (One-stage) Average ΔmAP**: {avg_yolo_delta:.4f}\n\n")
        
        if avg_frcnn_delta < avg_yolo_delta:
            f.write("**Conclusion**: Faster R-CNN (two-stage) shows better robustness.\n\n")
        else:
            f.write("**Conclusion**: YOLOv8 (one-stage) shows better robustness.\n\n")
        
        f.write("## Detailed Results by Corruption Type\n\n")
        for corruption_type, data in sorted(corruption_summaries.items()):
            f.write(f"### {corruption_type.capitalize()}\n\n")
            f.write(f"- Faster R-CNN Average ΔmAP: {np.mean(data['frcnn_deltas']):.4f}\n")
            f.write(f"- YOLOv8 Average ΔmAP: {np.mean(data['yolo_deltas']):.4f}\n\n")
    
    print(f"  Saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate robustness evaluation report.")
    parser.add_argument(
        "--results",
        type=str,
        default="experiments/robustness/robustness_results.json",
        help="Path to robustness results JSON.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/robustness",
        help="Output directory for reports.",
    )
    
    args = parser.parse_args()
    
    # 加载结果
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    
    results = load_robustness_results(results_path)
    with baseline_path.open("r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating robustness report...")
    
    # 生成图表
    print("\n1. Generating robustness curves...")
    plot_robustness_curves(results, baseline_results, output_dir)
    
    # 生成表格
    print("\n2. Creating robustness table...")
    create_robustness_table(results, baseline_results, output_dir)
    
    # 生成分析报告
    print("\n3. Generating summary analysis...")
    generate_summary_analysis(results, baseline_results, output_dir)
    
    print(f"\n✓ Robustness report generated in {output_dir}")


if __name__ == "__main__":
    main()

