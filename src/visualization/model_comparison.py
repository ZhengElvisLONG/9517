"""模型对比可视化工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_evaluation_results(result_paths: Dict[str, Path]) -> Dict[str, Dict]:
    """
    加载多个模型的评估结果。
    
    Args:
        result_paths: 模型名称到结果文件路径的映射
    
    Returns:
        模型名称到评估结果的映射
    """
    results = {}
    for model_name, path in result_paths.items():
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                results[model_name] = json.load(f)
        else:
            print(f"Warning: Results file not found: {path}")
    return results


def normalize_detection_metrics(metrics: Dict) -> Dict[str, float]:
    """
    标准化检测指标，统一不同模型的指标名称。
    
    Args:
        metrics: 原始检测指标字典
    
    Returns:
        标准化的指标字典
    """
    normalized = {}
    
    # mAP 指标（尝试多种可能的键名）
    if "mAP" in metrics:
        normalized["mAP"] = metrics["mAP"]
    elif "mAP@0.5" in metrics:
        normalized["mAP"] = metrics["mAP@0.5"]
    elif "metrics/mAP50(B)" in metrics:
        normalized["mAP"] = metrics["metrics/mAP50(B)"]
    else:
        normalized["mAP"] = 0.0
    
    # mAP@0.5
    if "mAP_50" in metrics:
        normalized["mAP@0.5"] = metrics["mAP_50"]
    elif "mAP@0.5" in metrics:
        normalized["mAP@0.5"] = metrics["mAP@0.5"]
    elif "metrics/mAP50(B)" in metrics:
        normalized["mAP@0.5"] = metrics["metrics/mAP50(B)"]
    else:
        normalized["mAP@0.5"] = normalized["mAP"]
    
    # mAP@0.5:0.95
    if "mAP@0.50:0.95" in metrics:
        normalized["mAP@0.5:0.95"] = metrics["mAP@0.50:0.95"]
    elif "metrics/mAP50-95(B)" in metrics:
        normalized["mAP@0.5:0.95"] = metrics["metrics/mAP50-95(B)"]
    else:
        normalized["mAP@0.5:0.95"] = normalized["mAP"]
    
    # Precision
    if "precision" in metrics:
        normalized["Precision"] = metrics["precision"]
    elif "metrics/precision(B)" in metrics:
        normalized["Precision"] = metrics["metrics/precision(B)"]
    else:
        normalized["Precision"] = 0.0
    
    # Recall
    if "recall" in metrics:
        normalized["Recall"] = metrics["recall"]
    elif "metrics/recall(B)" in metrics:
        normalized["Recall"] = metrics["metrics/recall(B)"]
    else:
        normalized["Recall"] = 0.0
    
    return normalized


def plot_model_comparison(
    results: Dict[str, Dict],
    output_path: Path,
    metric_type: str = "detection",
) -> None:
    """
    绘制模型对比图。
    
    Args:
        results: 模型名称到评估结果的映射
        output_path: 输出图像路径
        metric_type: 指标类型（"detection" 或 "classification"）
    """
    if metric_type == "detection":
        # 检测指标对比
        model_names = list(results.keys())
        metrics_to_plot = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
        
        # 提取指标值
        values = {metric: [] for metric in metrics_to_plot}
        for model_name in model_names:
            detection_metrics = results[model_name].get("detection", {})
            normalized = normalize_detection_metrics(detection_metrics)
            for metric in metrics_to_plot:
                values[metric].append(normalized.get(metric, 0.0))
        
        # 绘制柱状图
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, metric in enumerate(metrics_to_plot):
            offset = (idx - len(metrics_to_plot) / 2) * width + width / 2
            bars = ax.bar(x + offset, values[metric], width, label=metric)
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison: Detection Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    elif metric_type == "classification":
        # 分类指标对比
        model_names = list(results.keys())
        metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
        
        # 提取指标值
        values = {metric: [] for metric in metrics_to_plot}
        for model_name in model_names:
            classification_metrics = results[model_name].get("classification", {})
            for metric in metrics_to_plot:
                values[metric].append(classification_metrics.get(metric, 0.0))
        
        # 绘制柱状图
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, metric in enumerate(metrics_to_plot):
            offset = (idx - len(metrics_to_plot) / 2) * width + width / 2
            bars = ax.bar(x + offset, values[metric], width, label=metric.upper())
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison: Classification Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def create_comparison_table(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """
    创建模型对比表格。
    
    Args:
        results: 模型名称到评估结果的映射
        output_path: 输出 CSV 文件路径
    """
    rows = []
    
    for model_name, result in results.items():
        detection_metrics = result.get("detection", {})
        classification_metrics = result.get("classification", {})
        
        normalized_det = normalize_detection_metrics(detection_metrics)
        
        row = {
            "Model": model_name,
            "mAP@0.5": f"{normalized_det.get('mAP@0.5', 0.0):.4f}",
            "mAP@0.5:0.95": f"{normalized_det.get('mAP@0.5:0.95', 0.0):.4f}",
            "Precision (Det)": f"{normalized_det.get('Precision', 0.0):.4f}",
            "Recall (Det)": f"{normalized_det.get('Recall', 0.0):.4f}",
            "Precision (Cls)": f"{classification_metrics.get('precision', 0.0):.4f}",
            "Recall (Cls)": f"{classification_metrics.get('recall', 0.0):.4f}",
            "F1 (Cls)": f"{classification_metrics.get('f1', 0.0):.4f}",
            "Accuracy (Cls)": f"{classification_metrics.get('accuracy', 0.0):.4f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def plot_radar_chart(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """
    绘制雷达图对比模型性能。
    
    Args:
        results: 模型名称到评估结果的映射
        output_path: 输出图像路径
    """
    # 选择关键指标
    metrics = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "Accuracy"]
    
    # 准备数据
    model_names = list(results.keys())
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    
    for model_name in model_names:
        result = results[model_name]
        detection_metrics = result.get("detection", {})
        classification_metrics = result.get("classification", {})
        
        normalized_det = normalize_detection_metrics(detection_metrics)
        
        values = [
            normalized_det.get("mAP@0.5", 0.0),
            normalized_det.get("mAP@0.5:0.95", 0.0),
            normalized_det.get("Precision", 0.0),
            normalized_det.get("Recall", 0.0),
            classification_metrics.get("f1", 0.0),
            classification_metrics.get("accuracy", 0.0),
        ]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, "o-", linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar Chart", size=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

