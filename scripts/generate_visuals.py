from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_class_names(data_yaml: Path) -> List[str]:
    cfg = load_yaml(data_yaml)
    names = cfg.get("names")
    if not names:
        raise ValueError(f"No class names defined in {data_yaml}")
    return names


def collect_labels(label_dir: Path) -> Iterable[Tuple[int, float, float, float, float]]:
    for txt in label_dir.glob("*.txt"):
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                yield (int(float(parts[0])), *(float(x) for x in parts[1:]))


def compute_class_distribution(data_root: Path, splits: Iterable[str], class_names: List[str]) -> pd.DataFrame:
    records = []
    for split in splits:
        label_dir = data_root / split / "labels"
        counts = Counter(cls for cls, *_ in collect_labels(label_dir))
        for idx, name in enumerate(class_names):
            records.append(
                {
                    "split": split,
                    "class_id": idx,
                    "class_name": name,
                    "instances": counts.get(idx, 0),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot(index="class_name", columns="split", values="instances").fillna(0)
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", width=0.8)
    plt.ylabel("Bounding Box Count")
    plt.title("AgroPest-12 Class Distribution per Split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_metrics_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_detection_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    display_labels = ["Precision", "Recall", "mAP@0.50", "mAP@0.50:0.95"]
    values = [metrics.get(k, 0.0) for k in keys]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(display_labels, values, color="#4C72B0")
    plt.ylim(0, max(values) * 1.2 if values else 1)
    plt.ylabel("Score")
    plt.title("YOLOv8 Detection Metrics (Validation)")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005, f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_classification_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    keys = ["precision", "recall", "f1", "accuracy"]
    values = [metrics.get(k, 0.0) if metrics.get(k) is not None else 0.0 for k in keys]

    plt.figure(figsize=(8, 5))
    bars = plt.bar([k.upper() for k in keys], values, color="#55A868")
    plt.ylim(0, max(values) * 1.2 if values else 1)
    plt.ylabel("Score")
    plt.title("YOLOv8 Classification Metrics (Matched Detections)")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005, f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_prediction_categories(predictions_json: Path, output_path: Path, top_k: int = 15) -> None:
    with predictions_json.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    counter = Counter(int(pred["category_id"]) for pred in predictions)
    total = sum(counter.values())
    top_items = counter.most_common(top_k)
    labels = [str(k) for k, _ in top_items]
    values = [count / total if total else 0 for _, count in top_items]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#C44E52")
    plt.ylabel("Proportion")
    plt.xlabel("Predicted Category ID (COCO)")
    plt.title("Top Predicted Categories (YOLOv8 Baseline)")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{value:.2%}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_ground_truth_boxes(data_root: Path, split: str, class_names: List[str]) -> Dict[str, Dict]:
    result = {}
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"

    for image_path in images_dir.glob("*"):
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    cls, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append((int(cls), x_center, y_center, width, height))
        result[image_path.name] = {"path": image_path, "boxes": boxes}
    return result


def yolo_to_xyxy(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    xc, yc, w, h = box
    box_w = w * width
    box_h = h * height
    x1 = (xc * width) - box_w / 2
    y1 = (yc * height) - box_h / 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    return x1, y1, x2, y2


def draw_boxes(ax, boxes, color, labels=None, linewidth=2.0):
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        if labels:
            ax.text(x1, y1 - 2, labels[idx], fontsize=8, color=color, backgroundcolor="black", ha="left", va="top")


def render_qualitative_examples(
    data_root: Path,
    split: str,
    class_names: List[str],
    predictions_json: Path,
    output_dir: Path,
    max_images: int = 3,
) -> None:
    with predictions_json.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    predictions_by_file = defaultdict(list)
    for item in predictions:
        predictions_by_file[item["file_name"]].append(item)

    gt_cache = load_ground_truth_boxes(data_root, split, class_names)

    for idx, (file_name, gt_info) in enumerate(gt_cache.items()):
        if idx >= max_images:
            break

        image_path = gt_info["path"]
        image = plt.imread(image_path)
        height, width = image.shape[:2]

        gt_boxes = []
        gt_labels = []
        for cls_id, xc, yc, w, h in gt_info["boxes"]:
            x1, y1, x2, y2 = yolo_to_xyxy((xc, yc, w, h), width, height)
            gt_boxes.append((x1, y1, x2, y2))
            gt_labels.append(class_names[cls_id])

        pred_boxes = []
        pred_labels = []
        for pred in predictions_by_file.get(file_name, []):
            x, y, w, h = pred["bbox"]
            pred_boxes.append((x, y, x + w, y + h))
            pred_labels.append(f"id:{pred['category_id']} {pred['score']:.2f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(image)
        axes[0].set_title(f"Ground Truth: {file_name}")
        draw_boxes(axes[0], gt_boxes, color="#1b9e77", labels=gt_labels)
        axes[0].axis("off")

        axes[1].imshow(image)
        axes[1].set_title("YOLOv8 Predictions (COCO classes)")
        draw_boxes(axes[1], pred_boxes, color="#d95f02", labels=pred_labels)
        axes[1].axis("off")

        plt.tight_layout()
        output_path = output_dir / f"qualitative_{idx+1}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def save_tables(
    dataset_df: pd.DataFrame,
    detection_metrics: Dict[str, float],
    classification_metrics: Dict[str, float],
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_df.to_csv(tables_dir / "dataset_distribution.csv", index=False)

    detection_table = pd.DataFrame([detection_metrics])
    detection_table.to_csv(tables_dir / "yolov8_detection_metrics.csv", index=False)

    classification_table = pd.DataFrame([classification_metrics])
    classification_table.to_csv(tables_dir / "yolov8_classification_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual summaries for AgroPest-12 project.")
    parser.add_argument("--data-yaml", type=Path, default=Path("agropest.yaml"), help="Dataset YAML path.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory of AgroPest data.")
    parser.add_argument("--result-json", type=Path, default=Path("experiments/yolov8/val_metrics_baseline.json"), help="Evaluation metrics json.")
    parser.add_argument("--predictions-json", type=Path, default=Path("runs/detect/val9/predictions.json"), help="YOLO predictions JSON.")
    parser.add_argument("--fig-dir", type=Path, default=Path("reports/figures"), help="Directory to save figures.")
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"), help="Directory to save tables.")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split for qualitative examples.")
    parser.add_argument("--max-images", type=int, default=3, help="Number of qualitative examples to render.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(args.data_yaml)
    dataset_df = compute_class_distribution(args.data_root, ["train", "valid", "test"], class_names)
    plot_class_distribution(dataset_df, args.fig_dir / "class_distribution.png")

    metrics_json = load_metrics_json(args.result_json)
    detection_metrics = metrics_json.get("detection", {})
    classification_metrics = metrics_json.get("classification", {})
    plot_detection_metrics(detection_metrics, args.fig_dir / "yolov8_detection_metrics.png")
    plot_classification_metrics(classification_metrics, args.fig_dir / "yolov8_classification_metrics.png")

    analyze_prediction_categories(args.predictions_json, args.fig_dir / "yolov8_prediction_categories.png")
    render_qualitative_examples(
        args.data_root,
        args.split,
        class_names,
        args.predictions_json,
        args.fig_dir,
        max_images=args.max_images,
    )

    save_tables(dataset_df, detection_metrics, classification_metrics, args.table_dir)


if __name__ == "__main__":
    main()


