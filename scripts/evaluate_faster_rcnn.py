from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.datasets.agropest import AgroPestDataset, detection_collate_fn
from src.evaluation.coco import evaluate_coco_map
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.faster_rcnn import build_faster_rcnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Faster R-CNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root.")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"], help="Dataset split.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="", help="Device string (cpu or cuda).")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for classification metrics.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save metrics JSON.")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint missing 'config' entry.")
    return checkpoint


def evaluate(args: argparse.Namespace) -> Dict:
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = AgroPestDataset(data_root=args.data_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
    )

    model = build_faster_rcnn(
        num_classes=dataset.num_classes,
        backbone=cfg.get("backbone", "resnet50"),
        pretrained=False,
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    detection_metrics = evaluate_coco_map(model, dataloader, device=device)

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                matches = greedy_match_iou(
                    target["boxes"].to(device),
                    output["boxes"].to(device),
                    iou_threshold=args.iou_threshold,
                )
                for gt_idx, pred_idx in matches:
                    gt_label = int(target["labels"][gt_idx])
                    pred_label = int(output["labels"][pred_idx].item())
                    score_vector = [0.0] * dataset.num_classes
                    score_vector[pred_label] = float(output["scores"][pred_idx].item())
                    y_true.append(gt_label)
                    y_pred.append(pred_label)
                    y_prob.append(score_vector)

    classification_metrics = (
        compute_classification_metrics(y_true, y_pred, y_prob=y_prob, average="macro")
        if y_true
        else ClassificationMetrics(0.0, 0.0, 0.0, 0.0, None)
    )

    payload = {
        "split": args.split,
        "detection": detection_metrics,
        "classification": classification_metrics.to_dict(),
    }
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()


