from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.datasets.agropest import AgroPestDataset


def dataset_to_coco_dict(dataset: AgroPestDataset) -> Dict:
    categories = [{"id": idx, "name": name} for idx, name in enumerate(dataset.class_names)]
    images: List[Dict] = []
    annotations: List[Dict] = []

    annotation_id = 1
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        image_id = int(target["image_id"][0].item() if torch.is_tensor(target["image_id"]) else target["image_id"])
        height, width = target["orig_size"].tolist()
        file_name = Path(target["image_path"]).name

        images.append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": file_name,
            }
        )

        boxes = target["boxes"]
        labels = target["labels"]
        for box, label in zip(boxes.tolist(), labels.tolist()):
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            area = float((x2 - x1) * (y2 - y1))
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    # COCO 格式需要包含 'info' 和 'licenses' 字段
    return {
        "info": {
            "description": "AgroPest-12 Dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def evaluate_coco_map(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute COCO-style mAP metrics for the provided model and dataloader."""

    dataset: AgroPestDataset = data_loader.dataset  # type: ignore
    coco_dict = dataset_to_coco_dict(dataset)

    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()

    results = []
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id_tensor = target["image_id"]
                image_id = int(image_id_tensor[0].item() if torch.is_tensor(image_id_tensor) else image_id_tensor)

                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    x1, y1, x2, y2 = box
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                        }
                    )

    if not results:
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    keys = [
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    metrics = {key: float(value) for key, value in zip(keys, coco_eval.stats.tolist())}
    return metrics


