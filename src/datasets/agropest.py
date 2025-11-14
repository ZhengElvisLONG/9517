from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as F
import yaml


@dataclass(frozen=True)
class SampleMeta:
    """Lightweight container describing a dataset sample."""

    image_path: Path
    label_path: Path
    index: int


class AgroPestDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the AgroPest-12 YOLO-formatted annotations."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transforms: Optional[Callable] = None,
        class_names: Optional[Iterable[str]] = None,
        preload_images: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.preload_images = preload_images

        self.images_dir = self.data_root / split / "images"
        self.labels_dir = self.data_root / split / "labels"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        if class_names is None:
            yaml_path = self.data_root / "data.yaml"
            if yaml_path.exists():
                with yaml_path.open("r", encoding="utf-8") as f:
                    data_cfg = yaml.safe_load(f)
                class_names = data_cfg.get("names")
        if class_names is None:
            raise ValueError("Class names must be provided or defined in data/data.yaml")

        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)

        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples: List[SampleMeta] = []
        for idx, image_path in enumerate(sorted(self.images_dir.iterdir())):
            if image_path.suffix.lower() not in valid_suffixes:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            self.samples.append(SampleMeta(image_path=image_path, label_path=label_path, index=idx))

        if not self.samples:
            raise RuntimeError(f"No images found inside {self.images_dir}")

        self._cache: Dict[int, Tuple[Image.Image, Dict[str, torch.Tensor]]] = {}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not (0 <= idx < len(self.samples)):
            raise IndexError(idx)

        if self.preload_images and idx in self._cache:
            image, target = self._cache[idx]
            return image.clone(), {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}

        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        width, height = image.size

        boxes: List[List[float]] = []
        labels: List[int] = []

        if sample.label_path.exists():
            with sample.label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:])

                    box_w = w * width
                    box_h = h * height
                    x_center = xc * width
                    y_center = yc * height

                    x1 = max(0.0, x_center - box_w / 2.0)
                    y1 = max(0.0, y_center - box_h / 2.0)
                    x2 = min(float(width), x_center + box_w / 2.0)
                    y2 = min(float(height), y_center + box_h / 2.0)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if boxes_tensor.numel() > 0
            else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd_tensor = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)

        target: Dict[str, torch.Tensor | str] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "image_path": str(sample.image_path),
        }

        if self.transforms:
            transformed = self.transforms(image=image, target=target)
            if isinstance(transformed, tuple):
                image, target = transformed
            else:
                image = transformed["image"]
                target = transformed["target"]
        else:
            image = F.to_tensor(image)

        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.dtype != torch.float32:
                image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
        else:
            image = F.to_tensor(image)

        if self.preload_images:
            self._cache[idx] = (image.clone(), {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()})

        return image, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Custom collate function for object detection dataloaders."""

    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


