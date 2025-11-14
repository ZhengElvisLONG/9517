from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.agropest import AgroPestDataset, detection_collate_fn
from src.evaluation.coco import evaluate_coco_map
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.faster_rcnn import build_faster_rcnn


@dataclass
class TrainerConfig:
    data_root: str = "data"
    output_dir: str = "experiments/faster_rcnn"
    split_train: str = "train"
    split_val: str = "valid"
    backbone: str = "resnet50"
    pretrained: bool = True
    trainable_backbone_layers: Optional[int] = 3
    epochs: int = 25
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 0.005
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"  # ["sgd", "adamw"]
    scheduler: str = "step"  # ["step", "cosine", "plateau", "none"]
    step_size: int = 8
    gamma: float = 0.1
    device: str = "cuda"
    amp: bool = True
    patience: int = 10
    classification_iou: float = 0.5
    save_every: int = 5


class FasterRCNNTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset = AgroPestDataset(data_root=cfg.data_root, split=cfg.split_train, transforms=None)
        self.val_dataset = AgroPestDataset(data_root=cfg.data_root, split=cfg.split_val, transforms=None)

        self.model = build_faster_rcnn(
            num_classes=self.train_dataset.num_classes,
            backbone=cfg.backbone,  # type: ignore[arg-type]
            pretrained=cfg.pretrained,
            trainable_backbone_layers=cfg.trainable_backbone_layers,
        )
        self.model.to(self.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=detection_collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=detection_collate_fn,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    def _build_optimizer(self) -> Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer.lower() == "sgd":
            return SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        if self.cfg.optimizer.lower() == "adamw":
            return AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

    def _build_scheduler(self):
        if self.cfg.scheduler == "none":
            return None
        if self.cfg.scheduler == "step":
            return StepLR(self.optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
        if self.cfg.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)
        if self.cfg.scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="max", patience=3, factor=0.5, verbose=True)
        raise ValueError(f"Unsupported scheduler: {self.cfg.scheduler}")

    def _move_targets_to_device(self, targets: List[Dict]) -> List[Dict]:
        new_targets = []
        for target in targets:
            new_target = {}
            for key, value in target.items():
                if torch.is_tensor(value):
                    new_target[key] = value.to(self.device)
                else:
                    new_target[key] = value
            new_targets.append(new_target)
        return new_targets

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        batches = len(self.train_loader)

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}", unit="batch")
        for batch_idx, (images, targets) in enumerate(progress):
            images = [img.to(self.device) for img in images]
            targets = self._move_targets_to_device(targets)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = losses.item()
            running_loss += loss_value
            progress.set_postfix({"loss": f"{loss_value:.4f}"})

        avg_loss = running_loss / max(1, batches)
        return avg_loss

    def _evaluate_classification(self) -> ClassificationMetrics:
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[List[float]] = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for target, output in zip(targets, outputs):
                    gt_boxes = target["boxes"].to(self.device)
                    pred_boxes = output["boxes"].to(self.device)
                    match_pairs = greedy_match_iou(gt_boxes, pred_boxes, iou_threshold=self.cfg.classification_iou)

                    for gt_idx, pred_idx in match_pairs:
                        gt_label = int(target["labels"][gt_idx])
                        pred_label = int(output["labels"][pred_idx].item())
                        score_vec = [0.0] * self.train_dataset.num_classes
                        score_vec[pred_label] = float(output["scores"][pred_idx].item())
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                        y_prob.append(score_vec)

        if not y_true:
            return ClassificationMetrics(precision=0.0, recall=0.0, f1=0.0, accuracy=0.0, auc=None)
        return compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, average="macro")

    def _evaluate_detection(self) -> Dict[str, float]:
        return evaluate_coco_map(self.model, self.val_loader, device=self.device)

    def train(self) -> Dict[str, float]:
        best_map = -math.inf
        epochs_without_improvement = 0
        history: List[Dict[str, float]] = []

        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()
            train_loss = self._train_one_epoch(epoch)

            detection_metrics = self._evaluate_detection()
            classification_metrics = self._evaluate_classification()

            epoch_time = time.time() - epoch_start

            metrics_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "time_sec": float(epoch_time),
                **{f"detection_{k}": v for k, v in detection_metrics.items()},
                **{f"classification_{k}": v for k, v in classification_metrics.to_dict().items()},
            }
            history.append(metrics_record)
            self._log_metrics(metrics_record)

            current_map = detection_metrics.get("mAP", float("-inf"))
            if current_map > best_map:
                best_map = current_map
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, best=True)
            else:
                epochs_without_improvement += 1

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(current_map)
                else:
                    self.scheduler.step()

            if epochs_without_improvement >= self.cfg.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            if (epoch + 1) % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, best=False)

        history_path = self.output_dir / "training_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return history[-1] if history else {}

    def _save_checkpoint(self, epoch: int, *, best: bool) -> None:
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": asdict(self.cfg),
        }
        suffix = "best" if best else f"epoch{epoch+1}"
        ckpt_path = self.output_dir / f"faster_rcnn_{suffix}.pth"
        torch.save(state, ckpt_path)

    def _log_metrics(self, metrics_record: Dict[str, float]) -> None:
        log_path = self.output_dir / "metrics.log"
        line = json.dumps(metrics_record)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_training(config_path: Optional[str] = None) -> None:
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = TrainerConfig(**data)
    else:
        cfg = TrainerConfig()

    trainer = FasterRCNNTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    run_training()


