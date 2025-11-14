from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    auc: float | None

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "accuracy": float(self.accuracy),
            "auc": float(self.auc) if self.auc is not None else None,
        }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    y_prob: Sequence[Sequence[float]] | None = None,
    average: str = "macro",
) -> ClassificationMetrics:
    """Compute precision, recall, F1, accuracy, and AUC (if probabilities provided)."""

    if len(y_true) == 0:
        raise ValueError("Empty ground-truth labels supplied.")

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    auc: float | None = None
    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob)
            if y_prob.ndim == 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            auc = None

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        auc=auc,
    )


def summarise_metrics(metrics: Iterable[ClassificationMetrics]) -> Dict[str, float]:
    """Average a list of metric objects."""

    metrics = list(metrics)
    if not metrics:
        raise ValueError("No metrics to summarise.")

    precision = float(np.mean([m.precision for m in metrics]))
    recall = float(np.mean([m.recall for m in metrics]))
    f1 = float(np.mean([m.f1 for m in metrics]))
    accuracy = float(np.mean([m.accuracy for m in metrics]))

    auc_values = [m.auc for m in metrics if m.auc is not None and not np.isnan(m.auc)]
    auc = float(np.mean(auc_values)) if auc_values else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auc": auc,
    }


