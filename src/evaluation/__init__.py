"""评估模块：统一的模型评估接口。"""

from src.evaluation.coco import evaluate_coco_map, dataset_to_coco_dict
from src.evaluation.matching import greedy_match_iou
from src.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    summarise_metrics,
)
from src.evaluation.unified_evaluator import ModelType, UnifiedEvaluator

__all__ = [
    "evaluate_coco_map",
    "dataset_to_coco_dict",
    "greedy_match_iou",
    "ClassificationMetrics",
    "compute_classification_metrics",
    "summarise_metrics",
    "UnifiedEvaluator",
    "ModelType",
]

