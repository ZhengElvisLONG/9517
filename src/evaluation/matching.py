from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (x1, y1, x2, y2)."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = boxes1_area[:, None] + boxes2_area - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    return iou


def greedy_match_iou(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """Greedy IoU-based matching between ground-truth and predictions."""

    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return []

    iou_matrix = box_iou(gt_boxes, pred_boxes)
    matches: List[Tuple[int, int]] = []

    used_gt = set()
    used_pred = set()

    # Flatten indices sorted by IoU descending
    flat_indices = torch.argsort(iou_matrix.flatten(), descending=True)
    rows = flat_indices // iou_matrix.shape[1]
    cols = flat_indices % iou_matrix.shape[1]

    for gt_idx, pred_idx in zip(rows.tolist(), cols.tolist()):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        if iou_matrix[gt_idx, pred_idx] < iou_threshold:
            break
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx))

    return matches


