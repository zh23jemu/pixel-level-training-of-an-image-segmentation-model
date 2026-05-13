from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """累计混淆矩阵，并计算 Pixel Accuracy、per-class IoU 和 mIoU。"""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        label = target.detach().cpu().numpy().reshape(-1)
        valid = (label != self.ignore_index) & (label >= 0) & (label < self.num_classes)
        label = label[valid]
        pred = pred[valid]
        indices = self.num_classes * label + pred
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        correct = np.diag(self.confusion)
        total = self.confusion.sum()
        accuracy = float(correct.sum() / total) if total > 0 else 0.0
        union = (
            self.confusion.sum(axis=1)
            + self.confusion.sum(axis=0)
            - correct
        )
        iou = np.divide(correct, union, out=np.full_like(correct, np.nan, dtype=float), where=union > 0)
        miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
        return {
            "pixel_accuracy": accuracy,
            "mean_iou": miou,
            "per_class_iou": iou.tolist(),
        }
