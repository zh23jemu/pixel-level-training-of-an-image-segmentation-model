from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """多类别 Dice Loss，专门处理语义分割中的类别不均衡问题。"""

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        valid_mask = valid_mask.unsqueeze(1).to(dtype=probabilities.dtype)

        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dims)
        cardinality = torch.sum(probabilities + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """统一封装 CE、Dice 与组合损失，便于配置文件切换实验。"""

    def __init__(
        self,
        name: str,
        num_classes: int,
        ignore_index: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.name == "ce":
            return self.ce(logits, target)
        if self.name == "dice":
            return self.dice(logits, target)
        if self.name == "ce_dice":
            return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(
                logits, target
            )
        raise ValueError(f"未知损失函数配置：{self.name}")
