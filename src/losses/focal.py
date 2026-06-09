"""Focal Loss for multi-class classification.

FL(p_t) = -(1 - p_t)^γ · log(p_t)

Setting γ=0 recovers standard cross-entropy; γ=2 (default) down-weights
easy negatives and focuses training on hard examples — useful for HAM10000's
58× class imbalance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # register_buffer so the weight tensor moves with .to(device)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        modulating = (1.0 - pt) ** self.gamma

        if self.weight is not None:
            modulating = modulating * self.weight[targets]  # type: ignore[index]

        loss = -modulating * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
