"""Class-weighted cross-entropy loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, targets)
