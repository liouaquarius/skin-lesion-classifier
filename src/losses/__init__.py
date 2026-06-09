"""Loss functions and factory for training YAML configs."""

from __future__ import annotations

import torch
import torch.nn as nn

from .focal import FocalLoss
from .weighted_ce import WeightedCrossEntropy


def build_loss(
    name: str,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    """Instantiate a loss by its ``loss.name`` config key.

    Supported names: ``cross_entropy``, ``weighted_ce``, ``focal``.
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return WeightedCrossEntropy(weight=weight)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    raise ValueError(f"Unknown loss '{name}'. Choose from: cross_entropy, weighted_ce, focal")


__all__ = ["FocalLoss", "WeightedCrossEntropy", "build_loss"]
