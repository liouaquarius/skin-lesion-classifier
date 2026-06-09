"""Model factory — instantiate any architecture by its config name."""

from __future__ import annotations

import torch.nn as nn

from .efficientnet import build_efficientnet_b0
from .resnet import build_resnet18
from .vit import build_vit_tiny

_REGISTRY: dict[str, object] = {
    "resnet18": build_resnet18,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_tiny_patch16_224": build_vit_tiny,
}


def build_model(name: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Return an initialised model by its ``model.name`` config key."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained)  # type: ignore[operator]


__all__ = ["build_resnet18", "build_efficientnet_b0", "build_vit_tiny", "build_model"]
