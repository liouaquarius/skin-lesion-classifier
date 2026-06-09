"""EfficientNet-B0 built with timm."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_efficientnet_b0(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
