"""ResNet-18 baseline built with timm."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_resnet18(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
