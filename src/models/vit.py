"""ViT-Tiny (patch 16, 224×224) built with timm."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_vit_tiny(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        "vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes
    )
