"""Profile model architectures: parameter count, checkpoint size, CPU latency.

Builds each architecture (ResNet18, EfficientNet-B0, ViT-Tiny) and reports total
parameters, the on-disk checkpoint size, and mean single-image inference latency
on CPU (the deployment target). Used to populate the README model table.

Usage:
    uv run python scripts/profile_models.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from src.models import build_model

_CKPT = Path("results/checkpoints")

# label -> (build_model name, representative checkpoint, pretrained source)
_ARCHS = {
    "ResNet18": ("resnet18", "resnet18-ce-seed42_best.pt", "torchvision / ImageNet"),
    "EfficientNet-B0": (
        "efficientnet_b0", "efficientnet_b0-ce-seed42_best.pt", "torchvision / ImageNet",
    ),
    "ViT-Tiny": (
        "vit_tiny_patch16_224", "vit_tiny-wce-seed42_best.pt", "timm / ImageNet",
    ),
}
_WARMUP = 5
_RUNS = 30


def _latency_ms(model: torch.nn.Module) -> float:
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(_WARMUP):
            model(x)
        t0 = time.perf_counter()
        for _ in range(_RUNS):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / _RUNS * 1000


def main() -> None:
    header = f"{'Architecture':18s} {'Params(M)':>10s} {'Size(MB)':>9s} {'CPU lat(ms)':>12s}  Pretrained"
    print(header)
    print("-" * len(header))
    for label, (name, ckpt, pretrained) in _ARCHS.items():
        model = build_model(name, num_classes=7, pretrained=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        ckpt_path = _CKPT / ckpt
        size = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else float("nan")
        lat = _latency_ms(model)
        print(f"{label:18s} {params:10.2f} {size:9.1f} {lat:12.1f}  {pretrained}")


if __name__ == "__main__":
    main()
