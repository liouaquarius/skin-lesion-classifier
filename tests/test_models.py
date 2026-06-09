"""Forward-pass shape tests for all three architectures."""

from __future__ import annotations

import pytest
import torch

from src.models import build_model

_NUM_CLASSES = 7
_BATCH = 2
_ARCHITECTURES = ["resnet18", "efficientnet_b0", "vit_tiny_patch16_224"]


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    return torch.zeros(_BATCH, 3, 224, 224)


@pytest.mark.parametrize("name", _ARCHITECTURES)
def test_output_shape(name: str, dummy_input: torch.Tensor) -> None:
    model = build_model(name, num_classes=_NUM_CLASSES, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (_BATCH, _NUM_CLASSES), f"{name}: expected ({_BATCH}, {_NUM_CLASSES}), got {out.shape}"


def test_build_model_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model")
