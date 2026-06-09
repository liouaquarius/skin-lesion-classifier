"""Correctness tests for FocalLoss, WeightedCrossEntropy, and build_loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.losses import FocalLoss, WeightedCrossEntropy, build_loss

_BATCH = 8
_CLASSES = 7


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(_BATCH, _CLASSES), torch.randint(0, _CLASSES, (_BATCH,))


# ── FocalLoss ────────────────────────────────────────────────────────────────


def test_focal_loss_returns_scalar() -> None:
    logits, targets = _inputs()
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.shape == torch.Size([])


def test_focal_gamma_zero_matches_cross_entropy() -> None:
    """γ=0 collapses focal loss to standard cross-entropy."""
    logits, targets = _inputs()
    fl = FocalLoss(gamma=0.0)(logits, targets)
    ce = nn.CrossEntropyLoss()(logits, targets)
    assert torch.allclose(fl, ce, atol=1e-5)


def test_focal_higher_gamma_reduces_loss() -> None:
    """Higher γ down-weights easy samples → lower mean loss on random logits."""
    logits, targets = _inputs()
    loss_g0 = FocalLoss(gamma=0.0)(logits, targets).item()
    loss_g2 = FocalLoss(gamma=2.0)(logits, targets).item()
    assert loss_g2 < loss_g0


# ── WeightedCrossEntropy ──────────────────────────────────────────────────────


def test_weighted_ce_returns_scalar() -> None:
    logits, targets = _inputs()
    loss = WeightedCrossEntropy()(logits, targets)
    assert loss.shape == torch.Size([])


def test_weighted_ce_uniform_weight_matches_ce() -> None:
    """Uniform weights of 1 produce the same result as unweighted CE."""
    logits, targets = _inputs()
    weight = torch.ones(_CLASSES)
    wce = WeightedCrossEntropy(weight=weight)(logits, targets)
    ce = nn.CrossEntropyLoss()(logits, targets)
    assert torch.allclose(wce, ce, atol=1e-5)


# ── build_loss factory ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["cross_entropy", "weighted_ce", "focal"])
def test_build_loss_returns_module(name: str) -> None:
    loss_fn = build_loss(name)
    logits, targets = _inputs()
    out = loss_fn(logits, targets)
    assert out.isfinite()


def test_build_loss_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("unknown_loss")
