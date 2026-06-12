"""Aggregate cross-model comparison figure from per-run test metrics.

Reads ``results/metrics/*_test.json`` and plots test accuracy against melanoma
sensitivity for all experiments, visualising the accuracy / minority-class
trade-off in a single view. Saves ``results/visualizations/model_comparison.png``.

Usage:
    uv run python scripts/plot_comparison.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_METRICS = Path("results/metrics")
_OUT = Path("results/visualizations/model_comparison.png")

_MODEL_COLOR = {
    "resnet18": "#3b82f6",
    "efficientnet_b0": "#10b981",
    "vit_tiny": "#f59e0b",
}
_MODEL_LABEL = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_tiny": "ViT-Tiny",
}
_LOSS_MARKER = {"ce": "o", "wce": "s", "focal": "^"}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for f in sorted(glob.glob(str(_METRICS / "*_test.json"))):
        j = json.load(open(f, encoding="utf-8"))
        model, loss, _ = j["run_name"].split("-")
        x = j["accuracy"]
        y = j["per_class_sensitivity"]["mel"]
        ax.scatter(
            x, y, s=140, c=_MODEL_COLOR[model], marker=_LOSS_MARKER[loss],
            edgecolors="white", linewidths=1.3, zorder=3,
        )
        ax.annotate(
            loss, (x, y), textcoords="offset points", xytext=(8, 4),
            fontsize=8, color="#475569",
        )

    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=11, label=_MODEL_LABEL[m])
        for m, c in _MODEL_COLOR.items()
    ]
    loss_handles = [
        Line2D([0], [0], marker=mk, color="#475569", linestyle="",
               markersize=9, label=loss)
        for loss, mk in _LOSS_MARKER.items()
    ]
    leg_model = ax.legend(handles=model_handles, title="Architecture",
                          loc="lower left", fontsize=9)
    ax.add_artist(leg_model)
    ax.legend(handles=loss_handles, title="Loss", loc="lower right", fontsize=9)

    ax.set_xlabel("Test accuracy", fontsize=12)
    ax.set_ylabel("Melanoma sensitivity (recall)", fontsize=12)
    ax.set_title(
        "Accuracy vs melanoma sensitivity across 9 experiments",
        fontsize=13, pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {_OUT}")


if __name__ == "__main__":
    main()
