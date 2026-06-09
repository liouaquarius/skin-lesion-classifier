"""Post-training evaluation script.

Runs the held-out test split against a trained checkpoint and writes:
  results/metrics/<run_name>_test.json
  results/visualizations/<run_name>_confusion_matrix.png
  results/visualizations/<run_name>_sensitivity.png

Usage:
    uv run python scripts/bake_results.py \
        --config  configs/resnet18.yaml \
        --checkpoint results/checkpoints/resnet18-ce-seed42_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import SkinLesionDataset, build_transforms, lesion_aware_split
from src.data.dataset import CLASSES
from src.evaluate import evaluate
from src.models import build_model

_DATA_DIR = Path("data")
_METADATA = _DATA_DIR / "HAM10000_metadata.csv"
_IMAGE_DIR = _DATA_DIR / "HAM10000_images"
_METRICS_DIR = Path("results/metrics")
_VIZ_DIR = Path("results/visualizations")

_NUM_CLASSES = len(CLASSES)
_CLASS_LABELS = [
    "Actinic\nKeratosis",
    "Basal Cell\nCarcinoma",
    "Benign\nKeratosis",
    "Dermato-\nfibroma",
    "Melanoma",
    "Melanocytic\nNevi",
    "Vascular\nLesion",
]


def _plot_confusion_matrix(cm: list[list[int]], run_name: str) -> None:
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=_CLASS_LABELS,
        yticklabels=_CLASS_LABELS,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Normalised Confusion Matrix — {run_name}", fontsize=13, pad=12)
    plt.tight_layout()
    out = _VIZ_DIR / f"{run_name}_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_sensitivity(sensitivity: dict[str, float], run_name: str) -> None:
    classes = list(sensitivity.keys())
    values = [sensitivity[c] for c in classes]
    labels = [_CLASS_LABELS[CLASSES.index(c)] for c in classes]
    colors = ["#3b82f6" if v >= 0.7 else "#f59e0b" if v >= 0.4 else "#ef4444" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Sensitivity (Recall)", fontsize=11)
    ax.set_title(f"Per-class Sensitivity — {run_name}", fontsize=13, pad=10)
    ax.axvline(0.7, color="#64748b", linestyle="--", linewidth=0.8, label="0.70 target")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    out = _VIZ_DIR / f"{run_name}_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def bake(config_path: str, checkpoint_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name: str = cfg["mlflow"]["run_name"]
    seed: int = cfg["train"]["seed"]
    model_name: str = cfg["model"]["name"]
    image_size: int = cfg["data"]["image_size"]
    batch_size: int = cfg["data"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {model_name}  |  Run: {run_name}")

    # ── Test split (same seed as training) ────────────────────────────────────
    df = pd.read_csv(_METADATA)
    _, _, test_df = lesion_aware_split(df, seed=seed)
    test_ds = SkinLesionDataset(test_df, _IMAGE_DIR, build_transforms(image_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    model = build_model(model_name, num_classes=_NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device)

    print(f"\n  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    print("  Per-class sensitivity:")
    for cls, s in metrics["per_class_sensitivity"].items():
        print(f"    {cls}: {s:.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = _METRICS_DIR / f"{run_name}_test.json"
    with open(out_json, "w") as f:
        json.dump({"run_name": run_name, **metrics}, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # ── Save visualizations ───────────────────────────────────────────────────
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(metrics["confusion_matrix"], run_name)
    _plot_sensitivity(metrics["per_class_sensitivity"], run_name)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set.")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()
    bake(args.config, args.checkpoint)
