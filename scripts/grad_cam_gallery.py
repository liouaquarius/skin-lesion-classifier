"""Generate a Grad-CAM gallery: confident-correct vs confident-wrong cases.

Scans the held-out test split with the deployed model, then renders a 2x3 grid:
top row  = high-confidence **correct** predictions (class-diverse),
bottom row = high-confidence **misclassifications** (the most instructive
failures). Each panel shows the Grad-CAM overlay with true / predicted labels
and confidence. Saves ``results/visualizations/grad_cam_gallery.png``.

Usage:
    uv run python scripts/grad_cam_gallery.py --config configs/vit_tiny_wce.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from PIL import Image

from src.data import lesion_aware_split
from src.inference import Predictor

_DATA = Path("data")
_META = _DATA / "HAM10000_metadata.csv"
_IMG = _DATA / "HAM10000_images"
_CKPT = Path("results/checkpoints")
_OUT = Path("results/visualizations/grad_cam_gallery.png")


def _decode(data_uri: str) -> Image.Image:
    """Decode an ``explain()`` data-URI back into a PIL image."""
    b64 = data_uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _scan(predictor: Predictor, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in test_df.itertuples():
        out = predictor.predict(Image.open(_IMG / f"{r.image_id}.jpg").convert("RGB"))
        rows.append({
            "image_id": r.image_id, "true": r.dx,
            "pred": out["predicted_class"], "conf": out["confidence"],
        })
    return pd.DataFrame(rows)


def _pick(scan_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct = scan_df[scan_df.true == scan_df.pred].sort_values("conf", ascending=False)
    wrong = scan_df[scan_df.true != scan_df.pred].sort_values("conf", ascending=False)
    # class-diverse correct picks; most-confident wrong picks
    return correct.drop_duplicates("true").head(3), wrong.head(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/vit_tiny_wce.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run_name = cfg["mlflow"]["run_name"]
    ckpt = _CKPT / f"{run_name}_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {run_name}  |  device: {device}")

    predictor = Predictor(ckpt, cfg["model"]["name"], device=device)
    _, _, test_df = lesion_aware_split(pd.read_csv(_META), seed=cfg["train"]["seed"])

    scan_df = _scan(predictor, test_df)
    correct_sel, wrong_sel = _pick(scan_df)

    fig, axes = plt.subplots(2, 3, figsize=(11, 8.4), layout="constrained")
    for row, sel in enumerate([correct_sel, wrong_sel]):
        ok = row == 0
        color = "#16a34a" if ok else "#dc2626"
        for col, r in enumerate(sel.itertuples()):
            ax = axes[row, col]
            overlay = _decode(
                predictor.explain(Image.open(_IMG / f"{r.image_id}.jpg").convert("RGB"))[
                    "grad_cam_image"
                ]
            )
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"true: {r.true}  /  pred: {r.pred}\nconf {r.conf:.2f}",
                fontsize=10, color=color,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
        axes[row, 0].set_ylabel(
            "Correct" if ok else "Misclassified", fontsize=12, color=color, fontweight="bold"
        )

    fig.suptitle(
        f"Grad-CAM — {run_name}  (top: confident correct · bottom: confident errors)",
        fontsize=13,
    )
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {_OUT}")


if __name__ == "__main__":
    main()
