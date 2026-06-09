"""Training entry point.

Usage:
    uv run python -m src.train --config configs/resnet18.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data import SkinLesionDataset, build_transforms, lesion_aware_split
from src.data.dataset import CLASS_TO_IDX, CLASSES
from src.evaluate import evaluate
from src.losses import build_loss
from src.models import build_model

_DATA_DIR = Path("data")
_METADATA = _DATA_DIR / "HAM10000_metadata.csv"
_IMAGE_DIR = _DATA_DIR / "HAM10000_images"
_CHECKPOINT_DIR = Path("checkpoints")

_NUM_CLASSES = len(CLASSES)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _balanced_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    weights = compute_class_weight("balanced", classes=np.arange(_NUM_CLASSES), y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(config_path: str | Path) -> None:
    cfg = _load_config(config_path)
    seed = cfg["train"]["seed"]
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ── Data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(_METADATA)
    train_df, val_df, _ = lesion_aware_split(df, seed=seed)

    image_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]

    train_ds = SkinLesionDataset(train_df, _IMAGE_DIR, build_transforms(image_size, train=True))
    val_ds = SkinLesionDataset(val_df, _IMAGE_DIR, build_transforms(image_size, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_name = cfg["loss"]["name"]
    class_weight = None
    if loss_name in ("weighted_ce", "focal"):
        train_labels = train_df["dx"].map(CLASS_TO_IDX).to_numpy()
        class_weight = _balanced_weights(train_labels, device)

    loss_fn = build_loss(
        loss_name, weight=class_weight, gamma=cfg["loss"].get("focal_gamma", 2.0)
    )

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    num_epochs = cfg["train"]["num_epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    _CHECKPOINT_DIR.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.log_params({
            "model": cfg["model"]["name"],
            "loss": loss_name,
            "lr": cfg["train"]["learning_rate"],
            "weight_decay": cfg["train"]["weight_decay"],
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "seed": seed,
        })

        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # Train ────────────────────────────────────────────────────────────
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_ds)
            scheduler.step()

            # Validate ─────────────────────────────────────────────────────────
            metrics = evaluate(model, val_loader, device)
            val_acc = metrics["accuracy"]

            log_metrics: dict[str, float] = {
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": metrics["macro_f1"],
                "val_macro_auc": metrics["macro_auc"],
            }
            log_metrics.update({
                f"val_sensitivity_{cls}": v
                for cls, v in metrics["per_class_sensitivity"].items()
            })
            mlflow.log_metrics(log_metrics, step=epoch)

            print(
                f"[{epoch:3d}/{num_epochs}] "
                f"loss={train_loss:.4f}  "
                f"acc={val_acc:.4f}  "
                f"f1={metrics['macro_f1']:.4f}  "
                f"auc={metrics['macro_auc']:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = _CHECKPOINT_DIR / f"{cfg['mlflow']['run_name']}_best.pt"
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(str(ckpt))

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a skin lesion classifier.")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    args = parser.parse_args()
    train(args.config)
