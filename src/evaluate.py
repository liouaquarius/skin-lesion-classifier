"""Evaluation metrics for multi-class skin lesion classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import CLASSES

_NUM_CLASSES = len(CLASSES)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict:
    """Run inference over *dataloader* and return a metrics dict.

    Keys: ``accuracy``, ``per_class_sensitivity``, ``macro_f1``,
    ``macro_auc``, ``confusion_matrix``.
    """
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=-1).cpu()
            all_labels.append(labels)
            all_preds.append(probs.argmax(dim=-1))
            all_probs.append(probs)

    labels_np = torch.cat(all_labels).numpy()
    preds_np = torch.cat(all_preds).numpy()
    probs_np = torch.cat(all_probs).numpy()

    cm = confusion_matrix(labels_np, preds_np, labels=list(range(_NUM_CLASSES)))

    # Per-class sensitivity = TP / (TP + FN)
    per_class_sensitivity: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum()) - tp
        per_class_sensitivity[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        macro_auc = float(
            roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
        )
    except ValueError:
        macro_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "per_class_sensitivity": per_class_sensitivity,
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "macro_auc": macro_auc,
        "confusion_matrix": cm.tolist(),
    }
