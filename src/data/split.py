"""Lesion-aware train / val / test split.

Multiple images can share a ``lesion_id``; splitting by group prevents the same
physical lesion from leaking across splits (an EDA finding).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def lesion_aware_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, val, test) with no ``lesion_id`` overlap."""
    groups = df["lesion_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Express val as a fraction of the remaining train+val pool.
    rel_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(gss_val.split(trainval, groups=trainval["lesion_id"]))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)
    return train, val, test
