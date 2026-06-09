"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import CLASSES


@pytest.fixture
def synthetic_data(tmp_path):
    """A tiny HAM10000-like dataset: random JPGs + a metadata DataFrame.

    Returns ``(df, image_dir)``. Some lesions have multiple images so the
    lesion-aware split can be exercised. No real dataset is required.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    n_lesions = 20
    for i in range(n_lesions):
        lesion_id = f"L{i:03d}"
        dx = CLASSES[i % len(CLASSES)]
        for j in range(1 + i % 2):  # 1-2 images per lesion
            image_id = f"img_{i:03d}_{j}"
            arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(image_dir / f"{image_id}.jpg")
            rows.append({"lesion_id": lesion_id, "image_id": image_id, "dx": dx})

    return pd.DataFrame(rows), image_dir
