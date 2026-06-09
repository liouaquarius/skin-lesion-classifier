"""PyTorch dataset for HAM10000 skin lesion images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Canonical class order -> stable label indices, shared across train / eval /
# inference. Alphabetical so the mapping never depends on data ordering.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


class SkinLesionDataset(Dataset):
    """HAM10000 images keyed by a metadata table.

    Args:
        metadata: a path to ``HAM10000_metadata.csv`` or an in-memory
            DataFrame (e.g. a split produced by ``lesion_aware_split``).
        image_dir: directory containing ``<image_id>.jpg`` files.
        transform: optional torchvision transform applied to each PIL image.
    """

    def __init__(self, metadata: str | Path | pd.DataFrame, image_dir: str | Path, transform=None):
        if isinstance(metadata, (str, Path)):
            metadata = pd.read_csv(metadata)
        self.df = metadata.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / f"{row['image_id']}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = CLASS_TO_IDX[row["dx"]]
        return image, label
