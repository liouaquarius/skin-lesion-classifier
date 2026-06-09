"""Tests for the data pipeline: dataset, transforms, and lesion-aware split."""

from __future__ import annotations

from src.data import SkinLesionDataset, build_transforms, lesion_aware_split
from src.data.dataset import CLASSES


def test_dataset_len_matches_metadata(synthetic_data):
    df, image_dir = synthetic_data
    ds = SkinLesionDataset(df, image_dir, transform=build_transforms(train=False))
    assert len(ds) == len(df)


def test_getitem_shape_and_label(synthetic_data):
    df, image_dir = synthetic_data
    ds = SkinLesionDataset(df, image_dir, transform=build_transforms(train=False))
    image, label = ds[0]
    assert image.shape == (3, 224, 224)
    assert 0 <= label < len(CLASSES)


def test_train_transform_output_shape(synthetic_data):
    df, image_dir = synthetic_data
    ds = SkinLesionDataset(df, image_dir, transform=build_transforms(train=True))
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_lesion_aware_split_no_leakage(synthetic_data):
    df, _ = synthetic_data
    train, val, test = lesion_aware_split(df, seed=42)

    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    groups = [set(part["lesion_id"]) for part in (train, val, test)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
