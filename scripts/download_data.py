"""Download and prepare the HAM10000 dataset from Kaggle.

Fetches ``kmader/skin-cancer-mnist-ham10000`` into ``data/``, then consolidates
the two ``*_images_part_*`` folders into a single ``data/HAM10000_images/``
directory expected by the rest of the pipeline.

Requires Kaggle API credentials (``~/.kaggle/kaggle.json`` or the
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables) and the ``kaggle``
package from the ``data`` extra.

Usage:
    uv run --extra data python scripts/download_data.py
    uv run --extra data python scripts/download_data.py --force  # re-download
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET = "kmader/skin-cancer-mnist-ham10000"
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "HAM10000_images"
METADATA = DATA_DIR / "HAM10000_metadata.csv"


def download() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} into {DATA_DIR}/ (~5.5 GB), please wait...")
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)


def consolidate_images() -> None:
    """Merge the split ``*_images_part_*`` folders into one image directory."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(p for p in DATA_DIR.glob("*images_part_*") if p.is_dir())
    for part in parts:
        for img in part.glob("*.jpg"):
            target = IMAGE_DIR / img.name
            if not target.exists():
                shutil.move(str(img), str(target))
        if not any(part.iterdir()):
            part.rmdir()


def verify() -> None:
    if not METADATA.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA}")
    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    if n_images == 0:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")
    print(f"Ready: {n_images} images in {IMAGE_DIR}, metadata at {METADATA}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already appears present.",
    )
    args = parser.parse_args()

    if METADATA.exists() and not args.force:
        print(f"{METADATA} already exists; skipping download (use --force to redo).")
    else:
        download()

    consolidate_images()
    verify()


if __name__ == "__main__":
    main()
