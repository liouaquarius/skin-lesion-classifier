"""Smoke tests for the training configs.

These run in CI before any model code exists, validating the committed YAML
configs (schema + key invariants) rather than the not-yet-built ``src`` package.
"""

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))

REQUIRED_SECTIONS = ("model", "data", "train", "loss", "mlflow")
VALID_LOSSES = {"cross_entropy", "weighted_ce", "focal"}


def test_config_files_present():
    assert CONFIG_FILES, f"No config files found in {CONFIG_DIR}"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_schema(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for section in REQUIRED_SECTIONS:
        assert section in config, f"{config_path.name}: missing '{section}' section"

    assert config["model"]["num_classes"] == 7
    assert config["data"]["image_size"] > 0
    assert config["data"]["batch_size"] > 0
    assert config["loss"]["name"] in VALID_LOSSES
    assert config["train"]["seed"] == 42
