"""Run the full train -> test-bake pipeline for one or more configs.

Each config is processed in an isolated subprocess, so a failure (e.g. a CUDA
OOM on the ViT runs) is contained: that config is reported and skipped while the
remaining configs keep going. The checkpoint path is derived from the config's
``mlflow.run_name``, so train and bake always stay aligned.

Usage:
    # single config
    uv run python scripts/run_experiment.py configs/resnet18_ce.yaml

    # several configs, run in order
    uv run python scripts/run_experiment.py configs/resnet18_ce.yaml configs/vit_tiny_ce.yaml

    # all configs (PowerShell glob expansion)
    uv run python scripts/run_experiment.py (Get-ChildItem configs/*.yaml).FullName
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_CHECKPOINT_DIR = Path("results/checkpoints")


def _run_name(config_path: Path) -> str:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["mlflow"]["run_name"]


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def run_one(config_path: Path) -> bool:
    """Train then bake a single config. Returns True only if both stages pass."""
    run_name = _run_name(config_path)
    checkpoint = _CHECKPOINT_DIR / f"{run_name}_best.pt"

    print(f"\n{'=' * 70}\n  {run_name}  ({config_path})\n{'=' * 70}")

    print(f"\n[1/2] TRAIN  {run_name}")
    if _run([sys.executable, "-m", "src.train", "--config", str(config_path)]) != 0:
        print(f"!! TRAIN FAILED: {run_name} -- skipping bake")
        return False

    print(f"\n[2/2] BAKE   {run_name}")
    if _run([
        sys.executable, "scripts/bake_results.py",
        "--config", str(config_path), "--checkpoint", str(checkpoint),
    ]) != 0:
        print(f"!! BAKE FAILED: {run_name}")
        return False

    print(f"\n++ DONE: {run_name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run train + test-bake for one or more configs."
    )
    parser.add_argument(
        "configs", nargs="+", type=Path, help="One or more training YAML configs"
    )
    args = parser.parse_args()

    results: dict[str, bool] = {str(cfg): run_one(cfg) for cfg in args.configs}

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    for cfg, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {cfg}")

    # Confirm each config produced its full set of artifacts (mlflow run + the
    # four result files). Reuses the same check as `inspect_runs.py`.
    print(f"\n{'=' * 70}\n  VERIFY\n{'=' * 70}")
    from inspect_runs import client, verify_run

    c = client()
    complete = True
    for cfg in args.configs:
        ok, lines = verify_run(_run_name(Path(cfg)), c)
        complete = complete and ok
        print(f"\n{_run_name(Path(cfg))}  ->  {'COMPLETE' if ok else 'INCOMPLETE'}")
        print("\n".join(lines))

    pipeline_ok = all(results.values()) and complete
    return 0 if pipeline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
