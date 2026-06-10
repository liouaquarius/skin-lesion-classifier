"""Inspect and clean up experiment results (MLflow runs + on-disk artifacts).

The MLflow store is a SQLite db that can't be opened by hand, so this gives a
terminal view of what exists and whether each run is *complete* -- i.e. it has a
FINISHED MLflow run plus all four on-disk artifacts (checkpoint, test JSON, and
the two figures).

Create/Update of runs deliberately happen through training only (hand-editing
experiment records would break provenance), so this tool exposes just Read
(list / verify) and Delete (clean up failed or duplicate runs).

Usage:
    uv run python scripts/inspect_runs.py                         # list every run
    uv run python scripts/inspect_runs.py --run resnet18-ce-seed42
    uv run python scripts/inspect_runs.py --delete resnet18-ce-seed42 --status FAILED
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

_EXPERIMENT = "skin-lesion-classifier"
_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
_CKPT = Path("results/checkpoints")
_METRICS = Path("results/metrics")
_VIZ = Path("results/visualizations")


def expected_artifacts(run_name: str) -> dict[str, Path]:
    """The four on-disk files a fully-baked run should have."""
    return {
        "checkpoint": _CKPT / f"{run_name}_best.pt",
        "test.json": _METRICS / f"{run_name}_test.json",
        "confusion_png": _VIZ / f"{run_name}_confusion_matrix.png",
        "sensitivity_png": _VIZ / f"{run_name}_sensitivity.png",
    }


def client() -> MlflowClient:
    mlflow.set_tracking_uri(_TRACKING_URI)
    return MlflowClient()


def _runs_by_name(c: MlflowClient) -> dict[str, list]:
    exp = c.get_experiment_by_name(_EXPERIMENT)
    if exp is None:
        return {}
    out: dict[str, list] = {}
    for r in c.search_runs(exp.experiment_id):
        name = r.data.tags.get("mlflow.runName", "<unnamed>")
        out.setdefault(name, []).append(r)
    return out


def _disk_run_names() -> set[str]:
    """Run names inferred from result files (catches files with no MLflow run)."""
    names: set[str] = set()
    for p in _CKPT.glob("*_best.pt"):
        names.add(p.name[: -len("_best.pt")])
    for p in _METRICS.glob("*_test.json"):
        names.add(p.name[: -len("_test.json")])
    return names


def verify_run(run_name: str, c: MlflowClient | None = None) -> tuple[bool, list[str]]:
    """Return ``(complete, checklist_lines)`` for a single run."""
    c = c or client()
    runs = _runs_by_name(c).get(run_name, [])
    finished = [r for r in runs if r.info.status == "FINISHED"]

    mlflow_ok = len(finished) >= 1
    note = f"{len(finished)} FINISHED"
    if len(runs) > len(finished):
        note += f", {len(runs) - len(finished)} other"
    lines = [f"   [{'x' if mlflow_ok else ' '}] mlflow run ({note})"]

    ok = mlflow_ok
    for label, path in expected_artifacts(run_name).items():
        exists = path.exists()
        ok = ok and exists
        lines.append(f"   [{'x' if exists else ' '}] {label}: {path}")
    return ok, lines


def report(run_names: list[str], c: MlflowClient | None = None) -> bool:
    """Print a completeness checklist for each run. Returns True if all complete."""
    c = c or client()
    all_ok = True
    for name in run_names:
        ok, lines = verify_run(name, c)
        all_ok = all_ok and ok
        print(f"\n{name}  ->  {'COMPLETE' if ok else 'INCOMPLETE'}")
        print("\n".join(lines))
    return all_ok


def cmd_list(args: argparse.Namespace) -> int:
    c = client()
    names = sorted(set(_runs_by_name(c)) | _disk_run_names())
    if not names:
        print("No MLflow runs or result files found.")
        return 0
    all_ok = report(names, c)
    print(f"\n{'all runs complete' if all_ok else 'some runs INCOMPLETE'}")
    return 0 if all_ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    return 0 if report([args.run]) else 1


def cmd_delete(args: argparse.Namespace) -> int:
    c = client()
    runs = _runs_by_name(c).get(args.delete, [])
    if args.status:
        runs = [r for r in runs if r.info.status == args.status]
    if not runs:
        scope = f" with status={args.status}" if args.status else ""
        print(f"No runs named {args.delete!r}{scope}.")
        return 0
    for r in runs:
        c.delete_run(r.info.run_id)
        print(f"soft-deleted {args.delete} [{r.info.status}] {r.info.run_id}")
    print(
        "\nPurge permanently with:\n"
        f"  uv run mlflow gc --backend-store-uri {_TRACKING_URI}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", help="Verify a single run by name")
    group.add_argument("--delete", help="Soft-delete runs with this name")
    parser.add_argument(
        "--status", help="Restrict --delete to this status (e.g. FAILED)"
    )
    args = parser.parse_args()

    if args.delete:
        return cmd_delete(args)
    if args.run:
        return cmd_run(args)
    return cmd_list(args)


if __name__ == "__main__":
    raise SystemExit(main())
