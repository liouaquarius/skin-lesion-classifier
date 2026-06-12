"""Build static dashboard data for the frontend Results view.

Aggregates the 9 per-run test metrics into ``frontend/public/results_summary.json``
and copies the figures the dashboard displays into ``frontend/public/results/``,
so the Results page is fully static (works without the backend running).

Usage:
    uv run python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

from src.data.dataset import CLASSES

_METRICS = Path("results/metrics")
_VIZ = Path("results/visualizations")
_PUBLIC = Path("frontend/public")
_PUBLIC_RESULTS = _PUBLIC / "results"
_DEPLOYED = "vit_tiny-wce-seed42"

_ORDER_M = {"resnet18": 0, "efficientnet_b0": 1, "vit_tiny": 2}
_ORDER_L = {"ce": 0, "wce": 1, "focal": 2}
_AGGREGATE_FIGS = ["model_comparison.png", "grad_cam_gallery.png"]


def main() -> None:
    _PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)

    models = []
    for f in glob.glob(str(_METRICS / "*_test.json")):
        j = json.load(open(f, encoding="utf-8"))
        run_name = j["run_name"]
        model, loss, _ = run_name.split("-")
        cm_png = f"{run_name}_confusion_matrix.png"
        sens_png = f"{run_name}_sensitivity.png"
        models.append({
            "run_name": run_name,
            "model": model,
            "loss": loss,
            "accuracy": j["accuracy"],
            "macro_f1": j["macro_f1"],
            "macro_auc": j["macro_auc"],
            "per_class_sensitivity": j["per_class_sensitivity"],
            "confusion_matrix": j["confusion_matrix"],
            "confusion_png": f"results/{cm_png}",
            "sensitivity_png": f"results/{sens_png}",
        })
        for png in (cm_png, sens_png):
            src = _VIZ / png
            if src.exists():
                shutil.copy2(src, _PUBLIC_RESULTS / png)

    models.sort(key=lambda m: (_ORDER_M[m["model"]], _ORDER_L[m["loss"]]))

    for fig in _AGGREGATE_FIGS:
        src = _VIZ / fig
        if src.exists():
            shutil.copy2(src, _PUBLIC_RESULTS / fig)

    summary = {
        "classes": CLASSES,
        "deployed": _DEPLOYED,
        "aggregate_figures": _AGGREGATE_FIGS,
        "models": models,
    }
    out = _PUBLIC / "results_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out}  ({len(models)} models)")
    print(f"Copied figures to {_PUBLIC_RESULTS}/")


if __name__ == "__main__":
    main()
