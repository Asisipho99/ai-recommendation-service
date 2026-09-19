"""Train, persist, compare and serve the models.

One place that knows how to train everything, where the artefacts live, and how
to load them back. The comparison table it produces is the main evidence that
several algorithms were applied to the dataset and measured against each other.

Artefacts are written atomically (temp file then rename) for the same reason the
recommender does it: a process killed mid-write must never leave a half-written
model that fails to deserialise on the next read.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from dataset import prepare
from ml import classification, clustering

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
BUNDLE_FILE = MODEL_DIR / "supervised_models.joblib"
CLUSTER_FILE = MODEL_DIR / "segmentation.joblib"
REPORT_FILE = MODEL_DIR / "training_report.json"


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        writer(handle)
        temp_name = handle.name
    os.replace(temp_name, path)


def train_all() -> dict:
    """Run the whole training pass over the prepared features."""
    features = prepare.load_features()

    fitted, results, split_info = classification.train_all(features)
    cluster_pipeline, cluster_report = clustering.segment(features)

    _atomic_write(
        BUNDLE_FILE,
        lambda fh: joblib.dump(
            {"models": fitted, "feature_columns": split_info["feature_columns"]}, fh
        ),
    )
    _atomic_write(CLUSTER_FILE, lambda fh: joblib.dump(cluster_pipeline, fh))

    comparison = [
        {
            "algorithm": result.name,
            "family": result.family,
            "task": "Binary classification - repurchase prediction",
            **result.metrics,
        }
        for result in results
    ]
    # Clustering is scored on a different scale, so it is listed separately
    # rather than being forced into the same accuracy column.
    best = max(
        comparison,
        key=lambda row: (row.get("roc_auc") or 0, row["f1"]),
    )

    report = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "rows_used": int(len(features)),
            "source": "Online Retail II (UCI / Kaggle), prepared by dataset/prepare.py",
        },
        "split": split_info,
        "supervised_comparison": comparison,
        "best_supervised": {
            "algorithm": best["algorithm"],
            "chosen_by": "highest ROC-AUC, then F1",
        },
        "feature_importance": {
            result.name: result.feature_importance for result in results
        },
        "segmentation": cluster_report,
        "algorithm_count": len(comparison) + 1,
        "algorithms": [row["algorithm"] for row in comparison]
        + [cluster_report["algorithm"]],
    }

    _atomic_write(
        REPORT_FILE,
        lambda fh: fh.write(json.dumps(report, indent=2).encode("utf-8")),
    )
    return report


def load_report() -> dict | None:
    if not REPORT_FILE.exists():
        return None
    try:
        return json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_supervised() -> tuple[dict[str, Any], list[str]] | tuple[None, None]:
    if not BUNDLE_FILE.exists():
        return None, None
    try:
        bundle = joblib.load(BUNDLE_FILE)
        return bundle["models"], bundle["feature_columns"]
    except Exception:  # noqa: BLE001 - corrupt artefact, remove and retrain
        BUNDLE_FILE.unlink(missing_ok=True)
        return None, None


def load_segmentation():
    if not CLUSTER_FILE.exists():
        return None
    try:
        return joblib.load(CLUSTER_FILE)
    except Exception:  # noqa: BLE001
        CLUSTER_FILE.unlink(missing_ok=True)
        return None


def status() -> dict:
    """What is trained and ready, for the UI to show before offering a prediction."""
    models, feature_columns = load_supervised()
    report = load_report()

    return {
        "supervised_trained": models is not None,
        "available_models": sorted(models) if models else [],
        "feature_columns": feature_columns or [],
        "segmentation_trained": load_segmentation() is not None,
        "last_trained": report.get("trained_at") if report else None,
        "algorithm_count": report.get("algorithm_count") if report else 0,
    }


def predict(model_name: str, values: dict) -> dict:
    models, feature_columns = load_supervised()
    if not models:
        raise FileNotFoundError("no trained models; POST /api/ml/train first")
    if model_name not in models:
        raise KeyError(f"unknown model '{model_name}'; available: {sorted(models)}")

    result = classification.predict_one(models[model_name], feature_columns, values)
    result["model"] = model_name

    segmentation = load_segmentation()
    if segmentation is not None:
        try:
            result["segment"] = clustering.assign(segmentation, values)["cluster"]
        except ValueError:
            result["segment"] = None

    return result
