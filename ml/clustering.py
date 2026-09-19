"""Unsupervised customer segmentation with K-Means over RFM features.

Uses the same prepared feature table as the classifiers but ignores the label,
which is what makes it genuinely unsupervised rather than a relabelled
classifier.

Two details worth noting:

* RFM values are heavily right-skewed - a handful of customers spend orders of
  magnitude more than the rest - so a log transform is applied before scaling.
  Without it K-Means chases the outliers and produces one enormous cluster plus
  several singletons.
* `k` is chosen by silhouette score across a range rather than assumed, and the
  scores for every candidate are reported so the choice is visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RANDOM_STATE = 42
RFM_COLUMNS = ["recency_days", "frequency", "monetary_total"]
CANDIDATE_K = [2, 3, 4, 5, 6]


def _build_pipeline(k: int) -> Pipeline:
    return Pipeline(
        [
            # log1p handles the skew and tolerates zeros.
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)),
        ]
    )


def segment(features: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Fit K-Means, choosing k by silhouette score."""
    available = [c for c in RFM_COLUMNS if c in features.columns]
    if len(available) < 2:
        raise ValueError(f"need at least two RFM columns, found {available}")

    X = features[available].to_numpy(dtype=float)
    # Clip negatives so log1p is defined; recency can be 0 but never negative.
    X = np.clip(X, 0, None)

    scores = []
    for k in CANDIDATE_K:
        if k >= len(X):
            continue
        pipeline = _build_pipeline(k)
        labels = pipeline.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        transformed = pipeline[:-1].transform(X)
        scores.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(transformed, labels)), 4),
                "inertia": round(float(pipeline.named_steps["model"].inertia_), 2),
            }
        )

    if not scores:
        raise ValueError("could not fit any clustering; too few customers")

    best = max(scores, key=lambda s: s["silhouette"])
    pipeline = _build_pipeline(best["k"])
    labels = pipeline.fit_predict(X)

    profile = _profile_clusters(features, available, labels)

    report = {
        "algorithm": "K-Means",
        "family": "Unsupervised - clustering",
        "features_used": available,
        "preprocessing": "log1p then standard scaling, to control RFM skew",
        "chosen_k": best["k"],
        "selection_metric": "silhouette",
        "candidate_scores": scores,
        "metrics": {
            "silhouette": best["silhouette"],
            "inertia": best["inertia"],
        },
        "clusters": profile,
    }
    return pipeline, report


def _profile_clusters(
    features: pd.DataFrame, columns: list[str], labels: np.ndarray
) -> list[dict]:
    """Describe each cluster and give it a readable name."""
    frame = features.copy()
    frame["cluster"] = labels

    overall_monetary = frame["monetary_total"].median()
    overall_recency = frame["recency_days"].median()

    profiles = []
    for cluster_id, group in frame.groupby("cluster"):
        summary = {
            "cluster": int(cluster_id),
            "customers": int(len(group)),
            "share_pct": round(len(group) / len(frame) * 100, 2),
        }
        for column in columns:
            summary[f"median_{column}"] = round(float(group[column].median()), 2)

        if "repurchased" in group.columns:
            summary["repurchase_rate_pct"] = round(
                float(group["repurchased"].mean()) * 100, 2
            )

        high_value = group["monetary_total"].median() >= overall_monetary
        recent = group["recency_days"].median() <= overall_recency

        if high_value and recent:
            summary["label"] = "Champions - recent and high spend"
        elif high_value and not recent:
            summary["label"] = "At risk - valuable but lapsing"
        elif not high_value and recent:
            summary["label"] = "New or low spend - recently active"
        else:
            summary["label"] = "Dormant - low spend, long inactive"

        profiles.append(summary)

    return sorted(profiles, key=lambda p: p["cluster"])


def assign(pipeline: Pipeline, values: dict) -> dict:
    """Place a single hand-entered customer into a segment."""
    missing = [c for c in RFM_COLUMNS if c not in values]
    if missing:
        raise ValueError(f"missing features: {missing}")

    row = np.clip(
        np.array([[float(values[c]) for c in RFM_COLUMNS]], dtype=float), 0, None
    )
    return {"cluster": int(pipeline.predict(row)[0])}
