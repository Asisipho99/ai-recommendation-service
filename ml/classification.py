"""Supervised models predicting whether a customer will purchase again.

Three algorithms are trained on identical splits so their scores are directly
comparable - which is the point, since comparing algorithms is more informative
than reporting one number for one model.

    Logistic Regression  linear, interpretable baseline
    Random Forest        non-linear ensemble
    MLP                  feed-forward neural network (deep learning)

Every model is wrapped in a Pipeline with a StandardScaler. That matters for
logistic regression and the MLP, which are sensitive to feature scale, and
keeps the scaler bound to the model so a prediction later cannot accidentally
skip it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class ModelResult:
    name: str
    family: str
    metrics: dict[str, Any]
    feature_importance: list[dict] = field(default_factory=list)


def _build_models() -> dict[str, tuple[str, Any]]:
    """Name -> (family, estimator). Pipelines include scaling."""
    return {
        "Logistic Regression": (
            "Supervised - linear",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        "Random Forest": (
            "Supervised - ensemble",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=12,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        "Neural Network (MLP)": (
            "Deep learning",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            solver="adam",
                            max_iter=600,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
    else:
        result["roc_auc"] = None
    return result


def train_all(features: pd.DataFrame) -> tuple[dict[str, Any], list[ModelResult], dict]:
    """Train every classifier on the same split.

    Returns the fitted estimators, their results, and a description of the split.
    """
    label = "repurchased"
    feature_columns = [c for c in features.columns if c not in {"customer_id", label}]

    X = features[feature_columns].to_numpy(dtype=float)
    y = features[label].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "the label has only one class, so no classifier can be trained; "
            "widen the holdout window"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    split_info = {
        "feature_columns": feature_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "train_positive_pct": round(float(y_train.mean()) * 100, 2),
        "test_positive_pct": round(float(y_test.mean()) * 100, 2),
        "stratified": True,
        "random_state": RANDOM_STATE,
    }

    fitted: dict[str, Any] = {}
    results: list[ModelResult] = []

    for name, (family, estimator) in _build_models().items():
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        y_proba = (
            estimator.predict_proba(X_test)[:, 1]
            if hasattr(estimator[-1], "predict_proba")
            else None
        )

        importance: list[dict] = []
        final = estimator.named_steps["model"]
        if hasattr(final, "feature_importances_"):
            importance = [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in sorted(
                    zip(feature_columns, final.feature_importances_),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
            ]
        elif hasattr(final, "coef_"):
            importance = [
                {"feature": f, "coefficient": round(float(v), 4)}
                for f, v in sorted(
                    zip(feature_columns, final.coef_[0]),
                    key=lambda pair: abs(pair[1]),
                    reverse=True,
                )
            ]

        fitted[name] = estimator
        results.append(
            ModelResult(
                name=name,
                family=family,
                metrics=_metrics(y_test, y_pred, y_proba),
                feature_importance=importance,
            )
        )

    return fitted, results, split_info


def predict_one(estimator: Any, feature_columns: list[str], values: dict) -> dict:
    """Score a single hand-entered customer, for the UI's prediction form."""
    missing = [c for c in feature_columns if c not in values]
    if missing:
        raise ValueError(f"missing features: {missing}")

    row = np.array([[float(values[c]) for c in feature_columns]], dtype=float)
    label = int(estimator.predict(row)[0])

    probability = None
    if hasattr(estimator[-1], "predict_proba"):
        probability = round(float(estimator.predict_proba(row)[0, 1]), 4)

    return {
        "prediction": label,
        "meaning": "likely to purchase again" if label == 1
        else "unlikely to purchase again",
        "probability_repurchase": probability,
    }
