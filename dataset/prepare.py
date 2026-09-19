"""Cleaning and feature engineering.

Two stages:

1. `clean()` removes the data-quality problems that `eda.quality_findings()`
   identified, recording a before/after count for every step so the cleaning is
   auditable rather than a black box.

2. `build_customer_features()` collapses roughly a million transaction lines
   into one row per customer using RFM features (recency, frequency, monetary)
   plus a supervised label. That aggregation is what keeps this affordable on a
   small container: the models train on thousands of rows, not millions.

The label is "will this customer purchase again in the holdout window". It is
derived by splitting the timeline: features come from before the cutoff, the
label from after it. That avoids leaking the answer into the inputs, which is
the easiest way to accidentally produce an implausibly perfect model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DIR / "customer_features.csv.gz"

# The label window: customers are judged on whether they returned in the final
# 90 days of the dataset.
HOLDOUT_DAYS = 90


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the cleaning steps, returning the result and an audit trail."""
    steps: list[dict] = []
    working = frame.copy()
    start_rows = len(working)

    def record(step: str, before: int, reason: str) -> None:
        after = len(working)
        steps.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": before - after,
                "reason": reason,
            }
        )

    # Types first: everything else depends on these being real numbers/dates.
    working["quantity"] = pd.to_numeric(working["quantity"], errors="coerce")
    working["unit_price"] = pd.to_numeric(working["unit_price"], errors="coerce")
    working["invoice_date"] = pd.to_datetime(working["invoice_date"], errors="coerce")

    before = len(working)
    working = working.drop_duplicates()
    record("Remove exact duplicate rows", before, "The same line counted twice.")

    before = len(working)
    working = working[working["invoice_date"].notna()]
    record("Drop unparseable invoice dates", before, "A transaction with no usable date "
                                                     "cannot be placed on the timeline.")

    if "invoice" in working.columns:
        before = len(working)
        is_cancellation = (
            working["invoice"].astype(str).str.upper().str.startswith("C")
        )
        working = working[~is_cancellation]
        record("Drop cancellation invoices", before,
               "Invoices prefixed 'C' are cancelled sales, not purchases.")

    before = len(working)
    working = working[working["quantity"] > 0]
    record("Drop non-positive quantity", before,
           "Returns and stock adjustments are not sales.")

    before = len(working)
    working = working[working["unit_price"] > 0]
    record("Drop zero or negative unit price", before,
           "Free lines and credits distort monetary value.")

    if "description" in working.columns:
        filled = int(working["description"].isna().sum())
        working["description"] = working["description"].fillna("Unknown")
        steps.append(
            {
                "step": "Fill missing product descriptions",
                "rows_before": len(working),
                "rows_after": len(working),
                "rows_removed": 0,
                "values_imputed": filled,
                "reason": "Imputed with 'Unknown' rather than dropped - the "
                          "transaction is still valid without a description.",
            }
        )

    before = len(working)
    working = working[working["customer_id"].notna()]
    record("Drop rows with no customer identifier", before,
           "Cannot be attributed to a customer, so cannot form customer features.")

    working["customer_id"] = working["customer_id"].astype("int64")

    # Derived money column, used by every monetary feature downstream.
    working["line_total"] = working["quantity"] * working["unit_price"]

    steps.append(
        {
            "step": "TOTAL",
            "rows_before": start_rows,
            "rows_after": len(working),
            "rows_removed": start_rows - len(working),
            "retained_pct": round(len(working) / start_rows * 100, 2) if start_rows else 0.0,
            "reason": "Net effect of all cleaning steps.",
        }
    )

    return working.reset_index(drop=True), steps


def build_customer_features(clean_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregate transactions into one labelled row per customer.

    Features describe behaviour up to `cutoff`; the label describes what
    happened after it.
    """
    latest = clean_frame["invoice_date"].max()
    cutoff = latest - pd.Timedelta(days=HOLDOUT_DAYS)

    history = clean_frame[clean_frame["invoice_date"] <= cutoff]
    future = clean_frame[clean_frame["invoice_date"] > cutoff]

    if history.empty:
        raise ValueError("no transactions before the cutoff; dataset span too short")

    grouped = history.groupby("customer_id")

    features = pd.DataFrame(
        {
            # Recency: days since the customer's last purchase, as at the cutoff.
            "recency_days": (cutoff - grouped["invoice_date"].max()).dt.days,
            # Frequency: distinct invoices.
            "frequency": grouped["invoice"].nunique(),
            # Monetary: total and average spend.
            "monetary_total": grouped["line_total"].sum(),
            "monetary_avg_order": grouped["line_total"].sum()
            / grouped["invoice"].nunique(),
            "total_items": grouped["quantity"].sum(),
            "distinct_products": grouped["stock_code"].nunique(),
            "avg_unit_price": grouped["unit_price"].mean(),
            "tenure_days": (grouped["invoice_date"].max()
                            - grouped["invoice_date"].min()).dt.days,
        }
    )

    # Label: did this customer buy again during the holdout window?
    returners = set(future["customer_id"].unique())
    features["repurchased"] = features.index.isin(returners).astype(int)

    features = features.reset_index()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    positives = int(features["repurchased"].sum())
    meta = {
        "customers": int(len(features)),
        "feature_columns": [c for c in features.columns
                            if c not in {"customer_id", "repurchased"}],
        "label_column": "repurchased",
        "dataset_end": latest.isoformat(),
        "feature_window_end": cutoff.isoformat(),
        "holdout_days": HOLDOUT_DAYS,
        "positive_label_count": positives,
        "positive_label_pct": round(positives / len(features) * 100, 2) if len(features) else 0.0,
        "note": "Features are computed strictly before the cutoff and the label "
                "strictly after it, so no future information leaks into the inputs.",
    }
    return features, meta


def run_pipeline(limit: int | None = None) -> dict:
    """Load, clean, engineer and persist. Returns a report of what happened."""
    from dataset import eda, loader  # imported here to keep module import cheap

    raw = loader.load_raw(limit=limit)
    raw_profile = eda.overview(raw)
    findings = eda.quality_findings(raw)

    cleaned, steps = clean(raw)
    features, meta = build_customer_features(cleaned)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_FILE, index=False, compression="gzip")

    return {
        "raw": raw_profile,
        "quality_findings": findings,
        "cleaning_steps": steps,
        "features": meta,
        "output_file": str(FEATURES_FILE.relative_to(ROOT)),
    }


def load_features() -> pd.DataFrame:
    """Read the prepared feature table."""
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            "No prepared features. POST /api/data/prepare first "
            "(or run scripts/prepare_data.py)."
        )
    return pd.read_csv(FEATURES_FILE, compression="gzip")
