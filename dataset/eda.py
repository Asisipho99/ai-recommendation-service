"""Exploratory data analysis over the raw external dataset.

Everything here is read-only profiling: it describes the data as downloaded,
before any cleaning. The point is to make the data-quality problems visible and
quantified, so the cleaning decisions in prepare.py can be justified rather
than asserted.

Returned structures are plain dicts and lists so they serialise straight to
JSON for the API and the admin UI.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _clean_number(value: Any) -> Any:
    """JSON cannot carry NaN or infinity, so normalise them to None."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        if math.isnan(as_float) or math.isinf(as_float):
            return None
        return round(as_float, 4)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def overview(frame: pd.DataFrame) -> dict:
    """Shape, memory, duplicates and the column inventory."""
    duplicate_rows = int(frame.duplicated().sum())

    return {
        "source_file": frame.attrs.get("source_file"),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(c) for c in frame.columns],
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / len(frame) * 100, 2) if len(frame) else 0.0,
        "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
    }


def missing_values(frame: pd.DataFrame) -> list[dict]:
    """Per-column null counts, worst first.

    This is the headline data-quality finding for this dataset: a large share of
    rows carry no customer identifier, which makes them useless for any
    per-customer model and forces an explicit decision in cleaning.
    """
    total = len(frame)
    rows = []
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        rows.append(
            {
                "column": str(column),
                "dtype": str(frame[column].dtype),
                "missing": missing,
                "missing_pct": round(missing / total * 100, 2) if total else 0.0,
                "present": total - missing,
                "unique": int(frame[column].nunique(dropna=True)),
            }
        )
    return sorted(rows, key=lambda r: r["missing"], reverse=True)


def numeric_summary(frame: pd.DataFrame) -> list[dict]:
    """Descriptive statistics for the numeric columns."""
    summary = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        summary.append(
            {
                "column": str(column),
                "count": int(series.count()),
                "mean": _clean_number(series.mean()),
                "std": _clean_number(series.std()),
                "min": _clean_number(series.min()),
                "p25": _clean_number(series.quantile(0.25)),
                "median": _clean_number(series.median()),
                "p75": _clean_number(series.quantile(0.75)),
                "max": _clean_number(series.max()),
                "negative_values": int((series < 0).sum()),
                "zero_values": int((series == 0).sum()),
            }
        )
    return summary


def top_categories(frame: pd.DataFrame, column: str, limit: int = 10) -> list[dict]:
    """The most frequent values of a categorical column."""
    if column not in frame.columns:
        return []
    counts = frame[column].value_counts(dropna=True).head(limit)
    total = len(frame)
    return [
        {
            "value": str(index),
            "count": int(value),
            "pct": round(int(value) / total * 100, 2) if total else 0.0,
        }
        for index, value in counts.items()
    ]


def date_range(frame: pd.DataFrame, column: str = "invoice_date") -> dict:
    if column not in frame.columns:
        return {"available": False}

    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return {"available": False}

    span_days = int((dates.max() - dates.min()).days)
    return {
        "available": True,
        "earliest": dates.min().isoformat(),
        "latest": dates.max().isoformat(),
        "span_days": span_days,
        "unparseable": int(len(frame) - len(dates)),
    }


def distribution(frame: pd.DataFrame, column: str, bins: int = 20) -> dict:
    """A histogram of a numeric column, for charting in the UI."""
    if column not in frame.columns:
        return {"available": False, "reason": f"no column '{column}'"}

    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return {"available": False, "reason": "no numeric values"}

    # Clip to the 1st-99th percentile; this dataset has extreme outliers that
    # otherwise collapse the whole histogram into a single bar.
    low, high = series.quantile(0.01), series.quantile(0.99)
    clipped = series[(series >= low) & (series <= high)]
    if clipped.empty:
        clipped = series

    counts, edges = np.histogram(clipped, bins=bins)
    return {
        "available": True,
        "column": column,
        "clipped_to_percentile": [1, 99],
        "excluded_outliers": int(len(series) - len(clipped)),
        "bins": [
            {
                "from": _clean_number(edges[i]),
                "to": _clean_number(edges[i + 1]),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ],
    }


def quality_findings(frame: pd.DataFrame) -> list[dict]:
    """Named data-quality problems, each with a count and a stated consequence.

    This is the bridge from profiling to cleaning: every finding here maps to a
    step in prepare.py.
    """
    findings: list[dict] = []
    total = len(frame)

    if "customer_id" in frame.columns:
        missing = int(frame["customer_id"].isna().sum())
        if missing:
            findings.append(
                {
                    "issue": "Missing customer identifier",
                    "column": "customer_id",
                    "affected_rows": missing,
                    "affected_pct": round(missing / total * 100, 2),
                    "consequence": "Rows cannot be attributed to a customer, so they "
                                   "cannot contribute to per-customer features.",
                    "treatment": "Dropped during preparation.",
                }
            )

    if "description" in frame.columns:
        missing = int(frame["description"].isna().sum())
        if missing:
            findings.append(
                {
                    "issue": "Missing product description",
                    "column": "description",
                    "affected_rows": missing,
                    "affected_pct": round(missing / total * 100, 2),
                    "consequence": "Weakens any text-based product similarity.",
                    "treatment": "Filled with 'Unknown' rather than dropped, since "
                                 "the transaction itself is still valid.",
                }
            )

    if "invoice" in frame.columns:
        cancelled = int(
            frame["invoice"].astype(str).str.upper().str.startswith("C").sum()
        )
        if cancelled:
            findings.append(
                {
                    "issue": "Cancellation invoices",
                    "column": "invoice",
                    "affected_rows": cancelled,
                    "affected_pct": round(cancelled / total * 100, 2),
                    "consequence": "Cancelled sales would count as purchases and "
                                   "inflate every customer's spend.",
                    "treatment": "Dropped (invoice numbers prefixed with 'C').",
                }
            )

    if "quantity" in frame.columns:
        non_positive = int((pd.to_numeric(frame["quantity"], errors="coerce") <= 0).sum())
        if non_positive:
            findings.append(
                {
                    "issue": "Non-positive quantity",
                    "column": "quantity",
                    "affected_rows": non_positive,
                    "affected_pct": round(non_positive / total * 100, 2),
                    "consequence": "Returns and adjustments are not sales.",
                    "treatment": "Dropped.",
                }
            )

    if "unit_price" in frame.columns:
        non_positive = int(
            (pd.to_numeric(frame["unit_price"], errors="coerce") <= 0).sum()
        )
        if non_positive:
            findings.append(
                {
                    "issue": "Zero or negative unit price",
                    "column": "unit_price",
                    "affected_rows": non_positive,
                    "affected_pct": round(non_positive / total * 100, 2),
                    "consequence": "Free or adjustment lines distort monetary value.",
                    "treatment": "Dropped.",
                }
            )

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        findings.append(
            {
                "issue": "Exact duplicate rows",
                "column": "(all)",
                "affected_rows": duplicates,
                "affected_pct": round(duplicates / total * 100, 2),
                "consequence": "Double-counts the same transaction line.",
                "treatment": "Deduplicated.",
            }
        )

    return findings


def full_report(frame: pd.DataFrame) -> dict:
    """Everything the EDA screen needs, in one payload."""
    return {
        "overview": overview(frame),
        "missing_values": missing_values(frame),
        "numeric_summary": numeric_summary(frame),
        "date_range": date_range(frame),
        "top_countries": top_categories(frame, "country"),
        "top_products": top_categories(frame, "description"),
        "quality_findings": quality_findings(frame),
    }
