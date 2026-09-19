"""Tests for the EDA profiling module.

Asserts on a small fixture with known defects, so the counts are exact. Every
quality finding here maps to a cleaning step in dataset/prepare.py - these tests
are what keep the two in step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import eda  # noqa: E402


def sample_frame() -> pd.DataFrame:
    rows = [
        ("INV001", "A1", "Widget", 2, "2011-01-10 10:00", 5.0, 1001, "UK"),
        ("INV002", "A2", "Gadget", 1, "2011-06-01 10:00", 10.0, 1001, "UK"),
        ("INV003", "A1", "Widget", 3, "2011-01-15 10:00", 5.0, 1002, "UK"),
        ("INV003", "A1", "Widget", 3, "2011-01-15 10:00", 5.0, 1002, "UK"),
        ("C9001", "A1", "Widget", 1, "2011-02-01 10:00", 5.0, 1002, "UK"),
        ("INV004", "A3", "Doohickey", -4, "2011-02-02 10:00", 5.0, 1002, "UK"),
        ("INV005", "A4", "Freebie", 1, "2011-02-03 10:00", 0.0, 1002, "UK"),
        ("INV006", "A1", "Widget", 1, "2011-02-04 10:00", 5.0, None, "France"),
        ("INV007", "A5", None, 2, "2011-02-05 10:00", 7.5, 1003, "France"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "invoice", "stock_code", "description", "quantity",
            "invoice_date", "unit_price", "customer_id", "country",
        ],
    )


class TestOverview:
    def test_reports_shape_and_duplicates(self):
        result = eda.overview(sample_frame())
        assert result["rows"] == 9
        assert result["columns"] == 8
        assert result["duplicate_rows"] == 1
        assert "invoice" in result["column_names"]

    def test_includes_dtypes_for_every_column(self):
        frame = sample_frame()
        result = eda.overview(frame)
        assert set(result["dtypes"]) == set(str(c) for c in frame.columns)


class TestMissingValues:
    def test_counts_nulls_worst_first(self):
        result = eda.missing_values(sample_frame())
        by_column = {row["column"]: row for row in result}

        assert by_column["customer_id"]["missing"] == 1
        assert by_column["description"]["missing"] == 1
        assert by_column["invoice"]["missing"] == 0

        # Sorted descending by missing count.
        counts = [row["missing"] for row in result]
        assert counts == sorted(counts, reverse=True)

    def test_present_and_missing_sum_to_total(self):
        frame = sample_frame()
        for row in eda.missing_values(frame):
            assert row["missing"] + row["present"] == len(frame)


class TestQualityFindings:
    def test_identifies_every_defect_in_the_fixture(self):
        findings = {f["issue"]: f for f in eda.quality_findings(sample_frame())}

        assert findings["Missing customer identifier"]["affected_rows"] == 1
        assert findings["Missing product description"]["affected_rows"] == 1
        assert findings["Cancellation invoices"]["affected_rows"] == 1
        assert findings["Non-positive quantity"]["affected_rows"] == 1
        assert findings["Zero or negative unit price"]["affected_rows"] == 1
        assert findings["Exact duplicate rows"]["affected_rows"] == 1

    def test_every_finding_states_a_treatment(self):
        """A finding without a stated treatment is an unexplained decision."""
        for finding in eda.quality_findings(sample_frame()):
            assert finding["treatment"]
            assert finding["consequence"]
            assert finding["affected_pct"] >= 0

    def test_clean_data_produces_no_findings(self):
        clean = sample_frame().iloc[[0, 1]].copy()
        assert eda.quality_findings(clean) == []


class TestNumericSummary:
    def test_flags_negative_and_zero_values(self):
        summary = {row["column"]: row for row in eda.numeric_summary(sample_frame())}

        assert summary["quantity"]["negative_values"] == 1
        assert summary["unit_price"]["zero_values"] == 1

    def test_skips_non_numeric_columns(self):
        columns = {row["column"] for row in eda.numeric_summary(sample_frame())}
        assert "invoice" not in columns
        assert "quantity" in columns


class TestDateRange:
    def test_reports_span(self):
        result = eda.date_range(sample_frame())
        assert result["available"] is True
        assert result["span_days"] > 0
        assert result["earliest"] < result["latest"]

    def test_handles_a_missing_column(self):
        frame = sample_frame().drop(columns=["invoice_date"])
        assert eda.date_range(frame)["available"] is False


class TestDistribution:
    def test_produces_bins(self):
        result = eda.distribution(sample_frame(), "unit_price", bins=4)
        assert result["available"] is True
        assert len(result["bins"]) == 4
        assert all("count" in b for b in result["bins"])

    def test_reports_unavailable_for_an_unknown_column(self):
        result = eda.distribution(sample_frame(), "not_a_column")
        assert result["available"] is False


class TestJsonSafety:
    def test_nan_and_infinity_become_null(self):
        """The API serialises these straight to JSON, which cannot carry NaN."""
        frame = sample_frame()
        # Widen to float first: pandas 3 refuses to place inf/NaN into an
        # int64 column rather than silently upcasting.
        frame["quantity"] = frame["quantity"].astype(float)
        frame["unit_price"] = frame["unit_price"].astype(float)
        frame.loc[0, "unit_price"] = np.nan
        frame.loc[1, "quantity"] = np.inf

        for row in eda.numeric_summary(frame):
            for value in row.values():
                assert value == value  # not NaN
                assert value not in (float("inf"), float("-inf"))
