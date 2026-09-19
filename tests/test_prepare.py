"""Tests for the cleaning and feature-engineering pipeline.

Uses a small hand-built frame rather than the real dataset, so the tests run
fast and assert on exact numbers. The fixture deliberately contains one of every
data-quality problem the real dataset has.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import prepare  # noqa: E402


def dirty_frame() -> pd.DataFrame:
    """Nine rows: four clean, five with a specific defect each."""
    rows = [
        # customer 1 - two clean invoices, the second inside the holdout window
        ("INV001", "A1", "Widget", 2, "2011-01-10 10:00", 5.0, 1001, "UK"),
        ("INV002", "A2", "Gadget", 1, "2011-12-01 10:00", 10.0, 1001, "UK"),
        # customer 2 - one clean invoice, never returns
        ("INV003", "A1", "Widget", 3, "2011-01-15 10:00", 5.0, 1002, "UK"),
        # duplicate of the row above
        ("INV003", "A1", "Widget", 3, "2011-01-15 10:00", 5.0, 1002, "UK"),
        # cancellation
        ("C9001", "A1", "Widget", 1, "2011-02-01 10:00", 5.0, 1002, "UK"),
        # negative quantity (a return)
        ("INV004", "A3", "Doohickey", -4, "2011-02-02 10:00", 5.0, 1002, "UK"),
        # zero price
        ("INV005", "A4", "Freebie", 1, "2011-02-03 10:00", 0.0, 1002, "UK"),
        # no customer id
        ("INV006", "A1", "Widget", 1, "2011-02-04 10:00", 5.0, None, "UK"),
        # missing description, but otherwise a valid sale
        ("INV007", "A5", None, 2, "2011-02-05 10:00", 7.5, 1003, "France"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "invoice", "stock_code", "description", "quantity",
            "invoice_date", "unit_price", "customer_id", "country",
        ],
    )


class TestClean:
    def test_removes_each_defect_and_keeps_valid_rows(self):
        cleaned, steps = prepare.clean(dirty_frame())

        invoices = set(cleaned["invoice"])
        assert invoices == {"INV001", "INV002", "INV003", "INV007"}

        # The duplicate collapsed, so INV003 appears once.
        assert (cleaned["invoice"] == "INV003").sum() == 1

    def test_audit_trail_reports_every_step(self):
        _cleaned, steps = prepare.clean(dirty_frame())
        names = [s["step"] for s in steps]

        assert "Remove exact duplicate rows" in names
        assert "Drop cancellation invoices" in names
        assert "Drop non-positive quantity" in names
        assert "Drop zero or negative unit price" in names
        assert "Drop rows with no customer identifier" in names
        assert names[-1] == "TOTAL"

    def test_total_step_reconciles(self):
        frame = dirty_frame()
        cleaned, steps = prepare.clean(frame)
        total = steps[-1]

        assert total["rows_before"] == len(frame)
        assert total["rows_after"] == len(cleaned)
        assert total["rows_removed"] == len(frame) - len(cleaned)

    def test_missing_descriptions_are_imputed_not_dropped(self):
        cleaned, steps = prepare.clean(dirty_frame())

        # INV007 had no description but survived.
        assert "INV007" in set(cleaned["invoice"])
        assert cleaned.loc[cleaned["invoice"] == "INV007", "description"].iloc[0] == "Unknown"

        step = next(s for s in steps if s["step"] == "Fill missing product descriptions")
        assert step["values_imputed"] == 1
        assert step["rows_removed"] == 0

    def test_derives_line_total(self):
        cleaned, _ = prepare.clean(dirty_frame())
        row = cleaned[cleaned["invoice"] == "INV001"].iloc[0]
        assert row["line_total"] == pytest.approx(10.0)  # 2 x 5.00

    def test_customer_id_is_integer(self):
        cleaned, _ = prepare.clean(dirty_frame())
        assert cleaned["customer_id"].dtype.kind == "i"

    def test_does_not_mutate_the_input(self):
        frame = dirty_frame()
        before = len(frame)
        prepare.clean(frame)
        assert len(frame) == before


class TestCustomerFeatures:
    def test_one_row_per_customer(self):
        cleaned, _ = prepare.clean(dirty_frame())
        features, _meta = prepare.build_customer_features(cleaned)
        assert features["customer_id"].is_unique

    def test_label_marks_customers_who_returned_in_the_holdout_window(self):
        cleaned, _ = prepare.clean(dirty_frame())
        features, _meta = prepare.build_customer_features(cleaned)

        by_customer = features.set_index("customer_id")["repurchased"].to_dict()

        # 1001 bought again on 2011-12-01, inside the final 90 days.
        assert by_customer[1001] == 1
        # 1002 and 1003 did not come back.
        assert by_customer.get(1002, 0) == 0

    def test_features_exclude_the_label_and_the_id(self):
        cleaned, _ = prepare.clean(dirty_frame())
        _features, meta = prepare.build_customer_features(cleaned)

        assert "repurchased" not in meta["feature_columns"]
        assert "customer_id" not in meta["feature_columns"]
        assert "recency_days" in meta["feature_columns"]
        assert "monetary_total" in meta["feature_columns"]

    def test_feature_window_ends_before_the_dataset_does(self):
        """Guards against leaking the answer into the inputs."""
        cleaned, _ = prepare.clean(dirty_frame())
        _features, meta = prepare.build_customer_features(cleaned)

        assert meta["feature_window_end"] < meta["dataset_end"]
        assert meta["holdout_days"] == prepare.HOLDOUT_DAYS

    def test_rejects_a_dataset_that_is_too_short(self):
        frame = dirty_frame()
        # Squeeze everything into a single day, so nothing precedes the cutoff.
        frame["invoice_date"] = "2011-12-01 10:00"
        cleaned, _ = prepare.clean(frame)

        with pytest.raises(ValueError, match="cutoff"):
            prepare.build_customer_features(cleaned)
