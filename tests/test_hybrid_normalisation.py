"""The hybrid blend puts both scores on a 0-1 scale before the 60/40 weighting.

Collaborative scores are SVD rating estimates on 1-5, content scores are cosine
similarities on 0-1. Before normalisation the collaborative half dominated by
roughly an order of magnitude whatever the weights said.

No database and no model files: the two strategies and the product lookup are
monkeypatched on recommender.hybrid, the module that looks them up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import recommender.hybrid as hybrid_module  # noqa: E402
from recommender.hybrid import (  # noqa: E402
    get_hybrid_recommendations,
    normalise_collaborative_score,
    normalise_content_score,
)

CATALOGUE = [
    {"product_id": pid, "name": f"Product {pid}", "category": "Books",
     "price": 10.0, "image_url": ""}
    for pid in (1, 2, 3, 4)
]


def _patch(monkeypatch, collaborative, content):
    monkeypatch.setattr(hybrid_module, "get_customer_purchases", lambda cid: [9])
    monkeypatch.setattr(
        hybrid_module, "get_collaborative_recommendations",
        lambda cid, all_ids, purchased, n=5: collaborative)
    monkeypatch.setattr(
        hybrid_module, "get_content_recommendations",
        lambda purchased, all_ids, n=5: content)
    monkeypatch.setattr(
        hybrid_module, "get_product_details",
        lambda ids: [p for p in CATALOGUE if p["product_id"] in ids])


def _collab(pid, est):
    return {"product_id": pid, "score": est, "reason": "Customers like you also bought this"}


def _content(pid, sim):
    return {"product_id": pid, "score": sim, "reason": "Based on your purchase history"}


class TestNormalisers:
    @pytest.mark.parametrize("estimate, expected", [
        (1.0, 0.0), (3.0, 0.5), (5.0, 1.0), (4.2, 0.8),
    ])
    def test_collaborative_maps_the_1_5_rating_scale_onto_0_1(self, estimate, expected):
        assert normalise_collaborative_score(estimate) == pytest.approx(expected)

    @pytest.mark.parametrize("estimate, expected", [(0.2, 0.0), (5.4, 1.0)])
    def test_collaborative_estimates_outside_the_scale_are_clamped(
            self, estimate, expected):
        assert normalise_collaborative_score(estimate) == expected

    @pytest.mark.parametrize("similarity", [0.0, 0.25, 1.0])
    def test_content_similarity_is_already_0_1_and_is_unchanged(self, similarity):
        assert normalise_content_score(similarity) == similarity


class TestBlend:
    def test_a_perfect_content_match_can_beat_an_average_collaborative_one(
            self, monkeypatch):
        """Unnormalised, 3.0 * 0.6 = 1.8 always beat 1.0 * 0.4 = 0.4.
        Normalised, 0.5 * 0.6 = 0.30 against 1.0 * 0.4 = 0.40."""
        _patch(monkeypatch,
               collaborative=[_collab(1, 3.0)],
               content=[_content(2, 1.0)])
        results, source = get_hybrid_recommendations(42, [1, 2, 3, 4], n=2)
        assert source == "hybrid"
        assert [r["product_id"] for r in results] == [2, 1]
        assert results[0]["score"] == pytest.approx(0.40)
        assert results[1]["score"] == pytest.approx(0.30)

    def test_scores_are_on_a_0_1_scale_and_use_the_60_40_weights(self, monkeypatch):
        _patch(monkeypatch,
               collaborative=[_collab(1, 5.0), _collab(2, 1.0)],
               content=[_content(1, 1.0), _content(3, 0.5)])
        results, _ = get_hybrid_recommendations(42, [1, 2, 3, 4], n=4)
        by_id = {r["product_id"]: r for r in results}
        assert by_id[1]["score"] == pytest.approx(1.0)          # 0.6 + 0.4
        assert by_id[2]["score"] == pytest.approx(0.0)          # 0.6 * 0
        assert by_id[3]["score"] == pytest.approx(0.2)          # 0.4 * 0.5
        assert all(0.0 <= r["score"] <= 1.0 for r in results)

    def test_a_product_found_by_both_strategies_keeps_the_combined_reason(
            self, monkeypatch):
        _patch(monkeypatch,
               collaborative=[_collab(1, 4.0)],
               content=[_content(1, 0.5)])
        results, _ = get_hybrid_recommendations(42, [1, 2, 3, 4], n=2)
        assert results[0]["reason"] == "Recommended based on your activity"
        assert results[0]["score"] == pytest.approx(0.75 * 0.6 + 0.5 * 0.4)
