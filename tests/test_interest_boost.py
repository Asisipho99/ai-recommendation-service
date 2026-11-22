"""Interests give an established customer a modest nudge in the hybrid ranking.

The nudge is INTEREST_BOOST (0.05) added to the 0-1 blended score of a product
whose category the customer picked, only for customers with purchase history.
It reorders close scores. It must never filter and never dominate.

No database and no model files: everything the hybrid looks up is
monkeypatched on recommender.hybrid, the module that looks it up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402
from app import app as flask_app  # noqa: E402
import recommender.hybrid as hybrid_module  # noqa: E402
from recommender.hybrid import INTEREST_BOOST, get_hybrid_recommendations  # noqa: E402

# Product id -> category. Interests in the tests below are Books and Toys.
CATEGORY = {
    1: "Garden", 2: "Garden", 3: "Books", 4: "Garden", 5: "Toys",
    6: "Garden", 7: "Books", 8: "Garden", 9: "Toys", 10: "Garden",
}
CATALOGUE = [
    {"product_id": pid, "name": f"Product {pid}", "category": category,
     "price": 10.0, "image_url": ""}
    for pid, category in CATEGORY.items()
]
ALL_IDS = list(CATEGORY)


def _content(pid, similarity):
    return {"product_id": pid, "score": similarity,
            "reason": "Based on your purchase history"}


def _setup(monkeypatch, content, interests=("Books", "Toys"),
           purchased=(100,), interest_lookup=None):
    """Content-only candidates keep the arithmetic readable: blended = 0.4 x sim."""
    calls = []

    def lookup(customer_id):
        calls.append(customer_id)
        return list(interests)

    monkeypatch.setattr(hybrid_module, "get_customer_purchases",
                        lambda cid: list(purchased))
    monkeypatch.setattr(
        hybrid_module, "get_collaborative_recommendations",
        lambda cid, all_ids, purchased_ids, n=5: [])
    monkeypatch.setattr(
        hybrid_module, "get_content_recommendations",
        lambda purchased_ids, all_ids, n=5: content)
    monkeypatch.setattr(
        hybrid_module, "get_product_details",
        lambda ids: [p for p in CATALOGUE if p["product_id"] in ids])
    monkeypatch.setattr(hybrid_module, "get_customer_interests",
                        interest_lookup or lookup)
    return calls


def _ids(results):
    return [r["product_id"] for r in results]


# Blended scores (0.4 x similarity), best first, with the interest products
# (3, 5, 7, 9) sitting just below, level with and well below the top of the list:
#   1 Garden 0.40 | 2 Garden 0.36 | 3 Books 0.32 | 4 Garden 0.30
#   5 Toys   0.28 | 6 Garden 0.24 | 7 Books 0.20 | 8 Garden 0.16 | 9 Toys 0.08
CONTENT = [_content(1, 1.0), _content(2, 0.9), _content(3, 0.8),
           _content(4, 0.75), _content(5, 0.7), _content(6, 0.6),
           _content(7, 0.5), _content(8, 0.4), _content(9, 0.2)]


class TestBoostReordersButDoesNotDominate:
    def test_an_interest_product_just_behind_overtakes_a_near_neighbour(
            self, monkeypatch):
        _setup(monkeypatch, CONTENT)
        results, source = get_hybrid_recommendations(42, ALL_IDS, n=5)
        assert source == "hybrid"
        # Product 5 (Toys, 0.28 + 0.05 = 0.33) overtakes product 4 (0.30), and
        # product 3 (Books, 0.32 + 0.05 = 0.37) moves ahead of product 2 (0.36).
        assert _ids(results) == [1, 3, 2, 5, 4]

    def test_the_list_differs_from_the_unboosted_order(self, monkeypatch):
        _setup(monkeypatch, CONTENT, interests=())
        unboosted, _ = get_hybrid_recommendations(42, ALL_IDS, n=5)
        _setup(monkeypatch, CONTENT)
        boosted, _ = get_hybrid_recommendations(42, ALL_IDS, n=5)
        assert _ids(unboosted) == [1, 2, 3, 4, 5]
        assert _ids(boosted) != _ids(unboosted)

    def test_a_distant_interest_product_does_not_jump_the_queue(self, monkeypatch):
        _setup(monkeypatch, CONTENT)
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=9)
        order = _ids(results)
        # Product 7 (Books, 0.20 + 0.05 = 0.25) and 9 (Toys, 0.08 + 0.05 = 0.13)
        # stay behind every product scoring more than 0.05 above them.
        assert order.index(7) > order.index(4)
        assert order.index(9) == len(order) - 1

    def test_the_list_is_not_dominated_by_interest_products(self, monkeypatch):
        _setup(monkeypatch, CONTENT)
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=5)
        interest = [r for r in results if r["category"] in ("Books", "Toys")]
        other = [r for r in results if r["category"] not in ("Books", "Toys")]
        assert len(other) >= 3
        assert len(interest) < len(results)
        # The best-ranked product is still the top non-interest match.
        assert results[0]["product_id"] == 1

    def test_non_interest_products_are_never_filtered_out(self, monkeypatch):
        _setup(monkeypatch, CONTENT)
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=9)
        assert sorted(_ids(results)) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_the_boost_never_adds_a_product_the_strategies_did_not_surface(
            self, monkeypatch):
        _setup(monkeypatch, CONTENT[:5])
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=9)
        assert sorted(_ids(results)) == [1, 2, 3, 4, 5]
        assert 7 not in _ids(results) and 9 not in _ids(results)

    def test_the_boost_is_exactly_the_documented_size(self, monkeypatch):
        _setup(monkeypatch, [_content(3, 0.5)])
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=1)
        assert INTEREST_BOOST == 0.05
        assert results[0]["score"] == pytest.approx(0.5 * 0.4 + INTEREST_BOOST)

    def test_a_product_gains_the_boost_once_whatever_the_interest_count(
            self, monkeypatch):
        _setup(monkeypatch, [_content(3, 0.5)],
               interests=("Books", "books", "Books ", "Toys"))
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=1)
        assert results[0]["score"] == pytest.approx(0.5 * 0.4 + INTEREST_BOOST)

    def test_the_score_stays_on_the_0_1_scale(self, monkeypatch):
        _setup(monkeypatch, [])
        monkeypatch.setattr(
            hybrid_module, "get_collaborative_recommendations",
            lambda cid, all_ids, purchased_ids, n=5: [
                {"product_id": 3, "score": 5.0,
                 "reason": "Customers like you also bought this"}])
        monkeypatch.setattr(
            hybrid_module, "get_content_recommendations",
            lambda purchased_ids, all_ids, n=5: [_content(3, 1.0)])
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=1)
        assert results[0]["score"] == 1.0


class TestBoostOnlyForCustomersWithPurchaseHistory:
    def test_no_purchases_means_no_interest_lookup_and_no_boost(self, monkeypatch):
        calls = _setup(monkeypatch, CONTENT, purchased=())
        results, source = get_hybrid_recommendations(42, ALL_IDS, n=5)
        assert source == "hybrid"
        assert calls == []
        assert _ids(results) == [1, 2, 3, 4, 5]

    def test_a_customer_with_no_interests_is_ranked_as_before(self, monkeypatch):
        _setup(monkeypatch, CONTENT, interests=())
        results, _ = get_hybrid_recommendations(42, ALL_IDS, n=5)
        assert _ids(results) == [1, 2, 3, 4, 5]

    def test_a_failing_interest_lookup_does_not_fail_the_recommendation(
            self, monkeypatch):
        def broken(customer_id):
            raise RuntimeError("customer_interests unreachable")

        _setup(monkeypatch, CONTENT, interest_lookup=broken)
        results, source = get_hybrid_recommendations(42, ALL_IDS, n=5)
        assert source == "hybrid"
        assert _ids(results) == [1, 2, 3, 4, 5]


class TestContractIsUnchanged:
    @pytest.fixture()
    def client(self):
        flask_app.config.update(TESTING=True)
        with flask_app.test_client() as test_client:
            yield test_client

    def test_the_api_still_answers_hybrid_with_no_interest_seeded_flag(
            self, client, monkeypatch):
        _setup(monkeypatch, CONTENT)
        monkeypatch.setattr(app_module, "get_all_products", lambda: CATALOGUE)
        response = client.get("/api/recommendations/42?n=5")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["type"] == "hybrid"
        assert "interest_seeded" not in data
        assert [item["product_id"] for item in data["recommendations"]] == [1, 3, 2, 5, 4]
        for item in data["recommendations"]:
            assert not item["reason"].startswith("Based on your interest in")
