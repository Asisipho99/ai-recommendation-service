"""The interest-seeded tier: hybrid -> INTEREST-SEEDED -> popularity.

Flask test client, no server and no database: every query function is
monkeypatched on the module that LOOKS IT UP, since each module imported the
name into its own namespace. The recommender modules are imported inside the
Flask handler per request, so the patches take effect.

The case that matters most is TestHybridIsUnchanged: an existing customer with
interaction history must get exactly the result they get today, and the
interests lookup must never run for them.
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
import recommender.interests as interests_module  # noqa: E402


CATALOGUE = [
    {"product_id": 1, "name": "Kindle", "category": "Books", "price": 129.99},
    {"product_id": 2, "name": "Lego Set", "category": "Toys", "price": 49.5},
    {"product_id": 3, "name": "Novel", "category": "Books", "price": 12.0},
    {"product_id": 4, "name": "Cookbook", "category": "Books", "price": 24.0},
]

BOOKS_ROWS = [
    {"product_id": 1, "name": "Kindle", "category": "Books",
     "price": 129.99, "image_url": "", "purchase_count": 9},
    {"product_id": 3, "name": "Novel", "category": "Books",
     "price": 12.0, "image_url": "", "purchase_count": 4},
]

POPULAR_ROWS = [
    {"product_id": 2, "name": "Lego Set", "category": "Toys",
     "price": 49.5, "image_url": "", "purchase_count": 7},
    {"product_id": 4, "name": "Cookbook", "category": "Books",
     "price": 24.0, "image_url": "", "purchase_count": 2},
]


@pytest.fixture()
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


def body(response):
    return response.get_json()


@pytest.fixture()
def no_history(monkeypatch):
    """A catalogue, no purchases and a hybrid merge that produces nothing."""
    monkeypatch.setattr(app_module, "get_all_products", lambda: CATALOGUE)
    monkeypatch.setattr(hybrid_module, "get_customer_purchases", lambda cid: [])
    monkeypatch.setattr(
        hybrid_module, "get_collaborative_recommendations",
        lambda cid, all_ids, purchased, n=5: [])
    monkeypatch.setattr(
        hybrid_module, "get_content_recommendations",
        lambda purchased, all_ids, n=5: [])
    return monkeypatch


class TestInterestSeededTier:
    def test_a_customer_with_interests_gets_products_from_those_categories(
            self, client, no_history):
        no_history.setattr(
            interests_module, "get_customer_interests", lambda cid: ["Books"])
        no_history.setattr(
            interests_module, "get_top_products_by_categories",
            lambda categories, limit=5, exclude_ids=None: BOOKS_ROWS)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        data = body(response)["data"]
        assert data["type"] == "fallback"
        assert data["interest_seeded"] is True
        assert data["count"] == 2
        for item in data["recommendations"]:
            assert item["category"] == "Books"
            assert item["reason"] == "Based on your interest in Books"
            assert item["score"] == 0.5

    def test_a_customer_without_interests_gets_the_popularity_fallback(
            self, client, no_history):
        no_history.setattr(
            interests_module, "get_customer_interests", lambda cid: [])
        no_history.setattr(
            hybrid_module, "get_top_products",
            lambda limit=5, exclude_ids=None: POPULAR_ROWS)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        data = body(response)["data"]
        assert data["type"] == "fallback"
        assert "interest_seeded" not in data
        assert [item["reason"] for item in data["recommendations"]] == [
            "Popular with our customers", "Popular with our customers"]

    def test_a_thin_category_is_topped_up_from_popularity(
            self, client, no_history):
        no_history.setattr(
            interests_module, "get_customer_interests", lambda cid: ["Books"])
        no_history.setattr(
            interests_module, "get_top_products_by_categories",
            lambda categories, limit=5, exclude_ids=None: BOOKS_ROWS[:1])
        no_history.setattr(
            interests_module, "get_top_products",
            lambda limit=5, exclude_ids=None: POPULAR_ROWS)

        response = client.get("/api/recommendations/42?n=3")
        assert response.status_code == 200

        data = body(response)["data"]
        assert data["type"] == "fallback"
        assert data["interest_seeded"] is True
        assert data["count"] == 3
        assert [item["reason"] for item in data["recommendations"]] == [
            "Based on your interest in Books",
            "Popular with our customers",
            "Popular with our customers",
        ]


class TestHybridIsUnchanged:
    """The regression that matters: an existing customer's result is today's."""

    def test_scored_history_still_produces_the_0_6_0_4_merge_on_a_0_1_scale(
            self, client, monkeypatch):
        interest_calls = []

        def interests_recorder(customer_id):
            interest_calls.append(customer_id)
            return ["Books"]

        monkeypatch.setattr(app_module, "get_all_products", lambda: CATALOGUE)
        monkeypatch.setattr(hybrid_module, "get_customer_purchases", lambda cid: [])
        monkeypatch.setattr(
            hybrid_module, "get_collaborative_recommendations",
            lambda cid, all_ids, purchased, n=5: [
                {"product_id": 1, "score": 4.2, "reason": "Customers like you bought this"}])
        monkeypatch.setattr(
            hybrid_module, "get_content_recommendations",
            lambda purchased, all_ids, n=5: [
                {"product_id": 1, "score": 0.5, "reason": "Similar to what you bought"},
                {"product_id": 2, "score": 0.9, "reason": "Similar to what you bought"}])
        monkeypatch.setattr(
            hybrid_module, "get_product_details",
            lambda ids: [p for p in CATALOGUE if p['product_id'] in ids])
        monkeypatch.setattr(
            interests_module, "get_customer_interests", interests_recorder)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        data = body(response)["data"]
        assert data["type"] == "hybrid"
        assert "interest_seeded" not in data

        scores = {item["product_id"]: item["score"]
                  for item in data["recommendations"]}
        # 0.6 x collaborative + 0.4 x content, both on a 0-1 scale. The
        # collaborative score arrives as a 1-5 rating estimate: 4.2 -> 0.8.
        assert scores[1] == round(0.8 * 0.6 + 0.5 * 0.4, 4)
        assert scores[2] == round(0.9 * 0.4, 4)

        assert interest_calls == []


class TestPurchasedProductsAreExcluded:
    def test_both_tier_2_paths_receive_the_purchased_ids(
            self, client, no_history):
        recorded = {}

        def by_categories_recorder(categories, limit=5, exclude_ids=None):
            recorded["by_categories"] = exclude_ids
            return BOOKS_ROWS[:1]

        def top_products_recorder(limit=5, exclude_ids=None):
            recorded["top_products"] = exclude_ids
            return POPULAR_ROWS[:1]

        no_history.setattr(hybrid_module, "get_customer_purchases", lambda cid: [1])
        no_history.setattr(
            interests_module, "get_customer_interests", lambda cid: ["Books"])
        no_history.setattr(
            interests_module, "get_top_products_by_categories", by_categories_recorder)
        no_history.setattr(interests_module, "get_top_products", top_products_recorder)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        assert 1 in recorded["by_categories"]
        assert 1 in recorded["top_products"]

    def test_the_popularity_tier_receives_the_purchased_ids(
            self, client, no_history):
        recorded = {}

        def top_products_recorder(limit=5, exclude_ids=None):
            recorded["exclude_ids"] = exclude_ids
            return POPULAR_ROWS

        no_history.setattr(hybrid_module, "get_customer_purchases", lambda cid: [1])
        no_history.setattr(
            interests_module, "get_customer_interests", lambda cid: [])
        no_history.setattr(hybrid_module, "get_top_products", top_products_recorder)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        assert 1 in recorded["exclude_ids"]
