"""The degraded tier (tier 3) must exclude products the customer already bought.

No server and no database: the Flask test client is used as in test_api_smoke.py
and every database call is monkeypatched. The recommender modules are imported
inside the handler on each request, so patching a module attribute takes effect.
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


CATALOGUE = [
    {"product_id": 1, "name": "Kindle", "category": "Books", "price": 129.99},
    {"product_id": 2, "name": "Lego Set", "category": "Toys", "price": 49.5},
    {"product_id": 3, "name": "Novel", "category": "Books", "price": 12.0},
]

TOP_PRODUCTS = [
    {"product_id": 2, "name": "Lego Set", "category": "Toys",
     "price": 49.5, "image_url": "", "purchase_count": 7},
    {"product_id": 3, "name": "Novel", "category": "Books",
     "price": 12.0, "image_url": "", "purchase_count": 3},
]


@pytest.fixture()
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


def body(response):
    return response.get_json()


def _boom(*args, **kwargs):
    raise RuntimeError("hybrid exploded")


class TestDegradedFallbackExcludesPurchases:
    def test_purchased_ids_reach_the_popularity_query(self, client, monkeypatch):
        recorded = {}

        def recorder(limit=5, exclude_ids=None):
            recorded["limit"] = limit
            recorded["exclude_ids"] = exclude_ids
            return TOP_PRODUCTS

        monkeypatch.setattr(app_module, "get_all_products", lambda: CATALOGUE)
        monkeypatch.setattr(app_module, "get_customer_purchases", lambda cid: [1])
        monkeypatch.setattr(hybrid_module, "get_hybrid_recommendations", _boom)
        monkeypatch.setattr(hybrid_module, "get_top_products", recorder)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        data = body(response)["data"]
        assert data["degraded"] is True
        assert data["type"] == "fallback"
        assert recorded["exclude_ids"] == [1]

    def test_a_failing_purchase_lookup_still_returns_a_degraded_fallback(
            self, client, monkeypatch):
        """The degraded path is reached because something failed already."""

        def exploding_lookup(customer_id):
            raise RuntimeError("database unreachable")

        monkeypatch.setattr(app_module, "get_all_products", lambda: CATALOGUE)
        monkeypatch.setattr(app_module, "get_customer_purchases", exploding_lookup)
        monkeypatch.setattr(hybrid_module, "get_hybrid_recommendations", _boom)
        monkeypatch.setattr(
            hybrid_module, "get_top_products",
            lambda limit=5, exclude_ids=None: TOP_PRODUCTS)

        response = client.get("/api/recommendations/42?n=2")
        assert response.status_code == 200

        payload = body(response)
        assert payload["success"] is True

        data = payload["data"]
        assert data["degraded"] is True
        assert data["type"] == "fallback"
        assert data["count"] == 2
