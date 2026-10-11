"""Regression tests for content-based recommendation.

The bug these pin down: get_content_recommendations() raised
`TypeError: np.matrix is not supported` for every customer with purchase
history, because scipy's sparse .mean(axis=0) returns numpy.matrix and
scikit-learn's check_array refuses it. Customers with an empty purchased_ids
return early and never reach that line, so the failure only ever appeared for
real customers - and in production it surfaced as a "fallback" response rather
than an error, because the route's blanket handler swallowed it.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import recommender.content_based as cb

PRODUCTS = [
    {"product_id": 1, "name": "Kudu Adjustable Resistance Bands",
     "category": "Sports",
     "description": "Non slip textured surface for stable footing training."},
    {"product_id": 2, "name": "Ridgeline Lightweight Yoga Mat",
     "category": "Sports",
     "description": "Non slip textured surface shock absorbing training mat."},
    {"product_id": 3, "name": "Cape Quill Illustrated Paperback Novel",
     "category": "Books",
     "description": "Illustrated chapters with an annotated bibliography."},
    {"product_id": 4, "name": "Thornfield Annotated Hardcover Biography",
     "category": "Books",
     "description": "Revised edition printed on acid free paper annotated."},
]

ALL_IDS = [p["product_id"] for p in PRODUCTS]


@pytest.fixture
def model(tmp_path, monkeypatch):
    """Build a real TF-IDF model in a temp location."""
    monkeypatch.setattr(cb, "MODEL_PATH", str(tmp_path / "content_model.pkl"))
    cb.build_content_features(PRODUCTS)
    return cb.load_content_model()


def test_purchase_history_does_not_raise(model):
    """The regression. One purchased product used to be enough to crash."""
    out = cb.get_content_recommendations([3], ALL_IDS, n=3)
    assert isinstance(out, list)


def test_multiple_purchases_do_not_raise(model):
    out = cb.get_content_recommendations([1, 3], ALL_IDS, n=3)
    assert isinstance(out, list)


def test_recommends_same_category_as_purchase(model):
    """Buying a Book should surface the other Book above the Sports items."""
    out = cb.get_content_recommendations([3], ALL_IDS, n=3)
    assert out, "expected at least one recommendation"
    assert out[0]["product_id"] == 4
    assert all(r["score"] > 0 for r in out)


def test_purchased_products_are_excluded(model):
    out = cb.get_content_recommendations([3], ALL_IDS, n=10)
    assert 3 not in [r["product_id"] for r in out]


def test_scores_descend(model):
    out = cb.get_content_recommendations([1], ALL_IDS, n=10)
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_no_history_returns_empty(model):
    assert cb.get_content_recommendations([], ALL_IDS, n=3) == []


def test_unknown_purchase_returns_empty(model):
    """A product bought before the model was last trained is not in it."""
    assert cb.get_content_recommendations([999], ALL_IDS, n=3) == []


def test_average_vector_is_a_plain_ndarray(model):
    """Guard the specific type that broke it, not just the symptom."""
    matrix = model["tfidf_matrix"][[0, 2]]
    assert isinstance(matrix.mean(axis=0), np.matrix), (
        "scipy no longer returns np.matrix; the asarray guard may be "
        "removable, but verify before touching it"
    )
    assert not isinstance(np.asarray(matrix.mean(axis=0)), np.matrix)


def test_similar_products_within_category(model):
    out = cb.get_similar_products(1, n=3)
    assert out
    assert out[0]["product_id"] == 2


def test_similar_products_excludes_itself(model):
    assert 1 not in [r["product_id"] for r in cb.get_similar_products(1, n=10)]


def test_empty_catalogue_builds_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "MODEL_PATH", str(tmp_path / "none.pkl"))
    vectoriser, matrix = cb.build_content_features([])
    assert vectoriser is None and matrix is None
    assert not Path(cb.MODEL_PATH).exists()
