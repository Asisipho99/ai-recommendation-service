"""Smoke tests for the new data and ML routes.

Uses Flask's test client, so no server and no database are needed. The
recommender routes are not covered here because they require a live MySQL
connection; these tests target the dataset and ML endpoints, which do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app as flask_app  # noqa: E402


@pytest.fixture()
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


def body(response):
    return response.get_json()


class TestHealth:
    def test_reports_running(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = body(response)
        assert payload["success"] is True
        assert payload["data"]["status"] == "running"


class TestDataSource:
    def test_describes_the_dataset_provenance(self, client):
        response = client.get("/api/data/source")
        assert response.status_code == 200

        dataset = body(response)["data"]["dataset"]
        assert dataset["name"] == "Online Retail II"
        # The provenance matters: it must be an external dataset, not one this
        # platform generated about itself.
        assert "UCI" in dataset["origin"]


class TestMlStatus:
    def test_always_answers_even_before_training(self, client):
        response = client.get("/api/ml/status")
        assert response.status_code == 200

        data = body(response)["data"]
        assert "supervised_trained" in data
        assert "segmentation_trained" in data


class TestMlPredictValidation:
    def test_rejects_a_request_with_no_features(self, client):
        response = client.post("/api/ml/predict", json={})
        assert response.status_code == 400
        assert body(response)["success"] is False

    def test_rejects_features_of_the_wrong_shape(self, client):
        response = client.post("/api/ml/predict", json={"features": "not-an-object"})
        assert response.status_code == 400


class TestErrorEnvelope:
    def test_failures_use_the_same_envelope_as_successes(self, client):
        """Every response carries `success`, so callers need one code path."""
        ok = body(client.get("/api/ml/status"))
        failed = body(client.post("/api/ml/predict", json={}))

        assert ok["success"] is True and "data" in ok
        assert failed["success"] is False and "error" in failed
