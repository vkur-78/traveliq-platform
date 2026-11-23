"""
Tests for TRAVELIQ Lightweight Gender Classification Flask API (app/gender_api.py).
"""

from unittest.mock import patch

import pytest

from app.config import GENDER_MODEL_PATH
from app.gender_api import GenderModelRegistry, create_app


@pytest.fixture()
def gender_client():
    """Gender API test client using the real 40.4 KB Phase 1 sklearn artifact."""
    registry = GenderModelRegistry(model_path=GENDER_MODEL_PATH)
    app = create_app(model_registry=registry)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_gender_api_health_ok(gender_client):
    resp = gender_client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "healthy"
    assert body["service"] == "traveliq-gender-api"
    assert body["artifact_version"] == "2.0-lightweight-binary"
    assert set(body["classes"]) == {"female", "male"}
    assert body["artifact_size_kb"] < 200.0  # Lightweight sklearn artifact


def test_gender_api_health_missing_model_returns_503(tmp_path):
    registry = GenderModelRegistry(model_path=tmp_path / "missing_gender.joblib")
    app = create_app(model_registry=registry)
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.get_json()["status"] == "unhealthy"


def test_gender_predict_valid_inputs(gender_client):
    cases = [
        ({"name": "John Smith", "company": "4You", "age": 30}, "male"),
        ({"name": "Wilma Mcinnis", "company": "4You", "age": 48}, "female"),
        ({"name": "Priya Sharma", "company": "Acme Factory", "age": 28}, "female"),
    ]
    for payload, expected_gender in cases:
        resp = gender_client.post("/predict-gender", json=payload)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["predicted_gender"] == expected_gender
        assert 0.5 <= body["confidence"] <= 1.0
        assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
        assert body["experimental"] is True
        assert "experimental" in body["experimental_notice"].lower()


def test_gender_model_loaded_once_not_per_request():
    registry = GenderModelRegistry(model_path=GENDER_MODEL_PATH)
    with patch("app.gender_api.joblib.load", wraps=__import__("joblib").load) as spy_load:
        app = create_app(model_registry=registry)
        app.config["TESTING"] = True
        with app.test_client() as client:
            r1 = client.post("/predict-gender", json={"name": "John Smith", "company": "4You", "age": 30})
            r2 = client.post("/predict-gender", json={"name": "Paula Daniel", "company": "4You", "age": 25})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert spy_load.call_count == 1


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"name": "John Smith"},  # missing company & age
        {"name": "   ", "company": "4You", "age": 30},  # empty name
        {"name": "John Smith", "company": "", "age": 30},  # empty company
        {"name": "John Smith", "company": "4You", "age": "thirty"},  # non-numeric age
        {"name": "John Smith", "company": "4You", "age": -5},  # negative age
        {"name": "John Smith", "company": "4You", "age": 200},  # unrealistic age
        {"name": "John Smith", "company": "4You", "age": True},  # boolean age
    ],
)
def test_gender_predict_invalid_inputs_return_400(gender_client, bad_payload):
    resp = gender_client.post("/predict-gender", json=bad_payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_gender_predict_non_json_returns_400(gender_client):
    resp = gender_client.post("/predict-gender", data="plain text", content_type="text/plain")
    assert resp.status_code == 400
