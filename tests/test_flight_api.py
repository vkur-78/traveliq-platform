"""
Tests for TRAVELIQ Flight Price Prediction Flask API (app/app.py).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.app import FlightModelRegistry, create_app
from app.config import FLIGHT_LOCAL_MODEL_PATH


@pytest.fixture()
def flight_client():
    """Flight API test client backed by real Phase 1 local fallback or MLflow model."""
    registry = FlightModelRegistry(
        tracking_uri="",
        local_model_path=FLIGHT_LOCAL_MODEL_PATH,
        enable_fallback=True,
    )
    app = create_app(model_registry=registry)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_flight_api_health_ok(flight_client):
    resp = flight_client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "healthy"
    assert body["service"] == "traveliq-flight-api"
    assert body["model_source"] in ("mlflow_registry", "local_joblib_fallback")


def test_flight_api_health_unavailable_returns_503(tmp_path):
    registry = FlightModelRegistry(
        tracking_uri="http://127.0.0.1:59999",
        local_model_path=tmp_path / "does_not_exist.joblib",
        enable_fallback=True,
    )
    with patch("app.app.mlflow.pyfunc.load_model", side_effect=RuntimeError("MLflow down")):
        app = create_app(model_registry=registry)
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/health")
            assert resp.status_code == 503
            body = resp.get_json()
            assert body["status"] == "unhealthy"


def test_flight_api_local_fallback_when_mlflow_fails():
    """Verify safe automatic fallback to local joblib artifact when MLflow is unreachable."""
    registry = FlightModelRegistry(
        tracking_uri="http://127.0.0.1:59999",
        local_model_path=FLIGHT_LOCAL_MODEL_PATH,
        enable_fallback=True,
    )
    with patch("app.app.mlflow.pyfunc.load_model", side_effect=ConnectionError("MLflow unreachable")):
        app = create_app(model_registry=registry)
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["model_source"] == "local_joblib_fallback"


def test_flight_predict_valid_input_and_cached_model_loading():
    """Verify valid prediction and ensure model is loaded once (not per request)."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [1245.678]

    registry = FlightModelRegistry(
        tracking_uri="http://127.0.0.1:59999",
        local_model_path=Path("fake.joblib"),
        enable_fallback=True,
    )
    with patch("app.app.mlflow.pyfunc.load_model", return_value=mock_model) as mock_loader:
        app = create_app(model_registry=registry)
        app.config["TESTING"] = True
        payload = {
            "from": "Florianopolis (SC)",
            "to": "Aracaju (SE)",
            "flightType": "firstClass",
            "agency": "CloudFy",
            "time": 1.76,
            "distance": 676.53,
            "date": "2026-09-20",
        }
        with app.test_client() as client:
            r1 = client.post("/predict", json=payload)
            r2 = client.post("/predict", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.get_json()["predicted_price"] == 1245.68
        assert r1.get_json()["currency"] == "USD"
        assert mock_loader.call_count == 1  # Loaded once at startup, never per request


def test_flight_predict_real_model_inference(flight_client):
    payload = {
        "from": "Florianopolis (SC)",
        "to": "Aracaju (SE)",
        "flightType": "firstClass",
        "agency": "CloudFy",
        "time": 1.76,
        "distance": 676.53,
        "date": "2026-09-20",
    }
    resp = flight_client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body["predicted_price"], float)
    assert 200.0 <= body["predicted_price"] <= 3000.0
    assert body["input_summary"]["from"] == "Florianopolis (SC)"


@pytest.mark.parametrize(
    "bad_payload,expected_fragment",
    [
        ({"from": "Recife (PE)"}, "Missing required fields"),
        (
            {
                "from": "Recife (PE)",
                "to": "Recife (PE)",
                "flightType": "economic",
                "agency": "CloudFy",
                "time": 1.2,
                "distance": 450.0,
                "date": "2026-09-20",
            },
            "Invalid route",
        ),
        (
            {
                "from": "Recife (PE)",
                "to": "Salvador (BH)",
                "flightType": "economic",
                "agency": "CloudFy",
                "time": -1.5,
                "distance": 450.0,
                "date": "2026-09-20",
            },
            "Invalid numeric input",
        ),
        (
            {
                "from": "Recife (PE)",
                "to": "Salvador (BH)",
                "flightType": "economic",
                "agency": "CloudFy",
                "time": 1.2,
                "distance": 0,
                "date": "2026-09-20",
            },
            "Invalid numeric input",
        ),
        (
            {
                "from": "Recife (PE)",
                "to": "Salvador (BH)",
                "flightType": "economic",
                "agency": "CloudFy",
                "time": 1.2,
                "distance": 450.0,
                "date": "20-09-2026",
            },
            "Invalid date format",
        ),
        (
            {
                "from": "   ",
                "to": "Salvador (BH)",
                "flightType": "economic",
                "agency": "CloudFy",
                "time": 1.2,
                "distance": 450.0,
                "date": "2026-09-20",
            },
            "Invalid input field",
        ),
    ],
)
def test_flight_predict_invalid_inputs_return_400(flight_client, bad_payload, expected_fragment):
    resp = flight_client.post("/predict", json=bad_payload)
    assert resp.status_code == 400
    body = resp.get_json()
    assert expected_fragment in body["error"]


def test_flight_predict_non_json_returns_400(flight_client):
    resp = flight_client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
