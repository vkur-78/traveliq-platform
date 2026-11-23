"""
Tests for TRAVELIQ HTTP API Client error handling & connection failures (app/api_client.py).
"""

from unittest.mock import MagicMock, patch

import requests

from app.api_client import (
    check_api_health,
    request_flight_prediction,
    request_gender_prediction,
)


def test_check_api_health_online():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"status": "healthy", "model_source": "mlflow_registry"}
    with patch("app.api_client.requests.get", return_value=mock_resp):
        status = check_api_health("http://127.0.0.1:5001")
    assert status["online"] is True
    assert status["status"] == "healthy"
    assert status["details"]["model_source"] == "mlflow_registry"


def test_check_api_health_connection_error():
    with patch("app.api_client.requests.get", side_effect=requests.exceptions.ConnectionError()):
        status = check_api_health("http://127.0.0.1:5001")
    assert status["online"] is False
    assert status["status"] == "offline"
    assert "Unable to reach" in status["error"]


def test_check_api_health_timeout():
    with patch("app.api_client.requests.get", side_effect=requests.exceptions.Timeout()):
        status = check_api_health("http://127.0.0.1:5002")
    assert status["online"] is False
    assert status["status"] == "timeout"


def test_flight_prediction_connection_failure():
    with patch("app.api_client.requests.post", side_effect=requests.exceptions.ConnectionError()):
        res, err = request_flight_prediction({"from": "A"}, base_url="http://127.0.0.1:5001")
    assert res is None
    assert "Unable to connect to the Flight Price API" in err


def test_flight_prediction_timeout():
    with patch("app.api_client.requests.post", side_effect=requests.exceptions.Timeout()):
        res, err = request_flight_prediction({"from": "A"}, base_url="http://127.0.0.1:5001")
    assert res is None
    assert "timed out" in err


def test_flight_prediction_http_400_error_parsing():
    mock_resp = MagicMock(status_code=400)
    mock_resp.json.return_value = {"error": "Invalid date format.", "message": "Use YYYY-MM-DD."}
    with patch("app.api_client.requests.post", return_value=mock_resp):
        res, err = request_flight_prediction({"from": "A"}, base_url="http://127.0.0.1:5001")
    assert res is None
    assert "Invalid date format" in err
    assert "YYYY-MM-DD" in err


def test_gender_prediction_connection_failure():
    with patch("app.api_client.requests.post", side_effect=requests.exceptions.ConnectionError()):
        res, err = request_gender_prediction({"name": "Test"}, base_url="http://127.0.0.1:5002")
    assert res is None
    assert "Unable to connect to the Gender Classification API" in err


def test_gender_prediction_success():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"predicted_gender": "female", "confidence": 0.98}
    with patch("app.api_client.requests.post", return_value=mock_resp):
        res, err = request_gender_prediction({"name": "Wilma"}, base_url="http://127.0.0.1:5002")
    assert err is None
    assert res["predicted_gender"] == "female"
