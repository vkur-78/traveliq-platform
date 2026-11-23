"""
TRAVELIQ — Flight Price Prediction REST API.

Exposes:
  - GET  /health  : Liveness, readiness, and active model lineage details.
  - POST /predict : Validates flight inputs and returns predicted ticket price.

Loads the MLflow-registered `@champion` model once at startup with a safe,
automatic fallback to the local `models/flight_price_pipeline.joblib` artifact.
"""

import logging
import math
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, request

try:
    import mlflow
    import mlflow.pyfunc
except ImportError:  # Lean cloud containers intentionally omit MLflow admin/client deps
    mlflow = None  # type: ignore[assignment]

from .config import (
    DEBUG,
    ENABLE_LOCAL_MODEL_FALLBACK,
    FLASK_HOST,
    FLASK_PORT,
    FLIGHT_LOCAL_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    PROJECT_ROOT,
    WSGI_THREADS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("traveliq.flight_api")

REQUIRED_FIELDS = [
    "from",
    "to",
    "flightType",
    "time",
    "distance",
    "agency",
    "date",
]


def _resolve_mlflow_uri(configured_uri: str) -> Optional[str]:
    """Determine the best available MLflow tracking URI without hanging."""
    if configured_uri:
        if configured_uri.strip().lower() in {"disabled", "none", "off", "false"}:
            return None
        return configured_uri

    if Path("/.dockerenv").exists():
        return "http://host.docker.internal:5000"

    local_http = "http://127.0.0.1:5000"
    try:
        with urllib.request.urlopen(f"{local_http}/health", timeout=1.0):
            return local_http
    except Exception:
        sqlite_db = PROJECT_ROOT / "mlflow.db"
        if sqlite_db.exists():
            return f"sqlite:///{sqlite_db.resolve().as_posix()}"
        return None


class FlightModelRegistry:
    """Thread-safe singleton loader for the Flight Price model (MLflow + local fallback)."""

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        model_name: str = MODEL_NAME,
        model_alias: str = MODEL_ALIAS,
        local_model_path: Path = FLIGHT_LOCAL_MODEL_PATH,
        enable_fallback: bool = ENABLE_LOCAL_MODEL_FALLBACK,
    ) -> None:
        self.configured_tracking_uri = tracking_uri
        self.resolved_tracking_uri: Optional[str] = None
        self.model_name = model_name
        self.model_alias = model_alias
        self.local_model_path = Path(local_model_path)
        self.enable_fallback = enable_fallback

        self._lock = threading.Lock()
        self._model: Any = None
        self.model_source: str = "unloaded"
        self.fallback_reason: Optional[str] = None
        self.load_error: Optional[str] = None

    def get_model(self) -> Any:
        """Load the model once and cache it across all subsequent requests."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            # 1. Try MLflow Model Registry (@champion alias)
            uri = _resolve_mlflow_uri(self.configured_tracking_uri)
            self.resolved_tracking_uri = uri
            mlflow_error: Optional[str] = None

            if uri and mlflow is not None:
                try:
                    mlflow.set_tracking_uri(uri)
                    model_uri = f"models:/{self.model_name}@{self.model_alias}"
                    logger.info("Loading flight model from MLflow: %s (%s)", model_uri, uri)
                    self._model = mlflow.pyfunc.load_model(model_uri)
                    self.model_source = "mlflow_registry"
                    self.fallback_reason = None
                    self.load_error = None
                    logger.info("Successfully loaded MLflow model %s", model_uri)
                    return self._model
                except Exception as exc:
                    mlflow_error = str(exc)
                    logger.warning("MLflow model load failed (%s): %s", uri, mlflow_error)
            elif mlflow is None:
                mlflow_error = "MLflow client disabled in lean cloud deployment mode."
            else:
                mlflow_error = "MLflow disabled or no reachable MLflow server / local mlflow.db found."

            # 2. Safe local joblib fallback
            if self.enable_fallback and self.local_model_path.exists():
                try:
                    logger.info(
                        "Falling back to local flight model artifact: %s",
                        self.local_model_path,
                    )
                    self._model = joblib.load(self.local_model_path)
                    self.model_source = "local_joblib_fallback"
                    self.fallback_reason = mlflow_error
                    self.load_error = None
                    logger.info("Successfully loaded local fallback flight model.")
                    return self._model
                except Exception as exc:
                    self.load_error = f"MLflow failed ({mlflow_error}); local fallback failed ({exc})"
                    logger.error(self.load_error)
                    raise RuntimeError(self.load_error) from exc

            self.load_error = (
                f"Unable to load flight model. MLflow error: {mlflow_error}. "
                f"Local fallback path '{self.local_model_path}' "
                f"{'disabled' if not self.enable_fallback else 'not found'}."
            )
            logger.error(self.load_error)
            raise RuntimeError(self.load_error)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "alias": self.model_alias,
            "model_source": self.model_source,
            "tracking_uri": self.resolved_tracking_uri,
            "fallback_reason": self.fallback_reason,
        }


def _parse_positive_float(value: Any, field_name: str) -> Tuple[Optional[float], Optional[str]]:
    if isinstance(value, bool):
        return None, f"'{field_name}' must be a numeric value, not a boolean."
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"'{field_name}' must be numeric."
    if not math.isfinite(parsed) or parsed <= 0:
        return None, f"'{field_name}' must be a finite number greater than 0."
    return parsed, None


def _validate_non_empty_string(value: Any, field_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(value, str) or not value.strip():
        return None, f"'{field_name}' must be a non-empty string."
    return value.strip(), None


def create_app(model_registry: Optional[FlightModelRegistry] = None) -> Flask:
    """Application factory for TRAVELIQ Flight Price API."""
    flask_app = Flask(__name__)
    registry = model_registry or FlightModelRegistry()
    flask_app.config["MODEL_REGISTRY"] = registry

    # Attempt eager startup load once (does not crash import; health reports status cleanly)
    try:
        registry.get_model()
    except Exception as exc:
        logger.warning("Startup model preload deferred/failed: %s", exc)

    @flask_app.get("/health")
    def health():
        try:
            registry.get_model()
            meta = registry.metadata()
            return (
                jsonify(
                    {
                        "service": "traveliq-flight-api",
                        "status": "healthy",
                        "model": meta["model"],
                        "alias": meta["alias"],
                        "model_source": meta["model_source"],
                    }
                ),
                200,
            )
        except Exception as exc:
            return (
                jsonify(
                    {
                        "service": "traveliq-flight-api",
                        "status": "unhealthy",
                        "error": "Flight prediction model is unavailable.",
                        "message": str(exc),
                    }
                ),
                503,
            )

    @flask_app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return (
                jsonify(
                    {
                        "error": "Invalid JSON body.",
                        "message": "Request body must contain a valid JSON object.",
                    }
                ),
                400,
            )

        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
        if missing_fields:
            return (
                jsonify(
                    {
                        "error": "Missing required fields.",
                        "missing_fields": missing_fields,
                    }
                ),
                400,
            )

        # Validate categorical string inputs
        parsed_strings: Dict[str, str] = {}
        for field in ("from", "to", "flightType", "agency"):
            val, err = _validate_non_empty_string(data[field], field)
            if err:
                return jsonify({"error": "Invalid input field.", "message": err}), 400
            parsed_strings[field] = val  # type: ignore[assignment]

        if parsed_strings["from"].lower() == parsed_strings["to"].lower():
            return (
                jsonify(
                    {
                        "error": "Invalid route.",
                        "message": "Origin ('from') and destination ('to') must be different cities.",
                    }
                ),
                400,
            )

        # Validate positive numeric inputs
        flight_time, err = _parse_positive_float(data["time"], "time")
        if err:
            return jsonify({"error": "Invalid numeric input.", "message": err}), 400

        distance, err = _parse_positive_float(data["distance"], "distance")
        if err:
            return jsonify({"error": "Invalid numeric input.", "message": err}), 400

        # Validate date string (YYYY-MM-DD)
        raw_date = data["date"]
        if not isinstance(raw_date, str):
            return (
                jsonify(
                    {
                        "error": "Invalid date format.",
                        "message": "Use YYYY-MM-DD string format.",
                    }
                ),
                400,
            )
        try:
            prediction_date = datetime.strptime(raw_date.strip(), "%Y-%m-%d")
        except ValueError:
            return (
                jsonify(
                    {
                        "error": "Invalid date format.",
                        "message": "Use YYYY-MM-DD format (e.g., 2026-09-20).",
                    }
                ),
                400,
            )

        # Ensure cached model is ready
        try:
            active_model = registry.get_model()
        except Exception as exc:
            return (
                jsonify(
                    {
                        "error": "Model unavailable.",
                        "message": str(exc),
                    }
                ),
                503,
            )

        input_df = pd.DataFrame(
            {
                "from": pd.Series([parsed_strings["from"]], dtype="object"),
                "to": pd.Series([parsed_strings["to"]], dtype="object"),
                "flightType": pd.Series([parsed_strings["flightType"]], dtype="object"),
                "agency": pd.Series([parsed_strings["agency"]], dtype="object"),
                "time": pd.Series([flight_time], dtype="float64"),
                "distance": pd.Series([distance], dtype="float64"),
                "year": pd.Series([prediction_date.year], dtype="int32"),
                "month": pd.Series([prediction_date.month], dtype="int32"),
                "day": pd.Series([prediction_date.day], dtype="int32"),
                "day_of_week": pd.Series([prediction_date.weekday()], dtype="int32"),
            }
        )

        try:
            prediction = active_model.predict(input_df)
            predicted_price = round(float(prediction[0]), 2)
        except Exception as exc:
            logger.exception("Flight price inference failed.")
            return (
                jsonify(
                    {
                        "error": "Prediction failed.",
                        "message": str(exc),
                    }
                ),
                500,
            )

        meta = registry.metadata()
        return (
            jsonify(
                {
                    "predicted_price": predicted_price,
                    "currency": "USD",
                    "model": meta["model"],
                    "model_alias": meta["alias"],
                    "model_source": meta["model_source"],
                    "input_summary": {
                        "from": parsed_strings["from"],
                        "to": parsed_strings["to"],
                        "flightType": parsed_strings["flightType"],
                        "agency": parsed_strings["agency"],
                        "time_hours": flight_time,
                        "distance_km": distance,
                        "date": raw_date.strip(),
                    },
                }
            ),
            200,
        )

    return flask_app


# Default WSGI app callable (compatible with gunicorn / waitress)
app = create_app()


def serve_production() -> None:
    """Run the Flight API with Waitress WSGI in production mode, or Flask debug server."""
    if DEBUG:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
        return

    try:
        from waitress import serve

        logger.info(
            "Starting TRAVELIQ Flight API via Waitress WSGI on http://%s:%s (threads=%s)",
            FLASK_HOST,
            FLASK_PORT,
            WSGI_THREADS,
        )
        serve(app, host=FLASK_HOST, port=FLASK_PORT, threads=WSGI_THREADS)
    except ImportError:
        logger.warning("Waitress not installed; falling back to Flask built-in server.")
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    serve_production()
