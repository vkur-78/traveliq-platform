"""
TRAVELIQ — Lightweight Gender Classification REST API.

Exposes:
  - GET  /health         : Service readiness and model metadata.
  - POST /predict-gender : Validates name/company/age and predicts binary gender
                           using the Phase 1 40.4 KB pure scikit-learn pipeline.

Obsolete PyTorch / SentenceTransformer dependencies have been completely removed.
"""

import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from .config import (
    DEBUG,
    GENDER_API_HOST,
    GENDER_API_PORT,
    GENDER_MODEL_PATH,
    WSGI_THREADS,
)
from .gender_features import EXPERIMENTAL_GENDER_NOTICE, build_gender_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("traveliq.gender_api")

REQUIRED_FIELDS = ["name", "company", "age"]
MIN_AGE = 1.0
MAX_AGE = 120.0


class GenderModelRegistry:
    """Thread-safe singleton loader for the lightweight Phase 1 sklearn gender artifact."""

    def __init__(self, model_path: Path = GENDER_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._bundle: Optional[Dict[str, Any]] = None
        self.artifact_size_kb: Optional[float] = None
        self.load_error: Optional[str] = None

    def get_bundle(self) -> Dict[str, Any]:
        """Load the lightweight joblib bundle once and cache it for all requests."""
        if self._bundle is not None:
            return self._bundle

        with self._lock:
            if self._bundle is not None:
                return self._bundle

            if not self.model_path.exists():
                self.load_error = f"Gender model artifact not found: {self.model_path}"
                logger.error(self.load_error)
                raise FileNotFoundError(self.load_error)

            try:
                loaded = joblib.load(self.model_path)
                if not isinstance(loaded, dict) or "pipeline" not in loaded:
                    raise ValueError(
                        "Invalid gender artifact format: expected Phase 1 lightweight "
                        "bundle containing a scikit-learn 'pipeline' key."
                    )
                self._bundle = loaded
                self.artifact_size_kb = round(self.model_path.stat().st_size / 1024.0, 1)
                self.load_error = None
                logger.info(
                    "Loaded lightweight gender model (%s KB, version=%s)",
                    self.artifact_size_kb,
                    loaded.get("artifact_version", "unknown"),
                )
                return self._bundle
            except Exception as exc:
                self.load_error = str(exc)
                logger.exception("Failed to load gender classifier artifact.")
                raise

    def metadata(self) -> Dict[str, Any]:
        bundle = self.get_bundle()
        pipeline = bundle["pipeline"]
        classes: List[str] = [str(c) for c in bundle.get("classes", list(pipeline.classes_))]
        return {
            "model": "gender_classifier",
            "model_type": bundle.get("model_type", "scikit-learn LogisticRegression Pipeline"),
            "artifact_version": bundle.get("artifact_version", "2.0-lightweight-binary"),
            "classes": classes,
            "artifact_size_kb": self.artifact_size_kb,
        }


def _validate_string(value: Any, field_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(value, str) or not value.strip():
        return None, f"'{field_name}' must be a non-empty string."
    cleaned = value.strip()
    if len(cleaned) > 120:
        return None, f"'{field_name}' must be 120 characters or fewer."
    return cleaned, None


def _validate_age(value: Any) -> Tuple[Optional[float], Optional[str]]:
    if isinstance(value, bool):
        return None, "'age' must be numeric, not a boolean."
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, "'age' must be numeric."
    if not math.isfinite(parsed) or parsed < MIN_AGE or parsed > MAX_AGE:
        return None, f"'age' must be between {int(MIN_AGE)} and {int(MAX_AGE)}."
    return parsed, None


def create_app(model_registry: Optional[GenderModelRegistry] = None) -> Flask:
    """Application factory for TRAVELIQ Gender Classification API."""
    flask_app = Flask(__name__)
    registry = model_registry or GenderModelRegistry()
    flask_app.config["MODEL_REGISTRY"] = registry

    try:
        registry.get_bundle()
    except Exception as exc:
        logger.warning("Gender model startup preload deferred/failed: %s", exc)

    @flask_app.get("/health")
    def health():
        try:
            meta = registry.metadata()
            return (
                jsonify(
                    {
                        "service": "traveliq-gender-api",
                        "status": "healthy",
                        **meta,
                    }
                ),
                200,
            )
        except Exception as exc:
            return (
                jsonify(
                    {
                        "service": "traveliq-gender-api",
                        "status": "unhealthy",
                        "error": "Gender classifier model is unavailable.",
                        "message": str(exc),
                    }
                ),
                503,
            )

    @flask_app.post("/predict-gender")
    def predict_gender():
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

        name, err = _validate_string(data["name"], "name")
        if err:
            return jsonify({"error": "Invalid input field.", "message": err}), 400

        company, err = _validate_string(data["company"], "company")
        if err:
            return jsonify({"error": "Invalid input field.", "message": err}), 400

        age, err = _validate_age(data["age"])
        if err:
            return jsonify({"error": "Invalid age input.", "message": err}), 400

        try:
            bundle = registry.get_bundle()
            pipeline = bundle["pipeline"]
            meta = registry.metadata()
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

        try:
            raw_df = pd.DataFrame([{"name": name, "company": company, "age": age}])
            features_df = build_gender_features(raw_df)
            prediction = pipeline.predict(features_df)
            predicted_gender = str(prediction[0])

            probabilities: Dict[str, float] = {}
            confidence: Optional[float] = None
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(features_df)[0]
                classes = [str(c) for c in pipeline.classes_]
                probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
                confidence = round(float(max(probs)), 4)
        except Exception as exc:
            logger.exception("Gender inference failed.")
            return (
                jsonify(
                    {
                        "error": "Gender prediction failed.",
                        "message": str(exc),
                    }
                ),
                500,
            )

        return (
            jsonify(
                {
                    "predicted_gender": predicted_gender,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "model": meta["model"],
                    "artifact_version": meta["artifact_version"],
                    "experimental": True,
                    "experimental_notice": EXPERIMENTAL_GENDER_NOTICE,
                }
            ),
            200,
        )

    return flask_app


# Default WSGI app callable (compatible with gunicorn / waitress)
app = create_app()


def serve_production() -> None:
    """Run the Gender API with Waitress WSGI in production mode, or Flask debug server."""
    if DEBUG:
        app.run(host=GENDER_API_HOST, port=GENDER_API_PORT, debug=True)
        return

    try:
        from waitress import serve

        logger.info(
            "Starting TRAVELIQ Gender API via Waitress WSGI on http://%s:%s (threads=%s)",
            GENDER_API_HOST,
            GENDER_API_PORT,
            WSGI_THREADS,
        )
        serve(app, host=GENDER_API_HOST, port=GENDER_API_PORT, threads=WSGI_THREADS)
    except ImportError:
        logger.warning("Waitress not installed; falling back to Flask built-in server.")
        app.run(host=GENDER_API_HOST, port=GENDER_API_PORT, debug=False)


if __name__ == "__main__":
    serve_production()
