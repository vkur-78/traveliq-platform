"""
TRAVELIQ — Centralized Environment Configuration.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_setting(key: str, default: str = "") -> str:
    """Read setting from environment first, falling back safely to Streamlit Cloud st.secrets."""
    val = os.getenv(key)
    if val is not None and str(val).strip() != "":
        return str(val).strip()
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default

# Data & Model Paths
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(PROJECT_ROOT / "travel_capstone_dataset")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))

FLIGHT_LOCAL_MODEL_PATH = Path(
    os.getenv("FLIGHT_LOCAL_MODEL_PATH", str(MODELS_DIR / "flight_price_pipeline.joblib"))
)
GENDER_MODEL_PATH = Path(
    os.getenv("GENDER_MODEL_PATH", str(MODELS_DIR / "gender_classifier.joblib"))
)

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Voyage_Flight_Price_Model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
ENABLE_LOCAL_MODEL_FALLBACK = os.getenv("ENABLE_LOCAL_MODEL_FALLBACK", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Flight API Server Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))

# Gender API Server Configuration
GENDER_API_HOST = os.getenv("GENDER_API_HOST", "0.0.0.0")
GENDER_API_PORT = int(os.getenv("GENDER_API_PORT", "5002"))

# WSGI / Runtime Settings
WSGI_THREADS = int(os.getenv("WSGI_THREADS", "4"))
DEBUG = os.getenv("FLASK_DEBUG", "0").strip() == "1"

# Frontend -> Backend API URLs (for Streamlit — supports env vars & Streamlit Cloud st.secrets)
FLIGHT_API_URL = (
    _get_setting("FLIGHT_API_URL", "")
    or _get_setting("FLASK_API_URL", "http://127.0.0.1:5001")
).rstrip("/")
GENDER_API_URL = _get_setting("GENDER_API_URL", "http://127.0.0.1:5002").rstrip("/")
API_TIMEOUT_SECONDS = float(_get_setting("API_TIMEOUT_SECONDS", "10"))
HEALTH_TIMEOUT_SECONDS = float(_get_setting("HEALTH_TIMEOUT_SECONDS", "3.0"))
