# ============================================================
# TRAVELIQ — Flight Price Prediction REST API Container
# Image: traveliq-flight-api:2.0
# Build: docker build -f docker/Dockerfile -t traveliq-flight-api:2.0 .
# ============================================================

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install LightGBM OpenMP runtime (libgomp1) and curl for container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Flight API dependencies
COPY docker/requirements-flight.txt ./requirements-flight.txt
RUN pip install --upgrade pip \
    && pip install -r requirements-flight.txt

# Copy only the application modules and Flight model fallback artifact
COPY app/__init__.py ./app/__init__.py
COPY app/config.py ./app/config.py
COPY app/app.py ./app/app.py
COPY models/flight_price_pipeline.joblib ./models/flight_price_pipeline.joblib

# Run as non-root user for container hardening
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default runtime configuration (overridden by Docker Compose / Kubernetes ConfigMap)
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5001 \
    MODEL_NAME=Voyage_Flight_Price_Model \
    MODEL_ALIAS=champion \
    ENABLE_LOCAL_MODEL_FALLBACK=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:5001/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "2", "--timeout", "60", "app.app:app"]
