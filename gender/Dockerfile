# ============================================================
# TRAVELIQ — Lightweight Gender Classification API Container
# Image: traveliq-gender-api:2.0
# Build: docker build -f gender/Dockerfile -t traveliq-gender-api:2.0 .
#
# Uses the Phase 1 40.4 KB pure scikit-learn binary pipeline.
# No PyTorch, CUDA, or SentenceTransformer dependencies required.
# ============================================================

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install curl for container health checks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned lightweight scikit-learn dependencies
COPY gender/requirements-gender.txt ./requirements-gender.txt
RUN pip install --upgrade pip \
    && pip install -r requirements-gender.txt

# Copy only required gender API modules and the 40.4 KB model artifact
COPY app/__init__.py ./app/__init__.py
COPY app/config.py ./app/config.py
COPY app/gender_features.py ./app/gender_features.py
COPY app/gender_api.py ./app/gender_api.py
COPY models/gender_classifier.joblib ./models/gender_classifier.joblib

# Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV GENDER_API_HOST=0.0.0.0 \
    GENDER_API_PORT=5002

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:5002/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--workers", "2", "--threads", "2", "--timeout", "30", "app.gender_api:app"]
