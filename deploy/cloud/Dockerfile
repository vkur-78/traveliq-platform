# TRAVELIQ — Production Cloud Container (Hugging Face Spaces / Generic Docker Cloud)
# Runs internal-only Flight Price API (:5001) + Gender Classification API (:5002) via Gunicorn
# and serves ONLY the public TRAVELIQ Streamlit UI on HTTPS-fronted port 7860.
# MLflow admin UI is intentionally NOT exposed publicly.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    FLIGHT_API_URL=http://127.0.0.1:5001 \
    GENDER_API_URL=http://127.0.0.1:5002 \
    MLFLOW_TRACKING_URI=disabled \
    ENABLE_LOCAL_MODEL_FALLBACK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# libgomp1 required for LightGBM OpenMP runtime; curl required for startup & health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root UID 1000 (required by Hugging Face Spaces security sandbox)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user:user deploy/cloud/requirements-cloud.txt /home/user/app/requirements-cloud.txt
RUN pip install --no-cache-dir -r /home/user/app/requirements-cloud.txt

# Copy only runtime production code, assets, datasets, and trained model artifacts
COPY --chown=user:user .streamlit /home/user/app/.streamlit
COPY --chown=user:user app /home/user/app/app
COPY --chown=user:user assets /home/user/app/assets
COPY --chown=user:user models/flight_price_pipeline.joblib /home/user/app/models/flight_price_pipeline.joblib
COPY --chown=user:user models/gender_classifier.joblib /home/user/app/models/gender_classifier.joblib
COPY --chown=user:user travel_capstone_dataset /home/user/app/travel_capstone_dataset
COPY --chown=user:user streamlit_app.py /home/user/app/streamlit_app.py
COPY --chown=user:user ATTRIBUTION.md /home/user/app/ATTRIBUTION.md
COPY --chown=user:user deploy/cloud/start_traveliq.sh /home/user/app/start_traveliq.sh

RUN sed -i 's/\r$//' /home/user/app/start_traveliq.sh && chmod +x /home/user/app/start_traveliq.sh

USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:${PATH}"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5001/health && curl -fsS http://127.0.0.1:5002/health && curl -fsS http://127.0.0.1:${PORT}/_stcore/health || exit 1

CMD ["/home/user/app/start_traveliq.sh"]
