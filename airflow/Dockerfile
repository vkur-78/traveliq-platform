# ============================================================
# TRAVELIQ — Apache Airflow Worker/Scheduler Custom Image
# Installs pinned ML dependencies matching Phase 1 & Phase 2.
# ============================================================

FROM apache/airflow:2.10.5

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    pandas==3.1.1 \
    numpy==2.5.5 \
    scipy==1.18.1 \
    scikit-learn==1.9.1 \
    lightgbm==4.7.0 \
    mlflow==3.18.0 \
    skops==0.16.0 \
    joblib==1.6.0 \
    psutil==7.2.2 \
    cryptography==44.0.2
