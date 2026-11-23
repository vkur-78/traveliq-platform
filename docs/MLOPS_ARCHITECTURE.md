# TRAVELIQ — MLOps Architecture & Infrastructure Guide

This document explains **why** each infrastructure component exists in **TRAVELIQ — Travel Intelligence Platform**, how the components actually communicate based on the implemented files in this repository, and how they work together end-to-end.

> **Attribution Notice:** TRAVELIQ was redesigned and extended from the open-source `voyage-analytics-mlops` starter repository. All Phase 1 ML workflows, Phase 2 TRAVELIQ application services, and Phase 3 infrastructure configurations (`docker-compose.yml`, `docker/Dockerfile.*`, `kubernetes/*`, `scripts/mlops_pipeline.py`, `airflow/dags/traveliq_ml_pipeline.py`, and `Jenkinsfile`) were custom-engineered specifically for TRAVELIQ.

---

## 1. End-to-End TRAVELIQ System Architecture

```text
                      ┌────────────────────────────────────────┐
                      │         Jenkins CI/CD Pipeline         │
                      │  Checkout → Deps → Quality → Pytest    │
                      │  → Artifacts → Docker Build → Dry-Run  │
                      └───────────────────┬────────────────────┘
                                          │ validates & builds images
                                          ▼
┌────────────────────┐      ┌───────────────────────────────┐      ┌──────────────────────────┐
│ Travel Datasets    │─────►│   Apache Airflow DAG          │─────►│ MLflow Tracking &        │
│ flights / hotels / │      │   traveliq_ml_pipeline        │      │ Model Registry (:5000)   │
│ users (.csv)       │      │   1. validate_datasets        │      │ - Experiments & metrics  │
└─────────┬──────────┘      │   2. preprocess_retrain_gate  │      │ - Voyage_Flight_Price    │
          │                 │   3. train flight / gender /  │      │   Model (@champion)      │
          │                 │      build recommender        │      │ - TravelIQ_Gender_       │
          │                 │   4. evaluate_models (gates)  │      │   Classifier (@champion) │
          │                 │   5. register & promote       │      └────────────┬─────────────┘
          │                 └───────────────────────────────┘                   │ loads @champion
          │                                                                     │ (w/ local joblib fallback)
          │         ┌───────────────────────────────────────────────────────────┘
          │         ▼
          │   ┌────────────────────────────────────────────────────────────────────┐
          │   │                Docker Compose / Kubernetes Cluster                 │
          │   │                                                                    │
          │   │   ┌───────────────────────┐      ┌───────────────────────────┐     │
          │   │   │ traveliq-flight-api   │      │ traveliq-gender-api       │     │
          │   │   │ Gunicorn · Port 5001  │      │ Gunicorn · Port 5002      │     │
          │   │   │ HPA: 2 → 5 pods (70%) │      │ HPA: 1 → 4 pods (70%)     │     │
          │   │   └───────────▲───────────┘      └─────────────▲─────────────┘     │
          │   │               │ HTTP /predict                  │ HTTP /predict-gender
          │   │               └──────────────┬─────────────────┘                   │
          │   │                              │                                     │
          │   │               ┌──────────────┴───────────────┐                     │
          └──►└──────────────►│     traveliq-streamlit       │◄──── User Browser   │
                              │     Port 8501 (NodePort      │      :8501          │
                              │     30851) + Recommender     │                     │
                              └──────────────────────────────┘                     │
              └────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why Docker Is Used & How Containers Communicate

### Why Docker Matters for TRAVELIQ
TRAVELIQ combines four distinct runtime responsibilities with different system and library dependencies:
1. **`traveliq-flight-api:2.0`** ([docker/Dockerfile](../docker/Dockerfile)) needs `libgomp1` for LightGBM OpenMP threading plus `mlflow==3.18.0`, `lightgbm==4.7.0`, `scikit-learn==1.9.1`, and `gunicorn==23.0.0`.
2. **`traveliq-gender-api:2.0`** ([gender/Dockerfile](../gender/Dockerfile)) serves the lightweight **40.4 KB** pure `scikit-learn` pipeline from Phase 1—removing PyTorch and `sentence-transformers` completely shrinks the image by ~2 GB and brings cold-start time under 3 seconds.
3. **`traveliq-streamlit:2.0`** ([docker/Dockerfile.streamlit](../docker/Dockerfile.streamlit)) runs Streamlit, Plotly, and the shared `TravelPackageRecommender` without installing LightGBM or MLflow.
4. **`traveliq-mlflow:2.0`** ([docker/Dockerfile.mlflow](../docker/Dockerfile.mlflow)) hosts the MLflow Tracking Server and Model Registry backed by a persistent volume (`traveliq_mlflow_data`).

Separating these into lean `python:3.13-slim` non-root containers with pinned dependency files guarantees reproducible builds, prevents version drift against serialized `.joblib` models (`scikit-learn==1.9.1`), and isolates failures.

### How Containers Communicate in Docker Compose
In [docker-compose.yml](../docker-compose.yml), all four containers share a dedicated user-defined bridge network (`traveliq-net`) so Docker's embedded DNS resolves service names directly:
- `streamlit-ui` calls `FLIGHT_API_URL=http://flight-api:5001` and `GENDER_API_URL=http://gender-api:5002`.
- `flight-api` connects to `MLFLOW_TRACKING_URI=http://mlflow:5000` (with `--allowed-hosts` explicitly permitting `mlflow:5000` to avoid MLflow 3.x HTTP 403 host-header rejections) and mounts `./models:/app/models:ro` for automatic local fallback if the registry is empty or temporarily unreachable.
- `depends_on` with `condition: service_healthy` ensures `mlflow` passes `/health` before `flight-api` boots, and both Flask APIs pass `/health` before `streamlit-ui` starts serving traffic.

---

## 3. Why Kubernetes Is Used & How HPA Works

### Why Kubernetes Fits TRAVELIQ
While Docker Compose is ideal for single-host development, Kubernetes ([kubernetes/](../kubernetes)) provides production orchestration:
- **Zero-Downtime High Availability:** `traveliq-flight-api`, `traveliq-gender-api`, and `traveliq-streamlit` each run `replicas: 2` behind stable Kubernetes Services (`traveliq-flight-api:5001`, `traveliq-gender-api:5002`, `traveliq-streamlit:8501`).
- **Centralized Non-Secret Configuration:** [kubernetes/configmap.yaml](../kubernetes/configmap.yaml) injects internal service DNS (`http://traveliq-mlflow:5000`, `http://traveliq-flight-api:5001`, `http://traveliq-gender-api:5002`) into pods via `envFrom`—eliminating fragile `host.docker.internal` assumptions.
- **Self-Healing Probes:** Every Deployment defines `startupProbe`, `readinessProbe`, and `livenessProbe` on `/health` (Flask APIs) and `/_stcore/health` (Streamlit). Unhealthy pods are removed from Service endpoints automatically and restarted if deadlocked.
- **Security & Secret Hygiene:** Pods run with `runAsNonRoot: true`, explicit CPU/memory requests & limits, and zero embedded secrets ([kubernetes/secret.example.yaml](../kubernetes/secret.example.yaml) is provided strictly as a template if connecting external cloud DB/S3 storage).

### How Horizontal Pod Autoscaling (HPA) Works in TRAVELIQ
In [kubernetes/hpa.yaml](../kubernetes/hpa.yaml):
- **`traveliq-flight-api-hpa`** monitors CPU utilization (target **70%** of the `250m` CPU request) and memory utilization (target **80%** of `512Mi`) via Kubernetes `metrics-server`, automatically scaling `traveliq-flight-api` between **`minReplicas: 2`** and **`maxReplicas: 5`** when flight-search inference spikes occur.
- **`traveliq-gender-api-hpa`** scales `traveliq-gender-api` between **`minReplicas: 1`** and **`maxReplicas: 4`** at **70%** CPU utilization.
- Both HPAs define a **`300s` scale-down stabilization window** (`scaleDown.policies: max 1 pod / 60s`) to prevent replica flapping during bursty travel-booking traffic.

---

## 4. How Apache Airflow Orchestrates Model Workflows

Implemented in [airflow/dags/traveliq_ml_pipeline.py](../airflow/dags/traveliq_ml_pipeline.py) and [scripts/mlops_pipeline.py](../scripts/mlops_pipeline.py):

1. **`validate_datasets`**: Verifies `flights.csv`, `hotels.csv`, and `users.csv` existence, schema, zero nulls, positive ranges, valid dates, `"none"` gender audit, and computes a deterministic SHA-256 dataset fingerprint.
2. **`preprocess_and_retrain_gate` (`ShortCircuitOperator`)**: Checks preprocessing health and compares the dataset SHA-256 fingerprint against `models/pipeline_state.json`. If the data is unchanged and verified models already exist (unless `TRAVELIQ_FORCE_RETRAIN=1`), downstream training tasks are cleanly skipped—preventing unnecessary compute and redundant model versions.
3. **Parallel Candidate Training:**
   - **`train_flight_model`**: Leakage-free `GroupShuffleSplit` LightGBM evaluation (`MAE`, `MSE`, `RMSE`, `R²`) + candidate save (`models/candidates/flight_price_pipeline.joblib`) + MLflow candidate run logging.
   - **`train_gender_model`**: Binary `male`/`female` Logistic Regression (excluding `"none"`) + candidate save + MLflow candidate run logging.
   - **`build_recommendation_catalog`**: Builds the 1,568-package hybrid recommender catalog and validates budget compliance across 50 randomized queries.
4. **`evaluate_models` (Strict Promotion Quality Gates)**: Blocks promotion with `RuntimeError` if ANY candidate fails:
   - Flight LightGBM: `R² >= 0.95`, `RMSE <= 65.0`, finite positive smoke inference.
   - Gender Binary Classifier: `weighted F1 >= 0.85`, `ROC-AUC >= 0.90`, artifact size `< 500 KB`, valid binary smoke inference.
   - Recommender: `budget_compliance == 100%`, `package_count >= 1,000`.
5. **`register_and_promote_champion`**: Runs **only if `evaluate_models` passes**—promotes candidate `.joblib` artifacts into `models/`, registers both models in MLflow Model Registry, updates the `@champion` alias, and persists `models/pipeline_state.json`.

All tasks configure `retries=2`, `retry_delay=2m`, `execution_timeout=20m`, `max_active_runs=1`, and a structured `on_failure_callback`. [airflow/docker-compose.yml](../airflow/docker-compose.yml) enforces Postgres `service_healthy` and `airflow-init` `service_completed_successfully` startup ordering and uses environment-variable admin credentials.

---

## 5. How Jenkins Automates CI/CD

Implemented in [Jenkinsfile](../Jenkinsfile) using a cross-platform `runCmd()` helper supporting Linux (`sh`) and Windows (`bat`) agents:

| Stage | What It Verifies | Failure Behavior |
|---|---|---|
| **1. Checkout** | Checks out source revision from Git SCM | Stops on SCM failure |
| **2. Python Env & Dependencies** | Builds clean `.ci-venv`, installs `requirements-dev.txt`, runs `pip check` | Fails on missing/conflicting packages |
| **3. Code Quality Checks** | Parses AST of all `.py` files and parses all Kubernetes & Compose `.yaml` files | Fails on any syntax or YAML error |
| **4. Automated Pytest Suite** | Runs `pytest -v --junitxml=reports/junit.xml` across APIs, recommender, client, and MLOps gates | Fails immediately if any test fails; archives JUnit XML |
| **5. Notebook & Artifact Validation** | Ensures all 3 notebooks contain 0 error cells; loads both `.joblib` artifacts and verifies gender artifact `< 500 KB` | Fails on broken notebooks or corrupted/bloated artifacts |
| **6. Docker Image Build** | Builds all 4 tagged images (`:2.0`) and runs container dependency smoke tests | Fails on build or import errors |
| **7. Deployment Validation** | Runs `docker compose config --quiet` and `kubectl apply --dry-run=client` on all manifests | Validates deployability **without** auto-deploying to production |

---

## 6. How MLflow Tracks & Registers Models

- **Experiment Tracking:** Both [notebooks/01_flight_price_prediction.ipynb](../notebooks/01_flight_price_prediction.ipynb) and [scripts/mlops_pipeline.py](../scripts/mlops_pipeline.py) log hyperparameters, group-split metrics (`MAE`, `MSE`, `RMSE`, `R²`), input schemas, and `scikit-learn` pipelines to MLflow.
- **Model Registry & `@champion` Alias:** Validated models are registered as `Voyage_Flight_Price_Model` (and `TravelIQ_Gender_Classifier`) with versioned releases and promoted via `MlflowClient.set_registered_model_alias(..., "champion", version)`.
- **Decoupled Serving with Resilience:** [app/app.py](../app/app.py) loads `models:/Voyage_Flight_Price_Model@champion` once at startup (never per request) and automatically falls back to `models/flight_price_pipeline.joblib` if MLflow is unreachable, exposing `model_source` transparently via `GET /health`.
