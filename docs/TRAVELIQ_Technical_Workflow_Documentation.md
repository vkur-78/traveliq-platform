# TRAVELIQ — Travel Intelligence Platform
## Complete Technical Workflow Documentation

> **Document Purpose:** End-to-end engineering documentation of the TRAVELIQ Travel & Tourism Machine Learning and MLOps Capstone Project. Every metric, API output, test result, and screenshot reported in this document comes directly from live execution of this repository; components that could not be executed natively on the non-administrator Windows workstation are explicitly marked as **Runtime Execution Pending**.
>
> **Companion Google-Docs-Ready Format:** A fully formatted Word document containing embedded high-resolution screenshots is available at `docs/TRAVELIQ_Technical_Workflow_Documentation.docx` for direct upload to Google Drive / Google Docs.

---

## 1. Dataset Analysis

TRAVELIQ ingests three relational CSV datasets stored in `travel_capstone_dataset/`:

| File | Dimensions | Primary Keys / Join Keys | Nulls / Exact Duplicates | Operational Scope |
|---|---:|---|---:|---|
| `flights.csv` | `271,888 × 10` | `travelCode`, `userCode` | `0` / `0` | 9 Brazilian regional hub cities, 70 directional routes, 3 cabin classes (`economic`, `premium`, `firstClass`), 3 travel agencies (`CloudFy`, `Rainbow`, `FlyingDrops`), years `2019–2023`. |
| `hotels.csv` | `40,552 × 8` | `travelCode`, `userCode`, `place` | `0` / `0` | 9 partner hotels (`Hotel A`, `Hotel AF`, `Hotel AU`, `Hotel BD`, `Hotel BW`, `Hotel CB`, `Hotel K`, `Hotel Z`, `Hotel C`), stay durations `1–4` days. |
| `users.csv` | `1,340 × 5` | `code` (maps to `userCode`) | `0` nulls (`440` literal `"none"` in `gender`) | 5 corporate accounts (`4You`, `Acme Factory`, `Monsters CYA`, `Umbrella LTDA`, `Wonka Company`), traveler ages `21–65`. |

### Critical Exploratory Findings That Shaped Architecture

1. **Exact Fixed-Price Flight Configurations (`flights.csv`):** Although `flights.csv` contains 271,888 rows, grouping by `(from, to, flightType, agency, time, distance)` reveals **490 unique flight configurations** with near-deterministic pricing across booking dates. A standard random `train_test_split` places the exact same route/cabin/agency combinations in both train and test sets, causing artificial memorization (`R² > 0.999`). TRAVELIQ therefore mandates group-aware splitting (`GroupShuffleSplit` and `GroupKFold`).
2. **Flight Duration Units (`time` column):** Values span `0.44` to `2.44`, confirming duration is recorded in **hours** (not minutes). All UI sliders, API schemas, and validation checks enforce hours explicitly.
3. **One Partner Hotel per Destination City (`hotels.csv`):** Cross-tabulating `place` vs. `name` proves a strict 1-to-1 mapping across all 9 cities (`9 cities ↔ 9 hotels`), with each hotel charging a single constant nightly rate (`price`). Pretending the dataset offers competing hotels inside a single city would be factually dishonest. TRAVELIQ therefore joins `hotels.csv` with round-trip `flights.csv` via `travelCode` (verified **100.0% destination city match** across all 40,552 hotel stays) to build **1,568 real Flight + Hotel travel packages**.
4. **Statistical Proof that `"none"` Gender is Unlabeled / MCAR (`users.csv`):** Exactly `440` of `1,340` user rows (`32.84%`) carry the literal lowercase string `"none"`, alongside `452` `male` and `448` `female`. Hypothesis testing confirms `"none"` is Missing Completely At Random (MCAR):
   - Company independence: $\chi^2\ p = 0.3750$ (no corporate bias).
   - Age distribution: one-way ANOVA $p = 0.6185$ (no age bias).
   - First-name overlap: `66.8%` of `"none"` records share first names with labeled `male`/`female` users, including exact full-name duplicates (`David Robinson`, `Charlotte Johnson`).
   Treating `"none"` as an undisclosed label rather than a genuine 3rd target class fixes the starter repository's misframed classification task.

---

## 2. Data Preprocessing & Feature Engineering

Each ML track encapsulates all transformations inside deterministic `scikit-learn` pipelines or dedicated shared Python modules to guarantee zero training/serving skew and zero data leakage:

### Flight Price Pipeline (`notebooks/01_flight_price_prediction.ipynb` & `scripts/train_flight_model.py`)
- **Date Parsing & Explicit `int32` Alignment:** `date` is parsed to datetime and decomposed into `year`, `month`, `day`, and `day_of_week`. Because pandas 3.x produces `int32` for `.dt` accessors and MLflow Model Signatures strictly validate integer width at inference time, `app/app.py` explicitly constructs `int32` columns to match the registered signature.
- **Route Feature Engineering:** Combined `from_to = from + "_" + to` captures directional corridor pricing effects alongside `from`, `to`, `flightType`, and `agency`.
- **Encapsulated Preprocessing:** `ColumnTransformer` applies `OneHotEncoder(handle_unknown="ignore")` to categorical columns and passes numeric columns (`time`, `distance`, `year`, `month`, `day`, `day_of_week`) directly to tree/linear estimators, fitted strictly on training folds.

### Gender Classification Pipeline (`notebooks/02_gender_classification.ipynb` & `app/gender_features.py`)
- **Label Hygiene:** Drops literal `"none"` records (`440` rows) prior to supervised modeling, retaining `900` binary-labeled records (`452 male`, `448 female`).
- **Lightweight Pure-`scikit-learn` Feature Extractor (`app/gender_features.py`):** Replaces the starter codebase's 91.9 MB `SentenceTransformer` + PyTorch pipeline with deterministic first-name extraction plus `FeatureUnion`/`ColumnTransformer` combining:
  - Character n-gram TF-IDF (`analyzer="char_wb"`, `ngram_range=(2, 5)`) on normalized first names.
  - Word-level TF-IDF on normalized first names.
  - Morphological indicators (name length, vowel ratio, ending vowel, last-character suffixes such as `-a`, `-e`, `-i`, `-y`, `-n`, `-o`, `-r`, `-s`).
  - Corporate metadata (`company` one-hot encoded, `age` scaled).
- **Leakage Control:** All TF-IDF vocabularies and scalers fit exclusively on `X_train` inside the `scikit-learn` `Pipeline`.

### Recommendation Data Pipeline (`notebooks/03_travel_recommendation.ipynb` & `app/recommender.py`)
- Links outbound and return legs in `flights.csv` via `travelCode` to compute true round-trip airfare per itinerary, merges `hotels.csv` nightly rate × stay `days`, and aggregates valid historical combinations into a 1,568-package catalog keyed by `(origin, destination, hotel_name, flightType, agency, days)` with median round-trip flight cost, exact hotel total, and historical booking volume.

---

## 3. Flight Price Regression Workflow

### Validation Strategy
To mirror real deployment on unseen route/airline/cabin pricing schedules, data is partitioned using `GroupShuffleSplit(test_size=0.20, random_state=42)` grouped by `(from, to, flightType, agency, time, distance)`:
- **Training Set:** `221,274` rows across `392` flight configurations.
- **Holdout Test Set:** `50,614` rows across `98` completely unseen flight configurations (`0` group overlap).
- **Hyperparameter Cross-Validation:** 3-fold `GroupKFold` over training configurations.

### Model Comparison Results (50,614 Unseen-Config Test Rows)

| Model | MAE ($) | MSE | RMSE ($) | R² | Outcome |
|---|---:|---:|---:|---:|---|
| **LightGBM (Default)** | **41.73** | **2,754.14** | **52.48** | **0.9765** | **Selected Production Champion** |
| XGBoost | 43.09 | 2,966.09 | 54.46 | 0.9746 | Strong runner-up |
| LightGBM (`GroupKFold` Tuned) | 45.28 | 2,944.11 | 54.26 | 0.9748 | Slightly higher MAE on holdout |
| Random Forest | 54.63 | 4,976.75 | 70.55 | 0.9575 | Higher variance & larger artifact |
| Linear Regression | 82.13 | 10,001.39 | 100.01 | 0.9145 | Underfits non-linear route interactions |

### Artifact & Registry Promotion
The winning LightGBM `Pipeline` is saved locally to `models/flight_price_pipeline.joblib` (`896.3 KB`) and logged to MLflow with full input/output schema signature as `Voyage_Flight_Price_Model` Version 2 with the `@champion` alias. Prediction parity between the live MLflow registry model and the local `.joblib` artifact was verified to `0.000000` absolute difference (`$1,304.8800 == $1,304.8800`).

---

## 4. Gender Classification Workflow

### Audit & Resolution of the Historical `~0.52` vs. `>= 0.85` F1 Discrepancy
During the final capstone audit, the discrepancy between the starter project's `~0.52` macro F1 and TRAVELIQ's `>= 0.85` Airflow quality gate was audited mathematically:
1. **Why Starter F1 was `~0.52`:** The starter notebook kept all 440 `"none"` records as a third target class. Because `"none"` is statistically random missingness (MCAR), no legitimate feature can predict it, depressing `"none"` recall to `0.17–0.33` and dragging overall F1 down to `~0.52`.
2. **Leakage Audit on TRAVELIQ's Binary Split:** Verified `0` overlapping user `code` values and `0` overlapping full `name` strings between the `720`-row training set and `180`-row stratified test set, with all vectorizers strictly inside the `Pipeline`.
3. **Why Stratified Holdout F1 is `0.8830` (and What Happens on Unseen First Names):**
   - `110 / 180` (`61.1%`) of distinct test users share a common first name (e.g., *Michael*, *Patricia*) that also appeared in training -> Accuracy `0.9727`, Weighted F1 `0.9726`.
   - `70 / 180` (`38.9%`) of test users carry brand-new first names never seen during training -> Accuracy `0.7429`, Weighted F1 `0.7454`.
   - Under an even stricter `GroupShuffleSplit` holding out entire first-name groups (`n=168`, `0%` shared first names), pure morphological + character n-gram generalization yields Accuracy `0.6369`, Weighted F1 `0.6366`.

### 5-Model Benchmark (900 Labeled Records · 5-Fold Stratified CV + 180 Test Holdout)

| Model | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test Weighted F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Tuned Logistic Regression (`C=8.0`, Saved)** | **0.9083** | **0.8833** | **0.8872** | **0.8833** | **0.8830** | **0.9533** |
| Random Forest | 0.8989 | 0.8889 | 0.8920 | 0.8889 | 0.8887 | 0.9666 |
| Calibrated LinearSVC | 0.9040 | 0.8833 | 0.8872 | 0.8833 | 0.8830 | 0.9552 |
| XGBoost | 0.8636 | 0.8722 | 0.8745 | 0.8722 | 0.8720 | 0.9232 |
| LightGBM | 0.8672 | 0.8611 | 0.8648 | 0.8611 | 0.8608 | 0.9426 |

**Confusion Matrix (Saved Logistic Regression, `n=180`):**
- Actual `female` (`90`): `75` correctly classified as `female`, `15` misclassified as `male`.
- Actual `male` (`90`): `84` correctly classified as `male`, `6` misclassified as `female`.

**Engineering & Responsible AI Highlights:**
- **2,275× Artifact Reduction:** Shrank from `91.9 MB` (requiring PyTorch + HuggingFace downloads) to **`40.4 KB`** (`models/gender_classifier.joblib`), enabling sub-millisecond inference and a minimal container footprint.
- **Dual MLOps Promotion Gates:** `scripts/mlops_pipeline.py` requires both `stratified_f1 >= 0.85` AND `unseen_first_name_f1 >= 0.58`.
- **Ethical Guardrails:** Both the Gender REST API (`experimental_notice` field) and the Streamlit UI display prominent banners clarifying that outputs are non-verified statistical estimates and must never be used for consequential decisions.

---

## 5. Travel & Hotel Recommendation Workflow

### Architecture (`notebooks/03_travel_recommendation.ipynb` & `app/recommender.py`)
Given the verified 1-hotel-per-city constraint, TRAVELIQ implements a two-layer **Hybrid Flight + Hotel Package Recommender** over **1,568 real packages**:

1. **Collaborative Filtering Affinity Layer:** Constructs a user-by-destination interaction matrix from historical trips and computes User-Based cosine similarity across peers to predict destination affinity scores for returning travelers.
2. **Transparent Multi-Factor Scoring (`0–100` Scale):** Each candidate package matching optional origin/destination/cabin/budget filters receives an explainable weighted score combining:
   - **Budget Fit:** Rewards packages comfortably below the traveler's total budget ceiling.
   - **Duration Alignment:** Highest score for exact stay-day match, with graceful decay when flexible matching is enabled.
   - **Personalization / CF Affinity:** Uses User-Based CF affinity when a traveler profile is selected, falling back cleanly to cold-start popularity for anonymous sessions.
   - **Historical Route Popularity:** Incorporates empirical trip volume across the 40,552 paired stays.
   - **Destination Diversity Mode:** Optionally selects the top package per unique destination city when exploring broadly.

### Offline & Query Validation Results
- **Collaborative Filtering (`1,228` users, time-ordered 80/20 split, `K=3`):**
  - User-Based CF: `HitRate@3 = 0.8974`, `Precision@3 = 0.5532`, `Recall@3 = 0.3672`, `NDCG@3 = 0.5680`, **Catalog Coverage = `100.0%` (`9/9` cities)**.
  - Popularity Baseline: `HitRate@3 = 0.8933`, `Precision@3 = 0.5611`, `Recall@3 = 0.3712`, `NDCG@3 = 0.5754`, **Catalog Coverage = `33.3%` (`3/9` cities)**.
  - *Insight:* User-Based CF matches the Popularity baseline's hit rate while tripling destination catalog coverage from 33.3% to 100.0%.
- **200 Randomized Query Stress Test:** `100.0%` budget compliance, `89.7%` exact duration match, `4.78 / 5.00` average unique destinations per Top-5 explorer query, `95.5%` personalization differentiation vs. anonymous queries, and `6/6` edge cases handled gracefully (ultra-low budget, missing user ID, exact duration conflict, single destination, cold-start user, high top-N).
- **Notebook-to-Production Parity:** `app/recommender.py` reproduces Notebook 03's three canonical verification queries to exact hundredth precision (`71.28`, `76.99`, `54.91`).

---

## 6. Flask REST Microservices Workflow

TRAVELIQ decouples ML inference from the presentation tier via two focused Flask services served with **Waitress** locally (`python -m ...`) and **Gunicorn** in containers, configured centrally via `app/config.py`:

### Flight Price Microservice (`app/app.py`, Port `5001`)
- **Model Lifecycle:** On first inference or health check, queries `MLFLOW_TRACKING_URI` (preferring live HTTP `:5000`, then `sqlite:///mlflow.db`) for `models:/Voyage_Flight_Price_Model@champion` and caches the loaded pyfunc/sklearn object in memory. If MLflow is unreachable and `ENABLE_LOCAL_MODEL_FALLBACK=1`, transparently loads `models/flight_price_pipeline.joblib` and reports `model_source: "local_joblib_fallback"`.
- **Schema & Input Validation:** Rejects missing fields, non-JSON payloads, invalid dates, negative/zero durations or distances, and identical origin/destination cities with structured `400 Bad Request` JSON responses. Casts extracted date features to `int32` to honor MLflow signature validation.
- **Verified Live Output (MLflow HTTP Mode):**
  - `GET /health` -> `200 OK`, `{"alias":"champion","model":"Voyage_Flight_Price_Model","model_source":"mlflow_registry","service":"traveliq-flight-api","status":"healthy"}`.
  - `POST /predict` (`Recife (PE)` -> `Florianopolis (SC)`, `firstClass`, `FlyingDrops`, `2020-07-15`, `1.76h`, `676.53km`) -> `200 OK`, `predicted_price: 1429.45`, `model_source: "mlflow_registry"`.

### Gender Classification Microservice (`app/gender_api.py`, Port `5002`)
- **Model Lifecycle:** Loads `models/gender_classifier.joblib` (`40.4 KB`) once at startup into a cached singleton. Zero PyTorch, zero HuggingFace network calls, near-instant startup.
- **Validation & Responsible AI Output:** Enforces non-empty `name`, valid `company`, and numeric `age` (`18–100`), returning `predicted_gender`, calibrated `confidence`, per-class `probabilities`, `experimental: true`, and `experimental_notice`.
- **Verified Live Output:**
  - `GET /health` -> `200 OK`, `artifact_size_kb: 40.4`, `artifact_version: "2.0-lightweight-binary"`.
  - `POST /predict-gender` (`Charlotte Johnson`, `Acme Factory`, `31`) -> `200 OK`, `predicted_gender: "female"`, `confidence: 0.9729`.

---

## 7. TRAVELIQ Streamlit Web Application Workflow

`streamlit_app.py` implements an executive Navy (`#0B1F3A`), White (`#FFFFFF`), and Modern Blue (`#1D63ED`) Travel Intelligence interface styled via `.streamlit/config.toml` and custom CSS, communicating with backend microservices exclusively through `app/api_client.py` (configurable URLs, timeouts, and graceful offline error handling).

### Live-Verified Application Views

1. **Home Dashboard (`screenshots/traveliq_01_home_dashboard.png`):** Displays five KPI cards (`271,888` flights, `40,552` hotel stays, `1,340` traveler profiles, `9` destinations, `1,568` packages), Quick Launch action cards, sidebar live API health badges (`ONLINE` with `Source: mlflow_registry` and `Model: 2.0-lightweight-binary`), and two overview Plotly charts (Average Fare by Cabin & Agency; Destination Demand & Partner Hotel Daily Rate).
2. **Flight Price Predictor (`screenshots/traveliq_02_flight_predictor.png`):** Interactive route, cabin, agency, duration (hours), distance (km), and travel date controls calling `POST :5001/predict`. Displays the headline fare card (e.g., `$1,480.04 USD` for `Florianopolis (SC) -> Aracaju (SE)`, `firstClass`, `CloudFy`), historical route mean comparison (`$1,468.03`, delta `+$12.01`), live serving badge (`MLflow @champion`), and expandable request summary.
3. **Trip & Hotel Package Recommendations (`screenshots/traveliq_03_trip_recommendations.png`):** Highlights the one-hotel-per-city dataset limitation prominently at the top, provides origin, destination, cabin, duration, budget, top-N, diversity, and traveler personalization controls, renders KPI summary cards, a stacked horizontal bar chart comparing Round-Trip Flight vs. Hotel Stay cost against the dashed budget ceiling, and ranked package cards with explicit `"Why recommended"` score breakdowns.
4. **Gender Classification Demo (`screenshots/traveliq_04_gender_demo.png`):** Features a prominent amber Responsible AI notice above the input form, calls `POST :5002/predict-gender`, and renders the experimental estimate card alongside a calibrated class probability chart (`Female 98.3% / Male 1.6%` for sample `Wilma Mcinnis`).
5. **Travel & Pricing Analytics (`screenshots/traveliq_05_analytics.png`):** Five interactive Plotly analytics tabs covering Flight Price Distribution (cabin histograms + agency/cabin boxplots), Popular Routes, Destination Popularity, Travel Duration, and Hotel & Package Pricing.

---

## 8. MLflow Experiment Tracking & Model Registry Workflow

MLflow (`3.18.0`) serves as TRAVELIQ's unified experiment ledger and model registry using a SQLite backend store (`sqlite:///mlflow.db`) and local artifact store (`mlruns/`):

- **Logged Experiments:** `Voyage Analytics - Flight Price Prediction` contains 6 completed runs covering all 4 benchmark algorithms (`Benchmark_Linear_Regression`, `Benchmark_Random_Forest`, `Benchmark_XGBoost`, `Benchmark_LightGBM`) plus production training runs (`Production_LightGBM_Champion`, `Flight_Price_LightGBM_Production`) recording `mae`, `mse`, `rmse`, `r2`, parameters, input schemas, and serialized pipelines.
- **Registered Model & Alias Governance:** `Voyage_Flight_Price_Model` tracks Version 1 and Version 2, with the **`@champion`** alias assigned to Version 2 (`status: READY`). Decoupling consumers via `@champion` lets automated Airflow retraining promote superior future models without changing a single line of Flask API code.
- **Live Verification Evidence:** Captured directly from the running MLflow server (`http://127.0.0.1:5000`) in `screenshots/traveliq_06_mlflow_experiments.png` and `screenshots/traveliq_07_mlflow_registry.png`, and confirmed via `GET http://127.0.0.1:5001/health` returning `"model_source": "mlflow_registry"`.

---

## 9. Docker & Docker Compose Workflow

> **Verification Status: `PARTIAL` — Configurations & Hygiene Verified via Pytest; Live Container Execution Pending (Docker Desktop / WSL2 Not Installed on Non-Admin Windows Host).**

TRAVELIQ containerizes each microservice independently to minimize blast radius, isolate dependencies, and eliminate image bloat:

| Service | Dockerfile | Image Tag | Port | Key Engineering Decisions |
|---|---|---|---:|---|
| `mlflow` | `docker/Dockerfile.mlflow` | `traveliq-mlflow:2.0` | `5000` | Persistent volume `traveliq_mlflow_data` at `/mlflow`, `--allowed-hosts` configured for service-name DNS, non-root `traveliq` user, HTTP `/health` check. |
| `flight-api` | `docker/Dockerfile` | `traveliq-flight-api:2.0` | `5001` | Installs `libgomp1` required by LightGBM on slim Debian Linux; served via multi-worker Gunicorn; `MLFLOW_TRACKING_URI=http://mlflow:5000`; automatic local `.joblib` fallback mounted read-only. |
| `gender-api` | `gender/Dockerfile` | `traveliq-gender-api:2.0` | `5002` | Pure `scikit-learn` (`gender/requirements-gender.txt`) with **zero PyTorch and zero `sentence-transformers`**, shrinking image size by >2 GB compared to the starter container. |
| `streamlit-ui` | `docker/Dockerfile.streamlit` | `traveliq-streamlit:2.0` | `8501` | Connects to `http://flight-api:5001` and `http://gender-api:5002` over private bridge network `traveliq-net`; `depends_on` with `condition: service_healthy`; `/_stcore/health` check. |

All 4 Dockerfiles enforce non-root runtime users, per-service `.dockerignore` hygiene, pinned dependencies, and zero embedded credentials.

---

## 10. Kubernetes & Horizontal Pod Autoscaling (HPA) Workflow

> **Verification Status: `PARTIAL` — Manifests, Cross-Resource Wiring & Probes Verified via Pytest; Live Cluster Rollout & HPA Load Observation Pending (`kubectl` / Local Cluster Not Installed).**

All Kubernetes manifests in `kubernetes/` reference exact matching image tags (`traveliq-*:2.0`) with `imagePullPolicy: IfNotPresent` for local clusters:

- **`configmap.yaml` (`traveliq-config`):** Centralizes internal cluster DNS endpoints (`http://mlflow-service:5000`, `http://flight-api-service:5001`, `http://gender-api-service:5002`), `MODEL_NAME`, `MODEL_ALIAS`, and `ENABLE_LOCAL_MODEL_FALLBACK`.
- **`secret.example.yaml`:** Template for optional external registry/database credentials (real secrets excluded via `.gitignore`).
- **`mlflow.yaml`:** Provisions a `PersistentVolumeClaim` (`mlflow-pvc`, `2Gi`), Deployment, and ClusterIP `mlflow-service:5000`.
- **`deployment.yaml` & `service.yaml` (Flight API):** Runs `2` initial replicas with CPU/memory requests & limits plus distinct `startupProbe`, `readinessProbe`, and `livenessProbe` targeting `GET /health` on `:5001`, exposed via `NodePort` `30501`.
- **`gender-classifier.yaml` & `streamlit.yaml`:** Deploy Gender API (`:5002`, NodePort `30502`) and TRAVELIQ Streamlit UI (`:8501`, NodePort `30851`) with full probe coverage and resource ceilings.
- **`hpa.yaml` (`autoscaling/v2`):** Configures dynamic scaling for `flight-api-deployment` (`minReplicas: 2`, `maxReplicas: 5`, target `70%` CPU / `80%` memory) and `gender-api-deployment` (`minReplicas: 1`, `maxReplicas: 4`, target `70%` CPU) with stabilization windows to prevent thrashing.

---

## 11. Apache Airflow Retraining Workflow

> **Verification Status: `PARTIAL` — All 7 Python Pipeline Stages, Skip-Retrain Gate, Dual Quality Gates & Failure Blocking Verified Live in Isolated Sandbox (`PASS`); Dockerized Airflow Webserver/Scheduler UI Pending (Requires Docker Desktop).**

TRAVELIQ's continuous retraining pipeline separates core ML logic (`scripts/mlops_pipeline.py`) from DAG scheduling (`airflow/dags/traveliq_ml_pipeline.py`) so every stage is independently unit- and integration-testable:

```text
validate_datasets
       │
       ▼
preprocess_and_retrain_gate  (ShortCircuitOperator: skips cleanly if dataset SHA-256 unchanged)
       │
       ├─────────────────────────┬──────────────────────────┐
       ▼                         ▼                          ▼
train_flight_model      train_gender_model      build_recommendation_catalog
       │                         │                          │
       └─────────────────────────┴──────────────────────────┘
                                 ▼
                         evaluate_models
     (Enforces: Flight R² >= 0.90 & RMSE <= 80; Gender Stratified F1 >= 0.85
      AND Unseen-First-Name F1 >= 0.58; Recommender >= 1,000 packages & 100% budget fit)
                                 │
                                 ▼
                  register_and_promote_champion
        (Registers candidate in MLflow & promotes @champion ONLY if all gates pass)
```

**Operational Reliability Features:**
- `retries=2` with `retry_delay=timedelta(minutes=2)` and an explicit `on_failure_callback` logger.
- `airflow/docker-compose.yml` uses healthchecked PostgreSQL 15 (`service_healthy`), initializes Airflow 2.10.5 (`service_completed_successfully`), and contains zero legacy personal names or hardcoded secrets.
- Verified in `tests/test_integration.py` that a passing pipeline promotes `@champion` in an isolated sandbox MLflow registry AND that any failing quality gate raises a blocking `RuntimeError`, preventing bad models from reaching production.

---

## 12. Jenkins CI/CD Workflow

> **Verification Status: `PARTIAL` — 7-Stage Pipeline Syntax, Safety Guards & Cross-Platform Commands Verified via Pytest (`PASS`); Live Jenkins Server Run Pending (Jenkins Not Installed).**

The root `Jenkinsfile` implements a 7-stage declarative CI/CD workflow with Windows (`bat`) and Linux (`sh`) parity:

1. **Checkout:** Pulls the commit cleanly and records build metadata.
2. **Python Dependencies:** Creates/updates an isolated virtual environment from `requirements.txt` and `requirements-dev.txt`.
3. **Code Quality:** Compiles all Python sources (`py_compile`), parses all YAML manifests (`kubernetes/`, `docker-compose.yml`, `airflow/`), and verifies Airflow DAG syntax.
4. **Automated Tests & JUnit Reporting:** Executes `pytest -v --junitxml=reports/junit.xml` and archives JUnit test results in `post { always }` so test evidence is retained even on failures.
5. **Notebook & Artifact Validation:** Verifies all 3 capstone notebooks exist with valid JSON structure, zero execution tracebacks, and verified model artifacts (`flight_price_pipeline.joblib`, `gender_classifier.joblib`).
6. **Docker Image Build:** Builds all four versioned `traveliq-*:2.0` images when Docker is available on the build agent.
7. **Non-Production Deployment Validation:** Runs `kubectl apply --dry-run=client` manifest validation only (explicitly prohibiting automatic production deployment). Any failure in Stages 1–6 immediately halts the pipeline before deployment validation.

---

## 13. Automated Testing, Live Verification & Honest Evidence Matrix

### Automated Pytest Results (`57 Passed, 0 Failed`)

Executed cleanly via `.\.venv\Scripts\pytest.exe -v`:

| Test Module | Tests | Scope Verified | Result |
|---|---:|---|---|
| `tests/test_flight_api.py` | `12` | Flight `/health`, valid predictions, invalid payload `400` rejections, identical origin/dest rejection, `int32` schema alignment, MLflow failure -> local `.joblib` fallback, model caching | **12 PASS** |
| `tests/test_gender_api.py` | `12` | Gender `/health`, valid predictions, calibrated probability sum `= 1.0`, `experimental_notice` presence, invalid age/empty name rejections, single-load caching, lightweight `<1 MB` artifact check | **12 PASS** |
| `tests/test_recommender.py` | `13` | 1,568 package catalog integrity, 1-hotel-per-city accuracy, Notebook 03 exact score parity (`71.28`, `76.99`, `54.91`), budget ceiling compliance, personalized vs. anonymous differentiation, edge cases | **13 PASS** |
| `tests/test_api_client.py` | `8` | Streamlit HTTP client success parsing, connection-refused handling, timeouts, HTTP 400/500 propagation, configurable base URLs | **8 PASS** |
| `tests/test_mlops_infrastructure.py` | `7` | Dockerfiles (`libgomp1`, non-root, no PyTorch in gender, healthchecks), `docker-compose.yml` service wiring, Kubernetes YAML & image parity, HPA `autoscaling/v2`, Airflow DAG & skip/fail gates, `Jenkinsfile` stages | **7 PASS** |
| `tests/test_integration.py` | `5` | Live local Waitress Flight & Gender HTTP roundtrips, MLflow `@champion` vs. local `.joblib` zero-diff parity, end-to-end sandboxed 7-stage Airflow/MLflow training + dual-gate promotion, blocked promotion on failed gate, gender seen/unseen split integrity | **5 PASS** |
| **Total** | **`57`** | **Full Application, ML Parity, Infrastructure & Pipeline Verification** | **`57 / 57 PASS`** |

### Complete Evidence & Screenshot Matrix

| Evidence Item | Artifact / Location | Status |
|---|---|---|
| TRAVELIQ Home Dashboard | `screenshots/traveliq_01_home_dashboard.png` | **CAPTURED (Live)** |
| Flight Price Predictor (`MLflow @champion`) | `screenshots/traveliq_02_flight_predictor.png` | **CAPTURED (Live)** |
| Trip & Hotel Recommendations | `screenshots/traveliq_03_trip_recommendations.png` | **CAPTURED (Live)** |
| Gender Classification Demo | `screenshots/traveliq_04_gender_demo.png` | **CAPTURED (Live)** |
| Travel & Pricing Analytics | `screenshots/traveliq_05_analytics.png` | **CAPTURED (Live)** |
| MLflow Experiments Runs Table | `screenshots/traveliq_06_mlflow_experiments.png` | **CAPTURED (Live)** |
| MLflow Model Registry (`@champion` v2) | `screenshots/traveliq_07_mlflow_registry.png` | **CAPTURED (Live)** |
| Automated Pytest Suite (`57/57`) | Verbatim output recorded in Section 13 & `docs/SUBMISSION_CHECKLIST.md` | **VERIFIED (Live)** |
| Docker Containers (`docker compose ps`) | Pending host Docker Desktop + WSL2 installation | **PENDING (Not Fabricated)** |
| Kubernetes Pods & HPA (`kubectl get pods,hpa`) | Pending local cluster (`minikube` / Docker Desktop K8s) | **PENDING (Not Fabricated)** |
| Airflow Webserver UI (`:8080`) | Pending Docker Desktop (7-stage Python pipeline verified live in sandbox) | **PENDING (Not Fabricated)** |
| Jenkins Pipeline UI (`:8080`) | Pending Jenkins installation (`Jenkinsfile` statically verified) | **PENDING (Not Fabricated)** |
