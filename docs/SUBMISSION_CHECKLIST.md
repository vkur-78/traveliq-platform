# TRAVELIQ — Capstone Submission Checklist

Use this checklist to verify every deliverable prior to final capstone submission. Every **Completed** item below has been generated and verified directly in this workspace; every **User Action Required** item lists the exact smallest step you need to take before submitting.

---

## 1. Three Executed Jupyter Notebooks

| Deliverable | File Path | Status | Verified Details |
|---|---|---|---|
| **Notebook 1 — Flight Price Prediction** | `notebooks/01_flight_price_prediction.ipynb` | **READY** | 35 cells, 8 charts, sequential execution (`1..21`), `0` errors, Google Colab compatible, leakage-free `GroupShuffleSplit` (`R² = 0.9765`, `RMSE = $52.48`, `MAE = $41.73`, `MSE = 2,754.14`), MLflow integrated. |
| **Notebook 2 — Gender Classification** | `notebooks/02_gender_classification.ipynb` | **READY** | 27 cells, 5 charts, sequential execution (`1..14`), `0` errors, MCAR `"none"` analysis, 5-model binary comparison (`CV F1 = 0.9083`, `Test F1 = 0.8830`, `AUC = 0.9533`), confusion matrix, strict unseen-first-name group check (`0.6366`), `40.4 KB` artifact, Responsible AI section. |
| **Notebook 3 — Hotel & Travel Recommendation** | `notebooks/03_travel_recommendation.ipynb` | **READY** | 25 cells, 4 charts, sequential execution (`1..14`), `0` errors, honest 1-hotel-per-city analysis, 1,568 round-trip Flight + Hotel packages, User-Based CF (`HitRate@3 = 0.8974`, `100%` coverage) vs. Popularity (`33.3%`), 200-query validation & 6 edge cases. |

---

## 2. GitHub Repository Preparation

| Item | Status | Action / Verification |
|---|---|---|
| Clean `.gitignore` excluding `.venv/`, `backups/`, `mlflow.db`, `mlruns/`, `mlartifacts/`, `reports/`, `.pytest_cache/`, `.env`, `models/candidates/` | **READY** | Verified in `.gitignore`. |
| Production artifacts preserved (`models/flight_price_pipeline.joblib` 896.3 KB, `models/gender_classifier.joblib` 40.4 KB) | **READY** | Verified in `models/`. |
| Starter-project attribution & open-source licenses preserved (`ATTRIBUTION.md`, `README.md`) | **READY** | Zero legacy author personal names/paths in TRAVELIQ files; honest credit preserved. |
| Local Git repository initialized cleanly | **READY** | Verified locally (`git status`). |
| **Create your own GitHub repo & push** | **USER ACTION REQUIRED** | Run `git add .`, `git commit -m "Initial TRAVELIQ capstone release"`, `git remote add origin <your-repo-url>`, and `git push -u origin main` after reviewing `git status`. |

---

## 3. Technical Workflow Google Doc

| Deliverable | File Path | Status | Conversion Steps |
|---|---|---|---|
| **Formatted Word Document (with embedded live screenshots)** | `docs/TRAVELIQ_Technical_Workflow_Documentation.docx` | **READY** | Covers all 13 technical workflow chapters with real metrics, API JSON outputs, pytest table, and 7 embedded high-res PNGs. |
| **Markdown Mirror** | `docs/TRAVELIQ_Technical_Workflow_Documentation.md` | **READY** | Version-controlled plain-text mirror in Git. |
| **Upload to Google Docs** | — | **USER ACTION REQUIRED** | 1) Upload `docs/TRAVELIQ_Technical_Workflow_Documentation.docx` to Google Drive.<br>2) Right-click → **Open with → Google Docs** → **File → Save as Google Docs**.<br>3) Verify all 7 embedded figures & tables render cleanly, then set viewer sharing link for submission. |

---

## 4. Presentation Deck & 40-Minute Video Recording

| Deliverable | File Path | Status | Notes |
|---|---|---|---|
| **PowerPoint Presentation (22 Widescreen Slides)** | `docs/TRAVELIQ_Capstone_Presentation.pptx` | **READY** | Navy/White/Blue TRAVELIQ branding, 7 embedded live screenshots, metric tables, speaker notes. Replace `[Your Name]` on Slide 1. |
| **Timed 40-Minute Presentation Script** | `docs/TRAVELIQ_40Min_Presentation_Script.md` | **READY** | Natural student style mapped slide-by-slide (`00:00–40:00`) with `[YOUR DECISION / REFLECTION PROMPT: ...]` markers to fill in with your genuine engineering rationale. |
| **Record 40-Minute Video Walkthrough** | — | **USER ACTION REQUIRED** | 1) Personalize Slide 1 (`[Your Name]`) and review the 5 reflection prompts in the script.<br>2) Launch MLflow (`:5000`), Flight API (`:5001`), Gender API (`:5002`), and Streamlit (`:8501`) locally before recording.<br>3) Record slides + 8-minute live demo + `pytest -v` (`57/57`) run. |

---

## 5. Screenshots & Automated Test Evidence

### Captured Live Evidence (`READY`)
- [x] `screenshots/traveliq_01_home_dashboard.png` — Live TRAVELIQ Home Dashboard (`ONLINE` API badges, KPIs, overview Plotly charts)
- [x] `screenshots/traveliq_02_flight_predictor.png` — Live Flight Price Predictor (`$1,480.04 USD`, `MLflow @champion`, route mean delta)
- [x] `screenshots/traveliq_03_trip_recommendations.png` — Live Trip & Hotel Recommendations (1-hotel-per-city notice, stacked cost chart, rationale card)
- [x] `screenshots/traveliq_04_gender_demo.png` — Live Gender Classification Demo (amber Responsible AI banner, calibrated probability bar)
- [x] `screenshots/traveliq_05_analytics.png` — Live Travel & Pricing Analytics (Plotly fare histogram & agency/cabin boxplot)
- [x] `screenshots/traveliq_06_mlflow_experiments.png` — Live MLflow 3.18 Experiments runs table (6 logged runs)
- [x] `screenshots/traveliq_07_mlflow_registry.png` — Live MLflow 3.18 Model Registry (`Voyage_Flight_Price_Model` Version 2 `@champion`)
- [x] Automated `pytest -v` — **`57 passed, 0 failed`** across `test_flight_api.py` (12), `test_gender_api.py` (12), `test_recommender.py` (13), `test_api_client.py` (8), `test_mlops_infrastructure.py` (7), and `test_integration.py` (5)

### Pending Runtime Evidence (Requires Docker Desktop + WSL2 with Admin Elevation)
- [ ] **Docker Compose live containers (`docker compose ps`)** — Pending Docker Desktop installation (`winget install Docker.DockerDesktop` as Administrator + `wsl --install`).
- [ ] **Kubernetes pods & HPA live output (`kubectl get pods,svc,hpa`)** — Pending enabling Kubernetes in Docker Desktop (or Minikube + `metrics-server`).
- [ ] **Apache Airflow Webserver UI (`http://localhost:8080`)** — Pending Docker Desktop (`docker compose -f airflow/docker-compose.yml up --build -d`); core 7-stage Python pipeline + dual promotion gates already verified live in sandbox (`PASS`).
- [ ] **Jenkins Pipeline UI (`http://localhost:8080`)** — Pending local Jenkins controller; 7-stage cross-platform `Jenkinsfile` statically verified (`PASS`).
