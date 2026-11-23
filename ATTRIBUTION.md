# Attribution & Open-Source Acknowledgements

## Starter Codebase Attribution

**TRAVELIQ — Travel Intelligence Platform** was independently engineered and redesigned as a Travel & Tourism Machine Learning & MLOps Capstone Project building upon the open-source starter repository structure `voyage-analytics-mlops`.

### What Originated from the Starter Structure
- The initial problem framing (flight price regression, user gender classification, hotel/travel recommendation) and the `travel_capstone_dataset/` CSV files (`flights.csv`, `hotels.csv`, `users.csv`).
- The initial baseline directory layout (`notebooks/`, `app/`, `docker/`, `gender/`, `kubernetes/`, `airflow/`, `Jenkinsfile`).

### Independent Engineering & Redesign Performed in TRAVELIQ
- **Phase 1 — Machine Learning Engineering:** Complete notebook audit and redesign (`01_flight_price_prediction.ipynb`, `02_gender_classification.ipynb`, `03_travel_recommendation.ipynb`), leakage-free `GroupShuffleSplit`/`GroupKFold` flight regression validation, statistical MCAR investigation and binary refactoring of `"none"` gender records, replacement of the 91.9 MB PyTorch/`SentenceTransformer` gender model with a 40.4 KB pure `scikit-learn` pipeline, honest seen vs. unseen first-name generalization analysis, and a transparent 1,568-package round-trip Flight + Hotel hybrid recommender addressing the one-hotel-per-city dataset limitation.
- **Phase 2 — Full-Stack TRAVELIQ Application:** Original TRAVELIQ visual identity (Navy/White/Blue theme, vector logo, custom CSS, 5-section dashboard), decoupled REST API client, Waitress WSGI serving, MLflow `@champion` loading with `int32` schema alignment and automatic local fallback, Responsible AI disclaimers, and an initial 45-test `pytest` suite.
- **Phase 3 & Final Integration — MLOps & Verification:** Multi-service Dockerfiles with non-root users, health checks, and zero PyTorch bloat in the gender container; root `docker-compose.yml`; aligned Kubernetes Deployments, Services, ConfigMap, PVC, Secret template, and `autoscaling/v2` HPAs; a 7-stage quality-gated Apache Airflow retraining pipeline (`scripts/mlops_pipeline.py` + `traveliq_ml_pipeline.py`) with SHA-256 skip-retrain and dual F1 promotion gates; cross-platform 7-stage `Jenkinsfile`; expanded 57-test automated unit/infrastructure/integration suite; and original technical workflow & presentation documentation.

All legacy personal names, local filesystem paths, and third-party branding were removed from TRAVELIQ's own source code and documentation while preserving transparent credit to the starter codebase above.

---

## Third-Party Open-Source Libraries & Platforms

TRAVELIQ relies on the following open-source projects, each governed by its respective license:

| Component | License | Usage in TRAVELIQ |
|---|---|---|
| Python & Standard Library | PSF License | Core runtime (`3.11` in containers, `3.13` locally) |
| pandas & NumPy | BSD-3-Clause | Data loading, cleaning, feature engineering, aggregations |
| scikit-learn | BSD-3-Clause | Preprocessing pipelines, TF-IDF, Logistic Regression, RF, LinearSVC, CV/metrics |
| LightGBM | MIT License | Production champion flight fare regression model |
| XGBoost | Apache-2.0 | Benchmark flight regression & gender classification models |
| Flask & Waitress / Gunicorn | BSD-3-Clause / ZPL / MIT | REST API microservices and production WSGI serving |
| Streamlit | Apache-2.0 | TRAVELIQ interactive web dashboard |
| Plotly & Matplotlib / Seaborn | MIT / PSF / BSD | Interactive application charts & notebook EDA visualizations |
| MLflow | Apache-2.0 | Experiment tracking, model signatures, and `@champion` Model Registry |
| Apache Airflow | Apache-2.0 | 7-stage scheduled/gated ML retraining workflow orchestration |
| Docker & Kubernetes | Apache-2.0 | Container packaging, orchestration, and Horizontal Pod Autoscaling |
| Jenkins | MIT License | Automated CI/CD pipeline orchestration |
| pytest | MIT License | 57-test automated unit, infrastructure, and integration suite |
| python-docx & python-pptx | MIT License | Programmatic generation of Capstone Technical DOCX & PowerPoint deliverables |
