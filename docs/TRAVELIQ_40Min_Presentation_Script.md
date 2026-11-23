# TRAVELIQ — 40-Minute Capstone Presentation Script

> **How to Use This Script:**
> - Mapped slide-by-slide to `docs/TRAVELIQ_Capstone_Presentation.pptx` (22 slides) with realistic pacing for a **40-minute** capstone recording or live defense.
> - Written in a natural, conversational technical student voice — easy to speak aloud without sounding robotic.
> - **Honesty Note:** Places marked **`[YOUR DECISION / REFLECTION PROMPT: ...]`** are intentionally left for you to speak in your own words about your actual development choices, trade-offs, and learning takeaways. Never read placeholder text verbatim — substitute your real reflection!

---

## Pacing Overview (40 Minutes Total)

| Segment | Time | Slides | Focus |
|---|---:|---|---|
| 1. Intro, Problem & Architecture | `00:00 – 05:30` | Slides 1–4 | Identity, goals, 4-tier architecture |
| 2. Dataset Audit & Realities | `05:30 – 10:00` | Slide 5 | Flights configs, 1-hotel/city, MCAR `"none"` |
| 3. Flight Price Regression | `10:00 – 15:30` | Slides 6–7 | Leakage-free `GroupShuffleSplit`, 5-model table |
| 4. Gender Classification & Ethics | `15:30 – 21:00` | Slides 8–9 | 0.52 vs. 0.8830 audit, seen/unseen, 40.4 KB, ethics |
| 5. Hybrid Package Recommender | `21:00 – 26:00` | Slide 10 | 1,568 packages, CF vs. popularity, 200-query tests |
| 6. Backend APIs & Live Demo | `26:00 – 34:00` | Slides 11–16, 20 | Flask APIs, live Streamlit 5-page + MLflow walkthrough |
| 7. MLOps Stack & 57 Pytest Suite | `34:00 – 38:00` | Slides 17–19 | Docker, K8s/HPA, Airflow, Jenkins, real bugs caught |
| 8. Limitations, Roadmap & Close | `38:00 – 40:00` | Slides 21–22 | Honest pending items, roadmap, attribution, wrap-up |

---

## Segment 1 — Introduction, Problem & Architecture (`00:00 – 05:30`)

### Slide 1 — Title Slide (`00:00 – 01:00`)
"Hello everyone, and thank you for joining my Travel & Tourism Machine Learning and MLOps Capstone presentation. Today I am walking you through **TRAVELIQ — Travel Intelligence Platform**.

TRAVELIQ brings together three machine learning tracks — flight fare prediction, binary gender classification as an experimental demo, and hybrid round-trip flight-and-hotel recommendations — inside a single cohesive web platform backed by Flask REST microservices, MLflow model governance, and a full MLOps engineering stack.

Before diving in, I want to be completely transparent about provenance: this project started from an open-source travel MLOps starter structure (`voyage-analytics-mlops`), which is properly credited in `ATTRIBUTION.md`. Everything I am presenting today reflects an end-to-end technical audit, independent re-engineering of all three machine learning notebooks, a complete UI and API redesign under the TRAVELIQ identity, and a comprehensive MLOps and automated testing architecture."

### Slide 2 — Roadmap & Development Phases (`01:00 – 02:30`)
"Here is how we'll spend the next 40 minutes: first, the problem framing and what we actually discovered when we audited the datasets; then a deep dive into all three ML tracks, including some really interesting findings around data leakage and classification F1 discrepancies; after that, I will walk through the Flask APIs, the live TRAVELIQ Streamlit application, and MLflow Model Registry; and finally, I will cover our MLOps layer — Docker, Kubernetes with HPA, Airflow, Jenkins, and our 57 automated tests — along with an honest assessment of what ran live on my Windows machine versus what is ready and pending a Docker-enabled host.

**[YOUR DECISION / REFLECTION PROMPT: Share in 2–3 sentences why you decided to audit the starter codebase thoroughly first before redesigning it, and which phase was the most interesting or challenging for you to work through.]**"

### Slide 3 — Problem Statement & Objectives (`02:30 – 04:00`)
"When analyzing corporate travel booking systems, three real engineering problems show up immediately:

First, airfare forecasting fails silently when you use a standard random train/test split on repeated route schedules — your model looks like it has 99.9% accuracy only because it memorized exact route prices across both splits.

Second, travel datasets often have thin hotel inventories. In our dataset, there is literally only one partner hotel per destination city. A naive hotel-only recommender would pretend to compare competing hotels where none exist.

Third, notebooks don't automatically make reliable production services. Without schema enforcement, lightweight artifacts, fallback resilience, and quality-gated retraining, even accurate models break in deployment.

Our KPIs at the bottom frame the scale of our data: **271,888 flights**, **40,552 hotel stays**, **1,340 traveler profiles**, and **1,568 joined round-trip Flight + Hotel packages**."

### Slide 4 — System Architecture (`04:00 – 05:30`)
"Slide 4 shows TRAVELIQ's four-tier architecture from left to right.

On the left, raw CSVs feed our three Jupyter notebooks and our automated Python training modules (`train_flight_model.py` and `mlops_pipeline.py`).

In Tier 2, **MLflow** on port `5000` tracks every experiment run and registers `Voyage_Flight_Price_Model` with the `@champion` alias, while serialized production artifacts live in `models/` — `896.3 KB` for LightGBM and just `40.4 KB` for our refactored pure `scikit-learn` gender classifier.

In Tier 3, inference is decoupled into two dedicated Flask microservices — the Flight API on port `5001` and the Gender API on port `5002`, served via Waitress locally and Gunicorn in Docker — alongside our shared hybrid recommendation engine (`app/recommender.py`).

Finally, Tier 4 hosts the five-page TRAVELIQ Streamlit platform on port `8501` plus our Docker Compose, Kubernetes/HPA, Airflow, and Jenkins operational layer. Notice one key engineering choice: Streamlit never imports ML models directly for predictions; it talks to the APIs over HTTP via `app/api_client.py`."

---

## Segment 2 — Dataset Audit & Structural Realities (`05:30 – 10:00`)

### Slide 5 — Dataset Audit (`05:30 – 10:00`)
"Before training a single model, we audited every column, join key, and edge case in `travel_capstone_dataset/`. All three CSV files had zero `NaN` nulls and zero exact duplicate rows, **yet three hidden structural realities completely shaped our design**:

1. **In `flights.csv` (271,888 rows):** There are 9 Brazilian regional hub cities, 70 routes, 3 cabin classes (`economic`, `premium`, `firstClass`), and 3 agencies (`CloudFy`, `Rainbow`, `FlyingDrops`). Crucially, the `time` column ranges from `0.44` to `2.44` — confirming flight duration is in **hours**, not minutes. Even more importantly, grouping by `(from, to, flightType, agency, time, distance)` yields **only 490 unique fixed-price flight configurations**. That single discovery made group-aware validation mandatory.
2. **In `hotels.csv` (40,552 rows):** Cross-tabulating `place` and `name` proved an exact 1-to-1 bijection — **9 cities, 9 hotels**, each with one constant nightly rate and stay lengths of 1 to 4 days. However, joining `hotels.csv` with `flights.csv` on `travelCode` gave a **100.0% destination city match**, enabling genuine Flight + Hotel package recommendations.
3. **In `users.csv` (1,340 rows):** Gender contains `452` `male`, `448` `female`, and `440` literal lowercase `"none"` records — almost a third (`32.84%`) of the dataset! We didn't guess what `"none"` meant; we tested it statistically. A chi-square test against `company` gave `p = 0.3750`, and a one-way ANOVA against `age` gave `p = 0.6185`. Plus, `66.8%` of `"none"` rows share first names with labeled travelers, including exact full-name duplicates like *David Robinson* and *Charlotte Johnson*. That proves `"none"` is **Missing Completely At Random (MCAR)** — undisclosed labels, NOT a genuine third gender class.

**[YOUR DECISION / REFLECTION PROMPT: Explain briefly how discovering the 1-hotel-per-city limitation and the MCAR 'none' labels changed your approach to building trustworthy models rather than blindly running starter code.]**"

---

## Segment 3 — Flight Price Regression (`10:00 – 15:30`)

### Slide 6 — Flight Regression Methodology (`10:00 – 12:30`)
"Let's look at Track 1: Flight Price Prediction (`notebooks/01_flight_price_prediction.ipynb` and `scripts/train_flight_model.py`).

Because 271,888 rows come from just 490 unique configurations, a random 80/20 split puts every configuration into both train and test. Instead, we used **`GroupShuffleSplit`** grouped on the configuration key: `392` configurations (`221,274` rows) train the model, while `98` completely unseen configurations (`50,614` rows) form our holdout test set with **zero configuration overlap**. Hyperparameter tuning inside training also uses **3-fold `GroupKFold`**.

For feature engineering, we extract `year`, `month`, `day`, and `day_of_week` explicitly cast to `int32` — a subtle detail that turned out to be essential because pandas 3.x produces `int32` during training and MLflow's strict schema validation rejects `int64` at inference time! We also engineer `from_to` to capture directional corridor effects, and wrap `OneHotEncoder(handle_unknown="ignore")` and the estimator inside one atomic `scikit-learn` `Pipeline`."

### Slide 7 — Flight Regression Results (`12:30 – 15:30`)
"Here are our verified holdout metrics across all five models on those `50,614` unseen-configuration test rows:

- **Linear Regression baseline:** `MAE = $82.13`, `RMSE = $100.01`, `R² = 0.9145` — cannot model non-linear interactions between cabin class, agency, and route distance.
- **Random Forest:** improves sharply to `MAE = $54.63`, `RMSE = $70.55`, `R² = 0.9575`, but with a larger artifact and slower inference.
- **XGBoost:** `MAE = $43.09`, `RMSE = $54.46`, `R² = 0.9746`.
- **`GroupKFold`-Tuned LightGBM:** `MAE = $45.28`, `RMSE = $54.26`, `R² = 0.9748`.
- **Default LightGBM (Production Champion):** best across every key metric — **`MAE = $41.73`, `MSE = 2,754.14`, `RMSE = $52.48`, and `R² = 0.9765`**, cutting RMSE nearly in half versus Linear Regression on completely unseen flight schedules!

We saved this champion to `models/flight_price_pipeline.joblib` (`896.3 KB`), registered it in MLflow as `Voyage_Flight_Price_Model` Version 2 with the `@champion` alias, and confirmed **exact `$0.0000` prediction parity** between the live MLflow HTTP registry and the local joblib fallback."

---

## Segment 4 — Gender Classification & Responsible AI (`15:30 – 21:00`)

### Slide 8 — Resolving the 0.52 vs. 0.8830 F1 Audit (`15:30 – 18:30`)
"Track 2 (`notebooks/02_gender_classification.ipynb`) involved one of the most revealing technical audits in the whole project.

Initially, the starter repository reported a macro F1 of around `0.52`, whereas our Phase 3 MLOps pipeline had an F1 gate of `>= 0.85`. Whenever you see a jump like `0.52` to `0.88`, your first instinct as an ML engineer should be: *is there data leakage?*

So we audited the data split row by row:
1. First, why was the starter model at `0.52`? Strictly because it kept the 440 MCAR `"none"` rows as a third target class. Since `"none"` is statistically random missingness, recall on `"none"` was only `0.17` to `0.33`, dragging macro F1 down.
2. Second, on our clean 900-row binary dataset (`720 train / 180 stratified test`), we verified **zero overlapping user codes** and **zero overlapping full names**, and confirmed all TF-IDF vectorizers fit strictly inside the `Pipeline` on `X_train`.
3. Third — and this is the most important nuance — we dissected the 180 test rows by first-name familiarity:
   - `110 of 180` users (`61.1%`) share common first names (like *Michael* or *Patricia*) that also appeared in training -> **Accuracy `0.9727`, F1 `0.9726`**.
   - `70 of 180` users (`38.9%`) have brand-new first names -> **Accuracy `0.7429`, F1 `0.7454`**.
   - And under a strict **`GroupShuffleSplit`** holding out entire first-name groups (`n=168`, `0%` overlap), pure character n-grams (`2–5`) plus morphological suffix features achieve **Accuracy `0.6369`, F1 `0.6366`**.

That explains the entire behavior mathematically with zero leakage — AND it motivated us to upgrade `scripts/mlops_pipeline.py` to enforce **dual quality gates**: both `stratified_f1 >= 0.85` AND `unseen_first_name_f1 >= 0.58`.

At the same time, replacing the starter repo's `SentenceTransformer` + PyTorch pipeline with deterministic pure `scikit-learn` features in `app/gender_features.py` shrank the saved model from **`91.9 MB` down to `40.4 KB`** — a **2,275× reduction**!"

### Slide 9 — Model Comparison & Responsible AI (`18:30 – 21:00`)
"Slide 9 shows all five binary classifiers evaluated with 5-fold Stratified CV and the 180-row holdout test set. While Random Forest edged slightly higher on holdout accuracy (`0.8889`), **Tuned Logistic Regression (`C=8.0`)** achieved the highest 5-fold CV F1 (`0.9083`), strong test accuracy (`0.8833`), Weighted F1 (`0.8830`), ROC-AUC (`0.9533`), well-calibrated class probabilities, and that tiny `40.4 KB` footprint. On the confusion matrix, it correctly predicts `75/90` female and `84/90` male holdout records.

Just as importantly, look at the Responsible AI box on the right. Inferring gender statistically from first names is inherently limited and culturally dependent. In TRAVELIQ, this microservice is **strictly labeled experimental**: the API includes `experimental: true` and an `experimental_notice` field in every response, and the UI shows a prominent amber warning making clear that predictions are never verified personal identity and must never be used for consequential or policy-sensitive decisions.

**[YOUR DECISION / REFLECTION PROMPT: Share in 2–3 sentences why you chose to keep the lightweight Logistic Regression model and add explicit Responsible AI notices rather than trying to squeeze a slightly higher test score with a black-box deep model.]**"

---

## Segment 5 — Hybrid Travel & Hotel Recommender (`21:00 – 26:00`)

### Slide 10 — Hybrid 1,568-Package Recommender (`21:00 – 26:00`)
"Track 3 (`notebooks/03_travel_recommendation.ipynb` and `app/recommender.py`) tackles the one-hotel-per-city reality head-on.

Instead of pretending the dataset lets us pick between five hotels in Rio de Janeiro, we join outbound and return flights from `flights.csv` with `hotels.csv` on `travelCode` and `users.csv` to assemble **1,568 real round-trip Flight + Hotel travel packages** across origin cities, destinations, cabin classes, agencies, and stay durations.

Our hybrid engine has two layers:
1. First, an offline **User-Based Collaborative Filtering** layer using cosine similarity over historical user-destination matrices. Evaluated on `1,228` travelers with a time-ordered 80/20 split (`K=3`), User-Based CF achieves **`HitRate@3 = 0.8974`**, **`Precision@3 = 0.5532`**, **`NDCG@3 = 0.5680`**, and **`100.0%` catalog coverage (all 9 cities)** — tripling the Popularity baseline's `33.3%` coverage (`3/9` cities) while keeping hit rate virtually identical!
2. Second, a transparent **`0–100` multi-factor ranking score** combining Budget Fit, Duration Alignment, Personal/CF destination affinity (gracefully falling back to cold-start popularity for anonymous users), and historical route popularity, plus optional destination-diversity mode so travelers exploring broadly see different cities instead of five variations of one city.

On a stress test of 200 randomized queries, our engine achieved **`100.0%` budget compliance**, **`89.7%` exact duration match**, **`4.78 / 5.00` average unique cities** per Top-5 explorer query, and **`95.5%` personalization differentiation**. And because `app/recommender.py` is shared directly with the notebook, production reproduces Notebook 03's verification benchmark scores to the exact hundredth (`71.28`, `76.99`, `54.91`)."

---

## Segment 6 — Backend APIs, Live Application & MLflow Demo (`26:00 – 34:00`)

### Slide 11 — Flask REST Microservices (`26:00 – 27:30`)
"Moving into our full-stack serving layer: both APIs use Flask wrapped in **Waitress** locally and **Gunicorn** in Docker, configured centrally via `app/config.py`.

The **Flight API (`:5001`)** connects to `MLFLOW_TRACKING_URI` to load `Voyage_Flight_Price_Model@champion`, caches the pipeline in memory, validates all request fields strictly, casts date features to `int32`, and automatically falls back to `models/flight_price_pipeline.joblib` if MLflow goes offline — always declaring `model_source` transparently in `/health` and `/predict`.

The **Gender API (`:5002`)** loads the `40.4 KB` artifact once at startup with zero PyTorch dependencies and returns calibrated probabilities summing to `1.0` alongside the ethical disclaimer."

### Slides 12–16 & Slide 20 — Live Application & MLflow Walkthrough (`27:30 – 34:00`)
> *(Switch to live browser tabs or walk through the captured screenshots on Slides 12–16 following the Slide 20 6-step demo flow!)*

"Now let's see the live TRAVELIQ platform in action — starting with **Slide 12 (Home Dashboard)**:
- Notice the cohesive Navy, White, and Modern Blue design system and vector TRAVELIQ logo. On the left sidebar, the **Live API Services** widget actively polls `/health` on both microservices, displaying green **`ONLINE`** pills and explicitly confirming `Source: mlflow_registry` and `Model: 2.0-lightweight-binary`.
- Across the top, five KPI cards summarize the entire dataset, followed by Quick Launch cards and two interactive Plotly overview charts.

Moving to **Slide 13 (Flight Price Predictor)**:
- When I select *Florianopolis (SC)* to *Aracaju (SE)*, *firstClass*, *CloudFy*, duration `2.10` hours, distance `808.85` km, and click **Predict Flight Price**, Streamlit calls `POST http://127.0.0.1:5001/predict`.
- The right column prominently displays **`$1,480.04 USD`**, compares it directly to the **Historical Route Mean (`$1,468.03`, delta `+$12.01`)**, displays the **`MLflow @champion`** serving badge, and renders an expandable request summary table.

On **Slide 14 (Trip & Hotel Package Recommendations)**:
- Right below the header, you see the blue **Dataset Transparency banner** explaining the one-partner-hotel-per-city constraint.
- With a 3-day trip and `$2,500` budget ceiling, TRAVELIQ filters `924` eligible matches, shows KPI cards (`Top Match 88.1 / 100`, `Lowest Package $1,166.35`, `Avg Savings $1,042.04`), renders a stacked bar chart separating **Round-Trip Flight Cost** vs. **Hotel Stay Cost** against the dashed `$2,500` budget line, and displays ranked cards — each with an explicit **`Why recommended`** rationale showing exact Budget Fit, Duration, Personal/CF, and Popularity sub-scores. *(In live demo: toggle from Anonymous to User #0 to show personalization reordering live!)*

On **Slide 15 (Gender Classification Demo & Travel Analytics)**:
- On the left, the Gender Demo shows the amber Responsible AI warning above the form and returns a calibrated horizontal bar chart (`98.3% Female / 1.6% Male` for sample *Wilma Mcinnis*).
- On the right, the **Analytics** page offers five interactive Plotly tabs — Flight Price Distribution, Popular Routes, Destination Popularity, Travel Duration, and Hotel & Package Pricing.

And on **Slide 16 (Live MLflow 3.18 Server on `:5000`)**:
- The left view shows all **6 logged runs** under `Voyage Analytics - Flight Price Prediction`, letting us compare Linear Regression, Random Forest, XGBoost, and LightGBM side by side.
- The right view shows the **Model Registry** for `Voyage_Flight_Price_Model`, with Version 1 and Version 2 cleanly versioned and the **`@champion`** alias assigned to Version 2.

**[YOUR DECISION / REFLECTION PROMPT: Mention one UI/UX design decision you made when redesigning TRAVELIQ (for example, adding the route mean comparison delta, surfacing live API health badges, or showing transparent 'Why recommended' rationale) and why it matters to real users.]**"

---

## Segment 7 — MLOps Stack & 57 Automated Tests (`34:00 – 38:00`)

### Slide 17 — Docker Compose & Kubernetes + HPA (`34:00 – 35:30`)
"Now let's examine how TRAVELIQ is packaged for production MLOps.

In **Docker & Docker Compose**, every service has its own lean Dockerfile, non-root user, pinned per-service requirements, and HTTP health check:
- `flight-api` installs `libgomp1` specifically because LightGBM requires OpenMP on Debian slim images.
- `gender-api` removes PyTorch and `sentence-transformers` completely — cutting over 2 GB of unnecessary container bloat!
- All four containers communicate by service name (`http://mlflow:5000`, `http://flight-api:5001`, `http://gender-api:5002`) over a private `traveliq-net` bridge with a persistent `traveliq_mlflow_data` volume.

In **Kubernetes (`kubernetes/`)**, our manifests match exact `traveliq-*:2.0` image tags, centralize DNS URLs in `configmap.yaml`, keep credentials out of Git (`secret.example.yaml`), attach a `2Gi` PVC for MLflow, configure `startupProbe`, `readinessProbe`, and `livenessProbe`, and define `autoscaling/v2` **Horizontal Pod Autoscalers** scaling Flight API across `2–5` pods (`70% CPU / 80% Memory`) and Gender API across `1–4` pods (`70% CPU`).

I want to be fully transparent about runtime status: all container files, compose links, Kubernetes YAML schemas, and HPA rules are validated by automated tests, while **live `docker compose up` and live Kubernetes/HPA rollouts are pending** because Docker Desktop and WSL2 are not installed on my non-administrator Windows laptop."

### Slide 18 — Apache Airflow & Jenkins CI/CD (`35:30 – 36:45`)
"For continuous retraining, `airflow/dags/traveliq_ml_pipeline.py` orchestrates seven gated tasks backed by `scripts/mlops_pipeline.py`:
1. `validate_datasets` checks schemas, row counts, and missingness.
2. `preprocess_and_retrain_gate` uses a `ShortCircuitOperator` comparing SHA-256 dataset hashes so unchanged data doesn't waste compute.
3. Three parallel branches train the Flight model, Gender model, and 1,568-package Recommendation catalog.
4. `evaluate_models` enforces strict numerical quality gates — Flight `R² >= 0.90` & `RMSE <= 80`, Gender **both** `Stratified F1 >= 0.85` **and** `Unseen-First-Name F1 >= 0.58`, and Recommender `>= 1,000` packages with `100%` budget compliance.
5. Only when every gate passes does `register_and_promote_champion` register and promote `@champion` in MLflow.

By cleanly decoupling the core pipeline logic from the Airflow wrapper, we actually executed the entire 7-stage training, promotion, AND failed-gate blocking workflow live inside an isolated sandbox in `tests/test_integration.py`! The Dockerized Airflow UI (`:8080`) itself awaits Docker Desktop.

Similarly, our cross-platform root `Jenkinsfile` automates 7 stages from Checkout and AST/YAML linting through Pytest JUnit reporting, notebook/artifact verification, Docker builds, and non-production `kubectl --dry-run=client` validation — halting immediately on any test failure and never auto-deploying to production."

### Slide 19 — Automated Testing (`57 / 57` Passing) (`36:45 – 38:00`)
"Quality assurance ties everything together. Running `pytest -v` executes **57 automated tests across 6 test modules — 57 passed, 0 failed**:
- `12` Flight API tests, `12` Gender API tests, `13` Recommender tests, `8` API Client tests, `7` MLOps infrastructure tests, and `5` end-to-end integration tests.

And these tests aren't superficial — they caught two genuine production bugs during development:
1. Pandas 3.x produced `int32` date columns during training, causing MLflow signature enforcement to reject default `int64` frames at API inference time — fixed cleanly in `app/app.py`.
2. MLflow 3.18 cached the SQLAlchemy registry URI across sandbox runs even after `set_tracking_uri()` — fixed cleanly by pairing `mlflow.set_registry_uri()` in `scripts/mlops_pipeline.py`.

**[YOUR DECISION / REFLECTION PROMPT: Mention what writing real unit and integration tests taught you about deploying ML systems compared to stopping once a Jupyter notebook finishes.]**"

---

## Segment 8 — Honest Limitations, Roadmap & Conclusion (`38:00 – 40:00`)

### Slide 21 — Limitations & Future Roadmap (`38:00 – 39:15`)
"Every honest engineering project documents its boundaries clearly:
- **Data limitations:** One partner hotel per city prevents within-city hotel competition; 490 fixed fare configs simplify real-world dynamic airline pricing; and unseen first-name gender F1 (`0.6366`) underscores why the gender classifier remains strictly an experimental demo.
- **Environment limitations:** Live Docker container builds, Kubernetes/HPA load tests, Airflow Webserver UI, and Jenkins server builds require Docker Desktop + WSL2 with Administrator elevation and are clearly labeled pending rather than fabricated.

For our next production phase, the roadmap is straightforward: run live Docker/Kubernetes HPA benchmarks on a container-enabled host, add statistical data-drift detection (e.g., PSI/Evidently) before Airflow Stage 2, connect richer multi-hotel inventory feeds, and layer Prometheus/Grafana observability with API authentication."

### Slide 22 — Conclusion & Q&A (`39:15 – 40:00`)
"To summarize: **TRAVELIQ** delivers an independently engineered, thoroughly tested, and fully explainable Travel & Tourism ML and MLOps platform — uniting leakage-free LightGBM airfare forecasting (`R² = 0.9765`), an ethically scoped `40.4 KB` binary gender microservice, an explainable `1,568`-package hybrid recommender, a live MLflow `@champion` registry, a complete modern Streamlit UI, full MLOps infrastructure definitions, and **57 passing automated tests**, all with honest attribution and transparent verification.

Thank you very much for your time — I would be happy to take any questions!"
