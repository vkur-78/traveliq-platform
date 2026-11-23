"""
TRAVELIQ — Modular MLOps Pipeline Tasks (Invoked by Apache Airflow & CLI).

Stages:
  1. validate_datasets             : Schema, null, range, and SHA-256 fingerprint checks.
  2. preprocess_and_retrain_gate   : Preprocessing checks + skip gate to avoid unnecessary retraining.
  3. train_flight_model            : Leakage-free GroupShuffleSplit LightGBM candidate training.
  4. train_gender_model            : Lightweight binary LogisticRegression candidate training.
  5. build_recommendation_catalog  : Hybrid Flight+Hotel package catalog & constraint validation.
  6. evaluate_models               : Strict quality gates blocking unvalidated candidates.
  7. register_and_promote_champion : Promotes validated artifacts & updates MLflow @champion aliases.
"""

import hashlib
import json
import logging
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.gender_features import build_gender_features
from app.recommender import TravelPackageRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("traveliq.mlops_pipeline")

RANDOM_STATE = 42
FLIGHT_MODEL_NAME = os.getenv("MODEL_NAME", "Voyage_Flight_Price_Model")
GENDER_MODEL_NAME = os.getenv("GENDER_MODEL_NAME", "TravelIQ_Gender_Classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Quality gates that MUST pass before any model is registered or promoted
QUALITY_GATES = {
    "flight_min_r2": 0.95,
    "flight_max_rmse": 65.0,
    "gender_min_f1": 0.85,
    "gender_min_unseen_first_name_f1": 0.58,
    "gender_min_roc_auc": 0.90,
    "gender_max_artifact_kb": 500.0,
    "recommender_min_budget_compliance": 1.0,
    "recommender_min_packages": 1000,
}

FLIGHT_CAT_COLS = ["from", "to", "flightType", "agency"]
FLIGHT_NUM_COLS = ["time", "distance", "year", "month", "day", "day_of_week"]
FLIGHT_GROUP_COLS = ["from", "to", "flightType", "time", "distance", "agency"]
FLIGHT_LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


def resolve_project_root(project_root: Optional[Path] = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    env_root = os.getenv("TRAVELIQ_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if Path("/opt/airflow/project").exists():
        return Path("/opt/airflow/project").resolve()
    return Path(__file__).resolve().parents[1]


def resolve_mlflow_uri(project_root: Path) -> str:
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    if Path("/.dockerenv").exists():
        return "http://host.docker.internal:5000"
    local_http = "http://127.0.0.1:5000"
    try:
        with urllib.request.urlopen(f"{local_http}/health", timeout=1.5):
            return local_http
    except Exception:
        return f"sqlite:///{(project_root / 'mlflow.db').resolve().as_posix()}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_fingerprint(dataset_dir: Path) -> str:
    """Compute a combined SHA-256 fingerprint across all three capstone CSVs."""
    combined = hashlib.sha256()
    for name in ("flights.csv", "hotels.csv", "users.csv"):
        p = dataset_dir / name
        combined.update(name.encode("utf-8"))
        combined.update(_sha256_file(p).encode("utf-8"))
    return combined.hexdigest()


# ==============================================================================
# STAGE 1: DATASET VALIDATION
# ==============================================================================


def validate_datasets(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Validate flights.csv, hotels.csv, and users.csv schema, integrity, and fingerprint."""
    root = resolve_project_root(project_root)
    data_dir = root / "travel_capstone_dataset"

    expected_schemas = {
        "flights.csv": [
            "travelCode", "userCode", "from", "to", "flightType",
            "price", "time", "distance", "agency", "date",
        ],
        "hotels.csv": [
            "travelCode", "userCode", "name", "place", "days",
            "price", "total", "date",
        ],
        "users.csv": ["code", "company", "name", "gender", "age"],
    }

    summary: Dict[str, Any] = {"datasets": {}}
    for fname, req_cols in expected_schemas.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Required dataset missing: {path}")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Dataset is empty: {fname}")

        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset {fname} missing required columns: {missing_cols}")

        null_count = int(df[req_cols].isnull().sum().sum())
        if null_count > 0:
            raise ValueError(f"Dataset {fname} has {null_count} missing values in required columns.")

        if fname == "flights.csv":
            for num_col in ("price", "time", "distance"):
                if (df[num_col] <= 0).any():
                    raise ValueError(f"flights.csv contains non-positive values in '{num_col}'.")
            pd.to_datetime(df["date"], format="%m/%d/%Y")

        elif fname == "hotels.csv":
            for num_col in ("days", "price", "total"):
                if (df[num_col] <= 0).any():
                    raise ValueError(f"hotels.csv contains non-positive values in '{num_col}'.")
            pd.to_datetime(df["date"], format="%m/%d/%Y")

        elif fname == "users.csv":
            labeled = int(df["gender"].isin(["male", "female"]).sum())
            undisclosed = int((df["gender"] == "none").sum())
            if labeled < 100:
                raise ValueError(f"Insufficient labeled gender rows in users.csv: {labeled}")
            summary["users_labeled_gender"] = labeled
            summary["users_undisclosed_none"] = undisclosed

        summary["datasets"][fname] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicates": int(df.duplicated().sum()),
        }

    fingerprint = compute_dataset_fingerprint(data_dir)
    summary["dataset_fingerprint"] = fingerprint
    logger.info("Dataset validation PASSED. Fingerprint=%s", fingerprint[:12])
    return summary


# ==============================================================================
# STAGE 2: DATA PREPROCESSING & RETRAINING SKIP GATE
# ==============================================================================


def preprocess_and_check_retrain(
    project_root: Optional[Path] = None,
    force_retrain: Optional[bool] = None,
) -> bool:
    """
    Validate preprocessing readiness and decide whether retraining is necessary.

    Returns True to run downstream training tasks, or False to short-circuit cleanly
    when datasets are unchanged and validated production artifacts already exist.
    """
    root = resolve_project_root(project_root)
    data_dir = root / "travel_capstone_dataset"
    models_dir = root / "models"
    candidates_dir = models_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Verify preprocessing transformations cleanly before training
    flights_sample = pd.read_csv(data_dir / "flights.csv", nrows=500)
    parsed = pd.to_datetime(flights_sample["date"], format="%m/%d/%Y")
    assert parsed.notna().all(), "Flight date preprocessing failed."

    users_df = pd.read_csv(data_dir / "users.csv")
    labeled = users_df[users_df["gender"].isin(["male", "female"])]
    gender_feats = build_gender_features(labeled[["name", "company", "age"]].head(20))
    assert not gender_feats.isnull().any().any(), "Gender feature engineering produced nulls."

    fingerprint = compute_dataset_fingerprint(data_dir)
    state_file = models_dir / "pipeline_state.json"
    flight_artifact = models_dir / "flight_price_pipeline.joblib"
    gender_artifact = models_dir / "gender_classifier.joblib"

    if force_retrain is None:
        force_retrain = os.getenv("TRAVELIQ_FORCE_RETRAIN", "0").strip().lower() in (
            "1", "true", "yes",
        )

    if not force_retrain and flight_artifact.exists() and gender_artifact.exists() and state_file.exists():
        try:
            prev_state = json.loads(state_file.read_text(encoding="utf-8"))
            if (
                prev_state.get("dataset_fingerprint") == fingerprint
                and prev_state.get("status") == "promoted"
            ):
                logger.info(
                    "Dataset fingerprint %s unchanged and verified artifacts exist. "
                    "Skipping unnecessary retraining (set TRAVELIQ_FORCE_RETRAIN=1 to override).",
                    fingerprint[:12],
                )
                return False
        except Exception as exc:
            logger.warning("Could not parse pipeline_state.json (%s); proceeding with training.", exc)

    # Record current run context for downstream candidate tasks
    run_meta = {
        "dataset_fingerprint": fingerprint,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (candidates_dir / "run_context.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    logger.info("Preprocessing checks PASSED. Proceeding to candidate training.")
    return True


# ==============================================================================
# STAGE 3: FLIGHT MODEL CANDIDATE TRAINING (LEAKAGE-FREE GROUP SPLIT)
# ==============================================================================


def train_flight_candidate(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Train LightGBM flight candidate model, evaluate with GroupShuffleSplit, log to MLflow."""
    root = resolve_project_root(project_root)
    df = pd.read_csv(root / "travel_capstone_dataset" / "flights.csv")
    parsed_date = pd.to_datetime(df["date"], format="%m/%d/%Y")

    model_df = df.copy()
    model_df["year"] = parsed_date.dt.year.astype("int32")
    model_df["month"] = parsed_date.dt.month.astype("int32")
    model_df["day"] = parsed_date.dt.day.astype("int32")
    model_df["day_of_week"] = parsed_date.dt.day_of_week.astype("int32")

    flight_groups = df[FLIGHT_GROUP_COLS].astype(str).agg(" | ".join, axis=1)
    X = model_df[FLIGHT_CAT_COLS + FLIGHT_NUM_COLS].copy()
    y = model_df["price"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=flight_groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    overlap = set(flight_groups.iloc[train_idx]).intersection(set(flight_groups.iloc[test_idx]))
    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} overlapping flight groups.")

    def _make_pipe() -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            ("categorical", OneHotEncoder(handle_unknown="ignore"), FLIGHT_CAT_COLS),
                            ("numerical", "passthrough", FLIGHT_NUM_COLS),
                        ]
                    ),
                ),
                ("model", LGBMRegressor(**FLIGHT_LGBM_PARAMS)),
            ]
        )

    eval_pipe = _make_pipe()
    eval_pipe.fit(X_train, y_train)
    y_pred = eval_pipe.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    # Refit candidate production pipeline on full dataset after unbiased holdout scoring
    prod_pipe = _make_pipe()
    prod_pipe.fit(X, y)

    candidates_dir = root / "models" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidates_dir / "flight_price_pipeline.joblib"
    joblib.dump(prod_pipe, candidate_path)

    # Track candidate run in MLflow (NOT registered/promoted until evaluate_models passes!)
    tracking_uri = resolve_mlflow_uri(root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("TRAVELIQ - Flight Price Prediction")

    with mlflow.start_run(run_name="Airflow_Flight_Candidate") as run:
        mlflow.log_params({
            "model": "LightGBM",
            **FLIGHT_LGBM_PARAMS,
            "validation_strategy": "GroupShuffleSplit (unseen configurations)",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=prod_pipe,
            name="flight_price_pipeline_candidate",
            input_example=X_test.head(5),
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
            ],
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    report = {
        "artifact_path": str(candidate_path),
        "mlflow_run_id": run_id,
        "mlflow_model_uri": model_uri,
        "metrics": metrics,
        "sample_input": X_test.head(3).to_dict(orient="records"),
    }
    (candidates_dir / "flight_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Flight candidate trained: RMSE=%.2f, R2=%.4f", metrics["RMSE"], metrics["R2"])
    return report


# ==============================================================================
# STAGE 4: GENDER MODEL CANDIDATE TRAINING (LIGHTWEIGHT BINARY SKLEARN)
# ==============================================================================


def train_gender_candidate(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Train lightweight binary male/female LogisticRegression candidate & log to MLflow."""
    root = resolve_project_root(project_root)
    users = pd.read_csv(root / "travel_capstone_dataset" / "users.csv")
    labeled = users[users["gender"].isin(["male", "female"])].copy().reset_index(drop=True)

    X_all = build_gender_features(labeled[["name", "company", "age"]])
    y_all = labeled["gender"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )

    cat_cols = ["last_1", "last_2", "last_3", "first_2", "company"]
    num_cols = [
        "age", "first_name_len", "ends_with_vowel",
        "ends_with_a", "ends_with_e", "ends_with_y_ie",
    ]

    def _make_pipe() -> Pipeline:
        prep = ColumnTransformer(
            transformers=[
                (
                    "fn_char_ngrams",
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                    "first_name",
                ),
                ("fn_word", TfidfVectorizer(analyzer="word", min_df=1), "first_name"),
                ("cat_features", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num_features", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("preprocessor", prep),
                ("classifier", LogisticRegression(C=8.0, max_iter=3000, random_state=RANDOM_STATE)),
            ]
        )

    eval_pipe = _make_pipe()
    eval_pipe.fit(X_train, y_train)
    y_pred = eval_pipe.predict(X_test)
    female_idx = list(eval_pipe.classes_).index("female")
    y_prob = eval_pipe.predict_proba(X_test)[:, female_idx]

    # Strict unseen-first-name generalization check (zero overlapping first names)
    gss_fn = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    tr_u, te_u = next(gss_fn.split(X_all, y_all, groups=X_all["first_name"]))
    unseen_pipe = _make_pipe()
    unseen_pipe.fit(X_all.iloc[tr_u], y_all.iloc[tr_u])
    y_pred_unseen = unseen_pipe.predict(X_all.iloc[te_u])

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == "female").astype(int), y_prob)),
        "unseen_first_name_f1": float(
            f1_score(y_all.iloc[te_u], y_pred_unseen, average="weighted", zero_division=0)
        ),
    }

    prod_pipe = _make_pipe()
    prod_pipe.fit(X_all, y_all)

    candidates_dir = root / "models" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidates_dir / "gender_classifier.joblib"

    bundle = {
        "artifact_version": "2.0-lightweight-binary",
        "model_type": "LogisticRegression (First-Name Char/Word TF-IDF + Morphology + Demographics)",
        "pipeline": prod_pipe,
        "raw_input_fields": ["name", "company", "age"],
        "engineered_feature_columns": list(X_all.columns),
        "classes": list(prod_pipe.classes_),
        "target_handling": "Binary male/female trained on labeled rows; 'none' excluded as undisclosed.",
        "best_params": {"classifier__C": 8.0},
        "metrics": metrics,
    }
    joblib.dump(bundle, candidate_path, compress=3)
    size_kb = round(candidate_path.stat().st_size / 1024.0, 2)

    tracking_uri = resolve_mlflow_uri(root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("TRAVELIQ - Gender Classification")

    with mlflow.start_run(run_name="Airflow_Gender_Candidate") as run:
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": 8.0,
            "labeled_samples": len(labeled),
            "artifact_size_kb": size_kb,
        })
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=prod_pipe,
            name="gender_classifier_candidate",
            input_example=X_test.head(5),
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    report = {
        "artifact_path": str(candidate_path),
        "artifact_size_kb": size_kb,
        "mlflow_run_id": run_id,
        "mlflow_model_uri": model_uri,
        "metrics": metrics,
    }
    (candidates_dir / "gender_candidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Gender candidate trained: F1=%.4f, AUC=%.4f, size=%.1f KB", metrics["f1_weighted"], metrics["roc_auc"], size_kb)
    return report


# ==============================================================================
# STAGE 5: RECOMMENDATION PACKAGE CATALOG GENERATION & VALIDATION
# ==============================================================================


def build_recommendation_catalog(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Generate and validate the hybrid Flight + Hotel recommendation catalog."""
    root = resolve_project_root(project_root)
    recommender = TravelPackageRecommender.from_csv_dir(root / "travel_capstone_dataset")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_users = rng.choice(
        recommender.user_profiles.index.to_numpy(),
        size=min(50, len(recommender.user_profiles)),
        replace=False,
    )

    total_pkgs = 0
    budget_ok = 0
    dest_diversity = []
    for u in sample_users:
        req_days = int(rng.choice([1, 2, 3, 4]))
        req_budget = float(rng.choice([1800, 2200, 2600, 3200]))
        recs, meta = recommender.recommend(days=req_days, max_budget=req_budget, user_code=int(u), top_n=5)
        if meta["status"] == "ok" and not recs.empty:
            total_pkgs += len(recs)
            budget_ok += int((recs["package_total"] <= req_budget + 1e-6).sum())
            dest_diversity.append(int(recs["destination"].nunique()))

    compliance = float(budget_ok / max(total_pkgs, 1))
    summary = {
        "package_count": int(len(recommender.package_catalog)),
        "cities_count": int(len(recommender.city_to_hotel)),
        "hotels_count": int(len(set(recommender.city_to_hotel.values()))),
        "users_indexed": int(len(recommender.user_profiles)),
        "validation_queries": int(len(sample_users)),
        "budget_compliance": compliance,
        "avg_destination_diversity": round(float(np.mean(dest_diversity)) if dest_diversity else 0.0, 2),
    }

    candidates_dir = root / "models" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "recommendation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Recommendation catalog validated: %d packages, budget_compliance=%.2f%%",
        summary["package_count"],
        compliance * 100.0,
    )
    return summary


# ==============================================================================
# STAGE 6: MODEL EVALUATION & PROMOTION QUALITY GATES
# ==============================================================================


def evaluate_models(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Strictly validate candidate artifacts against metric, size, and smoke-inference gates.
    Raises RuntimeError if ANY check fails so unvalidated models can never be promoted.
    """
    root = resolve_project_root(project_root)
    candidates_dir = root / "models" / "candidates"

    flight_report = json.loads((candidates_dir / "flight_candidate_report.json").read_text(encoding="utf-8"))
    gender_report = json.loads((candidates_dir / "gender_candidate_report.json").read_text(encoding="utf-8"))
    rec_report = json.loads((candidates_dir / "recommendation_summary.json").read_text(encoding="utf-8"))

    failures = []

    # 1. Flight metric & smoke inference checks
    fm = flight_report["metrics"]
    if fm["R2"] < QUALITY_GATES["flight_min_r2"]:
        failures.append(f"Flight R2 {fm['R2']:.4f} < {QUALITY_GATES['flight_min_r2']}")
    if fm["RMSE"] > QUALITY_GATES["flight_max_rmse"]:
        failures.append(f"Flight RMSE {fm['RMSE']:.2f} > {QUALITY_GATES['flight_max_rmse']}")

    flight_pipe = joblib.load(Path(flight_report["artifact_path"]))
    sample_flight_df = pd.DataFrame(flight_report["sample_input"])
    flight_preds = flight_pipe.predict(sample_flight_df)
    if not np.all(np.isfinite(flight_preds)) or np.any(flight_preds <= 0):
        failures.append(f"Flight smoke inference returned invalid predictions: {flight_preds}")

    # 2. Gender metric, size & smoke inference checks
    gm = gender_report["metrics"]
    if gm["f1_weighted"] < QUALITY_GATES["gender_min_f1"]:
        failures.append(f"Gender F1 {gm['f1_weighted']:.4f} < {QUALITY_GATES['gender_min_f1']}")
    if (
        "unseen_first_name_f1" in gm
        and gm["unseen_first_name_f1"] < QUALITY_GATES["gender_min_unseen_first_name_f1"]
    ):
        failures.append(
            f"Gender unseen-first-name F1 {gm['unseen_first_name_f1']:.4f} < "
            f"{QUALITY_GATES['gender_min_unseen_first_name_f1']}"
        )
    if gm["roc_auc"] < QUALITY_GATES["gender_min_roc_auc"]:
        failures.append(f"Gender ROC-AUC {gm['roc_auc']:.4f} < {QUALITY_GATES['gender_min_roc_auc']}")
    if gender_report["artifact_size_kb"] > QUALITY_GATES["gender_max_artifact_kb"]:
        failures.append(
            f"Gender artifact {gender_report['artifact_size_kb']} KB exceeds "
            f"{QUALITY_GATES['gender_max_artifact_kb']} KB lightweight limit."
        )

    gender_bundle = joblib.load(Path(gender_report["artifact_path"]))
    sample_gender = build_gender_features(
        pd.DataFrame([{"name": "John Smith", "company": "4You", "age": 30}])
    )
    g_pred = gender_bundle["pipeline"].predict(sample_gender)[0]
    if g_pred not in ("male", "female"):
        failures.append(f"Gender smoke inference returned invalid label: {g_pred}")

    # 3. Recommender checks
    if rec_report["budget_compliance"] < QUALITY_GATES["recommender_min_budget_compliance"]:
        failures.append(f"Recommender budget compliance {rec_report['budget_compliance']:.4f} < 1.0")
    if rec_report["package_count"] < QUALITY_GATES["recommender_min_packages"]:
        failures.append(f"Recommender package count {rec_report['package_count']} too low.")

    evaluation_result = {
        "passed": len(failures) == 0,
        "failures": failures,
        "flight_metrics": fm,
        "gender_metrics": gm,
        "recommender_summary": rec_report,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (candidates_dir / "evaluation_gate_result.json").write_text(
        json.dumps(evaluation_result, indent=2), encoding="utf-8"
    )

    if failures:
        raise RuntimeError("Model quality gates FAILED (blocking registration): " + " | ".join(failures))

    logger.info("All model evaluation quality gates PASSED.")
    return evaluation_result


# ==============================================================================
# STAGE 7: MLFLOW REGISTRATION & CHAMPION ALIAS PROMOTION (GATED)
# ==============================================================================


def register_and_promote_champion(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Promote validated candidate artifacts into models/, register in MLflow Model Registry,
    and update the @champion alias ONLY when evaluation_gate_result.json passed.
    """
    root = resolve_project_root(project_root)
    models_dir = root / "models"
    candidates_dir = models_dir / "candidates"

    gate_file = candidates_dir / "evaluation_gate_result.json"
    if not gate_file.exists():
        raise RuntimeError("Cannot promote models: evaluation_gate_result.json is missing.")

    gate_result = json.loads(gate_file.read_text(encoding="utf-8"))
    if not gate_result.get("passed", False):
        raise RuntimeError("Refusing to register/promote unvalidated models.")

    flight_report = json.loads((candidates_dir / "flight_candidate_report.json").read_text(encoding="utf-8"))
    gender_report = json.loads((candidates_dir / "gender_candidate_report.json").read_text(encoding="utf-8"))
    run_context = json.loads((candidates_dir / "run_context.json").read_text(encoding="utf-8"))

    # Copy validated artifacts atomically into production models/
    shutil.copy2(flight_report["artifact_path"], models_dir / "flight_price_pipeline.joblib")
    shutil.copy2(gender_report["artifact_path"], models_dir / "gender_classifier.joblib")

    # Register validated runs in MLflow Model Registry & update @champion aliases
    tracking_uri = resolve_mlflow_uri(root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()

    flight_mv = mlflow.register_model(
        model_uri=flight_report["mlflow_model_uri"],
        name=FLIGHT_MODEL_NAME,
    )
    client.set_registered_model_alias(
        name=FLIGHT_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=str(flight_mv.version),
    )

    gender_mv = mlflow.register_model(
        model_uri=gender_report["mlflow_model_uri"],
        name=GENDER_MODEL_NAME,
    )
    client.set_registered_model_alias(
        name=GENDER_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=str(gender_mv.version),
    )

    state = {
        "status": "promoted",
        "dataset_fingerprint": run_context["dataset_fingerprint"],
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "flight_model": {
            "registered_name": FLIGHT_MODEL_NAME,
            "version": str(flight_mv.version),
            "alias": MODEL_ALIAS,
            "metrics": flight_report["metrics"],
        },
        "gender_model": {
            "registered_name": GENDER_MODEL_NAME,
            "version": str(gender_mv.version),
            "alias": MODEL_ALIAS,
            "metrics": gender_report["metrics"],
        },
        "recommender": gate_result["recommender_summary"],
    }
    (models_dir / "pipeline_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    logger.info(
        "Promoted %s v%s (@%s) and %s v%s (@%s).",
        FLIGHT_MODEL_NAME, flight_mv.version, MODEL_ALIAS,
        GENDER_MODEL_NAME, gender_mv.version, MODEL_ALIAS,
    )
    return state
