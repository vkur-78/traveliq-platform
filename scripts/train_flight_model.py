import os
import urllib.request
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Project paths & reproducibility
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "travel_capstone_dataset" / "flights.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "flight_price_pipeline.joblib"

# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "Voyage Analytics - Flight Price Prediction",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Voyage_Flight_Price_Model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")


def resolve_mlflow_tracking_uri() -> str:
    """Resolve MLflow tracking URI across Docker/Airflow, local server, or local SQLite fallback."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    if Path("/.dockerenv").exists():
        return "http://host.docker.internal:5000"

    local_http = "http://127.0.0.1:5000"
    try:
        with urllib.request.urlopen(f"{local_http}/health", timeout=1.5):
            return local_http
    except Exception:
        sqlite_path = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
        return f"sqlite:///{sqlite_path}"


# ============================================================
# Feature & pipeline builders (shared with Notebook 01)
# ============================================================

CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency"]
NUMERICAL_FEATURES = ["time", "distance", "year", "month", "day", "day_of_week"]
GROUP_COLUMNS = ["from", "to", "flightType", "time", "distance", "agency"]

LGBM_PARAMS = {
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


def build_pipeline() -> Pipeline:
    """Create the preprocessing + LightGBM regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ]
    )
    model = LGBMRegressor(**LGBM_PARAMS)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_features(df: pd.DataFrame):
    """Engineer date features and return (X, y, flight_groups)."""
    parsed_date = pd.to_datetime(df["date"], format="%m/%d/%Y")

    model_df = df.copy()
    model_df["year"] = parsed_date.dt.year.astype("int32")
    model_df["month"] = parsed_date.dt.month.astype("int32")
    model_df["day"] = parsed_date.dt.day.astype("int32")
    model_df["day_of_week"] = parsed_date.dt.day_of_week.astype("int32")

    flight_groups = df[GROUP_COLUMNS].astype(str).agg(" | ".join, axis=1)

    drop_cols = [c for c in ["travelCode", "userCode", "date"] if c in model_df.columns]
    model_df = model_df.drop(columns=drop_cols)

    X = model_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].copy()
    y = model_df["price"].copy()
    return X, y, flight_groups


def evaluate_regression(y_true, y_pred) -> dict:
    """Compute MAE, MSE, RMSE, and R2 regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ============================================================
# Main training workflow
# ============================================================


def main():
    print("=" * 70)
    print("Voyage Analytics - Flight Price Training")
    print("=" * 70)

    tracking_uri = resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment  : {MLFLOW_EXPERIMENT}")

    # --------------------------------------------------------
    # Validate and load input dataset
    # --------------------------------------------------------

    print(f"Dataset path: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Flights dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Flights dataset is empty.")

    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    required_columns = [
        "from",
        "to",
        "flightType",
        "time",
        "distance",
        "agency",
        "date",
        "price",
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = int(df[required_columns].isnull().sum().sum())
    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values in required columns."
        )

    for num_col in ["price", "time", "distance"]:
        if (df[num_col] <= 0).any():
            raise ValueError(f"Column '{num_col}' contains non-positive values.")

    duplicate_rows = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_rows}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    X, y, flight_groups = prepare_features(df)

    print(f"Features ({X.shape[1]}): {X.columns.tolist()}")
    print(f"Target: {y.name} | Total unique flight configurations: {flight_groups.nunique()}")

    # --------------------------------------------------------
    # Leakage-free GroupShuffleSplit evaluation (consistent with Notebook 01)
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=flight_groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(flight_groups.iloc[train_idx])
    test_groups = set(flight_groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} overlapping flight groups.")

    print("\nGroup-Based Split (Unseen Flight Configurations):")
    print(f"Training samples: {len(X_train):,} ({len(train_groups)} unique configurations)")
    print(f"Testing samples : {len(X_test):,} ({len(test_groups)} unique configurations)")
    print(f"Group overlap   : {len(overlap)}")

    # --------------------------------------------------------
    # Train & evaluate on unseen configurations, then refit final production model
    # --------------------------------------------------------

    eval_pipeline = build_pipeline()
    final_pipeline = build_pipeline()

    print("\nEvaluating LightGBM on unseen flight configurations...")
    with mlflow.start_run(run_name="Flight_Price_LightGBM_Production"):
        eval_pipeline.fit(X_train, y_train)
        y_pred = eval_pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)

        print("\nModel Performance (Unseen Configurations)")
        print("-" * 45)
        print(f"MAE  : {metrics['MAE']:.2f}")
        print(f"MSE  : {metrics['MSE']:.2f}")
        print(f"RMSE : {metrics['RMSE']:.2f}")
        print(f"R2   : {metrics['R2']:.4f}")

        # Refit final production pipeline on complete dataset after holdout evaluation
        print("\nRefitting final production LightGBM pipeline on full dataset...")
        final_pipeline.fit(X, y)

        mlflow.log_params(
            {
                "model": "LightGBM",
                "n_estimators": LGBM_PARAMS["n_estimators"],
                "learning_rate": LGBM_PARAMS["learning_rate"],
                "max_depth": LGBM_PARAMS["max_depth"],
                "num_leaves": LGBM_PARAMS["num_leaves"],
                "subsample": LGBM_PARAMS["subsample"],
                "colsample_bytree": LGBM_PARAMS["colsample_bytree"],
                "random_state": RANDOM_STATE,
                "validation_strategy": "GroupShuffleSplit (unseen flight configurations)",
                "test_size": 0.20,
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "training_groups": len(train_groups),
                "testing_groups": len(test_groups),
                "final_production_samples": len(X),
            }
        )

        mlflow.log_metrics(metrics)

        input_example = X_test.head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="flight_price_pipeline",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
            ],
        )

        if model_info.registered_model_version is not None:
            client = MlflowClient()
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias=MODEL_ALIAS,
                version=str(model_info.registered_model_version),
            )
            print(
                f"\nRegistered '{MODEL_NAME}' version {model_info.registered_model_version} "
                f"with alias '@{MODEL_ALIAS}'."
            )

        # Save local production artifact
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, MODEL_PATH)
        print(f"Local model saved to: {MODEL_PATH}")

    # --------------------------------------------------------
    # Reload verification
    # --------------------------------------------------------

    loaded_pipeline = joblib.load(MODEL_PATH)
    sample_predictions = loaded_pipeline.predict(X_test.head(5))
    print("\nSaved local model reload verification (first 5 test rows):")
    print(np.round(sample_predictions, 2))

    print("\nTraining workflow completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
