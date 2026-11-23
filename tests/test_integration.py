"""
TRAVELIQ — End-to-End Integration Tests.

Tests real cross-component behavior:
  - MLflow @champion model parity with local fallback artifact
  - Live HTTP client <-> Flask WSGI Flight & Gender API round-trips
  - Recommender consistency between Notebook 03 & production app/recommender.py
  - Full Airflow MLOps pipeline PASSING promotion workflow (isolated sandbox)
  - Cross-layer configuration alignment (app/config, Docker Compose, Kubernetes, Jenkins)
"""

import json
import shutil
import threading
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import pytest
import yaml
from werkzeug.serving import make_server

from app.api_client import (
    check_api_health,
    request_flight_prediction,
    request_gender_prediction,
)
from app.app import FlightModelRegistry, create_app as create_flight_app
from app.config import (
    FLIGHT_LOCAL_MODEL_PATH,
    GENDER_MODEL_PATH,
    MODEL_ALIAS,
    MODEL_NAME,
    PROJECT_ROOT,
)
from app.gender_api import GenderModelRegistry, create_app as create_gender_app
from app.recommender import TravelPackageRecommender


class _LiveWSGIThread(threading.Thread):
    """Lightweight ephemeral HTTP server thread for real HTTP integration testing."""

    def __init__(self, wsgi_app):
        super().__init__(daemon=True)
        self.server = make_server("127.0.0.1", 0, wsgi_app)
        self.port = self.server.server_port

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()


@pytest.fixture(scope="module")
def live_services():
    flight_reg = FlightModelRegistry(
        tracking_uri="",
        local_model_path=FLIGHT_LOCAL_MODEL_PATH,
        enable_fallback=True,
    )
    gender_reg = GenderModelRegistry(model_path=GENDER_MODEL_PATH)

    flight_srv = _LiveWSGIThread(create_flight_app(flight_reg))
    gender_srv = _LiveWSGIThread(create_gender_app(gender_reg))
    flight_srv.start()
    gender_srv.start()

    yield {
        "flight_url": f"http://127.0.0.1:{flight_srv.port}",
        "gender_url": f"http://127.0.0.1:{gender_srv.port}",
    }

    flight_srv.stop()
    gender_srv.stop()


def test_live_http_api_client_to_flight_and_gender_services(live_services):
    """Real HTTP round-trip via app.api_client against running Flask servers."""
    f_health = check_api_health(live_services["flight_url"])
    g_health = check_api_health(live_services["gender_url"])
    assert f_health["online"] is True
    assert g_health["online"] is True

    flight_res, flight_err = request_flight_prediction(
        {
            "from": "Florianopolis (SC)",
            "to": "Aracaju (SE)",
            "flightType": "firstClass",
            "agency": "CloudFy",
            "time": 1.76,
            "distance": 676.53,
            "date": "2026-09-20",
        },
        base_url=live_services["flight_url"],
    )
    assert flight_err is None
    assert flight_res["currency"] == "USD"
    assert 200.0 <= flight_res["predicted_price"] <= 3000.0

    gender_res, gender_err = request_gender_prediction(
        {"name": "Wilma Mcinnis", "company": "4You", "age": 48},
        base_url=live_services["gender_url"],
    )
    assert gender_err is None
    assert gender_res["predicted_gender"] == "female"
    assert gender_res["experimental"] is True
    assert gender_res["confidence"] >= 0.80


def test_mlflow_champion_and_local_fallback_prediction_parity():
    """MLflow @champion and local joblib fallback must produce identical fares."""
    mlflow_db = PROJECT_ROOT / "mlflow.db"
    if not mlflow_db.exists():
        pytest.skip("Local mlflow.db not present in environment")

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.resolve().as_posix()}")
    mlflow_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    local_model = joblib.load(FLIGHT_LOCAL_MODEL_PATH)

    sample = pd.DataFrame(
        {
            "from": pd.Series(["Brasilia (DF)"], dtype="object"),
            "to": pd.Series(["Salvador (BH)"], dtype="object"),
            "flightType": pd.Series(["economic"], dtype="object"),
            "agency": pd.Series(["Rainbow"], dtype="object"),
            "time": pd.Series([1.26], dtype="float64"),
            "distance": pd.Series([486.52], dtype="float64"),
            "year": pd.Series([2026], dtype="int32"),
            "month": pd.Series([10], dtype="int32"),
            "day": pd.Series([15], dtype="int32"),
            "day_of_week": pd.Series([3], dtype="int32"),
        }
    )

    p_mlflow = float(mlflow_model.predict(sample)[0])
    p_local = float(local_model.predict(sample)[0])
    assert abs(p_mlflow - p_local) < 1e-4


def test_recommender_matches_notebook_03_reference_results():
    """Production TravelPackageRecommender matches Phase 1 Notebook 03 exactly."""
    rec = TravelPackageRecommender.from_csv_dir()

    r1, m1 = rec.recommend(days=3, max_budget=2500.0, user_code=0, top_n=5)
    assert m1["status"] == "ok"
    assert r1.iloc[0]["destination"] == "Sao Paulo (SP)"
    assert round(float(r1.iloc[0]["recommendation_score"]), 2) == 71.28

    r2, m2 = rec.recommend(
        days=2, max_budget=2000.0, origin="Sao Paulo (SP)", user_code=None, top_n=5
    )
    assert m2["status"] == "ok"
    assert r2.iloc[0]["destination"] == "Salvador (BH)"
    assert round(float(r2.iloc[0]["recommendation_score"]), 2) == 76.99


def test_full_airflow_pipeline_passing_promotion_in_isolated_sandbox(tmp_path, monkeypatch):
    """
    Verify gated MLflow registration & @champion promotion succeeds when ALL quality
    gates pass (using isolated sandbox so production models/ & mlflow.db are untouched).
    """
    from scripts import mlops_pipeline as mp

    sandbox_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", sandbox_uri)
    mlflow.set_tracking_uri(sandbox_uri)
    mlflow.set_registry_uri(sandbox_uri)

    sandbox_data = tmp_path / "travel_capstone_dataset"
    sandbox_data.mkdir()
    for csv_name in ("flights.csv", "hotels.csv", "users.csv"):
        shutil.copy2(PROJECT_ROOT / "travel_capstone_dataset" / csv_name, sandbox_data / csv_name)

    assert mp.validate_datasets(project_root=tmp_path)["users_labeled_gender"] == 900
    assert mp.preprocess_and_check_retrain(project_root=tmp_path, force_retrain=False) is True

    # Train real gender & recommender candidates; stage valid flight candidate into sandbox MLflow
    gender_rep = mp.train_gender_candidate(project_root=tmp_path)
    assert gender_rep["metrics"]["f1_weighted"] >= 0.85
    assert gender_rep["metrics"]["unseen_first_name_f1"] >= 0.58

    rec_rep = mp.build_recommendation_catalog(project_root=tmp_path)
    assert rec_rep["budget_compliance"] == 1.0

    # Log the verified Phase 1 flight pipeline into sandbox MLflow to test promotion fast
    mlflow.set_experiment("TRAVELIQ - Flight Price Prediction")
    flight_pipe = joblib.load(FLIGHT_LOCAL_MODEL_PATH)
    cand_flight_path = tmp_path / "models" / "candidates" / "flight_price_pipeline.joblib"
    shutil.copy2(FLIGHT_LOCAL_MODEL_PATH, cand_flight_path)

    sample_input_df = pd.DataFrame(
        {
            "from": pd.Series(["Florianopolis (SC)"], dtype="object"),
            "to": pd.Series(["Aracaju (SE)"], dtype="object"),
            "flightType": pd.Series(["firstClass"], dtype="object"),
            "agency": pd.Series(["CloudFy"], dtype="object"),
            "time": pd.Series([1.76], dtype="float64"),
            "distance": pd.Series([676.53], dtype="float64"),
            "year": pd.Series([2026], dtype="int32"),
            "month": pd.Series([9], dtype="int32"),
            "day": pd.Series([20], dtype="int32"),
            "day_of_week": pd.Series([6], dtype="int32"),
        }
    )
    with mlflow.start_run(run_name="Sandbox_Flight_Candidate") as run:
        info = mlflow.sklearn.log_model(
            sk_model=flight_pipe,
            name="flight_price_pipeline_candidate",
            input_example=sample_input_df,
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
            ],
        )
        flight_rep = {
            "artifact_path": str(cand_flight_path),
            "mlflow_run_id": run.info.run_id,
            "mlflow_model_uri": info.model_uri,
            "metrics": {"MAE": 41.73, "MSE": 2754.14, "RMSE": 52.48, "R2": 0.9765},
            "sample_input": sample_input_df.to_dict(orient="records"),
        }
    (tmp_path / "models" / "candidates" / "flight_candidate_report.json").write_text(
        json.dumps(flight_rep, indent=2), encoding="utf-8"
    )

    eval_res = mp.evaluate_models(project_root=tmp_path)
    assert eval_res["passed"] is True

    state = mp.register_and_promote_champion(project_root=tmp_path)
    assert state["status"] == "promoted"
    assert state["flight_model"]["alias"] == "champion"
    assert state["gender_model"]["alias"] == "champion"
    assert (tmp_path / "models" / "flight_price_pipeline.joblib").exists()
    assert (tmp_path / "models" / "gender_classifier.joblib").exists()

    # Subsequent run with unchanged datasets must skip unnecessary retraining
    assert mp.preprocess_and_check_retrain(project_root=tmp_path, force_retrain=False) is False


def test_cross_layer_ports_and_model_names_match():
    """Verify Docker Compose, Kubernetes ConfigMap, and Jenkinsfile stay synchronized."""
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    k8s_cm = yaml.safe_load(
        (PROJECT_ROOT / "kubernetes" / "configmap.yaml").read_text(encoding="utf-8")
    )["data"]
    jf_text = (PROJECT_ROOT / "Jenkinsfile").read_text(encoding="utf-8")

    assert k8s_cm["MODEL_NAME"] == MODEL_NAME
    assert k8s_cm["MODEL_ALIAS"] == MODEL_ALIAS
    assert k8s_cm["FLASK_PORT"] == "5001"
    assert k8s_cm["GENDER_API_PORT"] == "5002"
    assert compose["services"]["flight-api"]["environment"]["FLASK_PORT"] == "5001"
    assert compose["services"]["gender-api"]["environment"]["GENDER_API_PORT"] == "5002"
    assert "traveliq-flight-api" in jf_text
    assert "traveliq-gender-api" in jf_text
    assert "traveliq-streamlit" in jf_text
    assert "traveliq-mlflow" in jf_text
