"""
TRAVELIQ — End-to-End Apache Airflow MLOps Pipeline DAG.

DAG ID: traveliq_ml_pipeline

Workflow Graph:
    validate_datasets
          │
          ▼
    preprocess_and_retrain_gate  (ShortCircuitOperator: skips if data unchanged)
          │
          ├──► train_flight_model
          ├──► train_gender_model
          └──► build_recommendation_catalog
                      │
                      ▼
               evaluate_models   (Quality gates block unvalidated models)
                      │
                      ▼
        register_and_promote_champion (MLflow Registry + @champion alias)
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Ensure project root is importable inside the Airflow container
PROJECT_ROOT = Path(os.getenv("TRAVELIQ_PROJECT_ROOT", "/opt/airflow/project"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlops_pipeline import (  # noqa: E402
    build_recommendation_catalog,
    evaluate_models,
    preprocess_and_check_retrain,
    register_and_promote_champion,
    train_flight_candidate,
    train_gender_candidate,
    validate_datasets,
)

logger = logging.getLogger("traveliq.airflow_dag")


def log_task_failure(context):
    """Structured failure callback for Airflow operational alerting."""
    ti = context.get("task_instance")
    dag_id = context.get("dag").dag_id if context.get("dag") else "traveliq_ml_pipeline"
    task_id = ti.task_id if ti else "unknown"
    exc = context.get("exception")
    logger.error(
        "TRAVELIQ Airflow Alert | DAG=%s | Task=%s FAILED | Error=%s",
        dag_id,
        task_id,
        exc,
    )


default_args = {
    "owner": "traveliq-mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=20),
    "on_failure_callback": log_task_failure,
}

# Default to manual trigger (None) or set TRAVELIQ_AIRFLOW_SCHEDULE="@weekly"
DAG_SCHEDULE = os.getenv("TRAVELIQ_AIRFLOW_SCHEDULE") or None

with DAG(
    dag_id="traveliq_ml_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=DAG_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["traveliq", "mlops", "flight-price", "gender", "recommendation"],
    description=(
        "TRAVELIQ end-to-end ML workflow: dataset validation, retrain gating, "
        "parallel flight/gender/recommender training, quality-gate evaluation, "
        "and gated MLflow @champion promotion."
    ),
) as dag:

    t_validate = PythonOperator(
        task_id="validate_datasets",
        python_callable=validate_datasets,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    t_preprocess_gate = ShortCircuitOperator(
        task_id="preprocess_and_retrain_gate",
        python_callable=preprocess_and_check_retrain,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    t_train_flight = PythonOperator(
        task_id="train_flight_model",
        python_callable=train_flight_candidate,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    t_train_gender = PythonOperator(
        task_id="train_gender_model",
        python_callable=train_gender_candidate,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    t_build_recommender = PythonOperator(
        task_id="build_recommendation_catalog",
        python_callable=build_recommendation_catalog,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    t_register_promote = PythonOperator(
        task_id="register_and_promote_champion",
        python_callable=register_and_promote_champion,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    (
        t_validate
        >> t_preprocess_gate
        >> [t_train_flight, t_train_gender, t_build_recommender]
        >> t_evaluate
        >> t_register_promote
    )
