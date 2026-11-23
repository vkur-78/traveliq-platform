"""
Phase 3 MLOps Infrastructure & Pipeline Validation Tests.

Covers:
  - Dockerfiles & docker-compose.yml static consistency and hygiene
  - Kubernetes manifests (YAML schema, selector/port/image/HPA/ConfigMap cross-refs)
  - Apache Airflow DAG parsing (via lightweight Airflow mock) & dependency verification
  - Live execution of scripts/mlops_pipeline.py stages in an isolated sandbox
    (including quality-gate blocking and unnecessary-retraining skip gate)
  - Jenkinsfile structural & stage verification
"""

import ast
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import joblib
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# 1. DOCKER & DOCKER COMPOSE VALIDATION
# ==============================================================================


def test_docker_compose_valid_and_service_networking_consistent():
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists()
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = data["services"]
    assert set(services.keys()) == {"mlflow", "flight-api", "gender-api", "streamlit-ui"}

    assert services["mlflow"]["image"] == "traveliq-mlflow:2.0"
    assert services["flight-api"]["image"] == "traveliq-flight-api:2.0"
    assert services["gender-api"]["image"] == "traveliq-gender-api:2.0"
    assert services["streamlit-ui"]["image"] == "traveliq-streamlit:2.0"

    # Verify service-name networking (no localhost between containers)
    assert services["flight-api"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert services["streamlit-ui"]["environment"]["FLIGHT_API_URL"] == "http://flight-api:5001"
    assert services["streamlit-ui"]["environment"]["GENDER_API_URL"] == "http://gender-api:5002"

    # Every service has a healthcheck and sits on traveliq-net
    for svc_name, svc in services.items():
        assert "healthcheck" in svc, f"{svc_name} missing healthcheck"
        assert "traveliq-net" in svc["networks"], f"{svc_name} not on traveliq-net"


def test_dockerfiles_best_practices_and_copy_sources_exist():
    dockerfiles = [
        PROJECT_ROOT / "docker" / "Dockerfile",
        PROJECT_ROOT / "gender" / "Dockerfile",
        PROJECT_ROOT / "docker" / "Dockerfile.streamlit",
        PROJECT_ROOT / "docker" / "Dockerfile.mlflow",
    ]

    for df_path in dockerfiles:
        text = df_path.read_text(encoding="utf-8")
        active_lines = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        assert "HEALTHCHECK" in active_lines, f"{df_path} missing HEALTHCHECK"
        assert "USER " in active_lines, f"{df_path} does not switch to a non-root USER"
        assert "torch" not in active_lines.lower(), f"{df_path} still installs torch"
        assert "sentence-transformers" not in active_lines.lower(), (
            f"{df_path} still installs sentence-transformers"
        )

        # Verify every COPY source exists in the repository
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY "):
                parts = stripped.split()
                src = parts[1]
                assert (PROJECT_ROOT / src).exists(), f"{df_path} COPY source missing: {src}"

    flight_df = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "libgomp1" in flight_df
    assert "gunicorn" in flight_df

    gender_reqs_active = "\n".join(
        line
        for line in (PROJECT_ROOT / "gender" / "requirements-gender.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if not line.strip().startswith("#")
    ).lower()
    assert "torch" not in gender_reqs_active
    assert "sentence-transformers" not in gender_reqs_active


# ==============================================================================
# 2. KUBERNETES MANIFESTS & HPA VALIDATION
# ==============================================================================


def test_kubernetes_manifests_valid_and_cross_referenced():
    k8s_dir = PROJECT_ROOT / "kubernetes"
    required_files = [
        "configmap.yaml",
        "mlflow.yaml",
        "deployment.yaml",
        "service.yaml",
        "gender-classifier.yaml",
        "streamlit.yaml",
        "hpa.yaml",
        "secret.example.yaml",
    ]

    docs = []
    for fname in required_files:
        p = k8s_dir / fname
        assert p.exists(), f"Missing Kubernetes manifest: {fname}"
        loaded = [d for d in yaml.safe_load_all(p.read_text(encoding="utf-8")) if d is not None]
        assert loaded, f"Empty YAML file: {fname}"
        docs.extend(loaded)

    deployments = {d["metadata"]["name"]: d for d in docs if d["kind"] == "Deployment"}
    services = {d["metadata"]["name"]: d for d in docs if d["kind"] == "Service"}
    configmaps = {d["metadata"]["name"]: d for d in docs if d["kind"] == "ConfigMap"}
    hpas = {d["metadata"]["name"]: d for d in docs if d["kind"] == "HorizontalPodAutoscaler"}

    assert "traveliq-config" in configmaps
    assert set(deployments.keys()) == {
        "traveliq-mlflow",
        "traveliq-flight-api",
        "traveliq-gender-api",
        "traveliq-streamlit",
    }
    assert set(services.keys()) == set(deployments.keys())

    # Verify exact Docker Compose image tag consistency
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    expected_images = {
        "traveliq-mlflow": compose["services"]["mlflow"]["image"],
        "traveliq-flight-api": compose["services"]["flight-api"]["image"],
        "traveliq-gender-api": compose["services"]["gender-api"]["image"],
        "traveliq-streamlit": compose["services"]["streamlit-ui"]["image"],
    }

    for dep_name, dep in deployments.items():
        container = dep["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == expected_images[dep_name]
        assert "readinessProbe" in container
        assert "livenessProbe" in container
        assert "requests" in container["resources"]
        assert "limits" in container["resources"]

        # Verify Service targets match Deployment labels and container port
        svc = services[dep_name]
        assert svc["spec"]["selector"] == dep["spec"]["selector"]["matchLabels"]
        assert svc["spec"]["ports"][0]["targetPort"] == container["ports"][0]["containerPort"]

    # Verify HPA targets exist and have valid min/max bounds
    assert "traveliq-flight-api-hpa" in hpas
    assert "traveliq-gender-api-hpa" in hpas
    for hpa_name, hpa in hpas.items():
        assert hpa["apiVersion"] == "autoscaling/v2"
        target_name = hpa["spec"]["scaleTargetRef"]["name"]
        assert target_name in deployments, f"HPA {hpa_name} targets unknown deployment {target_name}"
        assert hpa["spec"]["minReplicas"] < hpa["spec"]["maxReplicas"]


# ==============================================================================
# 3. APACHE AIRFLOW DAG PARSING & PIPELINE EXECUTION TESTS
# ==============================================================================


def test_airflow_compose_has_healthchecks_and_no_legacy_personal_info():
    af_compose = (PROJECT_ROOT / "airflow" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "vinay" not in af_compose.lower()
    assert "pandey" not in af_compose.lower()
    data = yaml.safe_load(af_compose)
    assert "healthcheck" in data["services"]["postgres"]
    assert data["services"]["airflow-scheduler"]["depends_on"]["airflow-init"]["condition"] == "service_completed_successfully"


def test_airflow_dag_parses_and_defines_expected_task_graph():
    dag_file = PROJECT_ROOT / "airflow" / "dags" / "traveliq_ml_pipeline.py"
    assert dag_file.exists()
    source = dag_file.read_text(encoding="utf-8")
    ast.parse(source)

    # Import DAG with lightweight Airflow stub to verify actual task IDs & edge wiring
    created_tasks = {}
    edges = []

    class FakeOperator:
        def __init__(self, task_id, python_callable, op_kwargs=None, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
            self.op_kwargs = op_kwargs or {}
            created_tasks[task_id] = self

        def __rshift__(self, other):
            targets = other if isinstance(other, list) else [other]
            for t in targets:
                edges.append((self.task_id, t.task_id))
            return other

        def __rrshift__(self, other):
            sources = other if isinstance(other, list) else [other]
            for s in sources:
                edges.append((s.task_id, self.task_id))
            return self

    class FakeDAG:
        def __init__(self, dag_id, default_args=None, **kwargs):
            self.dag_id = dag_id
            self.default_args = default_args or {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_airflow = types.ModuleType("airflow")
    fake_airflow.DAG = FakeDAG
    fake_ops = types.ModuleType("airflow.operators")
    fake_py_ops = types.ModuleType("airflow.operators.python")
    fake_py_ops.PythonOperator = FakeOperator
    fake_py_ops.ShortCircuitOperator = FakeOperator

    with patch.dict(
        sys.modules,
        {
            "airflow": fake_airflow,
            "airflow.operators": fake_ops,
            "airflow.operators.python": fake_py_ops,
        },
    ):
        ns = {}
        exec(compile(source, str(dag_file), "exec"), ns)

    expected_tasks = {
        "validate_datasets",
        "preprocess_and_retrain_gate",
        "train_flight_model",
        "train_gender_model",
        "build_recommendation_catalog",
        "evaluate_models",
        "register_and_promote_champion",
    }
    assert set(created_tasks.keys()) == expected_tasks
    assert ns["dag"].dag_id == "traveliq_ml_pipeline"
    assert ns["dag"].default_args["retries"] >= 1
    assert callable(ns["dag"].default_args["on_failure_callback"])
    assert ("validate_datasets", "preprocess_and_retrain_gate") in edges
    assert ("preprocess_and_retrain_gate", "train_flight_model") in edges
    assert ("preprocess_and_retrain_gate", "train_gender_model") in edges
    assert ("preprocess_and_retrain_gate", "build_recommendation_catalog") in edges
    assert ("train_flight_model", "evaluate_models") in edges
    assert ("train_gender_model", "evaluate_models") in edges
    assert ("build_recommendation_catalog", "evaluate_models") in edges
    assert ("evaluate_models", "register_and_promote_champion") in edges


def test_mlops_pipeline_validation_retrain_gate_and_quality_gate_blocking(tmp_path, monkeypatch):
    """
    Execute real scripts/mlops_pipeline.py stages in an isolated tmp_path sandbox
    so Phase 1 production models/ are never altered while verifying real behavior.
    """
    import mlflow
    from scripts import mlops_pipeline as mp

    sandbox_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", sandbox_uri)
    mlflow.set_tracking_uri(sandbox_uri)
    mlflow.set_registry_uri(sandbox_uri)

    # 1. Live dataset validation on actual capstone datasets
    val_summary = mp.validate_datasets(project_root=PROJECT_ROOT)
    assert val_summary["datasets"]["flights.csv"]["rows"] == 271888
    assert val_summary["datasets"]["hotels.csv"]["rows"] == 40552
    assert val_summary["datasets"]["users.csv"]["rows"] == 1340
    assert val_summary["users_labeled_gender"] == 900
    assert val_summary["users_undisclosed_none"] == 440

    # 2. Prepare isolated sandbox with dataset copies (cross-platform, no admin symlink required)
    import shutil

    sandbox_data = tmp_path / "travel_capstone_dataset"
    sandbox_data.mkdir()
    for csv_name in ("flights.csv", "hotels.csv", "users.csv"):
        shutil.copy2(PROJECT_ROOT / "travel_capstone_dataset" / csv_name, sandbox_data / csv_name)

    sandbox_models = tmp_path / "models"
    sandbox_models.mkdir()

    # Retrain gate should return True on fresh sandbox (no promoted artifacts yet)
    assert mp.preprocess_and_check_retrain(project_root=tmp_path, force_retrain=False) is True

    # Run real gender training & real recommendation catalog generation in sandbox
    gender_report = mp.train_gender_candidate(project_root=tmp_path)
    assert gender_report["metrics"]["f1_weighted"] >= 0.85
    assert gender_report["artifact_size_kb"] < 200.0

    rec_report = mp.build_recommendation_catalog(project_root=tmp_path)
    assert rec_report["package_count"] == 1568
    assert rec_report["budget_compliance"] == 1.0

    # 3. Verify quality gate BLOCKS promotion when flight metrics fail threshold
    bad_flight_report = {
        "artifact_path": str(PROJECT_ROOT / "models" / "flight_price_pipeline.joblib"),
        "mlflow_run_id": "fake-bad-run",
        "mlflow_model_uri": "models:/fake",
        "metrics": {"MAE": 250.0, "MSE": 90000.0, "RMSE": 300.0, "R2": 0.40},
        "sample_input": [
            {
                "from": "Florianopolis (SC)",
                "to": "Aracaju (SE)",
                "flightType": "firstClass",
                "agency": "CloudFy",
                "time": 1.76,
                "distance": 676.53,
                "year": 2026,
                "month": 9,
                "day": 20,
                "day_of_week": 6,
            }
        ],
    }
    (sandbox_models / "candidates" / "flight_candidate_report.json").write_text(
        json.dumps(bad_flight_report), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="quality gates FAILED"):
        mp.evaluate_models(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="unvalidated"):
        mp.register_and_promote_champion(project_root=tmp_path)

    # 4. Verify retrain gate returns False (skips unnecessary retraining) once promoted with same fingerprint
    joblib.dump({"ok": True}, sandbox_models / "flight_price_pipeline.joblib")
    joblib.dump({"ok": True}, sandbox_models / "gender_classifier.joblib")
    (sandbox_models / "pipeline_state.json").write_text(
        json.dumps({"status": "promoted", "dataset_fingerprint": val_summary["dataset_fingerprint"]}),
        encoding="utf-8",
    )
    assert mp.preprocess_and_check_retrain(project_root=tmp_path, force_retrain=False) is False
    assert mp.preprocess_and_check_retrain(project_root=tmp_path, force_retrain=True) is True


# ==============================================================================
# 4. JENKINSFILE STRUCTURAL VALIDATION
# ==============================================================================


def test_jenkinsfile_structure_stages_and_safety_rules():
    jf_text = (PROJECT_ROOT / "Jenkinsfile").read_text(encoding="utf-8")

    # Balanced braces
    assert jf_text.count("{") == jf_text.count("}"), "Unbalanced curly braces in Jenkinsfile"
    assert jf_text.count("(") == jf_text.count(")"), "Unbalanced parentheses in Jenkinsfile"

    # All 7 required stages present
    for expected_stage in [
        "1. Checkout",
        "2. Python Environment & Dependency Validation",
        "3. Code Quality Checks",
        "4. Automated Pytest Suite",
        "5. Notebook & Model Artifact Validation",
        "6. Docker Image Build",
        "7. Deployment Validation",
    ]:
        assert f"stage('{expected_stage}')" in jf_text, f"Missing Jenkins stage: {expected_stage}"

    # Safety rules: dry-run only (no production auto-deploy), no hardcoded credentials/author
    assert "--dry-run=client" in jf_text
    assert "pytest" in jf_text
    assert "vinay" not in jf_text.lower()
    assert "password=" not in jf_text.lower()


# ==============================================================================
# 5. PUBLIC CLOUD DEPLOYMENT BUNDLE VALIDATION
# ==============================================================================


def test_cloud_deployment_bundle_security_and_completeness():
    cloud_dir = PROJECT_ROOT / "deploy" / "cloud"
    dockerfile = (cloud_dir / "Dockerfile").read_text(encoding="utf-8")
    startup = (cloud_dir / "start_traveliq.sh").read_text(encoding="utf-8")
    reqs = (cloud_dir / "requirements-cloud.txt").read_text(encoding="utf-8").lower()
    hf_readme = (cloud_dir / "README_HF_SPACE.md").read_text(encoding="utf-8")
    render_yaml = yaml.safe_load((PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8"))

    # Security: non-root UID 1000, internal-only APIs on 127.0.0.1, MLflow admin NOT exposed
    assert "useradd -m -u 1000" in dockerfile
    assert "USER user" in dockerfile
    assert "libgomp1" in dockerfile
    assert "MLFLOW_TRACKING_URI=disabled" in dockerfile
    assert "EXPOSE 7860" in dockerfile
    assert "HEALTHCHECK" in dockerfile

    # Internal loopback bindings for Flask APIs + health-wait before Streamlit
    assert "--bind 127.0.0.1:5001" in startup
    assert "--bind 127.0.0.1:5002" in startup
    assert "wait_for_health" in startup
    assert "streamlit run streamlit_app.py" in startup

    # Zero PyTorch / SentenceTransformers / MLflow bloat in public cloud container
    active_reqs = "\n".join(ln for ln in reqs.splitlines() if not ln.strip().startswith("#"))
    assert "torch" not in active_reqs
    assert "sentence-transformers" not in active_reqs
    assert "mlflow" not in active_reqs

    # Valid Hugging Face Spaces Docker frontmatter + Render fallback blueprint
    assert "sdk: docker" in hf_readme
    assert "app_port: 7860" in hf_readme
    assert render_yaml["services"][0]["dockerfilePath"] == "./deploy/cloud/Dockerfile"
