// ==============================================================================
// TRAVELIQ — Travel Intelligence Platform CI/CD Pipeline (Declarative Jenkinsfile)
//
// Stages:
//   1. Checkout
//   2. Python Environment & Dependency Validation
//   3. Code Quality Checks
//   4. Automated Pytest Suite
//   5. Notebook & Model Artifact Validation
//   6. Docker Image Build
//   7. Deployment Validation (Non-Production Dry-Run Only)
//
// Safety Guarantees:
//   - Fails immediately if dependency checks, code quality, pytest, or model checks fail.
//   - Contains zero hardcoded credentials (uses Jenkins credentials only if push enabled).
//   - NEVER deploys automatically to production.
// ==============================================================================

def runCmd(String command) {
    if (isUnix()) {
        sh command
    } else {
        bat command
    }
}

def venvPython() {
    return isUnix() ? '.ci-venv/bin/python' : '.ci-venv\\Scripts\\python.exe'
}

pipeline {
    agent any

    parameters {
        string(
            name: 'IMAGE_TAG',
            defaultValue: '2.0',
            description: 'Docker image tag aligned with Kubernetes manifests'
        )
        booleanParam(
            name: 'RUN_DOCKER_BUILD',
            defaultValue: true,
            description: 'Build TRAVELIQ Docker images when Docker daemon is available on the agent'
        )
    }

    environment {
        PYTHONDONTWRITEBYTECODE = '1'
        PYTHONUNBUFFERED        = '1'
        FLIGHT_IMAGE            = "traveliq-flight-api:${params.IMAGE_TAG}"
        GENDER_IMAGE            = "traveliq-gender-api:${params.IMAGE_TAG}"
        STREAMLIT_IMAGE         = "traveliq-streamlit:${params.IMAGE_TAG}"
        MLFLOW_IMAGE            = "traveliq-mlflow:${params.IMAGE_TAG}"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {

        stage('1. Checkout') {
            steps {
                echo 'Checking out TRAVELIQ source repository...'
                checkout scm
                script {
                    runCmd('git rev-parse --short HEAD || echo "Non-git checkout workspace"')
                }
            }
        }

        stage('2. Python Environment & Dependency Validation') {
            steps {
                echo 'Creating isolated CI virtual environment and validating pinned dependencies...'
                script {
                    if (isUnix()) {
                        sh '''
                            python3 -m venv .ci-venv
                            .ci-venv/bin/python -m pip install --upgrade pip
                            .ci-venv/bin/python -m pip install -r requirements-dev.txt
                            .ci-venv/bin/python -m pip check
                        '''
                    } else {
                        bat '''
                            python -m venv .ci-venv
                            .ci-venv\\Scripts\\python.exe -m pip install --upgrade pip
                            .ci-venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt
                            .ci-venv\\Scripts\\python.exe -m pip check
                        '''
                    }
                }
            }
        }

        stage('3. Code Quality Checks') {
            steps {
                echo 'Running Python AST syntax verification and YAML manifest validation...'
                script {
                    def py = venvPython()
                    runCmd("${py} -B -c \"import ast, pathlib, yaml; " +
                        "py_files = [p for d in ['app','scripts','tests','airflow'] for p in pathlib.Path(d).rglob('*.py')] + [pathlib.Path('streamlit_app.py')]; " +
                        "[ast.parse(p.read_text(encoding='utf-8')) for p in py_files]; " +
                        "yaml_files = list(pathlib.Path('kubernetes').glob('*.yaml')) + [pathlib.Path('docker-compose.yml'), pathlib.Path('airflow/docker-compose.yml')]; " +
                        "[list(yaml.safe_load_all(p.read_text(encoding='utf-8'))) for p in yaml_files]; " +
                        "print(f'Code Quality OK: {len(py_files)} Python files, {len(yaml_files)} YAML files validated.')\"")
                }
            }
        }

        stage('4. Automated Pytest Suite') {
            steps {
                echo 'Running automated pytest test suite (pipeline fails if any test fails)...'
                script {
                    def py = venvPython()
                    if (isUnix()) {
                        sh 'mkdir -p reports'
                    } else {
                        bat 'if not exist reports mkdir reports'
                    }
                    runCmd("${py} -B -m pytest -v --junitxml=reports/junit.xml")
                }
            }
        }

        stage('5. Notebook & Model Artifact Validation') {
            steps {
                echo 'Validating Phase 1 notebooks (zero saved errors) and serialized model artifacts...'
                script {
                    def py = venvPython()
                    runCmd("${py} -B -c \"import json, pathlib, joblib, pandas as pd; " +
                        "nbs = sorted(pathlib.Path('notebooks').glob('*.ipynb')); " +
                        "assert len(nbs) == 3, f'Expected 3 notebooks, found {len(nbs)}'; " +
                        "errs = [(n.name, i) for n in nbs for i, c in enumerate(json.loads(n.read_text(encoding='utf-8'))['cells']) for o in c.get('outputs', []) if o.get('output_type') == 'error']; " +
                        "assert not errs, f'Notebook execution errors found: {errs}'; " +
                        "fp = pathlib.Path('models/flight_price_pipeline.joblib'); " +
                        "gp = pathlib.Path('models/gender_classifier.joblib'); " +
                        "assert fp.exists() and gp.exists(), 'Missing required model artifacts'; " +
                        "assert gp.stat().st_size < 512000, f'Gender artifact too large: {gp.stat().st_size} bytes'; " +
                        "joblib.load(fp); gb = joblib.load(gp); assert 'pipeline' in gb; " +
                        "print('Notebooks & Model Artifacts Validated OK.')\"")
                }
            }
        }

        stage('6. Docker Image Build') {
            when {
                expression { return params.RUN_DOCKER_BUILD }
            }
            steps {
                echo 'Building TRAVELIQ Docker images (Flight API, Gender API, Streamlit, MLflow)...'
                script {
                    runCmd("docker build -f docker/Dockerfile -t ${env.FLIGHT_IMAGE} .")
                    runCmd("docker build -f gender/Dockerfile -t ${env.GENDER_IMAGE} .")
                    runCmd("docker build -f docker/Dockerfile.streamlit -t ${env.STREAMLIT_IMAGE} .")
                    runCmd("docker build -f docker/Dockerfile.mlflow -t ${env.MLFLOW_IMAGE} .")
                    runCmd("docker run --rm ${env.FLIGHT_IMAGE} python -c \"import flask, mlflow, lightgbm, sklearn, gunicorn; print('Flight image OK')\"")
                    runCmd("docker run --rm ${env.GENDER_IMAGE} python -c \"import flask, sklearn, joblib, gunicorn; print('Gender image OK')\"")
                    runCmd("docker run --rm ${env.STREAMLIT_IMAGE} python -c \"import streamlit, plotly, requests, sklearn; print('Streamlit image OK')\"")
                }
            }
        }

        stage('7. Deployment Validation') {
            steps {
                echo 'Validating Docker Compose & Kubernetes manifests (dry-run only; no automatic production deployment)...'
                script {
                    runCmd('docker compose config --quiet')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/configmap.yaml')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/mlflow.yaml')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/deployment.yaml')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/service.yaml')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/gender-classifier.yaml')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/streamlit.yaml')
                    runCmd('kubectl apply --dry-run=client -f kubernetes/hpa.yaml')
                }
                echo 'Non-production deployment validation passed. Production rollout requires explicit manual approval outside CI.'
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit.xml'
            echo 'Cleaning up temporary CI virtual environment...'
            script {
                if (isUnix()) {
                    sh 'rm -rf .ci-venv'
                } else {
                    bat 'if exist .ci-venv rmdir /s /q .ci-venv'
                }
            }
        }
        success {
            echo 'TRAVELIQ CI/CD pipeline passed all quality, test, artifact, and dry-run deployment checks.'
        }
        failure {
            echo 'TRAVELIQ CI/CD pipeline FAILED. Review stage logs and JUnit test results.'
        }
    }
}
