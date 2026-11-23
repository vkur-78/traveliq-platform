# Contributing to TRAVELIQ

Thank you for your interest in **TRAVELIQ — Travel Intelligence Platform**. This guide outlines the engineering, testing, security, and Responsible AI standards maintained across the repository.

---

## Development Setup

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Engineering & Pull Request Standards

1. **Preserve Leakage-Free Validation:**
   - Flight fare modeling must keep group-aware validation (`GroupShuffleSplit` / `GroupKFold`) across the 490 unique flight configurations.
   - Gender classification must exclude undisclosed `"none"` records from supervised training and fit all vectorizers inside a `scikit-learn` `Pipeline`.
2. **Keep Serving Artifacts Lightweight:**
   - Do not introduce PyTorch or `sentence-transformers` into `requirements.txt`, `gender/requirements-gender.txt`, or `deploy/cloud/`.
   - Preserve exact schema alignment (`int32` calendar columns) between training pipelines and `app/app.py`.
3. **Run the Full Automated Test Suite Before Opening a PR:**
   ```bash
   pytest -v
   ```
   All tests in `tests/` must pass without weakening assertions.
4. **Security & Credential Hygiene:**
   - Never commit `.env`, `.streamlit/secrets.toml`, `kubernetes/secret.yaml`, API keys, tokens, local `mlflow.db`, or machine-specific local file paths.
   - Do not expose MLflow administration interfaces on public internet endpoints without authentication.
5. **Responsible AI & Dataset Transparency:**
   - Preserve the explicit experimental disclaimer on the Gender Classification API and UI.
   - Preserve the transparent disclosure of the one-partner-hotel-per-city dataset limitation in the recommendation engine and UI.
   - Preserve starter-codebase and third-party open-source attribution in [ATTRIBUTION.md](ATTRIBUTION.md).
