---
title: TRAVELIQ — Travel Intelligence Platform
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: ML-powered flight pricing, hybrid trip packages & travel analytics
---

# TRAVELIQ — Travel Intelligence Platform

**TRAVELIQ** is an end-to-end Travel & Tourism Machine Learning and MLOps platform combining:

1. **Flight Price Predictor** — Leakage-free LightGBM regression (`GroupShuffleSplit` R² = `0.9765`, RMSE = `\$52.48`) served via an internal Gunicorn Flask REST API (`127.0.0.1:5001`).
2. **Trip & Hotel Package Recommendations** — Hybrid User-Based Collaborative Filtering + multi-factor engine ranking **1,568 real round-trip Flight + Hotel packages** with transparent budget/duration/affinity rationale and upfront disclosure of the one-hotel-per-city dataset constraint.
3. **Gender Classification Demo (Experimental)** — Lightweight **40.4 KB** pure `scikit-learn` binary classifier (`127.0.0.1:5002`) with calibrated probabilities and clear Responsible AI disclaimers.
4. **Interactive Market Analytics** — Five Plotly analytics dashboards spanning `271,888` flights, `40,552` hotel stays, and `1,340` traveler profiles.

### Production Cloud Security Architecture
- Only the **TRAVELIQ Streamlit frontend** is exposed over HTTPS (port `7860`).
- Both **Flask REST APIs** (`:5001` and `:5002`) bind strictly to loopback `127.0.0.1` inside the container.
- The **MLflow administration interface** is intentionally excluded from public internet exposure.
- Non-root container execution (`uid=1000`), zero hardcoded credentials, and zero PyTorch bloat.
