---
title: F1 WIM Pit Wall
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# 🏎️ F1-WIM: F1 "What-If" Machine
### Pit Wall Intelligence — Predictive Lap Time Engine
**🔴 Live Demo:** https://huggingface.co/spaces/NandDerek/f1-wim-pitwall

An end-to-end ML pipeline that predicts Formula 1 lap times across all 22 circuits of the 2023 season. Built with XGBoost, explained with SHAP, containerized with Docker.

**Model accuracy:** 0.327s MAE | 56% error reduction over naive baseline | 73.9% physics R²

---

## What It Does

Given a driver, circuit, tyre compound, fuel load, and tyre age — the model predicts the expected lap time and breaks down exactly how much time was lost or gained from each physical factor using SHAP explainability.

---

## Architecture
Jolpica F1 API + FastF1

↓

Data Cleaning (pit/SC lap removal)

↓

Feature Engineering

(fuel burn, tyre deg index, log tyre penalty)

↓

Target Encoding (split-then-encode, anti-leakage)

↓

XGBoost Regressor (Optuna tuned, 950 trees)

↓

SHAP TreeExplainer (waterfall plots per lap)

↓

Streamlit Dashboard → Docker Container

---

## Results

| Metric | Value |
|--------|-------|
| Naive baseline MAE (driver avg) | 0.751s |
| Optimized model MAE | 0.327s |
| Error reduction | 56% |
| Physics R² (delta only) | 73.9% |
| Training laps | 20,110 (across 22 circuits) |
| Circuits covered | All 22 from 2023 F1 season |

---

## Run with Docker (Recommended)

```bash
docker build -t f1-wim:latest .
docker run -p 8501:8501 f1-wim:latest
```

Open `http://localhost:8501`

> **Note:** Run via Docker only. Direct Windows execution not supported due to XGBoost/SHAP version constraints (see below).

---

## Version Pins — Do Not Upgrade

| Package | Version | Reason |
|---------|---------|--------|
| `xgboost` | `==1.7.6` | XGBoost 2.x serializes `base_score` as a bracketed array `[-0.00...]` which crashes SHAP's C-level memory parser |
| `shap` | `==0.45.0` | Stable integration with XGBoost 1.7.x tree structure |
| `numpy` | `<2.0` | NumPy 2.0 type deprecations break SHAP's internal color arrays |

These pins were established after a 4-layer dependency collapse during Docker containerization (XGBoost 2.x → SHAP C-buffer crash → NumPy 2.0 color array failure → WSL2 file-locking).

---

## Tech Stack

- **Data:** FastF1, Jolpica F1 API, Pandas, NumPy
- **ML:** XGBoost (Optuna hyperparameter tuning), Scikit-learn
- **Explainability:** SHAP (TreeExplainer, waterfall plots)
- **UI:** Streamlit (dark theme, custom F1 CSS)
- **DevOps:** Multi-stage Docker, WSL2

---

## Project Structure
f1-telemetry-ml/

├── src/

│   ├── app.py                          # Streamlit dashboard

│   ├── batch_pipeline.py               # Multi-circuit data ingestion

│   ├── circuit_encoding.py             # Target encoding pipeline

│   ├── cross_validation.py             # 5-fold leak-free CV

│   ├── data_clean.py                   # Lap filtering

│   ├── data_fetch.py                   # Jolpica API integration

│   ├── fastf1_fetch.py                 # FastF1 session loader

│   ├── feature_engineering.py          # Feature creation

│   ├── hyperparameter_tune.py          # Optuna tuning engine

│   ├── model_training.py               # Baseline XGBoost

│   ├── physics_refinement.py           # Track fuel burn maps

│   ├── shap_analysis.py                # SHAP export script

│   └── train_final_model.py            # Production retraining

├── models/

│   ├── xgb_lap_predictor.json          # Trained model weights

│   ├── circuit_means.npy               # Track baseline speeds

│   ├── driver_track_means.json         # Driver-circuit averages

│   ├── compound_track_means.json       # Tyre-circuit averages

│   └── track_to_id.json                # Circuit ID mapping

├── Dockerfile

├── requirements.txt

├── .gitignore

└── README.md

---

## Status

**Week 1** ✅ Data pipeline + baseline model
**Week 2** ✅ Multi-circuit scaling + Optuna tuning
**Week 3** ✅ SHAP explainability + Streamlit UI + Docker
**Week 4** 🔄 Deployment + portfolio integration