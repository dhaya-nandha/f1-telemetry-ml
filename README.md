# 🏁 F1 Telemetry ML Project

Formula 1 lap time prediction using machine learning on real race data.

## What It Does

Predicts F1 lap times across 22 circuits given driver, tyre compound, fuel load, and tyre age.

## Current Status

**Week 2 Complete:**
- ✅ Data pipeline (Jolpica API + FastF1)
- ✅ Feature engineering (fuel, tyre degradation, baselines)
- ✅ XGBoost model trained (MAE: 0.314s)
- ✅ Interactive Streamlit dashboard

**Week 3 & 4:** Optimization, deployment, final polish

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

## Tech Stack

- **Data:** FastF1, Jolpica F1 API, Pandas
- **ML:** XGBoost, Scikit-learn
- **UI:** Streamlit
- **Deployment:** HuggingFace Spaces (coming Week 4)

## Results

- Model accuracy: MAE 0.314 seconds
- Training data: 23,366 laps from 22 circuits
- Prediction time: <50ms

---

More details coming after project completion.