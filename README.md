# F1-WIM: Pit Wall Intelligence

**Real-Time F1 Lap Time Prediction & Explainability Engine**

An end-to-end machine learning pipeline and interactive dashboard that predicts Formula 1 lap-time evolution. The system utilizes race telemetry, tire degradation, fuel burn, and track-specific topologies to forecast absolute pace, while exposing the feature attributions of every prediction via live SHAP (SHapley Additive exPlanations) waterfall plots.

## 📌 Overview

Given a driver, circuit, tire compound, fuel load, and tire age, the XGBoost engine predicts the expected lap time. It calculates the physical tradeoffs (e.g., time gained from fuel burn versus time lost to rubber degradation) and visualizes those exact second-contributions in real-time.

The application also enforces strict physical constraints, intercepting physically impossible telemetry inputs (e.g., a Soft compound running for 40 laps) before they reach the inference model.

## 📊 Model Performance

| Metric | Value |
| --- | --- |
| **Naive Baseline MAE** (driver-track average) | `0.751s` |
| **Optimized Model MAE** | `0.327s` |
| **Error Reduction** (over baseline) | `56%` |
| **Physics $R^2$** (delta predictions only) | `73.9%` |
| **Total $R^2$** (includes circuit/driver variance) | `99.8%` |
| **Training Laps** (post-clean-air filter) | `20,110` |
| **Circuits Covered** | All `22` (2023 F1 Season) |
| **Validation Method** | 5-Fold Cross-Validation |

## 🏗️ System Architecture

```text
[ Jolpica F1 API + FastF1 Library ]
                ↓
[ Data Cleaning Layer ] 
  (Pit lap removal, safety car laps, FIA 107% filter)
                ↓
[ Feature Engineering Layer ]
  (Fuel burn estimation, degradation index, log-tire penalty)
                ↓
[ Target Encoding Layer ]
  (Split-then-encode, strict anti-leakage isolation)
                ↓
[ XGBoost Regressor ]
  (Optuna-tuned, 950 trees, MAE objective)
                ↓
[ SHAP TreeExplainer ]
  (Live C-level memory parsing for waterfall plots)
                ↓
[ Streamlit UI → Multi-Stage Docker → Hugging Face Spaces ]

```

## ⚙️ Core Features

* **Live Prediction Dashboard:** Select any of the 22 circuits from the 2023 season. Adjust driver, compound, lap number, tire age, and fuel load to see absolute pace predictions format (`MM:SS.sss`).
* **SHAP Explainability Engine:** Renders a live waterfall plot for every inference request, quantifying the exact impact of `FuelLoadKg`, `TyreLife`, and `TyreDegradationIndex`.
* **Out-of-Distribution Guardrails:**
* **Rule A:** Soft compound beyond 30 laps triggers a structural failure error.
* **Rule B:** Late-race lap combined with an impossible heavy fuel load throws a physics warning.
* **Rule C:** Tire age exceeding the current session lap number triggers an alert.



## 🛠️ Technology Stack

| Layer | Technology |
| --- | --- |
| **Data Ingestion** | FastF1, Jolpica F1 API |
| **Data Processing** | Pandas, NumPy |
| **Modeling** | XGBoost, Optuna (Hyperparameter Tuning) |
| **Explainability (XAI)** | SHAP (TreeExplainer) |
| **Validation** | Scikit-Learn (KFold, MAE, $R^2$) |
| **UI / Frontend** | Streamlit |
| **Infrastructure** | Docker (Multi-stage, `python:3.10-slim`), Git LFS |
| **Deployment** | Hugging Face Spaces |

## 📂 Project Structure

```text
f1-telemetry-ml/
├── src/
│   ├── app.py                    # Streamlit UI (entry point)
│   ├── batch_pipeline.py         # Multi-circuit data ingestion
│   ├── circuit_encoding.py       # Target encoding pipeline
│   ├── cross_validation.py       # 5-fold CV evaluation
│   ├── data_clean.py             # Lap filtering protocols
│   ├── data_fetch.py             # Jolpica API integration
│   ├── fastf1_fetch.py           # FastF1 session loader
│   ├── feature_engineering.py    # Physical feature creation
│   ├── hyperparameter_tune.py    # Optuna tuning engine
│   ├── physics_refinement.py     # Track-specific fuel burn maps
│   ├── shap_analysis.py          # SHAP export generator
│   └── train_final_model.py      # Production retraining script
├── models/
│   ├── xgb_lap_predictor.json    # Trained model weights
│   ├── circuit_means.npy         # Track baseline speeds
│   ├── driver_track_means.json   # Driver-circuit averages
│   ├── compound_track_means.json # Tire-circuit averages
│   └── track_to_id.json          # Circuit ID mapping
├── Dockerfile
├── requirements.txt
└── README.md

```

## 🚀 Running Locally

**Prerequisites:** Docker and Git installed.

Using Docker is strictly recommended to avoid operating system dependency conflicts and WSL2 file-locking behaviors.

```bash
git clone https://github.com/dhaya-nandha/f1-telemetry-ml.git
cd f1-telemetry-ml
docker build -t f1-wim:latest .
docker run -p 8501:8501 f1-wim:latest

```

Access the application at `http://localhost:8501`.

### ⚠️ Critical Version Pins

Do not upgrade the following packages in `requirements.txt`. These specific versions resolve known C-level serialization conflicts between XGBoost and SHAP.

* `xgboost==1.7.6`: XGBoost 2.x serializes `base_score` as a bracketed array (`[-0.00...]`), which breaks SHAP's memory parser.
* `shap==0.45.0`: Ensures stable integration with XGBoost 1.7.x tree structures.
* `numpy<2.0`: Prevents type deprecations in NumPy 2.0 from breaking SHAP's internal color array logic.

## 🔄 Retraining the Pipeline

To rebuild the models from scratch with updated data:

```bash
# 1. Ingest and Process Data
python src/batch_pipeline.py       
python src/circuit_encoding.py     
python src/physics_refinement.py   

# 2. Hyperparameter Tuning (Optional)
python src/hyperparameter_tune.py  

# 3. Train & Validate
python src/train_final_model.py    
python src/cross_validation.py     
python src/shap_analysis.py        

# 4. Rebuild Container
docker build -t f1-wim:latest .

```

## 🧠 Key Architectural Decisions

* **Target Encoding over One-Hot Encoding:** Representing 22 circuits as one-hot features introduces heavy sparsity. Target encoding replaces the circuit name with its mean lap time, providing a single, continuous feature that carries physical pace context while keeping the feature matrix lightweight.
* **Predicting Deltas:** Rather than predicting absolute lap times, the model predicts the delta (`LapTime - DriverBaseline`). This forces the model to learn the actual physical causes of pace variation (tire age, fuel burn) rather than simply memorizing that certain drivers are inherently faster than others.
* **Multi-Stage Docker Builds:** Separating the build environment (which requires heavy compilation toolchains) from the runtime environment (`python:3.10-slim`) significantly reduces the final production image size and surface area for vulnerabilities.

## 📝 Deployment Notes

* **Hugging Face Spaces:** The application is hosted on the free tier. The container spins down after 48 hours of inactivity and requires ~60 seconds to cold-boot on the next visit.
* **Git LFS:** Visualization assets and compiled models are tracked using Git Large File Storage (LFS) to comply with Hugging Face commit size restrictions.

## 👨‍💻 Author

**Nand Derek (Dhayanandha M.)**

B.Tech Computer Science (AI/ML)

🔗 [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/dhayanandham/) | [Hugging Face](https://huggingface.co/NandDerek)