import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import os

def train_season_engine(data_path):
    print(f" Jane loading finalized season dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Final feature matrix not found at {data_path}. Complete Day 10 first!")
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total racing rows. Executing high-dimensional Target Encoding...")
    
    # ─── 1. FIXED: DUAL-TARGET ENCODING FOR DRIVER & COMPOUND ────────────────
    # Map driver skill baseline profiles
    driver_means = df.groupby('Driver')['LapTimeSeconds'].mean().to_dict()
    df['Driver_Baseline'] = df['Driver'].map(driver_means)
    
    # Map tyre compound performance profiles
    compound_means = df.groupby('Compound')['LapTimeSeconds'].mean().to_dict()
    df['Compound_Baseline'] = df['Compound'].map(compound_means)
    
    print(f"✓ Engineered baseline nodes for {len(driver_means)} drivers and {len(compound_means)} compounds.")
    
    # ─── 2. UPGRADED MODEL FEATURE SELECTION ──────────────────────────────────
    features = [
        'LapNumber', 
        'TyreLife', 
        'FuelLoadKg', 
        'TyreDegradationIndex', 
        'Circuit_Pace_Baseline',  # Track Profile Info
        'Driver_Baseline',       # Driver Skill Info (FIXED)
        'Compound_Baseline'      # Tyre Performance Info (FIXED)
    ]
    target = 'LapTimeSeconds'
    
    X = df[features]
    y = df[target]
    
    # ─── 3. DATASET SPLIT (80% Training, 20% Unseen Validation) ─────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Dataset split: {len(X_train)} training rows | {len(X_test)} validation rows.")
    
    # ─── 4. INITIALIZE HIGH-PERFORMANCE XGBOOST ENGINE ──────────────────────
    print("⚡ Training multi-circuit XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1  # Utilize all available CPU threads
    )
    
    model.fit(X_train, y_train)
    
    # ─── 5. PERFORMANCE METRICS EVALUATION ──────────────────────────────────
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n🏆 UPGRADED SPRINT TRAINING SCORING BOARD:")
    print("=============================================================")
    print(f"⏱️ Mean Absolute Error (MAE) : {mae:.3f} seconds")
    print(f"📈 Variance Explained (R² Score): {r2:.4f} ({r2*100:.1f}%)")
    print("=============================================================")
    
    # ─── 6. SECURE MODEL ARTIFACT LOCKDOWN ─────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model_output_path = 'models/xgb_lap_predictor.json'
    model.save_model(model_output_path)
    print(f"✅ Enhanced model framework securely locked at: {model_output_path}")

if __name__ == "__main__":
    train_season_engine('data/multi_track_features_final_2023.csv')