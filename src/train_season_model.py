import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import os
import json

def train_season_engine(data_path):
    print(f"🏎️ Loading finalized season dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Final feature matrix not found at {data_path}!")
        
    df = pd.read_csv(data_path)
    initial_count = len(df)
    
    # ─── 1. THE FIA 107% RULE FILTER ───────────────────────────────────────
    print("⚡ Applying strict FIA 107% Rule to isolate pure physics racing laps...")
    fastest_laps = df.groupby('Circuit')['LapTimeSeconds'].min().to_dict()
    df['Circuit_Fastest'] = df['Circuit'].map(fastest_laps)
    
    # Any lap slower than 107% of the fastest lap is traffic, a mistake, or an in/out lap
    clean_df = df[df['LapTimeSeconds'] <= df['Circuit_Fastest'] * 1.07].copy()
    
    dropped = initial_count - len(clean_df)
    print(f"✓ Purged {dropped} non-competitive/traffic anomalies. Retained {len(clean_df)} elite laps.")

    # ─── 2. NON-LINEAR TYRE CLIFF ENGINEERING ──────────────────────────────
    # Transforms linear tire age into a logarithmic curve to mimic the real grip drop-off
    clean_df['Tyre_Log_Penalty'] = np.log1p(clean_df['TyreLife'])

    # ─── 3. TRACK-LOCALIZED TARGET ENCODING ────────────────────────────────
    print("⚡ Building track-localized driver and compound baseline maps...")
    circuit_means = clean_df.groupby('Circuit')['LapTimeSeconds'].mean().to_dict()
    clean_df['Circuit_Pace_Baseline'] = clean_df['Circuit'].map(circuit_means)
    
    driver_track_means = clean_df.groupby(['Circuit', 'Driver'])['LapTimeSeconds'].mean().to_dict()
    compound_track_means = clean_df.groupby(['Circuit', 'Compound'])['LapTimeSeconds'].mean().to_dict()
    
    clean_df['Driver_Track_Baseline'] = clean_df.apply(
        lambda row: driver_track_means.get((row['Circuit'], row['Driver']), row['Circuit_Pace_Baseline']), axis=1
    )
    clean_df['Compound_Track_Baseline'] = clean_df.apply(
        lambda row: compound_track_means.get((row['Circuit'], row['Compound']), row['Circuit_Pace_Baseline']), axis=1
    )

    # ─── 4. TARGET TRANSFORMATION ──────────────────────────────────────────
    clean_df['LapTime_Delta'] = clean_df['LapTimeSeconds'] - clean_df['Driver_Track_Baseline']
    
    unique_tracks = sorted(clean_df['Circuit'].unique())
    track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
    clean_df['Circuit_ID'] = clean_df['Circuit'].map(track_to_id)
    
    # Added the Tyre_Log_Penalty feature!
    features = [
        'Circuit_ID', 
        'LapNumber', 
        'TyreLife', 
        'Tyre_Log_Penalty', 
        'FuelLoadKg', 
        'TyreDegradationIndex', 
        'Driver_Track_Baseline',   
        'Compound_Track_Baseline'  
    ]
    
    X_matrix = clean_df[features].copy()
    y_vector = clean_df['LapTime_Delta'].copy()
    
    meta_columns = ['Circuit_Pace_Baseline', 'Driver_Track_Baseline', 'LapTimeSeconds']
    X_meta = clean_df[meta_columns].copy()

    # ─── 5. TRAIN-TEST SPLIT ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test_delta = train_test_split(
        X_matrix, y_vector, test_size=0.2, random_state=42
    )
    _, X_test_meta = train_test_split(X_meta, test_size=0.2, random_state=42)
    
    y_test_actual = X_test_meta['LapTimeSeconds']
    test_driver_baselines = X_test_meta['Driver_Track_Baseline']
    
    y_train = y_train.squeeze()
    y_test_delta = y_test_delta.squeeze()
    
    # ─── 6. MAE-OPTIMIZED XGBOOST ENGINE ───────────────────────────────────
    print("⚡ Launching elite MAE-optimized multi-circuit XGBoost configuration...")
    model = xgb.XGBRegressor(
        n_estimators=850,       # Increased to map the new logarithmic features
        max_depth=8,
        learning_rate=0.03,     # Slower descent for extreme precision
        subsample=0.85,
        colsample_bytree=0.85,
        objective='reg:absoluteerror',  # CRITICAL: Forces model to ignore traffic anomalies
        eval_metric='mae',              # Optimize directly for MAE
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test_delta)],
        verbose=False
    )
    
    # ─── 7. BACK-TRANSFORMATION & SCOREBOARD ───────────────────────────────
    predicted_deltas = model.predict(X_test)
    final_lap_predictions = predicted_deltas + test_driver_baselines
    
    mae = mean_absolute_error(y_test_actual, final_lap_predictions)
    r2 = r2_score(y_test_actual, final_lap_predictions)
    
    print("\n🏆 ELITE PHYSICS-OPTIMIZED SCOREBOARD:")
    print("=============================================================")
    print(f"⏱️ True Racing Mean Absolute Error (MAE) : {mae:.3f} seconds")
    print(f"📈 Variance Explained (R² Score): {r2:.4f} ({r2*100:.1f}%)")
    print("=============================================================")
    
    # ─── 8. SERIALIZE METADATA MAPS ────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgb_lap_predictor.json')
    
    dt_serializable = {f"{k[0]}_{k[1]}": v for k, v in driver_track_means.items()}
    ct_serializable = {f"{k[0]}_{k[1]}": v for k, v in compound_track_means.items()}
    
    with open('models/driver_track_means.json', 'w') as f:
        json.dump(dt_serializable, f)
    with open('models/compound_track_means.json', 'w') as f:
        json.dump(ct_serializable, f)
    with open('models/track_to_id.json', 'w') as f:
        json.dump(track_to_id, f)
    np.save('models/circuit_means.npy', circuit_means)
        
    print("✅ Elite precision weights and localized layers securely locked.")

if __name__ == "__main__":
    train_season_engine('data/multi_track_features_final_2023.csv')