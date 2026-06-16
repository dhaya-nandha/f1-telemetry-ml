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
    
    # ─── 1. TELEMETRY CLEANING (FIA 107% RULE) ─────────────────────────────
    fastest_laps = df.groupby('Circuit')['LapTimeSeconds'].min().to_dict()
    df['Circuit_Fastest'] = df['Circuit'].map(fastest_laps)
    clean_df = df[df['LapTimeSeconds'] <= df['Circuit_Fastest'] * 1.07].copy()
    print(f"✓ Clean Air Filter: Retained {len(clean_df)} pure racing laps.")

    # Feature engineering for non-linear tires
    clean_df['Tyre_Log_Penalty'] = np.log1p(clean_df['TyreLife'])
    
    unique_tracks = sorted(clean_df['Circuit'].unique())
    track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
    clean_df['Circuit_ID'] = clean_df['Circuit'].map(track_to_id)
    
    # ─── 2. CRITICAL FIX: TRAIN-TEST SPLIT FIRST (ANTI-LEAKAGE) ───────────
    print("⚡ Splitting dataset before calculating target encodings...")
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

    # ─── 3. CALCULATE COHORT AVERAGES STRICTLY ON TRAINING DATA ───────────
    print("⚡ Compiling target encodings from training split only...")
    circuit_means = train_df.groupby('Circuit')['LapTimeSeconds'].mean().to_dict()
    driver_track_means = train_df.groupby(['Circuit', 'Driver'])['LapTimeSeconds'].mean().to_dict()
    compound_track_means = train_df.groupby(['Circuit', 'Compound'])['LapTimeSeconds'].mean().to_dict()
    
    # ─── 4. MAP MAPPINGS BACK TO TRAIN AND TEST SEPARATELY ─────────────────
    def apply_baselines(target_df):
        target_df = target_df.copy()
        target_df['Circuit_Pace_Baseline'] = target_df['Circuit'].map(circuit_means)
        
        target_df['Driver_Track_Baseline'] = target_df.apply(
            lambda row: driver_track_means.get((row['Circuit'], row['Driver']), circuit_means.get(row['Circuit'], 85.0)), axis=1
        )
        target_df['Compound_Track_Baseline'] = target_df.apply(
            lambda row: compound_track_means.get((row['Circuit'], row['Compound']), circuit_means.get(row['Circuit'], 85.0)), axis=1
        )
        # Target transformation
        target_df['LapTime_Delta'] = target_df['LapTimeSeconds'] - target_df['Driver_Track_Baseline']
        return target_df

    train_df = apply_baselines(train_df)
    test_df = apply_baselines(test_df)

    # Define features
    features = [
        'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
        'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
    ]
    
    X_train = train_df[features]
    y_train = train_df['LapTime_Delta']
    
    X_test = test_df[features]
    y_test_delta = test_df['LapTime_Delta']
    y_test_actual = test_df['LapTimeSeconds']
    test_driver_baselines = test_df['Driver_Track_Baseline']
    
    # ─── 5. MAE-OPTIMIZED XGBOOST ENGINE (TRAINING STEP) ───────────────────
    print("⚡ Launching production leak-free XGBoost configuration...")
    model = xgb.XGBRegressor(
        n_estimators=850,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='reg:absoluteerror',
        eval_metric='mae',
        random_state=42,
        n_jobs=-1
    )
    
    # THIS IS WHAT WAS DELETED IN YOUR VERSION:
    model.fit(X_train, y_train, eval_set=[(X_test, y_test_delta)], verbose=False)
    
    # ─── 6. EVALUATION & NAIVE BASELINE TEST ───────────────────────────────
    predicted_deltas = model.predict(X_test)
    final_lap_predictions = predicted_deltas + test_driver_baselines
    
    # Calculate Model MAE
    mae = mean_absolute_error(y_test_actual, final_lap_predictions)
    r2_total = r2_score(y_test_actual, final_lap_predictions)
    
    # Calculate Naive MAE (What if the model just guessed the baseline?)
    naive_mae = mean_absolute_error(y_test_actual, test_driver_baselines)
    
    # Calculate R2 on the Deltas (How well did we predict the physics?)
    r2_delta = r2_score(y_test_delta, predicted_deltas)
    
    print("\n🏆 THE TRUE LEAK-FREE SCORING BOARD:")
    print("=============================================================")
    print(f"🧠 Naive Baseline MAE (Driver Average) : {naive_mae:.3f} seconds")
    print(f"🤖 XGBoost Model MAE (Physics Engine)  : {mae:.3f} seconds")
    print(f"🔥 Model's Physical Value-Add          : {naive_mae - mae:.3f} seconds")
    print("=============================================================")
    print(f"📈 Total R² (Including Track Length)   : {r2_total*100:.1f}%")
    print(f"📉 Physics R² (Predicting Deltas Only) : {r2_delta*100:.1f}%")
    print("=============================================================")
    
    # ─── 7. SERIALIZE METADATA MAPS ────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgb_lap_predictor.json')
    
    dt_serializable = {f"{k[0]}_{k[1]}": v for k, v in driver_track_means.items()}
    ct_serializable = {f"{k[0]}_{k[1]}": v for k, v in compound_track_means.items()}
    
    with open('models/driver_track_means.json', 'w') as f: json.dump(dt_serializable, f)
    with open('models/compound_track_means.json', 'w') as f: json.dump(ct_serializable, f)
    with open('models/track_to_id.json', 'w') as f: json.dump(track_to_id, f)
    np.save('models/circuit_means.npy', circuit_means)
        
    print("✅ Elite precision weights and localized layers securely locked.")

if __name__ == "__main__":
    train_season_engine('data/multi_track_features_final_2023.csv')