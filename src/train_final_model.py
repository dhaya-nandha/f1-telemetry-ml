import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import os
import json

def retrain_optimized_engine(data_path):
    print(f"🏎️ Loading season dataset for definitive retraining: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # ─── 1. CLEAN AIR FILTER & FEATURE PREPARATION ────────────────────────
    fastest_laps = df.groupby('Circuit')['LapTimeSeconds'].min().to_dict()
    df['Circuit_Fastest'] = df['Circuit'].map(fastest_laps)
    clean_df = df[df['LapTimeSeconds'] <= df['Circuit_Fastest'] * 1.07].copy()
    
    clean_df['Tyre_Log_Penalty'] = np.log1p(clean_df['TyreLife'])
    unique_tracks = sorted(clean_df['Circuit'].unique())
    track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
    clean_df['Circuit_ID'] = clean_df['Circuit'].map(track_to_id)
    clean_df = clean_df.reset_index(drop=True)

    # ─── 2. SEPARATE HELD-OUT TEST SET (ANTI-LEAKAGE TRAP) ────────────────
    print("⚡ Isolating 20% held-out test split for final generalization audit...")
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

    # ─── 3. EXTRACT COHORT TARGET ENCODINGS FROM TRAINING ONLY ────────────
    print("⚡ Compiling precision baseline maps from training cohort...")
    circuit_means = train_df.groupby('Circuit')['LapTimeSeconds'].mean().to_dict()
    driver_track_means = train_df.groupby(['Circuit', 'Driver'])['LapTimeSeconds'].mean().to_dict()
    compound_track_means = train_df.groupby(['Circuit', 'Compound'])['LapTimeSeconds'].mean().to_dict()
    
    def apply_baselines(target_df):
        target_df = target_df.copy()
        target_df['Circuit_Pace_Baseline'] = target_df['Circuit'].map(circuit_means)
        
        target_df['Driver_Track_Baseline'] = target_df.apply(
            lambda row: driver_track_means.get((row['Circuit'], row['Driver']), circuit_means.get(row['Circuit'], 85.0)), axis=1
        )
        target_df['Compound_Track_Baseline'] = target_df.apply(
            lambda row: compound_track_means.get((row['Circuit'], row['Compound']), circuit_means.get(row['Circuit'], 85.0)), axis=1
        )
        target_df['LapTime_Delta'] = target_df['LapTimeSeconds'] - target_df['Driver_Track_Baseline']
        return target_df

    train_final = apply_baselines(train_df)
    test_final = apply_baselines(test_df)

    features = [
        'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
        'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
    ]
    
    X_train = train_final[features]
    y_train = train_final['LapTime_Delta']
    
    X_test = test_final[features]
    y_test_delta = test_final['LapTime_Delta']
    y_test_actual = test_final['LapTimeSeconds']
    test_driver_baselines = test_final['Driver_Track_Baseline']

    # ─── 4. LOAD THE VERIFIED OPTUNA OPTIMIZED HYPERPARAMETERS ────────────
    print("⚡ Injecting verified Optuna hyperparameter frontier weights into XGBoost...")
    optimized_params = {
        "n_estimators": 950,
        "max_depth": 9,
        "learning_rate": 0.07689014919558045,
        "subsample": 0.9362895675257343,
        "colsample_bytree": 0.7532901555078426,
        "min_child_weight": 9,
        "reg_alpha": 0.014641790464002647,
        "reg_lambda": 0.008620323847455766,
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "random_state": 42,
        "n_jobs": -1
    }
    
    model = xgb.XGBRegressor(**optimized_params)
    model.fit(X_train, y_train, verbose=False)

    # ─── 5. FINAL GENERALIZATION AUDIT ─────────────────────────────────────
    predicted_deltas = model.predict(X_test)
    final_lap_predictions = predicted_deltas + test_driver_baselines
    
    mae = mean_absolute_error(y_test_actual, final_lap_predictions)
    naive_mae = mean_absolute_error(y_test_actual, test_driver_baselines)
    total_r2 = r2_score(y_test_actual, final_lap_predictions)
    physics_r2 = r2_score(y_test_delta, predicted_deltas)
    
    print("\n🏆 THE FINAL DEFINITIVE LEAK-FREE SCOREBOARD:")
    print("=============================================================")
    print(f"🧠 Naive Baseline MAE (Driver Average) : {naive_mae:.3f} seconds")
    print(f"🤖 Optimized XGBoost MAE (Final Engine): {mae:.3f} seconds")
    print(f"🔥 True Physical Value-Add             : {naive_mae - mae:.3f} seconds")
    print("=============================================================")
    print(f"📈 Total Season R² Score               : {total_r2*100:.2f}%")
    print(f"📉 Isolated Physics R² Score           : {physics_r2*100:.2f}%")
    print("=============================================================")

    # ─── 6. PRODUCTION BINARY SERIALIZATION ───────────────────────────────
    print("⚡ Overwriting localized deployment binaries with optimized parameters...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgb_lap_predictor.json')
    
    dt_serializable = {f"{k[0]}_{k[1]}": v for k, v in driver_track_means.items()}
    ct_serializable = {f"{k[0]}_{k[1]}": v for k, v in compound_track_means.items()}
    
    with open('models/driver_track_means.json', 'w') as f: json.dump(dt_serializable, f)
    with open('models/compound_track_means.json', 'w') as f: json.dump(ct_serializable, f)
    with open('models/track_to_id.json', 'w') as f: json.dump(track_to_id, f)
    np.save('models/circuit_means.npy', circuit_means)
    
    print("✅ Success. Final production artifacts are locked and fully synchronized.")

if __name__ == "__main__":
    retrain_optimized_engine('data/multi_track_features_final_2023.csv')