import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import optuna
import os
import json

# Silence Optuna's native verbose logs to keep our terminal output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, clean_df):
    # ─── 1. DEFINE THE OPTIMIZATION SEARCH SPACE ───────────────────────────
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.70, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.70, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'random_state': 42,
        'n_jobs': -1
    }

    # ─── 2. K-FOLD CROSS-VALIDATION WITH TARGET ISOLATION ─────────────────
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3 splits optimized for tuning speed
    fold_maes = []

    for train_idx, val_idx in kf.split(clean_df):
        train_df = clean_df.iloc[train_idx].copy()
        val_df = clean_df.iloc[val_idx].copy()

        # Calculate baselines strictly inside the training fold to block target leakage
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

        train_fold = apply_baselines(train_df)
        val_fold = apply_baselines(val_df)

        features = [
            'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
            'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
        ]

        X_train = train_fold[features]
        y_train = train_fold['LapTime_Delta']
        X_val = val_fold[features]
        y_val_delta = val_fold['LapTime_Delta']
        y_val_actual = val_fold['LapTimeSeconds']
        val_driver_baselines = val_fold['Driver_Track_Baseline']

        # Train model with early stopping to prevent over-fitting to the fold
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val_delta)], verbose=False)

        # Invert target transformation to evaluate on true absolute seconds
        predicted_deltas = model.predict(X_val)
        final_predictions = predicted_deltas + val_driver_baselines
        fold_maes.append(mean_absolute_error(y_val_actual, final_predictions))

    return np.mean(fold_maes)

def run_tuning_study(data_path):
    print(f"🏎️ Ingesting dataset for optimization: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Pre-split feature preparation (safe transformations)
    fastest_laps = df.groupby('Circuit')['LapTimeSeconds'].min().to_dict()
    df['Circuit_Fastest'] = df['Circuit'].map(fastest_laps)
    clean_df = df[df['LapTimeSeconds'] <= df['Circuit_Fastest'] * 1.07].copy()
    
    clean_df['Tyre_Log_Penalty'] = np.log1p(clean_df['TyreLife'])
    unique_tracks = sorted(clean_df['Circuit'].unique())
    track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
    clean_df['Circuit_ID'] = clean_df['Circuit'].map(track_to_id)
    clean_df = clean_df.reset_index(drop=True)

    print("\n⚡ Initializing Bayesian TPE Search Space (10 Optimization Trials)...")
    print("=============================================================")
    
    # Initialize optimization study direction
    study = optuna.create_study(direction="minimize")
    
    # Real-time progress tracker callback
    def logging_callback(study, trial):
        print(f"Trial {trial.number:2d} | Current Trial MAE: {trial.value:.3f}s | Historic Best MAE: {study.best_value:.3f}s")

    study.optimize(lambda trial: objective(trial, clean_df), n_trials=10, callbacks=[logging_callback])

    print("=============================================================")
    print("🏆 OPTUNA HYPERPARAMETER TUNING TARGET FRONTIER FOUND:")
    print(f"⏱️ Absolute Best Cross-Validation MAE: {study.best_value:.3f} seconds")
    print("\n💡 Optimized Architecture Configuration:")
    print(json.dumps(study.best_params, indent=4))
    print("=============================================================")

if __name__ == "__main__":
    run_tuning_study('data/multi_track_features_final_2023.csv')