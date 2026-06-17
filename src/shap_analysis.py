import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def run_shap_analysis(data_path, model_path):
    print("🏎️ Loading production model and dataset for Explainable AI (SHAP) audit...")
    
    # ─── 1. RECONSTRUCT THE EXACT DAY 17 TEST SET ─────────────────────────
    df = pd.read_csv(data_path)
    
    fastest_laps = df.groupby('Circuit')['LapTimeSeconds'].min().to_dict()
    df['Circuit_Fastest'] = df['Circuit'].map(fastest_laps)
    clean_df = df[df['LapTimeSeconds'] <= df['Circuit_Fastest'] * 1.07].copy()
    
    clean_df['Tyre_Log_Penalty'] = np.log1p(clean_df['TyreLife'])
    unique_tracks = sorted(clean_df['Circuit'].unique())
    track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
    clean_df['Circuit_ID'] = clean_df['Circuit'].map(track_to_id)
    clean_df = clean_df.reset_index(drop=True)

    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

    # Use training data to map baselines (Strict Isolation)
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

    test_final = apply_baselines(test_df)

    features = [
        'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
        'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
    ]
    
    X_test = test_final[features]

    # ─── 2. LOAD THE OPTIMIZED MODEL ─────────────────────────────────────
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # ─── 3. INITIALIZE SHAP TREE EXPLAINER ───────────────────────────────
    print("⚡ Calculating Shapley values (this takes a few seconds)...")
    explainer = shap.TreeExplainer(model)
    
    # We use a random sample of 1000 laps from the test set for visual clarity
    X_sample = X_test.sample(1000, random_state=42)
    shap_values = explainer(X_sample)

    # ─── 4. GENERATE MACRO SUMMARY PLOT ──────────────────────────────────
    print("📊 Rendering SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    plt.style.use('dark_background')
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("XGBoost Engine: Feature Impact on Lap Time Deltas", color='white', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.close()

    # ─── 5. GENERATE MICRO WATERFALL PLOT (SINGLE LAP) ───────────────────
    print("📊 Rendering SHAP Waterfall Plot for a single specific lap...")
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    # FIXED: Modern SHAP API call
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Micro-Analysis: How the AI calculated this exact lap", color='white', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.close()

    print("✅ SHAP analysis complete! Two high-resolution images saved to your project folder.")

if __name__ == "__main__":
    run_shap_analysis('data/multi_track_features_final_2023.csv', 'models/xgb_lap_predictor.json')