import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

def analyze_model_residuals(data_path, model_path):
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Make sure Day 11 training completed and files exist!")

    df = pd.read_csv(data_path)
    
    # Reconstruct the exact encoding layers used in training
    driver_means = df.groupby('Driver')['LapTimeSeconds'].mean().to_dict()
    df['Driver_Baseline'] = df['Driver'].map(driver_means)
    compound_means = df.groupby('Compound')['LapTimeSeconds'].mean().to_dict()
    df['Compound_Baseline'] = df['Compound'].map(compound_means)
    
    features = ['LapNumber', 'TyreLife', 'FuelLoadKg', 'TyreDegradationIndex', 
                'Circuit_Pace_Baseline', 'Driver_Baseline', 'Compound_Baseline']
    
    # Split exactly the same way to isolate the test set rows
    _, X_test, _, y_test = train_test_split(df[features], df['LapTimeSeconds'], test_size=0.2, random_state=42)
    
    # Load model and predict
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    preds = model.predict(X_test)
    
    # Create an evaluation dataframe matching indices back to the original metadata
    test_indices = X_test.index
    analysis_df = df.loc[test_indices].copy()
    analysis_df['PredictedTime'] = preds
    analysis_df['AbsoluteError'] = np.abs(analysis_df['LapTimeSeconds'] - preds)
    
    print("\n📊 TRACK-BY-TRACK ERROR PROFILED LEADERBOARD:")
    print("=============================================================")
    track_errors = analysis_df.groupby('Circuit')['AbsoluteError'].mean().sort_values(ascending=False)
    
    for track, track_mae in track_errors.items():
        print(f"🏁 {track.ljust(15)} : MAE = {track_mae:.3f} seconds")
    print("=============================================================")
    
    # Global Summary Indicator
    print(f"⚠️ Global Aggregated MAE: {analysis_df['AbsoluteError'].mean():.3f}s")

if __name__ == "__main__":
    analyze_model_residuals('data/multi_track_features_final_2023.csv', 'models/xgb_lap_predictor.json')