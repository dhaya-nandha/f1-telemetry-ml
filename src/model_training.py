import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

def train_lap_time_model(filepath):
    """
    Trains an XGBoost regression model to predict lap times based on
    driver, tire compound, fuel weight, and tire degradation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found at {filepath}. Run Day 4 first!")
        
    print(f"Loading feature matrix from {filepath}...")
    df = pd.read_csv(filepath)
    
    # ─── 1. SELECTING OUR INPUTS (FEATURES) AND OUTPUT (TARGET) ──────────────
    features = ['Driver', 'Compound', 'FuelLoadKg', 'TyreDegradationIndex']
    target = 'LapTimeSeconds'
    
    X = df[features].copy()
    y = df[target]
    
    # ─── 2. FORMATTING TEXT DATA FOR THE AI ──────────────────────────────────
    # XGBoost is smart, but we need to tell it which columns are text/categories
    X['Driver'] = X['Driver'].astype('category')
    X['Compound'] = X['Compound'].astype('category')
    
    # ─── 3. SPLIT THE DATA (TRAINING VS. EXAM) ───────────────────────────────
    # We give the model 80% of the laps to study, and keep 20% hidden to test it later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining XGBoost Engine (This simulates the racing physics)...")
    # enable_categorical=True allows it to read 'Driver' and 'Compound' directly
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        enable_categorical=True,
        random_state=42
    )
    
    # 🧠 This is the exact moment the machine learns
    model.fit(X_train, y_train)
    
    # ─── 4. ADMINISTER THE FINAL EXAM ────────────────────────────────────────
    print("Scoring the model on the hidden 20% test data...")
    predictions = model.predict(X_test)
    
    # Calculate Mean Absolute Error (How many seconds off is it on average?)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\n🏆 Model Accuracy (Mean Absolute Error): {mae:.3f} seconds")
    print(f"-> On average, the model's prediction is off by just {mae:.3f}s from the real lap time!")
    
    # ─── 5. SAVE THE TRAINED BRAIN ───────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgb_lap_predictor.json'
    model.save_model(model_path)
    print(f"\n✓ Trained model securely saved to {model_path}")

if __name__ == "__main__":
    train_lap_time_model('data/features_monaco_2023.csv')