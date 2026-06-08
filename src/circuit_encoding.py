import pandas as pd
import numpy as np
import os

def encode_circuit_features(input_filepath):
    print(f"📊 Loading master season dataset from {input_filepath}...")
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Master file not found at {input_filepath}. Run Day 8 first!")
        
    df = pd.read_csv(input_filepath)
    print(f"Loaded {len(df)} total racing laps. Initializing Target Encoding...")
    
    # ─── 1. CALCULATE TRACK PACE BASELINES (THE TARGET MAP) ─────────────────
    # Map the mean lap time for every single unique track in our dataset
    circuit_means = df.groupby('Circuit')['LapTimeSeconds'].mean().to_dict()
    
    print("\n⏱️ Calculated Baseline Track Speed Mapping (Average Lap Times):")
    print("=============================================================")
    for track, mean_time in sorted(circuit_means.items(), key=lambda x: x[1]):
        print(f"🏁 {track.ljust(15)} : {mean_time:.3f} seconds average pace")
    print("=============================================================")
    
    # ─── 2. MAP VALUES TO INJECT THE NEW NUMERICAL FEATURE ──────────────────
    # We create the new column while keeping the text 'Circuit' for debugging/UI mapping
    df['Circuit_Pace_Baseline'] = df['Circuit'].map(circuit_means)
    
    # ─── 3. FIXED SANITY CHECKS ─────────────────────────────────────────────
    # Verify no rows were missed or left empty
    missing_count = df['Circuit_Pace_Baseline'].isna().sum()
    if missing_count == 0:
        print("✅ Success: All text track profiles successfully converted to numerical baselines!")
    else:
        print(f"⚠️ Warning: Found {missing_count} rows with unmapped track profiles.")
        
    # Check total unique circuits captured
    print(f"✓ Unique circuits encoded: {len(circuit_means)}")
        
    # FIXED: Save to an engineering copy to prevent overwriting raw source data
    output_path = 'data/multi_track_features_encoded_2023.csv'
    df.to_csv(output_path, index=False)
    print(f"\n🏆 Enriched feature matrix securely saved locally to: {output_path}")

if __name__ == "__main__":
    encode_circuit_features('data/multi_track_features_2023.csv')