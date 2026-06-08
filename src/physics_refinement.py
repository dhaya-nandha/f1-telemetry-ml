import pandas as pd
import numpy as np
import os

def refine_fuel_physics(input_filepath):
    print(f"📊 Loading encoded dataset from {input_filepath}...")
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Encoded file not found at {input_filepath}. Complete Day 9 first!")
        
    df = pd.read_csv(input_filepath)
    print(f"Loaded {len(df)} total racing rows. Injecting track-specific physics profiles...")
    
    # ─── 1. FIXED: ALIGNED DIRECTLY WITH FASTF1 VENUE STRINGS FROM DATASET ───
    fuel_burn_map = {
        'Monaco': 1.30,
        'Singapore': 1.35,
        'Hungary': 1.38,
        'Zandvoort': 1.40,        # Aligned from 'Netherlands'
        'Mexico': 1.40,
        'Azerbaijan': 1.42,
        'Miami': 1.43,
        'Australia': 1.44,
        'Bahrain': 1.45,
        'Silverstone': 1.45,     # Aligned from 'Great Britain'
        'Spain': 1.46,
        'Austin': 1.47,          # Aligned from 'USA'
        'Japan': 1.48,
        'Qatar': 1.50,
        'Saudi Arabia': 1.52,
        'Austria': 1.52,
        'Sao Paulo': 1.54,       # Aligned from 'Brazil'
        'Abu Dhabi': 1.55,
        'Las Vegas': 1.56,
        'Canada': 1.58,
        'Spa': 1.60,             # Aligned from 'Belgium'
        'Monza': 1.62            # Aligned from 'Italy'
    }
    
    GLOBAL_AVERAGE_BURN = 1.45
    
    # ─── 2. DYNAMICALLY RE-ENGINEER FUEL LOADS ───────────────────────────────
    MAX_FUEL_KG = 110.0
    
    def calculate_precise_fuel(row):
        track = row['Circuit']
        lap = row['LapNumber']
        burn_rate = fuel_burn_map.get(track, GLOBAL_AVERAGE_BURN)
        
        # Calculate load remaining and apply a strict physical safety floor
        remaining_fuel = MAX_FUEL_KG - (lap * burn_rate)
        return max(remaining_fuel, 0.5)

    print("⚡ Vectorizing consumption curves across 23,366 data rows...")
    df['FuelLoadKg'] = df.apply(calculate_precise_fuel, axis=1)
    
    # ─── 3. VALIDATION SANITY CHECK ─────────────────────────────────────────
    print("\n🔍 Verifying Physics Refinement Metrics:")
    print("=============================================================")
    monaco_sample = df[df['Circuit'] == 'Monaco']['FuelLoadKg'].min()
    monza_sample = df[df['Circuit'] == 'Monza']['FuelLoadKg'].min()
    
    print(f"🏎️ Monaco Minimum Fuel Remaining Profile: {monaco_sample:.2f} kg")
    print(f"🚀 Monza Minimum Fuel Remaining Profile:  {monza_sample:.2f} kg")
    print("=============================================================")
    
    # Save the output to a distinct, final feature matrix file
    output_path = 'data/multi_track_features_final_2023.csv'
    df.to_csv(output_path, index=False)
    print(f"\n🏆 Final optimized feature matrix written to: {output_path}")
    print("⚠️ READY FOR DAY 11 MULTI-CIRCUIT XGBOOST TRAINING SEQUENCE.")

if __name__ == "__main__":
    refine_fuel_physics('data/multi_track_features_encoded_2023.csv')