import pandas as pd
import numpy as np
import os

def engineer_f1_features(filepath):
    """
    Reads the cleaned lap data and engineers advanced racing features
    like fuel consumption estimation and compound-specific tire degradation indexes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned data file not found at {filepath}. Did you run Day 3's script?")
        
    print(f"Loading cleaned data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # ─── 1. ESTIMATING FUEL LOAD (THE 'LIGHTER CAR' VARIABLE) ────────────────
    # F1 regulation limit is roughly 110kg at start. Monaco is a 78-lap race.
    # Average fuel burn rate at Monaco is ~1.30 kg per lap (estimate).
    TOTAL_RACE_LAPS = 78
    MAX_FUEL_KG = 110.0
    BURN_RATE_PER_LAP = 1.30
    
    print("Engineering 'FuelLoadKg' feature...")
    # Fuel load decreases linearly as LapNumber goes up
    df['FuelLoadKg'] = MAX_FUEL_KG - (df['LapNumber'] * BURN_RATE_PER_LAP)
    # Clip at 0.5 just in case of any weird outlier tracking values
    df['FuelLoadKg'] = df['FuelLoadKg'].clip(lower=0.5)
    
    # ─── 2. TYRE DEGRADATION INDEX (THE 'GRIP DROP' VARIABLE) ────────────────
    # Soft compounds degrade faster than Mediums, which degrade faster than Hards.
    compound_multipliers = {
        'SOFT': 1.5,
        'MEDIUM': 1.0,
        'HARD': 0.6,
        'INTERMEDIATE': 0.8,
        'WET': 0.5
    }
    
    print("Engineering 'TyreDegradationIndex' feature...")
    # Map the compound column to its respective decay multiplier
    df['CompoundMultiplier'] = df['Compound'].str.upper().map(compound_multipliers).fillna(1.0)
    
    # Degradation Index = how many laps the tire has run * its compound wear severity
    df['TyreDegradationIndex'] = df['TyreLife'] * df['CompoundMultiplier']
    
    # Drop the temporary multiplier column so we keep our final matrix clean
    df = df.drop(columns=['CompoundMultiplier'])
    
    return df

if __name__ == "__main__":
    input_file = 'data/cleaned_monaco_2023.csv'
    
    # Run the engineering pipeline
    engineered_df = engineer_f1_features(input_file)
    
    print("\n✓ Sample of Engineered Dataframe Rows:")
    # Print the specific columns we engineered so we can verify the math
    print(engineered_df[['LapNumber', 'Driver', 'LapTimeSeconds', 'FuelLoadKg', 'TyreLife', 'TyreDegradationIndex']].head(10))
    
    # Check the fuel load on the final lap to ensure our math didn't break
    final_lap = engineered_df['LapNumber'].max()
    final_fuel = engineered_df[engineered_df['LapNumber'] == final_lap]['FuelLoadKg'].iloc[0]
    print(f"\nSanity Check: Fuel load on final lap ({final_lap}) is {final_fuel:.2f} kg")
    
    # Save the updated dataset locally
    output_file = 'data/features_monaco_2023.csv'
    engineered_df.to_csv(output_file, index=False)
    print(f"\n✓ Feature matrix successfully generated and saved to {output_file}")