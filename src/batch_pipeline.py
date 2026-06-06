import fastf1
import pandas as pd
import numpy as np
import os

# Enable cache to accelerate multi-race telemetry hits
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

def process_race_data(year, track_name):
    """Downloads, cleans, and engineers features for a single race."""
    print(f"\n⚙️ Processing {year} {track_name} Grand Prix...")
    try:
        session = fastf1.get_session(year, track_name, 'R')
        session.load(telemetry=False, weather=False) # Keep it lightweight for downloading
        
        laps = session.laps
        
        # ─── 1. FIXED CLEANING (Day 3 Proven Boolean Mask) ─────────────────────
        clean_laps = laps[
            (laps['PitOutTime'].isna()) &  # Strip out out-laps
            (laps['LapTime'].notna()) &    # Must have a valid lap time
            (laps['Driver'].notna())       # Must have a valid driver mapping
        ].copy()
        
        # Convert Timedelta to Seconds
        clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
        
        # ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────
        MAX_FUEL_KG = 110.0
        # FIXED: Track-specific burn rates placeholder note
        # TODO: Move to track-specific map (Monaco: 1.30, Silverstone: 1.45, Monza: 1.60)
        BURN_RATE_PER_LAP = 1.30  
        
        clean_laps['FuelLoadKg'] = MAX_FUEL_KG - (clean_laps['LapNumber'] * BURN_RATE_PER_LAP)
        clean_laps['FuelLoadKg'] = clean_laps['FuelLoadKg'].clip(lower=0.5)
        
        compound_multipliers = {'SOFT': 1.5, 'MEDIUM': 1.0, 'HARD': 0.6, 'INTERMEDIATE': 0.8, 'WET': 0.5}
        clean_laps['CompoundMultiplier'] = clean_laps['Compound'].str.upper().map(compound_multipliers).fillna(1.0)
        clean_laps['TyreDegradationIndex'] = clean_laps['TyreLife'] * clean_laps['CompoundMultiplier']
        
        # ─── 3. NEW FEATURE: CIRCUIT ACCUMULATION ─────────────────────────────
        clean_laps['Circuit'] = track_name
        
        # Slice only the critical model attributes
        final_features = clean_laps[['Circuit', 'Driver', 'LapNumber', 'Compound', 'TyreLife', 'FuelLoadKg', 'TyreDegradationIndex', 'LapTimeSeconds']]
        
        return final_features

    except Exception as e:
        print(f"⚠️ Failed to process {track_name}: {e}")
        return None

if __name__ == "__main__":
    target_tracks = ['Monaco', 'Silverstone', 'Monza']
    year = 2023
    
    all_races_data = []
    
    for track in target_tracks:
        track_df = process_race_data(year, track)
        if track_df is not None:
            all_races_data.append(track_df)
            print(f"✓ {track} processed: {len(track_df)} valid racing laps extracted.")
            
    # Concatenate all tracks into a single master matrix
    master_df = pd.concat(all_races_data, ignore_index=True)
    
    # Save output data layer locally
    os.makedirs('data', exist_ok=True)
    output_path = 'data/multi_track_features_2023.csv'
    master_df.to_csv(output_path, index=False)
    
    print(f"\n🏆 BATCH PIPELINE COMPLETE!")
    print(f"Total Laps in Database: {len(master_df)}")
    print(f"File saved locally to: {output_path}")
    
    # FIXED: Warning note for the upcoming training loop transition
    print("\n⚠️ NOTE: Model was trained on Monaco only.")
    print("This multi-circuit data requires model retraining in Week 2!")