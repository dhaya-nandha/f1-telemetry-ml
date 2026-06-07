import fastf1
import pandas as pd
import numpy as np
import os

# Enable cache to maximize download speed and reuse locally saved data
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

def process_race_data(year, track_name):
    """Downloads, cleans, and engineers features for a single race."""
    print(f"⚙️ Ingesting: {year} {track_name} Grand Prix...")
    try:
        session = fastf1.get_session(year, track_name, 'R')
        session.load(telemetry=False, weather=False)  # Lightweight configuration
        
        laps = session.laps
        
        # ─── 1. SANITIZATION MASK ─────────────────────────────────────────────
        clean_laps = laps[
            (laps['PitOutTime'].isna()) &  # Eliminate pit out-laps
            (laps['LapTime'].notna()) &    # Ensure a valid lap timing exists
            (laps['Driver'].notna())       # Validate driver assignment
        ].copy()
        
        # Convert Timedelta to Float Seconds
        clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
        
        # ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────
        MAX_FUEL_KG = 110.0
        # TODO: Implement track-dependent mapping on Day 10 (e.g., Monza: 1.60, Monaco: 1.30)
        BURN_RATE_PER_LAP = 1.30  # Baseline global constant
        
        clean_laps['FuelLoadKg'] = MAX_FUEL_KG - (clean_laps['LapNumber'] * BURN_RATE_PER_LAP)
        clean_laps['FuelLoadKg'] = clean_laps['FuelLoadKg'].clip(lower=0.5)
        
        compound_multipliers = {'SOFT': 1.5, 'MEDIUM': 1.0, 'HARD': 0.6, 'INTERMEDIATE': 0.8, 'WET': 0.5}
        clean_laps['CompoundMultiplier'] = clean_laps['Compound'].str.upper().map(compound_multipliers).fillna(1.0)
        clean_laps['TyreDegradationIndex'] = clean_laps['TyreLife'] * clean_laps['CompoundMultiplier']
        
        # ─── 3. CATEGORICAL PROFILE MAPPING ──────────────────────────────────
        clean_laps['Circuit'] = track_name
        
        # Slice core dimensional data columns
        final_features = clean_laps[['Circuit', 'Driver', 'LapNumber', 'Compound', 'TyreLife', 'FuelLoadKg', 'TyreDegradationIndex', 'LapTimeSeconds']]
        return final_features

    except Exception as e:
        print(f"⚠️ Track Skip Notice: Failed to process {track_name} -> {e}")
        return None

if __name__ == "__main__":
    # Corrected official FastF1 strings for the 2023 calendar
    season_tracks = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 
        'Monaco', 'Spain', 'Canada', 'Austria', 'Silverstone', 
        'Hungary', 'Spa', 'Zandvoort', 'Monza', 'Singapore', 
        'Japan', 'Qatar', 'Austin', 'Mexico', 'Sao Paulo', 'Las Vegas', 'Abu Dhabi'
    ]
    year = 2023
    
    all_season_data = []
    
    print("🚀 STARTING BULK CHAMPIONSHIP DATA INGESTION MATRIX...")
    print("=====================================================")
    
    for track in season_tracks:
        track_df = process_race_data(year, track)
        if track_df is not None:
            all_season_data.append(track_df)
            print(f"✓ Data Lock: {track} yielded {len(track_df)} structured racing rows.")
            
    # Combine individual track frames into a continuous master database
    master_season_df = pd.concat(all_season_data, ignore_index=True)
    
    # Save compilation to the local data directory
    os.makedirs('data', exist_ok=True)
    output_path = 'data/multi_track_features_2023.csv'
    master_season_df.to_csv(output_path, index=False)
    
    print("=====================================================")
    print(f"🏆 WEEK 2 MASTER DATA ARCHIVE COMPILED!")
    print(f"Total Season Database Capacity: {len(master_season_df)} Total Laps Stacked.")
    print(f"Output File Destination: {output_path}")
    print("\n⚠️ SYSTEM NOTE: Ready for Day 9 categorical circuit encoding pipeline adjustments.")