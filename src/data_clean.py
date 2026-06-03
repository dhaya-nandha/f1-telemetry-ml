import fastf1
import pandas as pd
import os

# Enable our clean local storage locker
fastf1.Cache.enable_cache('cache')

def clean_session_data(year, location, session_type="R"):
    """
    Loads a cached session, filters out non-representative laps, 
    and normalizes time columns into numerical seconds.
    """
    print(f"Loading cached data for {year} {location}...")
    session = fastf1.get_session(year, location, session_type)
    session.load(telemetry=False, laps=True) # Only the lap matrix is needed for this step
    
    raw_laps = session.laps
    print(f"Initial raw lap count: {len(raw_laps)}")
    
    # ─── FIXED STEP 1: CORRECT PIT FILTERS & RELAXED TRACK STATUS ────────────
    # 1. PitOutTime.isna() safely removes slow out-laps without crashing.
    # 2. PitInTime.isna() drops the slow in-laps where cars enter the pits.
    # 3. TrackStatus checking '1', '2', '3' handles caution periods gracefully.
    clean_laps = raw_laps[
        (raw_laps['PitOutTime'].isna()) & 
        (raw_laps['PitInTime'].isna()) &
        (raw_laps['TrackStatus'].isin(['1', '2', '3'])) & 
        (raw_laps['LapTime'].notna())
    ].copy()
    
    # ─── STEP 2: CONVERT TIME TO NUMERICAL SECONDS ────────────────────────────
    # Translates timedelta format into clean numbers for machine learning models
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
    
    print(f"Cleaned representative lap count: {len(clean_laps)}")
    
    # ─── FIXED STEP 3: RE-CONFIGURED CLEAN FEATURE LIST ──────────────────────
    features = [
        'LapNumber', 'Driver', 'Team', 
        'LapTimeSeconds', 'Compound', 'TyreLife'
    ]
    
    return clean_laps[features]

if __name__ == "__main__":
    # Process and clean the cached Monaco data
    cleaned_df = clean_session_data(2023, "Monaco", "R")
    
    # Print out a clear preview of our model-ready rows
    print("\n✓ Sample of Processed Dataframe Rows:")
    print(cleaned_df.head(10))
    
    # Ensure local directory exists and save clean output
    os.makedirs('data', exist_ok=True)
    cleaned_df.to_csv('data/cleaned_monaco_2023.csv', index=False)
    print("\n✓ Cleaned data exported to data/cleaned_monaco_2023.csv")