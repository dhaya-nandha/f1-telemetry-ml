import fastf1
import os

def initialize_fastf1_pipeline(cache_dir="cache"):
    """
    Creates a local cache directory and enables FastF1's caching engine.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created local storage volume at: {cache_dir}/")
        
    # Enable the local cache mechanism
    fastf1.Cache.enable_cache(cache_dir)
    print("✓ FastF1 persistent caching layer activated successfully.")

def download_and_cache_session(year, location, session_type="R"):
    """
    Downloads a complete F1 session and saves it directly to local disk cache.
    """
    print(f"\nRequesting {year} {location} Grand Prix [{session_type}]...")
    
    # Target specific event
    session = fastf1.get_session(year, location, session_type)
    
    print("Ingesting session telemetry and lap timing matrix from servers...")
    session.load(telemetry=True, laps=True, weather=True)
    
    print(f"✓ Data loaded. Verified {len(session.laps)} total lap records.")
    return session

if __name__ == "__main__":
    initialize_fastf1_pipeline()
    
    # Ingest Monaco 2023 Race session
    session_data = download_and_cache_session(2023, "Monaco", "R")
    
    # UPDATED: Using the reliable singular pick_driver check
    try:
        fastest_lap = session_data.laps.pick_driver('VER').pick_fastest()
    except AttributeError:
        # UPDATED: Use pick_drivers() to avoid the FutureWarning
        fastest_lap = session_data.laps.pick_drivers('VER').pick_fastest()
        
    print(f"\n--- FastF1 Ingestion Diagnostics ---")
    print(f"Fastest Lap Driver: {fastest_lap['Driver']}")
    print(f"Lap Time: {fastest_lap['LapTime']}")
    print(f"Compound Used: {fastest_lap['Compound']}")