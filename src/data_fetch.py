import requests
import pandas as pd
import json

# Target URL pointing to the live Jolpica Ergast-compatible mirror
BASE_URL = "https://api.jolpi.ca/ergast/f1"

def fetch_season_races(season):
    """
    Fetch all race metadata and scheduling for a specific season.
    """
    url = f"{BASE_URL}/{season}.json?limit=100"
    resp = requests.get(url)
    resp.raise_for_status()  # Check for HTTP network errors
    return resp.json()

def fetch_race_laps(season, round_num):
    """
    Fetch all historical lap times for a specific race using a pagination loop.
    """
    all_laps = []
    # Loop over pages (API handles a maximum of 20 telemetry rows per page)
    for page in range(0, 50):  
        offset = page * 20
        url = f"{BASE_URL}/{season}/{round_num}/laps.json?limit=20&offset={offset}"
        resp = requests.get(url)
        resp.raise_for_status()
        
        data = resp.json()
        races_found = data['MRData']['RaceTable'].get('Races', [])
        
        # If no more lap records are returned, terminate the loop
        if not races_found:
            break
            
        all_laps.extend(races_found[0]['Laps'])
    return all_laps

if __name__ == "__main__":
    print("Testing Jolpica API Connection...")
    
    # Verification: 2023 Season Calendar
    races_data = fetch_season_races(2023)
    total_races = len(races_data['MRData']['RaceTable']['Races'])
    print(f"✓ Connection Success! 2023 season has {total_races} races.")
    
    # Verification: 2023 Monaco Grand Prix (CORRECTED: Round 8)
    print("Testing paginated lap ingestion pipeline for Monaco...")
    laps_data = fetch_race_laps(2023, 8)
    print(f"✓ Data Pipeline Verified! Monaco 2023 retrieved with {len(laps_data)} lap records.")