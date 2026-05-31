import sys
import os

# PATH FIX: Ensures Python can find data_fetch.py when run from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_fetch import fetch_season_races

def generate_and_save_index(season=2023):
    """
    Queries season details and saves a standard CSV index file.
    """
    print(f"Building master index for the {season} Formula 1 season...")
    raw_json = fetch_season_races(season)
    race_list = raw_json['MRData']['RaceTable']['Races']
    
    # Parse unstructured API components directly into a flat structure
    races_df = pd.DataFrame([
        {
            'season': int(r['season']),
            'round': int(r['round']),
            'race_name': r['raceName'],
            'race_date': r['date'],
            'circuit_id': r['Circuit']['circuitId'],
        }
        for r in race_list
    ])
    
    # Explicit verification print to verify true round numbers on your terminal
    print("\n--- Verifying Schedule & Round Numbers ---")
    for idx, row in races_df.iterrows():
        print(f"Round {row['round']}: {row['race_name']}")
    print("------------------------------------------\n")
    
    output_path = 'data/races_2023.csv'
    races_df.to_csv(output_path, index=False)
    print(f"✓ Indexing complete. File saved to: {output_path}")

if __name__ == "__main__":
    generate_and_save_index(2023)