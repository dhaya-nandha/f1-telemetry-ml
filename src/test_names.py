import fastf1

# Diagnostic array to probe FastF1's naming parsing engine
test_tracks = ['Bahrain', 'Monaco', 'Great Britain', 'Italy', 'USA']

print("🔍 Probing FastF1 API naming conventions...")
print("=====================================================")

for track in test_tracks:
    try:
        # Attempting lightweight session initialization
        session = fastf1.get_session(2023, track, 'R')
        print(f"✓ '{track}' is VALID")
    except Exception as e:
        print(f"✗ '{track}' FAILED: {e}")