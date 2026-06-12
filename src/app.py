import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os

# ─── PAGE CONFIGURATION ──────────────────────────────────────────────────
st.set_page_config(page_title="F1 AI Telemetry Engine", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Multi-Circuit AI Telemetry Predictor")
st.markdown("Powered by a Deep-Tree XGBoost Engine | Optimized for Absolute Physics Error (MAE: 0.314s)")

# ─── 1. SECURE ARTIFACT LOADING (CACHED FOR INFRASTRUCTURE SPEED) ────────
@st.cache_resource
def load_telemetry_engine():
    model = xgb.XGBRegressor()
    model.load_model('models/xgb_lap_predictor.json')
    
    with open('models/track_to_id.json', 'r') as f:
        track_to_id = json.load(f)
    with open('models/driver_track_means.json', 'r') as f:
        driver_track_means = json.load(f)
    with open('models/compound_track_means.json', 'r') as f:
        compound_track_means = json.load(f)
        
    circuit_means = np.load('models/circuit_means.npy', allow_pickle=True).item()
    
    return model, track_to_id, driver_track_means, compound_track_means, circuit_means

try:
    model, track_to_id, driver_track_means, compound_track_means, circuit_means = load_telemetry_engine()
except Exception as e:
    st.error(f"⚠️ Engine Offline: Missing model components. Run Day 12 first. Error: {e}")
    st.stop()

# Generate dynamic lists directly from your serialized dataset schemas
track_list = sorted(list(track_to_id.keys()))
driver_list = sorted(list(set([key.split('_')[1] for key in driver_track_means.keys()])))
compound_list = sorted(list(set([key.split('_')[1] for key in compound_track_means.keys()])))

# ─── 2. USER INTERFACE CONTROLS ──────────────────────────────────────────
st.sidebar.header("🔧 Telemetry Parameters")

col1, col2 = st.columns([1, 2])

with st.sidebar:
    selected_track = st.selectbox("🏁 Grand Prix Circuit", track_list, index=track_list.index("Monza") if "Monza" in track_list else 0)
    selected_driver = st.selectbox("🧑‍🚀 Driver Node", driver_list, index=driver_list.index("VER") if "VER" in driver_list else 0)
    selected_compound = st.selectbox("🛞 Tyre Compound", compound_list, index=compound_list.index("SOFT") if "SOFT" in compound_list else 0)
    
    st.markdown("---")
    
    lap_number = st.slider("🔄 Lap Number", min_value=1, max_value=80, value=15)
    tyre_life = st.slider("🔥 Tyre Age (Laps)", min_value=1, max_value=40, value=5)
    fuel_load = st.slider("⛽ Fuel Weight (kg)", min_value=1.0, max_value=110.0, value=85.0, step=1.0)

# ─── 3. FEATURE ENGINEERING & FIXED DELTA MATHEMATICS ───────────────────
track_id = track_to_id[selected_track]
tyre_log_penalty = np.log1p(tyre_life)
tyre_deg_idx = tyre_life * (fuel_load / 110.0)

dt_key = f"{selected_track}_{selected_driver}"
ct_key = f"{selected_track}_{selected_compound}"

# FIXED: Fallback baseline defaults contextually to the specific circuit pace mean instead of 80.0s!
circuit_base = circuit_means.get(selected_track, 85.0)
driver_base = driver_track_means.get(dt_key, circuit_base)
compound_base = compound_track_means.get(ct_key, circuit_base)

# Build inputs exactly matching Day 12's 8 structural parameters
input_features = pd.DataFrame([[
    track_id, 
    lap_number, 
    tyre_life, 
    tyre_log_penalty, 
    fuel_load, 
    tyre_deg_idx, 
    driver_base, 
    compound_base
]], columns=[
    'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
    'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
])

# Execute inference loop on the transformed delta target
predicted_delta = model.predict(input_features)[0]
final_lap_time_seconds = predicted_delta + driver_base

# ─── 4. RENDER TELEMETRY DASHBOARD ───────────────────────────────────────
def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.3f}"

with col1:
    st.subheader("⏱️ Predicted Lap Time")
    st.metric(
        label=f"{selected_driver} Predictions", 
        value=format_time(final_lap_time_seconds), 
        delta=f"{predicted_delta:+.3f}s vs Baseline", 
        delta_color="inverse"
    )

with col2:
    st.subheader("📊 Live Feature Vector Output")
    st.code(f"""
    [TRACK ENGINE STATUS: ONLINE]
    Track Base Average : {circuit_base:.3f}s
    Driver Delta Anchor: {driver_base:.3f}s
    Log tyre Deg Curve : {tyre_log_penalty:.3f}
    XGBoost Output Delta: {predicted_delta:+.3f}s
    """, language="markdown")
    
    st.progress(fuel_load / 110.0, text=f"Fuel Load: {fuel_load}kg")
    st.progress(tyre_life / 40.0, text=f"Tyre Age: {tyre_life} Laps")