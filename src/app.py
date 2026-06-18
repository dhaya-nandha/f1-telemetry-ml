import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import shap
import matplotlib.pyplot as plt

# ─── PATH RESOLUTION ──────────────────────────────────────────────────
# Since app.py stays in the 'src' folder, we locate 'models' one level up
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# ─── EMBEDDED LIVE SHAP VISUALIZER FUNCTION ───────────────────────────
def generate_live_shap_plot(input_data, model_instance):
    """
    Calculates Shapley values on the fly for current dashboard states
    and returns a clean matplotlib figure layout to display.
    """
    explainer = shap.TreeExplainer(model_instance)
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    
    # Render the modern SHAP waterfall layout directly into the axis
    shap.plots.waterfall(shap_values[0], show=False)
    
    plt.title("Live Strategy Impact: Telemetry Weight Breakdown", color='white', pad=15, fontsize=12)
    plt.tight_layout()
    return fig

# ─── ARTIFACT HYDRATION ───────────────────────────────────────────────
@st.cache_resource
def load_production_artifacts():
    """Hydrate all serialized tracking maps and model binaries into memory once."""
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODELS_DIR, 'xgb_lap_predictor.json'))
    
    with open(os.path.join(MODELS_DIR, 'driver_track_means.json'), 'r') as f:
        driver_track_means = json.load(f)
    with open(os.path.join(MODELS_DIR, 'compound_track_means.json'), 'r') as f:
        compound_track_means = json.load(f)
    with open(os.path.join(MODELS_DIR, 'track_to_id.json'), 'r') as f:
        track_to_id = json.load(f)
        
    circuit_means = np.load(os.path.join(MODELS_DIR, 'circuit_means.npy'), allow_pickle=True).item()
    return model, driver_track_means, compound_track_means, track_to_id, circuit_means

try:
    model, driver_track_means, compound_track_means, track_to_id, circuit_means = load_production_artifacts()
except Exception as e:
    st.error(f"❌ Critical Error: Production artifacts missing. Looking in: {MODELS_DIR}")
    st.stop()

# ─── REAL-WORLD F1 CIRCUIT LAP MAPPING ────────────────────────────────
track_max_laps = {
    'Monaco': 78, 'Spa': 44, 'Monza': 53, 'Singapore': 62, 'Suzuka': 53, 
    'Silverstone': 52, 'Interlagos': 71, 'Red Bull Ring': 71, 'Zandvoort': 72, 
    'Hungaroring': 70, 'Circuit of the Americas': 56, 'Albert Park': 58,
    'Bahrain': 57, 'Jeddah': 50, 'Baku': 51, 'Miami': 57, 'Catalunya': 66,
    'Montreal': 70, 'Qatar': 57, 'Mexico City': 71, 'Las Vegas': 50, 'Abu Dhabi': 58
}

st.title("🏎️ F1 Pit Wall Strategy & Telemetry Engine")
st.markdown("---")

available_drivers = sorted(list(set([key.split('_')[1] for key in driver_track_means.keys() if '_' in key])))

# ─── SIDEBAR: INPUT CONTROL CONFIGURATIONS ────────────────────────────
st.sidebar.header("🎯 Race Strategy Parameters")

selected_circuit = st.sidebar.selectbox("Select Circuit", list(track_to_id.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", available_drivers)
selected_compound = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])

max_laps_for_track = track_max_laps.get(selected_circuit, 70)
lap_number = st.sidebar.slider(f"Current Lap Number (Max {max_laps_for_track})", 1, max_laps_for_track, min(45, max_laps_for_track))

tyre_life = st.sidebar.slider("Current Tyre Age (Laps)", 1, 50, 11)
fuel_load = st.sidebar.slider("Estimated Fuel Load (Kg)", 0, 110, 46)

# ─── STATE CALCULATION & ALIGNMENT ENGINE ─────────────────────────────
circuit_id = track_to_id[selected_circuit]
dt_key = f"{selected_circuit}_{selected_driver}"
ct_key = f"{selected_circuit}_{selected_compound}"

base_track_pace = circuit_means.get(selected_circuit, 85.0)
compound_baseline = compound_track_means.get(ct_key, base_track_pace)

if dt_key in driver_track_means:
    driver_baseline = driver_track_means[dt_key]
    baseline_status = "✅ Driver-Specific Baseline Used"
else:
    driver_baseline = base_track_pace
    baseline_status = "⚠️ Fallback: Generic Circuit Pace"

compound_multipliers = {'SOFT': 1.5, 'MEDIUM': 1.0, 'HARD': 0.6}
tyre_deg_index = tyre_life * compound_multipliers.get(selected_compound, 1.0)

input_payload = pd.DataFrame([{
    'Circuit_ID': circuit_id,
    'LapNumber': lap_number,
    'TyreLife': tyre_life,
    'Tyre_Log_Penalty': np.log1p(tyre_life),
    'FuelLoadKg': fuel_load,
    'TyreDegradationIndex': tyre_deg_index, 
    'Driver_Track_Baseline': driver_baseline,
    'Compound_Track_Baseline': compound_baseline
}])

features_signature = [
    'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
    'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
]
input_payload = input_payload[features_signature]

# ─── LIVE INFERENCE EXECUTION ─────────────────────────────────────────
predicted_delta = model.predict(input_payload)[0]
final_predicted_time = driver_baseline + predicted_delta

def format_lap_time(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes:02d}:{rem_seconds:06.3f}"

# ─── METRICS DISPLAY INTERFACE ────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="🤖 Predicted Absolute Lap Time", value=format_lap_time(final_predicted_time))
with col2:
    st.metric(label="⏱️ Model Pace Delta", value=f"{predicted_delta:+.3f} s", delta=f"{predicted_delta:.3f} s", delta_color="inverse")
with col3:
    st.metric(label="🧠 Track Baseline", value=f"{driver_baseline:.3f} s")
    if "Fallback" in baseline_status:
        st.caption(f"_{baseline_status}_")

st.markdown("---")

# ─── LIVE SHAP VISUALIZATION INTEGRATION ─────────────────────────────
st.subheader("🏎️ Live Strategy Impact: Telemetry Weight Breakdown")
st.markdown("This chart updates in real-time. It opens the black box of the AI to show exactly how many seconds were gained or lost due to track position, tyre degradation, and fuel weight.")

with st.spinner("Calculating live physics telemetry..."):
    # Render using the model object directly inside memory
    shap_fig = generate_live_shap_plot(input_payload, model)
    st.pyplot(shap_fig)

st.markdown("---")
st.subheader("📊 Raw Payload Context (Model Inputs)")
st.dataframe(input_payload, use_container_width=True)