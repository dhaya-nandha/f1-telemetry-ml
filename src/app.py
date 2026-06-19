import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import shap
import matplotlib.pyplot as plt
import tempfile
import re

# ─── APP CONFIGURATION & CUSTOM CSS ───────────────────────────────────
st.set_page_config(page_title="F1 Telemetry Predictive Engine", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
    <style>
        /* F1 Dark Theme Backgrounds */
        .stApp {
            background-color: #0E1117;
        }
        /* Custom F1 Red Accents for Headers */
        h1, h2, h3 {
            color: #FF1801 !important;
            font-family: 'Helvetica Neue', sans-serif;
        }
        /* Style the Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #1A1C23;
            border: 1px solid #2B2E35;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="metric-container"] label {
            color: #A0AEC0 !important;
            font-weight: 600;
        }
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111318;
            border-right: 2px solid #FF1801;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ─── PATH RESOLUTION & CACHED SHAP EXPLAINER ──────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

@st.cache_resource
def get_cached_explainer(_model_instance):
    """Caches the explainer tree so it doesn't rebuild on every slider drag."""
    return shap.TreeExplainer(_model_instance)

def generate_live_shap_plot(input_data, model_instance):
    """Generates the live waterfall plot cleanly."""
    explainer = get_cached_explainer(model_instance)
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Live Strategy Impact: Telemetry Weight Breakdown", color='#FF1801', pad=15, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ─── ARTIFACT HYDRATION & BUG FIX ─────────────────────────────────────
@st.cache_resource
def load_production_artifacts():
    # ─── XGBOOST 2.1+ / SHAP BUG FIX (WINDOWS SAFE) ───────────────────
    # 1. Read the raw JSON as pure text
    model_path = os.path.join(MODELS_DIR, 'xgb_lap_predictor.json')
    with open(model_path, 'r') as f:
        raw_data = f.read()
        
    # 2. Use Regex to find "base_score": "[-0.00995]" and cleanly strip the brackets
    clean_data = re.sub(r'"base_score":\s*"\[(.*?)\]"', r'"base_score": "\1"', raw_data)
        
    # 3. Create a temporary file, write the clean data, and explicitly CLOSE it 
    # so Windows doesn't block XGBoost from accessing it.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(tmp_fd, 'w') as tmp:
        tmp.write(clean_data)
        
    # 4. Load the perfectly clean model into XGBoost and delete the temp file
    model = xgb.XGBRegressor()
    model.load_model(tmp_path)
    os.remove(tmp_path)
    # ──────────────────────────────────────────────────────────────────
    
    with open(os.path.join(MODELS_DIR, 'driver_track_means.json'), 'r') as f:
        driver_track_means = json.load(f)
    with open(os.path.join(MODELS_DIR, 'compound_track_means.json'), 'r') as f:
        compound_track_means = json.load(f)
    with open(os.path.join(MODELS_DIR, 'track_to_id.json'), 'r') as f:
        track_to_id = json.load(f)
        
    circuit_means = np.load(os.path.join(MODELS_DIR, 'circuit_means.npy'), allow_pickle=True).item()
    return model, driver_track_means, compound_track_means, track_to_id, circuit_means

# >>> CRITICAL ADDITION: ACTUALLY LOAD THE MODEL AND ARTIFACTS <<<
try:
    model, driver_track_means, compound_track_means, track_to_id, circuit_means = load_production_artifacts()
except Exception as e:
    st.error(f"❌ Critical Error loading models: {e}")
    st.stop()

# ─── REAL-WORLD ALIGNED F1 CIRCUIT LAP MAPPING ────────────────────────
track_max_laps = {
    'Monaco': 78, 'Belgium': 44, 'Italy': 53, 'Singapore': 62, 'Japan': 53, 
    'Great Britain': 52, 'Brazil': 71, 'Austria': 71, 'Netherlands': 72, 
    'Hungary': 70, 'USA': 56, 'Australia': 58, 'Bahrain': 57, 'Saudi Arabia': 50, 
    'Azerbaijan': 51, 'Miami': 57, 'Spain': 66, 'Canada': 70, 'Qatar': 57, 
    'Mexico': 71, 'Las Vegas': 50, 'Abu Dhabi': 58
}

available_drivers = sorted(list(set([key.split('_')[1] for key in driver_track_means.keys() if '_' in key])))

# ─── SIDEBAR UI ───────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
st.sidebar.markdown("## Race Strategy Setup")

st.sidebar.markdown("### 📍 Location & Driver")
selected_circuit = st.sidebar.selectbox("Select Circuit", list(track_to_id.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", available_drivers)

st.sidebar.markdown("### ⚙️ Car Setup & Phase")
selected_compound = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])

max_laps_for_track = track_max_laps.get(selected_circuit, 70)
lap_number = st.sidebar.slider(f"Current Lap Number", 1, max_laps_for_track, min(45, max_laps_for_track))

tyre_life = st.sidebar.slider("Current Tyre Age (Laps)", 1, 50, 11)
fuel_load = st.sidebar.slider("Estimated Fuel Load (Kg)", 0, 110, 46)

# ─── STATE CALCULATION & INFERENCE ────────────────────────────────────
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
    'Circuit_ID': circuit_id, 'LapNumber': lap_number, 'TyreLife': tyre_life,
    'Tyre_Log_Penalty': np.log1p(tyre_life), 'FuelLoadKg': fuel_load,
    'TyreDegradationIndex': tyre_deg_index, 'Driver_Track_Baseline': driver_baseline,
    'Compound_Track_Baseline': compound_baseline
}])

features_signature = [
    'Circuit_ID', 'LapNumber', 'TyreLife', 'Tyre_Log_Penalty', 
    'FuelLoadKg', 'TyreDegradationIndex', 'Driver_Track_Baseline', 'Compound_Track_Baseline'
]
input_payload = input_payload[features_signature]

predicted_delta = model.predict(input_payload)[0]
final_predicted_time = driver_baseline + predicted_delta

def format_lap_time(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes:02d}:{rem_seconds:06.3f}"

# ─── MAIN DASHBOARD LAYOUT ────────────────────────────────────────────
st.title("PIT WALL INTELLIGENCE")
st.markdown("Real-time predictive telemetry engine. Adjust sidebar parameters to recalculate strategy.")

# Top Row Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🏁 Predicted Absolute Pace", value=format_lap_time(final_predicted_time))
with col2:
    st.metric(label="⏱️ AI Pace Delta (vs Normal)", value=f"{predicted_delta:+.3f} s", delta=f"{predicted_delta:.3f} s", delta_color="inverse")
with col3:
    st.metric(label="🧠 Track Normal Pace", value=f"{driver_baseline:.3f} s")

if "Fallback" in baseline_status:
    st.warning(baseline_status)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for visual organization
tab1, tab2 = st.tabs(["🏎️ LIVE SHAP PHYSICS TELEMETRY", "📊 RAW ALGORITHM PAYLOAD"])

with tab1:
    with st.spinner("Calculating live physics telemetry..."):
        shap_fig = generate_live_shap_plot(input_payload, model)
        st.pyplot(shap_fig)

with tab2:
    st.markdown("### Matrix View: Features mapped for XGBoost Engine")
    st.dataframe(input_payload, use_container_width=True)