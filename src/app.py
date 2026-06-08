import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Widescreen tactical layout
st.set_page_config(page_title="F1 Pit-Wall Strategy Engine", layout="wide")

# ─── PREMIUM TELEMETRY THEME (CSS GLOW INJECTIONS) ──────────────────────────
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    h1, h2, h3 { 
        color: #ff1801 !important; 
        font-family: 'Helvetica Neue', sans-serif; 
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .stSlider label { color: #f0f6fc !important; font-size: 0.95rem; }
    
    /* Sleek Pit Wall Container */
    .pitwall-container {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-left: 5px solid #ff1801;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Strategy Delta Cards */
    .delta-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-top: 4px solid #00ffb9;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
    }
    .delta-val { font-size: 2rem; font-weight: 700; color: #ffffff; font-family: monospace; }
    .delta-lbl { color: #8b949e; font-size: 0.85rem; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_model():
    model_path = 'models/xgb_lap_predictor.json'
    if not os.path.exists(model_path):
        st.error("⚠️ Model engine not found! Ensure models/xgb_lap_predictor.json exists.")
        return None
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

model = load_saved_model()

if model:
    st.title("🏎️ F1 PIT-WALL STRATEGY COMMAND CENTER")
    st.markdown("##### INFRASTRUCTURE STATUS: ACTIVE | SIMULATION ENVIRONMENT: MONACO GP")
    st.write("---")
    
    # Grid layout splitting inputs and visualization panels
    col_inputs, col_charts = st.columns([1, 2])
    
    with col_inputs:
        st.markdown("<div class='pitwall-container'><h3> STINT VARIABLES</h3>", unsafe_allow_html=True)
        
        selected_driver = st.selectbox("DRIVER PROFILE:", ['VER', 'ALO', 'LEC', 'HAM', 'RUS', 'SAI', 'NOR', 'PIA', 'GAS', 'OCO'])
        
        # Current Stint Context
        current_lap = st.slider("CURRENT RACE LAP:", min_value=1, max_value=78, value=15)
        starting_fuel = st.slider("FUEL LOAD AT CURRENT LAP (kg):", min_value=5.0, max_value=110.0, value=85.0, step=0.5)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='pitwall-container'><h3> REAL-TIME CALCULATION</h3>", unsafe_allow_html=True)
        st.write("Simulating a 30-lap stint trajectory based on your variables:")
        
        # Quick metrics readout
        st.write("")
        st.markdown("""
            <div class='delta-card'>
                <div class='delta-lbl'>STRATEGY DEPLOYMENT STATUS</div>
                <div class='delta-val' style='color:#00ffb9;'>LIVE GRAPH GENERATED</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_charts:
        st.markdown("###  SIMULATED RACE STRATEGY TIMELINE")
        st.write("Instead of checking one lap, this projects your lap-by-lap tire degradation and changing weight over the entire next stint:")
        
        # ─── MULTI-COMPOUND STRATEGY SIMULATION ──────────────────────────────
        # We simulate the exact same stint window for Softs, Mediums, and Hards 
        # so the user can actually make a tactical comparison!
        fig = go.Figure()
        
        compounds_to_plot = [
            {'name': '🔴 SOFT', 'id': 'SOFT', 'color': '#ff1801', 'mult': 1.5},
            {'name': '🟡 MEDIUM', 'id': 'MEDIUM', 'color': '#ffcc00', 'mult': 1.0},
            {'name': '⚪ HARD', 'id': 'HARD', 'color': '#ffffff', 'mult': 0.6}
        ]
        
        stint_length = 30
        future_laps = np.arange(current_lap, min(current_lap + stint_length, 79))
        stint_age = np.arange(1, len(future_laps) + 1)
        
        # Simulating linear fuel burn over the stint (1.3kg per lap)
        sim_fuel = np.clip(starting_fuel - (stint_age * 1.3), a_min=0.5, a_max=None)
        
        for comp in compounds_to_plot:
            sim_deg = stint_age * comp['mult']
            
            sim_df = pd.DataFrame({
                'Driver': [selected_driver] * len(future_laps),
                'Compound': [comp['id']] * len(future_laps),
                'FuelLoadKg': sim_fuel,
                'TyreDegradationIndex': sim_deg
            })
            sim_df['Driver'] = sim_df['Driver'].astype('category')
            sim_df['Compound'] = sim_df['Compound'].astype('category')
            
            # Predict timeline
            predicted_stint_times = model.predict(sim_df)
            
            fig.add_trace(go.Scatter(
                x=future_laps, 
                y=predicted_stint_times,
                mode='lines',
                line=dict(color=comp['color'], width=3),
                name=comp['name']
            ))
            
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            xaxis=dict(title="Race Lap Number", gridcolor='#30363d'),
            yaxis=dict(title="Predicted Pace (Seconds)", gridcolor='#30363d'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=10, b=50),
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)