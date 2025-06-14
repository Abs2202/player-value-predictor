import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="Football Player Value Predictor", layout="centered")
st.title("⚽ Football Player Market Value Predictor")
st.markdown("Estimate a player's transfer value using position-specific performance attributes.")

# Select position
position = st.selectbox("Select Player Position", ["Defender", "Midfielder", "Attacker"])

# Load correct model based on position
@st.cache_resource
def load_model(pos):
    if pos == "Defender":
        return joblib.load("xgb_model_real_defender.pkl")
    elif pos == "Midfielder":
        return joblib.load("xgb_model_real_midfielder.pkl")
    elif pos == "Attacker":
        return joblib.load("xgb_model_real_attacker.pkl")

model = load_model(position)

# Streamlit sliders (input fields)
playmaker_index = st.slider("Playmaker Index", 0.0, 100.0, 70.0)
finisher_index = st.slider("Finisher Index", 0.0, 100.0, 72.0)
defender_index = st.slider("Defender Index", 0.0, 100.0, 65.0)
speed_index = st.slider("Speed Index", 0.0, 100.0, 75.0)
technical_index = st.slider("Technical Index", 0.0, 100.0, 68.0)
physical_index = st.slider("Physical Index", 0.0, 100.0, 80.0)
composure = st.slider("Composure", 0, 100, 78)
age = st.slider("Age", 16, 45, 24)

# Input DataFrame (MUST match model training column names)
input_df = pd.DataFrame([{
    "Playmaker_Index": playmaker_index,
    "Finisher_Index": finisher_index,
    "Defender_Index": defender_index,
    "Speed_Index": speed_index,
    "Technical_Index": technical_index,
    "Physical_Index": physical_index,
    "Composure": composure,
    "Age": age
}])


# Predict and show results
if st.button("Predict Market Value"):
    log_value = model.predict(input_df)[0]
    market_value = np.exp(log_value)  # log to real value (assumes np.log(value) was used)
    st.success(f"💰 Estimated Market Value: €{market_value:,.0f}")
