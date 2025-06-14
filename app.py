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

# Attribute sliders (must match features used in training)
st.subheader("Enter Player Attributes")
vision = st.slider("Vision", 0, 100, 70)
short_passing = st.slider("Short Passing", 0, 100, 75)
sprint_speed = st.slider("Sprint Speed", 0, 100, 80)
composure = st.slider("Composure", 0, 100, 78)
strength = st.slider("Strength", 0, 100, 85)
reactions = st.slider("Reactions", 0, 100, 82)
ball_control = st.slider("Ball Control", 0, 100, 75)
age = st.slider("Age", 16, 45, 24)  # Only if used in your model!

# Create input dataframe
input_df = pd.DataFrame([{
    "Vision": vision,
    "Short passing": short_passing,
    "Sprint speed": sprint_speed,
    "Composure": composure,
    "Strength": strength,
    "Reactions": reactions,
    "Ball control": ball_control,
    "Age": age  # Remove if not used
}])

# Predict and show results
if st.button("Predict Market Value"):
    log_value = model.predict(input_df)[0]
    market_value = np.exp(log_value)  # log to real value (assumes np.log(value) was used)
    st.success(f"💰 Estimated Market Value: €{market_value:,.0f}")
