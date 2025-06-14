import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="Football Player Value Predictor", layout="centered")

st.title("⚽ Football Player Market Value Predictor")
st.markdown("Estimate a player's transfer value using position-specific performance attributes.")

# ---------------------- Index Explanation ----------------------
with st.expander("ℹ️ What do the indexes mean?"):
    st.markdown("""
    **Each index is a simplified average of related football attributes. Here's how they're calculated:**

    - **Playmaker Index** = avg of *Vision, Short passing, Long passing, Ball control, Composure*  
    - **Finisher Index** = avg of *Finishing, Shot power, Att. Position, Reactions, Composure*  
    - **Defender Index** = avg of *Interceptions, Standing tackle, Sliding tackle, Defensive awareness, Strength*  
    - **Physical Index** = avg of *Stamina, Jumping, Strength, Aggression*  
    - **Technical Index** = avg of *Dribbling, Ball control, Curve, FK Accuracy, Vision, Composure*  
    - **Speed Index** = avg of *Acceleration, Sprint speed, Agility*  
    - **Reactions** and **Composure** are also used individually.

    You can either manually input these index values or let the app calculate them from raw attributes.
    """)

# ---------------------- Position Selector ----------------------
position = st.selectbox("Select Player Position", ["Defender", "Midfielder", "Attacker"])

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model(pos):
    if pos == "Defender":
        return joblib.load("xgb_model_real_defender.pkl")
    elif pos == "Midfielder":
        return joblib.load("xgb_model_real_midfielder.pkl")
    elif pos == "Attacker":
        return joblib.load("xgb_model_real_attacker.pkl")

model = load_model(position)

st.markdown("---")

# ---------------------- Input Mode ----------------------
input_mode = st.radio("How would you like to input values?", ["Manual Index Input", "Auto Calculate from Attributes"])

# ---------------------- Reactions & Composure ----------------------
reactions = st.slider("Reactions", 0, 100, 78)
composure = st.slider("Composure", 0, 100, 80)

# ---------------------- Manual Index Input ----------------------
if input_mode == "Manual Index Input":
    st.subheader("Enter Index Values Manually")
    playmaker_index = st.slider("Playmaker Index", 0.0, 100.0, 70.0)
    finisher_index = st.slider("Finisher Index", 0.0, 100.0, 72.0)
    defender_index = st.slider("Defender Index", 0.0, 100.0, 68.0)
    physical_index = st.slider("Physical Index", 0.0, 100.0, 75.0)
    technical_index = st.slider("Technical Index", 0.0, 100.0, 74.0)
    speed_index = st.slider("Speed Index", 0.0, 100.0, 77.0)

# ---------------------- Auto Calculate Indexes ----------------------
else:
    st.subheader("Input Raw Attributes (app calculates indexes)")

    # Playmaker
    vision = st.slider("Vision", 0, 100, 75)
    short_passing = st.slider("Short Passing", 0, 100, 78)
    long_passing = st.slider("Long Passing", 0, 100, 76)
    ball_control = st.slider("Ball Control", 0, 100, 77)

    # Finisher
    finishing = st.slider("Finishing", 0, 100, 70)
    shot_power = st.slider("Shot Power", 0, 100, 72)
    att_position = st.slider("Att. Position", 0, 100, 73)

    # Defender
    interceptions = st.slider("Interceptions", 0, 100, 68)
    standing_tackle = st.slider("Standing Tackle", 0, 100, 70)
    sliding_tackle = st.slider("Sliding Tackle", 0, 100, 65)
    def_awareness = st.slider("Def. Awareness", 0, 100, 67)
    strength = st.slider("Strength", 0, 100, 75)

    # Physical
    stamina = st.slider("Stamina", 0, 100, 74)
    jumping = st.slider("Jumping", 0, 100, 71)
    aggression = st.slider("Aggression", 0, 100, 69)

    # Technical
    dribbling = st.slider("Dribbling", 0, 100, 76)
    curve = st.slider("Curve", 0, 100, 70)
    fk_accuracy = st.slider("FK Accuracy", 0, 100, 68)

    # Speed
    acceleration = st.slider("Acceleration", 0, 100, 78)
    sprint_speed = st.slider("Sprint Speed", 0, 100, 79)
    agility = st.slider("Agility", 0, 100, 75)

    # Compute indexes
    playmaker_index = np.mean([vision, short_passing, long_passing, ball_control, composure])
    finisher_index = np.mean([finishing, shot_power, att_position, reactions, composure])
    defender_index = np.mean([interceptions, standing_tackle, sliding_tackle, def_awareness, strength])
    physical_index = np.mean([stamina, jumping, strength, aggression])
    technical_index = np.mean([dribbling, ball_control, curve, fk_accuracy, vision, composure])
    speed_index = np.mean([acceleration, sprint_speed, agility])

# ---------------------- Prediction ----------------------
st.markdown("---")
if st.button("Predict Market Value"):
    input_df = pd.DataFrame([{
        "Reactions": reactions,
        "Composure": composure,
        "Playmaker_Index": playmaker_index,
        "Finisher_Index": finisher_index,
        "Defender_Index": defender_index,
        "Physical_Index": physical_index,
        "Technical_Index": technical_index,
        "Speed_Index": speed_index
    }])
    log_value = model.predict(input_df)[0]
    market_value = np.exp(log_value)
    st.success(f"💰 Estimated Market Value: €{market_value:,.0f}")

