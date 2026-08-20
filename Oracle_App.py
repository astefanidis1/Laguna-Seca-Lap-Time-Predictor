import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === Load model and scaler ===
model = load_model("LapTimePredictor_MLP_v10_best.h5")
scaler = joblib.load("scaler_v10.pkl")

# === Page Config ===
st.set_page_config(page_title="The Oracle — Lap Time Predictor", layout="centered")
st.title("🏁 The Oracle — Predict Your Lap Time")
st.markdown("Estimate your car's lap time at **Laguna Seca** (other tracks coming soon).")

# === Sidebar: Track Selection ===
st.sidebar.header("Track Selector")
track = st.sidebar.selectbox("Choose Track", ["Laguna Seca", "Nürburgring (coming soon)", "Spa (coming soon)"])
if track != "Laguna Seca":
    st.sidebar.warning("Track support coming soon. Current model is trained only on Laguna Seca.")

# === Main Form ===
st.subheader("Enter Your Car's Specs")

with st.form("car_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        zero_to_sixty = st.number_input("0–60 mph (s)", min_value=1.5, max_value=10.0, value=3.5, step=0.1)
        quarter_mile_et = st.number_input("1/4 Mile ET (s)", min_value=7.0, max_value=15.0, value=11.2, step=0.1)
        trap_speed = st.number_input("Trap Speed (mph)", min_value=90, max_value=180, value=130, step=1)
    with col2:
        sixty_to_130 = st.number_input("60–130 mph (s)", min_value=3.0, max_value=15.0, value=7.6, step=0.1)
        lateral_g = st.number_input("Lateral G @ 120 mph", min_value=0.70, max_value=1.50, value=1.09, step=0.01)
        braking_dist = st.number_input("100–0 Braking (ft)", min_value=80.0, max_value=400.0, value=266.3, step=1.0)

    submitted = st.form_submit_button("Predict Lap Time")

# === Prediction Logic ===
if submitted:
    accel_curve = sixty_to_130 / zero_to_sixty

    features = [
        zero_to_sixty,
        quarter_mile_et,
        trap_speed,
        sixty_to_130,
        lateral_g,
        braking_dist,
        accel_curve
    ]

    input_scaled = scaler.transform([features])
    prediction = model.predict(input_scaled)[0][0]

    minutes = int(prediction // 60)
    seconds = prediction % 60

    st.success(f"🏁 Predicted Lap Time at {track}: **{minutes}:{seconds:06.3f}**")
    st.caption("Model prediction based on acceleration, grip, and braking performance.\n")