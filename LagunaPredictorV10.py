import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# === Load trained model and scaler ===
model = load_model("LapTimePredictor_MLP_v10_best.h5")
scaler = joblib.load("scaler_v10.pkl")

# === Define your car's stats here ===
car = {
    '0-60 (s)': 3.5,
    '1/4 Mile ET (s)': 11.2,
    'Trap Speed (mph)': 130,
    '60-130 (s)': 7.6,
    'Lateral G @ 120 mph': 1.09,
    '100-0 Braking (ft)': 266.3
}

# === Define feature order — must match training ===
features = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)', 'Acceleration Curve'
]

# === Validate inputs ===
for key in features[:-1]:  # exclude Acceleration Curve
    if key not in car:
        raise ValueError(f"Missing input for: {key}")

# === Engineer features ===
car['Acceleration Curve'] = car['60-130 (s)'] / car['0-60 (s)']

# === Prepare input ===
input_df = pd.DataFrame([car])[features]
input_scaled = scaler.transform(input_df.values)

# === Predict lap time ===
predicted_time = model.predict(input_scaled)[0][0]

# === Format output ===
minutes = int(predicted_time // 60)
seconds = predicted_time % 60
print(f"🏁 Predicted Lap Time: {minutes}:{seconds:06.3f}")