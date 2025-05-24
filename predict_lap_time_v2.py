import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# === Load the trained model ===
model = XGBRegressor()
model.load_model("LapTimePredictor_XGBoost_v5.json")

# === Define your car's stats here ===
car = {
    '0-60 (s)': 2.2,
    '1/4 Mile ET (s)': 9.0,
    'Trap Speed (mph)': 161,
    'Top Speed (mph)': 232,
    'Drive Type Encoded': 2,  # RWD=0, FWD=1, AWD=2
    'Weight (lb)': 2774,
    '60-130 (s)': 3.8,
    'Lateral G @ 120 mph': 1.88,
    '100-0 Braking (ft)': 212.9
}

# === Engineer features ===
car['Composite Grip Index'] = car['Lateral G @ 120 mph'] / car['100-0 Braking (ft)']
car['Acceleration Curve'] = car['60-130 (s)'] / car['0-60 (s)']
car['Powerband Balance'] = (car['Trap Speed (mph)'] / car['Top Speed (mph)']) * car['60-130 (s)']
car['Track Dominance Index'] = (car['Lateral G @ 120 mph'] ** 2) / car['100-0 Braking (ft)']

# === Define feature order — must match training ===
features = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Drive Type Encoded', 'Weight (lb)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)',
    'Composite Grip Index', 'Acceleration Curve', 'Powerband Balance',
    'Track Dominance Index'
]

# === Predict lap time ===
input_df = pd.DataFrame([car])[features]
predicted_time = model.predict(input_df)[0]

# === Format output ===
minutes = int(predicted_time // 60)
seconds = predicted_time % 60
print(f"Predicted Lap Time: {minutes}:{seconds:06.3f} (Total: {predicted_time:.3f} seconds)")

# === Plot Global Feature Importances ===
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("XGBoost Global Feature Importances")
plt.tight_layout()
plt.show()
