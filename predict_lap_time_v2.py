import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# === Load the trained model ===
model = XGBRegressor()
model.load_model("LapTimePredictor_XGBoost_v5.json")

# === Define your car's stats here ===
car = {
    '0-60 (s)': 2.4,
    '1/4 Mile ET (s)': 9.6,
    'Trap Speed (mph)': 150,
    'Top Speed (mph)': 243,
    'Lateral G @ 120 mph': 1.85,
    '100-0 Braking (ft)': 232.6,
    'Drive Type Encoded': 2,  # RWD=0, FWD=1, AWD=2
    'Weight (lb)': 3637,
    '60-130 (s)': 4.9
}

# === Dataset z-score stats ===
GRIP_MEAN = 1.2391
GRIP_STD = 0.3063
BRAKE_MEAN = 273.47
BRAKE_STD = 45.07

# === Engineer features ===
car['Composite Grip Index'] = car['Lateral G @ 120 mph'] / car['100-0 Braking (ft)']
car['Acceleration Curve'] = car['60-130 (s)'] / car['0-60 (s)']
car['Powerband Balance'] = (car['Trap Speed (mph)'] / car['Top Speed (mph)']) * car['60-130 (s)']
car['Grip Z'] = (car['Lateral G @ 120 mph'] - GRIP_MEAN) / GRIP_STD
car['Braking Z'] = -(car['100-0 Braking (ft)'] - BRAKE_MEAN) / BRAKE_STD  # lower is better

# === Define feature order — must match training ===
features = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Drive Type Encoded', 'Weight (lb)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)',
    'Composite Grip Index', 'Acceleration Curve', 'Powerband Balance',
    'Grip Z', 'Braking Z'
]

# === Predict lap time ===
input_df = pd.DataFrame([car])[features]
predicted_time = model.predict(input_df)[0]

# === Optional Z-Aware Correction Bonus ===
grip_z = car['Grip Z']
brake_z = car['Braking Z']

# Cap each z-score to avoid runaway bonuses
effective_grip_z = min(grip_z, 3.0)
effective_brake_z = min(brake_z, 3.0)
total_z = effective_grip_z + effective_brake_z

# Print Z-score summary
print(f"Z-summary → Grip Z: {grip_z:.2f}, Braking Z: {brake_z:.2f}, Combined: {total_z:.2f}")

# Apply bonus only for truly elite combined performance
if total_z > 3.9:
    bonus = (total_z - 3.9) * 0.6
    predicted_time -= min(bonus, 2.0)
    print(f"⚠️ Correction Bonus Applied: -{min(bonus, 2.0):.3f} sec (Extreme Grip + Braking Adjustment)")

# === Format output ===
minutes = int(predicted_time // 60)
seconds = predicted_time % 60
print(f"Predicted Lap Time: {minutes}:{seconds:06.3f} (Total: {predicted_time:.3f} seconds)")
