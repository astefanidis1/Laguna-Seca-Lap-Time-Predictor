import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# === Load the trained model ===
model = XGBRegressor()
model.load_model("LapTimePredictor_XGBoost_v9.json")

# === Define your car's stats here ===
car = {
    '0-60 (s)': 3.5,
    '1/4 Mile ET (s)': 11.2,
    'Trap Speed (mph)': 130,
    '60-130 (s)': 7.6,
    'Lateral G @ 120 mph': 1.09,
    '100-0 Braking (ft)': 266.3
}

# === Engineer features ===
car['Acceleration Curve'] = car['60-130 (s)'] / car['0-60 (s)']

# === Define feature order — must match training ===
features = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)', 'Acceleration Curve'
]

# === Predict lap time ===
input_df = pd.DataFrame([car])[features]
predicted_time = model.predict(input_df)[0]

# === Format output ===
minutes = int(predicted_time // 60)
seconds = predicted_time % 60
print(f"Predicted Lap Time: {minutes}:{seconds:06.3f}")

import matplotlib.pyplot as plt
import xgboost as xgb

# Plot feature importances
xgb.plot_importance(model, importance_type='gain', title='Feature Importance by Gain')
plt.tight_layout()
plt.show()
