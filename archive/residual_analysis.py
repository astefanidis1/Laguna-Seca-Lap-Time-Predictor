# Residual Analysis Script (Zandros Projects)
# Evaluates prediction errors across dataset using latest model (v7 or v8)

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# === Load model ===
model = XGBRegressor()
model.load_model("LapTimePredictor_XGBoost_v7.json")

# === Load dataset ===
df = pd.read_csv("Lap Regression V4.csv")

# === Convert lap time to seconds ===
def convert_lap_time(lap_str):
    try:
        minutes, seconds = lap_str.split(":")
        return int(minutes) * 60 + float(seconds)
    except:
        return np.nan

df["Lap Time (s)"] = df["Laguna Seca Lap Time (s)"].apply(convert_lap_time)

# === Select core features (latest model input) ===
keep_cols = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)',
    '60-130 (s)', 'Lateral G @ 120 mph', '100-0 Braking (ft)',
    'Lap Time (s)'
]

# === Drop missing values on target ===
df = df[keep_cols].dropna(subset=['Lap Time (s)'])

# === Impute missing input values ===
imputer = SimpleImputer(strategy='median')
X_base = df.drop(columns=['Lap Time (s)'])
X_imputed = pd.DataFrame(imputer.fit_transform(X_base), columns=X_base.columns)

# === Add engineered feature: Acceleration Curve ===
X_imputed["Acceleration Curve"] = X_imputed["60-130 (s)"] / X_imputed["0-60 (s)"]

# === Final feature order (must match model) ===
feature_cols = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)',
    '60-130 (s)', 'Lateral G @ 120 mph', '100-0 Braking (ft)',
    'Acceleration Curve'
]

# === Predict lap times ===
y_actual = df["Lap Time (s)"].values
y_pred = model.predict(X_imputed[feature_cols])

# === Calculate residuals ===
df_results = df.copy()
df_results["Predicted Lap Time (s)"] = y_pred
df_results["Prediction Error"] = y_pred - y_actual
df_results["Absolute Error"] = np.abs(df_results["Prediction Error"])

# === Sort and export ===
df_sorted = df_results.sort_values(by="Absolute Error", ascending=False)
df_sorted.to_csv("lap_time_residuals.csv", index=False)
print("✅ Residual analysis saved as 'lap_time_residualsV2.csv'")