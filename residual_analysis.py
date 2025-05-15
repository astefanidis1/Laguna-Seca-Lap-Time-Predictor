
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# === Load trained model ===
model = XGBRegressor()
model.load_model("C:/Users/astef/OneDrive/Desktop/LapTimePredictor_XGBoost_v2.json")

# === Load full dataset ===
df = pd.read_csv("C:/Users/astef/Downloads/Lap Regression V2 - Sheet1.csv")

# === Convert lap time to seconds ===
def convert_lap_time(lap_str):
    try:
        minutes, seconds = lap_str.split(":")
        return int(minutes) * 60 + float(seconds)
    except:
        return np.nan

df["Lap Time (s)"] = df["Laguna Seca Lap Time (s)"].apply(convert_lap_time)
df['Drive Type Encoded'] = df['Drive Type'].map({'RWD': 0, 'FWD': 1, 'AWD': 2})

# === Select and clean key columns ===
keep_cols = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)', 'Drive Type Encoded',
    'Weight (lb)', '60-130 (s)', 'Lap Time (s)'
]

car_names = df['Car'] if 'Car' in df.columns else None
df = df[keep_cols]
df = df.dropna(subset=['Lap Time (s)'])  # only keep cars with real lap times

# === Impute missing values ===
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['Lap Time (s)'])), columns=keep_cols[:-1])

# === Add engineered features ===
df_imputed["Speed Efficiency"] = df_imputed["Trap Speed (mph)"] / df_imputed["1/4 Mile ET (s)"]
df_imputed["Composite Grip Index"] = df_imputed["Lateral G @ 120 mph"] / df_imputed["100-0 Braking (ft)"]
df_imputed["Acceleration Curve"] = df_imputed["60-130 (s)"] / df_imputed["0-60 (s)"]
df_imputed["Powerband Balance"] = (df_imputed["Trap Speed (mph)"] / df_imputed["Top Speed (mph)"]) * df_imputed["60-130 (s)"]

# === Predict lap times ===
features = df_imputed.columns
predicted_times = model.predict(df_imputed[features])

# === Compile results ===
df_results = df.copy()
df_results["Predicted Lap Time (s)"] = predicted_times
df_results["Prediction Error"] = df_results["Predicted Lap Time (s)"] - df_results["Lap Time (s)"]
df_results["Absolute Error"] = df_results["Prediction Error"].abs()
if car_names is not None:
    df_results["Car"] = car_names.values[:len(df_results)]

# === Sort by worst predictions ===
df_sorted = df_results.sort_values(by="Absolute Error", ascending=False)

# === Save the result to CSV ===
df_sorted.to_csv("C:/Users/astef/OneDrive/Desktop/lap_time_residuals.csv", index=False)
print("✅ Residual analysis saved as 'lap_time_residuals.csv' on your Desktop.")
