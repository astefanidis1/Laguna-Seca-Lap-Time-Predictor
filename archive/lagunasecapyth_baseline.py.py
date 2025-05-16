# === DEPRECATED TRAINING SCRIPT ===
# This is the older non-Optuna training pipeline.
# Kept for reference and comparison with Optuna-tuned version.

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import zscore

# === Step 1: Load the CSV ===
df = pd.read_csv("sample_input_data.csv")

# === Step 2: Convert lap times to seconds ===
def convert_lap_time(lap_str):
    try:
        minutes, seconds = lap_str.split(":")
        return int(minutes) * 60 + float(seconds)
    except:
        return np.nan

df["Lap Time (s)"] = df["Laguna Seca Lap Time (s)"].apply(convert_lap_time)
df["Drive Type Encoded"] = df["Drive Type"].map({'RWD': 0, 'FWD': 1, 'AWD': 2})

# === Step 3: Select columns ===
keep_cols = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Drive Type Encoded', 'Weight (lb)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)'
]

df = df[keep_cols + ['Lap Time (s)']]
df = df.dropna()

# === Step 4: Impute missing values ===
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=keep_cols + ['Lap Time (s)'])

# === Step 5: Engineer features ===
df_imputed["Composite Grip Index"] = df_imputed["Lateral G @ 120 mph"] / df_imputed["100-0 Braking (ft)"]
df_imputed["Acceleration Curve"] = df_imputed["60-130 (s)"] / df_imputed["0-60 (s)"]
df_imputed["Powerband Balance"] = (df_imputed["Trap Speed (mph)"] / df_imputed["Top Speed (mph)"]) * df_imputed["60-130 (s)"]
df_imputed["Grip Z"] = zscore(df_imputed["Lateral G @ 120 mph"])
df_imputed["Braking Z"] = zscore(-df_imputed["100-0 Braking (ft)"])  # lower is better

# === Step 6: Define final features ===
X = df_imputed[[
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Drive Type Encoded', 'Weight (lb)', '60-130 (s)',
    'Composite Grip Index', 'Acceleration Curve', 'Powerband Balance',
    'Grip Z', 'Braking Z'
]]
y = df_imputed["Lap Time (s)"]

# === Step 7: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 8: Train model with deeper trees ===
model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# === Step 9: Evaluate model ===
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# === Step 10: Save model ===
import os
os.makedirs("models", exist_ok=True)
model.save_model("models/LapTimePredictor_XGBoost_v4.json")
print("✅ Model saved as models/LapTimePredictor_XGBoost_v4.json")

