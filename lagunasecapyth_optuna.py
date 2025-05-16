import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore

# === Load and prepare the data ===
df = pd.read_csv("sample_input_data.csv")

def convert_lap_time(lap_str):
    try:
        minutes, seconds = lap_str.split(":")
        return int(minutes) * 60 + float(seconds)
    except:
        return np.nan

df["Lap Time (s)"] = df["Laguna Seca Lap Time (s)"].apply(convert_lap_time)
df["Drive Type Encoded"] = df["Drive Type"].map({'RWD': 0, 'FWD': 1, 'AWD': 2})

cols = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Drive Type Encoded', 'Weight (lb)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)'
]

df = df[cols + ['Lap Time (s)']].dropna()

# Impute and engineer features
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols + ['Lap Time (s)'])

df_imputed["Composite Grip Index"] = df_imputed["Lateral G @ 120 mph"] / df_imputed["100-0 Braking (ft)"]
df_imputed["Acceleration Curve"] = df_imputed["60-130 (s)"] / df_imputed["0-60 (s)"]
df_imputed["Powerband Balance"] = (df_imputed["Trap Speed (mph)"] / df_imputed["Top Speed (mph)"]) * df_imputed["60-130 (s)"]
df_imputed["Grip Z"] = zscore(df_imputed["Lateral G @ 120 mph"])
df_imputed["Braking Z"] = zscore(-df_imputed["100-0 Braking (ft)"])

X = df_imputed[[
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', 'Top Speed (mph)',
    'Drive Type Encoded', 'Weight (lb)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)',  # raw
    'Composite Grip Index', 'Acceleration Curve', 'Powerband Balance',
    'Grip Z', 'Braking Z'  # engineered
]]
y = df_imputed["Lap Time (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Optuna tuning ===
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# === Train final model with best params ===
print("Best Parameters:", study.best_params)

final_model = XGBRegressor(**study.best_params)
final_model.fit(X_train, y_train)

# Save model
final_model.save_model("LapTimePredictor_XGBoost_v5.json")
print("✅ Model saved as LapTimePredictor_XGBoost_v5.json")
