import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import shap
import os

# === Load and prepare the data ===
df = pd.read_csv("TrainingDataV10.csv")

def convert_lap_time(lap_str):
    try:
        minutes, seconds = lap_str.split(":")
        return int(minutes) * 60 + float(seconds)
    except:
        return np.nan

if "Laguna Seca Lap Time (s)" in df.columns:
    df["Lap Time (s)"] = df["Laguna Seca Lap Time (s)"].apply(convert_lap_time)

# === Add Acceleration Curve before defining final feature set ===
df["Acceleration Curve"] = df["60-130 (s)"] / df["0-60 (s)"]

feature_cols = [
    '0-60 (s)', '1/4 Mile ET (s)', 'Trap Speed (mph)', '60-130 (s)',
    'Lateral G @ 120 mph', '100-0 Braking (ft)', 'Acceleration Curve'
]

# Drop rows with missing values in the selected features
df = df[feature_cols + ['Lap Time (s)']].dropna()

X = df[feature_cols].values
y = df['Lap Time (s)'].values

# === Normalize Inputs ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_v10.pkl")

# === Define Optuna Objective with K-Fold CV + SHAP ===
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    def build_model():
        model = keras.Sequential()
        model.add(layers.Input(shape=(X.shape[1],)))
        for i in range(n_layers):
            units = trial.suggest_int(f"units_l{i}", 32, 256, step=32)
            activation = trial.suggest_categorical(f"activation_l{i}", ["relu", "tanh"])
            model.add(layers.Dense(units, activation=activation))
            if dropout > 0:
                model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mae")
        return model

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = build_model()
        model.fit(
            X_train, y_train_fold,
            validation_data=(X_val, y_val_fold),
            batch_size=batch_size,
            epochs=100,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            ]
        )

        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val_fold, val_pred)
        val_scores.append(val_mae)

        # SHAP analysis (optional, for the first fold only)
        if len(val_scores) == 1:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_val[:50])  # sample for speed
            shap.summary_plot(shap_values, feature_names=feature_cols, show=False)

    return np.mean(val_scores)

# === Run Optuna Study ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("\n🔍 Best Hyperparameters:")
print(study.best_params)

# === Train Final Model on Full Dataset ===
def build_final_model(best_params):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))
    for i in range(best_params["n_layers"]):
        units = best_params[f"units_l{i}"]
        activation = best_params[f"activation_l{i}"]
        model.add(layers.Dense(units, activation=activation))
        if best_params["dropout"] > 0:
            model.add(layers.Dropout(best_params["dropout"]))
    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"]), loss="mae")
    return model

final_model = build_final_model(study.best_params)
final_model.fit(
    X_scaled, y,
    batch_size=study.best_params["batch_size"],
    epochs=100,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)]
)

final_model.save("LapTimePredictor_MLP_v10_best.h5")
print("\n✅ Final model saved as LapTimePredictor_MLP_v10_best.h5")