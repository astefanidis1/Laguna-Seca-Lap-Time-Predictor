# Laguna Seca Lap Time Prediction Model (XGBoost v5)

## ✅ Model Type
- **XGBoost Regressor**
  - Final Parameters (tuned via Optuna):
    ```
    {
      'n_estimators': 248,
      'max_depth': 5,
      'learning_rate': 0.15486,
      'subsample': 0.6099,
      'colsample_bytree': 0.8632,
      'gamma': 2.608
    }
    ```
- **Evaluation**:
  - RMSE: **1.43 seconds** (lowest achieved)
  - 5-Fold Cross-Validation used in tuning

---

## 📊 Input Features (14 total)

| Feature | Description | Direction (Better When...) |
|--------|-------------|-----------------------------|
| `0-60 (s)` | Time to 60 mph | ↓ Lower |
| `1/4 Mile ET (s)` | Quarter mile time | ↓ Lower |
| `Trap Speed (mph)` | Speed at end of 1/4 | ↑ Higher |
| `Top Speed (mph)` | Max speed | ↑ Higher |
| `Drive Type Encoded` | 0 = RWD, 1 = FWD, 2 = AWD | Auto-handled |
| `Weight (lb)` | Curb weight | ↓ Lower |
| `60-130 (s)` | Time from 60 to 130 mph | ↓ Lower |
| `Lateral G @ 120 mph` | Cornering grip | ↑ Higher |
| `100-0 Braking (ft)` | Braking distance | ↓ Lower |
| `Composite Grip Index` | Lateral G / Braking ft | ↑ Higher |
| `Acceleration Curve` | 60–130 / 0–60 | ~ Ideal near 1.0–1.4 |
| `Powerband Balance` | (Trap / Top Speed) × 60–130 | Context-dependent |
| `Grip Z` | Z-score of Lateral G | ↑ Higher |
| `Braking Z` | Negative Z-score of Braking (lower is better) | ↑ Higher |

---

## 🧠 Key Learnings
- Most influential factors:
  - `Trap Speed`
  - `Lateral G`
  - `0–60 time`
  - `Weight`
  - `Braking Z` when extreme
- Pure grip/braking is underweighted in typical models — addressed here using:
  - Z-scores for outlier scaling
  - Bonus correction (see below)

---

## ⚠️ Correction Bonus System (v5+)

- Applied **after** model prediction, only for extreme cases:
  - `Grip Z + Braking Z > 3.9`
  - Bonus = `(Total Z - 3.9) × 0.6`, max -2.0s
  - Helps fix prototype-level cornering anomalies

Example:
```
Grip Z = 2.55, Braking Z = 1.41 → Z = 3.96
Correction: (3.96 - 3.9) × 0.6 = -0.036s
```

---

## 🧪 How to Use

1. Define a car dictionary with all required stats (see README).
2. Run `predict_lap_time_v2.py`.
3. It returns the predicted lap time with optional correction bonus applied.

---

## 🧾 What to Save

- `sample_input_data.csv` – Public sample of cleaned dataset
- `LapTimePredictor_XGBoost_v5.json` – Final trained model
- `lagunasecapyth.py` – Manual trainer (non-Optuna)
- `lagunasecapyth_optuna.py` – Full Optuna tuner
- `predict_lap_time_v2.py` – Main prediction script (w/ bonus)
- `residual_analysis_v2.py` – Error analysis script
- `CHANGELOG.md` – Full documentation of process and improvements
