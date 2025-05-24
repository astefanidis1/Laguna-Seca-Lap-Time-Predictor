# Laguna Seca Lap Time Prediction Model (XGBoost v6)

## ✅ Model Type
- **XGBoost Regressor** (v6)
  - Tuned via **Optuna** with 50 trials
  - Monotonic constraints applied to enforce logical feature influence
  - **Correction bonus system removed** — model now learns grip/braking importance inherently

### Best Parameters (v6):
```python
{
  'n_estimators': 482,
  'max_depth': 6,
  'learning_rate': 0.0664,
  'subsample': 0.7211,
  'colsample_bytree': 0.6036,
  'gamma': 3.4919
}
```

---

## 📊 Input Features (13 total)

| Feature | Description | Direction (Better When...) |
|--------|-------------|-----------------------------|
| `0-60 (s)` | Time to 60 mph | ↓ Lower |
| `1/4 Mile ET (s)` | Quarter mile time | ↓ Lower |
| `Trap Speed (mph)` | Speed at end of 1/4 | ↑ Higher |
| `Top Speed (mph)` | Max speed | ↑ Higher |
| `Drive Type Encoded` | 0 = RWD, 1 = FWD, 2 = AWD | Neutral |
| `Weight (lb)` | Curb weight | ↓ Lower |
| `60-130 (s)` | Time from 60 to 130 mph | ↓ Lower |
| `Lateral G @ 120 mph` | Cornering grip | ↑ Higher |
| `100-0 Braking (ft)` | Braking distance | ↓ Lower |
| `Composite Grip Index` | Lateral G / Braking ft | ↑ Higher |
| `Acceleration Curve` | 60–130 / 0–60 | Ideal around 1.0–1.4 |
| `Powerband Balance` | (Trap / Top Speed) × 60–130 | Context-dependent |
| `Track Dominance Index` | (Lateral G ^ 2) / Braking ft | ↑ Higher |

---

## 📈 Model Performance

- **RMSE**: **1.63 seconds** (5-fold CV average)
- **Correction Bonus System Removed** — no longer needed
- Predicts prototype-level cars naturally, without hacky adjustments

---

## 📐 Monotonic Constraints Applied

| Feature                  | Direction Enforced |
|--------------------------|--------------------|
| `0–60 (s)`               | ↓ Faster = Better  |
| `1/4 Mile ET (s)`        | ↓ Faster = Better  |
| `Trap Speed`             | ↑ Higher = Better  |
| `Top Speed`              | ↑ Higher = Better  |
| `Drive Type Encoded`     | Neutral            |
| `Weight`                 | ↓ Lower = Better   |
| `60–130 (s)`             | ↓ Faster = Better  |
| `Lateral G`              | ↑ Higher = Better  |
| `Braking (ft)`           | ↓ Shorter = Better |
| `Composite Grip Index`   | ↑ Higher = Better  |
| `Acceleration Curve`     | ↓ Optimal Range    |
| `Powerband Balance`      | Neutral            |
| `Track Dominance Index`  | ↑ Higher = Better  |

---

## 🧠 Why This Version Is Better

- **No bonus patches** — everything learned directly
- **Handles extreme hypercars** like Ducati Evoluzione R with perfect logical ordering
- **Cleaner, smaller feature set (13 total)**
- **Physics-aligned logic** via Track Dominance Index and monotonic constraints

---

## 🧾 What to Save

- `Lap Regression V3.csv` – Cleaned dataset
- `LapTimePredictor_XGBoost_v5.json` – Final trained model (v6)
- `lagunasecapyth_optuna.py` – Full Optuna tuner (with constraints)
- `predict_lap_time_v2.py` – Main predictor script
- `CHANGELOG.md` – Full version history and upgrades

---

🏁 **This is the final v6 model — fast, fair, and physics-aware.**