# Laguna Seca Lap Time Prediction Model (XGBoost v9)

## ✅ Model Type

* **XGBoost Regressor** (v9)

  * Tuned via **Optuna** with 100 trials
  * Monotonic constraints applied to enforce logical feature influence
  * Trained only on **high-quality data**:

    * Residual Z-score ≤ 3
    * Fewer than 3 missing values (pre-imputation)
  * All bonus/patch systems removed

### Best Parameters (v9):

```python
{
  'n_estimators': 209,
  'max_depth': 7,
  'learning_rate': 0.0791,
  'subsample': 0.8303,
  'colsample_bytree': 0.9085,
  'gamma': 3.5639
}
```

---

## 📊 Final Input Features (7 total)

| Feature               | Description             | Direction (Better When...) |
| --------------------- | ----------------------- | -------------------------- |
| `0-60 (s)`            | Time to 60 mph          | ↓ Lower                    |
| `1/4 Mile ET (s)`     | Quarter mile time       | ↓ Lower                    |
| `Trap Speed (mph)`    | Speed at end of 1/4     | ↑ Higher                   |
| `60-130 (s)`          | Time from 60 to 130 mph | ↓ Lower                    |
| `Lateral G @ 120 mph` | Cornering grip          | ↑ Higher                   |
| `100-0 Braking (ft)`  | Braking distance        | ↓ Lower                    |
| `Acceleration Curve`  | 60–130 / 0–60 ratio     | ↓ Optimal around 1.0–1.4   |

---

## 📈 Model Performance

* **Validation RMSE**: **1.64 seconds**
* Fully monotonic and physics-aligned
* Outperforms all previous versions in realism and ranking logic
* Reproduces lap time gaps between cars without clustering or flattening

---

## ▱ Monotonic Constraints Applied

| Feature              | Direction Enforced |
| -------------------- | ------------------ |
| `0–60 (s)`           | ↓ Faster = Better  |
| `1/4 Mile ET (s)`    | ↓ Faster = Better  |
| `Trap Speed`         | ↑ Higher = Better  |
| `60–130 (s)`         | ↓ Faster = Better  |
| `Lateral G`          | ↑ Higher = Better  |
| `Braking (ft)`       | ↓ Shorter = Better |
| `Acceleration Curve` | ↓ Lower = Better   |

---

## 🧠 Why This Version Is Final

* ✅ Most accurate predictions to date
* ✅ Realistic performance spread (hypercar vs street vs classic)
* ✅ Cleaned feature set and data make it easier to maintain
* ✅ No more stat inflation or overfitting from engineered features
* ✅ Fully reproducible using public-safe sample input

---

## 📄 Key Files

* `LapTimePredictor_XGBoost_v9.json` – Final trained model
* `lagunasecapyth_optuna.py` – Optuna trainer with constraints
* `predict_lap_time_v2.py` – Lap time predictor
* `sample_input_data.csv` – Public testing set
* `CHANGELOG.md` – Full model history

---

🏁 **This is the final v9 model: accurate, transparent, and battle-tested.**