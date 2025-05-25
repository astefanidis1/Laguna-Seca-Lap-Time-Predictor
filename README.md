# 🏁 Laguna Seca Lap Time Predictor

Predict Laguna Seca lap times for real or fictional cars using real-world performance stats and a finely-tuned XGBoost regression model.

---

## 📆 What's Included

| File                               | Purpose                                         |
| ---------------------------------- | ----------------------------------------------- |
| `sample_input_data.csv`            | Public sample data users can try predictions on |
| `Laguna_Seca_Model_Summary.md`     | Full model and feature documentation            |
| `predict_lap_time_v2.py`           | Main prediction script                          |
| `lagunasecapyth_optuna.py`         | XGBoost training script with Optuna tuning      |
| `LapTimePredictor_XGBoost_v9.json` | Final model (locked)                            |
| `residual_analysis.py`             | Residual z-score + missing data filtering tool  |
| `CHANGELOG.md`                     | Full chronological list of improvements         |

---

## 🚀 How to Predict a Lap Time

1. Open `predict_lap_time_v2.py`
2. Scroll to the `car = { ... }` block
3. Replace the values with your car's stats:

   ```python
   car = {
       '0-60 (s)': 2.4,
       '1/4 Mile ET (s)': 9.6,
       'Trap Speed (mph)': 150,
       '60-130 (s)': 4.9,
       'Lateral G @ 120 mph': 1.85,
       '100-0 Braking (ft)': 232.6
   }
   ```
4. Save and run:

   ```bash
   python predict_lap_time_v2.py
   ```
5. The output will show:

   * Predicted time in `MM:SS.mmm`

---

## 🧠 Model Overview (v9)

* Model: `XGBoostRegressor`, Optuna-tuned with monotonic constraints
* Final validation RMSE: **1.64 seconds**
* Feature Set:

  * Core stats: 0–60, ¼ mile ET, trap speed, 60–130, lateral G, braking distance
  * One engineered feature: `Acceleration Curve` (60–130 / 0–60)
* Removed:

  * Weight, top speed, drive type (redundant or misleading)
  * Engineered noise features (e.g., Grip Index, Powerband Balance)
* Trained only on high-quality rows with:

  * Prediction Z-score ≤ 3
  * Fewer than 3 missing features

---

## 💡 Example Output

```
Predicted Lap Time: 1:24.030 (Total: 84.030 seconds)
```

---

## 🛠 How to Retrain the Model

If needed, retrain with:

```bash
python lagunasecapyth_optuna.py
```

This will regenerate:

```
LapTimePredictor_XGBoost_v9.json
```

---

## 📊 Residual Analysis + Filtering

To analyze model error and identify noisy training rows:

```bash
python residual_analysis.py
```

It outputs:

* `lap_time_residuals.csv`: model predictions, errors, and Z-scores
* Use this to identify and remove high-residual rows

---

## ✅ Final Notes

* Final model: **v9** — clean, realistic, and locked in
* Built only from trusted data — no anchor laps or hardcoded targets
* Sample file provided (`sample_input_data.csv`) for safe demo use
* Full version history in `CHANGELOG.md`

---

Made with precision by Zandros. 🏎️