# 🏁 Laguna Seca Lap Time Predictor

Predict Laguna Seca lap times for real or fictional cars using real-world performance stats and a finely-tuned XGBoost regression model.

---

## 📦 What's Included

| File | Purpose |
|------|---------|
| `Lap Regression V3.csv` | Cleaned dataset with 356 cars |
| `Laguna_Seca_Model_Summary.md` | Full model and feature documentation (v6) |
| `predict_lap_time_v2.py` | Main prediction script (uses trained model) |
| `lagunasecapyth.py` | XGBoost trainer (basic) |
| `lagunasecapyth_optuna.py` | XGBoost trainer with Optuna tuning + monotonic constraints |
| `LapTimePredictor_XGBoost_v5.json` | Final trained model (v6) |
| `residual_analysis_v2.py` | Residual z-score based outlier tool |
| `CHANGELOG.md` | Full chronological list of improvements |

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
       'Top Speed (mph)': 243,
       'Lateral G @ 120 mph': 1.85,
       '100-0 Braking (ft)': 232.6,
       'Drive Type Encoded': 2,  # 0=RWD, 1=FWD, 2=AWD
       'Weight (lb)': 3637,
       '60-130 (s)': 4.9
   }
   ```
4. Save and run:
   ```bash
   python predict_lap_time_v2.py
   ```

5. The output will show:
   - Predicted time in MM:SS

---

## 🧠 Model Overview (v6)

- Model: `XGBoostRegressor` (v6, Optuna-tuned, monotonic constraints)
- Engineered features:
  - Composite Grip Index, Acceleration Curve, Powerband Balance
  - Track Dominance Index = (Lateral G ^ 2) / Braking Distance
- Removed bonus system — model now inherently understands grip/braking impact

---

## 💡 Example Prediction Output
```
Predicted Lap Time: 1:20.348 (Total: 80.348 seconds)
```

---

## 🛠 How to Retrain the Model

If needed, retrain with:
```bash
python lagunasecapyth.py            # for quick retrain
python lagunasecapyth_optuna.py     # for hyperparameter tuning (w/ constraints)
```
This uses the cleaned CSV and will regenerate `LapTimePredictor_XGBoost_v5.json`

---

## 📈 Residual Cleanup
To see which cars are hurting model accuracy the most:
```bash
python residual_analysis_v2.py
```
It saves `lap_time_residuals.csv` with prediction errors and z-scores.

---

## ✅ Final Notes
- Cleaned dataset: 356 valid cars
- v6 model is fully physics-aligned and logical
- Handles hypercars and prototypes without correction hacks
- See `CHANGELOG.md` for full version history

---

Made with precision by Zandros. 🏎️