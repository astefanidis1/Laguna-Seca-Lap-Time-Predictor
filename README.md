# 🏁 Laguna Seca Lap Time Predictor

Predict Laguna Seca lap times for real or fictional cars using real-world performance stats and a finely-tuned XGBoost regression model.

---

## 📦 What's Included

| File | Purpose |
|------|---------|
| `sample_input_data.csv` | Cleaned dataset sample |
| `Laguna_Seca_Model_Summary.md` | Full model and feature documentation |
| `predict_lap_time_v2.py` | Main prediction script (uses trained model) |
| `archive/lagunasecapyth_baseline.py` | Archived baseline trainer (non-Optuna) |
| `lagunasecapyth_optuna.py` | XGBoost trainer with Optuna tuning |
| `LapTimePredictor_XGBoost_v5.json` | Final trained model (v5) |
| `residual_analysis_v2.py` | Residual z-score based outlier tool |
| `CHANGELOG.md` | Full chronological list of improvements |

---

## 🚀 How to Predict a Lap Time

## 🧰 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

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
   - Correction bonus applied (if any)

---

## 🧠 Model Overview

- Model: `XGBoostRegressor` (v5, Optuna-tuned)
- Engineered features:
  - Composite Grip Index, Acceleration Curve, Powerband Balance
  - Grip Z and Braking Z for outlier scaling
- Bonus System:
  - If `Grip Z + Braking Z > 3.9`, a small penalty is subtracted to correct high-end track cars

---

## 💡 Example Prediction Output
```
Z-summary → Grip Z: 2.55, Braking Z: 1.41, Combined: 3.96
⚠️ Correction Bonus Applied: -0.036 sec (Extreme Grip + Braking Adjustment)
Predicted Lap Time: 1:21.108 (Total: 81.108 seconds)
```

---

## 🛠 How to Retrain the Model

If needed, retrain with:
```bash
python lagunasecapyth.py            # for quick retrain
python lagunasecapyth_optuna.py     # for hyperparameter tuning
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
- Lap time prediction tuned to extreme precision
- Correction bonus logic improves realism near prototype levels
- See `CHANGELOG.md` for every step of the process

---

Made with precision by Zandros. 🏎️
