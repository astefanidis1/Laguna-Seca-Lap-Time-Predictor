## 🧾 CHANGELOG – Laguna Seca Lap Time Predictor

### **Initial Version**

- Started with a dataset of 359 real-world cars
- Features included: 0-60, 1/4 mile time, trap speed, top speed, lateral G @ 120 mph, braking distance, drive type, lap time (in MM:SS format)

---

### **🔧 Data Preprocessing**

- Converted lap time strings to total seconds
- Encoded drive type as integers (RWD = 0, FWD = 1, AWD = 2)
- Applied median imputation to fill missing values

---

### **🧠 Feature Engineering**

- Added 4 key derived features:
  - `Speed Efficiency` = Trap Speed / 1/4 ET
  - `Composite Grip Index` = Lateral G / Braking
  - `Acceleration Curve` = 60-130 / 0-60
  - `Powerband Balance` = (Trap / Top Speed) × 60-130
- Later added:
  - `Weight (lb)`
  - `60-130 (s)`

---

### **🧼 Data Cleaning (via Residual Analysis)**

- Created residual analysis script
- Identified outlier cars based on z-score of prediction error
- Removed or corrected cars like Furai, Dart, Nova, Zonda R, etc.

---

### **📊 Model Training Progression**

| Version | Model            | Key Changes                                            | R^2    | RMSE      |
| ------- | ---------------- | ------------------------------------------------------ | ------ | --------- |
| v1      | XGBoost          | Baseline, 9 features                                   | ~0.93 | ~2.6s     |
| v2      | XGBoost          | +4 engineered features                                 | ~0.94 | ~2.2s     |
| v3      | XGBoost          | Cleaned dataset (355 rows)                             | 0.951  | 2.27s     |
| v4      | XGBoost          | Z-score grip & braking, removed redundant features     | 0.960  | 1.69s     |
| v5      | XGBoost + Optuna | Tuned 50 trials (best RMSE: 1.43s, depth 5, gamma 2.6) | —      | **1.43s** |

---

### **📐 Z-Score Scaling**

- Added Grip Z and Braking Z
  - Grip Z = (Lateral G – mean) / std
  - Braking Z = -(Braking – mean) / std
- Captures extremeness relative to dataset

---

### **🧪 Model Tuning with Optuna (v5)**

- Tuned parameters:
  - `n_estimators`: 100–500
  - `max_depth`: 3–8
  - `learning_rate`: 0.01–0.3
  - `gamma`: 0–5
  - `subsample`, `colsample_bytree`
- Best params found:
  {
    'n_estimators': 248,
    'max_depth': 5,
    'learning_rate': 0.15486,
    'subsample': 0.6099,
    'colsample_bytree': 0.8632,
    'gamma': 2.608
  }

---

### **⚠️ Correction Bonus System (v5+)**

- Post-prediction bonus applied **only** if:
  - Combined `Grip Z + Braking Z > 3.9`
  - Bonus = (Total Z – 3.9) × 0.6, capped at -2.0s
- Caps extreme stats (Z > 3.0)
- Gives light adjustment for prototype-level grip/braking
- ✅ Example: Velocita SP (Z = 3.96) → -0.036s bonus

---

### **📂 Final File Outputs**

- `sample_input_data.csv` – Cleaned dataset (public sample)
- `LapTimePredictor_XGBoost_v5.json` – Final model
- `lagunasecapyth.py` – Training script (non-Optuna)
- `lagunasecapyth_optuna.py` – Full Optuna tuner
- `predict_lap_time_v2.py` – Prediction script (with correction bonus)
- `residual_analysis_v2.py` – Outlier flagging

---

🎉 **This is the final version of the project — accurate, transparent, and fully documented.**
