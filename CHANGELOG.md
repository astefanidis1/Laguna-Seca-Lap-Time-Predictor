## 🧾 CHANGELOG – Laguna Seca Lap Time Predictor

### **v6 (May 23, 2025)**

- ⭐ Major model logic upgrade and cleanup
- ❌ Removed unused and underperforming features:
  - `Grip Z`
  - `Braking Z`
- ✅ Added smarter engineered feature:
  - `Track Dominance Index` = (Lateral G ^ 2) / Braking Distance
- ✅ Applied monotonic constraints to ensure logical behavior:
  - e.g., More grip = better, more weight = worse, more trap speed = better
- ✅ Removed correction bonus system entirely — no more post-prediction hacks
- 🧠 Model now inherently learns what makes a car fast, even at the extreme high end
- ✅ Retrained using cleaned dataset (`Lap Regression V3.csv`)
- 🏁 Fixed core issue where better hypercars (like Ducati V4 Evoluzione R) were predicted slower than inferior ones
- 🎯 Achieved new best RMSE: **1.63s**

---

### **🔧 Data Preprocessing**

- Converted lap time strings to total seconds
- Encoded drive type as integers (RWD = 0, FWD = 1, AWD = 2)
- Applied median imputation to fill missing values

---

### **🧠 Feature Engineering**

- Added 5 key derived features:
  - `Speed Efficiency` = Trap Speed / 1/4 ET
  - `Composite Grip Index` = Lateral G / Braking
  - `Acceleration Curve` = 60-130 / 0-60
  - `Powerband Balance` = (Trap / Top Speed) × 60-130
  - `Track Dominance Index` = (Lateral G ^ 2) / Braking ft
- Removed:
  - `Grip Z`, `Braking Z` (previously used for bonus system)

---

### **🧼 Data Cleaning (via Residual Analysis)**

- Created residual analysis script
- Identified outlier cars based on z-score of prediction error
- Removed or corrected cars like Furai, Dart, Nova, Zonda R, etc.

---

### **📊 Model Training Progression**

| Version | Model            | Key Changes                                            | R²     | RMSE      |
| ------- | ---------------- | ------------------------------------------------------ | ------ | --------- |
| v1      | XGBoost          | Baseline, 9 features                                   | ~0.93  | ~2.6s     |
| v2      | XGBoost          | +4 engineered features                                 | ~0.94  | ~2.2s     |
| v3      | XGBoost          | Cleaned dataset (355 rows)                             | 0.951  | 2.27s     |
| v4      | XGBoost          | Z-score grip & braking, removed redundant features     | 0.960  | 1.69s     |
| v5      | XGBoost + Optuna | Tuned 50 trials, Z-bonus correction added              | —      | **1.43s** |
| v6      | XGBoost + Optuna | Track Dom. Index added, bonus system removed, mono.    | —      | **1.63s** |

---

### **🧪 Model Tuning with Optuna**

- Tuned parameters:
  - `n_estimators`: 100–500
  - `max_depth`: 3–8
  - `learning_rate`: 0.01–0.3
  - `gamma`: 0–5
  - `subsample`, `colsample_bytree`
- Best params (v6):
  ```python
  {
    'n_estimators': 482,
    'max_depth': 6,
    'learning_rate': 0.0664,
    'subsample': 0.7211,
    'colsample_bytree': 0.6036,
    'gamma': 3.4919
  }