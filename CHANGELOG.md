# 🕒 CHANGELOG

## 🚀 v9 — Finalized Model (May 25, 2025)

- ✅ Removed upstream features that caused redundancy or inflated importance:
  - Dropped: `Weight`, `Top Speed`, and `Drive Type`
  - Dropped engineered stats like `Composite Grip Index`, `Track Dominance Index`, and `Powerband Balance`
- ✅ Retained only raw inputs and one derived feature: `Acceleration Curve`
- ✅ Identified and removed cars with high prediction error (Z-score > 3) based on residual analysis
- ✅ Also removed cars with 3 or more missing input values to avoid over-reliance on imputation
- ✅ Retrained using Optuna (100 trials) and achieved best validation RMSE to date: **1.64 seconds**
- ✅ Manually validated outputs on edge cases to confirm model now:
  - Separates cars with realistic time gaps
  - No longer clusters different builds into identical lap times
  - Aligns with real-world benchmarks like the F40 LM
- ✅ Saved as: `LapTimePredictor_XGBoost_v9.json`
- ✅ Locked as **final production model**

---

## ✅ v8 — Trimmed Outliers (May 25, 2025)
- Trimmed dataset using lap time percentiles (5th to 95th)
- Retrained on cleaned data using Optuna tuning
- Improved generalization compared to v7

---

## ✅ v7 — Streamlined Feature Set (May 25, 2025)
- Removed `Horsepower`, `Weight`, `Drive Type`, and `Top Speed`
- Only retained meaningful acceleration, grip, and braking features
- Introduced monotone constraints and cleaned engineered stats down to 1 (`Acceleration Curve`)
- RMSE: ~1.75s

---

## ✅ v6 — Pre-Refactor Baseline
- Full feature set, included some redundant inputs
- Observed flat predictions and unrealistic clustering
- Used as the baseline for current refinement effort