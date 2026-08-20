# 🕒 CHANGELOG

## 🚀 v2.0 — The Oracle App Launch (July 2025)

* 🌟 Rebranded "Laguna Seca Lap Time Predictor v10" into **The Oracle**, part of TheCarBible suite
* ✅ Streamlit MVP scoped and planned:

  * Manual input fields (0–60, trap, etc.)
  * Derived `Acceleration Curve`
  * MM\:SS.sss prediction output
  * Track dropdown (Laguna only for now)
  * Placeholder for closest-car matching
* 🔐 Private dataset confirmed as behind-the-scenes engine (not exposed)
* 🖋️ README and model summary rewritten for new vision

---

## 🚀 v1.95 — Final Neural Network Model (May 25, 2025)

* ✅ Migrated from XGBoost to a Keras-based **neural network (MLP)**
* ✅ Used **Optuna** to tune hyperparameters over 100 trials
* ✅ Final architecture: 1 hidden layer, 256 units, `tanh` activation, slight dropout
* ✅ Trained only on **clean, realistic data** (no residual outliers, ≤2 missing values)
* ✅ Inputs normalized via `StandardScaler`
* ✅ All 7 features used and confirmed relevant via:

  * **SHAP analysis**
  * **Manual perturbation tests**
* ✅ Real-world anchor lap times (LFA, NSX) predicted accurately
* ✅ Fictional trap-heavy cars no longer overestimated — model generalizes well
* ✅ Saved as: `LapTimePredictor_MLP_v10_best.h5`

---

## ✅ v1.9 — Finalized XGBoost Model (May 25, 2025)

* ✅ Removed upstream features that caused redundancy or inflated importance:

  * Dropped: `Weight`, `Top Speed`, and `Drive Type`
  * Dropped engineered stats like `Composite Grip Index`, `Track Dominance Index`, and `Powerband Balance`
* ✅ Retained only raw inputs and one derived feature: `Acceleration Curve`
* ✅ Identified and removed cars with high prediction error (Z-score > 3) based on residual analysis
* ✅ Also removed cars with 3 or more missing input values to avoid over-reliance on imputation
* ✅ Retrained using Optuna (100 trials) and achieved best validation RMSE to date: **1.64 seconds**
* ✅ Manually validated outputs on edge cases to confirm model now:

  * Separates cars with realistic time gaps
  * No longer clusters different builds into identical lap times
  * Aligns with real-world benchmarks like the F40 LM
* ✅ Saved as: `LapTimePredictor_XGBoost_v9.json`
* ✅ Locked as **final production model**

---

## ✅ v1.8 — Trimmed Outliers (May 25, 2025)

* Trimmed dataset using lap time percentiles (5th to 95th)
* Retrained on cleaned data using Optuna tuning
* Improved generalization compared to v7

---

## ✅ v1.7 — Streamlined Feature Set (May 25, 2025)

* Removed `Horsepower`, `Weight`, `Drive Type`, and `Top Speed`
* Only retained meaningful acceleration, grip, and braking features
* Introduced monotone constraints and cleaned engineered stats down to 1 (`Acceleration Curve`)
* RMSE: \~1.75s

---

## ✅ v1.6 — Pre-Refactor Baseline

* Full feature set, included some redundant inputs
* Observed flat predictions and unrealistic clustering
* Used as the baseline for current refinement effort