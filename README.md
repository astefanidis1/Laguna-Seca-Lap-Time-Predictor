# 🧠 The Oracle — Lap Time Prediction Tool (Powered by TheCarBible)

## 💡 What This Tool Does

**The Oracle** predicts how fast any car would lap Laguna Seca (and eventually other tracks) using machine learning. Users can input key performance stats — like acceleration, trap speed, grip, and braking — and receive a realistic lap time estimate.

This evolved from the original "Laguna Seca Lap Time Predictor v10" project into a polished tool within *TheCarBible* suite.

---

## 📦 Included Files & Purpose

| File                               | Purpose                                              |
| ---------------------------------- | ---------------------------------------------------- |
| `README.md`                        | You’re here! Project overview and usage instructions |
| `Oracle_Model_Summary.md`          | Technical summary of the ML model (v9 → v10)         |
| `sample_input_data.csv`            | (Legacy/testing) Example cars for prediction testing |
| `LapTimePredictor_MLP_v10_best.h5` | Trained neural network (Keras + Optuna-tuned)        |
| `scaler_v10.pkl`                   | StandardScaler for feature normalization             |
| `LagunaPredictorV10.py`            | Script for prediction using the model                |
| `CHANGELOG.md`                     | Version history and updates                          |

---

## 🚀 MVP App (v1.0)

A Streamlit-based app will:

* Accept manual car spec inputs (0–60, trap speed, etc.)
* Auto-calculate Acceleration Curve
* Predict lap time using the neural model
* Display the result in clean MM\:SS.sss format
* Offer a dropdown to select track (only Laguna Seca supported currently)

### Example Input:

```python
car = {
    '0-60 (s)': 3.2,
    '1/4 Mile ET (s)': 11.0,
    'Trap Speed (mph)': 130,
    '60-130 (s)': 7.5,
    'Lateral G @ 120 mph': 1.15,
    '100-0 Braking (ft)': 265.0
}
```

---

## 🤖 Model Details (v10)

* Built using **Keras / TensorFlow**
* Tuned using **Optuna** over 100 trials
* Validated against real-world benchmarks (e.g., LFA, NSX)
* Final MAE: ≈ **1.05 seconds**
* Inputs normalized via `StandardScaler`
* No trap-speed overfitting (avoids the "trap speed trap")

### Final Input Features (7):

* 0–60 (s)
* 1/4 Mile ET (s)
* Trap Speed (mph)
* 60–130 (s)
* Lateral G @ 120 mph
* 100–0 Braking (ft)
* Acceleration Curve (derived)

---

## 🛠 Future Features (Planned)

* Track support for Spa, Nürburgring, and others
* Closest-car comparator from private dataset
* Residual range/confidence output
* Car vs car comparison mode
* Bulk CSV upload support
* Leaderboard system

---

## 🔒 Dataset Notice

This project is powered by a **private proprietary dataset** of 490+ vehicles.

The raw dataset:

* Will never be shown or exposed
* Powers predictions behind the scenes
* Enables AI-car integration, lore, and ranking systems

---

## ✅ TL;DR

* `LagunaPredictorV10.py` — makes predictions using final v10 model
* `oracle_app.py` — (coming soon) Streamlit frontend
* `LapTimePredictor_MLP_v10_best.h5` — trained neural model
* `scaler_v10.pkl` — input normalizer
* `Oracle_Model_Summary.md` — deep dive on the modeling process