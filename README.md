# 🏁 Laguna Seca Lap Time Predictor — v10

## 💡 What This Project Does

This tool predicts how fast a car will lap Laguna Seca using real-world performance specs. You enter core metrics like 0–60 time, lateral Gs, braking, etc., and the model returns an estimated lap time.

---

## 📆 Files & Purpose

| File                               | Purpose                                                   |
| ---------------------------------- | --------------------------------------------------------- |
| `README.md`                        | You’re here! Overview and usage instructions              |
| `Laguna_Seca_Model_Summary.md`     | Full explanation of the model evolution (v9 → v10)        |
| `sample_input_data.csv`            | Demo cars for trying out predictions                      |
| `LapTimePredictor_MLP_v10_best.h5` | Neural network model trained with Optuna tuning (v10)     |
| `scaler_v10.pkl`                   | Scaler used to normalize inputs during training           |
| `predict_lap_time_nn.py`           | Main script to run predictions using the v10 neural model |
| `predict_lap_time_v2.py`           | (Legacy) Script using the older XGBoost v9 model          |
| `residual_analysis.py`             | Optional: analyze prediction errors and performance       |
| `CHANGELOG.md`                     | Tracks version history and updates                        |

---


🛠️ How to Use It
1. Open `LagunaPredictorV10.py`
2. Modify the `car` dictionary with your car’s specs:

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

3. Run the script
4. It will print your predicted lap time at Laguna Seca

---

## 🤖 Model Details (v10)

* Built using a **deep neural network (Keras)**
* Trained with **Optuna** to tune hyperparameters for realism
* Uses **7 real-world features**, normalized via `StandardScaler`
* Designed to generalize well — avoids tunnel vision on trap speed or other features

### Final features used:

* 0–60 (s)
* 1/4 Mile ET (s)
* Trap Speed (mph)
* 60–130 (s)
* Lateral G @ 120 mph
* 100–0 Braking (ft)
* Acceleration Curve (derived)

---

## 🔄 Legacy (v9 XGBoost Model)

The previous version (`predict_lap_time_v2.py`) used an XGBoost regression model. While strong on paper, it occasionally overemphasized trap speed and struggled to generalize on unusual car types.

You can still use this version for comparison.

---

## ✅ TL;DR

* `predict_lap_time_nn.py` — neural network model (v10) ✅
* `predict_lap_time_v2.py` — legacy XGBoost model (v9)
* `README.md` — usage guide
* `Laguna_Seca_Model_Summary.md` — model evolution and insights
