# Laguna Seca Lap Time Prediction Model (MLP v10)

## ✅ Model Type

* **Multilayer Perceptron (Neural Network)** (v10)

  * Built with **TensorFlow / Keras**
  * Tuned via **Optuna** with 100 trials
  * Trained on high-quality, real-world data only
  * Inputs normalized using `StandardScaler`
  * Simple, fully connected architecture with dropout
  * SHAP + perturbation checks confirm balanced feature use

### Best Hyperparameters (Optuna v10):

```python
{
  'n_layers': 1,
  'dropout': 0.0172,
  'learning_rate': 0.00129,
  'batch_size': 16,
  'units_l0': 256,
  'activation_l0': 'tanh'
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

* **Validation MAE**: \~**1.05 seconds**
* Closely matches real-world anchor laps (LFA, NSX)
* Generalizes well to fictional cars without trap speed bias
* No monotonic constraints used — learned behavior naturally
* All 7 features have measurable influence (confirmed via perturbation)

---

## 📄 Key Files

* `LapTimePredictor_MLP_v10_best.h5` – Final trained MLP model
* `neural_model_optuna.py` – Optuna tuning script
* `predict_lap_time_v10.py` – Inference script with scaler + NN
* `scaler_v10.pkl` – Saved `StandardScaler`
* `TrainingDataV10.csv` – Final clean training data
* `CHANGELOG.md` – Full version history

---

🏁 **This is the final v10 model: neural-powered, realistic, and deeply validated.**
