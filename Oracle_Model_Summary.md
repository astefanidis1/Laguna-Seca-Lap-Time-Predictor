# 🧠 Oracle Model Summary (MLP v10)

## ✅ Model Type

**Multilayer Perceptron (Neural Network)** — Version 10

* Built with **TensorFlow / Keras**
* Tuned using **Optuna** (100 trials)
* Trained on clean, real-world car data only
* Inputs normalized using `StandardScaler`
* SHAP + manual perturbation used to validate feature importance
* Architecture: 1 hidden layer, dropout, `tanh` activation

### 🔧 Best Hyperparameters (Optuna v10)

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

| Feature               | Description             | Better When...       |
| --------------------- | ----------------------- | -------------------- |
| `0-60 (s)`            | Time to 60 mph          | Lower                |
| `1/4 Mile ET (s)`     | Quarter mile time       | Lower                |
| `Trap Speed (mph)`    | Speed at end of 1/4     | Higher               |
| `60-130 (s)`          | Time from 60 to 130 mph | Lower                |
| `Lateral G @ 120 mph` | Cornering grip          | Higher               |
| `100-0 Braking (ft)`  | Braking distance        | Lower                |
| `Acceleration Curve`  | Ratio of 60–130 to 0–60 | Optimal near 1.0–1.4 |

---

## 📈 Model Performance

* **Validation MAE**: ≈ **1.05 seconds**
* Tracks real-world cars well (e.g., LFA, NSX)
* Generalizes across fictional builds and AI-created entries
* Avoids trap speed bias seen in earlier models
* All 7 features contribute meaningfully to predictions

---

## 📄 Key Assets

| File                               | Purpose                       |
| ---------------------------------- | ----------------------------- |
| `LapTimePredictor_MLP_v10_best.h5` | Final trained model           |
| `scaler_v10.pkl`                   | StandardScaler for input data |
| `TrainingDataV10.csv`              | Final clean training dataset  |
| `LagunaPredictorV10.py`            | Inference script (v10)        |
| `OptunaNNTuner.py`                 | Hyperparameter tuning logic   |
| `CHANGELOG.md`                     | Full version history          |

---

🏁 **This is the final v10 model that powers The Oracle — deeply validated, realistic, and tuned for generalization.**