# Oracle Model Summary (MLP v10)

## Model type

**Multilayer Perceptron (Neural Network) - Version 10**

- Built with **TensorFlow / Keras**
- Tuned using **Optuna** over 100 trials
- Uses a compact, domain-driven seven-feature input set
- Inputs normalized with `StandardScaler`
- Feature behavior inspected with **SHAP** and manual perturbation tests
- Architecture selected through hyperparameter search

### Saved v10 hyperparameters

```python
{
  "n_layers": 1,
  "dropout": 0.0172,
  "learning_rate": 0.00129,
  "batch_size": 16,
  "units_l0": 256,
  "activation_l0": "tanh"
}
```

## Final input features

| Feature | Description |
| --- | --- |
| `0-60 (s)` | Low-speed acceleration |
| `1/4 Mile ET (s)` | Quarter-mile elapsed time |
| `Trap Speed (mph)` | Quarter-mile terminal speed |
| `60-130 (s)` | High-speed acceleration |
| `Lateral G @ 120 mph` | High-speed cornering grip |
| `100-0 Braking (ft)` | Braking performance |
| `Acceleration Curve` | Derived ratio of 60-130 to 0-60 performance |

## Performance and evaluation

- Historical reported validation MAE for the saved v10 model: approximately **1.05 seconds**
- Real-world benchmark checks included cars such as the Lexus LFA and NSX
- Residual analysis was used to identify large errors and problematic training examples
- SHAP and manual perturbation testing were used to check whether feature influence behaved sensibly
- The current tuning script fits preprocessing separately within each cross-validation fold to avoid leakage

The saved model artifact predates the preprocessing cleanup in the tuning script, so a future retraining run on the private dataset should be used to refresh the headline cross-validation metric under the updated pipeline.

## Key assets

| File | Purpose |
| --- | --- |
| `LapTimePredictor_MLP_v10_best.h5` | Saved v10 neural-network model |
| `scaler_v10.pkl` | Saved scaler used for inference |
| `LagunaPredictorV10.py` | Inference script |
| `Oracle_App.py` | Streamlit interface |
| `OptunaNNTuner.py` | Hyperparameter tuning and cross-validation |
| `CHANGELOG.md` | Development history |

The raw training dataset is private and is intentionally not stored in the public repository.
