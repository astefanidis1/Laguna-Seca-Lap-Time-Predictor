# The Oracle - Automotive Lap-Time Prediction

The Oracle is a machine-learning project that predicts **Laguna Seca lap times** from vehicle performance data.

The goal is not just to fit a model, but to build a prediction system that behaves credibly across very different cars. The project evolved through multiple model versions, feature sets, outlier analyses, and benchmark checks before arriving at the current neural-network model.

## What it does

Users provide seven performance inputs:

- 0-60 mph time
- 1/4-mile elapsed time
- 1/4-mile trap speed
- 60-130 mph time
- lateral grip at 120 mph
- 100-0 braking distance
- **Acceleration Curve** - a derived feature relating high-speed and low-speed acceleration

The model returns an estimated Laguna Seca lap time.

## Current model

**Model:** Multilayer Perceptron (TensorFlow / Keras)

- Tuned with **Optuna** over 100 trials
- Uses standardized numerical inputs
- Evaluated with cross-validation during tuning
- Feature behavior checked with **SHAP** and manual perturbation tests
- Model errors investigated with residual analysis and real-world benchmark comparisons
- Historical reported validation MAE: approximately **1.05 seconds**
- Training data comes from a private dataset of **490+ vehicles**

Earlier XGBoost iterations are preserved in the repository for comparison and development history.

## Why the model changed

Earlier versions exposed several failure modes, including redundant features, unrealistic prediction clustering, sensitivity to outliers, and excessive influence from trap speed.

The final feature set was intentionally simplified. Weight, top speed, drive type, and several engineered composite metrics were removed after testing showed they added redundancy or distorted model behavior. The retained inputs focus on acceleration, grip, braking, and one derived acceleration-shape feature.

## Validation approach

The project uses multiple checks rather than relying on a single score:

1. **Cross-validation** during hyperparameter tuning
2. **Residual analysis** to identify systematic errors and extreme outliers
3. **SHAP analysis** to inspect feature influence
4. **Manual perturbation tests** to verify sensible directional behavior
5. **Real-world benchmark checks** against known cars such as the Lexus LFA and Acura/Honda NSX

The tuning pipeline fits preprocessing on each training fold before transforming its validation fold.

## Streamlit app

The repository includes a working Streamlit interface in `Oracle_App.py`.

Run locally:

```bash
pip install -r requirements.txt
streamlit run Oracle_App.py
```

The current model supports Laguna Seca. Additional tracks are future work.

## Repository map

| File | Purpose |
| --- | --- |
| `Oracle_App.py` | Streamlit prediction interface |
| `LagunaPredictorV10.py` | Lightweight inference script |
| `OptunaNNTuner.py` | Hyperparameter tuning and cross-validation |
| `LapTimePredictor_MLP_v10_best.h5` | Saved neural-network model |
| `scaler_v10.pkl` | Saved input scaler for inference |
| `Oracle_Model_Summary.md` | Technical model summary |
| `CHANGELOG.md` | Model-development history |
| `archive/` | Earlier XGBoost and analysis iterations |

## Dataset

The raw training dataset is proprietary and intentionally not included in the public repository. A small sample-input file is provided for format/testing purposes.

## Tech stack

Python - Pandas - NumPy - Scikit-learn - TensorFlow/Keras - Optuna - SHAP - Streamlit
