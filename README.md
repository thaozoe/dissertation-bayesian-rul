# Bayesian Machine Learning for Real-Time RUL Prediction of Industrial Rotating Equipment

This repository contains the full modeling pipeline for a dissertation project on
**Bayesian Machine Learning for Real-Time Remaining Useful Life Prediction of Industrial Rotating Equipment**, using the
**IMS Bearing Dataset** (Center for Intelligent Maintenance Systems, University of
Cincinnati). The project progresses from exploratory data analysis, through
classical point-prediction baselines, to Bayesian models that quantify predictive
uncertainty, and finally a real-time streaming simulation that demonstrates the
full pipeline running under deployment-like conditions.

## Dataset

- **Source:** IMS Bearing Dataset ("Experiments on Bearings"), Center for
  Intelligent Maintenance Systems (IMS), University of Cincinnati — publicly
  available via the [NASA Open Data Portal](https://data.nasa.gov/).
- **Content:** vibration signals from run-to-failure tests on four bearings per
  test rig (test1, test2, test3), sampled at 20 kHz, with a new snapshot recorded
  roughly every 10 minutes until failure.
- The raw dataset is **not included** in this repository. Download it separately
  and update the folder paths at the top of each notebook to point to your local
  copy of `test1/`, `test2/`, `test3/`.

## Notebooks

The notebooks are meant to be run **in order** — each one consumes the outputs
of the previous stage.

| Notebook | Purpose |
|---|---|
| `1_EDA.ipynb` | Exploratory data analysis: raw signal inspection, theoretical fault-frequency calculations, time/frequency-domain feature behaviour, distribution analysis, autocorrelation, and degradation-trend exploration. Establishes the feature set and the bandpass band used later. |
| `2_Preprocessing.ipynb` | Full preprocessing pipeline for test2 + test3: time/frequency-domain feature extraction, per-test bandpass band selection via spectral kurtosis (kurtogram), RUL label generation, Hampel + EWMA smoothing, Health Index (PCA) computation, temporal train/val/test split, feature normalization, and sliding-window sequence generation. |
| `3_Baseline_Models.ipynb` | Non-Bayesian point-prediction baselines, in increasing order of complexity: mean-predictor floor, Linear Regression, XGBoost, MLP, and LSTM. Evaluates RMSE/MAE and an asymmetric CMAPSS-style score on the held-out test3 run. |
| `4_Bayesian_Models.ipynb` | Bayesian counterparts that output a predictive distribution instead of a point estimate: Bayesian Ridge Regression, Gaussian Process Regression, and MC Dropout LSTM (reusing the NB3 LSTM backbone). Adds uncertainty metrics (PICP, MPIW, NLL, CRPS) and a robustness check on test1 as an out-of-distribution run. |
| `5_RealTime_Prediction.ipynb` | Deployment simulation: replays test3 snapshot-by-snapshot to simulate live arrival, runs feature extraction + prediction + a maintenance decision (NORMAL / WARNING / CRITICAL) per snapshot, and checks that per-snapshot latency stays well within the ~10-minute sensor cadence. |

### Supporting modules

These local Python modules are imported by the notebooks above and must sit in
the same working directory:

- `realtime_utils.py` — defines `RealTimeRULPredictor`, the streaming inference
  class used in Notebook 5 (feature extraction, scaling, Health Index transform,
  and per-snapshot prediction across all three Bayesian models).
- `robustness_test1_utils.py` — defines `load_test1_for_robustness()`, used in
  Notebook 4 to load and prepare test1 for the out-of-distribution robustness
  check.

## Pipeline Overview

```
Raw vibration files (test1 / test2 / test3)
        │
        ▼
1_EDA.ipynb              → feature/band selection understanding
        │
        ▼
2_Preprocessing.ipynb     → features, RUL labels, splits, sequences
        │  (./preprocessing_output)
        ▼
3_Baseline_Models.ipynb   → point-prediction models
        │  (./baseline_output)
        ▼
4_Bayesian_Models.ipynb   → uncertainty-aware models + OOD robustness check
        │  (./bayesian_output)
        ▼
5_RealTime_Prediction.ipynb → streaming simulation + maintenance alerts
        (./realtime_output)
```

Each notebook writes its results (feature arrays, trained models, scalers,
metric tables, and figures) into its own `*_output/` folder, which the next
notebook reads from — no manual copying is required as long as all notebooks
share the same working directory.

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the IMS Bearing Dataset and update the data directory paths in
   `1_EDA.ipynb` and `2_Preprocessing_v10_kurtogram.ipynb` to match your local
   folder structure.
4. Launch Jupyter and run the notebooks in the order listed above:
   ```bash
   jupyter lab
   ```

## Models Implemented

**Point-prediction baselines (Notebook 3):**
- Mean predictor (sanity/feasibility floor)
- Linear Regression
- XGBoost
- Multi-Layer Perceptron (MLP)
- LSTM

**Bayesian / uncertainty-aware models (Notebook 4):**
- Bayesian Ridge Regression
- Gaussian Process Regression
- MC Dropout LSTM (same backbone as the Notebook 3 LSTM)

## Evaluation Metrics

- **Point accuracy:** RMSE, MAE (normalized and in minutes), and an
  asymmetric CMAPSS-style score that penalizes late predictions more than
  early ones.
- **Uncertainty calibration (Bayesian models only):**
  - **PICP** — Prediction Interval Coverage Probability
  - **MPIW** — Mean Prediction Interval Width
  - **NLL** — Negative Log-Likelihood
  - **CRPS** — Continuous Ranked Probability Score
- **Deployment feasibility:** per-snapshot inference latency vs. the sensor's
  ~10-minute sampling cadence.

## Notes

- test2 is used for training/validation; test3 is the primary in-distribution
  held-out test run; test1 is reserved exclusively for the out-of-distribution
  robustness check in Notebook 4.
- `RUL_normalized` and `MAX_RUL` are computed offline (they require knowing the
  time of failure), so in Notebook 5 "real-time" refers to inference speed,
  not to knowing this normalization constant in advance for a genuinely new,
  in-progress run.
