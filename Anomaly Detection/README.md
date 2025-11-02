# Credit Card Anomaly Detection (PyCaret)

This folder contains `CreditCardAnomalyDetection.ipynb`, a complete unsupervised anomaly detection workflow on credit-card transactions using PyCaret's anomaly module.

## What this notebook does

- Environment repair and clean install for PyCaret 3.3.2 and compatible pins (scikit-learn 1.4.2, numpy 1.26.x, pandas 2.1.x, etc.).
  - Includes cleanup cells that uninstall conflicting packages and purge pip cache.
  - Provides a one‑shot pinned install block and optional extras (lightgbm, xgboost, catboost, shap, plotly, etc.).
- Data loading and preparation
  - Reads `/Users/spartan/Downloads/creditcard.csv`.
  - Uses the ground‑truth label `Class` (1=fraud, 0=legit) only for evaluation, not for training features.
  - Optionally downsamples to ~60,000 rows for faster experiments.
- PyCaret anomaly experiment
  - `setup(data=data, session_id=123, normalize=True, transformation=False, verbose=True)`
  - Trains multiple detectors: Isolation Forest (`'iforest'`), KNN (`'knn'`), and LOF (`'lof'`).
  - `assign_model` attaches per‑row anomaly flags and scores; `predict_model` scores arbitrary data.
- Evaluation against ground truth
  - Computes AUC‑ROC, PR‑AUC, Precision, Recall, F1, and Confusion Matrix for each detector using `sklearn.metrics`.
  - Contains a simple percentile‑based threshold tuner for anomaly scores to maximize F1 or meet a recall target.
- Model persistence
  - Saves and reloads the trained pipeline with `save_model` / `load_model`.

## Files and paths

- Expected dataset: `/Users/spartan/Downloads/creditcard.csv` (update the path in the notebook if different).
- No additional project files are required; all steps are in the notebook.

## Key code steps (PyCaret)

- Import: `from pycaret.anomaly import setup, create_model, assign_model, predict_model, plot_model, save_model, load_model, models`
- Initialize: `setup(data=data, session_id=123, normalize=True, transformation=False)`
- Train: `create_model('iforest')`, `create_model('knn')`, `create_model('lof')`
- Label/Score: `assign_model(model)`, `predict_model(model, data=data)`
- Inspect catalog: `models()`
- Persist: `save_model(model, 'iforest_pipeline_cc')` and `load_model('iforest_pipeline_cc')`

## How to run

1. Open `CreditCardAnomalyDetection.ipynb` in VS Code or Jupyter.
2. Run the cleanup/installation cells first to ensure compatible versions. If you already have a working environment, you can skip the cleanup.
3. Make sure the CSV path is correct. If you don't have `creditcard.csv`, download it (e.g., from Kaggle) and update the path.
4. Execute cells top‑to‑bottom. The evaluation cells print AUC/PR‑AUC/Precision/Recall/F1 and a confusion matrix for each model.

Optional environment setup (one possible baseline):

```bash
python -m pip install --upgrade pip
python -m pip install "numpy==1.26.4" "pandas==2.1.4" "scipy==1.11.4" "scikit-learn==1.4.2" "matplotlib==3.7.5"
python -m pip install "pycaret==3.3.2" "imbalanced-learn==0.12.3" PyYAML "umap-learn>=0.5.5" "shap~=0.44.0"
# Optional extras often used by PyCaret
python -m pip install lightgbm xgboost catboost optuna mlflow plotly kaleido ipywidgets
```

## Notes and tips

- If `setup` errors due to LightGBM import, the notebook removes LightGBM and retries. You can skip uninstalling if your environment already works.
- Downsampling greatly speeds up experimentation; rerun on the full dataset when finalizing.
- Threshold tuning shows how to trade precision vs recall based on the anomaly score distribution.

---
If you want, I can run this notebook end‑to‑end and generate the metrics table, or extract a minimal `requirements.txt` from the pinned versions inside the notebook.
