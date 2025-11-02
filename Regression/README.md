# House Price Regression (PyCaret)

This folder contains `Regression.ipynb`, a thorough PyCaret regression workflow on the Ames Housing dataset, including model comparison, tuning, ensembling/blending/stacking, interpretation, final predictions, and pipeline persistence.

## Dataset and setup

- Train CSV: `Regression/dataset/train.csv`
- Test  CSV: `Regression/dataset/test.csv`
- Sample submission: `Regression/dataset/sample_submission.csv` (for column format reference)
- Target column: `SalePrice`
- ID column: `Id` (ignored in modeling)

## Workflow overview (what happens)

1. Installation and imports
   - Installs `pycaret xgboost lightgbm catboost shap` if needed.
   - Uses `from pycaret.regression import *`.
2. Load and sanity checks
   - Asserts the presence of train/test files and expected columns.
   - Prints shapes, missing values summary, and target statistics.
3. Initialize PyCaret for regression
   - `setup(data=train, target='SalePrice', ignore_features=['Id'], fold=5, session_id=42, numeric_imputation='median', categorical_imputation='mode', normalize=False, remove_multicollinearity=False, log_experiment=False, verbose=True)`
   - Displays available models via `models()` to build an environment‑aware include list.
4. Model comparison
   - Builds an include list across linear/knn/trees/boosters if available.
   - `best = compare_models(include=include or None, sort='MAE')` with leaderboard via `pull()`.
5. Baseline diagnostics
   - `plot_model(best, plot='residuals')`, `plot_model(best, plot='error')`, `plot_model(best, plot='feature')` (some plots skip if unsupported).
   - `predict_model(best)` to evaluate on hold‑out and snapshot metrics via `pull()`.
6. Hyperparameter tuning
   - `tuned_best = tune_model(best, optimize='MAE', choose_better=True, n_iter=50)` with diagnostic plots.
7. Ensembles, blending, and stacking
   - Selects `top_models = compare_models(n_select=5, sort='MAE')`.
   - `ensemble_model` (Bagging/Boosting), `blend_models` (soft voting), and `stack_models` with a simple meta‑learner.
   - Compares hold‑out summaries of candidates and picks a “champion”.
8. Interpretation
   - `interpret_model(champion, plot='summary'|'correlation'|'reason')` (SHAP‑based where supported).
9. Finalization and test predictions
   - `final_model = finalize_model(model)` (locks preprocessing + refits on full data).
   - `predict_model(final_model, data=test)` to produce `prediction_label` column.
   - Builds a Kaggle‑style `submission_pycaret.csv` with `Id` and `SalePrice`.
10. Persistence and sanity check
   - `save_model(final_model, 'pycaret_ames_champion')`, `load_model('pycaret_ames_champion')` and quick inference on `test.head(5)`.
11. Optional dashboard and config introspection
   - Attempts `dashboard(model)` if available.
   - Uses `get_config` to inspect training/validation splits and pipelines; exports transformed matrices.

## Key functions used

- Setup: `setup`, `models`, `get_config`
- Selection: `compare_models`, `pull`
- Diagnostics: `plot_model`, `predict_model`
- Tuning/Advanced: `tune_model`, `ensemble_model`, `blend_models`, `stack_models`
- Interpretation: `interpret_model`
- Finalization/IO: `finalize_model`, `save_model`, `load_model`

## How to run

1. Ensure the dataset CSVs exist under `Regression/dataset/`.
2. Open `Regression.ipynb` and execute cells top‑to‑bottom.
3. Inspect leaderboards and plots; tune/ensemble as needed; generate submission and saved pipeline.

Suggested environment setup:

```bash
python -m pip install --upgrade pip
python -m pip install pycaret xgboost lightgbm catboost shap
```

## Notes

- The workflow sorts models by MAE, which is robust to outliers; adapt to RMSE/R2 if preferred.
- Some plots (feature importance, SHAP) require tree/booster models and the `shap` package.
- The notebook is resilient to environment differences (skips or falls back when an estimator/plot isn’t available).
