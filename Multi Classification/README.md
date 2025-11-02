# Multiclass Classification (PyCaret)

This folder contains `MultiClass Classification.ipynb`, a PyCaret multiclass classification workflow using the Dry Bean dataset.

## Dataset and setup

- Excel file used: `Multi Classification/dataset/Dry_Bean_Dataset.xlsx`
- The notebook installs `openpyxl` if missing and reads the Excel into a DataFrame.
- Target column: `Class`
- Sampling: uses a 30% subset (`df.sample(frac=0.3, random_state=42)`) for quicker experimentation.

## Workflow overview (what happens)

1. Installation and imports
   - Installs PyCaret if needed (`!pip install pycaret`) and imports `from pycaret.classification import *`.
   - Ensures Excel engine via `openpyxl`.
2. Load data
   - Reads the Dry Bean dataset from Excel into `df`.
3. Initialize PyCaret
   - `setup(data=df_sampled, target='Class', session_id=123, fix_imbalance=True, normalize=True, verbose=False)`
4. Model selection
   - `best_model = compare_models()` (default sort is Accuracy for multiclass), with `pull()` to view the leaderboard.
5. Evaluation and plots
   - Interactive dashboard: `evaluate_model(best_model)`.
   - Static plots: `plot_model(best_model, plot='confusion_matrix')`, `plot_model(best_model, plot='class_report')`, `plot_model(best_model, plot='feature')`.
6. Predictions
   - `predict_model(best_model)` on the internal hold‑out split.
7. (Optional) Hyperparameter tuning
   - `tune_model(best_model, optimize='Accuracy')` and compare plots again.
8. Finalize and persist
   - `final_model = finalize_model(tuned_model)` and `save_model(final_model, 'drybean_multiclass_model')`.
   - Verifies `load_model` and runs a quick prediction sanity check.

## Key functions used

- Setup: `setup`
- Selection: `compare_models`, `pull`
- Evaluation: `evaluate_model`, `plot_model`
- Tuning: `tune_model`
- Finalization/IO: `finalize_model`, `save_model`, `load_model`, `predict_model`

## How to run

1. Ensure `Dry_Bean_Dataset.xlsx` is present under `Multi Classification/dataset/`.
2. Open `MultiClass Classification.ipynb` and run cells top‑to‑bottom.
3. Review the leaderboard and plots; optionally tune the best model, then finalize and save.

Suggested environment setup:

```bash
python -m pip install --upgrade pip
python -m pip install pycaret openpyxl
```

## Notes

- If some models lack certain plots (e.g., feature importance), PyCaret will skip silently or raise a helpful message; try a tree‑based model (rf, xgboost, lightgbm, catboost) for SHAP/feature plots.
- Use `optimize` in `tune_model` to target your preferred metric (e.g., F1‑Macro for imbalanced multiclass datasets).
