# Binary Classification (PyCaret)

This folder contains `Binary Classification.ipynb`, a complete PyCaret classification workflow that trains and evaluates binary classifiers, predicts on a held‑out test set, and persists the final model.

## Dataset and setup

- Train CSV: `Binary Classification/dataset/train.csv`
- Test  CSV: `Binary Classification/dataset/test.csv`
- Target column: `y`
- The notebook samples 30% of the train set for faster experimentation.

## Workflow overview (what happens)

1. Installation and imports
   - Installs PyCaret if needed: `!pip install pycaret`
   - `from pycaret.classification import *`
2. Load data
   - Reads train/test CSVs and prints basic shapes.
3. Optional sampling
   - `train.sample(frac=0.3, random_state=42)` to reduce runtime.
4. Initialize PyCaret
   - `setup(data=train_sampled, target='y', session_id=123, fix_imbalance=True, normalize=True, verbose=False)`
   - Fixes class imbalance, standardizes numeric features, and seeds folds for reproducibility.
5. Model selection
   - `best_model = compare_models()` evaluates a wide range of classifiers.
   - Uses `pull()` to view the leaderboard table of cross‑validated metrics.
6. Evaluation and plots
   - `evaluate_model(best_model)` opens the interactive evaluation UI.
   - Key plots: `plot_model(best_model, plot='auc')`, `plot_model(best_model, plot='confusion_matrix')`, `plot_model(best_model, plot='feature')`.
7. Predictions
   - `predict_model(best_model)` on the internal hold‑out split.
   - `predict_model(best_model, data=test)` on external test data; saves `test_predictions.csv`.
8. Finalize and persist
   - `final_model = finalize_model(best_model)` retrains on full training data.
   - `save_model(final_model, 'bank_classifier_model')` to persist the pipeline.
9. (Optional) Interpretation and AutoML
   - `interpret_model(best_model)` (works best with tree‑based models; the notebook falls back to a Random Forest if needed).
   - `get_leaderboard()` and `automl(optimize='AUC')` to pick the best model for a metric.

## Key functions used

- Setup: `setup`
- Selection: `compare_models`, `get_leaderboard`, `automl`
- Training/usage: `predict_model`, `finalize_model`
- Visualization: `evaluate_model`, `plot_model`
- Persistence: `save_model`, `load_model`

## How to run

1. Place `train.csv` and `test.csv` under `Binary Classification/dataset/`.
2. Open `Binary Classification.ipynb` in VS Code/Jupyter.
3. Run cells top‑to‑bottom. After `compare_models`, inspect the leaderboard; evaluation/plots will render inline.
4. The notebook writes `test_predictions.csv` in the working directory.

Suggested environment setup:

```bash
python -m pip install --upgrade pip
python -m pip install pycaret
```

## Notes

- If imbalance is severe, consider adjusting `fix_imbalance_method` or using threshold tuning on prediction scores for your target metric (e.g., F1, Recall).
- `finalize_model` refits on all available training rows; use it before producing your final test predictions for deployment/submission.
