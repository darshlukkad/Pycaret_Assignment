## Time Series Forecasting with PyCaret

This notebook demonstrates univariate time series forecasting using PyCaret’s time_series module. It walks through loading a sample series, initializing the experiment, comparing models, visualizing forecasts and residuals, generating future predictions, and saving models.

### What’s inside
- Load a sample dataset (`airline`) via `pycaret.datasets.get_data`
- Initialize the experiment with `setup(data, fh=3, session_id=123)`
- Compare baseline models with `compare_models()` (functional) and `exp.compare_models()` (OOP)
- Analyze with `plot_model(best, plot='forecast'|'residuals')` and interact via `evaluate_model`
- Predict hold-out and future horizons with `predict_model`
- Save / load models (notebook sections demonstrate additional plots and utilities)

### Requirements
- PyCaret Time Series (PyCaret 3.x recommended for `pycaret.time_series`).
- Python 3.8–3.10 is commonly supported for PyCaret 3.x.

If you haven’t installed PyCaret time series yet, run inside the active kernel:

```
%pip install -q "pycaret[time_series]"
```

Note: If you are also using the Association Rules notebook constrained to PyCaret 2.3.5, keep that in a separate kernel (e.g., the provided “Python 3.8 (PyCaret 2.3.5 x86)”) to avoid dependency conflicts. For this Time Series notebook, a newer kernel with PyCaret 3.x is recommended.

### How to run (high level)
1) Import and load data
   - `from pycaret.datasets import get_data`
   - `data = get_data('airline')`

2) Initialize setup
   - `from pycaret.time_series import *`
   - `s = setup(data, fh=3, session_id=123)`

3) Compare and select model
   - `best = compare_models()`
   - Optional OOP style:
     - `from pycaret.time_series import TSForecastingExperiment`
     - `exp = TSForecastingExperiment(); exp.setup(data, fh=3, session_id=123); exp.compare_models()`

4) Analyze and forecast
   - `plot_model(best, plot='forecast')`
   - `plot_model(best, plot='forecast', data_kwargs={'fh': 36})` for longer horizon
   - `plot_model(best, plot='residuals')` for diagnostic checks

5) Predict
   - `holdout_pred = predict_model(best)`  # uses `fh` from setup
   - `predict_model(best, fh=36)`          # custom future horizon

6) Persist (optional)
   - `save_model(best, 'best_ts_model')`
   - `loaded = load_model('best_ts_model')`

### Tips
- Ensure your series index is a proper DatetimeIndex at a consistent frequency.
- For multivariate/exogenous variables, see `setup(..., seasonal_period, exogenous, ...)` in docs.
- Use `get_metrics()` to list scoring metrics and `add_metric()` to register custom ones.

### Troubleshooting
- Missing module `pycaret.time_series`: Install `pycaret[time_series]` in the selected kernel.
- Plot rendering issues: Run in a Jupyter environment with widget support; consider `%pip install ipywidgets` and enabling it.
- Version conflicts: Keep time series work in a separate environment from legacy PyCaret 2.3.5 tasks to avoid dependency pins clashing.
