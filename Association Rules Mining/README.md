## Association Rules Mining (PyCaret 2.3.5, Python 3.8)

This notebook mines market-basket association rules using PyCaret’s Association Rules module on Python 3.8 with PyCaret 2.3.5. It supports downloading a dataset from Kaggle, auto-shaping it into a TransactionID/Item table, training the model, visualizing rules, and exporting the top rules.

### What’s inside
- Environment bootstrap for Python 3.8 + PyCaret 2.3.5 (installs pycaret==2.3.5, mlxtend, kaggle into the active kernel)
- Kaggle authentication via kaggle.json and dataset download using the Kaggle Python API (no CLI required)
- Robust CSV loader that accepts either:
  - Transaction rows: columns like `InvoiceNo` (transaction) and `Description` (item)
  - Basket-rows layout: one row per transaction with many item columns → automatically stacks into two columns `TransactionID, Item`
- PyCaret Association Rules workflow: `setup` → `create_model` → `plot_model`
- Exports top rules to `association_rules.csv`

### Requirements
- Python 3.8 (select kernel: “Python 3.8 (PyCaret 2.3.5 x86)”).
- Internet access if you’re downloading from Kaggle.
- A Kaggle account and API token if the dataset requires authentication.

### How to run (cell order)
1) Environment setup
   - Installs required packages in the current kernel. If asked, restart the kernel after install and continue.

2) Kaggle configuration & download
   - Configure Kaggle credentials in one of two ways:
     - Set `KAGGLE_JSON_PATH` env var to your `kaggle.json` file path, or
     - Set `KAGGLE_JSON` env var to the JSON string content of your token.
   - Choose your dataset/file via env vars (or edit the cell):
     - `KAGGLE_DATASET` (e.g., `hariharan29/market-basket-optimisation`)
     - `KAGGLE_FILE` (e.g., `Market_Basket_Optimisation.csv`)
   - The file is downloaded into `./data/` and unzipped automatically.

3) Load and transform data
   - If the hardcoded CSV path isn’t present, the cell auto-discovers a CSV in `./data` (prefers the filename from `KAGGLE_FILE` when available).
   - Outputs a two-column DataFrame `data` with `TRANSACTION_ID_COL` and `ITEM_COL` set appropriately.

4) Train and analyze rules
   - `from pycaret.arules import *`
   - `setup(data, transaction_id=TRANSACTION_ID_COL, item_id=ITEM_COL, session_id=123, silent=True, html=False)`
   - `a1 = create_model()` builds rules; use `plot_model(a1)` and `plot_model(a1, plot='3d')` for visuals.

5) Export results
   - Saves the top rules (by lift, when available) to `association_rules.csv` in the notebook folder and displays the top 20.

### Customization
- You can control rule mining by passing parameters to `create_model`, for example:
  - `create_model(metric='confidence', min_support=0.01, min_threshold=0.5)`
- Change the Kaggle dataset by setting `KAGGLE_DATASET` and `KAGGLE_FILE`, then re-run the Kaggle and Load cells.
- If you already have a local CSV, place it in `./data/` and skip the Kaggle cell.

### Troubleshooting
- “Kaggle CLI not available”: The notebook now uses the Python Kaggle API; re-run the updated Kaggle cell. Ensure the environment setup cell has run so `kaggle` (Python package) is installed.
- “Authentication failed”: Verify `~/.kaggle/kaggle.json` exists with permission `0600`, or set `KAGGLE_JSON_PATH` / `KAGGLE_JSON` and re-run.
- “No CSV found”: Confirm the download succeeded and a CSV exists under `./data/`, or update the hardcoded path/`KAGGLE_FILE`.

### Outputs
- `association_rules.csv` — top mined rules with support, confidence, lift, and related columns.
- Plots — 2D/3D visuals from `plot_model` rendered inline in the notebook.
