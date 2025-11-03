# PyCaret Assignment

This repository contains multiple Jupyter notebooks demonstrating common machine learning workflows using PyCaret. Each top-level folder focuses on a particular ML task and contains notebooks, example data, and a short README describing the work in that folder.

Video walkthroughs for all notebooks are available in this Google Drive folder:

https://drive.google.com/drive/folders/1j3AUBwi-nxgY45VpTWCDXsXvxTviNL3u?usp=sharing

These videos explain the notebooks step-by-step.

## Repository structure

- `Anomaly Detection/` — Notebook(s) and README for anomaly detection (credit card example included).
- `Association Rules Mining/` — Notebook(s) demonstrating association rule mining with PyCaret.
- `Binary Classification/` — Example notebook for binary classification tasks.
- `Clustering/` — Notebooks and notes on clustering workflows.
- `Multi Classification/` — Multi-class classification notebooks.
- `Regression/` — Regression notebooks and guidance.
- `Time series/` — Time series forecasting notebooks.

## How to run the notebooks

1. Create a Python environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install Jupyter and project dependencies. Many notebooks use PyCaret; install a compatible version for your Python:

   pip install --upgrade pip
   pip install jupyter pycaret

3. Start Jupyter Lab/Notebook and open the notebook you want to run:

   jupyter lab

4. Follow the corresponding video in the Google Drive folder for a guided walkthrough.

## Notes

- Some notebooks may require additional packages (plotting libs, dataset-specific packages). If a notebook fails to run, check the first cells for imports and install missing packages.
- Notebooks were created for demonstration and learning. Results may vary depending on library versions.

## Contact / Credits

Created by the PyCaret assignment author. For questions about the notebooks, refer to the video walkthroughs linked above.
