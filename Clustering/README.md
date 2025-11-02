# Customer Clustering (PyCaret)

This folder contains `Clustering.ipynb`, an end‑to‑end clustering workflow on the Mall Customers dataset using PyCaret’s clustering module, with extra analysis and export utilities.

## Dataset and preparation

- CSV path used in the notebook: `Clustering/dataset/Mall_Customers.csv`
- Columns are standardized (strip spaces, lowercased, underscores for spaces).
- Identifier (`customerid`) is dropped before modeling.

## Workflow overview (what happens)

1. Setup
   - `from pycaret.clustering import *`
   - `setup(data=df, session_id=42, normalize=True, normalize_method='zscore', verbose=False, html=False)`
   - Normalizes numeric features and seeds the experiment.
2. Model sweep and selection
   - Trains multiple candidates:
     - KMeans across several k values: `create_model('kmeans', num_clusters=k)` for k ∈ {3,4,5,6,7,8}
     - Additional algorithms: `birch`, `hclust`, `dbscan`, `optics`
   - Captures the results via `pull()` and selects the best model based on Silhouette score.
3. Label assignment and profiles
   - `assign_model(best_model)` returns the dataset with a `Cluster` column (and distances/scores if available).
   - Builds per‑cluster numeric profiles (means) and size distributions.
4. Visualization
   - `plot_model(best_model, plot='cluster')` (PCA scatter)
   - `plot_model(best_model, plot='silhouette')`
   - `plot_model(best_model, plot='elbow')` (for KMeans)
   - A custom PCA projection cell also visualizes clusters in 2D.
5. Exports and reports
   - Writes clustered data and summaries (sizes, numeric profiles) to CSVs.
   - Generates a lightweight HTML and Markdown report (no extra deps) in `final_artifacts/`.
   - Optionally zips artifacts for sharing.
6. Predict on new data
   - Uses `predict_model` on raw samples or the entire dataset with the saved/selected model.
7. Reusable pipeline function
   - `run_clustering_pipeline(...)` demonstrates how to run a complete clustering pass on any similar CSV (sweep K, pick best by Silhouette, save outputs and pipeline).

## Key functions used

- Setup: `setup`
- Training: `create_model`, `pull`
- Selection: Silhouette-based ranking over assembled results
- Labeling/Prediction: `assign_model`, `predict_model`
- Visualization: `plot_model('cluster'|'silhouette'|'elbow')`
- Persistence: `save_model`, `load_model`

## Outputs written by the notebook

- `mall_customers_clustered.csv` — data with `Cluster` labels
- `cluster_outputs/` — size and numeric profile CSVs
- `final_artifacts/` — HTML/MD reports, optional images and mappings
- `mall_clustering_artifacts.zip` — consolidated archive (optional)

## How to run

1. Ensure `Clustering/dataset/Mall_Customers.csv` exists.
2. Open `Clustering.ipynb` and run cells sequentially.
3. Inspect cluster sizes/profile tables and visualizations; adjust `k_values` or features as desired.

Suggested environment setup:

```bash
python -m pip install --upgrade pip
python -m pip install pycaret numpy pandas scikit-learn matplotlib
# Optional for richer plots/reports
python -m pip install plotly kaleido ipywidgets
```

## Notes

- The best model is chosen by Silhouette score; for KMeans, the notebook also provides an elbow study and a final lock‑in of the preferred `k`.
- Some `plot_model` calls are skipped gracefully when not supported by a model/back‑end.
- Cluster names/segments can be constructed heuristically from profile means (the notebook shows an example).
