# Driver Assessment - Driving Telemetry Analysis

End-to-end exploration of vehicle telemetry from 10 drivers (A-J,
94,380 rows at ~1 Hz). The project covers data cleaning, EDA,
unsupervised clustering of driving behavior, anomaly-based risk
flagging, and a regression-based risk score, with all caveats made
explicit (the dataset has **no ground-truth risk labels**).

## Repository layout

```
.
+-- data/
|   +-- raw/              data.csv               (~17 MB, raw telemetry)
|   +-- processed/        cleaned_data.csv       (output of notebook 01)
+-- notebooks/
|   +-- 01_cleaning_eda.ipynb
|   +-- 02_clustering_kmeans.ipynb
|   +-- 03_clustering_dbscan.ipynb
|   +-- 04_anomaly_detection.ipynb
|   +-- 05_regression.ipynb
+-- src/                  shared helpers used by every modelling notebook
|   +-- io.py             paths and dataset loaders
|   +-- features.py       window-level feature engineering
+-- models/
|   +-- risk_model.pkl    trained regressor + feature list (notebook 05)
+-- reports/
|   +-- phase1_feature_screening.pdf   original feature-screening report
|   +-- anomaly_detection_report.md    methodology note for notebook 04
|   +-- outputs/                        CSV artifacts emitted by notebooks
+-- requirements.txt
+-- README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the notebooks

Notebooks are designed to run top-to-bottom, in order, from any working
directory (each notebook auto-detects the project root and imports
`src/`). To execute the full pipeline end-to-end and write results back
into the notebooks:

```powershell
jupyter nbconvert --to notebook --execute --inplace `
  notebooks/01_cleaning_eda.ipynb `
  notebooks/02_clustering_kmeans.ipynb `
  notebooks/03_clustering_dbscan.ipynb `
  notebooks/04_anomaly_detection.ipynb `
  notebooks/05_regression.ipynb
```

End-to-end runtime is roughly 5-10 minutes on a laptop. The largest
single step is fitting `OneClassSVM` in notebook 04.

## Phase-by-phase report

### Phase 0 - Feature screening (`reports/phase1_feature_screening.pdf`)

Manual review of the 54 raw columns. Engine-tuning and TCU /
maintenance signals (`Fuel_Pressure`, `Engine_soacking_time`, spark
timing, glow-plug control, ...) were marked irrelevant because they
describe what the *vehicle* is doing internally, not what the *driver*
is doing.

### Phase 1 - Cleaning + EDA (`notebooks/01_cleaning_eda.ipynb`)

- Drops the 22 phase-0 irrelevant columns.
- Drops constant columns (`Filtered_Accelerator_Pedal_value`).
- Inventories outliers per feature; does **not** remove them (extreme
  values are usually the events of interest).
- Plots boxplots, histograms, a short telemetry slice for driver A,
  and the full correlation matrix.
- Flags ~15 near-duplicate feature pairs (`|r| > 0.9`), notably the
  four wheel velocities + `Vehicle_speed`, and the various engine /
  flywheel torque columns.
- Snake-cases column names, renames `Class` -> `driver_id`, and writes
  `data/processed/cleaned_data.csv` (94380 x 31).

### Phase 2 - Clustering

Both clustering notebooks use the same 5-second windows on 10
behavior-relevant sensors (10 sensors x 6 stats = 60 features), built by
`src.features.build_window_feature_table`.

**`02_clustering_kmeans.ipynb`** - elbow + silhouette diagnostics over
k=2..12, KMeans at the silhouette-optimal `k`, then a heatmap of the
z-scored cluster signatures and a stacked-bar of each driver's
cluster mix. The PCA scatter already shows that drivers overlap
heavily in window space - cluster identity is *behavioural*, not
per-driver.

**`03_clustering_dbscan.ipynb`** - k-distance plot to pick `eps`, then
a grid search over `(eps, min_samples)` reporting n_clusters,
noise-pct, and silhouette. Final clusters are profiled the same way as
in notebook 02, and the labels are compared to the KMeans labels via
adjusted Rand index + a cross-tab.

### Phase 3 - Anomaly detection (`notebooks/04_anomaly_detection.ipynb`)

See `reports/anomaly_detection_report.md` for the full methodology. In
short:

- Isolation Forest + One-Class SVM + Local Outlier Factor, each with
  ~15% contamination.
- Per-window consensus and unanimous flags.
- Per-driver anomaly-rate rollup with arbitrary low/medium/high buckets.
- Permutation importance against the Isolation-Forest score to surface
  which sensor-statistics actually drive the anomaly decisions.
- A heuristic 6-component risk score is computed *separately* and used
  only to sanity-check that the unsupervised flags overlap with what an
  analyst would call "aggressive".

### Phase 4 - Regression-based risk (`notebooks/05_regression.ipynb`)

A synthetic continuous risk score is built from six aggressive-driving
indicators (peak brake pressure, lateral / longitudinal acceleration
std, steering-speed std, throttle peak, speed std), normalised to
[0, 1]. The model has to predict this score from the **other** sensor
statistics - every column derived from one of the six target-source
sensors is removed from the feature matrix. Without that step the model
would simply learn the target's defining formula.

- Models compared: Ridge, RandomForest, GradientBoosting.
- Evaluation: 5-fold `GroupKFold` by driver - the test driver(s) are
  never seen during training.
- Diagnostics: predicted-vs-actual scatter, residuals-vs-prediction,
  residual histogram, top feature importances, per-driver predicted
  risk bar chart bucketed into safe / moderate / dangerous tertiles.

The trained estimator is saved to `models/risk_model.pkl` together with
the list of features it expects.

## Assumptions and caveats

- **No risk labels.** Driver identity (A-J) is **not** a risk label. It
  is just an ID. Every "risk" output in this project is built from
  unsupervised signal or from a synthetic target - none of it is
  validated against ground truth.
- **Outliers are kept.** In driving telemetry, the extreme values *are*
  the events we care about. We inventory and visualize them but do not
  drop them.
- **Driver A scores highest on every method.** Both anomaly detection
  and the regression model rank driver A above the rest. That is a
  property of *this dataset* - removing or replacing driver A would
  shift the ranking.

## Outputs

`reports/outputs/`

| file | source notebook | content |
|---|---|---|
| `windows_kmeans.csv`                 | 02 | per-window KMeans label |
| `windows_dbscan.csv`                 | 03 | per-window DBSCAN label (-1 = noise) |
| `driver_risk_windows_anomaly.csv`    | 04 | per-window anomaly flags + scores from IF / OC-SVM / LOF + heuristic risk |
| `driver_risk_summary_anomaly.csv`    | 04 | per-driver anomaly rollup |
| `driver_risk_summary_regression.csv` | 05 | per-driver predicted-risk rollup |
| `risk_assessment_results.csv`        | 05 | per-window target + prediction |
