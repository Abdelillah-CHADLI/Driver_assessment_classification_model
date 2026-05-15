# Driver Risk via Anomaly Detection - Technical Note

This note documents the design of `notebooks/04_anomaly_detection.ipynb`.
For exact numbers, run the notebook - the per-window and per-driver
results are written to `reports/outputs/`.

## Problem framing

The dataset we have (`data/processed/cleaned_data.csv`, 94,380 rows from
10 drivers labelled A-J) contains **no risk labels**. It is driver-
identification telemetry. We therefore cannot train a supervised "is
this risky?" classifier.

The workable alternative is to treat *aggressive* driving as
*statistically unusual* driving:

- Most rows of telemetry are uneventful (cruising, light traffic).
- The aggressive maneuvers we care about (hard braking, sharp lane
  changes, jerky throttle) are rare and statistically extreme.
- Anomaly detection methods are designed for exactly this regime.

This re-frames "risk classification" as "unusual-window detection on the
raw sensor distribution", which we can run unsupervised.

## Pipeline

1. **Windowing.** Per-second telemetry is grouped into non-overlapping
   5-second windows per driver (`src/features.py:build_window_feature_table`).
2. **Per-window features.** For each window and each of the 10
   behavior-relevant sensors (speed, longitudinal/lateral acceleration,
   engine RPM, throttle position + pedal value, brake-cylinder pressure,
   steering speed + angle, fuel consumption) we compute 6 summary
   statistics: mean, std, min, max, range, skew. That gives ~60 features
   per window.
3. **Anomaly detectors.** Three are run independently, each with a
   contamination prior of ~15%:
   - **Isolation Forest** (300 trees) - global, tree-based.
   - **One-Class SVM** with RBF kernel - boundary-based.
   - **Local Outlier Factor** (`n_neighbors=35`) - density-based.
4. **Consensus.** A window is flagged as `consensus_anom` if at least
   2 of 3 detectors agree, and `unanimous_anom` if all 3 do.
5. **Heuristic prior.** Separately, we build a "risk-score" from six
   common-sense aggressive-driving indicators (brake peak, lateral and
   longitudinal acceleration std, steering speed std, throttle peak,
   speed std), scaled with `RobustScaler`. The notebook plots its
   distribution alongside the model-flagged anomalies to sanity-check
   that the two views agree on direction.
6. **Driver-level rollup.** For each driver, the fraction of their
   windows in the consensus anomaly set is reported. We bucket drivers
   into low / medium / high using the thresholds 15% and 30% **as a
   reporting convention only** - these are not validated against any
   ground truth.

## Why three detectors

The detectors disagree where it matters:

- Isolation Forest isolates points whose axis-aligned bounding box is
  small. It tends to flag globally extreme magnitudes.
- One-Class SVM with RBF flags points outside a learned smooth
  envelope. It is more sensitive to combinations of features.
- LOF flags points that are sparse *relative to their neighbours*. It
  picks up windows that look "out of context" even if not globally
  extreme.

Looking at the agreement matrix (printed in the notebook) tells you
whether the three detectors are seeing the same thing. The consensus
flag is conservative on purpose: it keeps the windows that look unusual
under several different definitions of "unusual".

## What feature importance tells us

The notebook runs permutation importance against the Isolation-Forest
anomaly score. This reveals which sensor-statistics push the score up.
Across runs the lateral-acceleration and steering-speed statistics tend
to dominate, which lines up with the intuition that sharp cornering and
jerky steering are the clearest visible signal of aggressive driving in
this sensor mix.

## Caveats

- **No ground truth.** Every "risky" claim in this notebook is "more
  unusual than the rest of the data", not "objectively dangerous". The
  driver-level buckets are descriptive, not normative.
- **Contamination is a knob.** We fix `contamination = 0.15`. Bigger
  numbers will flag more windows. In a real deployment you'd calibrate
  this against expert-labelled events.
- **Self-comparison.** The pool that defines "normal" is *this 10-driver
  dataset*. Driver A is unusual *relative to the other 9*. If the pool
  changes, the rankings will too.
- **Windowing choices matter.** 5 seconds and 6 stats per sensor are
  defensible defaults but not unique. Longer windows smooth more, more
  stats add resolution. Calibrate to the use case.

## Outputs

`notebooks/04_anomaly_detection.ipynb` writes:

- `reports/outputs/driver_risk_windows_anomaly.csv` - per-window
  anomaly flags and scores from all three detectors plus the heuristic
  risk score.
- `reports/outputs/driver_risk_summary_anomaly.csv` - per-driver
  rollup (window count, mean risk, anomaly rate per detector, consensus
  rate, risk-level bucket).
