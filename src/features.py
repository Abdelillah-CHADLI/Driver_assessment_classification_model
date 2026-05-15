"""Window-level feature engineering shared by the modelling notebooks.

The raw telemetry is sampled at ~1 Hz. Most driving-behavior research works
at the level of short *windows* (a few seconds) rather than individual rows,
because individual rows are too noisy and too correlated in time. This
module provides one canonical implementation of that pipeline so that the
clustering, anomaly-detection and regression notebooks stay consistent.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.stats import skew


# Sensors that actually reflect driver behavior (chosen in phase 1 of the
# project). They are the ones we aggregate inside each window.
BEHAVIOR_SENSORS: list[str] = [
    "vehicle_speed",
    "acceleration_speed_longitudinal",
    "acceleration_speed_lateral",
    "engine_speed",
    "throttle_position_signal",
    "accelerator_pedal_value",
    "master_cylinder_pressure",
    "steering_wheel_speed",
    "steering_wheel_angle",
    "fuel_consumption",
]


def make_time_windows(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Add a `window_id` column that groups rows of the same driver into
    consecutive non-overlapping windows of `window_size` seconds.

    The cleaned dataset has a per-driver `time_s` column that resets at the
    start of each driver. We use integer division on it to assign windows.
    """
    out = df.sort_values(["driver_id", "time_s"]).reset_index(drop=True).copy()
    out["window_id"] = (out["time_s"].astype(int) // window_size).astype(int)
    out["global_window_id"] = out["driver_id"].astype(str) + "_w" + out["window_id"].astype(str)
    return out


def _window_stats(values: np.ndarray) -> dict[str, float]:
    """Six summary stats for one sensor inside one window."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "skew": 0.0}
    vmean = float(np.mean(values))
    vstd = float(np.std(values, ddof=0)) if n > 1 else 0.0
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if n > 2 and vstd > 0:
        try:
            vskew = float(skew(values, bias=False))
        except Exception:
            vskew = 0.0
    else:
        vskew = 0.0
    return {"mean": vmean, "std": vstd, "min": vmin, "max": vmax, "range": vmax - vmin, "skew": vskew}


def aggregate_window_features(
    window_df: pd.DataFrame,
    sensors: Sequence[str] = BEHAVIOR_SENSORS,
) -> dict[str, float]:
    """Compute the 6 summary stats for every sensor in `sensors` over a single window."""
    feats: dict[str, float] = {}
    for col in sensors:
        if col not in window_df.columns:
            continue
        vals = window_df[col].to_numpy(dtype=float, copy=False)
        for stat, val in _window_stats(vals).items():
            feats[f"{col}__{stat}"] = val
    return feats


def build_window_feature_table(
    df: pd.DataFrame,
    window_size: int = 5,
    sensors: Sequence[str] = BEHAVIOR_SENSORS,
    min_rows_per_window: int = 2,
) -> pd.DataFrame:
    """Turn the per-second telemetry into one row per window.

    Returns columns:
      driver_id, window_id, global_window_id, n_rows,
      <sensor>__mean, <sensor>__std, <sensor>__min, <sensor>__max,
      <sensor>__range, <sensor>__skew  (for every sensor in `sensors`)
    """
    windowed = make_time_windows(df, window_size=window_size)

    rows: list[dict[str, float]] = []
    grouped = windowed.groupby(["driver_id", "window_id"], sort=True)
    for (driver, wid), w in grouped:
        if len(w) < min_rows_per_window:
            continue
        feats = aggregate_window_features(w, sensors)
        feats["driver_id"] = driver
        feats["window_id"] = int(wid)
        feats["global_window_id"] = f"{driver}_w{int(wid)}"
        feats["n_rows"] = int(len(w))
        rows.append(feats)

    out = pd.DataFrame(rows)
    # Move metadata columns to the front.
    meta = ["driver_id", "window_id", "global_window_id", "n_rows"]
    feat_cols = [c for c in out.columns if c not in meta]
    return out[meta + feat_cols]


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Return the names of the numeric feature columns (drops metadata)."""
    meta = {"driver_id", "window_id", "global_window_id", "n_rows"}
    return [c for c in table.columns if c not in meta]


def filter_sensors(table: pd.DataFrame, sensors: Iterable[str]) -> list[str]:
    """Return the feature columns that belong to any of `sensors`."""
    keep = set(sensors)
    return [c for c in feature_columns(table) if c.split("__", 1)[0] in keep]
