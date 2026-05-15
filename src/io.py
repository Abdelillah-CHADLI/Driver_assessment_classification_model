"""Path helpers and dataset loading utilities.

Notebooks live in `notebooks/`. They import this module and call
`load_cleaned()` so that the same code works no matter the current
working directory.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def project_root() -> Path:
    """Return the repository root (the directory containing `src/`)."""
    return Path(__file__).resolve().parent.parent


def raw_data_path() -> Path:
    return project_root() / "data" / "raw" / "data.csv"


def cleaned_data_path() -> Path:
    return project_root() / "data" / "processed" / "cleaned_data.csv"


def load_cleaned() -> pd.DataFrame:
    """Load the standardized cleaned dataset produced by notebook 01."""
    return pd.read_csv(cleaned_data_path())


def load_raw() -> pd.DataFrame:
    """Load the original raw CSV (with the messy column names)."""
    return pd.read_csv(raw_data_path())
