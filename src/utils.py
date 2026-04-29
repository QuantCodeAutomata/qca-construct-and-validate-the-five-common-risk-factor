"""
Shared utility functions for the Fama-French five-factor replication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Sample period ──────────────────────────────────────────────────────────────
SAMPLE_START = "1963-07"
SAMPLE_END = "1991-12"
N_OBS = 342  # July 1963 – December 1991


def align_monthly_series(*series: pd.Series, start: str = SAMPLE_START, end: str = SAMPLE_END) -> pd.DataFrame:
    """
    Align multiple monthly return series to the common sample period.

    Parameters
    ----------
    *series : pd.Series
        Monthly return series with PeriodIndex or DatetimeIndex.
    start : str
        Start period (YYYY-MM).
    end : str
        End period (YYYY-MM).

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with PeriodIndex (monthly frequency).
    """
    frames = []
    for s in series:
        if isinstance(s.index, pd.DatetimeIndex):
            s = s.copy()
            s.index = s.index.to_period("M")
        frames.append(s)
    df = pd.concat(frames, axis=1)
    idx = pd.period_range(start=start, end=end, freq="M")
    return df.reindex(idx)


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a series at given quantile bounds."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def value_weighted_return(returns: pd.Series, weights: pd.Series) -> float:
    """
    Compute value-weighted portfolio return.

    Parameters
    ----------
    returns : pd.Series
        Individual stock returns for the period.
    weights : pd.Series
        Lagged market equity (weights before normalisation).

    Returns
    -------
    float
        Value-weighted return.
    """
    valid = returns.notna() & weights.notna() & (weights > 0)
    r = returns[valid]
    w = weights[valid]
    if w.sum() == 0:
        return np.nan
    return (r * w).sum() / w.sum()


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, t-stat, and first-order autocorrelation for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly return series (percent units).

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    n = df.shape[0]
    means = df.mean()
    stds = df.std(ddof=1)
    t_stats = means / (stds / np.sqrt(n))
    autocorr = df.apply(lambda x: x.autocorr(lag=1))
    summary = pd.DataFrame({
        "Mean (%)": means,
        "Std (%)": stds,
        "t-stat": t_stats,
        "AR(1)": autocorr,
    })
    return summary


def save_table(df: pd.DataFrame, filename: str, title: str = "") -> None:
    """Save a DataFrame as a CSV and print a formatted version."""
    path = RESULTS_DIR / filename
    df.to_csv(path)
    logger.info("Saved %s to %s", title or filename, path)


def period_range_monthly(start: str = SAMPLE_START, end: str = SAMPLE_END) -> pd.PeriodIndex:
    """Return the standard monthly PeriodIndex for the sample."""
    return pd.period_range(start=start, end=end, freq="M")
