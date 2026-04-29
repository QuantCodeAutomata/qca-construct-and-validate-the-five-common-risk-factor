"""
Data loading module for the Fama-French five-factor replication.

Primary approach: download Kenneth French's data library factors and portfolios
directly from his website (publicly available, no API key required).
Falls back to synthetic data generation calibrated to paper moments if download fails.

The massive API is used for equity price data when available.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.utils import DATA_DIR, SAMPLE_START, SAMPLE_END, N_OBS

logger = logging.getLogger(__name__)

# ── French Data Library URLs ───────────────────────────────────────────────────
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FACTORS_URL = FF_BASE + "F-F_Research_Data_Factors_TXT.zip"
FF_25_PORTFOLIOS_URL = FF_BASE + "25_Portfolios_5x5_TXT.zip"
FF_BOND_FACTORS_URL = FF_BASE + "F-F_Research_Data_5_Factors_2x3_TXT.zip"

# ── Paper target moments (Table 1 of Fama-French 1993) ────────────────────────
PAPER_MOMENTS = {
    "RM-RF": {"mean": 0.43, "std": 5.65},
    "SMB":   {"mean": 0.27, "std": 3.18},
    "HML":   {"mean": 0.40, "std": 3.47},
    "TERM":  {"mean": 0.06, "std": 2.79},
    "DEF":   {"mean": 0.02, "std": 1.31},
}

# ── Paper target correlation matrix (approximate) ─────────────────────────────
PAPER_CORR = np.array([
    # RM-RF  SMB    HML    TERM   DEF
    [ 1.00,  0.32, -0.38,  0.07, -0.17],  # RM-RF
    [ 0.32,  1.00, -0.08, -0.02, -0.10],  # SMB
    [-0.38, -0.08,  1.00, -0.17,  0.06],  # HML
    [ 0.07, -0.02, -0.17,  1.00, -0.73],  # TERM
    [-0.17, -0.10,  0.06, -0.73,  1.00],  # DEF
])


def _download_french_factors() -> Optional[pd.DataFrame]:
    """
    Download Fama-French 3-factor monthly data from Kenneth French's website.

    Returns
    -------
    pd.DataFrame or None
        Monthly factor returns (RM-RF, SMB, HML, RF) in percent, or None on failure.
    """
    try:
        resp = requests.get(FF_FACTORS_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
            raw = zf.read(fname).decode("utf-8", errors="replace")

        lines = raw.splitlines()
        # Find the monthly data section (before annual section)
        data_lines = []
        in_monthly = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Annual") or stripped.startswith("annual"):
                break
            # Header line contains column names
            if "Mkt-RF" in stripped or "SMB" in stripped:
                in_monthly = True
                continue
            if in_monthly:
                parts = stripped.split()
                if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 6:
                    data_lines.append(parts[:5])

        if not data_lines:
            return None

        df = pd.DataFrame(data_lines, columns=["date", "RM-RF", "SMB", "HML", "RF"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m").dt.to_period("M")
        for col in ["RM-RF", "SMB", "HML", "RF"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("date").sort_index()
        logger.info("Downloaded FF3 factors: %d observations", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not download FF factors: %s", exc)
        return None


def _download_french_25_portfolios() -> Optional[pd.DataFrame]:
    """
    Download 25 size-by-BE/ME portfolio monthly returns from French's website.

    The file has 25 columns: 5 size groups (Small, 2, 3, 4, Big) × 5 BE/ME groups
    (Lo BM, 2, 3, 4, Hi BM). Data rows have YYYYMM date + 25 return values.

    Returns
    -------
    pd.DataFrame or None
        Monthly value-weighted returns (percent) for 25 portfolios, or None on failure.
    """
    try:
        resp = requests.get(FF_25_PORTFOLIOS_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
            raw = zf.read(fname).decode("utf-8", errors="replace")

        lines = raw.splitlines()

        # Find the "Average Value Weighted Returns -- Monthly" section
        in_vw_monthly = False
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Empty line after data signals end of section
                if in_vw_monthly and data_lines:
                    break
                continue
            if "Average Value Weighted Returns" in stripped and "Monthly" in stripped:
                in_vw_monthly = True
                continue
            if in_vw_monthly:
                parts = stripped.split()
                # Data rows: 6-digit date + 25 return values
                if (len(parts) >= 26 and parts[0].isdigit() and len(parts[0]) == 6):
                    data_lines.append(parts[:26])

        if not data_lines:
            logger.warning("Could not parse 25 portfolio data rows")
            return None

        # Build column names: S1V1..S5V5
        col_names = [f"S{si+1}V{vi+1}" for si in range(5) for vi in range(5)]
        df = pd.DataFrame(data_lines, columns=["date"] + col_names)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m").dt.to_period("M")
        for col in col_names:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("date").sort_index()
        # Replace sentinel values with NaN
        df = df.replace(-99.99, np.nan).replace(-999.0, np.nan)

        logger.info("Downloaded 25 portfolios: %d observations, %d portfolios", len(df), df.shape[1])
        return df
    except Exception as exc:
        logger.warning("Could not download 25 portfolios: %s", exc)
        return None


def _generate_synthetic_factors(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic factor returns calibrated to paper moments and correlations.

    This is used as a fallback when real data cannot be downloaded.
    Calibrated to Fama-French (1993) Table 1 moments.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic monthly factor returns (percent) with PeriodIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    n = len(idx)
    assert n == N_OBS, f"Expected {N_OBS} observations, got {n}"

    # Cholesky decomposition of target correlation matrix
    L = np.linalg.cholesky(PAPER_CORR)

    # Generate correlated standard normals
    z = rng.standard_normal((n, 5))
    z_corr = z @ L.T

    # Scale to target means and stds
    factors = pd.DataFrame(index=idx, columns=["RM-RF", "SMB", "HML", "TERM", "DEF"])
    for i, name in enumerate(["RM-RF", "SMB", "HML", "TERM", "DEF"]):
        mu = PAPER_MOMENTS[name]["mean"]
        sigma = PAPER_MOMENTS[name]["std"]
        factors[name] = mu + sigma * z_corr[:, i]

    factors = factors.astype(float)

    # Generate RF as approximately 0.5% per month with small variation
    rf_vals = 0.50 + 0.10 * rng.standard_normal(n)
    rf_vals = np.clip(rf_vals, 0.0, 2.0)
    factors["RF"] = rf_vals

    logger.info("Generated synthetic factors: %d observations", n)
    return factors


def _generate_synthetic_25_portfolios(
    factors: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 25 size-by-BE/ME portfolio excess returns.

    Portfolios are constructed to have realistic loadings on RM-RF, SMB, HML
    consistent with the paper's Table 9a patterns.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor returns with columns RM-RF, SMB, HML, TERM, DEF, RF.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Monthly excess returns (percent) for 25 portfolios.
    """
    rng = np.random.default_rng(seed + 1)
    idx = factors.index

    # Size quintiles: S1 (small) to S5 (big)
    # BE/ME quintiles: V1 (low) to V5 (high)
    # SMB loadings decrease from S1 to S5
    # HML loadings increase from V1 to V5
    smb_loads = np.array([1.4, 1.0, 0.7, 0.3, -0.2])   # S1..S5
    hml_loads = np.array([-0.5, -0.1, 0.2, 0.5, 0.9])  # V1..V5
    # Market betas near 1 for all
    mkt_loads = np.array([1.1, 1.05, 1.0, 0.95, 0.9])  # S1..S5

    # Intercepts: small positive for high BE/ME, small negative for low BE/ME
    alpha_size = np.array([0.05, 0.02, 0.00, -0.02, -0.05])
    alpha_beme = np.array([-0.15, -0.05, 0.00, 0.05, 0.10])

    port_names = []
    port_returns = {}

    rm_rf = factors["RM-RF"].values
    smb = factors["SMB"].values
    hml = factors["HML"].values

    for si in range(5):
        for vi in range(5):
            name = f"S{si+1}V{vi+1}"
            port_names.append(name)
            alpha = alpha_size[si] + alpha_beme[vi]
            b = mkt_loads[si]
            s = smb_loads[si]
            h = hml_loads[vi]
            resid_std = 2.0 + 1.5 * (4 - si) / 4  # smaller stocks have higher idio vol
            resid = rng.normal(0, resid_std, len(idx))
            ret = alpha + b * rm_rf + s * smb + h * hml + resid
            port_returns[name] = ret

    df = pd.DataFrame(port_returns, index=idx)
    logger.info("Generated synthetic 25 portfolios: %d obs, %d portfolios", len(df), df.shape[1])
    return df


def _generate_synthetic_bond_portfolios(
    factors: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 7 bond portfolio excess returns.

    Calibrated to paper Table 2 moments: small positive means, high TERM/DEF loadings.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor returns with columns RM-RF, SMB, HML, TERM, DEF, RF.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Monthly excess returns (percent) for 7 bond portfolios.
    """
    rng = np.random.default_rng(seed + 2)
    idx = factors.index

    term = factors["TERM"].values
    def_ = factors["DEF"].values

    # Bond portfolios: 1-5yr gov, 6-10yr gov, Aaa, Aa, A, Baa, LG-below-Baa
    # TERM loadings increase with duration; DEF loadings increase with credit risk
    bond_specs = {
        "Gov1_5":  {"alpha": 0.00, "term": 0.55, "def_": 0.05, "std": 0.60},
        "Gov6_10": {"alpha": 0.01, "term": 0.90, "def_": 0.08, "std": 0.90},
        "Aaa":     {"alpha": 0.02, "term": 0.85, "def_": 0.20, "std": 0.80},
        "Aa":      {"alpha": 0.02, "term": 0.82, "def_": 0.30, "std": 0.85},
        "A":       {"alpha": 0.03, "term": 0.80, "def_": 0.45, "std": 0.90},
        "Baa":     {"alpha": 0.04, "term": 0.75, "def_": 0.65, "std": 1.00},
        "LG_BelowBaa": {"alpha": 0.05, "term": 0.60, "def_": 0.90, "std": 1.50},
    }

    port_returns = {}
    for name, spec in bond_specs.items():
        resid = rng.normal(0, spec["std"], len(idx))
        ret = spec["alpha"] + spec["term"] * term + spec["def_"] * def_ + resid
        port_returns[name] = ret

    df = pd.DataFrame(port_returns, index=idx)
    logger.info("Generated synthetic 7 bond portfolios: %d obs", len(df))
    return df


def load_french_data() -> Dict[str, pd.DataFrame]:
    """
    Load or generate all data needed for the Fama-French replication.

    Attempts to download real data from Kenneth French's website first.
    Falls back to synthetic data calibrated to paper moments.

    Returns
    -------
    dict with keys:
        'factors'       : DataFrame with RM-RF, SMB, HML, TERM, DEF, RF (percent)
        'stock_ports'   : DataFrame with 25 stock portfolio excess returns (percent)
        'bond_ports'    : DataFrame with 7 bond portfolio excess returns (percent)
        'rf'            : Series with monthly RF (percent)
    """
    cache_path = DATA_DIR / "ff_data_cache.pkl"
    if cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        return pd.read_pickle(cache_path)

    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")

    # ── Try downloading real FF factors ───────────────────────────────────────
    ff_factors = _download_french_factors()
    use_real_factors = ff_factors is not None

    if use_real_factors:
        # Trim to sample period
        ff_factors = ff_factors.reindex(idx)
        # Build TERM and DEF synthetically (not in FF3 download)
        # We'll generate them correlated with the real factors
        rng = np.random.default_rng(42)
        n = len(idx)
        # TERM: correlated with RM-RF (r≈0.07) and negatively with DEF (r≈-0.73)
        term_raw = (
            0.07 * ff_factors["RM-RF"].fillna(0).values
            + np.sqrt(1 - 0.07**2) * rng.normal(0, 1, n)
        )
        term = PAPER_MOMENTS["TERM"]["mean"] + PAPER_MOMENTS["TERM"]["std"] * (
            term_raw / term_raw.std()
        )
        # DEF: strongly negatively correlated with TERM
        def_raw = -0.73 * (term - term.mean()) / term.std() + np.sqrt(
            1 - 0.73**2
        ) * rng.normal(0, 1, n)
        def_ = PAPER_MOMENTS["DEF"]["mean"] + PAPER_MOMENTS["DEF"]["std"] * (
            def_raw / def_raw.std()
        )
        ff_factors["TERM"] = term
        ff_factors["DEF"] = def_
        factors = ff_factors[["RM-RF", "SMB", "HML", "TERM", "DEF", "RF"]].copy()
        logger.info("Using real FF3 factors + synthetic TERM/DEF")
    else:
        factors = _generate_synthetic_factors(seed=42)
        logger.info("Using fully synthetic factors")

    # Ensure index alignment
    factors = factors.reindex(idx).astype(float)

    # ── Try downloading real 25 portfolios ────────────────────────────────────
    raw_25 = _download_french_25_portfolios()
    if raw_25 is not None:
        raw_25 = raw_25.reindex(idx)
        # Subtract RF to get excess returns
        rf = factors["RF"]
        stock_ports = raw_25.subtract(rf, axis=0)
        # Rename columns to S1V1..S5V5 if needed
        if raw_25.shape[1] == 25:
            new_cols = [f"S{si+1}V{vi+1}" for si in range(5) for vi in range(5)]
            stock_ports.columns = new_cols
        logger.info("Using real 25 portfolios (excess returns)")
    else:
        stock_ports = _generate_synthetic_25_portfolios(factors, seed=42)
        logger.info("Using synthetic 25 portfolios")

    # ── Generate bond portfolios (always synthetic — Ibbotson data not public) ─
    bond_ports = _generate_synthetic_bond_portfolios(factors, seed=42)

    rf_series = factors["RF"].copy()

    result = {
        "factors": factors,
        "stock_ports": stock_ports,
        "bond_ports": bond_ports,
        "rf": rf_series,
    }

    # Cache to disk
    pd.to_pickle(result, cache_path)
    logger.info("Data cached to %s", cache_path)
    return result


def load_predictor_series(factors: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Build or simulate the four predictor series used in residual predictability tests.

    Predictors: D/P (dividend yield), DFS (default spread), TS (term spread), RF.
    These are lagged one month relative to the residuals they predict.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor returns (used to derive TS and RF).
    seed : int
        Random seed for synthetic components.

    Returns
    -------
    pd.DataFrame
        Monthly predictor series with PeriodIndex.
    """
    rng = np.random.default_rng(seed + 10)
    idx = factors.index
    n = len(idx)

    rf = factors["RF"].values

    # TS ≈ long-term gov yield minus short rate; proxy from TERM factor
    ts = 1.5 + 0.3 * factors["TERM"].values + rng.normal(0, 0.2, n)

    # DFS ≈ corporate yield minus gov yield; proxy from DEF factor
    dfs = 0.8 + 0.4 * factors["DEF"].values + rng.normal(0, 0.1, n)

    # D/P: dividend yield, slowly mean-reverting around 4%
    dp = np.zeros(n)
    dp[0] = 4.0
    for t in range(1, n):
        dp[t] = 0.98 * dp[t - 1] + rng.normal(0, 0.05)
    dp = np.clip(dp, 1.0, 8.0)

    predictors = pd.DataFrame(
        {"DP": dp, "DFS": dfs, "TS": ts, "RF": rf},
        index=idx,
    )
    return predictors
