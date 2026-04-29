"""
Experiment 1: Construct and validate the five common risk factors and dependent
test portfolios in U.S. stock and bond returns.

Implements:
- RM-RF, SMB, HML (stock factors)
- TERM, DEF (bond factors)
- RMO (orthogonalized market factor)
- 25 size-by-BE/ME stock portfolios
- 7 bond excess-return portfolios
- Validation against paper moments (Fama-French 1993, Table 1)

Using statsmodels OLS — Context7 confirmed
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from src.utils import (
    RESULTS_DIR,
    SAMPLE_END,
    SAMPLE_START,
    N_OBS,
    compute_summary_stats,
    save_table,
)

logger = logging.getLogger(__name__)

# ── Paper target moments (Fama-French 1993, Table 1) ──────────────────────────
PAPER_FACTOR_MEANS = {
    "RM-RF": 0.43,
    "SMB":   0.27,
    "HML":   0.40,
    "TERM":  0.06,
    "DEF":   0.02,
}
PAPER_FACTOR_STDS = {
    "RM-RF": 5.65,
    "SMB":   3.18,
    "HML":   3.47,
    "TERM":  2.79,
    "DEF":   1.31,
}

# ── Paper target RMO regression coefficients (Section 3) ─────────────────────
PAPER_RMO_INTERCEPT = 0.50
PAPER_RMO_SMB_LOAD  = 0.44
PAPER_RMO_HML_LOAD  = -0.63
PAPER_RMO_TERM_LOAD = 0.81
PAPER_RMO_DEF_LOAD  = 0.79
PAPER_RMO_R2        = 0.38


def construct_rmo(factors: pd.DataFrame) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Construct the orthogonalized market factor RMO.

    Full-sample regression of RM-RF on SMB, HML, TERM, DEF.
    RMO = intercept_hat + residual_hat.

    This removes the linear dependence of RM-RF on the other four factors,
    making the bond-factor exposures in stock returns more visible.

    Parameters
    ----------
    factors : pd.DataFrame
        Must contain columns: RM-RF, SMB, HML, TERM, DEF.

    Returns
    -------
    rmo : pd.Series
        Orthogonalized market factor (percent, monthly).
    result : RegressionResultsWrapper
        Full OLS result for validation.
    """
    y = factors["RM-RF"]
    X = sm.add_constant(factors[["SMB", "HML", "TERM", "DEF"]])
    result = sm.OLS(y, X, missing="drop").fit()

    # RMO = constant + residual (strips out SMB/HML/TERM/DEF variation)
    rmo = result.params["const"] + result.resid
    rmo.name = "RMO"

    logger.info(
        "RMO regression: intercept=%.3f, SMB=%.3f, HML=%.3f, TERM=%.3f, DEF=%.3f, R2=%.3f",
        result.params["const"],
        result.params["SMB"],
        result.params["HML"],
        result.params["TERM"],
        result.params["DEF"],
        result.rsquared,
    )
    return rmo, result


def build_all_factors(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Assemble the full factor DataFrame including RMO.

    Parameters
    ----------
    data : dict
        Output of data_loader.load_french_data().

    Returns
    -------
    pd.DataFrame
        Monthly factor returns (percent) with columns:
        RM-RF, SMB, HML, TERM, DEF, RMO, RF.
    """
    factors = data["factors"].copy()
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    factors = factors.reindex(idx)

    # Validate required columns
    required = ["RM-RF", "SMB", "HML", "TERM", "DEF", "RF"]
    missing = [c for c in required if c not in factors.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")

    # Construct RMO
    rmo, rmo_result = construct_rmo(factors)
    factors["RMO"] = rmo

    logger.info("Factor DataFrame shape: %s", factors.shape)
    return factors, rmo_result


def compute_factor_correlation_matrix(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix for the six factors.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor returns with columns RM-RF, SMB, HML, TERM, DEF, RMO.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    cols = ["RM-RF", "RMO", "SMB", "HML", "TERM", "DEF"]
    available = [c for c in cols if c in factors.columns]
    return factors[available].corr()


def validate_factors(factors: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate factor moments against paper targets.

    Checks that means and standard deviations are within reasonable tolerance
    of the paper's reported values (Fama-French 1993, Table 1).

    Parameters
    ----------
    factors : pd.DataFrame
        Factor returns (percent, monthly).

    Returns
    -------
    dict
        Validation results: True if within tolerance, False otherwise.
    """
    results = {}
    tol_mean = 0.30  # percentage points tolerance on mean
    tol_std = 1.50   # percentage points tolerance on std

    for factor, target_mean in PAPER_FACTOR_MEANS.items():
        if factor not in factors.columns:
            continue
        actual_mean = factors[factor].mean()
        actual_std = factors[factor].std(ddof=1)
        target_std = PAPER_FACTOR_STDS[factor]

        mean_ok = abs(actual_mean - target_mean) < tol_mean
        std_ok = abs(actual_std - target_std) < tol_std
        results[f"{factor}_mean_ok"] = mean_ok
        results[f"{factor}_std_ok"] = std_ok

        logger.info(
            "%s: mean=%.3f (target %.3f, ok=%s), std=%.3f (target %.3f, ok=%s)",
            factor, actual_mean, target_mean, mean_ok,
            actual_std, target_std, std_ok,
        )

    # Check correlation signs
    corr = factors[["RM-RF", "SMB", "HML", "TERM", "DEF"]].corr()
    results["corr_RMRF_SMB_positive"] = corr.loc["RM-RF", "SMB"] > 0
    results["corr_RMRF_HML_negative"] = corr.loc["RM-RF", "HML"] < 0
    results["corr_SMB_HML_near_neg008"] = abs(corr.loc["SMB", "HML"] + 0.08) < 0.15
    results["corr_TERM_DEF_strongly_negative"] = corr.loc["TERM", "DEF"] < -0.50

    return results


def validate_rmo_regression(rmo_result: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, bool]:
    """
    Validate the RMO regression coefficients against paper targets.

    Parameters
    ----------
    rmo_result : RegressionResultsWrapper
        OLS result from construct_rmo().

    Returns
    -------
    dict
        Validation results.
    """
    params = rmo_result.params
    r2 = rmo_result.rsquared
    tol = 0.25  # tolerance on loadings

    results = {
        "intercept_ok": abs(params.get("const", 0) - PAPER_RMO_INTERCEPT) < tol,
        "smb_load_ok":  abs(params.get("SMB", 0) - PAPER_RMO_SMB_LOAD) < tol,
        "hml_load_ok":  abs(params.get("HML", 0) - PAPER_RMO_HML_LOAD) < tol,
        "term_load_ok": abs(params.get("TERM", 0) - PAPER_RMO_TERM_LOAD) < tol,
        "def_load_ok":  abs(params.get("DEF", 0) - PAPER_RMO_DEF_LOAD) < tol,
        "r2_ok":        abs(r2 - PAPER_RMO_R2) < 0.15,
    }
    return results


def compute_portfolio_summary(
    stock_ports: pd.DataFrame,
    bond_ports: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean excess returns for the 25 stock and 7 bond portfolios.

    Parameters
    ----------
    stock_ports : pd.DataFrame
        Monthly excess returns for 25 stock portfolios.
    bond_ports : pd.DataFrame
        Monthly excess returns for 7 bond portfolios.

    Returns
    -------
    stock_means : pd.DataFrame
        5x5 matrix of mean excess returns (percent).
    bond_means : pd.Series
        Mean excess returns for 7 bond portfolios.
    """
    # Stock portfolio means reshaped to 5x5
    stock_means_flat = stock_ports.mean()
    # Reshape to 5x5 (size rows, BE/ME columns)
    size_labels = ["Small", "2", "3", "4", "Big"]
    beme_labels = ["Low", "2", "3", "4", "High"]
    vals = stock_means_flat.values.reshape(5, 5)
    stock_means = pd.DataFrame(vals, index=size_labels, columns=beme_labels)

    bond_means = bond_ports.mean()
    return stock_means, bond_means


def validate_portfolio_moments(
    stock_means: pd.DataFrame,
    bond_means: pd.Series,
) -> Dict[str, bool]:
    """
    Validate portfolio mean excess returns against paper ranges.

    Paper reports stock portfolio means approximately 0.32% to 1.05%,
    bond portfolio means below about 0.15%.

    Parameters
    ----------
    stock_means : pd.DataFrame
        5x5 mean excess returns for stock portfolios.
    bond_means : pd.Series
        Mean excess returns for bond portfolios.

    Returns
    -------
    dict
        Validation results.
    """
    flat = stock_means.values.flatten()
    results = {
        "stock_means_in_range": bool(
            (flat >= 0.10).all() and (flat <= 1.50).all()
        ),
        "stock_beme_pattern": bool(
            stock_means.iloc[:, -1].mean() > stock_means.iloc[:, 0].mean()
        ),
        "bond_means_small": bool((bond_means.abs() < 0.30).all()),
    }
    return results


def run_experiment_1(data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run Experiment 1: construct and validate all factors and portfolios.

    Parameters
    ----------
    data : dict
        Output of data_loader.load_french_data().

    Returns
    -------
    dict
        All constructed series and validation results.
    """
    logger.info("=== Experiment 1: Factor Construction and Validation ===")

    # ── Build factors ─────────────────────────────────────────────────────────
    factors, rmo_result = build_all_factors(data)
    stock_ports = data["stock_ports"].copy()
    bond_ports = data["bond_ports"].copy()

    # Align to sample period
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    factors = factors.reindex(idx)
    stock_ports = stock_ports.reindex(idx)
    bond_ports = bond_ports.reindex(idx)

    # ── Summary statistics ────────────────────────────────────────────────────
    factor_cols = ["RM-RF", "RMO", "SMB", "HML", "TERM", "DEF"]
    factor_summary = compute_summary_stats(factors[factor_cols])
    logger.info("\nFactor Summary Statistics:\n%s", factor_summary.to_string())

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr_matrix = compute_factor_correlation_matrix(factors)
    logger.info("\nFactor Correlation Matrix:\n%s", corr_matrix.to_string())

    # ── Portfolio means ───────────────────────────────────────────────────────
    stock_means, bond_means = compute_portfolio_summary(stock_ports, bond_ports)
    logger.info("\n25 Stock Portfolio Mean Excess Returns (%%/month):\n%s", stock_means.to_string())
    logger.info("\n7 Bond Portfolio Mean Excess Returns (%%/month):\n%s", bond_means.to_string())

    # ── Validation ────────────────────────────────────────────────────────────
    factor_validation = validate_factors(factors)
    rmo_validation = validate_rmo_regression(rmo_result)
    port_validation = validate_portfolio_moments(stock_means, bond_means)

    all_validation = {**factor_validation, **rmo_validation, **port_validation}
    n_pass = sum(all_validation.values())
    n_total = len(all_validation)
    logger.info("Validation: %d/%d checks passed", n_pass, n_total)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_table(factor_summary, "exp1_factor_summary.csv", "Factor Summary Statistics")
    save_table(corr_matrix, "exp1_factor_correlations.csv", "Factor Correlations")
    save_table(stock_means, "exp1_stock_portfolio_means.csv", "25 Stock Portfolio Means")
    bond_means.to_frame("Mean Excess Return (%)").to_csv(
        RESULTS_DIR / "exp1_bond_portfolio_means.csv"
    )

    # RMO regression summary
    rmo_params = pd.DataFrame({
        "Coefficient": rmo_result.params,
        "t-stat": rmo_result.tvalues,
        "p-value": rmo_result.pvalues,
    })
    save_table(rmo_params, "exp1_rmo_regression.csv", "RMO Regression")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_factor_time_series(factors)
    _plot_factor_correlations(corr_matrix)
    _plot_stock_portfolio_means(stock_means)

    return {
        "factors": factors,
        "stock_ports": stock_ports,
        "bond_ports": bond_ports,
        "rmo_result": rmo_result,
        "factor_summary": factor_summary,
        "corr_matrix": corr_matrix,
        "stock_means": stock_means,
        "bond_means": bond_means,
        "validation": all_validation,
    }


def _plot_factor_time_series(factors: pd.DataFrame) -> None:
    """Plot cumulative factor returns over the sample period."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    factor_cols = ["RM-RF", "SMB", "HML", "TERM", "DEF", "RMO"]
    colors = ["steelblue", "darkorange", "green", "red", "purple", "brown"]

    for ax, col, color in zip(axes.flatten(), factor_cols, colors):
        if col not in factors.columns:
            continue
        series = factors[col].dropna()
        cum = (1 + series / 100).cumprod() - 1
        # Convert PeriodIndex to timestamps for plotting
        x = series.index.to_timestamp()
        ax.plot(x, cum * 100, color=color, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"{col} Cumulative Return (%)", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Fama-French Factor Cumulative Returns (Jul 1963 – Dec 1991)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_factor_cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved factor cumulative returns plot")


def _plot_factor_correlations(corr_matrix: pd.DataFrame) -> None:
    """Plot factor correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Factor Correlation Matrix (Jul 1963 – Dec 1991)", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_factor_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved factor correlation heatmap")


def _plot_stock_portfolio_means(stock_means: pd.DataFrame) -> None:
    """Plot 5x5 heatmap of mean excess returns for stock portfolios."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        stock_means,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "25 Stock Portfolio Mean Excess Returns (%/month)\n"
        "Rows: Size (Small→Big), Cols: BE/ME (Low→High)",
        fontsize=11,
    )
    ax.set_xlabel("BE/ME Quintile")
    ax.set_ylabel("Size Quintile")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_stock_portfolio_means.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved stock portfolio means heatmap")
