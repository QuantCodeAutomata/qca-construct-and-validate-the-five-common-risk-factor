"""
Experiment 3: Test cross-sectional pricing through individual and joint intercepts.

Implements:
- Individual alpha tests (t-statistics for each portfolio intercept)
- GRS-style joint F-test across 32 portfolios
- Paired-month bootstrap with 10,000 resamples for finite-sample inference

Reference: Gibbons, Ross, and Shanken (1989); Fama and French (1993) Section 5.

Using statsmodels OLS — Context7 confirmed
Custom GRS implementation — Context7 found no library equivalent (paper Eq. 5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from src.regressions import MODEL_SPECS, MODEL_LABELS, run_single_regression
from src.utils import RESULTS_DIR, N_OBS, save_table

logger = logging.getLogger(__name__)

# ── Bootstrap parameters ───────────────────────────────────────────────────────
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

# ── Paper target F-statistics (approximate, Table 5 of FF 1993) ───────────────
PAPER_F_STATS = {
    "bond_factor":       2.09,
    "market_only":       1.91,
    "style_factor":      1.78,
    "three_factor":      1.56,
    "five_factor":       1.66,
    "ortho_five_factor": 1.66,
}


def compute_grs_fstat(
    alphas: np.ndarray,
    residuals: np.ndarray,
    X: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute the GRS-style joint F-statistic for zero intercepts.

    Formula (Fama-French 1993, Section 5):
        F = [(A' Σ^{-1} A) * (N - K - L + 1)] / [L * (N - K) * ω_{1,1}]

    where:
        N = number of time-series observations
        L = number of test assets
        K = number of regressors (including constant)
        A = vector of intercepts
        Σ = unbiased residual covariance matrix (L × L)
        ω_{1,1} = (1,1) element of (X'X)^{-1} (intercept variance scaling)

    Custom implementation — Context7 found no library equivalent (paper Eq. 5)

    Parameters
    ----------
    alphas : np.ndarray, shape (L,)
        Vector of OLS intercepts.
    residuals : np.ndarray, shape (N, L)
        Matrix of OLS residuals.
    X : np.ndarray, shape (N, K)
        Common regressor matrix (including constant column).

    Returns
    -------
    f_stat : float
        GRS F-statistic.
    p_value : float
        Theoretical p-value from F(L, N-K-L+1) distribution.
    df_num : float
        Numerator degrees of freedom (L).
    """
    N, L = residuals.shape
    K = X.shape[1]  # includes constant

    # Unbiased residual covariance matrix
    Sigma = (residuals.T @ residuals) / (N - K)

    # Regularize if near-singular
    cond = np.linalg.cond(Sigma)
    if cond > 1e10:
        logger.warning("Sigma condition number %.2e — adding ridge regularization", cond)
        Sigma += np.eye(L) * 1e-6 * np.trace(Sigma) / L

    Sigma_inv = np.linalg.inv(Sigma)

    # ω_{1,1}: (1,1) element of (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    omega_11 = XtX_inv[0, 0]

    # Quadratic form A' Σ^{-1} A
    quad_form = alphas @ Sigma_inv @ alphas

    # GRS F-statistic
    df_num = L
    df_denom = N - K - L + 1
    if df_denom <= 0:
        logger.warning("Negative denominator df: N=%d, K=%d, L=%d", N, K, L)
        df_denom = 1

    f_stat = (quad_form * df_denom) / (L * (N - K) * omega_11)

    # Theoretical p-value
    p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_denom)

    return float(f_stat), float(p_value), float(df_num)


def run_joint_alpha_test(
    portfolios: pd.DataFrame,
    factors: pd.DataFrame,
    factor_names: List[str],
) -> Dict:
    """
    Run the GRS joint alpha test for a given model specification.

    Parameters
    ----------
    portfolios : pd.DataFrame
        Monthly excess returns for all test portfolios.
    factors : pd.DataFrame
        Factor returns.
    factor_names : list of str
        Factor columns to use.

    Returns
    -------
    dict
        Contains: alphas, alpha_tstats, f_stat, p_value, residuals, X.
    """
    # Align data
    X_df = sm.add_constant(factors[factor_names])
    combined = pd.concat([portfolios, X_df], axis=1).dropna()
    port_cols = portfolios.columns.tolist()
    factor_cols_with_const = ["const"] + factor_names

    Y = combined[port_cols].values  # (N, L)
    X = combined[factor_cols_with_const].values  # (N, K)

    N, L = Y.shape
    K = X.shape[1]

    # OLS for each portfolio
    alphas = np.zeros(L)
    alpha_tstats = np.zeros(L)
    residuals = np.zeros((N, L))

    for j in range(L):
        result = sm.OLS(Y[:, j], X).fit()
        alphas[j] = result.params[0]  # intercept
        alpha_tstats[j] = result.tvalues[0]
        residuals[:, j] = result.resid

    # GRS F-statistic
    f_stat, p_value, df_num = compute_grs_fstat(alphas, residuals, X)

    logger.info(
        "GRS F-stat=%.4f, p-value=%.4f, df=(%d, %d)",
        f_stat, p_value, int(df_num), N - K - L + 1,
    )

    return {
        "alphas": alphas,
        "alpha_tstats": alpha_tstats,
        "f_stat": f_stat,
        "p_value": p_value,
        "residuals": residuals,
        "X": X,
        "Y": Y,
        "port_names": port_cols,
        "factor_names": factor_names,
        "N": N,
        "L": L,
        "K": K,
    }


def bootstrap_joint_alpha_test(
    joint_test_result: Dict,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """
    Bootstrap the GRS joint F-statistic under the null of zero intercepts.

    Procedure (Fama-French 1993, Section 5):
    1. Fix slopes at estimated values, set intercepts to zero.
    2. Generate null returns: Y_null = X_slopes @ beta_hat + residuals.
    3. Resample 342 paired (Y_null, X) observations with replacement.
    4. Re-estimate regressions and compute F-statistic.
    5. Repeat 10,000 times.
    6. Bootstrap p-value = fraction of bootstrap F-stats < empirical F-stat.

    Custom implementation — Context7 found no library equivalent (paper Section 5)

    Parameters
    ----------
    joint_test_result : dict
        Output of run_joint_alpha_test().
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Bootstrap distribution, empirical F-stat, bootstrap p-value.
    """
    rng = np.random.default_rng(seed)

    alphas = joint_test_result["alphas"]
    residuals = joint_test_result["residuals"]
    X = joint_test_result["X"]
    Y = joint_test_result["Y"]
    N, L = Y.shape
    K = X.shape[1]
    empirical_f = joint_test_result["f_stat"]

    # Estimate slopes (excluding intercept) for each portfolio
    # Y = X @ [alpha; beta] => slopes are params[1:]
    slopes = np.zeros((K, L))
    for j in range(L):
        result = sm.OLS(Y[:, j], X).fit()
        slopes[:, j] = result.params

    # Generate null returns: set intercept to 0, keep slopes
    slopes_null = slopes.copy()
    slopes_null[0, :] = 0.0  # zero out intercepts
    Y_null = X @ slopes_null + residuals  # (N, L)

    # Bootstrap
    bootstrap_f_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample paired observations
        idx = rng.integers(0, N, size=N)
        Y_boot = Y_null[idx, :]
        X_boot = X[idx, :]

        # Re-estimate regressions
        alphas_boot = np.zeros(L)
        resid_boot = np.zeros((N, L))
        for j in range(L):
            res = sm.OLS(Y_boot[:, j], X_boot).fit()
            alphas_boot[j] = res.params[0]
            resid_boot[:, j] = res.resid

        # Compute F-statistic
        try:
            f_b, _, _ = compute_grs_fstat(alphas_boot, resid_boot, X_boot)
            bootstrap_f_stats[b] = f_b
        except np.linalg.LinAlgError:
            bootstrap_f_stats[b] = np.nan

    # Remove NaN bootstrap samples
    valid = bootstrap_f_stats[~np.isnan(bootstrap_f_stats)]
    bootstrap_p = float((valid < empirical_f).mean())

    logger.info(
        "Bootstrap: empirical F=%.4f, bootstrap p=%.4f (n_valid=%d/%d)",
        empirical_f, bootstrap_p, len(valid), n_bootstrap,
    )

    return {
        "empirical_f": empirical_f,
        "bootstrap_f_stats": bootstrap_f_stats,
        "bootstrap_p": bootstrap_p,
        "n_valid": len(valid),
    }


def run_all_joint_tests(
    stock_ports: pd.DataFrame,
    bond_ports: pd.DataFrame,
    factors: pd.DataFrame,
    run_bootstrap: bool = True,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict:
    """
    Run GRS joint alpha tests for all model specifications.

    Parameters
    ----------
    stock_ports : pd.DataFrame
        25 stock portfolio excess returns.
    bond_ports : pd.DataFrame
        7 bond portfolio excess returns.
    factors : pd.DataFrame
        Factor returns.
    run_bootstrap : bool
        Whether to run bootstrap (slow).
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict
        Results for each model.
    """
    all_ports = pd.concat([stock_ports, bond_ports], axis=1)
    results = {}

    for model_name, factor_names in MODEL_SPECS.items():
        missing = [f for f in factor_names if f not in factors.columns]
        if missing:
            logger.warning("Skipping %s: missing factors %s", model_name, missing)
            continue

        logger.info("Joint alpha test: %s", MODEL_LABELS[model_name])

        joint_result = run_joint_alpha_test(all_ports, factors, factor_names)

        if run_bootstrap:
            boot_result = bootstrap_joint_alpha_test(
                joint_result, n_bootstrap=n_bootstrap, seed=BOOTSTRAP_SEED
            )
        else:
            boot_result = {"bootstrap_p": np.nan, "empirical_f": joint_result["f_stat"]}

        results[model_name] = {
            "joint_test": joint_result,
            "bootstrap": boot_result,
        }

    return results


def build_alpha_summary_table(all_joint_tests: Dict) -> pd.DataFrame:
    """
    Build a summary table of F-statistics and p-values across models.

    Parameters
    ----------
    all_joint_tests : dict
        Output of run_all_joint_tests().

    Returns
    -------
    pd.DataFrame
        Summary table with F-stat, theoretical p-value, bootstrap p-value.
    """
    rows = []
    for model_name, res in all_joint_tests.items():
        jt = res["joint_test"]
        bt = res["bootstrap"]
        paper_f = PAPER_F_STATS.get(model_name, np.nan)
        rows.append({
            "Model": MODEL_LABELS.get(model_name, model_name),
            "F-stat": jt["f_stat"],
            "Theoretical p-value": jt["p_value"],
            "Bootstrap p-value": bt.get("bootstrap_p", np.nan),
            "Paper F-stat (approx)": paper_f,
        })
    return pd.DataFrame(rows).set_index("Model")


def build_individual_alpha_table(
    all_joint_tests: Dict,
    port_names: List[str],
) -> pd.DataFrame:
    """
    Build a table of individual alphas and t-statistics across models.

    Parameters
    ----------
    all_joint_tests : dict
        Output of run_all_joint_tests().
    port_names : list of str
        Portfolio names.

    Returns
    -------
    pd.DataFrame
        Multi-column table: alpha and t-stat for each model.
    """
    frames = {}
    for model_name, res in all_joint_tests.items():
        jt = res["joint_test"]
        label = MODEL_LABELS.get(model_name, model_name)
        frames[f"{label}_alpha"] = pd.Series(
            jt["alphas"], index=jt["port_names"]
        )
        frames[f"{label}_t"] = pd.Series(
            jt["alpha_tstats"], index=jt["port_names"]
        )
    return pd.DataFrame(frames)


def run_experiment_3(
    exp1_results: Dict,
    exp2_results: Dict,
    run_bootstrap: bool = True,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict:
    """
    Run Experiment 3: joint and individual alpha tests.

    Parameters
    ----------
    exp1_results : dict
        Output of run_experiment_1().
    exp2_results : dict
        Output of run_experiment_2().
    run_bootstrap : bool
        Whether to run bootstrap.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict
        All test results and validation.
    """
    logger.info("=== Experiment 3: Joint Alpha Tests (GRS + Bootstrap) ===")

    factors = exp1_results["factors"]
    stock_ports = exp1_results["stock_ports"]
    bond_ports = exp1_results["bond_ports"]

    # Run all joint tests
    all_joint_tests = run_all_joint_tests(
        stock_ports, bond_ports, factors,
        run_bootstrap=run_bootstrap,
        n_bootstrap=n_bootstrap,
    )

    # Build summary tables
    summary_table = build_alpha_summary_table(all_joint_tests)
    all_port_names = list(stock_ports.columns) + list(bond_ports.columns)
    individual_alpha_table = build_individual_alpha_table(all_joint_tests, all_port_names)

    logger.info("\nJoint Alpha Test Summary:\n%s", summary_table.to_string())

    # ── Validation ────────────────────────────────────────────────────────────
    validation = {}
    if "three_factor" in all_joint_tests:
        tf_f = all_joint_tests["three_factor"]["joint_test"]["f_stat"]
        validation["three_factor_f_reasonable"] = bool(tf_f < 5.0)
        validation["three_factor_f_value"] = float(tf_f)

    if "five_factor" in all_joint_tests:
        ff_f = all_joint_tests["five_factor"]["joint_test"]["f_stat"]
        validation["five_factor_f_reasonable"] = bool(ff_f < 5.0)
        validation["five_factor_f_value"] = float(ff_f)

    # Check that three-factor has lower F than bond-only (paper finding)
    if "three_factor" in all_joint_tests and "bond_factor" in all_joint_tests:
        tf_f = all_joint_tests["three_factor"]["joint_test"]["f_stat"]
        bf_f = all_joint_tests["bond_factor"]["joint_test"]["f_stat"]
        validation["three_factor_better_than_bond_only"] = bool(tf_f < bf_f)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_table(summary_table, "exp3_joint_alpha_summary.csv", "Joint Alpha Test Summary")
    save_table(individual_alpha_table, "exp3_individual_alphas.csv", "Individual Alphas")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_f_stat_comparison(summary_table)
    _plot_bootstrap_distributions(all_joint_tests)
    _plot_individual_alphas(all_joint_tests)

    return {
        "all_joint_tests": all_joint_tests,
        "summary_table": summary_table,
        "individual_alpha_table": individual_alpha_table,
        "validation": validation,
    }


def _plot_f_stat_comparison(summary_table: pd.DataFrame) -> None:
    """Plot F-statistics across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = summary_table.index.tolist()
    f_stats = summary_table["F-stat"].values
    paper_f = summary_table["Paper F-stat (approx)"].values

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width / 2, f_stats, width, label="Computed F-stat", color="steelblue")
    bars2 = ax.bar(x + width / 2, paper_f, width, label="Paper F-stat (approx)", color="darkorange", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.split("(")[0].strip() for m in models],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("GRS F-statistic")
    ax.set_title("Joint Alpha Test: GRS F-Statistics by Model", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_f_stat_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved F-stat comparison plot")


def _plot_bootstrap_distributions(all_joint_tests: Dict) -> None:
    """Plot bootstrap F-statistic distributions."""
    n_models = len(all_joint_tests)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for i, (model_name, res) in enumerate(all_joint_tests.items()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        bt = res["bootstrap"]
        boot_f = bt.get("bootstrap_f_stats", np.array([]))
        emp_f = bt.get("empirical_f", res["joint_test"]["f_stat"])

        if len(boot_f) > 0:
            valid = boot_f[~np.isnan(boot_f)]
            ax.hist(valid, bins=50, density=True, alpha=0.7, color="steelblue", label="Bootstrap")
            ax.axvline(emp_f, color="red", linewidth=2, linestyle="--", label=f"Empirical F={emp_f:.3f}")
            boot_p = bt.get("bootstrap_p", np.nan)
            ax.set_title(
                f"{MODEL_LABELS.get(model_name, model_name)[:30]}\nBoot p={boot_p:.3f}",
                fontsize=9,
            )
        else:
            ax.text(0.5, 0.5, "No bootstrap", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(MODEL_LABELS.get(model_name, model_name)[:30], fontsize=9)

        ax.set_xlabel("F-statistic")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Bootstrap Distributions of GRS F-Statistics", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_bootstrap_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved bootstrap distributions plot")


def _plot_individual_alphas(all_joint_tests: Dict) -> None:
    """Plot individual alphas for three-factor and five-factor models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, model_name, title in [
        (axes[0], "three_factor", "Three-Factor Model"),
        (axes[1], "five_factor", "Five-Factor Model"),
    ]:
        if model_name not in all_joint_tests:
            continue
        jt = all_joint_tests[model_name]["joint_test"]
        alphas = jt["alphas"]
        alpha_t = jt["alpha_tstats"]
        port_names = jt["port_names"]

        colors = ["red" if abs(t) > 2.0 else "steelblue" for t in alpha_t]
        ax.bar(range(len(alphas)), alphas, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(port_names, rotation=90, fontsize=7)
        ax.set_title(f"Individual Alphas — {title}\n(Red = |t| > 2)", fontsize=10)
        ax.set_xlabel("Portfolio")
        ax.set_ylabel("Alpha (%/month)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Individual Pricing Errors (Alphas) by Portfolio", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_individual_alphas.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved individual alphas plot")
