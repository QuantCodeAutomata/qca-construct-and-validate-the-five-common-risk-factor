"""
Tests for Experiment 3: GRS joint alpha tests and bootstrap.

Verifies:
- GRS F-statistic computation is mathematically correct
- Bootstrap procedure preserves cross-sectional dependence
- F-statistics are positive
- Bootstrap p-values are in [0, 1]
- Known regression properties hold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from src.alpha_tests import (
    N_BOOTSTRAP,
    bootstrap_joint_alpha_test,
    build_alpha_summary_table,
    compute_grs_fstat,
    run_all_joint_tests,
    run_joint_alpha_test,
)
from src.data_loader import load_french_data
from src.factor_construction import build_all_factors
from src.utils import N_OBS, SAMPLE_START, SAMPLE_END


@pytest.fixture(scope="module")
def exp1_data():
    """Load data and build factors."""
    data = load_french_data()
    factors, rmo_result = build_all_factors(data)
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    factors = factors.reindex(idx)
    stock_ports = data["stock_ports"].reindex(idx)
    bond_ports = data["bond_ports"].reindex(idx)
    return factors, stock_ports, bond_ports


@pytest.fixture(scope="module")
def three_factor_joint_test(exp1_data):
    """Run three-factor joint alpha test."""
    factors, stock_ports, bond_ports = exp1_data
    all_ports = pd.concat([stock_ports, bond_ports], axis=1)
    return run_joint_alpha_test(all_ports, factors, ["RM-RF", "SMB", "HML"])


# ── GRS F-statistic computation tests ─────────────────────────────────────────

def test_grs_fstat_positive():
    """GRS F-statistic should be positive."""
    rng = np.random.default_rng(42)
    N, L, K = 100, 10, 4
    alphas = rng.normal(0, 0.1, L)
    residuals = rng.normal(0, 1, (N, L))
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])
    f_stat, p_value, _ = compute_grs_fstat(alphas, residuals, X)
    assert f_stat > 0, f"F-stat={f_stat:.4f} should be positive"


def test_grs_fstat_pvalue_in_01():
    """GRS p-value should be in [0, 1]."""
    rng = np.random.default_rng(42)
    N, L, K = 100, 10, 4
    alphas = rng.normal(0, 0.1, L)
    residuals = rng.normal(0, 1, (N, L))
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])
    _, p_value, _ = compute_grs_fstat(alphas, residuals, X)
    assert 0.0 <= p_value <= 1.0, f"p-value={p_value:.4f} out of [0,1]"


def test_grs_fstat_zero_alphas_small_fstat():
    """When all alphas are zero, F-statistic should be near zero."""
    rng = np.random.default_rng(42)
    N, L, K = 200, 10, 4
    alphas = np.zeros(L)
    residuals = rng.normal(0, 1, (N, L))
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])
    f_stat, p_value, _ = compute_grs_fstat(alphas, residuals, X)
    assert f_stat < 1.0, f"F-stat={f_stat:.4f} should be near zero for zero alphas"
    assert p_value > 0.10, f"p-value={p_value:.4f} should be large for zero alphas"


def test_grs_fstat_large_alphas_large_fstat():
    """When alphas are large, F-statistic should be large."""
    rng = np.random.default_rng(42)
    N, L, K = 200, 5, 3
    alphas = np.ones(L) * 2.0  # large alphas
    residuals = rng.normal(0, 0.5, (N, L))  # small residuals
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])
    f_stat, p_value, _ = compute_grs_fstat(alphas, residuals, X)
    assert f_stat > 5.0, f"F-stat={f_stat:.4f} should be large for large alphas"
    assert p_value < 0.01, f"p-value={p_value:.4f} should be small for large alphas"


def test_grs_fstat_degrees_of_freedom():
    """GRS F-statistic uses correct degrees of freedom."""
    rng = np.random.default_rng(42)
    N, L, K = 342, 32, 4  # paper dimensions
    alphas = rng.normal(0, 0.1, L)
    residuals = rng.normal(0, 1, (N, L))
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])
    f_stat, p_value, df_num = compute_grs_fstat(alphas, residuals, X)
    assert df_num == L, f"Numerator df={df_num}, expected {L}"


# ── Joint alpha test tests ─────────────────────────────────────────────────────

def test_joint_alpha_test_returns_required_keys(three_factor_joint_test):
    """run_joint_alpha_test returns all required keys."""
    required_keys = ["alphas", "alpha_tstats", "f_stat", "p_value", "residuals", "X", "Y"]
    for key in required_keys:
        assert key in three_factor_joint_test, f"Missing key: {key}"


def test_joint_alpha_test_alpha_count(three_factor_joint_test):
    """Joint alpha test has 32 alphas (25 stocks + 7 bonds)."""
    alphas = three_factor_joint_test["alphas"]
    assert len(alphas) == 32, f"Expected 32 alphas, got {len(alphas)}"


def test_joint_alpha_test_f_stat_positive(three_factor_joint_test):
    """Joint F-statistic should be positive."""
    f_stat = three_factor_joint_test["f_stat"]
    assert f_stat > 0, f"F-stat={f_stat:.4f} should be positive"


def test_joint_alpha_test_p_value_in_01(three_factor_joint_test):
    """Joint p-value should be in [0, 1]."""
    p_value = three_factor_joint_test["p_value"]
    assert 0.0 <= p_value <= 1.0, f"p-value={p_value:.4f} out of [0,1]"


def test_joint_alpha_test_residuals_shape(three_factor_joint_test):
    """Residuals matrix has correct shape."""
    residuals = three_factor_joint_test["residuals"]
    N = three_factor_joint_test["N"]
    L = three_factor_joint_test["L"]
    assert residuals.shape == (N, L), (
        f"Residuals shape {residuals.shape} != ({N}, {L})"
    )


def test_joint_alpha_test_residuals_near_zero_mean(three_factor_joint_test):
    """OLS residuals should have mean near zero."""
    residuals = three_factor_joint_test["residuals"]
    means = residuals.mean(axis=0)
    assert (np.abs(means) < 0.01).all(), (
        f"Some residual means not near zero: max={np.abs(means).max():.6f}"
    )


# ── Bootstrap tests ────────────────────────────────────────────────────────────

def test_bootstrap_returns_required_keys(three_factor_joint_test):
    """Bootstrap returns required keys."""
    # Use small bootstrap for speed
    boot_result = bootstrap_joint_alpha_test(
        three_factor_joint_test, n_bootstrap=100, seed=42
    )
    required_keys = ["empirical_f", "bootstrap_f_stats", "bootstrap_p", "n_valid"]
    for key in required_keys:
        assert key in boot_result, f"Missing key: {key}"


def test_bootstrap_p_value_in_01(three_factor_joint_test):
    """Bootstrap p-value should be in [0, 1]."""
    boot_result = bootstrap_joint_alpha_test(
        three_factor_joint_test, n_bootstrap=100, seed=42
    )
    p = boot_result["bootstrap_p"]
    assert 0.0 <= p <= 1.0, f"Bootstrap p-value={p:.4f} out of [0,1]"


def test_bootstrap_f_stats_positive(three_factor_joint_test):
    """Bootstrap F-statistics should be positive."""
    boot_result = bootstrap_joint_alpha_test(
        three_factor_joint_test, n_bootstrap=100, seed=42
    )
    f_stats = boot_result["bootstrap_f_stats"]
    valid = f_stats[~np.isnan(f_stats)]
    assert (valid > 0).all(), f"Some bootstrap F-stats non-positive"


def test_bootstrap_n_valid(three_factor_joint_test):
    """Most bootstrap resamples should be valid (no NaN)."""
    n_boot = 100
    boot_result = bootstrap_joint_alpha_test(
        three_factor_joint_test, n_bootstrap=n_boot, seed=42
    )
    assert boot_result["n_valid"] >= n_boot * 0.90, (
        f"Too many invalid bootstrap samples: {boot_result['n_valid']}/{n_boot}"
    )


def test_bootstrap_reproducible(three_factor_joint_test):
    """Bootstrap results are reproducible with same seed."""
    boot1 = bootstrap_joint_alpha_test(
        three_factor_joint_test, n_bootstrap=50, seed=99
    )
    boot2 = bootstrap_joint_alpha_test(
        three_factor_joint_test, n_bootstrap=50, seed=99
    )
    np.testing.assert_array_equal(
        boot1["bootstrap_f_stats"], boot2["bootstrap_f_stats"],
        err_msg="Bootstrap not reproducible with same seed"
    )


# ── All models joint test ──────────────────────────────────────────────────────

def test_all_joint_tests_run(exp1_data):
    """All model joint tests run without error."""
    factors, stock_ports, bond_ports = exp1_data
    results = run_all_joint_tests(
        stock_ports, bond_ports, factors,
        run_bootstrap=False,
    )
    assert len(results) > 0, "No joint test results"
    for model_name, res in results.items():
        assert "joint_test" in res, f"Missing joint_test for {model_name}"


def test_all_joint_tests_f_stats_positive(exp1_data):
    """All model F-statistics should be positive."""
    factors, stock_ports, bond_ports = exp1_data
    results = run_all_joint_tests(
        stock_ports, bond_ports, factors,
        run_bootstrap=False,
    )
    for model_name, res in results.items():
        f_stat = res["joint_test"]["f_stat"]
        assert f_stat > 0, f"Model {model_name}: F-stat={f_stat:.4f} not positive"


def test_summary_table_structure(exp1_data):
    """Summary table has correct structure."""
    factors, stock_ports, bond_ports = exp1_data
    results = run_all_joint_tests(
        stock_ports, bond_ports, factors,
        run_bootstrap=False,
    )
    summary = build_alpha_summary_table(results)
    assert "F-stat" in summary.columns
    assert "Theoretical p-value" in summary.columns
    assert len(summary) == len(results)


# ── Mathematical property tests ────────────────────────────────────────────────

def test_grs_fstat_invariant_to_scale():
    """GRS F-statistic should be invariant to scaling of returns."""
    rng = np.random.default_rng(42)
    N, L, K = 100, 5, 3
    alphas = rng.normal(0, 0.1, L)
    residuals = rng.normal(0, 1, (N, L))
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])

    f1, _, _ = compute_grs_fstat(alphas, residuals, X)
    # Scale returns by 100 (percent to basis points)
    f2, _, _ = compute_grs_fstat(alphas * 100, residuals * 100, X)
    # F-stat should be the same (scale-invariant)
    assert abs(f1 - f2) < 1e-6, f"F-stat not scale-invariant: {f1:.6f} vs {f2:.6f}"


def test_grs_fstat_increases_with_alpha_magnitude():
    """Larger alphas should produce larger F-statistics."""
    rng = np.random.default_rng(42)
    N, L, K = 200, 5, 3
    residuals = rng.normal(0, 1, (N, L))
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, K - 1))])

    f_small, _, _ = compute_grs_fstat(np.ones(L) * 0.1, residuals, X)
    f_large, _, _ = compute_grs_fstat(np.ones(L) * 1.0, residuals, X)
    assert f_large > f_small, (
        f"Larger alphas should give larger F: {f_large:.4f} vs {f_small:.4f}"
    )
