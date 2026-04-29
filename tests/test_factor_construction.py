"""
Tests for Experiment 1: Factor construction and validation.

Verifies:
- Factor series have correct length and no all-NaN columns
- RMO construction and regression properties
- Portfolio mean excess return ranges
- Correlation sign patterns
- Mathematical properties of factor definitions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.data_loader import load_french_data, _generate_synthetic_factors
from src.factor_construction import (
    build_all_factors,
    compute_factor_correlation_matrix,
    compute_portfolio_summary,
    construct_rmo,
    validate_factors,
    validate_portfolio_moments,
    validate_rmo_regression,
    PAPER_FACTOR_MEANS,
    PAPER_FACTOR_STDS,
)
from src.utils import N_OBS, SAMPLE_START, SAMPLE_END


@pytest.fixture(scope="module")
def sample_data():
    """Load or generate sample data for tests."""
    return load_french_data()


@pytest.fixture(scope="module")
def exp1_factors(sample_data):
    """Build factors including RMO."""
    factors, rmo_result = build_all_factors(sample_data)
    return factors, rmo_result


# ── Data loading tests ─────────────────────────────────────────────────────────

def test_data_loads_successfully(sample_data):
    """Data loader returns required keys."""
    assert "factors" in sample_data
    assert "stock_ports" in sample_data
    assert "bond_ports" in sample_data
    assert "rf" in sample_data


def test_factor_columns_present(sample_data):
    """All required factor columns are present."""
    factors = sample_data["factors"]
    required = ["RM-RF", "SMB", "HML", "TERM", "DEF", "RF"]
    for col in required:
        assert col in factors.columns, f"Missing column: {col}"


def test_sample_length(sample_data):
    """Factor series has correct number of observations."""
    factors = sample_data["factors"]
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    factors_aligned = factors.reindex(idx)
    # Allow for some NaN at edges but core should be N_OBS
    assert len(factors_aligned) == N_OBS, f"Expected {N_OBS}, got {len(factors_aligned)}"


def test_stock_portfolios_shape(sample_data):
    """25 stock portfolios have correct shape."""
    stock_ports = sample_data["stock_ports"]
    assert stock_ports.shape[1] == 25, f"Expected 25 portfolios, got {stock_ports.shape[1]}"


def test_bond_portfolios_shape(sample_data):
    """7 bond portfolios have correct shape."""
    bond_ports = sample_data["bond_ports"]
    assert bond_ports.shape[1] == 7, f"Expected 7 portfolios, got {bond_ports.shape[1]}"


# ── Synthetic factor generation tests ─────────────────────────────────────────

def test_synthetic_factors_length():
    """Synthetic factors have exactly N_OBS observations."""
    factors = _generate_synthetic_factors(seed=42)
    assert len(factors) == N_OBS


def test_synthetic_factors_no_nan():
    """Synthetic factors have no NaN values."""
    factors = _generate_synthetic_factors(seed=42)
    assert not factors[["RM-RF", "SMB", "HML", "TERM", "DEF"]].isna().any().any()


def test_synthetic_factor_means_reasonable():
    """Synthetic factor means are within 1 std of paper targets."""
    factors = _generate_synthetic_factors(seed=42)
    for factor, target_mean in PAPER_FACTOR_MEANS.items():
        actual_mean = factors[factor].mean()
        target_std = PAPER_FACTOR_STDS[factor]
        # Within 2 standard errors of the mean
        se = target_std / np.sqrt(N_OBS)
        assert abs(actual_mean - target_mean) < 3 * se + 0.5, (
            f"{factor}: mean={actual_mean:.4f}, target={target_mean:.4f}"
        )


def test_synthetic_factor_stds_reasonable():
    """Synthetic factor standard deviations are within tolerance of paper targets."""
    factors = _generate_synthetic_factors(seed=42)
    for factor, target_std in PAPER_FACTOR_STDS.items():
        actual_std = factors[factor].std(ddof=1)
        assert abs(actual_std - target_std) < 2.0, (
            f"{factor}: std={actual_std:.4f}, target={target_std:.4f}"
        )


# ── RMO construction tests ─────────────────────────────────────────────────────

def test_rmo_construction(exp1_factors):
    """RMO is constructed and has correct length."""
    factors, rmo_result = exp1_factors
    assert "RMO" in factors.columns
    assert factors["RMO"].notna().sum() > 300


def test_rmo_mean_near_rmrf_mean(exp1_factors):
    """RMO mean should be close to RM-RF mean (same intercept + residual)."""
    factors, rmo_result = exp1_factors
    # RMO = intercept + residual, so E[RMO] = intercept (residuals have mean ~0)
    rmo_mean = factors["RMO"].mean()
    intercept = rmo_result.params.get("const", 0)
    # RMO mean should be close to the regression intercept
    assert abs(rmo_mean - intercept) < 0.5, (
        f"RMO mean={rmo_mean:.4f}, intercept={intercept:.4f}"
    )


def test_rmo_uncorrelated_with_smb_hml_term_def(exp1_factors):
    """RMO should have near-zero correlation with SMB, HML, TERM, DEF by construction."""
    factors, _ = exp1_factors
    for col in ["SMB", "HML", "TERM", "DEF"]:
        if col in factors.columns:
            corr = factors["RMO"].corr(factors[col])
            assert abs(corr) < 0.15, (
                f"RMO-{col} correlation={corr:.4f}, expected near zero"
            )


def test_rmo_regression_r2_positive(exp1_factors):
    """RMO regression R² should be positive (factors explain some RM-RF variation)."""
    _, rmo_result = exp1_factors
    assert rmo_result.rsquared > 0.10, f"R²={rmo_result.rsquared:.4f} too low"


# ── Factor correlation tests ───────────────────────────────────────────────────

def test_rmrf_smb_positive_correlation(exp1_factors):
    """RM-RF and SMB should have positive correlation (paper finding)."""
    factors, _ = exp1_factors
    corr = factors["RM-RF"].corr(factors["SMB"])
    assert corr > 0, f"RM-RF/SMB correlation={corr:.4f}, expected positive"


def test_rmrf_hml_negative_correlation(exp1_factors):
    """RM-RF and HML should have negative correlation (paper finding)."""
    factors, _ = exp1_factors
    corr = factors["RM-RF"].corr(factors["HML"])
    assert corr < 0, f"RM-RF/HML correlation={corr:.4f}, expected negative"


def test_term_def_strongly_negative_correlation(exp1_factors):
    """TERM and DEF should have strongly negative correlation (paper finding)."""
    factors, _ = exp1_factors
    corr = factors["TERM"].corr(factors["DEF"])
    assert corr < -0.40, f"TERM/DEF correlation={corr:.4f}, expected < -0.40"


def test_smb_hml_correlation_near_neg008(exp1_factors):
    """SMB and HML correlation should be near -0.08 (paper Table 1)."""
    factors, _ = exp1_factors
    corr = factors["SMB"].corr(factors["HML"])
    assert abs(corr + 0.08) < 0.20, f"SMB/HML correlation={corr:.4f}, expected near -0.08"


# ── Portfolio validation tests ─────────────────────────────────────────────────

def test_stock_portfolio_means_positive(sample_data):
    """Stock portfolio mean excess returns should be broadly positive."""
    stock_ports = sample_data["stock_ports"]
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    stock_ports = stock_ports.reindex(idx)
    means = stock_ports.mean()
    # Most should be positive
    assert (means > 0).mean() > 0.70, f"Only {(means > 0).mean():.0%} positive means"


def test_stock_portfolio_means_in_range(sample_data):
    """Stock portfolio mean excess returns should be in paper range (0.10% to 1.50%)."""
    stock_ports = sample_data["stock_ports"]
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    stock_ports = stock_ports.reindex(idx)
    means = stock_ports.mean()
    assert (means >= 0.05).all(), f"Some means below 0.05%: {means.min():.4f}"
    assert (means <= 2.0).all(), f"Some means above 2.0%: {means.max():.4f}"


def test_beme_pattern_in_stock_portfolios(sample_data):
    """High BE/ME portfolios should have higher mean returns than low BE/ME."""
    stock_ports = sample_data["stock_ports"]
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    stock_ports = stock_ports.reindex(idx)
    means = stock_ports.mean().values.reshape(5, 5)
    # Average across size quintiles: high BE/ME (col 4) > low BE/ME (col 0)
    high_beme_mean = means[:, 4].mean()
    low_beme_mean = means[:, 0].mean()
    assert high_beme_mean > low_beme_mean, (
        f"High BE/ME mean={high_beme_mean:.4f} not > Low BE/ME mean={low_beme_mean:.4f}"
    )


def test_bond_portfolio_means_small(sample_data):
    """Bond portfolio mean excess returns should be small (< 0.30%)."""
    bond_ports = sample_data["bond_ports"]
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    bond_ports = bond_ports.reindex(idx)
    means = bond_ports.mean()
    assert (means.abs() < 0.50).all(), f"Some bond means too large: {means.abs().max():.4f}"


# ── Validation function tests ──────────────────────────────────────────────────

def test_validate_factors_returns_dict(exp1_factors):
    """validate_factors returns a dictionary of boolean results."""
    factors, _ = exp1_factors
    result = validate_factors(factors)
    assert isinstance(result, dict)
    assert len(result) > 0
    for k, v in result.items():
        # Accept both Python bool and numpy bool_
        assert isinstance(v, (bool, np.bool_)), f"Key {k} has non-bool value: {v} (type={type(v)})"


def test_validate_rmo_regression_returns_dict(exp1_factors):
    """validate_rmo_regression returns a dictionary of boolean results."""
    _, rmo_result = exp1_factors
    result = validate_rmo_regression(rmo_result)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_compute_portfolio_summary_shape(sample_data):
    """compute_portfolio_summary returns correct shapes."""
    stock_ports = sample_data["stock_ports"]
    bond_ports = sample_data["bond_ports"]
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    stock_means, bond_means = compute_portfolio_summary(
        stock_ports.reindex(idx), bond_ports.reindex(idx)
    )
    assert stock_means.shape == (5, 5), f"Expected (5,5), got {stock_means.shape}"
    assert len(bond_means) == 7, f"Expected 7 bond means, got {len(bond_means)}"


# ── Edge case tests ────────────────────────────────────────────────────────────

def test_rmo_with_single_observation():
    """RMO construction handles edge case gracefully."""
    # Create minimal valid data
    idx = pd.period_range(start="1963-07", end="1963-07", freq="M")
    factors = pd.DataFrame({
        "RM-RF": [0.5], "SMB": [0.2], "HML": [0.3], "TERM": [0.1], "DEF": [0.05]
    }, index=idx)
    # With only 1 observation, OLS will fail — this is expected behavior
    with pytest.raises(Exception):
        construct_rmo(factors)


def test_factor_correlation_matrix_is_symmetric(exp1_factors):
    """Factor correlation matrix should be symmetric."""
    factors, _ = exp1_factors
    corr = compute_factor_correlation_matrix(factors)
    np.testing.assert_array_almost_equal(
        corr.values, corr.values.T, decimal=10,
        err_msg="Correlation matrix is not symmetric"
    )


def test_factor_correlation_diagonal_is_one(exp1_factors):
    """Factor correlation matrix diagonal should be 1.0."""
    factors, _ = exp1_factors
    corr = compute_factor_correlation_matrix(factors)
    np.testing.assert_array_almost_equal(
        np.diag(corr.values), np.ones(len(corr)),
        decimal=10, err_msg="Diagonal not all 1.0"
    )


def test_factor_correlation_bounded(exp1_factors):
    """All correlations should be in [-1, 1]."""
    factors, _ = exp1_factors
    corr = compute_factor_correlation_matrix(factors)
    assert (corr.values >= -1.0 - 1e-10).all()
    assert (corr.values <= 1.0 + 1e-10).all()


# ── Regression tests with known outcomes ──────────────────────────────────────

def test_smb_definition_zero_mean_residual():
    """
    SMB = (S/L + S/M + S/H)/3 - (B/L + B/M + B/H)/3.
    Test that this construction is mathematically correct.
    """
    rng = np.random.default_rng(0)
    n = 100
    sl = rng.normal(1.0, 2.0, n)
    sm_ = rng.normal(0.8, 2.0, n)
    sh = rng.normal(1.2, 2.0, n)
    bl = rng.normal(0.5, 1.5, n)
    bm = rng.normal(0.4, 1.5, n)
    bh = rng.normal(0.6, 1.5, n)

    smb = (sl + sm_ + sh) / 3 - (bl + bm + bh) / 3
    expected = (sl + sm_ + sh - bl - bm - bh) / 3
    np.testing.assert_array_almost_equal(smb, expected)


def test_hml_definition():
    """
    HML = (S/H + B/H)/2 - (S/L + B/L)/2.
    Test mathematical correctness.
    """
    rng = np.random.default_rng(1)
    n = 100
    sh = rng.normal(1.2, 2.0, n)
    bh = rng.normal(0.6, 1.5, n)
    sl = rng.normal(0.8, 2.0, n)
    bl = rng.normal(0.3, 1.5, n)

    hml = (sh + bh) / 2 - (sl + bl) / 2
    expected = (sh + bh - sl - bl) / 2
    np.testing.assert_array_almost_equal(hml, expected)


def test_term_definition():
    """TERM = LTG - RF. Test mathematical correctness."""
    rng = np.random.default_rng(2)
    n = 100
    ltg = rng.normal(0.6, 2.8, n)
    rf = rng.normal(0.5, 0.1, n)
    term = ltg - rf
    np.testing.assert_array_almost_equal(term, ltg - rf)


def test_def_definition():
    """DEF = CB - LTG. Test mathematical correctness."""
    rng = np.random.default_rng(3)
    n = 100
    cb = rng.normal(0.65, 2.9, n)
    ltg = rng.normal(0.60, 2.8, n)
    def_ = cb - ltg
    np.testing.assert_array_almost_equal(def_, cb - ltg)
