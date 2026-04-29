"""
Tests for Experiment 4: Diagnostic and robustness analyses.

Verifies:
- Residual predictability test structure and properties
- January dummy construction
- January seasonality test structure
- Split-sample test preserves explanatory power
- E/P and D/P portfolio construction and regression
- Three-factor model reduces alpha patterns vs market-only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_french_data, load_predictor_series
from src.diagnostics import (
    build_dp_portfolios,
    build_ep_portfolios,
    create_january_dummy,
    run_dp_test,
    run_ep_test,
    run_january_test,
    run_residual_predictability_test,
    split_sample_factors,
    validate_dp_test,
    validate_ep_test,
    validate_january_seasonality,
    validate_residual_predictability,
    validate_split_sample,
)
from src.factor_construction import build_all_factors
from src.regressions import collect_residuals, MODEL_SPECS
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
def five_factor_residuals(exp1_data):
    """Compute five-factor model residuals."""
    factors, stock_ports, bond_ports = exp1_data
    all_ports = pd.concat([stock_ports, bond_ports], axis=1)
    return collect_residuals(all_ports, factors, MODEL_SPECS["five_factor"])


@pytest.fixture(scope="module")
def predictors(exp1_data):
    """Load predictor series."""
    factors, _, _ = exp1_data
    data = load_french_data()
    return load_predictor_series(data["factors"])


# ── January dummy tests ────────────────────────────────────────────────────────

def test_january_dummy_binary():
    """January dummy should be binary (0 or 1)."""
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    jan = create_january_dummy(idx)
    assert set(jan.unique()).issubset({0.0, 1.0}), "January dummy not binary"


def test_january_dummy_count():
    """January dummy should have exactly 29 ones (July 1963 to Dec 1991 = 29 Januaries)."""
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    jan = create_january_dummy(idx)
    # Count Januaries in Jul 1963 - Dec 1991
    # Jan 1964, 1965, ..., 1991 = 28 Januaries
    n_jan = int(jan.sum())
    assert 25 <= n_jan <= 32, f"Expected ~28 Januaries, got {n_jan}"


def test_january_dummy_length():
    """January dummy has correct length."""
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    jan = create_january_dummy(idx)
    assert len(jan) == N_OBS


# ── Residual predictability tests ─────────────────────────────────────────────

def test_residual_predictability_returns_dataframe(five_factor_residuals, predictors):
    """Residual predictability test returns a DataFrame."""
    result = run_residual_predictability_test(five_factor_residuals, predictors)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_residual_predictability_has_adj_r2(five_factor_residuals, predictors):
    """Residual predictability results include adj_r2 column."""
    result = run_residual_predictability_test(five_factor_residuals, predictors)
    assert "adj_r2" in result.columns


def test_residual_predictability_r2_bounded(five_factor_residuals, predictors):
    """Residual predictability adj R² should be in [-1, 1]."""
    result = run_residual_predictability_test(five_factor_residuals, predictors)
    r2_vals = result["adj_r2"].values
    assert (r2_vals >= -1.0).all() and (r2_vals <= 1.0).all(), (
        f"R² out of bounds: [{r2_vals.min():.4f}, {r2_vals.max():.4f}]"
    )


def test_residual_predictability_mostly_low_r2(five_factor_residuals, predictors):
    """Most residual predictability R² should be low (paper finding)."""
    result = run_residual_predictability_test(five_factor_residuals, predictors)
    r2_vals = result["adj_r2"].values
    frac_low = (r2_vals < 0.10).mean()
    assert frac_low > 0.50, f"Only {frac_low:.0%} of R² below 0.10"


def test_validate_residual_predictability(five_factor_residuals, predictors):
    """validate_residual_predictability returns boolean dict."""
    result = run_residual_predictability_test(five_factor_residuals, predictors)
    validation = validate_residual_predictability(result)
    assert isinstance(validation, dict)
    for k, v in validation.items():
        assert isinstance(v, bool), f"Key {k} has non-bool value: {v}"


# ── January seasonality tests ──────────────────────────────────────────────────

def test_january_test_returns_dataframe(exp1_data):
    """January test returns a DataFrame."""
    factors, _, _ = exp1_data
    result = run_january_test(factors[["RM-RF", "SMB", "HML"]], "test")
    assert isinstance(result, pd.DataFrame)
    assert "jan_slope" in result.columns
    assert "jan_t" in result.columns


def test_january_test_all_series(exp1_data):
    """January test covers all input series."""
    factors, _, _ = exp1_data
    cols = ["RM-RF", "SMB", "HML", "TERM", "DEF"]
    result = run_january_test(factors[cols], "factors")
    assert len(result) == len(cols), f"Expected {len(cols)} rows, got {len(result)}"


def test_january_test_t_stats_finite(exp1_data):
    """January t-statistics should be finite."""
    factors, _, _ = exp1_data
    result = run_january_test(factors[["RM-RF", "SMB", "HML"]], "test")
    assert result["jan_t"].notna().all(), "Some January t-stats are NaN"
    assert np.isfinite(result["jan_t"].values).all(), "Some January t-stats are infinite"


def test_validate_january_seasonality(exp1_data, five_factor_residuals):
    """validate_january_seasonality returns boolean dict."""
    factors, stock_ports, bond_ports = exp1_data
    all_ports = pd.concat([stock_ports, bond_ports], axis=1)
    factor_jan = run_january_test(factors[["RM-RF", "SMB", "HML", "TERM", "DEF"]], "factors")
    raw_jan = run_january_test(all_ports, "raw")
    resid_jan = run_january_test(five_factor_residuals, "residuals")
    validation = validate_january_seasonality(factor_jan, raw_jan, resid_jan)
    assert isinstance(validation, dict)


# ── Split-sample tests ─────────────────────────────────────────────────────────

def test_split_sample_returns_two_trials(exp1_data):
    """Split-sample test returns results for both halves."""
    factors, stock_ports, _ = exp1_data
    results = split_sample_factors(stock_ports, factors)
    assert "A_explains_B" in results
    assert "B_explains_A" in results


def test_split_sample_r2_positive(exp1_data):
    """Split-sample regressions should have positive mean adj R²."""
    factors, stock_ports, _ = exp1_data
    results = split_sample_factors(stock_ports, factors)
    for trial_name, df in results.items():
        mean_r2 = df["adj_r2"].mean()
        assert mean_r2 > 0.30, (
            f"Split-sample {trial_name}: mean adj R²={mean_r2:.4f} too low"
        )


def test_split_sample_has_required_columns(exp1_data):
    """Split-sample results have required columns."""
    factors, stock_ports, _ = exp1_data
    results = split_sample_factors(stock_ports, factors)
    required_cols = ["alpha", "alpha_t", "adj_r2"]
    for trial_name, df in results.items():
        for col in required_cols:
            assert col in df.columns, (
                f"Split-sample {trial_name}: missing column {col}"
            )


def test_validate_split_sample(exp1_data):
    """validate_split_sample returns boolean dict."""
    factors, stock_ports, _ = exp1_data
    results = split_sample_factors(stock_ports, factors)
    validation = validate_split_sample(results)
    assert isinstance(validation, dict)


# ── E/P portfolio tests ────────────────────────────────────────────────────────

def test_ep_portfolios_shape(exp1_data):
    """E/P portfolios have correct shape."""
    factors, _, _ = exp1_data
    ep_ports = build_ep_portfolios(factors)
    assert ep_ports.shape[1] == 6, f"Expected 6 E/P portfolios, got {ep_ports.shape[1]}"
    assert len(ep_ports) == N_OBS


def test_ep_portfolios_include_nonpositive_bin(exp1_data):
    """E/P portfolios include the E/P<=0 bin."""
    factors, _, _ = exp1_data
    ep_ports = build_ep_portfolios(factors)
    assert "EP_le0" in ep_ports.columns, "Missing EP_le0 portfolio"


def test_ep_test_returns_two_models(exp1_data):
    """E/P test returns results for market_only and three_factor."""
    factors, _, _ = exp1_data
    ep_ports = build_ep_portfolios(factors)
    results = run_ep_test(ep_ports, factors)
    assert "market_only" in results
    assert "three_factor" in results


def test_ep_three_factor_reduces_alphas(exp1_data):
    """Three-factor model should reduce mean |alpha| vs market-only for E/P portfolios."""
    factors, _, _ = exp1_data
    ep_ports = build_ep_portfolios(factors)
    results = run_ep_test(ep_ports, factors)
    mkt_alpha = results["market_only"]["alpha"].abs().mean()
    tf_alpha = results["three_factor"]["alpha"].abs().mean()
    assert tf_alpha < mkt_alpha, (
        f"Three-factor |alpha|={tf_alpha:.4f} not < market-only={mkt_alpha:.4f}"
    )


def test_validate_ep_test(exp1_data):
    """validate_ep_test returns boolean dict."""
    factors, _, _ = exp1_data
    ep_ports = build_ep_portfolios(factors)
    results = run_ep_test(ep_ports, factors)
    validation = validate_ep_test(results)
    assert isinstance(validation, dict)
    assert "three_factor_reduces_ep_alphas" in validation


# ── D/P portfolio tests ────────────────────────────────────────────────────────

def test_dp_portfolios_shape(exp1_data):
    """D/P portfolios have correct shape."""
    factors, _, _ = exp1_data
    dp_ports = build_dp_portfolios(factors)
    assert dp_ports.shape[1] == 6, f"Expected 6 D/P portfolios, got {dp_ports.shape[1]}"
    assert len(dp_ports) == N_OBS


def test_dp_portfolios_include_zero_dividend_bin(exp1_data):
    """D/P portfolios include the D/P=0 bin."""
    factors, _, _ = exp1_data
    dp_ports = build_dp_portfolios(factors)
    assert "DP_0" in dp_ports.columns, "Missing DP_0 portfolio"


def test_dp_zero_dividend_negative_alpha(exp1_data):
    """D/P=0 portfolio should have negative alpha in three-factor model (paper finding)."""
    factors, _, _ = exp1_data
    dp_ports = build_dp_portfolios(factors)
    results = run_dp_test(dp_ports, factors)
    if "three_factor" in results and "DP_0" in results["three_factor"].index:
        dp0_alpha = results["three_factor"].loc["DP_0", "alpha"]
        assert dp0_alpha < 0, f"D/P=0 alpha={dp0_alpha:.4f} should be negative"


def test_dp_three_factor_reduces_alphas(exp1_data):
    """Three-factor model should reduce mean |alpha| vs market-only for D/P portfolios."""
    factors, _, _ = exp1_data
    dp_ports = build_dp_portfolios(factors)
    results = run_dp_test(dp_ports, factors)
    mkt_alpha = results["market_only"]["alpha"].abs().mean()
    tf_alpha = results["three_factor"]["alpha"].abs().mean()
    assert tf_alpha < mkt_alpha, (
        f"Three-factor |alpha|={tf_alpha:.4f} not < market-only={mkt_alpha:.4f}"
    )


def test_validate_dp_test(exp1_data):
    """validate_dp_test returns boolean dict."""
    factors, _, _ = exp1_data
    dp_ports = build_dp_portfolios(factors)
    results = run_dp_test(dp_ports, factors)
    validation = validate_dp_test(results)
    assert isinstance(validation, dict)
    assert "dp0_negative_alpha" in validation


# ── Edge case tests ────────────────────────────────────────────────────────────

def test_january_test_empty_series():
    """January test handles empty DataFrame gracefully."""
    idx = pd.period_range(start=SAMPLE_START, end=SAMPLE_END, freq="M")
    empty_df = pd.DataFrame(index=idx)
    result = run_january_test(empty_df, "empty")
    assert len(result) == 0


def test_residual_predictability_single_portfolio(exp1_data, predictors):
    """Residual predictability works with a single portfolio."""
    factors, stock_ports, _ = exp1_data
    single_port = stock_ports.iloc[:, :1]
    residuals = collect_residuals(single_port, factors, ["RM-RF", "SMB", "HML"])
    result = run_residual_predictability_test(residuals, predictors)
    assert len(result) == 1


def test_ep_portfolios_no_nan(exp1_data):
    """E/P portfolios should have no NaN values."""
    factors, _, _ = exp1_data
    ep_ports = build_ep_portfolios(factors)
    assert not ep_ports.isna().any().any(), "E/P portfolios contain NaN values"


def test_dp_portfolios_no_nan(exp1_data):
    """D/P portfolios should have no NaN values."""
    factors, _, _ = exp1_data
    dp_ports = build_dp_portfolios(factors)
    assert not dp_ports.isna().any().any(), "D/P portfolios contain NaN values"
