"""
Tests for Experiment 2: Time-series OLS regressions.

Verifies:
- All six model specifications run without error
- Regression outputs have correct structure
- Three-factor model shows expected loading patterns
- Bond-factor model shows high R² for bond portfolios
- Adjusted R² values are in valid range [-1, 1]
- Residuals are collected correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.data_loader import load_french_data
from src.factor_construction import build_all_factors
from src.regressions import (
    MODEL_SPECS,
    collect_residuals,
    run_all_models,
    run_model_regressions,
    run_single_regression,
    validate_bond_factor_patterns,
    validate_three_factor_patterns,
)
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
def all_regression_results(exp1_data):
    """Run all model regressions."""
    factors, stock_ports, bond_ports = exp1_data
    return run_all_models(stock_ports, bond_ports, factors)


# ── Single regression tests ────────────────────────────────────────────────────

def test_single_regression_returns_result(exp1_data):
    """run_single_regression returns a valid OLS result."""
    factors, stock_ports, _ = exp1_data
    y = stock_ports.iloc[:, 0]
    result = run_single_regression(y, factors, ["RM-RF", "SMB", "HML"])
    assert result is not None
    assert hasattr(result, "params")
    assert hasattr(result, "rsquared_adj")
    assert "const" in result.params.index


def test_single_regression_nobs(exp1_data):
    """Single regression uses approximately N_OBS observations."""
    factors, stock_ports, _ = exp1_data
    y = stock_ports.iloc[:, 0]
    result = run_single_regression(y, factors, ["RM-RF"])
    assert result.nobs >= N_OBS - 5  # allow for small NaN trimming


def test_single_regression_r2_bounded(exp1_data):
    """Adjusted R² should be in [-1, 1]."""
    factors, stock_ports, _ = exp1_data
    for i in range(5):  # test first 5 portfolios
        y = stock_ports.iloc[:, i]
        result = run_single_regression(y, factors, ["RM-RF", "SMB", "HML"])
        assert -1.0 <= result.rsquared_adj <= 1.0, (
            f"Portfolio {i}: adj R²={result.rsquared_adj:.4f} out of bounds"
        )


# ── Model specification tests ──────────────────────────────────────────────────

def test_all_models_run(all_regression_results):
    """All six model specifications produce results."""
    expected_models = list(MODEL_SPECS.keys())
    for model_name in expected_models:
        assert model_name in all_regression_results, f"Missing model: {model_name}"


def test_model_results_have_required_columns(all_regression_results):
    """Each model result has required columns."""
    required_cols = ["alpha", "alpha_t", "adj_r2", "s_e"]
    for model_name, model_results in all_regression_results.items():
        for asset_class in ["stocks", "bonds"]:
            df = model_results[asset_class]
            for col in required_cols:
                assert col in df.columns, (
                    f"Model {model_name}, {asset_class}: missing column {col}"
                )


def test_stock_portfolio_count(all_regression_results):
    """Each model has results for all 25 stock portfolios."""
    for model_name, model_results in all_regression_results.items():
        n_stocks = len(model_results["stocks"])
        assert n_stocks == 25, (
            f"Model {model_name}: expected 25 stock portfolios, got {n_stocks}"
        )


def test_bond_portfolio_count(all_regression_results):
    """Each model has results for all 7 bond portfolios."""
    for model_name, model_results in all_regression_results.items():
        n_bonds = len(model_results["bonds"])
        assert n_bonds == 7, (
            f"Model {model_name}: expected 7 bond portfolios, got {n_bonds}"
        )


def test_adj_r2_in_valid_range(all_regression_results):
    """All adjusted R² values are in [-1, 1]."""
    for model_name, model_results in all_regression_results.items():
        for asset_class in ["stocks", "bonds"]:
            r2_vals = model_results[asset_class]["adj_r2"].values
            assert (r2_vals >= -1.0).all() and (r2_vals <= 1.0).all(), (
                f"Model {model_name}, {asset_class}: R² out of bounds"
            )


# ── Three-factor model specific tests ─────────────────────────────────────────

def test_three_factor_stock_r2_high(all_regression_results):
    """Three-factor model should have high R² for most stock portfolios."""
    if "three_factor" not in all_regression_results:
        pytest.skip("three_factor model not available")
    r2_vals = all_regression_results["three_factor"]["stocks"]["adj_r2"].values
    n_high = (r2_vals > 0.70).sum()
    assert n_high >= 15, f"Only {n_high}/25 portfolios have adj R² > 0.70"


def test_three_factor_smb_loading_pattern(all_regression_results):
    """SMB loadings should decrease from small to big portfolios."""
    if "three_factor" not in all_regression_results:
        pytest.skip("three_factor model not available")
    stock_res = all_regression_results["three_factor"]["stocks"]
    if "b_SMB" not in stock_res.columns:
        pytest.skip("SMB loadings not available")
    smb_loads = stock_res["b_SMB"].values
    # Small portfolios (first 5) should have higher SMB loading than big (last 5)
    smb_small = smb_loads[:5].mean()
    smb_big = smb_loads[-5:].mean()
    assert smb_small > smb_big, (
        f"SMB small mean={smb_small:.4f} not > big mean={smb_big:.4f}"
    )


def test_three_factor_hml_loading_pattern(all_regression_results):
    """HML loadings should increase from low to high BE/ME portfolios."""
    if "three_factor" not in all_regression_results:
        pytest.skip("three_factor model not available")
    stock_res = all_regression_results["three_factor"]["stocks"]
    if "b_HML" not in stock_res.columns:
        pytest.skip("HML loadings not available")
    hml_loads = stock_res["b_HML"].values
    # Low BE/ME (columns 0,5,10,15,20) vs high BE/ME (columns 4,9,14,19,24)
    hml_low = hml_loads[[0, 5, 10, 15, 20]].mean()
    hml_high = hml_loads[[4, 9, 14, 19, 24]].mean()
    assert hml_high > hml_low, (
        f"HML high mean={hml_high:.4f} not > low mean={hml_low:.4f}"
    )


def test_three_factor_market_betas_near_one(all_regression_results):
    """Market betas in three-factor model should be near 1 for stock portfolios."""
    if "three_factor" not in all_regression_results:
        pytest.skip("three_factor model not available")
    stock_res = all_regression_results["three_factor"]["stocks"]
    if "b_RM-RF" not in stock_res.columns:
        pytest.skip("RM-RF loadings not available")
    betas = stock_res["b_RM-RF"].values
    assert (betas > 0.5).all(), f"Some market betas below 0.5: {betas.min():.4f}"
    assert (betas < 1.8).all(), f"Some market betas above 1.8: {betas.max():.4f}"


# ── Bond-factor model specific tests ──────────────────────────────────────────

def test_bond_factor_model_high_r2_for_bonds(all_regression_results):
    """Bond-factor model should have high R² for bond portfolios."""
    if "bond_factor" not in all_regression_results:
        pytest.skip("bond_factor model not available")
    r2_vals = all_regression_results["bond_factor"]["bonds"]["adj_r2"].values
    assert (r2_vals > 0.20).all(), f"Some bond R² below 0.20: {r2_vals.min():.4f}"


def test_bond_factor_term_loadings_positive(all_regression_results):
    """TERM loadings should be positive for bond portfolios."""
    if "bond_factor" not in all_regression_results:
        pytest.skip("bond_factor model not available")
    bond_res = all_regression_results["bond_factor"]["bonds"]
    if "b_TERM" not in bond_res.columns:
        pytest.skip("TERM loadings not available")
    term_loads = bond_res["b_TERM"].values
    assert (term_loads > 0).all(), f"Some TERM loadings negative: {term_loads.min():.4f}"


# ── Five-factor model tests ────────────────────────────────────────────────────

def test_five_factor_stock_alphas_small(all_regression_results):
    """Five-factor model should have small alphas for stock portfolios."""
    if "five_factor" not in all_regression_results:
        pytest.skip("five_factor model not available")
    alphas = all_regression_results["five_factor"]["stocks"]["alpha"].values
    mean_abs_alpha = np.abs(alphas).mean()
    assert mean_abs_alpha < 0.50, f"Mean |alpha| too large: {mean_abs_alpha:.4f}"


def test_five_factor_vs_three_factor_stock_r2(all_regression_results):
    """Five-factor model R² should be at least as high as three-factor for stocks."""
    if "five_factor" not in all_regression_results or "three_factor" not in all_regression_results:
        pytest.skip("Models not available")
    r2_5f = all_regression_results["five_factor"]["stocks"]["adj_r2"].mean()
    r2_3f = all_regression_results["three_factor"]["stocks"]["adj_r2"].mean()
    # Five-factor should not be much worse than three-factor for stocks
    assert r2_5f >= r2_3f - 0.05, (
        f"Five-factor mean R²={r2_5f:.4f} much lower than three-factor={r2_3f:.4f}"
    )


# ── Residual collection tests ──────────────────────────────────────────────────

def test_collect_residuals_shape(exp1_data):
    """Collected residuals have correct shape."""
    factors, stock_ports, bond_ports = exp1_data
    all_ports = pd.concat([stock_ports, bond_ports], axis=1)
    residuals = collect_residuals(all_ports, factors, ["RM-RF", "SMB", "HML"])
    assert residuals.shape[1] == 32, f"Expected 32 residual series, got {residuals.shape[1]}"


def test_collect_residuals_near_zero_mean(exp1_data):
    """OLS residuals should have mean near zero."""
    factors, stock_ports, _ = exp1_data
    residuals = collect_residuals(stock_ports, factors, ["RM-RF", "SMB", "HML"])
    means = residuals.mean()
    assert (means.abs() < 0.01).all(), f"Some residual means not near zero: {means.abs().max():.6f}"


# ── Validation function tests ──────────────────────────────────────────────────

def test_validate_three_factor_patterns(all_regression_results):
    """validate_three_factor_patterns returns expected keys."""
    if "three_factor" not in all_regression_results:
        pytest.skip("three_factor model not available")
    result = validate_three_factor_patterns(all_regression_results["three_factor"]["stocks"])
    assert "r2_mostly_high" in result
    assert "smb_small_gt_big" in result
    assert "hml_high_gt_low" in result


def test_validate_bond_factor_patterns(all_regression_results):
    """validate_bond_factor_patterns returns expected keys."""
    if "bond_factor" not in all_regression_results:
        pytest.skip("bond_factor model not available")
    result = validate_bond_factor_patterns(all_regression_results["bond_factor"]["bonds"])
    assert "bond_r2_range_ok" in result
    assert "bond_r2_min" in result
    assert "bond_r2_max" in result


# ── Orthogonalized model tests ─────────────────────────────────────────────────

def test_ortho_five_factor_uses_rmo(all_regression_results):
    """Orthogonalized five-factor model uses RMO instead of RM-RF."""
    if "ortho_five_factor" not in all_regression_results:
        pytest.skip("ortho_five_factor model not available")
    stock_res = all_regression_results["ortho_five_factor"]["stocks"]
    assert "b_RMO" in stock_res.columns, "RMO loading column missing"
    assert "b_RM-RF" not in stock_res.columns, "RM-RF should not be in ortho model"


def test_ortho_five_factor_rmo_loadings_near_one(all_regression_results):
    """RMO loadings in orthogonalized model should be near 1 for stock portfolios."""
    if "ortho_five_factor" not in all_regression_results:
        pytest.skip("ortho_five_factor model not available")
    stock_res = all_regression_results["ortho_five_factor"]["stocks"]
    if "b_RMO" not in stock_res.columns:
        pytest.skip("RMO loadings not available")
    rmo_loads = stock_res["b_RMO"].values
    assert (rmo_loads > 0.5).all(), f"Some RMO loadings below 0.5: {rmo_loads.min():.4f}"
