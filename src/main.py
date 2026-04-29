"""
Main experiment runner for the Fama-French five-factor replication.

Runs all four experiments and saves results to results/RESULTS.md.

Usage:
    python -m src.main
    python src/main.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / "results" / "experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

from src.data_loader import load_french_data, load_predictor_series
from src.factor_construction import run_experiment_1
from src.regressions import run_experiment_2
from src.alpha_tests import run_experiment_3
from src.diagnostics import run_experiment_4
from src.utils import RESULTS_DIR


def run_all_experiments(
    run_bootstrap: bool = True,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Run all four experiments and return results.

    Parameters
    ----------
    run_bootstrap : bool
        Whether to run the bootstrap (slow, ~10 min for 10,000 resamples).
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict
        All experiment results.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Fama-French Five-Factor Replication")
    logger.info("Fama and French (1993): Common Risk Factors in Stock and Bond Returns")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    data = load_french_data()
    predictors = load_predictor_series(data["factors"])

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    exp1_results = run_experiment_1(data)

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    exp2_results = run_experiment_2(exp1_results)

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    exp3_results = run_experiment_3(
        exp1_results, exp2_results,
        run_bootstrap=run_bootstrap,
        n_bootstrap=n_bootstrap,
    )

    # ── Experiment 4 ──────────────────────────────────────────────────────────
    exp4_results = run_experiment_4(exp1_results, exp2_results, predictors)

    elapsed = time.time() - t0
    logger.info("All experiments completed in %.1f seconds", elapsed)

    # ── Save RESULTS.md ───────────────────────────────────────────────────────
    save_results_markdown(exp1_results, exp2_results, exp3_results, exp4_results)

    return {
        "exp1": exp1_results,
        "exp2": exp2_results,
        "exp3": exp3_results,
        "exp4": exp4_results,
    }


def save_results_markdown(
    exp1: dict,
    exp2: dict,
    exp3: dict,
    exp4: dict,
) -> None:
    """
    Save all experiment results to results/RESULTS.md.

    Parameters
    ----------
    exp1, exp2, exp3, exp4 : dict
        Experiment result dictionaries.
    """
    lines = []
    lines.append("# Fama-French Five-Factor Replication — Results\n")
    lines.append("**Reference**: Fama, E.F. and French, K.R. (1993). "
                 "Common Risk Factors in the Returns on Stocks and Bonds. "
                 "*Journal of Financial Economics*, 33(1), 3–56.\n")
    lines.append(f"**Sample**: July 1963 – December 1991 (342 monthly observations)\n")
    lines.append("---\n")

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    lines.append("## Experiment 1: Factor Construction and Validation\n")

    lines.append("### Factor Summary Statistics\n")
    lines.append(exp1["factor_summary"].to_markdown(floatfmt=".4f"))
    lines.append("\n")

    lines.append("### Factor Correlation Matrix\n")
    lines.append(exp1["corr_matrix"].to_markdown(floatfmt=".3f"))
    lines.append("\n")

    lines.append("### RMO Regression (RM-RF on SMB, HML, TERM, DEF)\n")
    rmo_res = exp1["rmo_result"]
    rmo_table = pd.DataFrame({
        "Coefficient": rmo_res.params,
        "t-stat": rmo_res.tvalues,
        "p-value": rmo_res.pvalues,
    })
    lines.append(rmo_table.to_markdown(floatfmt=".4f"))
    lines.append(f"\n**R² = {rmo_res.rsquared:.4f}**\n")
    lines.append(f"*Paper targets: intercept≈0.50, SMB≈0.44, HML≈-0.63, TERM≈0.81, DEF≈0.79, R²≈0.38*\n")

    lines.append("### 25 Stock Portfolio Mean Excess Returns (%/month)\n")
    lines.append(exp1["stock_means"].to_markdown(floatfmt=".4f"))
    lines.append("\n")

    lines.append("### 7 Bond Portfolio Mean Excess Returns (%/month)\n")
    bond_df = exp1["bond_means"].to_frame("Mean (%/month)")
    lines.append(bond_df.to_markdown(floatfmt=".4f"))
    lines.append("\n")

    lines.append("### Validation Results\n")
    val = exp1["validation"]
    for k, v in val.items():
        status = "✅" if (v is True) else ("❌" if (v is False) else f"📊 {v:.4f}")
        lines.append(f"- `{k}`: {status}\n")
    lines.append("\n")

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    lines.append("## Experiment 2: Time-Series Regressions\n")

    all_results = exp2["all_results"]
    for model_name, model_results in all_results.items():
        from src.regressions import MODEL_LABELS
        label = MODEL_LABELS.get(model_name, model_name)
        lines.append(f"### {label}\n")

        for asset_class in ["stocks", "bonds"]:
            if asset_class not in model_results:
                continue
            df = model_results[asset_class]
            lines.append(f"**{asset_class.capitalize()}** — "
                         f"Mean adj R²: {df['adj_r2'].mean():.4f}, "
                         f"Mean |alpha|: {df['alpha'].abs().mean():.4f}%\n")

            # Show alpha and adj R² columns
            display_cols = ["alpha", "alpha_t", "adj_r2", "s_e"]
            display_cols = [c for c in display_cols if c in df.columns]
            lines.append(df[display_cols].to_markdown(floatfmt=".4f"))
            lines.append("\n")

    lines.append("### Validation Results\n")
    for k, v in exp2["validation"].items():
        if isinstance(v, bool):
            status = "✅" if v else "❌"
        else:
            status = f"📊 {v}"
        lines.append(f"- `{k}`: {status}\n")
    lines.append("\n")

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    lines.append("## Experiment 3: Joint Alpha Tests (GRS + Bootstrap)\n")

    lines.append("### Joint F-Statistics Summary\n")
    lines.append(exp3["summary_table"].to_markdown(floatfmt=".4f"))
    lines.append("\n")
    lines.append("*Paper target F-stats (approx): bond_factor≈2.09, market_only≈1.91, "
                 "style_factor≈1.78, three_factor≈1.56, five_factor≈1.66*\n\n")

    lines.append("### Validation Results\n")
    for k, v in exp3["validation"].items():
        if isinstance(v, bool):
            status = "✅" if v else "❌"
        else:
            status = f"📊 {v:.4f}"
        lines.append(f"- `{k}`: {status}\n")
    lines.append("\n")

    # ── Experiment 4 ──────────────────────────────────────────────────────────
    lines.append("## Experiment 4: Diagnostics and Robustness\n")

    lines.append("### A. Residual Predictability\n")
    pred = exp4["pred_results"]
    lines.append(f"- Adj R² range: [{pred['adj_r2'].min():.4f}, {pred['adj_r2'].max():.4f}]\n")
    lines.append(f"- Mean adj R²: {pred['adj_r2'].mean():.4f}\n")
    lines.append(f"- Fraction with R² < 0.05: {(pred['adj_r2'] < 0.05).mean():.2%}\n\n")

    lines.append("### B. January Seasonality\n")
    fj = exp4["factor_jan"]
    lines.append("**Factor January t-statistics:**\n")
    lines.append(fj[["jan_slope", "jan_t"]].to_markdown(floatfmt=".4f"))
    lines.append("\n")

    lines.append("### C. Split-Sample Test\n")
    for trial_name, df in exp4["split_results"].items():
        lines.append(f"**{trial_name}**: Mean adj R²={df['adj_r2'].mean():.4f}, "
                     f"Mean |alpha|={df['alpha'].abs().mean():.4f}%\n")
    lines.append("\n")

    lines.append("### D. E/P Portfolio Test\n")
    for model_name, df in exp4["ep_results"].items():
        lines.append(f"**{model_name}**: Mean |alpha|={df['alpha'].abs().mean():.4f}%\n")
        lines.append(df[["alpha", "alpha_t", "adj_r2"]].to_markdown(floatfmt=".4f"))
        lines.append("\n")

    lines.append("### E. D/P Portfolio Test\n")
    for model_name, df in exp4["dp_results"].items():
        lines.append(f"**{model_name}**: Mean |alpha|={df['alpha'].abs().mean():.4f}%\n")
        lines.append(df[["alpha", "alpha_t", "adj_r2"]].to_markdown(floatfmt=".4f"))
        lines.append("\n")

    lines.append("### Validation Results\n")
    for k, v in exp4["validation"].items():
        if isinstance(v, bool):
            status = "✅" if v else "❌"
        else:
            status = f"📊 {v:.4f}" if isinstance(v, float) else f"📊 {v}"
        lines.append(f"- `{k}`: {status}\n")
    lines.append("\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Overall Validation Summary\n")
    all_validations = {
        **exp1["validation"],
        **exp2["validation"],
        **exp3["validation"],
        **exp4["validation"],
    }
    bool_vals = {k: v for k, v in all_validations.items() if isinstance(v, bool)}
    n_pass = sum(bool_vals.values())
    n_total = len(bool_vals)
    lines.append(f"**Total checks passed: {n_pass}/{n_total}**\n\n")

    lines.append("### Generated Files\n")
    result_files = sorted(RESULTS_DIR.glob("*.csv")) + sorted(RESULTS_DIR.glob("*.png"))
    for f in result_files:
        lines.append(f"- `{f.name}`\n")

    # Write to file
    md_path = RESULTS_DIR / "RESULTS.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Results saved to %s", md_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Fama-French five-factor replication")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap (faster)")
    parser.add_argument("--n-bootstrap", type=int, default=10_000, help="Bootstrap resamples")
    args = parser.parse_args()

    run_all_experiments(
        run_bootstrap=not args.no_bootstrap,
        n_bootstrap=args.n_bootstrap,
    )
