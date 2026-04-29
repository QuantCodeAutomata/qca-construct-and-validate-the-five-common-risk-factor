# Repository: qca-construct-and-validate-the-five-common-risk-factor

## Overview
Replication of Fama & French (1993) "Common Risk Factors in the Returns on Stocks and Bonds".
Implements five common risk factors (RM-RF, SMB, HML, TERM, DEF) and orthogonalized market factor (RMO),
25 size-by-BE/ME stock portfolios, 7 bond portfolios, and all associated tests.

## Structure
- `src/` — all implementation modules
  - `data_loader.py` — data loading via massive API
  - `factor_construction.py` — Exp1: factor and portfolio construction
  - `regressions.py` — Exp2: OLS time-series regressions
  - `alpha_tests.py` — Exp3: GRS F-test and bootstrap
  - `diagnostics.py` — Exp4: residual predictability, January, split-sample, E/P, D/P
  - `utils.py` — shared utilities
- `tests/` — pytest test suite
- `results/` — output tables, plots, RESULTS.md
- `data/` — cached intermediate data

## Key Conventions
- All returns in **percent units** (monthly)
- Sample: July 1963 – December 1991 (342 observations)
- NYSE breakpoints only; applied to NYSE+Amex+NASDAQ universe
- 2-year COMPUSTAT seasoning rule
- Negative-BE firms excluded from BE/ME sorts but included in RM
- Value-weighted portfolios using lagged ME (prior month-end)
- statsmodels OLS (no HAC) — confirmed via Context7
- Bootstrap: 10,000 resamples, paired months, seed=42

## Data Sources (via massive API / simulated)
- CRSP monthly stock returns, prices, shares outstanding
- COMPUSTAT annual accounting data
- CRSP Treasury bill rates
- Ibbotson bond returns (LTG, corporate by rating)
- CRSP government bond portfolio returns (1-5yr, 6-10yr)

## Known Ambiguities
- Within-year value-weight updating: use prior month-end ME (standard FF convention)
- Split-sample partition: sort by PERMNO within sort universe, alternate A/B
- Dividend estimation: Fama-French (1988) procedure approximated
