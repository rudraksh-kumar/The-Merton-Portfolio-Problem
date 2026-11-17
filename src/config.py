"""Global settings shared by all five parts of the Merton-problem project (MAC F244).

Everything that is an *assumption* (not an estimate from data) lives here so that it is
visible in one place and can be changed for sensitivity runs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for _p in (RESULTS, FIGS):
    _p.mkdir(exist_ok=True)

# ---- synthetic "textbook" parameters used in Parts 2, 3, 5 (same as the proposal) ----
MU, R, SIGMA, GAMMA, T = 0.12, 0.06, 0.20, 3.0, 10.0
X0 = 1.0

# ---- empirical-part assumptions ----
# Risk-free rate.  Part 1 uses the RBI 91-day T-bill auction yields in data/tbill_91d_auctions.csv (weekly,
# 1993-2026; mean 6.4% over Sep-2007..Apr-2026).  RF_CONST is the fallback if that file is missing, the r used
# where a single constant is needed (Part 4), and the centre of the constant-r sensitivity sweep.
RF_CONST = 0.065
RF_SENSITIVITY = (0.05, 0.06, 0.065, 0.07, 0.08)
GAMMA_GRID = (2.0, 3.0, 5.0)
TRADING_DAYS = 252
EST_WINDOWS = (3, 5, 7, 10)           # years of history used for estimation in the rolling backtest
PRIOR_PREMIUM, PRIOR_SD = 0.05, 0.04  # Bayesian-shrinkage prior on the equity premium (assumption)
SEED = 20261112
