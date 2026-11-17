# The Merton Portfolio Problem: Stochastic Asset Allocation & Empirical Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end quantitative finance project investigating **Merton's Portfolio Problem**—from rigorous continuous-time stochastic calculus and numerical partial differential equation (PDE) solvers, to Monte Carlo simulation and a 19-year empirical backtest on Indian financial markets (NSE Nifty 50 and RBI Treasury Bills).

---

## 📌 Project Overview in Plain English

Imagine you have a sum of money to invest over time. You have two choices:
1. A **risk-free asset** (like government Treasury bills) that pays a steady, guaranteed interest rate.
2. One or more **risky assets** (like stocks or index funds) that yield higher expected returns over the long run, but fluctuate randomly every day.

**The big question:** *What percentage of your wealth should you keep in stocks versus safe cash at any given moment to maximize your long-term satisfaction (expected utility)?*

In 1969, Nobel laureate Robert C. Merton solved this continuous-time optimal investment problem using stochastic calculus. 

This repository:
1. **Derives and verifies** Merton's theoretical solution across multiple mathematical frameworks (Hamilton-Jacobi-Bellman PDEs, Martingale/Girsanov change of measure, and symbolic proofs).
2. **Implements numerical solvers** (Finite Difference PDE methods and Dynamic Programming with quadrature) to solve the problem even when closed-form solutions do not exist (e.g., leverage limits, borrowing constraints, minimum wealth floors).
3. **Tests numerical convergence** using Monte Carlo schemes (Euler–Maruyama vs. Milstein) across hundreds of thousands of stochastic paths.
4. **Applies the theory to 19.2 years of real-world Indian market data** (2007–2026), discovering why textbook formulas fail in the real world when returns must be estimated, and how constraints and Bayesian shrinkage fix them.

---

## 🧠 The Core Mathematics (Simplified)

### 1. The Market Model
Stock prices **S<sub>t</sub>** follow Geometric Brownian Motion (GBM):

```
dS_t = μ · S_t dt + σ · S_t dW_t
```

Where:
* **μ** (mu) = expected rate of return (drift)
* **σ** (sigma) = market volatility (risk)
* **W<sub>t</sub>** = standard Brownian motion (random market shocks)

Portfolio wealth **X<sub>t</sub>** evolves over time as:

```
dX_t = X_t · [ r + π_t(μ - r) ] dt + π_t · σ · X_t dW_t
```

Where:
* **r** = risk-free rate (e.g., Treasury bills)
* **π<sub>t</sub>** (pi) = fraction of total wealth invested in the risky stock

---

### 2. The Investor's Goal
An investor with Constant Relative Risk Aversion (CRRA) utility:

```
U(x) = (x^(1 - γ)) / (1 - γ)    [for γ > 0, γ ≠ 1]
```

wants to choose the allocation **π<sub>t</sub>** over time to maximize their expected terminal utility: **E[ U(X_T) ]** at time horizon **T**.

---

### 3. Merton's Optimal Allocation
Using continuous-time stochastic control (Hamilton-Jacobi-Bellman equation), the exact analytical optimal allocation fraction is constant:

```
           μ - r
    π* = ─────────
          γ · σ²
```

Intuition:
* **Higher excess return (μ − r)** ➔ Invest more in stocks.
* **Higher risk aversion (γ)** ➔ Invest less in stocks, hold more cash.
* **Higher volatility (σ²)** ➔ Dramatically cut stock exposure (scales inversely with variance).

---

## 🔬 Project Modules & Implementation

The project is structured into 5 self-contained, fully documented modules in `src/`:

### Part 1: Empirical Data & Real-World Backtesting (`part1_data_backtest.py`)
* **Real Market Data:** 19.2 years (Sep 2007 – Nov 2026) of official daily Total Returns Index (TRI) data for Nifty 50, alongside weekly RBI 91-day Treasury Bill auction yields.
* **Rolling Estimation:** Estimates expected drift **μ̂** and volatility **σ̂** over rolling 3, 5, 7, and 10-year windows.
* **Allocation Strategies Evaluated:**
  * *Raw Merton:* Directly uses sample estimates (often swings wildly or demands extreme leverage).
  * *Capped Merton:* Restricts allocation to `[0, 1]` (no shorting, no leverage).
  * *Bayesian Shrinkage:* Blends noisy historical estimates with a sensible prior equity risk premium.
  * *Shrunk + Capped:* Combines Bayesian regularization with bound constraints.
* **Benchmarks:** Buy-and-Hold Nifty 50, standard 60/40 equity/bond split, and pure 91-day T-bills.

### Part 2: Monte Carlo Simulation & Convergence (`part2_montecarlo.py`)
* Compares **Euler–Maruyama** (`O(√Δt)` strong order) vs. **Milstein** (`O(Δt)` strong order) pathwise discretization against the exact analytical GBM solution.
* Demonstrates that for expected utility (weak convergence), the simpler Euler–Maruyama scheme performs at `O(Δt)`, making it computationally optimal for valuation.
* Simulates empirical estimation error to confirm Merton's theoretical loss rule:

```
E[ Certainty Equivalent Loss ] ≈ 1 / (2 · γ · T_e)
```
where **T<sub>e</sub>** is the length of the estimation window in years.

### Part 3: Analytical Verification & Martingale Methods (`part3_theory.py`, `theory.py`)
* **HJB Verification:** Proves that the candidate value function is a true martingale at **π*** and a supermartingale for any suboptimal **π**.
* **Girsanov's Theorem / Dual Martingale Method:** Changes to the risk-neutral measure **Q** and uses state-price deflators **ξ<sub>t</sub>** to confirm the duality bound: `E[ ξ_T · X_T ] ≤ x₀`.
* Symbolic zero-residual validation using `SymPy`.

### Part 4: Multi-Asset Extension (N Correlated Assets) (`part4_multiasset.py`)
* Extends allocation to multiple correlated assets (Nifty 50, Nifty Bank, Nifty IT, Nifty FMCG, Nifty Pharma):

```
π* = (1 / γ) · Σ⁻¹ · (μ - r · 1)
```

* Demonstrates the danger of matrix inversion with sample covariance matrices **Σ** (extreme gross leverage due to collinearity between sectors).
* Implements **Ledoit–Wolf Covariance Shrinkage**, SLSQP constrained optimization (long-only, no leverage), and CAPM-implied drift estimation.
* Validates DeMiguel et al.'s classic finding: Naive **1/N equal-weighting** often beats unconstrained optimization on real-world data due to parameter uncertainty.

### Part 5: Numerical PDE & Dynamic Programming Solvers (`part5_numerical_hjb.py`)
* **Finite Difference HJB Solver:** Transforms wealth coordinate to `y = ln(x)`, using an implicit time-stepping scheme with hybrid central/upwind spatial discretization and Howard's Policy Iteration.
* **Dynamic Programming (Discrete Bellman Recursion):** Discretizes time into rebalancing dates with Gauss–Hermite quadrature and cubic splines.
* Solves problems with **realistic non-linear constraints** where textbook formulas fail:
  * Hard borrowing/shorting constraints (`π ∈ [0, 1]`).
  * Guaranteed wealth floors (`X_T ≥ F`).

---

## 📊 Key Findings

| Finding | Explanation |
|---|---|
| **The "Estimation Risk" Curse** | To estimate expected return **μ** with standard error < 1%, an investor would need over 400 years of data! With typical 3–5 year windows, raw Merton rules fluctuate violently and destroy portfolio value. |
| **Why Naive / Constrained Rules Win** | Simple constraints (capping weights to `[0, 1]`) or naive equal weighting (`1/N`) drastically outperform unconstrained mathematical optimums out-of-sample by preventing extreme bets caused by estimation noise. |
| **Euler vs. Milstein in Finance** | While Milstein provides higher pathwise (strong) precision, the expected utility (weak error) converges at order 1 for both. Simple Euler-Maruyama is practically optimal for portfolio simulations. |
| **Finite Difference Efficiency** | Finite-difference HJB with policy iteration converges to errors < 10⁻⁶ in under 0.2 seconds, outperforming discrete dynamic programming in continuous-time settings. |

---

## 📁 Repository Structure

```text
The Merton Portfolio Problem/
│
├── data/                               # Historical financial datasets
│   ├── nse_tri_daily.csv               # Official NSE daily Total Return Index (2007-2026)
│   ├── tbill_91d_auctions.csv          # Weekly RBI 91-day T-bill auction cut-off yields
│   └── raw_nse/                        # Raw scraped JSON archives from niftyindices.com
│
├── figures/                            # Output plots generated by code
│   ├── fig_data.png                    # Empirical backtest performance & allocations
│   ├── fig_mc.png                      # Monte Carlo convergence rates & validation
│   ├── fig_hjb.png                     # Finite difference vs DP numerical convergence
│   ├── fig_multi.png                   # Multi-asset allocation & correlation sweeps
│   └── fig_sweep.png                   # Sensitivity analysis across parameters
│
├── report/                             # Academic paper & documentation
│   ├── report.tex                      # Full LaTeX manuscript
│   └── report.pdf                      # Compiled PDF report
│
├── results/                            # CSV & JSON outputs from all runs
│   ├── part1_backtest.csv              # Metric summary of all backtested strategies
│   ├── part2_summary.json              # Monte Carlo convergence statistics
│   ├── part4_backtest.csv              # Multi-asset empirical backtest results
│   └── part5_summary.json              # Numerical PDE runtime & accuracy results
│
├── src/                                # Core Python source code
│   ├── config.py                       # Global paths, seeds, and financial assumptions
│   ├── theory.py                       # Closed-form theoretical formulas & benchmarks
│   ├── prep_nse_tri.py                 # Data parser for NSE total returns data
│   ├── prep_rbi_tbill.py               # Data parser for RBI T-bill auction yields
│   ├── part1_data_backtest.py          # Part 1: Rolling backtest on Indian markets
│   ├── part2_montecarlo.py             # Part 2: SDE simulation & estimation error
│   ├── part3_theory.py                 # Part 3: Martingale & symbolic verifications
│   ├── part4_multiasset.py             # Part 4: Multi-asset allocation & covariance shrinkage
│   ├── part5_numerical_hjb.py          # Part 5: Numerical HJB & Dynamic Programming
│   └── run_all.py                      # Master script reproducing all results & figures
│
├── .gitignore                          # Clean repository filter (ignores pycache, build files)
├── requirements.txt                    # Python package dependencies
└── README.md                           # Documentation
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.10 or higher installed.

### Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "The Merton Portfolio Problem"
pip install -r requirements.txt
```

### Reproduce Everything
To execute the complete pipeline and regenerate every table, figure, and numerical benchmark:
```bash
cd src
python run_all.py
```
* Execution time is approximately **2 to 4 minutes** on a modern laptop.
* All random seeds are fixed (`config.py`), ensuring bit-for-bit reproducible numbers.

---

## 🛠️ Built With

* **[NumPy](https://numpy.org/)** & **[Pandas](https://pandas.pydata.org/)** – Vectorized financial time-series processing
* **[SciPy](https://scipy.org/)** – SLSQP optimization, numerical quadrature, and cubic spline interpolation
* **[SymPy](https://www.sympy.org/)** – Symbolic differentiation and analytical verification of the HJB PDE
* **[Scikit-learn](https://scikit-learn.org/)** – Ledoit–Wolf empirical covariance shrinkage
* **[Matplotlib](https://matplotlib.org/)** – Publication-quality visual figures

---

## 📜 Disclaimer
This project is for educational and quantitative research purposes only. The models and historical backtests presented herein do not constitute financial or investment advice.
