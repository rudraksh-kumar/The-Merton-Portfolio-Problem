"""Analytic benchmarks for the Merton problem  (used by every other part).

Model:  dB = rB dt,  dS = mu S dt + sigma S dW,  self-financing wealth
        dX = X[(r + pi (mu - r)) dt + pi sigma dW],     U(x) = x^(1-gamma)/(1-gamma).

Every function is a closed form derived in the report (Sections 2-3); the other modules only
use them as checks, never as inputs to the numerical solvers.
"""
import numpy as np


# ---------------------------------------------------------------- one risky asset
def merton_fraction(mu, r, sigma, gamma):
    """pi* = (mu - r) / (gamma sigma^2)."""
    return (mu - r) / (gamma * sigma ** 2)


def ce_growth(pi, mu, r, sigma, gamma):
    """Certainty-equivalent growth rate of a constant-fraction strategy:
    rho(pi) = r + pi (mu - r) - 0.5 gamma sigma^2 pi^2   (so E U(X_T) = U(x0 e^{rho T}))."""
    return r + pi * (mu - r) - 0.5 * gamma * sigma ** 2 * np.asarray(pi) ** 2


def kappa(mu, r, sigma, gamma):
    """kappa = rho(pi*) = r + (mu - r)^2 / (2 gamma sigma^2)."""
    return r + (mu - r) ** 2 / (2 * gamma * sigma ** 2)


def crra(x, gamma):
    """CRRA utility x^(1-gamma)/(1-gamma), or log x when gamma = 1."""
    x = np.asarray(x, dtype=float)
    return np.log(x) if gamma == 1 else x ** (1 - gamma) / (1 - gamma)


def expected_utility_const_pi(pi, x0, T, mu, r, sigma, gamma):
    """E[U(X_T)] for a constant fraction pi (log-normal moment formula)."""
    rho = ce_growth(pi, mu, r, sigma, gamma)
    if gamma == 1:
        return np.log(x0) + rho * T
    return x0 ** (1 - gamma) * np.exp((1 - gamma) * rho * T) / (1 - gamma)


def value_function(t, x, T, mu, r, sigma, gamma):
    """V(t,x) = exp{(1-gamma) kappa (T-t)} x^(1-gamma)/(1-gamma)."""
    return np.exp((1 - gamma) * kappa(mu, r, sigma, gamma) * (T - t)) * crra(x, gamma)


# ---------------------------------------------------------------- estimation risk
def loss_rate(pi, pi_star, sigma, gamma):
    """Loss in CE growth per year from holding pi instead of pi*:  0.5 gamma sigma^2 (pi - pi*)^2."""
    return 0.5 * gamma * sigma ** 2 * (np.asarray(pi) - pi_star) ** 2


def expected_loss_estimated_drift(gamma, T_est, n_assets=1):
    """E[loss] when pi is built from a drift estimated on T_est years: N / (2 gamma T_est).
    (Standard error of the drift is sigma/sqrt(T_est) whatever the sampling frequency.)"""
    return n_assets / (2.0 * gamma * T_est)


def sd_of_estimated_fraction(sigma, gamma, T_est):
    """sd(pi_hat) = 1 / (gamma sigma sqrt(T_est))."""
    return 1.0 / (gamma * sigma * np.sqrt(T_est))


# ---------------------------------------------------------------- extensions
def constrained_fraction(pi_star, lo=0.0, hi=1.0):
    """rho(pi) is a concave parabola, so the optimum on [lo,hi] is pi* clipped to the interval."""
    return np.clip(pi_star, lo, hi)


def floor_fraction(x, t, T, F, mu, r, sigma, gamma):
    """Utility (x-F)^(1-gamma)/(1-gamma): the investor keeps F e^{-r(T-t)} in bonds and applies
    Merton to the surplus s = x - F e^{-r(T-t)}.  Returns the fraction of *total* wealth in stock."""
    s = x - F * np.exp(-r * (T - t))
    return merton_fraction(mu, r, sigma, gamma) * s / x


def value_function_floor(t, x, T, F, mu, r, sigma, gamma):
    """Value function with a wealth floor F: invest the surplus x - F e^{-r(T-t)} in the unconstrained Merton problem."""
    s = x - F * np.exp(-r * (T - t))
    return np.exp((1 - gamma) * kappa(mu, r, sigma, gamma) * (T - t)) * crra(s, gamma)


def consumption_rate(beta, mu, r, sigma, gamma):
    """Infinite horizon, discount rate beta, utility of consumption c^(1-gamma)/(1-gamma):
    c* = nu x with nu = (beta - (1-gamma) kappa)/gamma  (requires nu > 0);  pi* unchanged."""
    return (beta - (1 - gamma) * kappa(mu, r, sigma, gamma)) / gamma


# ---------------------------------------------------------------- several assets
def multi_fraction(mu_vec, r, Sigma, gamma):
    """pi* = Sigma^{-1} (mu - r 1) / gamma."""
    return np.linalg.solve(Sigma, np.asarray(mu_vec) - r) / gamma


def multi_ce_growth(pi, mu_vec, r, Sigma, gamma):
    """CE growth rate r + pi'(mu - r) - gamma/2 pi' Sigma pi of a constant N-asset portfolio."""
    pi = np.asarray(pi)
    return r + pi @ (np.asarray(mu_vec) - r) - 0.5 * gamma * pi @ Sigma @ pi


def multi_kappa(mu_vec, r, Sigma, gamma):
    """Optimal N-asset CE growth r + (mu - r)' Sigma^{-1} (mu - r) / (2 gamma)."""
    ex = np.asarray(mu_vec) - r
    return r + 0.5 * ex @ np.linalg.solve(Sigma, ex) / gamma
