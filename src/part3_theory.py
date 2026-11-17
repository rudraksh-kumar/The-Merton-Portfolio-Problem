"""Part 3: stochastic control theory.

(a) Symbolic proof (sympy) that V(t,x) = exp{(1-g)k(T-t)} x^(1-g)/(1-g) solves the HJB equation
    V_t + sup_pi {(r+pi(mu-r)) x V_x + 0.5 pi^2 sigma^2 x^2 V_xx} = 0,  V(T,x)=U(x),
    that the supremum is attained at pi* = (mu-r)/(g sigma^2) and that it is a *maximum*.
(b) Symbolic derivation of the consumption extension  c* = nu x.
(c) Martingale method: state-price density xi_t = exp{-(r+lambda^2/2)t - lambda W_t},
    budget constraint E[xi_T X_T] = x0, Girsanov change of measure, and the pathwise identity
    X*_T (wealth under pi*) = (y xi_T)^(-1/gamma).  All checked by Monte Carlo.
(d) Verification-theorem sanity check: no constant fraction beats pi*; clip rule for pi in [0,1].
Outputs: results/part3_theory.json
"""
import json
import numpy as np
import sympy as sp
from config import *            # noqa
import theory as th


def symbolic_checks():
    """SymPy: the ansatz V solves the HJB equation, the maximiser is pi*, and the consumption extension gives c* = nu x."""
    t, x, T_, mu, r, sg, g = sp.symbols("t x T mu r sigma gamma", positive=True)
    pi = sp.symbols("pi", real=True)
    kap = r + (mu - r) ** 2 / (2 * g * sg ** 2)
    V = sp.exp((1 - g) * kap * (T_ - t)) * x ** (1 - g) / (1 - g)
    Vt, Vx, Vxx = sp.diff(V, t), sp.diff(V, x), sp.diff(V, x, 2)
    H = Vt + (r + pi * (mu - r)) * x * Vx + sp.Rational(1, 2) * pi ** 2 * sg ** 2 * x ** 2 * Vxx
    foc = sp.solve(sp.diff(H, pi), pi)[0]
    out = {
        "pi_star_matches": sp.simplify(foc - (mu - r) / (g * sg ** 2)) == 0,
        "HJB_residual_at_pi_star": str(sp.simplify(H.subs(pi, foc))),
        "terminal_condition_ok": sp.simplify(V.subs(t, T_) - x ** (1 - g) / (1 - g)) == 0,
        # d^2H/dpi^2 = sigma^2 x^2 V_xx < 0  (concave in pi, so the FOC gives a maximum)
        "second_derivative": str(sp.simplify(sp.diff(H, pi, 2))),
    }
    # consumption extension (infinite horizon, discount beta): V = nu^(-g) x^(1-g)/(1-g)
    beta, nu, lam = sp.symbols("beta nu lambda", positive=True)
    Vc = nu ** (-g) * x ** (1 - g) / (1 - g)
    lam_ = (mu - r) / sg
    # sup over pi of pi(mu-r) x Vx + 0.5 pi^2 sigma^2 x^2 Vxx  = -(mu-r)^2 Vx^2 / (2 sigma^2 Vxx)
    Vcx, Vcxx = sp.diff(Vc, x), sp.diff(Vc, x, 2)
    sup_pi = -(mu - r) ** 2 * Vcx ** 2 / (2 * sg ** 2 * Vcxx)
    hjb_c = -beta * Vc + (nu * x) ** (1 - g) / (1 - g) + (r * x - nu * x) * Vcx + sup_pi
    sol = sp.solve(sp.simplify(hjb_c / x ** (1 - g) * nu ** g), nu)
    out["consumption_nu_solution"] = [str(sp.simplify(s)) for s in sol]
    nu_formula = (beta - (1 - g) * kap) / g
    out["consumption_nu_matches"] = any(sp.simplify(s - nu_formula) == 0 for s in sol)
    return out


def martingale_checks(n=2_000_000, seed=SEED):
    """Monte Carlo checks of the martingale method and Girsanov: E[xi_T X_T] = x0 for every pi, X*_T = (y xi_T)^(-1/gamma) equals Merton wealth, E^Q[e^{-rT} X_T] = x0."""
    rng = np.random.default_rng(seed)
    lam = (MU - R) / SIGMA
    W = np.sqrt(T) * rng.standard_normal(n)
    xi = np.exp(-(R + lam ** 2 / 2) * T - lam * W)
    pi_s = th.merton_fraction(MU, R, SIGMA, GAMMA)

    def wealth(pi):  # exact GBM wealth under the *physical* measure
        """Exact GBM wealth under the physical measure for a constant fraction pi."""
        return X0 * np.exp((R + pi * (MU - R) - 0.5 * pi ** 2 * SIGMA ** 2) * T + pi * SIGMA * W)

    out = {"lambda": lam, "n_paths": n}
    out["E_xi_vs_exp(-rT)"] = (xi.mean(), np.exp(-R * T), xi.std() / np.sqrt(n))
    # budget constraint holds for *every* admissible strategy (xi*X is a martingale)
    for pi in (0.0, 0.3, pi_s, 1.0, 1.5):
        v = xi * wealth(pi)
        out[f"E[xi_T X_T] pi={pi:.2f}"] = (v.mean(), v.std() / np.sqrt(n))
    # optimiser from the Lagrangian: X*_T = (y xi_T)^(-1/gamma), y from E[xi X*] = x0
    c = X0 / np.mean(xi ** (1 - 1 / GAMMA))
    # exact y: E[xi^(1-1/g)] = exp{-(1-1/g)(r+lam^2/2)T + 0.5 (1-1/g)^2 lam^2 T}
    a = 1 - 1 / GAMMA
    Exi = np.exp(-a * (R + lam ** 2 / 2) * T + 0.5 * a ** 2 * lam ** 2 * T)
    c_exact = X0 / Exi                           # = y^(-1/gamma)
    X_opt = c_exact * xi ** (-1 / GAMMA)
    out["max_abs_diff  X*_T(pi*) vs (y xi_T)^(-1/g)"] = float(np.max(np.abs(X_opt / wealth(pi_s) - 1)))
    # Girsanov: under Q, W~ = W + lam t is Brownian and E^Q[e^{-rT} X_T] = x0 for every strategy
    Z = np.exp(-lam * W - 0.5 * lam ** 2 * T)
    for pi in (0.0, pi_s, 1.5):
        q = Z * np.exp(-R * T) * wealth(pi)
        out[f"E^Q[e^-rT X_T] pi={pi:.2f}"] = (q.mean(), q.std() / np.sqrt(n))
    out["E^Q[W_T]  (should be -lam T)"] = (float((Z * W).mean()), -lam * T)
    return out


def verification_checks():
    """Checks that pi* maximises rho(pi) with max rho = kappa, the loss formula, and that the [0,1]-constrained optimum is the clipped pi*."""
    pis = np.linspace(-1, 3, 4001)
    rho = th.ce_growth(pis, MU, R, SIGMA, GAMMA)
    out = {
        "argmax_rho_grid": float(pis[np.argmax(rho)]),
        "pi_star": float(th.merton_fraction(MU, R, SIGMA, GAMMA)),
        "max_rho": float(rho.max()), "kappa": float(th.kappa(MU, R, SIGMA, GAMMA)),
    }
    # loss of a wrong fraction: formula vs direct difference of CE growth rates
    pi_star = th.merton_fraction(MU, R, SIGMA, GAMMA)
    for pi in (0.0, 0.6, 1.0, 1.5):
        out[f"loss pi={pi}"] = (float(th.kappa(MU, R, SIGMA, GAMMA) - th.ce_growth(pi, MU, R, SIGMA, GAMMA)),
                                float(th.loss_rate(pi, pi_star, SIGMA, GAMMA)))
    # constrained optimum = clipped Merton fraction (example where pi* > 1)
    mu2, g2 = 0.20, 2.0
    ps = th.merton_fraction(mu2, R, SIGMA, g2)
    grid = np.linspace(0, 1, 100001)
    out["constrained_example"] = {"mu": mu2, "gamma": g2, "pi_star": float(ps),
                                  "argmax_on_[0,1]": float(grid[np.argmax(th.ce_growth(grid, mu2, R, SIGMA, g2))])}
    # consumption rate for the base parameters
    for beta in (0.05, 0.08):
        out[f"nu(beta={beta})"] = float(th.consumption_rate(beta, MU, R, SIGMA, GAMMA))
    return out


if __name__ == "__main__":
    res = {"symbolic": symbolic_checks(), "martingale": martingale_checks(), "verification": verification_checks()}
    (RESULTS / "part3_theory.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
