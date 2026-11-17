"""Part 5: numerical HJB and dynamic programming.

Solvers (neither uses the closed form; theory.py is only used to *check* the answers):

 1. solve_hjb_fd : finite differences for the HJB equation in log-wealth y = ln x,
        v_tau = sup_pi { b(pi) v_y + 0.5 a(pi) v_yy },  b = r + pi(mu-r) - 0.5 pi^2 sigma^2,  a = pi^2 sigma^2,
        v(0,y) = U(e^y)   (tau = T - t),
    implicit Euler in time, hybrid central/first-order-upwind drift (always a monotone scheme), and Howard policy iteration
    at each time step (the nonlinear maximisation is solved by iterating: freeze pi -> solve a
    tridiagonal linear system -> update pi pointwise).  Boundary condition at y = +-L is the exact
    homogeneity condition v_y = (1-gamma) v of power utility.  Constraints pi in [lo,hi] are imposed
    in the pointwise maximisation.
 2. bellman_dp   : discrete-time Bellman recursion with periodic rebalancing every dt = T/N,
        V_n(y) = max_pi E[ V_{n+1}( y + ln((1-pi)e^{r dt} + pi e^{(mu-s^2/2)dt + s sqrt(dt) Z}) ) ],
    expectation by Gauss-Hermite quadrature, cubic-spline interpolation in y, grid search + parabolic
    refinement in pi.   As N -> infinity it converges to the continuous-time (Merton) answer.
 3. wealth floor : for utility (x-F)^(1-gamma)/(1-gamma) the HJB is solved by the *same* FD code on the
    surplus s = x - F e^{-r(T-t)} (reduction proved in the report), and mapped back to fractions of x.
"""
import json, time
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *       # noqa
import theory as th


# ============================================================================ 1. finite differences
def _policy_from_v(v, h, mu, r, sigma, lo, hi):
    """pointwise maximiser of  pi (mu-r) v_y + 0.5 pi^2 sigma^2 (v_yy - v_y)  over [lo,hi]."""
    vy = np.zeros_like(v); vyy = np.zeros_like(v)
    vy[1:-1] = (v[2:] - v[:-2]) / (2 * h)
    vyy[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / h ** 2
    vy[0], vy[-1] = vy[1], vy[-2]; vyy[0], vyy[-1] = vyy[1], vyy[-2]
    q = vyy - vy                                    # coefficient of 0.5 pi^2 sigma^2
    lin = (mu - r) * vy
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(q < 0, -lin / (sigma ** 2 * q), np.nan)
    pi = np.clip(pi, lo, hi)
    bad = ~np.isfinite(pi)                          # non-concave node: compare the end points
    if bad.any():
        f = lambda p: p * lin + 0.5 * p ** 2 * sigma ** 2 * q
        pi[bad] = np.where(f(hi) >= f(lo), hi, lo)[bad] if np.isfinite(hi) and np.isfinite(lo) else 0.0
    return pi


def solve_hjb_fd(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA, T_=1.0, h=0.05, Nt=40, L=4.0,
                 lo=-20.0, hi=20.0, keep_policy=False, max_pol_iter=60):
    """Finite-difference HJB solver in log-wealth y = ln x: implicit time steps with Howard policy iteration; returns V(0, 1) and pi(0, 1)."""
    y = np.arange(-L, L + 1e-12, h); Ny = len(y); dtau = T_ / Nt
    v = th.crra(np.exp(y), gamma)
    pi = _policy_from_v(v, h, mu, r, sigma, lo, hi)
    policies = [] if keep_policy else None
    n_iter = []
    for k in range(Nt):
        v_old = v.copy()
        for it in range(max_pol_iter):
            b = r + pi * (mu - r) - 0.5 * pi ** 2 * sigma ** 2
            a = pi ** 2 * sigma ** 2
            # hybrid scheme: central differences (2nd order) wherever they keep the matrix monotone
            # (a >= |b| h), first-order upwind elsewhere
            central = a >= np.abs(b) * h
            bp = np.where(central, 0.0, np.maximum(b, 0)); bm = np.where(central, 0.0, np.minimum(b, 0))
            bc = np.where(central, b, 0.0)
            lower = -dtau * (-bm / h - bc / (2 * h) + a / (2 * h ** 2))
            diag = 1 - dtau * (-bp / h + bm / h - a / h ** 2)
            upper = -dtau * (bp / h + bc / (2 * h) + a / (2 * h ** 2))
            ab = np.zeros((3, Ny))
            ab[0, 1:] = upper[:-1]; ab[1, :] = diag; ab[2, :-1] = lower[1:]
            rhs = v_old.copy()
            # boundary rows: v_y = (1-gamma) v  (one-sided difference)
            c = 1 - gamma
            ab[1, 0], ab[0, 1], rhs[0] = -(1 + h * c), 1.0, 0.0           # v1 - v0 - h c v0 = 0
            ab[1, -1], ab[2, -2], rhs[-1] = 1 - h * c, -1.0, 0.0          # vN - v_{N-1} - h c vN = 0
            v_new = solve_banded((1, 1), ab, rhs)
            pi_new = _policy_from_v(v_new, h, mu, r, sigma, lo, hi)
            done = np.max(np.abs(pi_new - pi)) < 1e-10
            pi, v = pi_new, v_new
            if done:
                break
        n_iter.append(it + 1)
        if keep_policy:
            policies.append(pi.copy())
    i0 = int(round(L / h))
    out = dict(y=y, v=v, pi=pi, V0=v[i0], pi0=pi[i0], mean_policy_iters=float(np.mean(n_iter)))
    if keep_policy:
        out["policies"] = np.array(policies)
    return out


def fd_convergence(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA, T_=1.0, lo=-20.0, hi=20.0, exact=None):
    """Separate refinement in space (fine time grid) and in time (fine space grid)."""
    if exact is None:
        exact = (float(th.value_function(0.0, 1.0, T_, mu, r, sigma, gamma)),
                 float(th.merton_fraction(mu, r, sigma, gamma)))
    rows = []
    for h in (0.2, 0.05, 0.025, 0.0125, 0.00625):
        s = solve_hjb_fd(mu, r, sigma, gamma, T_, h=h, Nt=320, lo=lo, hi=hi)
        rows.append(dict(kind="space", h=h, Nt=320, V0=s["V0"], relerr_V=abs(s["V0"] / exact[0] - 1),
                         err_pi=abs(s["pi0"] - exact[1]), pol_iters=s["mean_policy_iters"]))
    ref = solve_hjb_fd(mu, r, sigma, gamma, T_, h=0.025, Nt=1280, lo=lo, hi=hi)["V0"]
    for Nt in (5, 10, 20, 40, 80, 160):
        s = solve_hjb_fd(mu, r, sigma, gamma, T_, h=0.025, Nt=Nt, lo=lo, hi=hi)
        rows.append(dict(kind="time", h=0.025, Nt=Nt, V0=s["V0"], relerr_V=abs(s["V0"] / exact[0] - 1),
                         err_pi=abs(s["pi0"] - exact[1]), pol_iters=s["mean_policy_iters"],
                         self_err_V=abs(s["V0"] / ref - 1)))
    df = pd.DataFrame(rows)
    df["exact_V0"], df["exact_pi"] = exact
    return df


# ============================================================================ 2. Bellman recursion
def bellman_dp(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA, T_=1.0, N=12, h=0.025, L=3.5, n_gh=24,
               pi_grid=None, lo=-1.0, hi=3.0, n_pi=81):
    """Discrete-time Bellman recursion with N rebalancing dates, Gauss-Hermite quadrature over the return shock and a grid search over pi."""
    y = np.arange(-L, L + 1e-12, h)
    dt = T_ / N
    z, w = np.polynomial.hermite_e.hermegauss(n_gh); w = w / w.sum()
    pis = np.linspace(lo, hi, n_pi) if pi_grid is None else pi_grid
    # gross return G[j,k] of the portfolio with fraction pis[j] and shock z[k]
    G = (1 - pis)[:, None] * np.exp(r * dt) + pis[:, None] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[None, :])
    valid = (G > 1e-9).all(axis=1)                  # fractions that could bankrupt the investor give -inf
    lnG = np.log(np.where(G > 1e-9, G, 1.0))
    V = th.crra(np.exp(y), gamma)

    def cont(spl, pi_vec):                          # E[V(y + ln G(pi))] for a vector of pi (one per y node)
        """Continuation value E[V(y + ln G(pi))] by Gauss-Hermite quadrature, one pi per grid node (-inf if wealth can hit zero)."""
        Gp = (1 - pi_vec)[:, None] * np.exp(r * dt) + pi_vec[:, None] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[None, :])
        ok = (Gp > 1e-9).all(axis=1)
        vals = spl(y[:, None] + np.log(np.where(Gp > 1e-9, Gp, 1.0))) @ w
        return np.where(ok, vals, -np.inf)

    for n in range(N):
        spl = CubicSpline(y, V, extrapolate=True)
        vals = np.empty((len(y), len(pis)))
        for j in range(len(pis)):
            vals[:, j] = spl(y + lnG[j][:, None]).reshape(len(y), n_gh) @ w if False else \
                np.stack([spl(y + lnG[j, k]) for k in range(n_gh)], axis=1) @ w
        vals[:, ~valid] = -np.inf
        j = np.clip(np.argmax(vals, axis=1), 1, len(pis) - 2)
        # parabolic refinement of the maximiser (vertex of the parabola through three grid points)
        i = np.arange(len(y)); f0, f1, f2 = vals[i, j - 1], vals[i, j], vals[i, j + 1]
        dpi = pis[1] - pis[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            shift = 0.5 * (f0 - f2) / (f0 - 2 * f1 + f2)
        shift = np.where(np.isfinite(shift), np.clip(shift, -1, 1), 0.0)
        pi_star = np.clip(pis[j] + shift * dpi, lo, hi)
        V = cont(spl, pi_star)
    i0 = int(round(L / h))
    return dict(y=y, V0=V[i0], pi0=pi_star[i0], pi_profile=pi_star)


def dp_convergence(mu=MU, r=R, sigma=SIGMA, gamma=GAMMA, T_=1.0, lo=-1.0, hi=3.0, exact=None):
    """DP error in V and pi against the closed form as the number of rebalancing dates N grows."""
    if exact is None:
        exact = (float(th.value_function(0.0, 1.0, T_, mu, r, sigma, gamma)), float(th.merton_fraction(mu, r, sigma, gamma)))
    rows = []
    for N in (1, 2, 4, 12, 52):
        s = bellman_dp(mu, r, sigma, gamma, T_, N=N, lo=lo, hi=hi)
        rows.append(dict(N=N, dt=T_ / N, V0=s["V0"], relerr_V=abs(s["V0"] / exact[0] - 1), pi0=s["pi0"], err_pi=abs(s["pi0"] - exact[1])))
    df = pd.DataFrame(rows); df["exact_V0"], df["exact_pi"] = exact
    return df


def accuracy_runtime(gamma=5.0, T_=1.0):
    """Accuracy vs wall-clock time for both solvers on the SAME problem (gamma = 5, pi* = 0.3), errors against
    the continuous-time closed form.  DP solves the discretely-rebalanced problem, so its error also contains the
    O(dt) rebalancing gap; FD solves the continuous problem directly."""
    exact_V = float(th.value_function(0.0, 1.0, T_, MU, R, SIGMA, gamma)); exact_pi = float(th.merton_fraction(MU, R, SIGMA, gamma))
    rows = []
    # h <= 0.025 keeps the drift discretisation central (a >= |b| h) at gamma = 5; coarser h switches to upwind
    for h, Nt in ((0.025, 20), (0.025, 80), (0.0125, 320), (0.00625, 1280)):
        t = time.perf_counter(); s = solve_hjb_fd(MU, R, SIGMA, gamma, T_, h=h, Nt=Nt); el = time.perf_counter() - t
        rows.append(dict(method="FD (HJB)", setting=f"h={h}, Nt={Nt}", seconds=el,
                         relerr_V=abs(s["V0"] / exact_V - 1), err_pi=abs(s["pi0"] - exact_pi)))
    for N in (4, 12, 52, 252):
        t = time.perf_counter(); d = bellman_dp(MU, R, SIGMA, gamma, T_, N=N); el = time.perf_counter() - t
        rows.append(dict(method="DP (Bellman)", setting=f"N={N}", seconds=el,
                         relerr_V=abs(d["V0"] / exact_V - 1), err_pi=abs(d["pi0"] - exact_pi)))
    return pd.DataFrame(rows)


def order(df, xcol, ycol):
    """Empirical convergence order: slope of log(error) against log(step)."""
    return float(np.polyfit(np.log(df[xcol]), np.log(df[ycol]), 1)[0])


# ============================================================================ 3. constrained & floor cases
def constrained_case(mu=0.20, gamma=2.0, T_=1.0):
    """pi* = 1.75 > 1 so the long-only/no-leverage constraint binds; exact V uses rho(1)."""
    pi_s = th.merton_fraction(mu, R, SIGMA, gamma)
    exact_V = float(th.expected_utility_const_pi(1.0, 1.0, T_, mu, R, SIGMA, gamma))
    s = solve_hjb_fd(mu, R, SIGMA, gamma, T_, h=0.025, Nt=160, lo=0.0, hi=1.0)
    d = bellman_dp(mu, R, SIGMA, gamma, T_, N=252, lo=0.0, hi=1.0)
    # second constrained example where the constraint is slack: base parameters, pi* = 0.5 in [0,1]
    s2 = solve_hjb_fd(MU, R, SIGMA, GAMMA, T_, h=0.025, Nt=160, lo=0.0, hi=1.0)
    exact2 = float(th.value_function(0, 1.0, T_, MU, R, SIGMA, GAMMA))
    return dict(mu=mu, gamma=gamma, pi_star_unconstrained=pi_s, exact_V0_constrained=exact_V,
                fd_V0=s["V0"], fd_relerr=abs(s["V0"] / exact_V - 1), fd_pi0=s["pi0"],
                fd_pi_min=float(s["pi"].min()), fd_pi_max=float(s["pi"].max()),
                dp_V0=d["V0"], dp_relerr=abs(d["V0"] / exact_V - 1), dp_pi0=d["pi0"],
                slack_case_fd_relerr=abs(s2["V0"] / exact2 - 1), slack_case_pi0=s2["pi0"])


def floor_case(F=0.5, T_=1.0, x_grid=(0.6, 0.8, 1.0, 1.5, 2.0)):
    """Utility (x-F)^(1-g)/(1-g).  FD on the surplus variable; fractions mapped back to total wealth."""
    s = solve_hjb_fd(MU, R, SIGMA, GAMMA, T_, h=0.025, Nt=160)
    # value on surplus s = e^y: v(0,y) ; V(0,x) = v(0, ln(x - F e^{-rT}))
    y = s["y"]; v = s["v"]
    rows = []
    for x in x_grid:
        sur = x - F * np.exp(-R * T_)
        V_num = np.interp(np.log(sur), y, v)
        V_ex = float(th.value_function_floor(0.0, x, T_, F, MU, R, SIGMA, GAMMA))
        pi_num = np.interp(np.log(sur), y, s["pi"]) * sur / x           # surplus fraction -> total-wealth fraction
        pi_ex = float(th.floor_fraction(x, 0.0, T_, F, MU, R, SIGMA, GAMMA))
        rows.append(dict(x0=x, floor_PV=F * np.exp(-R * T_), V_num=V_num, V_exact=V_ex, relerr_V=abs(V_num / V_ex - 1),
                         pi_num=pi_num, pi_exact=pi_ex))
    return pd.DataFrame(rows)


def time_dependence():
    """Policy pi(t,y) at several times from the FD solver (homogeneity => independent of t and y)."""
    s = solve_hjb_fd(MU, R, SIGMA, GAMMA, 1.0, h=0.05, Nt=80, keep_policy=True)
    P = s["policies"]; mid = (np.abs(s["y"]) <= 2)
    return dict(pi_t_min=float(P[:, mid].min()), pi_t_max=float(P[:, mid].max()))


def make_figure(fdc, dpc, cons):
    """Figure fig_hjb.png: FD errors, DP errors, and unconstrained vs pi in [0,1] FD policies."""
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3))
    sp_, ti = fdc[fdc.kind == "space"], fdc[fdc.kind == "time"]
    ax[0].loglog(sp_.h, sp_.relerr_V, "o-", label="value, space ($h$)"); ax[0].loglog(sp_.h, sp_.err_pi, "s--", label="$\\pi$, space ($h$)")
    ax[0].loglog(1.0 / ti.Nt, ti.self_err_V, "^-", label="value, time ($\\Delta\\tau$, vs fine ref.)")
    ax[0].set_xlabel("$h$ or $\\Delta\\tau$"); ax[0].set_ylabel("error"); ax[0].legend(fontsize=7); ax[0].set_title("(a) finite-difference HJB", fontsize=9)
    ax[1].loglog(dpc.dt, dpc.relerr_V, "o-", label="value"); ax[1].loglog(dpc.dt, dpc.err_pi, "s--", label="$\\pi$")
    ax[1].loglog(dpc.dt, dpc.err_pi.iloc[0] * dpc.dt / dpc.dt.iloc[0], "k:", lw=.8)
    ax[1].set_xlabel("rebalancing interval $\\Delta t$ (years)"); ax[1].legend(fontsize=7); ax[1].set_title("(b) Bellman recursion, $\\gamma=5$ (dotted: order 1)", fontsize=9)
    mu2, g2 = 0.20, 2.0
    s = solve_hjb_fd(mu2, R, SIGMA, g2, 1.0, h=0.05, Nt=40, lo=0.0, hi=1.0)
    su = solve_hjb_fd(mu2, R, SIGMA, g2, 1.0, h=0.05, Nt=40)
    ax[2].plot(su["y"], su["pi"], label="unconstrained (FD)"); ax[2].plot(s["y"], s["pi"], label="$\\pi\\in[0,1]$ (FD)")
    ax[2].axhline(th.merton_fraction(mu2, R, SIGMA, g2), c="k", ls=":", lw=.8)
    ax[2].set_ylim(0, 2.1); ax[2].set_xlim(-2, 2); ax[2].set_xlabel("log-wealth $y$"); ax[2].set_ylabel("$\\pi(0,y)$")
    ax[2].legend(fontsize=7, loc="center right"); ax[2].set_title("(c) $\\mu=0.20,\\gamma=2$: $\\pi^*=1.75$ clipped to 1", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "fig_hjb.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    fdc = fd_convergence(); fdc.to_csv(RESULTS / "part5_fd_convergence.csv", index=False)
    dpc3 = dp_convergence().assign(gamma=3.0); dpc = dp_convergence(gamma=5.0).assign(gamma=5.0)
    pd.concat([dpc3, dpc]).to_csv(RESULTS / "part5_dp_convergence.csv", index=False)
    cons = constrained_case(); flo = floor_case(); flo.to_csv(RESULTS / "part5_wealth_floor.csv", index=False)
    td = time_dependence()
    art = accuracy_runtime(); art.to_csv(RESULTS / "part5_accuracy_runtime.csv", index=False)
    sp_, ti = fdc[fdc.kind == "space"], fdc[fdc.kind == "time"]
    summ = dict(order_space_V=order(sp_[sp_.h <= 0.05], "h", "relerr_V"), order_space_pi=order(sp_[sp_.h <= 0.05], "h", "err_pi"),
                order_time_V=order(ti.assign(dt=1.0 / ti.Nt), "dt", "self_err_V"),
                order_dp_V_g5=order(dpc[dpc.N <= 12], "dt", "relerr_V"), order_dp_pi_g5=order(dpc[dpc.N <= 12], "dt", "err_pi"),
                constrained=cons, time_dependence=td, runtime_s=time.time() - t0)
    (RESULTS / "part5_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    make_figure(fdc, dpc, cons)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(fdc.round(7)); print(dpc.round(7)); print(flo.round(6)); print(art); print(json.dumps(summ, indent=2, default=float))
