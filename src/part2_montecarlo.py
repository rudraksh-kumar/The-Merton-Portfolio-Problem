"""Part 2: Monte Carlo study.

 A. Expected utility of five constant-fraction strategies: 10^6 exact GBM paths vs closed form.
 B. Strong convergence of Euler-Maruyama and Milstein for dX = X[(r+pi(mu-r))dt + pi sigma dW].
 B2. Weak convergence: error in E[U(X_T)] (CRRA utility, the quantity the investor cares about), estimated with
    common random numbers against the exact solution, plus the error in E[X_T], which is known in closed form.
 B3. Noise-free weak check: exact recursion for E[X_T^N] and E[(X_T^N)^2] of both schemes (results/part2_weak_moments_exact.csv).
 C. Monte Carlo error decays like M^(-1/2).
 D. Estimation risk: build pi from a drift estimated on T_est years of daily data and compare the
    simulated loss in certainty-equivalent growth with the formula 1/(2 gamma T_est).
 E. Parameter sweep (gamma, sigma, mu, r, T).
Outputs: results/part2_*.csv, results/part2_summary.json, figures/fig_mc.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *           # noqa
import theory as th

rng = np.random.default_rng(SEED)
PI_STAR = th.merton_fraction(MU, R, SIGMA, GAMMA)
STRATS = {"Merton $\\pi^*$": PI_STAR, "All-in": 1.0, "60/40": 0.6, "1.5x leverage": 1.5, "Risk-free": 0.0}


# ------------------------------------------------------------------ A
def exact_terminal(pi, W_T, T_=T, mu=MU, r=R, sigma=SIGMA):
    """Exact terminal wealth under a constant fraction pi given the Brownian value W_T (lognormal solution of the wealth SDE)."""
    b, s = r + pi * (mu - r), pi * sigma
    return X0 * np.exp((b - 0.5 * s ** 2) * T_ + s * W_T)


def part_A(M=1_000_000, batches=4):
    """Monte Carlo expected utility and CE growth of the five strategies against the closed form (common random numbers)."""
    rows = []
    U = {k: [] for k in STRATS}
    for _ in range(batches):
        W = np.sqrt(T) * rng.standard_normal(M // batches)      # common random numbers for all strategies
        for k, pi in STRATS.items():
            U[k].append(th.crra(exact_terminal(pi, W), GAMMA))
    for k, pi in STRATS.items():
        u = np.concatenate(U[k]); EU, se = u.mean(), u.std(ddof=1) / np.sqrt(len(u))
        EU_th = th.expected_utility_const_pi(pi, X0, T, MU, R, SIGMA, GAMMA)
        # CE growth: rho = ln( (EU (1-g))^(1/(1-g)) / x0 ) / T ; delta-method s.e.
        rho_mc = np.log(((1 - GAMMA) * EU) ** (1 / (1 - GAMMA)) / X0) / T
        se_rho = se / (T * abs(1 - GAMMA) * abs(EU))
        rho_th = float(th.ce_growth(pi, MU, R, SIGMA, GAMMA))
        rows.append(dict(strategy=k, pi=pi, EU_mc=EU, EU_se=se, EU_formula=EU_th,
                         # risk-free: deterministic, se ~ 1e-20, so a 1-ulp rounding difference would give a huge z
                         z=(EU - EU_th) / se if se > 1e-12 * abs(EU_th) else 0.0, rho_mc=rho_mc, rho_se=se_rho, rho_formula=rho_th))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B
def strong_convergence(pi, T_=1.0, Ns=(2, 4, 8, 16, 32, 64, 128, 256), M=200_000, chunk=20_000):
    """Strong error E|X_T - X_T^N| of Euler-Maruyama and Milstein against the exact solution on the same Brownian path."""
    b, s = R + pi * (MU - R), pi * SIGMA
    Nf = max(Ns)
    err = {N: {"EM": [], "Mil": []} for N in Ns}
    for _ in range(M // chunk):
        dW = np.sqrt(T_ / Nf) * rng.standard_normal((chunk, Nf))
        exact = X0 * np.exp((b - 0.5 * s ** 2) * T_ + s * dW.sum(1))
        for N in Ns:
            dt = T_ / N
            dWc = dW.reshape(chunk, N, Nf // N).sum(2)
            XE = np.full(chunk, X0); XM = np.full(chunk, X0)
            for n in range(N):
                w = dWc[:, n]
                XE = XE * (1 + b * dt + s * w)
                XM = XM * (1 + b * dt + s * w + 0.5 * s ** 2 * (w ** 2 - dt))
            err[N]["EM"].append(np.abs(exact - XE)); err[N]["Mil"].append(np.abs(exact - XM))
    rows = []
    for N in Ns:
        e1, e2 = np.concatenate(err[N]["EM"]), np.concatenate(err[N]["Mil"])
        rows.append(dict(pi=pi, N=N, dt=T_ / N, EM=e1.mean(), EM_se=e1.std() / np.sqrt(len(e1)),
                         Milstein=e2.mean(), Mil_se=e2.std() / np.sqrt(len(e2))))
    df = pd.DataFrame(rows)
    slope = lambda c: np.polyfit(np.log(df.dt), np.log(df[c]), 1)[0]
    return df, slope("EM"), slope("Milstein")


# ------------------------------------------------------------------ B2
def weak_convergence(pi, T_=1.0, Ns=(4, 8, 16, 32, 64), M=2_000_000, chunk=50_000):
    """Weak error |E U(X_T^N) - E U(X_T)| for EM and Milstein.
    E U(X_T^N) - E U(X_T) is estimated as the mean of U(X_T^N) - U(X_T^exact) on the SAME Brownian path
    (common random numbers): E U(X_T^exact) equals the closed form, and the difference has a tiny variance,
    so an O(dt) bias is visible far below the plain Monte Carlo error.
    Also returns the exact weak error in E[X_T]: E[X_T^N] = x0 (1 + b dt)^N for both schemes (the Milstein
    correction has mean zero), against x0 e^{bT}."""
    b, s = R + pi * (MU - R), pi * SIGMA
    Nf = max(Ns)
    diff = {N: {"EM": [], "Mil": []} for N in Ns}; neg = {N: 0 for N in Ns}
    for _ in range(M // chunk):
        dW = np.sqrt(T_ / Nf) * rng.standard_normal((chunk, Nf))
        U_ex = th.crra(X0 * np.exp((b - 0.5 * s ** 2) * T_ + s * dW.sum(1)), GAMMA)
        for N in Ns:
            dt = T_ / N
            dWc = dW.reshape(chunk, N, Nf // N).sum(2)
            XE = np.full(chunk, X0); XM = np.full(chunk, X0)
            for n in range(N):
                w = dWc[:, n]
                XE = XE * (1 + b * dt + s * w)
                XM = XM * (1 + b * dt + s * w + 0.5 * s ** 2 * (w ** 2 - dt))
            # EM/Milstein wealth can step below 0 for very coarse steps (dt = 1 at pi = 1.5), where CRRA is
            # undefined; Ns starts at 4 steps so that this has probability < 1e-11.  Count it anyway.
            neg[N] += int((XE <= 0).sum() + (XM <= 0).sum())
            diff[N]["EM"].append(th.crra(np.maximum(XE, 1e-12), GAMMA) - U_ex)
            diff[N]["Mil"].append(th.crra(np.maximum(XM, 1e-12), GAMMA) - U_ex)
    EU_th = th.expected_utility_const_pi(pi, X0, T_, MU, R, SIGMA, GAMMA)
    rows = []
    for N in Ns:
        dt = T_ / N
        d1, d2 = np.concatenate(diff[N]["EM"]), np.concatenate(diff[N]["Mil"])
        rows.append(dict(pi=pi, N=N, dt=dt,
                         EM_U=abs(d1.mean()) / abs(EU_th), EM_U_se=d1.std() / np.sqrt(len(d1)) / abs(EU_th),
                         Mil_U=abs(d2.mean()) / abs(EU_th), Mil_U_se=d2.std() / np.sqrt(len(d2)) / abs(EU_th),
                         mean_X_exact=abs(X0 * np.exp(b * T_) - X0 * (1 + b * dt) ** N), n_negative=neg[N]))
    df = pd.DataFrame(rows)
    slope = lambda c: np.polyfit(np.log(df.dt), np.log(df[c]), 1)[0]
    return df, slope("EM_U"), slope("Mil_U"), slope("mean_X_exact")


def weak_moments_exact(pi, T_=1.0, Ns=(2, 4, 8, 16, 32, 64, 128, 256), M_check=2_000_000):
    """Noise-free weak check: for GBM the one-step conditional moments of EM and Milstein are polynomials in dt,
    so E[X_N] and E[X_N^2] follow EXACTLY from a recursion (no sampling error at all):
      EM:       E[X_{n+1}|X_n] = a X_n,   E[X_{n+1}^2|X_n] = (a^2 + s^2 dt) X_n^2,                   a = 1 + b dt
      Milstein: E[X_{n+1}|X_n] = a X_n,   E[X_{n+1}^2|X_n] = (a^2 + s^2 dt + s^4 dt^2 / 2) X_n^2
    against the exact E X_T = x0 e^{bT} and E X_T^2 = x0^2 e^{(2b + s^2) T}.  The weak order is the log-log slope.
    A Monte Carlo run at N = 4 confirms the Milstein second-moment recursion.  (Approach from a teammate's version;
    it uses its own generator so the random numbers of the other Part 2 experiments are unchanged.)"""
    b, s = R + pi * (MU - R), pi * SIGMA
    e1, e2 = X0 * np.exp(b * T_), X0 ** 2 * np.exp((2 * b + s ** 2) * T_)
    rows = []
    for N in Ns:
        dt = T_ / N; a = 1 + b * dt
        rows.append(dict(pi=pi, N=N, dt=dt, mean_both=abs(X0 * a ** N / e1 - 1),
                         m2_EM=abs(X0 ** 2 * (a ** 2 + s ** 2 * dt) ** N / e2 - 1),
                         m2_Mil=abs(X0 ** 2 * (a ** 2 + s ** 2 * dt + 0.5 * s ** 4 * dt ** 2) ** N / e2 - 1)))
    df = pd.DataFrame(rows)
    sl = lambda c: float(np.polyfit(np.log(df.dt), np.log(df[c]), 1)[0])
    g = np.random.default_rng(SEED + 7)
    N = 4; dt = T_ / N; a = 1 + b * dt; X = np.full(M_check, X0)
    for _ in range(N):
        w = np.sqrt(dt) * g.standard_normal(M_check)
        X = X * (1 + b * dt + s * w + 0.5 * s ** 2 * (w ** 2 - dt))
    chk = dict(mc_m2_mil_N4=float((X ** 2).mean()), mc_se=float((X ** 2).std() / np.sqrt(M_check)),
               recursion_m2_mil_N4=float(X0 ** 2 * (a ** 2 + s ** 2 * dt + 0.5 * s ** 4 * dt ** 2) ** N))
    chk["z"] = (chk["mc_m2_mil_N4"] - chk["recursion_m2_mil_N4"]) / chk["mc_se"]
    return df, dict(order_mean=sl("mean_both"), order_m2_EM=sl("m2_EM"), order_m2_Mil=sl("m2_Mil"), **chk)


# ------------------------------------------------------------------ C
def mc_error_decay(Ms=(10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6), reps=40):
    """RMSE of the Monte Carlo CE-growth estimate for increasing path counts M (should fall like M^-1/2)."""
    pi = PI_STAR; rows = []
    rho_th = float(th.ce_growth(pi, MU, R, SIGMA, GAMMA))
    for M in Ms:
        est = []
        for _ in range(reps):
            W = np.sqrt(T) * rng.standard_normal(M)
            EU = th.crra(exact_terminal(pi, W), GAMMA).mean()
            est.append(np.log(((1 - GAMMA) * EU) ** (1 / (1 - GAMMA)) / X0) / T)
        est = np.array(est)
        rows.append(dict(M=M, rmse=np.sqrt(np.mean((est - rho_th) ** 2)), bias=est.mean() - rho_th))
    df = pd.DataFrame(rows)
    return df, np.polyfit(np.log(df.M), np.log(df.rmse), 1)[0]


# ------------------------------------------------------------------ D
def estimation_risk(Tests=(3, 5, 7, 10), reps=400_000, n_per_year=TRADING_DAYS):
    """Exact sampling distribution of (m_hat, sigma_hat^2) from n = T_est*252 daily log-returns."""
    rows = []
    m_true = MU - 0.5 * SIGMA ** 2
    for Te in Tests:
        n = int(Te * n_per_year); dt = 1.0 / n_per_year
        m_hat = m_true + SIGMA / np.sqrt(Te) * rng.standard_normal(reps)          # mean daily log-ret, annualised
        s2_hat = SIGMA ** 2 * rng.chisquare(n - 1, reps) / (n - 1)                  # independent of m_hat
        mu_hat = m_hat + 0.5 * s2_hat
        pi_hat = (mu_hat - R) / (GAMMA * s2_hat)
        pi_hat_known_sigma = (m_hat + 0.5 * SIGMA ** 2 - R) / (GAMMA * SIGMA ** 2)
        loss = th.ce_growth(PI_STAR, MU, R, SIGMA, GAMMA) - th.ce_growth(pi_hat, MU, R, SIGMA, GAMMA)
        loss_k = th.ce_growth(PI_STAR, MU, R, SIGMA, GAMMA) - th.ce_growth(pi_hat_known_sigma, MU, R, SIGMA, GAMMA)
        rows.append(dict(T_est=Te, formula=th.expected_loss_estimated_drift(GAMMA, Te),
                         sim_mean_loss=loss.mean(), sim_mean_loss_sigma_known=loss_k.mean(),
                         sim_median_loss=np.median(loss), sim_p95_loss=np.quantile(loss, .95),
                         sd_pi_hat=pi_hat.std(), sd_formula=th.sd_of_estimated_fraction(SIGMA, GAMMA, Te),
                         P_pi_hat_gt_1=(pi_hat > 1).mean(), P_pi_hat_lt_0=(pi_hat < 0).mean(),
                         loss_capped_01=(th.ce_growth(PI_STAR, MU, R, SIGMA, GAMMA)
                                         - th.ce_growth(np.clip(pi_hat, 0, 1), MU, R, SIGMA, GAMMA)).mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E
def parameter_sweep():
    """Closed-form pi*, kappa and the 60/40 loss over a grid of gamma, sigma, mu and r."""
    rows = []
    for g in (2.0, 3.0, 5.0):
        for sg in (0.15, 0.20, 0.30):
            for mu in (0.09, 0.12, 0.15):
                for r in (0.04, 0.06, 0.08):
                    ps = th.merton_fraction(mu, r, sg, g)
                    rows.append(dict(gamma=g, sigma=sg, mu=mu, r=r, pi_star=ps,
                                     kappa=th.kappa(mu, r, sg, g), ce_gain_over_rf=th.kappa(mu, r, sg, g) - r,
                                     loss_60_40=th.loss_rate(0.6, ps, sg, g)))
    return pd.DataFrame(rows)


def sweep_figure(S):
    """pi* against gamma and sigma, kappa against mu (Part 2 hyper-parameter study; closed forms from theory.py)."""
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.2))
    g = np.linspace(1.5, 8, 100)
    for sg in (0.15, 0.20, 0.30): ax[0].plot(g, th.merton_fraction(MU, R, sg, g), label=f"$\\sigma={sg}$")
    ax[0].axhline(1, c="k", lw=.5, ls=":"); ax[0].set_xlabel("risk aversion $\\gamma$"); ax[0].set_ylabel("$\\pi^*$"); ax[0].legend(fontsize=7); ax[0].set_title("(a) $\\pi^*$ vs $\\gamma$", fontsize=9)
    sg = np.linspace(0.1, 0.4, 100)
    for gm in (2, 3, 5): ax[1].plot(sg, th.merton_fraction(MU, R, sg, gm), label=f"$\\gamma={gm}$")
    ax[1].axhline(1, c="k", lw=.5, ls=":"); ax[1].set_xlabel("volatility $\\sigma$"); ax[1].legend(fontsize=7); ax[1].set_title("(b) $\\pi^*$ vs $\\sigma$", fontsize=9)
    for rr in (0.04, 0.06, 0.08): ax[2].plot(np.linspace(0.07, 0.16, 50), th.merton_fraction(np.linspace(0.07, 0.16, 50), rr, SIGMA, GAMMA), label=f"$r={rr}$")
    ax[2].axhline(1, c="k", lw=.5, ls=":"); ax[2].set_xlabel("drift $\\mu$"); ax[2].legend(fontsize=7); ax[2].set_title("(c) $\\pi^*$ vs $\\mu$ and $r$", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "fig_sweep.png", dpi=200); plt.close(fig)


def horizon_check():
    """The CE growth rate does not depend on T (E U(X_T) = U(x0 e^{rho T})): check T = 1, 5, 20."""
    rows = []
    for T_ in (1.0, 5.0, 20.0):
        W = np.sqrt(T_) * rng.standard_normal(1_000_000)
        EU = th.crra(exact_terminal(PI_STAR, W, T_=T_), GAMMA).mean()
        rows.append(dict(T=T_, rho_mc=np.log(((1 - GAMMA) * EU) ** (1 / (1 - GAMMA)) / X0) / T_,
                         rho_formula=float(th.ce_growth(PI_STAR, MU, R, SIGMA, GAMMA))))
    return pd.DataFrame(rows)


def make_figure(A, sc1, sc2, errdf, wk, wm):
    """Figure fig_mc.png: CE growth vs pi, strong order, Monte Carlo error and weak order."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    pis = np.linspace(-0.2, 1.8, 300)
    for g, c in ((2.0, "tab:green"), (3.0, "tab:blue"), (5.0, "tab:red")):
        ax[0].plot(pis, 100 * th.ce_growth(pis, MU, R, SIGMA, g), c=c, label=f"$\\gamma={g:g}$")
    ax[0].errorbar(A.pi, 100 * A.rho_mc, yerr=200 * A.rho_se, fmt="ko", ms=4, label="MC ($\\gamma=3$)")
    ax[0].axhline(100 * R, c="grey", ls=":", lw=.8)
    ax[0].set_xlabel("stock fraction $\\pi$"); ax[0].set_ylabel("CE growth rate $\\rho(\\pi)$ (% p.a.)")
    ax[0].set_ylim(-4, 9); ax[0].legend(fontsize=7); ax[0].set_title("(a) formula vs Monte Carlo", fontsize=9)
    for df, mk, lab in ((sc1, "o", r"$\pi^*=0.5$"), (sc2, "s", r"$\pi=1.5$")):
        ax[1].loglog(df.dt, df.EM, mk + "-", c="tab:red", label=f"Euler, {lab}", ms=4)
        ax[1].loglog(df.dt, df.Milstein, mk + "-", c="tab:blue", label=f"Milstein, {lab}", ms=4)
    ref = sc1.dt.values; ax[1].loglog(ref, 0.8 * ref ** 0.5 * sc1.EM.iloc[0] / (0.8 * ref[0] ** .5), "k:", lw=.8)
    ax[1].loglog(ref, sc1.Milstein.iloc[0] * ref / ref[0], "k--", lw=.8)
    ax[1].set_xlabel("step $\\Delta t$"); ax[1].set_ylabel("strong error $E|X_T-X_T^N|$")
    ax[1].legend(fontsize=7); ax[1].set_title("(b) strong order (dotted 1/2, dashed 1)", fontsize=9)
    ax[3].loglog(wk.dt, wk.EM_U, "o-", c="tab:red", ms=4, label=r"Euler, $E\,U(X_T)$")
    ax[3].loglog(wk.dt, wk.Mil_U, "s-", c="tab:blue", ms=4, label=r"Milstein, $E\,U(X_T)$")
    ax[3].loglog(wm.dt, wm.m2_EM, ":", c="tab:red", lw=1.4, label=r"Euler, $E\,X_T^2$ (exact)")
    ax[3].loglog(wm.dt, wm.m2_Mil, ":", c="tab:blue", lw=1.4, label=r"Milstein, $E\,X_T^2$ (exact)")
    ax[3].loglog(wm.dt, wm.mean_both, "-", c="grey", lw=1, label=r"both, $E\,X_T$ (exact)")
    ax[3].loglog(wm.dt, wk.EM_U.iloc[0] * wm.dt / wk.dt.iloc[0], "k--", lw=.8)
    ax[3].set_xlabel(r"step $\Delta t$"); ax[3].set_ylabel(r"relative weak error")
    ax[3].legend(fontsize=6.5); ax[3].set_title(r"(d) weak order, $\pi=1.5$ (dashed 1)", fontsize=9)
    ax[2].loglog(errdf.M, errdf.rmse, "ko-", ms=4); ax[2].loglog(errdf.M, errdf.rmse.iloc[0] * (errdf.M / errdf.M.iloc[0]) ** -0.5, "k:", lw=.8)
    ax[2].set_xlabel("paths $M$"); ax[2].set_ylabel("RMSE of $\\hat\\rho(\\pi^*)$"); ax[2].set_title("(c) Monte Carlo error $\\propto M^{-1/2}$", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "fig_mc.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    A = part_A(); A.to_csv(RESULTS / "part2_expected_utility.csv", index=False)
    s1, e1, m1 = strong_convergence(PI_STAR); s2, e2, m2 = strong_convergence(1.5)
    pd.concat([s1, s2]).to_csv(RESULTS / "part2_strong_convergence.csv", index=False)
    W1, we1, wm1, wx1 = weak_convergence(PI_STAR); W2, we2, wm2, wx2 = weak_convergence(1.5)
    pd.concat([W1, W2]).to_csv(RESULTS / "part2_weak_convergence.csv", index=False)
    C, slopeC = mc_error_decay(); C.to_csv(RESULTS / "part2_mc_error.csv", index=False)
    D = estimation_risk(); D.to_csv(RESULTS / "part2_estimation_risk.csv", index=False)
    S = parameter_sweep(); S.to_csv(RESULTS / "part2_param_sweep.csv", index=False); sweep_figure(S)
    H = horizon_check(); H.to_csv(RESULTS / "part2_horizon.csv", index=False)
    Wm1, wmo1 = weak_moments_exact(PI_STAR); Wm2, wmo2 = weak_moments_exact(1.5)
    pd.concat([Wm1, Wm2]).to_csv(RESULTS / "part2_weak_moments_exact.csv", index=False)
    make_figure(A, s1, s2, C, W2, Wm2)
    summ = dict(weak_moments_exact_pi_star=wmo1, weak_moments_exact_lev=wmo2,
strong_slope_EM_pi_star=e1, strong_slope_Mil_pi_star=m1, strong_slope_EM_lev=e2, strong_slope_Mil_lev=m2,
                weak_slope_EM_pi_star=we1, weak_slope_Mil_pi_star=wm1, weak_slope_EM_lev=we2, weak_slope_Mil_lev=wm2,
                weak_slope_meanX=wx2, mc_rmse_slope=slopeC, max_abs_z_EU=float(A.z.abs().max()))
    (RESULTS / "part2_summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(A.round(5)); print(summ); print(C); print(D.round(5)); print(H); print(s1.round(6)); print(s2.round(6)); print(W1); print(W2)
