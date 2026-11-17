"""Part 4: multi-asset extension.

Model: dS_i/S_i = mu_i dt + sigma_i dW_i,  corr(dW_i,dW_j) = rho_ij,  Sigma = covariance matrix of returns.
Unconstrained optimum  pi* = Sigma^{-1}(mu - r 1)/gamma,  kappa = r + (mu-r)' Sigma^{-1} (mu-r)/(2 gamma),
loss of an estimated rule  0.5 gamma (pi-pi*)' Sigma (pi-pi*)  with expectation  N/(2 gamma T_est)  if only mu is estimated.

Data: NSE's official daily Total Returns Indices (data/nse_tri_daily.csv, from niftyindices.com via prep_nse_tri.py) for
Nifty 50, Bank, IT, FMCG, Pharma, 2007-09-17 onwards, assets added one at a time.  Falls back to the 94 weekly
NSE closes from TapeTide (data/tapetide_weekly_sector_indices.csv) if the daily file is absent.
Experiments
  A. one-asset consistency (N=1 reduces to Part 3) and kappa as assets are added
  B. weights with sample vs Ledoit-Wolf covariance; unconstrained, long-only, long-only no-leverage
  C. sensitivity of pi* to gamma, mu, sigma, r
  D. effect of correlation (synthetic two assets): sd(pi_hat) ~ 1/sqrt(1-rho^2) while the CE loss stays N/(2 gamma T)
  E. out-of-sample simulation with a known truth: sample vs LW covariance, unconstrained vs long-only
  F. rolling out-of-sample backtest on the real daily data (monthly rebalancing, trailing 5- and 10-year windows),
     same evaluation period and metrics as Part 1
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *          # noqa
import theory as th

NAMES = ["nifty50", "nifty_bank", "nifty_it", "nifty_fmcg", "nifty_pharma"]
LABELS = ["Nifty 50", "Bank", "IT", "FMCG", "Pharma"]
rng = np.random.default_rng(SEED + 4)


def load_weekly():
    """Returns (log-returns DataFrame, years, periods per year, label).  Daily TRI if available, else weekly closes."""
    f = DATA / "nse_tri_daily.csv"
    if f.exists():
        px = pd.read_csv(f, parse_dates=["date"]).set_index("date")[NAMES].sort_index()
        px = px[px.index >= pd.Timestamp("2007-09-17")].dropna()
        label = "NSE daily Total Returns Indices"
    else:
        px = pd.read_csv(DATA / "tapetide_weekly_sector_indices.csv", parse_dates=["date"]).set_index("date")[NAMES]
        label = "NSE weekly closes (TapeTide)"
    lr = np.log(px).diff().dropna()
    years = (px.index[-1] - px.index[0]).days / 365.25
    return lr, years, len(lr) / years, label


def risk_free_mean(index):
    """mean RBI 91-day T-bill auction yield over the sample window (same source as Part 1); RF_CONST if absent."""
    f = DATA / "tbill_91d_auctions.csv"
    if not f.exists():
        return RF_CONST, f"constant {RF_CONST}"
    s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["yield_pct"].sort_index() / 100.0
    return float(s[index[0]:index[-1]].mean()), "mean RBI 91-day T-bill auction yield over the sample"


def moments(lr, ann, shrink=False):
    """Annualised arithmetic drifts and covariance (sample or Ledoit-Wolf) from log returns."""
    X = lr.values
    C = np.atleast_2d(np.cov(X.T, ddof=1))                      # (N,N) even when N=1
    S = (LedoitWolf().fit(X).covariance_ if shrink else C) * ann
    m = X.mean(0) * ann
    return m + 0.5 * np.diag(C) * ann, S     # arithmetic drift (uses the sample variance), covariance


def long_only(mu, r, Sigma, gamma, leverage_cap=None):
    """maximise rho(pi) s.t. pi >= 0 (and sum(pi) <= leverage_cap).  Concave QP -> SLSQP is reliable at this size."""
    N = len(mu); f = lambda p: -(p @ (mu - r) - 0.5 * gamma * p @ Sigma @ p)
    g = lambda p: -(mu - r) + gamma * Sigma @ p
    cons = [] if leverage_cap is None else [{"type": "ineq", "fun": lambda p: leverage_cap - p.sum(), "jac": lambda p: -np.ones(N)}]
    res = minimize(f, np.full(N, 1.0 / (N + 1)), jac=g, bounds=[(0, None)] * N, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return res.x


def weights_table(lr, ann, r=RF_CONST, gamma=GAMMA, years=None):
    """Merton weights, kappa and leverage as sectors are added one at a time, for sample and Ledoit-Wolf covariance."""
    rows = []
    for k in range(1, 6):
        cols = list(range(k))
        for cov_name, shrink in (("sample", False), ("Ledoit-Wolf", True)):
            mu, S = moments(lr.iloc[:, cols], ann, shrink)
            pi_u = th.multi_fraction(mu, r, S, gamma)
            pi_l = long_only(mu, r, S, gamma); pi_ln = long_only(mu, r, S, gamma, leverage_cap=1.0)
            vol = np.sqrt(np.diag(S)); C = S / np.outer(vol, vol)
            rows.append(dict(N=k, assets="+".join(LABELS[:k]), cov=cov_name, cond=np.linalg.cond(S),
                             kappa_unc=th.multi_kappa(mu, r, S, gamma), kappa_long=th.multi_ce_growth(pi_l, mu, r, S, gamma),
                             kappa_longnolev=th.multi_ce_growth(pi_ln, mu, r, S, gamma),
                             gross_unc=np.abs(pi_u).sum(), net_unc=pi_u.sum(), **{f"w_unc_{LABELS[i]}": pi_u[i] for i in range(k)},
                             **{f"w_long_{LABELS[i]}": pi_l[i] for i in range(k)}, **{f"w_longnolev_{LABELS[i]}": pi_ln[i] for i in range(k)}))
    return pd.DataFrame(rows)


def capm_mu(lr, ann, r, premium=PRIOR_PREMIUM):
    """alternative drift: mu_i - r = beta_i * premium (beta from a regression on Nifty 50 returns), removing the
    large sampling error of sector means (s.e. of each mean is sigma_i/sqrt(T))."""
    x = lr.values; b = np.array([np.cov(x[:, i], x[:, 0])[0, 1] / np.var(x[:, 0], ddof=1) for i in range(x.shape[1])])
    return r + b * premium, b


def sensitivity(lr, ann, r=RF_CONST, gamma=GAMMA):
    """Sensitivity of the five-asset Merton weights to gamma, the drifts, the covariance and r."""
    mu_s, S = moments(lr, ann, shrink=True)
    mu_c, beta = capm_mu(lr, ann, r)
    base = th.multi_fraction(mu_c, r, S, gamma)
    rows = [dict(case="base (CAPM drift, LW cov)", **{LABELS[i]: base[i] for i in range(5)}, net=base.sum(), gross=np.abs(base).sum())]
    def add(name, mu, Sg, rr, g):
        """Append one sensitivity case to the table."""
        p = th.multi_fraction(mu, rr, Sg, g); rows.append(dict(case=name, **{LABELS[i]: p[i] for i in range(5)}, net=p.sum(), gross=np.abs(p).sum()))
    for g in (2.0, 5.0): add(f"gamma={g:g}", mu_c, S, r, g)
    add("mu - r up 20%", r + 1.2 * (mu_c - r), S, r, gamma); add("mu - r down 20%", r + 0.8 * (mu_c - r), S, r, gamma)
    add("sigma x1.2", mu_c, S * 1.44, r, gamma); add("sigma x0.8", mu_c, S * 0.64, r, gamma)
    add("r = 5% (mu fixed)", mu_c, S, 0.05, gamma)      # drifts held fixed, so the premium widens/narrows with r
    add("r = 8% (mu fixed)", mu_c, S, 0.08, gamma)
    add("sample drift, sample cov", *[moments(lr, ann, False)[0]], moments(lr, ann, False)[1], r, gamma)
    add("sample drift, LW cov", mu_s, S, r, gamma)
    return pd.DataFrame(rows), beta, mu_c


def correlation_experiment(rho_grid=(0.0, 0.5, 0.8, 0.95, 0.99), T_est=5.0, reps=40_000, gamma=GAMMA, r=R):
    """Two assets with identical sigma=0.2, mu=0.12, true correlation rho.  Estimate mu from T_est years (Sigma known)."""
    rows = []
    sig = 0.2
    for rho in rho_grid:
        S = sig ** 2 * np.array([[1, rho], [rho, 1]]); mu = np.array([0.12, 0.12])
        L = np.linalg.cholesky(S)
        mu_hat = mu + (L @ rng.standard_normal((2, reps))).T / np.sqrt(T_est)
        pi_hat = np.linalg.solve(S, (mu_hat - r).T).T / gamma
        pi_star = th.multi_fraction(mu, r, S, gamma)
        loss = np.array([th.multi_kappa(mu, r, S, gamma) - th.multi_ce_growth(p, mu, r, S, gamma) for p in pi_hat[:4000]])
        rows.append(dict(rho=rho, sd_pi1=pi_hat[:, 0].std(), sd_formula=1.0 / (gamma * sig * np.sqrt(T_est * (1 - rho ** 2))),
                         corr_pi1_pi2=np.corrcoef(pi_hat.T)[0, 1], mean_loss=loss.mean(), loss_formula=th.expected_loss_estimated_drift(gamma, T_est, 2),
                         pi_star_each=pi_star[0], cond=np.linalg.cond(S)))
    return pd.DataFrame(rows)


def oos_simulation(Sigma_true, mu_true, T_ests=(2, 5, 10), reps=1500, gamma=GAMMA, r=RF_CONST, ann=52):
    """Draw weekly log-returns from a known GBM, estimate, build the rule, evaluate the CE-growth loss under the truth."""
    N = len(mu_true); L = np.linalg.cholesky(Sigma_true)
    kap = th.multi_kappa(mu_true, r, Sigma_true, gamma)
    rows = []
    for Te in T_ests:
        n = int(Te * ann); acc = {k: [] for k in ("sample_unc", "LW_unc", "sample_long", "LW_long", "LW_long_nolev", "equal_1/N_cap")}
        for _ in range(reps):
            z = rng.standard_normal((n, N)) @ L.T / np.sqrt(ann)
            X = z + (mu_true - 0.5 * np.diag(Sigma_true)) / ann
            for cov, shrink in (("sample", False), ("LW", True)):
                S = (LedoitWolf().fit(X).covariance_ if shrink else np.cov(X.T, ddof=1)) * ann
                mu_h = X.mean(0) * ann + 0.5 * np.diag(S)
                acc[f"{cov}_unc"].append(kap - th.multi_ce_growth(th.multi_fraction(mu_h, r, S, gamma), mu_true, r, Sigma_true, gamma))
                p = long_only(mu_h, r, S, gamma); acc[f"{cov}_long"].append(kap - th.multi_ce_growth(p, mu_true, r, Sigma_true, gamma))
                if shrink:
                    p2 = long_only(mu_h, r, S, gamma, 1.0); acc["LW_long_nolev"].append(kap - th.multi_ce_growth(p2, mu_true, r, Sigma_true, gamma))
            pe = np.full(N, 0.5 / N); acc["equal_1/N_cap"].append(kap - th.multi_ce_growth(pe, mu_true, r, Sigma_true, gamma))
        rows.append(dict(T_est=Te, N=N, formula_unc=th.expected_loss_estimated_drift(gamma, Te, N),
                         **{k: float(np.mean(v)) for k, v in acc.items()}, **{k + "_median": float(np.median(v)) for k, v in acc.items() if k in ("sample_unc", "LW_unc")}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ F. rolling backtest on real data
def rule_weights_multi(win, rf, gamma=GAMMA):
    """Weights of each rule from a trailing window of daily log-returns (DataFrame), at risk-free rate rf."""
    yrs = (win.index[-1] - win.index[0]).days / 365.25
    ann = len(win) / yrs
    mu_s, S_s = moments(win, ann, False)
    mu_l, S_l = moments(win, ann, True)
    mu_c, _ = capm_mu(win, ann, rf)
    return {"sample, unconstrained": th.multi_fraction(mu_s, rf, S_s, gamma),
            "Ledoit-Wolf, unconstrained": th.multi_fraction(mu_l, rf, S_l, gamma),
            "LW, long-only no leverage": long_only(mu_l, rf, S_l, gamma, leverage_cap=1.0),
            "CAPM drifts + LW": th.multi_fraction(mu_c, rf, S_l, gamma),
            "Equal weight 5 x 20%": np.full(len(NAMES), 0.2)}


def multi_backtest(lr, rf, windows=(5, 10), eval_years=max(EST_WINDOWS), gamma=GAMMA):
    """Monthly rebalancing on the evaluation period used in Part 1 (first date + eval_years to the end).
    Between rebalancing dates positions drift; wealth is floored at 0 (ruin is absorbing)."""
    import part1_data_backtest as p1
    idx = lr.index
    me = pd.Series(np.arange(len(idx)), index=idx).resample("ME").last().dropna().astype(int).values
    start = int(me[me >= np.searchsorted(idx, idx[0] + pd.DateOffset(years=eval_years))][0])
    rebal = [int(i) for i in me if i >= start]
    R = np.expm1(lr.values)
    rfd = rf / TRADING_DAYS
    rules = {}
    for T_est in windows:
        for i in rebal:
            j0 = int(np.searchsorted(idx, idx[i] - pd.DateOffset(years=T_est), side="right"))
            for k, w in rule_weights_multi(lr.iloc[j0:i + 1], rf[i], gamma).items():
                rules.setdefault(k if "Equal" in k else f"{k} ({T_est}y)", {})[i] = w
    rules["Nifty 50 buy & hold"] = {start: np.eye(len(NAMES))[0]}
    rows, paths = [], {}
    for name, wmap in rules.items():
        W = np.ones(len(idx))
        hold = np.zeros(len(NAMES))
        cash, ruined, used = 0.0, False, []
        for t in range(start, len(idx)):
            if t > start:
                hold = hold * (1 + R[t])
                cash *= 1 + rfd[t]
                Wt = hold.sum() + cash
                if Wt <= 0:
                    Wt, ruined, hold, cash = 0.0, True, np.zeros(len(NAMES)), 0.0
                W[t] = Wt
            else:
                Wt = 1.0
            if t in wmap and not ruined:
                w = wmap[t]
                used.append(w)
                hold = w * Wt
                cash = (1 - w.sum()) * Wt
        Wp = W[start:]
        m = p1.metrics(Wp, idx[start:], float(rf[start:].mean()), gamma)
        u = np.array(used)
        m.update(rule=name, ruined=ruined, mean_gross=float(np.abs(u).sum(1).mean()), max_gross=float(np.abs(u).sum(1).max()))
        rows.append(m)
        paths[name] = Wp
    info = dict(start=str(idx[start].date()), end=str(idx[-1].date()), n_rebal=len(rebal), rf_eval=float(rf[start:].mean()))
    return pd.DataFrame(rows), paths, info, idx[start:]


def make_figure(wt, corr, oos, bt_paths=None, bt_idx=None):
    """Figure fig_multi.png: weights, correlation experiment, out-of-sample loss and (if given) the real-data backtest."""
    fig, ax = plt.subplots(1, 4 if bt_paths else 3, figsize=(16.5 if bt_paths else 12.5, 3.4))
    last = wt[wt.N == 5]; x = np.arange(5); wdt = 0.27
    for j, (cov, c) in enumerate((("sample", "tab:red"), ("Ledoit-Wolf", "tab:blue"))):
        row = last[last["cov"] == cov].iloc[0]
        ax[0].bar(x + (j - .5) * wdt, [row[f"w_unc_{l}"] for l in LABELS], wdt, color=c, label=f"unconstrained, {cov} cov.")
    row = last[last["cov"] == "Ledoit-Wolf"].iloc[0]
    ax[0].bar(x + 1.5 * wdt, [row[f"w_longnolev_{l}"] for l in LABELS], wdt, color="tab:green", label="long-only, no leverage (LW)")
    ax[0].axhline(0, c="k", lw=.5); ax[0].set_xticks(x); ax[0].set_xticklabels(LABELS, fontsize=8); ax[0].set_ylabel("fraction of wealth")
    ax[0].set_title("(a) 5-asset weights, sample-mean drifts", fontsize=9); ax[0].legend(fontsize=6.5)
    ax[1].semilogy(corr.rho, corr.sd_pi1, "ko", label="simulated sd$(\\hat\\pi_1)$"); rr = np.linspace(0, .99, 100)
    ax[1].semilogy(rr, 1.0 / (GAMMA * 0.2 * np.sqrt(5 * (1 - rr ** 2))), "k-", lw=.8, label="$1/(\\gamma\\sigma\\sqrt{T(1-\\rho^2)})$")
    ax[1].axhline(corr.mean_loss.mean(), c="tab:red", ls="--", label="mean CE loss (flat)")
    ax[1].set_xlabel("correlation $\\rho$"); ax[1].legend(fontsize=7); ax[1].set_title("(b) weights blow up, loss does not", fontsize=9)
    for col, mk, lab in (("sample_unc", "o-", "sample, unconstrained"), ("LW_unc", "s-", "Ledoit-Wolf, unconstr."), ("LW_long_nolev", "^-", "LW, long-only no-lev."), ("equal_1/N_cap", "k--", "fixed 10%/asset")):
        ax[2].loglog(oos.T_est, oos[col], mk, label=lab, ms=4)
    ax[2].set_xlabel("estimation window $T_{est}$ (years)"); ax[2].set_ylabel("mean loss in CE growth"); ax[2].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 0.1)); ax[2].set_title("(c) out-of-sample, known truth", fontsize=9)
    ax[2].set_xticks(list(oos.T_est)); ax[2].set_xticks([], minor=True); ax[2].set_xticklabels([f"{t:g}" for t in oos.T_est])
    if bt_paths:
        for name, c in (("Nifty 50 buy & hold", "k"), ("Equal weight 5 x 20%", "tab:gray"), ("CAPM drifts + LW (5y)", "tab:purple"),
                        ("LW, long-only no leverage (5y)", "tab:green"), ("Ledoit-Wolf, unconstrained (5y)", "tab:blue"),
                        ("sample, unconstrained (5y)", "tab:red")):
            if name in bt_paths:
                ax[3].semilogy(bt_idx, np.maximum(bt_paths[name], 1e-3), c=c, lw=.9, label=name.replace(" (5y)", ""))
        ax[3].set_ylim(bottom=0.05)
        ax[3].legend(fontsize=6)
        ax[3].set_title("(d) real-data backtest, 5-year windows", fontsize=9)
        ax[3].set_ylabel("wealth (log scale)")
        import matplotlib.dates as mdates                # one label every two years so the years do not overlap
        ax[3].xaxis.set_major_locator(mdates.YearLocator(2)); ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(FIGS / "fig_multi.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    lr, years, ann, data_label = load_weekly()
    r, r_src = risk_free_mean(lr.index)
    # A. consistency with the one-asset rule
    mu1, S1 = moments(lr.iloc[:, :1], ann)
    cons = dict(multi_N1=float(th.multi_fraction(mu1, r, S1, GAMMA)[0]), one_asset=float(th.merton_fraction(mu1[0], r, np.sqrt(S1[0, 0]), GAMMA)),
                kappa_multi_N1=float(th.multi_kappa(mu1, r, S1, GAMMA)), kappa_one=float(th.kappa(mu1[0], r, np.sqrt(S1[0, 0]), GAMMA)))
    wt = weights_table(lr, ann, r=r); wt.to_csv(RESULTS / "part4_weights.csv", index=False)
    sens, beta, mu_c = sensitivity(lr, ann, r=r); sens.to_csv(RESULTS / "part4_sensitivity.csv", index=False)
    mu_s, S_lw = moments(lr, ann, True); _, S_sm = moments(lr, ann, False)
    vol = np.sqrt(np.diag(S_sm)); corrm = S_sm / np.outer(vol, vol)
    desc = pd.DataFrame(dict(asset=LABELS, mu_hat=moments(lr, ann)[0], se_mu=vol / np.sqrt(years), sigma=vol, beta=beta, mu_capm=mu_c))
    desc.to_csv(RESULTS / "part4_descriptives.csv", index=False); pd.DataFrame(corrm, index=LABELS, columns=LABELS).to_csv(RESULTS / "part4_corr.csv")
    corr_exp = correlation_experiment(); corr_exp.to_csv(RESULTS / "part4_correlation_experiment.csv", index=False)
    # E. truth = LW covariance of the data and CAPM drifts (a plausible, well-conditioned 5-asset world)
    oos = oos_simulation(S_lw, mu_c, r=r); oos.to_csv(RESULTS / "part4_oos.csv", index=False)
    import part1_data_backtest as p1
    if data_label.startswith("NSE daily"):
        rf_d = np.asarray(p1.risk_free_daily(lr.index)[0])
        bt, bt_paths, bt_info, bt_idx = multi_backtest(lr, rf_d)
        bt.to_csv(RESULTS / "part4_backtest.csv", index=False)
    else:
        bt, bt_paths, bt_info, bt_idx = None, None, None, None
    make_figure(wt, corr_exp, oos, bt_paths, bt_idx)
    summ = dict(data=data_label, r=r, r_source=r_src, backtest=bt_info, n_obs=len(lr), years=years, ann_factor=ann, consistency=cons, max_corr_offdiag=float(np.max(corrm - np.eye(5))),
                se_mu_range=(float(desc.se_mu.min()), float(desc.se_mu.max())))
    (RESULTS / "part4_summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(summ); print(desc.round(4)); print(bt.round(4) if bt is not None else ''); print(pd.DataFrame(corrm, index=LABELS, columns=LABELS).round(2)); print(wt.round(3).T if False else wt[["N", "cov", "cond", "kappa_unc", "kappa_long", "kappa_longnolev", "gross_unc", "net_unc"]].round(4))
    print(wt[wt.N == 5].filter(regex="w_").round(3).T); print(sens.round(3)); print(corr_exp.round(4)); print(oos.round(5))
