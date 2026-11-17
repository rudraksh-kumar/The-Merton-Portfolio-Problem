"""Part 1: data, estimation and backtest.

Data
  * Nifty 50: NSE's official daily Total Returns Index (dividends reinvested), data/nse_tri_daily.csv, built from
    niftyindices.com by prep_nse_tri.py, used from 2007-09-17 to its last date.  If that file is absent the code
    falls back to the earlier proxy: the price index (data/nifty50_daily_github.csv, checked against NSE-official
    weekly closes from TapeTide) plus a constant dividend yield equal to NSE's average reported Nifty 50 yield.
    With the TRI present, the proxy is still computed on the common window to report how far off it was.
  * Risk-free rate: RBI 91-day T-bill auction cut-off yields, weekly (data/tbill_91d_auctions.csv, built from the
    RBI workbook by prep_rbi_tbill.py), carried forward to each trading day.  Without that file the constant
    RF_CONST (config.py) is used.  A constant-r sensitivity sweep is always run on the same period.
What it does
  1. estimate mu, sigma with standard errors; frequency (in)variance of the drift s.e.; normality and
     autocorrelation tests; Merton fraction with its s.e.
  2. rolling-window backtest (3,5,7,10 yr windows, monthly rebalancing, four rules) against buy-and-hold,
     T-bills and a 60/40 rebalanced portfolio, common evaluation period for all strategies
  3. compare the dispersion / loss of the estimated fraction with the formulas 1/(gamma sigma sqrt(T)) and 1/(2 gamma T)
Outputs: results/part1_*.csv, results/part1_summary.json, results/nifty_returns.csv (for Part 4), figures/fig_data.png
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *          # noqa
import theory as th


# ------------------------------------------------------------------ data
def load_nifty():
    """Returns (level series, q, check dict).  q is the constant dividend yield of the proxy, or None when the
    official total-return index is used (its log-returns already include dividends)."""
    d = pd.read_csv(DATA / "nifty50_daily_github.csv", parse_dates=["Date"]).set_index("Date")["Close"].astype(float)
    w = pd.read_csv(DATA / "tapetide_weekly_sector_indices.csv", parse_dates=["date"]).set_index("date")
    q = float(w["nifty50_div_yield_pct"].dropna().mean()) / 100.0
    tri_file = DATA / "nse_tri_daily.csv"
    if not tri_file.exists():
        chk = pd.concat([w["nifty50"], d.reindex(w.index)], axis=1, keys=["nse", "file"]).dropna()
        return d, q, dict(source="price index + constant yield (proxy)", n_check=len(chk),
                          max_rel_diff=float(((chk.nse / chk.file) - 1).abs().max()))
    tri = pd.read_csv(tri_file, parse_dates=["date"]).set_index("date")["nifty50"].dropna().sort_index()
    tri = tri[tri.index >= d.index[0]]
    # how good was the old proxy?  Annualised log growth on the common window
    c = d.index.intersection(tri.index); yrs = (c[-1] - c[0]).days / 365.25
    g_tri = np.log(tri[c[-1]] / tri[c[0]]) / yrs
    g_price = np.log(d[c[-1]] / d[c[0]]) / yrs
    return tri, None, dict(source="NSE Total Returns Index (official)", common_start=str(c[0].date()), common_end=str(c[-1].date()),
                           n_common=len(c), tri_growth=g_tri, price_growth=g_price, implied_div_yield=g_tri - g_price,
                           proxy_yield=q, proxy_minus_tri=(g_price + q) - g_tri)


def risk_free_daily(index):
    """annual risk-free rate aligned to the trading days: the latest RBI 91-day T-bill auction yield known on each
    day (weekly auctions, carried forward; no look-ahead), or the constant RF_CONST if the file is absent."""
    f = DATA / "tbill_91d_auctions.csv"
    if f.exists():
        s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["yield_pct"].sort_index() / 100.0
        assert s.index[0] <= index[0], "T-bill series must start before the equity data"
        return s.reindex(index, method="ffill").values, "RBI 91-day T-bill auction cut-off yield (weekly, data/tbill_91d_auctions.csv)"
    return np.full(len(index), RF_CONST), f"constant {RF_CONST:.3f}"


def build_returns(rf_const=None):
    """Daily log total returns, price-only log returns (NaN when the official TRI is used) and the aligned risk-free rate."""
    px, q, chk = load_nifty()
    days = px.index.to_series().diff().dt.days.fillna(1).values
    if q is None:                                        # official TRI: dividends already reinvested
        lr = np.log(px).diff().fillna(0.0).values
        lr_price = np.full(len(px), np.nan)
    else:                                                # proxy: price index + constant dividend yield
        lr_price = np.log(px).diff().fillna(0.0).values
        lr = lr_price + q * days / 365.25
    rf, rf_label = risk_free_daily(px.index)
    if rf_const is not None:
        rf = np.full(len(px), rf_const)
    df = pd.DataFrame({"px": px.values, "lr": lr, "lr_price": lr_price, "days": days, "rf": rf}, index=px.index)
    df.iloc[0, df.columns.get_loc("lr")] = 0.0
    return df, q, chk, rf_label


# ------------------------------------------------------------------ estimation
def estimate(lr, years, rf, gamma=GAMMA):
    """lr: log returns over `years` calendar years.  Returns annualised estimates."""
    m = lr.sum() / years                                   # annualised mean log-return = ln(P_T/P_0)/years
    n = len(lr); ppy = n / years
    s2 = lr.var(ddof=1) * ppy
    mu = m + 0.5 * s2
    se_mu = np.sqrt(s2 / years)                            # sigma / sqrt(T), independent of sampling frequency
    pi = (mu - rf) / (gamma * s2)
    return dict(m=m, sigma=np.sqrt(s2), mu=mu, se_mu=se_mu, pi=pi, se_pi=se_mu / (gamma * s2))


def ljung_box(x, lags=10):
    """Ljung-Box statistic, p-value and autocorrelations of x up to `lags` (applied to squared returns for volatility clustering)."""
    x = np.asarray(x) - np.mean(x); n = len(x)
    ac = np.array([np.sum(x[k:] * x[:n - k]) / np.sum(x * x) for k in range(1, lags + 1)])
    Q = n * (n + 2) * np.sum(ac ** 2 / (n - np.arange(1, lags + 1)))
    return Q, 1 - stats.chi2.cdf(Q, lags), ac


def estimation_study(df):
    """Full-sample estimates of mu, sigma and pi with standard errors, normality and Ljung-Box tests, daily/weekly/monthly frequency check."""
    years = (df.index[-1] - df.index[0]).days / 365.25
    lr = df.lr.iloc[1:]
    rf = float(df.rf.mean())
    out = {"years": years, "n_days": len(lr), "rf_mean": rf}
    out["full"] = estimate(lr.values, years, rf)
    if df.lr_price.notna().any():
        out["full_price_only"] = estimate(df.lr_price.iloc[1:].values, years, rf)
    # frequency invariance: daily / weekly / monthly returns from the same price series
    rows = []
    lv = df.lr.cumsum()
    for name, rule in (("daily", None), ("weekly", "W-FRI"), ("monthly", "ME")):
        x = lv if rule is None else lv.resample(rule).last().dropna()
        r_ = x.diff().dropna()
        e = estimate(r_.values, years, rf)
        jb = stats.jarque_bera(r_.values)
        rows.append(dict(freq=name, n=len(r_), m=e["m"], sigma=e["sigma"], mu=e["mu"], se_mu=e["se_mu"],
                         skew=stats.skew(r_), exkurt=stats.kurtosis(r_), JB=jb.statistic, JB_p=jb.pvalue))
    freq = pd.DataFrame(rows)
    # autocorrelation
    Q_r, p_r, ac_r = ljung_box(lr.values, 10)
    Q_r2, p_r2, ac_r2 = ljung_box(lr.values ** 2, 10)
    out["ljung_box_returns"] = dict(Q10=Q_r, p=p_r, ac1=ac_r[0])
    out["ljung_box_squared"] = dict(Q10=Q_r2, p=p_r2, ac1=ac_r2[0], ac5=ac_r2[4])
    # Merton fraction for several gamma and r
    rows = []
    for g in GAMMA_GRID:
        for rr in RF_SENSITIVITY:
            e = estimate(lr.values, years, rr, g)
            rows.append(dict(gamma=g, r=rr, pi_hat=e["pi"], se_pi=e["se_pi"], mu_hat=e["mu"], sigma_hat=e["sigma"]))
    merton = pd.DataFrame(rows)
    # volatility regimes: rolling 1y annualised vol
    vol1y = lr.rolling(252).std() * np.sqrt(252)
    out["vol1y_min_max"] = (float(vol1y.min()), float(vol1y.max()))
    return out, freq, merton


# ------------------------------------------------------------------ rules
def posterior_premium(mu_hat, rf, s2, T_est, m0=PRIOR_PREMIUM, s0=PRIOR_SD):
    """Normal prior N(m0, s0^2) on mu-r, data precision T/sigma^2 -> posterior mean (James-Stein-type shrinkage)."""
    w_data = (T_est / s2) / (T_est / s2 + 1 / s0 ** 2)
    return w_data * (mu_hat - rf) + (1 - w_data) * m0, w_data


def rule_weights(lr_win, rf, T_est, gamma):
    """Stock fraction for each rule (raw, capped to [0,1], Bayesian-shrunk, shrunk and capped) from one estimation window."""
    e = estimate(lr_win, T_est, rf, gamma)
    s2 = e["sigma"] ** 2
    raw = e["pi"]
    post, wd = posterior_premium(e["mu"], rf, s2, T_est)
    sh = post / (gamma * s2)
    return dict(raw=raw, capped=float(np.clip(raw, 0, 1)), shrunk=sh, shrunk_capped=float(np.clip(sh, 0, 1)),
                mu_hat=e["mu"], sigma_hat=e["sigma"], w_data=wd)


# ------------------------------------------------------------------ backtest engine
def simulate(df, weight_fn, start_idx, rebal_idx):
    """Daily wealth path.  weight_fn(i) -> fraction in the index chosen at the close of day i (rebalancing days only).
    Between rebalancing days positions drift.  Wealth is floored at 0 (ruin is absorbing)."""
    R = np.expm1(df.lr.values); rf_d = df.rf.values / TRADING_DAYS
    W = np.ones(len(df)); stock = cash = 0.0; ruined = False; weights = {}
    rebal = set(rebal_idx)
    for i in range(start_idx, len(df)):
        if i == start_idx:
            Wt = 1.0
        else:
            stock *= 1 + R[i]; cash *= 1 + rf_d[i]; Wt = stock + cash
            if Wt <= 0:
                Wt = 0.0; ruined = True
            W[i] = Wt
        if i in rebal and not ruined:
            p = weight_fn(i); weights[i] = p
            stock, cash = p * Wt, (1 - p) * Wt
        elif i == start_idx:
            W[i] = 1.0
        if i == start_idx:
            W[i] = 1.0
    return W[start_idx:], weights, ruined


def metrics(W, idx, rf_ann, gamma=GAMMA):
    """CAGR, volatility, Sharpe ratio, maximum drawdown and realised certainty-equivalent return of a wealth path."""
    s = pd.Series(W, index=idx)
    r_d = s.pct_change().dropna()
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if s.iloc[-1] > 0 else -1.0
    vol = r_d.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r_d.mean() * TRADING_DAYS - rf_ann) / vol if vol > 0 else np.nan
    mdd = (s / s.cummax() - 1).min()
    m = s.resample("ME").last().pct_change().dropna()
    gross = 1 + m.values
    if (gross <= 0).any():
        ce = -1.0
    else:
        ce_m = np.mean(gross ** (1 - gamma)) ** (1 / (1 - gamma)); ce = ce_m ** 12 - 1
    return dict(CAGR=cagr, vol=vol, Sharpe=sharpe, maxDD=mdd, CE_return=ce, terminal=s.iloc[-1])


def backtest(df, gamma=GAMMA, windows=EST_WINDOWS, eval_years=max(EST_WINDOWS)):
    """eval_years fixes the start of the evaluation period (first date + eval_years) so that runs with
    different window sets (e.g. the r-sensitivity) are evaluated on the SAME period as the main backtest."""
    idx = df.index
    eval_start_date = idx[0] + pd.DateOffset(years=max(eval_years, max(windows)))
    me = pd.Series(np.arange(len(idx)), index=idx).resample("ME").last().dropna().astype(int).values  # month-end positions
    start_idx = int(me[me >= np.searchsorted(idx, eval_start_date)][0])
    rebal = [int(i) for i in me if i >= start_idx]
    lr = df.lr.values
    rf_ann = float(df.rf.iloc[start_idx:].mean())
    results, weights_store, paths = [], {}, {}
    for T_est in windows:
        n0 = {}
        # precompute rule weights at each rebalancing date
        W_cache = {}
        for i in rebal:
            lo_date = idx[i] - pd.DateOffset(years=T_est)
            j0 = int(np.searchsorted(idx, lo_date, side="right"))
            win = lr[j0 + 1:i + 1]
            yrs = (idx[i] - idx[j0]).days / 365.25
            W_cache[i] = rule_weights(win, df.rf.iloc[i], yrs, gamma)
        for rule in ("raw", "capped", "shrunk", "shrunk_capped"):
            W, wts, ruined = simulate(df, lambda i, rule=rule: W_cache[i][rule], start_idx, rebal)
            m = metrics(W, idx[start_idx:], rf_ann, gamma)
            m.update(window=T_est, rule=rule, ruined=ruined, mean_pi=np.mean(list(wts.values())),
                     sd_pi=np.std(list(wts.values())), min_pi=np.min(list(wts.values())), max_pi=np.max(list(wts.values())))
            results.append(m); paths[(T_est, rule)] = W
        weights_store[T_est] = pd.DataFrame(W_cache).T.set_index(idx[list(W_cache.keys())])
    # benchmarks
    bench = {"Buy & hold Nifty 50 (TR)": lambda i: 1.0, "T-bills": lambda i: 0.0, "60/40 (monthly rebal.)": lambda i: 0.6}
    for name, f in bench.items():
        rb = rebal if "60/40" in name else [start_idx]
        W, _, ruined = simulate(df, f, start_idx, rb)
        m = metrics(W, idx[start_idx:], rf_ann, gamma); m.update(window=np.nan, rule=name, ruined=ruined,
                                                                  mean_pi=f(0), sd_pi=0.0, min_pi=f(0), max_pi=f(0))
        results.append(m); paths[(0, name)] = W
    # hindsight benchmark: full-evaluation-period Merton fraction (uses future information; reference only)
    e = estimate(lr[start_idx + 1:], (idx[-1] - idx[start_idx]).days / 365.25, rf_ann, gamma)
    pi_hind = float(np.clip(e["pi"], -5, 5))
    W, _, ruined = simulate(df, lambda i: pi_hind, start_idx, rebal)
    m = metrics(W, idx[start_idx:], rf_ann, gamma); m.update(window=np.nan, rule=f"Hindsight Merton (pi={pi_hind:.2f})", ruined=ruined,
                                                              mean_pi=pi_hind, sd_pi=0.0, min_pi=pi_hind, max_pi=pi_hind)
    results.append(m); paths[(0, "hindsight")] = W
    return pd.DataFrame(results), weights_store, paths, dict(start=str(idx[start_idx].date()), end=str(idx[-1].date()),
                                                             n_rebal=len(rebal), rf_eval=rf_ann, pi_hindsight=pi_hind)


def loss_formula_check(weights_store, df, start_info, gamma=GAMMA):
    """Observed dispersion of the raw fraction vs the sampling-theory prediction sd = 1/(gamma sigma sqrt(T_est)),
    and the empirical loss 0.5 gamma sigma^2 mean((pi_t - pi_bar)^2) vs 1/(2 gamma T_est)."""
    full = estimate(df.lr.iloc[1:].values, (df.index[-1] - df.index[0]).days / 365.25, float(df.rf.mean()), gamma)
    rows = []
    for T_est, W in weights_store.items():
        s = W.sigma_hat.mean()
        rows.append(dict(T_est=T_est, sd_obs_raw=W.raw.std(), sd_theory=th.sd_of_estimated_fraction(s, gamma, T_est),
                         loss_obs=0.5 * gamma * s ** 2 * np.mean((W.raw - full["pi"]) ** 2),
                         loss_theory=th.expected_loss_estimated_drift(gamma, T_est),
                         sd_obs_shrunk=W.shrunk.std(), mean_weight_on_data=W.w_data.mean(),
                         frac_raw_outside_01=float(((W.raw < 0) | (W.raw > 1)).mean())))
    return pd.DataFrame(rows), full


def block_check(df, gamma=GAMMA, blocks=(1, 2, 3, 5)):
    """Non-overlapping T-year blocks of the real data: sd of the block estimates pi_hat vs 1/(gamma sigma sqrt(T)),
    and the mean empirical loss 0.5 gamma sigma^2 (pi_hat - pi_bar)^2 vs 1/(2 gamma T) (pi_bar = full-sample estimate)."""
    rf = float(df.rf.mean()); full = estimate(df.lr.iloc[1:].values, (df.index[-1] - df.index[0]).days / 365.25, rf, gamma)
    rows = []
    for Tb in blocks:
        edges = [df.index[0] + pd.DateOffset(years=Tb * k) for k in range(0, 40)]
        edges = [e for e in edges if e <= df.index[-1]]
        pis, sigs = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            x = df.loc[(df.index > a) & (df.index <= b), "lr"].values
            e = estimate(x, Tb, rf, gamma); pis.append(e["pi"]); sigs.append(e["sigma"])
        pis, sig = np.array(pis), float(np.sqrt(np.mean(np.square(sigs))))
        rows.append(dict(T_block=Tb, n_blocks=len(pis), sd_obs=pis.std(ddof=1), sd_theory=th.sd_of_estimated_fraction(sig, gamma, Tb),
                         loss_obs=0.5 * gamma * sig ** 2 * np.mean((pis - full["pi"]) ** 2), loss_theory=th.expected_loss_estimated_drift(gamma, Tb),
                         frac_outside_01=float(((pis < 0) | (pis > 1)).mean())))
    return pd.DataFrame(rows)


def make_figure(df, weights_store, paths, bt):
    """Figure fig_data.png: rolling estimated pi, backtest wealth paths and the Nifty 50 total-return index."""
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    w5 = weights_store[5]
    ax[0].plot(w5.index, w5.raw, lw=.9, label="raw $\\hat\\pi_t$ (5y window)")
    ax[0].plot(w5.index, w5.shrunk, lw=.9, label="shrunk")
    sd = th.sd_of_estimated_fraction(w5.sigma_hat.mean(), GAMMA, 5)
    ax[0].axhspan(0, 1, color="grey", alpha=.12); ax[0].axhline(0, c="k", lw=.4); ax[0].axhline(1, c="k", lw=.4)
    ax[0].set_ylim(-3, 4); ax[0].legend(fontsize=7); ax[0].set_ylabel("stock fraction")
    ax[0].set_title(f"(a) estimated fraction ($\\gamma={GAMMA:g}$); sampling sd $\\approx${sd:.2f}", fontsize=9)
    sel = {(0, "Buy & hold Nifty 50 (TR)"): "Buy & hold", (0, "T-bills"): "T-bills", (0, "60/40 (monthly rebal.)"): "60/40",
           (5, "raw"): "Merton raw (5y)", (5, "shrunk_capped"): "Merton shrunk+cap (5y)", (10, "shrunk_capped"): "Merton shrunk+cap (10y)"}
    idx = df.index[len(df) - len(next(iter(paths.values()))):]
    for k, lab in sel.items():
        ax[1].semilogy(idx, paths[k], lw=1, label=lab)
    ax[1].legend(fontsize=6.5); ax[1].set_title("(b) wealth, common evaluation period", fontsize=9)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")      # 1, 2, 3 instead of 10^0, 2x10^0
    ax[1].yaxis.set_major_formatter(plain); ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].plot(df.index, df.px, lw=.8); ax[2].set_yscale("log"); ax[2].set_title("(c) Nifty 50 total-return index (NSE), log scale" if df.lr_price.isna().all() else "(c) Nifty 50 (price index, log scale)", fontsize=9)
    import matplotlib.dates as mdates
    for a in ax[:2]:                                     # one label every two years so the years do not overlap
        a.xaxis.set_major_locator(mdates.YearLocator(2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(FIGS / "fig_data.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    df, q, chk, rf_label = build_returns()
    est, freq, merton = estimation_study(df)
    bt, wstore, paths, binfo = backtest(df)
    lossdf, full = loss_formula_check(wstore, df, binfo)
    blk = block_check(df); blk.to_csv(RESULTS / "part1_block_check.csv", index=False)
    # sensitivity of the headline strategies to r (constant-r runs)
    sens_rows = []
    for rr in RF_SENSITIVITY:
        d2, *_ = build_returns(rf_const=rr)
        b2, _, _, i2 = backtest(d2)            # same windows and same evaluation period as the main backtest
        assert i2["start"] == binfo["start"]
        for _, row in b2.iterrows():
            sens_rows.append(dict(r=rr, window=row.window, rule=row.rule, CAGR=row.CAGR, CE_return=row.CE_return, Sharpe=row.Sharpe))
    sens = pd.DataFrame(sens_rows)
    freq.to_csv(RESULTS / "part1_frequency.csv", index=False); merton.to_csv(RESULTS / "part1_merton_full_sample.csv", index=False)
    bt.to_csv(RESULTS / "part1_backtest.csv", index=False); lossdf.to_csv(RESULTS / "part1_loss_check.csv", index=False)
    sens.to_csv(RESULTS / "part1_r_sensitivity.csv", index=False)
    for T_est, W in wstore.items():
        W.to_csv(RESULTS / f"part1_weights_{T_est}y.csv")
    df[["px", "lr", "rf"]].to_csv(RESULTS / "nifty_returns.csv")
    summ = dict(q_dividend_yield=q, data_source=chk.get("source"), data_check=chk, rf_source=rf_label, estimation=est, full_sample_loss_ref=full, backtest=binfo)
    (RESULTS / "part1_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    make_figure(df, wstore, paths, bt)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(json.dumps(summ, indent=1, default=float)); print(freq.round(4)); print(merton.round(3)); print(bt.round(4)); print(lossdf.round(4)); print(blk.round(4))
