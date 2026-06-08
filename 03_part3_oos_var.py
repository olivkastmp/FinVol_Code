# ============================================================
# Part III: Out-of-Sample Forecast Evaluation & VaR Backtesting
# All 6 models: sGARCH-norm, sGARCH-t, EGARCH-norm, EGARCH-t,
#               GJR-norm, GJR-t
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# ============================================================
# Section 2: Load data
# ============================================================

BTC_clean = pd.read_csv("output/BTC_clean.csv", parse_dates=["Date"])
ETH_clean = pd.read_csv("output/ETH_clean.csv", parse_dates=["Date"])

# ARMA orders from Part I
p_btc, q_btc = 0, 0
p_eth, q_eth = 0, 0

os.makedirs("output", exist_ok=True)
log = open("output/04_Output_Part3.txt", "w")


def out(text=""):
    print(text)
    log.write(text + "\n")


out("=" * 56)
out("  Part III: Forecast Evaluation & VaR Backtesting")
out("  All 6 models — Financial Volatility: BTC & ETH")
out("=" * 56)
out()

# ============================================================
# Section 5: OOS window
# ============================================================

n_oos   = 300
n_total = len(BTC_clean)
n_is    = n_total - n_oos

out(f"Total observations   : {n_total}")
out(f"In-sample window     : {n_is}")
out(f"Out-of-sample window : {n_oos}")
out()

# ============================================================
# Helper: fit one GARCH-type model
# ============================================================

def fit_garch_full(series, p_arma, q_arma, vol="GARCH", dist="normal"):
    if q_arma > 0:
        arma_fit  = ARIMA(series, order=(p_arma, 0, q_arma)).fit()
        data      = arma_fit.resid
        mean_spec = "Constant"
        lags      = 0
    else:
        data      = series
        mean_spec = "ARX"
        lags      = list(range(1, p_arma + 1)) if p_arma > 0 else 0
    am = arch_model(data, mean=mean_spec, lags=lags, vol=vol, p=1, q=1, dist=dist)
    return am.fit(disp="off", show_warning=False)


# ============================================================
# Section 6: Rolling-window forecasts
# Expanding window, 1-step ahead, refit every observation.
# This can take several minutes.
# ============================================================

def rolling_forecast(series, p_arma, q_arma, vol, dist, n_is, n_oos):
    """
    Expanding-window 1-step-ahead forecasts.
    Returns arrays: mu_hat, sigma_hat, r_actual.
    """
    mu_hat    = np.zeros(n_oos)
    sigma_hat = np.zeros(n_oos)
    r_actual  = series[n_is:]

    for j in range(n_oos):
        window = series[:n_is + j]
        try:
            fit = fit_garch_full(window, p_arma, q_arma, vol, dist)
            fc  = fit.forecast(horizon=1, reindex=False)
            mu_hat[j]    = fc.mean.iloc[-1, 0]
            sigma_hat[j] = np.sqrt(fc.variance.iloc[-1, 0])
        except Exception:
            mu_hat[j]    = np.nan
            sigma_hat[j] = np.nan

        if (j + 1) % 50 == 0:
            print(f"  {vol}/{dist}: {j+1}/{n_oos} done")

    return pd.DataFrame({
        "mu_hat":    mu_hat,
        "sigma_hat": sigma_hat,
        "var_hat":   sigma_hat ** 2,
        "r_actual":  r_actual,
        "proxy":     r_actual ** 2,
    })


r_btc = BTC_clean["Return"].values
r_eth = ETH_clean["Return"].values

model_specs = [
    ("sGARCH", "normal", "GARCH",     "normal"),
    ("sGARCH", "t",      "GARCH",     "t"),
    ("EGARCH", "normal", "EGARCH",    "normal"),
    ("EGARCH", "t",      "EGARCH",    "t"),
    ("GJR",    "normal", "GJR-GARCH", "normal"),
    ("GJR",    "t",      "GJR-GARCH", "t"),
]

model_labels = ["sGARCH-Normal", "sGARCH-t",
                "EGARCH-Normal", "EGARCH-t",
                "GJR-Normal",    "GJR-t"]

out("--- Running rolling-window forecasts (this will take several minutes) ---")
out()

fc_btc, fc_eth = {}, {}
for label, _, vol, dist in model_specs:
    key = f"{label}_{dist}"
    print(f"BTC {key}...")
    fc_btc[key] = rolling_forecast(r_btc, p_btc, q_btc, vol, dist, n_is, n_oos)
    print(f"ETH {key}...")
    fc_eth[key] = rolling_forecast(r_eth, p_eth, q_eth, vol, dist, n_is, n_oos)

out("Rolling forecasts done.")
out()

oos_dates_btc = BTC_clean["Date"].values[-n_oos:]
oos_dates_eth = ETH_clean["Date"].values[-n_oos:]

# ============================================================
# Section 8: QLIKE Loss Function
# ============================================================

def qlike(proxy, var_hat):
    var_hat = np.maximum(var_hat, 1e-10)
    return proxy / var_hat - np.log(proxy / var_hat) - 1


keys = ["sGARCH_normal", "sGARCH_t", "EGARCH_normal", "EGARCH_t", "GJR_normal", "GJR_t"]

ql_btc = {k: qlike(fc_btc[k]["proxy"].values, fc_btc[k]["var_hat"].values) for k in keys}
ql_eth = {k: qlike(fc_eth[k]["proxy"].values, fc_eth[k]["var_hat"].values) for k in keys}

out("=== Average QLIKE Loss -- BTC ===")
out("(Lower = better volatility forecasts)")
for label, k in zip(model_labels, keys):
    out(f"  {label:<20}  {np.nanmean(ql_btc[k]):.6f}")
out()
out("=== Average QLIKE Loss -- ETH ===")
for label, k in zip(model_labels, keys):
    out(f"  {label:<20}  {np.nanmean(ql_eth[k]):.6f}")
out()

# ============================================================
# Section 9: DMW Pairwise Tests — QLIKE
# Newey-West HAC variance for the loss differential
# ============================================================

def newey_west_var(x, max_lag=None):
    """Simple Newey-West long-run variance estimator."""
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    x = x - x.mean()
    var = np.dot(x, x) / n
    for lag in range(1, max_lag + 1):
        w    = 1 - lag / (max_lag + 1)
        cov  = np.dot(x[lag:], x[:-lag]) / n
        var += 2 * w * cov
    return var


def dmw_test(loss_a, loss_b, label_a, label_b):
    d    = loss_a - loss_b
    d    = d[np.isfinite(d)]
    T    = len(d)
    se   = np.sqrt(newey_west_var(d) / T)
    dmw  = d.mean() / se if se > 0 else np.nan
    pval = 2 * stats.norm.sf(abs(dmw)) if np.isfinite(dmw) else np.nan
    sig  = "SIGNIFICANT" if pval < 0.05 else "not significant"
    out(f"  {label_a:<20} vs {label_b:<20}  |  DMW = {dmw:>7.3f}  |  p = {pval:.4f}  |  {sig}")


pairs = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
]

out("=== DMW Pairwise Tests -- BTC ===")
for i, j in pairs:
    dmw_test(ql_btc[keys[i]], ql_btc[keys[j]], model_labels[i], model_labels[j])
out()

out("=== DMW Pairwise Tests -- ETH ===")
for i, j in pairs:
    dmw_test(ql_eth[keys[i]], ql_eth[keys[j]], model_labels[i], model_labels[j])
out()

# ============================================================
# Section 10b: RMSE
# ============================================================

def rmse(proxy, var_hat):
    return np.sqrt(np.nanmean((proxy - var_hat) ** 2))


def rmse_table(fc_dict, keys, model_labels):
    rows = [{"Model": lab, "RMSE": rmse(fc_dict[k]["proxy"].values, fc_dict[k]["var_hat"].values)}
            for lab, k in zip(model_labels, keys)]
    df = pd.DataFrame(rows)
    df["Rank"] = df["RMSE"].rank().astype(int)
    df["RMSE"] = df["RMSE"].round(8)
    return df.sort_values("Rank")


out("=== RMSE Loss -- BTC ===")
out(rmse_table(fc_btc, keys, model_labels).to_string(index=False))
out()
out("=== RMSE Loss -- ETH ===")
out(rmse_table(fc_eth, keys, model_labels).to_string(index=False))
out()

# DMW on squared error loss
def dmw_rmse(fc_a, fc_b, label_a, label_b):
    d = (fc_a["proxy"].values - fc_a["var_hat"].values)**2 - \
        (fc_b["proxy"].values - fc_b["var_hat"].values)**2
    d = d[np.isfinite(d)]
    T = len(d)
    se = np.sqrt(newey_west_var(d) / T)
    dmw_stat = d.mean() / se if se > 0 else np.nan
    pval     = 2 * stats.norm.sf(abs(dmw_stat)) if np.isfinite(dmw_stat) else np.nan
    sig      = "SIGNIFICANT" if pval < 0.05 else "not significant"
    out(f"  {label_a:<20} vs {label_b:<20}  |  DMW = {dmw_stat:>7.3f}  |  p = {pval:.4f}  |  {sig}")


out("=== DMW Tests on Squared Error Loss -- BTC ===")
for i, j in pairs:
    dmw_rmse(fc_btc[keys[i]], fc_btc[keys[j]], model_labels[i], model_labels[j])
out()
out("=== DMW Tests on Squared Error Loss -- ETH ===")
for i, j in pairs:
    dmw_rmse(fc_eth[keys[i]], fc_eth[keys[j]], model_labels[i], model_labels[j])
out()

# ============================================================
# Section 11: VaR Computation — all 6 models
# ============================================================

alphas       = [0.01, 0.025, 0.05]
alpha_labels = ["1", "2.5", "5"]


def compute_var(fc_df, dist="norm", nu=None):
    """
    Compute VaR at each confidence level.
    dist: 'norm' or 'std' (Student-t)
    nu  : degrees of freedom for Student-t
    """
    var_mat = {}
    for a, al in zip(alphas, alpha_labels):
        if dist == "norm":
            q = stats.norm.ppf(a)
        else:
            q = stats.t.ppf(a, df=nu)
        var_mat[f"VaR_{al}pct"] = fc_df["mu_hat"].values + fc_df["sigma_hat"].values * q
    return pd.DataFrame(var_mat)


# Fit full in-sample models to extract Student-t degrees of freedom
fit_btc_t_is    = fit_garch_full(r_btc, p_btc, q_btc, "GARCH",     "t")
fit_btc_et_is   = fit_garch_full(r_btc, p_btc, q_btc, "EGARCH",    "t")
fit_btc_gjrt_is = fit_garch_full(r_btc, p_btc, q_btc, "GJR-GARCH", "t")
fit_eth_t_is    = fit_garch_full(r_eth, p_eth, q_eth, "GARCH",     "t")
fit_eth_et_is   = fit_garch_full(r_eth, p_eth, q_eth, "EGARCH",    "t")
fit_eth_gjrt_is = fit_garch_full(r_eth, p_eth, q_eth, "GJR-GARCH", "t")

def get_nu(fit):
    for key in ["nu", "eta", "shape"]:
        if key in fit.params.index:
            return float(fit.params[key])
    return None

nu_btc_sgarch = get_nu(fit_btc_t_is)
nu_btc_egarch = get_nu(fit_btc_et_is)
nu_btc_gjr    = get_nu(fit_btc_gjrt_is)
nu_eth_sgarch = get_nu(fit_eth_t_is)
nu_eth_egarch = get_nu(fit_eth_et_is)
nu_eth_gjr    = get_nu(fit_eth_gjrt_is)

var_btc = {
    "sGARCH_normal": compute_var(fc_btc["sGARCH_normal"], "norm"),
    "sGARCH_t":      compute_var(fc_btc["sGARCH_t"],      "std", nu_btc_sgarch),
    "EGARCH_normal": compute_var(fc_btc["EGARCH_normal"], "norm"),
    "EGARCH_t":      compute_var(fc_btc["EGARCH_t"],      "std", nu_btc_egarch),
    "GJR_normal":    compute_var(fc_btc["GJR_normal"],    "norm"),
    "GJR_t":         compute_var(fc_btc["GJR_t"],         "std", nu_btc_gjr),
}

var_eth = {
    "sGARCH_normal": compute_var(fc_eth["sGARCH_normal"], "norm"),
    "sGARCH_t":      compute_var(fc_eth["sGARCH_t"],      "std", nu_eth_sgarch),
    "EGARCH_normal": compute_var(fc_eth["EGARCH_normal"], "norm"),
    "EGARCH_t":      compute_var(fc_eth["EGARCH_t"],      "std", nu_eth_egarch),
    "GJR_normal":    compute_var(fc_eth["GJR_normal"],    "norm"),
    "GJR_t":         compute_var(fc_eth["GJR_t"],         "std", nu_eth_gjr),
}

# ============================================================
# Section 12: Benchmarks — Historical Simulation & RiskMetrics
# ============================================================

def compute_benchmarks(returns, n_is, n_oos):
    all_ret   = np.asarray(returns, dtype=float)
    T         = len(all_ret)
    var_hist  = np.full((n_oos, 3), np.nan)
    var_rm    = np.full((n_oos, 3), np.nan)
    lam       = 0.94
    sigma2    = np.zeros(T)
    sigma2[0] = np.var(all_ret[:n_is])
    for t in range(1, T):
        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * all_ret[t-1]**2
    for j in range(n_oos):
        t_idx    = n_is + j
        window   = all_ret[max(0, t_idx - 250): t_idx]
        for i, a in enumerate(alphas):
            var_hist[j, i] = np.quantile(window, a)
            var_rm[j, i]   = stats.norm.ppf(a) * np.sqrt(sigma2[t_idx])
    cols = [f"VaR_{al}pct" for al in alpha_labels]
    return pd.DataFrame(var_hist, columns=cols), pd.DataFrame(var_rm, columns=cols)


var_btc_hist, var_btc_rm = compute_benchmarks(r_btc, n_is, n_oos)
var_eth_hist, var_eth_rm = compute_benchmarks(r_eth, n_is, n_oos)
out("VaR computation complete for all models and benchmarks.")
out()

# ============================================================
# Section 13: VaR Backtesting — Kupiec (UC) + Christoffersen (CC)
# ============================================================

def kupiec_test(violations, n, alpha):
    """Kupiec unconditional coverage test. H0: correct coverage."""
    x   = violations
    pi  = x / n
    if pi == 0 or pi == 1:
        return np.nan, np.nan
    lr = -2 * (x * np.log(alpha / pi) + (n - x) * np.log((1 - alpha) / (1 - pi)))
    pval = stats.chi2.sf(lr, df=1)
    return lr, pval


def cc_test(hits, alpha):
    """Christoffersen conditional coverage test."""
    n    = len(hits)
    n00  = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01  = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10  = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11  = np.sum((hits[:-1] == 1) & (hits[1:] == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi   = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi == 0 or pi == 1 or pi01 == 0 or pi01 == 1 or pi11 == 0 or pi11 == 1:
        return np.nan, np.nan
    lr = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
        - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    pval = stats.chi2.sf(lr, df=2)
    return lr, pval


def run_backtest(var_vec, r_actual, alpha, model_label):
    if np.any(np.isnan(var_vec)):
        out(f"  {model_label:<25} | alpha={alpha*100:4.1f}% | SKIPPED (NA in VaR)")
        return
    hits     = (r_actual < var_vec).astype(int)
    n_viol   = hits.sum()
    expected = alpha * len(r_actual)
    uc_lr, uc_p = kupiec_test(n_viol, len(r_actual), alpha)
    cc_lr, cc_p = cc_test(hits, alpha)
    out(f"  {model_label:<25} | alpha={alpha*100:4.1f}% | "
        f"Violations: {n_viol:3d} (expected {expected:.1f}) | "
        f"Kupiec p={uc_p:.4f} | CC p={cc_p:.4f}")


r_btc_oos = r_btc[-n_oos:]
r_eth_oos = r_eth[-n_oos:]

all_var_btc = {**var_btc, "HistSim": var_btc_hist, "RiskMetrics": var_btc_rm}
all_var_eth = {**var_eth, "HistSim": var_eth_hist, "RiskMetrics": var_eth_rm}
all_model_names = model_labels + ["Hist Sim (250d)", "RiskMetrics"]
all_keys        = keys + ["HistSim", "RiskMetrics"]

out("=== VaR Backtesting -- BTC ===")
out("(p > 0.05 = model passes — cannot reject correct coverage)")
out()
for a, al in zip(alphas, alpha_labels):
    out(f"-- alpha = {a*100:.1f}% --")
    col = f"VaR_{al}pct"
    for name, k in zip(all_model_names, all_keys):
        run_backtest(all_var_btc[k][col].values, r_btc_oos, a, name)
    out()

out("=== VaR Backtesting -- ETH ===")
out()
for a, al in zip(alphas, alpha_labels):
    out(f"-- alpha = {a*100:.1f}% --")
    col = f"VaR_{al}pct"
    for name, k in zip(all_model_names, all_keys):
        run_backtest(all_var_eth[k][col].values, r_eth_oos, a, name)
    out()

# ============================================================
# Section 14: Violation count summary table
# ============================================================

def violation_table(var_dict, r_actual, all_keys, all_model_names):
    rows = []
    for name, k in zip(all_model_names, all_keys):
        row = {"Model": name}
        for a, al in zip(alphas, alpha_labels):
            col = f"VaR_{al}pct"
            row[f"{al}%"] = int((r_actual < var_dict[k][col].values).sum())
        rows.append(row)
    return pd.DataFrame(rows)


out("=== Violation Count Summary -- BTC ===")
out(f"(OOS: {n_oos} days | Expected at 1%: {n_oos*0.01:.1f} | 2.5%: {n_oos*0.025:.1f} | 5%: {n_oos*0.05:.1f})")
out(violation_table(all_var_btc, r_btc_oos, all_keys, all_model_names).to_string(index=False))
out()
out("=== Violation Count Summary -- ETH ===")
out(violation_table(all_var_eth, r_eth_oos, all_keys, all_model_names).to_string(index=False))
out()

# ============================================================
# Section 15: VaR Plots (GJR-t as best model)
# ============================================================

def plot_var(dates, r_actual, var_garch, var_hist, var_rm,
             alpha_label, asset, filename):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, r_actual,  color="black",     linewidth=0.7, label="Realised Return")
    ax.plot(dates, var_garch, color="blue",      linewidth=1.3, linestyle="--",
            label=f"GJR-t ({alpha_label} VaR)")
    ax.plot(dates, var_hist,  color="darkgreen", linewidth=1.3, linestyle=":",
            label="Hist Sim 250d")
    ax.plot(dates, var_rm,    color="orange",    linewidth=1.3, linestyle="-.",
            label="RiskMetrics")
    violations = np.where(r_actual < var_garch)[0]
    if len(violations):
        ax.scatter(dates[violations], r_actual[violations],
                   color="red", s=15, zorder=5, label="Violations")
    ax.set_title(f"{asset} — Out-of-Sample Returns & {alpha_label} VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return / VaR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


for a, al in zip(alphas, alpha_labels):
    col = f"VaR_{al}pct"
    plot_var(oos_dates_btc, r_btc_oos,
             var_btc["GJR_t"][col].values, var_btc_hist[col].values, var_btc_rm[col].values,
             f"{float(al)}%", "BTC", f"output/04_VaR_BTC_{al}pct.png")
    plot_var(oos_dates_eth, r_eth_oos,
             var_eth["GJR_t"][col].values, var_eth_hist[col].values, var_eth_rm[col].values,
             f"{float(al)}%", "ETH", f"output/04_VaR_ETH_{al}pct.png")

out("VaR plots saved.")
out()

# ============================================================
# Section 16: Conditional volatility — all 6 models OOS
# ============================================================

colors = ["black", "blue", "red", "darkred", "darkgreen", "forestgreen"]
ltypes = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0, (5,1))]

for coin, oos_dates, fc_dict in [("BTC", oos_dates_btc, fc_btc),
                                  ("ETH", oos_dates_eth, fc_eth)]:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, k, c, lt in zip(model_labels, keys, colors, ltypes):
        ax.plot(oos_dates, fc_dict[k]["sigma_hat"].values,
                label=label, color=c, linestyle=lt, linewidth=1.1)
    ax.set_title(f"{coin} — Out-of-Sample Conditional Volatility (All Models)")
    ax.set_xlabel("Date")
    ax.set_ylabel("σ̂_t")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"output/04_ConVola_{coin}_OOS.png", dpi=150)
    plt.close()

out("Conditional volatility plots saved.")
out()

log.close()
out("=" * 56)
out("  Part III complete.")
out("  Outputs saved to output/")
out("=" * 56)
