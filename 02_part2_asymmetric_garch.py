# ============================================================
# Part II: Asymmetric GARCH extensions for BTC and ETH
# Models: EGARCH(1,1) and GJR-GARCH(1,1) — Normal and Student-t
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from scipy.stats import chi2 as stats_chi2

# ============================================================
# Load data and Part I results
# ============================================================

BTC_clean = pd.read_csv("output/BTC_clean.csv", parse_dates=["Date"])
ETH_clean = pd.read_csv("output/ETH_clean.csv", parse_dates=["Date"])

# ARMA orders from Part I (update these if Part I gives different orders)
p_btc, q_btc = 0, 0
p_eth, q_eth = 0, 0

os.makedirs("output", exist_ok=True)
log = open("output/03_Output_Part2.txt", "w")


def out(text=""):
    print(text)
    log.write(text + "\n")


# ============================================================
# Helper: fit ARMA(p,q) + GARCH-type model
# arch doesn't support MA terms natively, so for MA > 0 we
# pre-filter with ARIMA and fit GARCH on residuals.
# ============================================================

def fit_garch_full(series, p_arma, q_arma, vol="GARCH", dist="normal"):
    if q_arma > 0:
        arma_fit = ARIMA(series, order=(p_arma, 0, q_arma)).fit()
        data     = arma_fit.resid
        mean_spec = "Constant"
        lags      = 0
    else:
        data      = series
        mean_spec = "ARX"
        lags      = list(range(1, p_arma + 1)) if p_arma > 0 else 0
    am = arch_model(data, mean=mean_spec, lags=lags, vol=vol, p=1, q=1, dist=dist)
    return am.fit(disp="off", show_warning=False)


def arch_lm_test(resid, lags=10):
    """Engle ARCH-LM test via OLS on squared residuals."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    sq  = resid ** 2
    y   = sq[lags:]
    X   = np.column_stack([sq[lags - i - 1: len(sq) - i - 1] for i in range(lags)])
    X   = add_constant(X)
    res = OLS(y, X).fit()
    stat = len(y) * res.rsquared
    pval = 1 - stats_chi2.cdf(stat, df=lags)
    return stat, pval


# ============================================================
# Sections 3–6: Fit all asymmetric models
# ============================================================

r_btc = BTC_clean["Return"].values
r_eth = ETH_clean["Return"].values

fit_btc_egarch_norm = fit_garch_full(r_btc, p_btc, q_btc, "EGARCH", "normal")
fit_eth_egarch_norm = fit_garch_full(r_eth, p_eth, q_eth, "EGARCH", "normal")
fit_btc_egarch_t    = fit_garch_full(r_btc, p_btc, q_btc, "EGARCH", "t")
fit_eth_egarch_t    = fit_garch_full(r_eth, p_eth, q_eth, "EGARCH", "t")
fit_btc_gjr_norm    = fit_garch_full(r_btc, p_btc, q_btc, "GJR-GARCH", "normal")
fit_eth_gjr_norm    = fit_garch_full(r_eth, p_eth, q_eth, "GJR-GARCH", "normal")
fit_btc_gjr_t       = fit_garch_full(r_btc, p_btc, q_btc, "GJR-GARCH", "t")
fit_eth_gjr_t       = fit_garch_full(r_eth, p_eth, q_eth, "GJR-GARCH", "t")

# also need symmetric fits from Part I for comparison
fit_btc_norm = fit_garch_full(r_btc, p_btc, q_btc, "GARCH", "normal")
fit_btc_t    = fit_garch_full(r_btc, p_btc, q_btc, "GARCH", "t")
fit_eth_norm = fit_garch_full(r_eth, p_eth, q_eth, "GARCH", "normal")
fit_eth_t    = fit_garch_full(r_eth, p_eth, q_eth, "GARCH", "t")

for label, fit in [
    ("EGARCH(1,1) Normal – BTC", fit_btc_egarch_norm),
    ("EGARCH(1,1) Normal – ETH", fit_eth_egarch_norm),
    ("EGARCH(1,1) Student-t – BTC", fit_btc_egarch_t),
    ("EGARCH(1,1) Student-t – ETH", fit_eth_egarch_t),
    ("GJR-GARCH(1,1) Normal – BTC", fit_btc_gjr_norm),
    ("GJR-GARCH(1,1) Normal – ETH", fit_eth_gjr_norm),
    ("GJR-GARCH(1,1) Student-t – BTC", fit_btc_gjr_t),
    ("GJR-GARCH(1,1) Student-t – ETH", fit_eth_gjr_t),
]:
    out(f"=== {label} ===")
    out(str(fit.summary()))
    out()

# ============================================================
# Section 7: Full model comparison — AIC/BIC
# ============================================================

def ic(fit):
    n  = fit.nobs
    k  = len(fit.params)
    ll = fit.loglikelihood
    return -2 * ll / n + 2 * k / n, -2 * ll / n + k * np.log(n) / n


all_fits = {
    "BTC_sGARCH_norm": fit_btc_norm,
    "BTC_sGARCH_t":    fit_btc_t,
    "BTC_EGARCH_norm":  fit_btc_egarch_norm,
    "BTC_EGARCH_t":     fit_btc_egarch_t,
    "BTC_GJR_norm":     fit_btc_gjr_norm,
    "BTC_GJR_t":        fit_btc_gjr_t,
    "ETH_sGARCH_norm": fit_eth_norm,
    "ETH_sGARCH_t":    fit_eth_t,
    "ETH_EGARCH_norm":  fit_eth_egarch_norm,
    "ETH_EGARCH_t":     fit_eth_egarch_t,
    "ETH_GJR_norm":     fit_eth_gjr_norm,
    "ETH_GJR_t":        fit_eth_gjr_t,
}

ic_rows = []
for name, fit in all_fits.items():
    a, b = ic(fit)
    ll   = fit.loglikelihood
    ic_rows.append({"Model": name, "LogLik": round(ll, 4),
                    "AIC": round(a, 6), "BIC": round(b, 6)})

ic_full = pd.DataFrame(ic_rows)
out("=== Full model comparison – all 12 models ===")
out(ic_full.to_string(index=False))
out()

# ============================================================
# Section 8: Asymmetry coefficients
# EGARCH: gamma (asymmetry term); GJR: gamma (leverage)
# arch parameter names: 'gamma[1]' for EGARCH, 'gamma[1]' for GJR too
# ============================================================

out("=== Asymmetry coefficients ===")

def extract_asym(fit, label):
    params   = fit.params
    pvalues  = fit.pvalues
    tvalues  = fit.tvalues
    # arch uses 'gamma[1]' for both EGARCH and GJR asymmetry
    asym_key = [k for k in params.index if "gamma" in k.lower()]
    if not asym_key:
        return
    key = asym_key[0]
    out(f"  {label:<30}  gamma = {params[key]:>8.4f}  "
        f"t = {tvalues[key]:>7.3f}  p = {pvalues[key]:.4f}")

for label, fit in [
    ("BTC EGARCH Normal",   fit_btc_egarch_norm),
    ("BTC EGARCH Student-t",fit_btc_egarch_t),
    ("BTC GJR Normal",      fit_btc_gjr_norm),
    ("BTC GJR Student-t",   fit_btc_gjr_t),
    ("ETH EGARCH Normal",   fit_eth_egarch_norm),
    ("ETH EGARCH Student-t",fit_eth_egarch_t),
    ("ETH GJR Normal",      fit_eth_gjr_norm),
    ("ETH GJR Student-t",   fit_eth_gjr_t),
]:
    extract_asym(fit, label)
out()

# ============================================================
# Section 9: ARCH-LM on standardised residuals
# ============================================================

for label, fit in [
    ("BTC_EGARCH_norm", fit_btc_egarch_norm), ("BTC_EGARCH_t", fit_btc_egarch_t),
    ("BTC_GJR_norm",    fit_btc_gjr_norm),    ("BTC_GJR_t",    fit_btc_gjr_t),
    ("ETH_EGARCH_norm", fit_eth_egarch_norm), ("ETH_EGARCH_t", fit_eth_egarch_t),
    ("ETH_GJR_norm",    fit_eth_gjr_norm),    ("ETH_GJR_t",    fit_eth_gjr_t),
]:
    stat, pval = arch_lm_test(fit.std_resid.dropna().values, lags=10)
    out(f"=== ARCH-LM – {label} ===  stat={stat:.4f}  p={pval:.4f}")
out()

# ============================================================
# Section 10: ACF/PACF — preferred Student-t asymmetric models
# ============================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def save_acf_pacf(stdres, title_prefix, filename):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_acf (stdres,    ax=axes[0, 0], title=f"{title_prefix} ACF")
    plot_pacf(stdres,    ax=axes[0, 1], title=f"{title_prefix} PACF")
    plot_acf (stdres**2, ax=axes[1, 0], title=f"{title_prefix} Squared ACF")
    plot_pacf(stdres**2, ax=axes[1, 1], title=f"{title_prefix} Squared PACF")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


save_acf_pacf(fit_btc_egarch_t.std_resid.dropna().values,
              "BTC EGARCH-t Std Residuals",   "output/03_Residuals_BTC_Asym_ACF.png")
save_acf_pacf(fit_btc_gjr_t.std_resid.dropna().values,
              "BTC GJR-t Std Residuals",      "output/03_Residuals_BTC_GJR_ACF.png")
save_acf_pacf(fit_eth_egarch_t.std_resid.dropna().values,
              "ETH EGARCH-t Std Residuals",   "output/03_Residuals_ETH_Asym_ACF.png")
save_acf_pacf(fit_eth_gjr_t.std_resid.dropna().values,
              "ETH GJR-t Std Residuals",      "output/03_Residuals_ETH_GJR_ACF.png")

# ============================================================
# Section 11: News-impact curves
#   x-axis: shock size ε_{t-1}
#   y-axis: implied σ²_t
# ============================================================

def news_impact_curve(fit, n_points=200):
    """
    Compute news-impact curve for a fitted arch model.
    Evaluates σ²_t as a function of ε_{t-1} at long-run variance level.
    """
    params = fit.params
    eps    = np.linspace(-0.2, 0.2, n_points)

    vol_type = fit.model.volatility.__class__.__name__.upper()
    omega = params.get("omega", params.get("Cst(V)", 0))

    if "EGARCH" in vol_type:
        alpha = params.get("alpha[1]", 0)
        gamma = params.get("gamma[1]", 0)
        beta  = params.get("beta[1]", 0)
        lrv   = np.exp(omega / (1 - beta))   # unconditional log-variance
        log_h = omega + alpha * (np.abs(eps) - np.sqrt(2 / np.pi)) + gamma * eps + beta * np.log(lrv)
        nic_y = np.exp(log_h)
    elif "GJR" in vol_type or "GJRGARCH" in vol_type:
        alpha = params.get("alpha[1]", 0)
        gamma = params.get("gamma[1]", 0)
        beta  = params.get("beta[1]", 0)
        lrv   = omega / max(1 - alpha - 0.5 * gamma - beta, 1e-8)
        nic_y = omega + (alpha + gamma * (eps < 0).astype(float)) * eps**2 + beta * lrv
    else:   # standard GARCH
        alpha = params.get("alpha[1]", 0)
        beta  = params.get("beta[1]", 0)
        lrv   = omega / max(1 - alpha - beta, 1e-8)
        nic_y = omega + alpha * eps**2 + beta * lrv

    return eps, nic_y


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, (coin, fits) in enumerate([
    ("BTC", [("sGARCH-t",  fit_btc_t),
             ("EGARCH-t",  fit_btc_egarch_t),
             ("GJR-t",     fit_btc_gjr_t)]),
    ("ETH", [("sGARCH-t",  fit_eth_t),
             ("EGARCH-t",  fit_eth_egarch_t),
             ("GJR-t",     fit_eth_gjr_t)]),
]):
    for col, (label, fit) in enumerate(fits):
        x, y = news_impact_curve(fit)
        axes[row, col].plot(x, y, linewidth=1.5, color="black")
        axes[row, col].set_title(f"{coin} {label}", fontsize=9)
        axes[row, col].set_xlabel("ε_{t-1}")
        axes[row, col].set_ylabel("σ²_t")
        axes[row, col].grid(alpha=0.2)
plt.suptitle("News-Impact Curves", fontsize=12)
plt.tight_layout()
plt.savefig("output/03_NewsImpact.png", dpi=150)
plt.close()

# ============================================================
# Section 12: Conditional volatility comparison
#   All three Student-t models overlaid per asset
# ============================================================

btc_dates = BTC_clean["Date"]
eth_dates = ETH_clean["Date"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for ax, coin, dates, fits in [
    (axes[0], "BTC", btc_dates, [
        ("sGARCH",    "black", "-",  fit_btc_t),
        ("EGARCH",    "blue",  "--", fit_btc_egarch_t),
        ("GJR-GARCH", "red",   ":",  fit_btc_gjr_t),
    ]),
    (axes[1], "ETH", eth_dates, [
        ("sGARCH",    "black", "-",  fit_eth_t),
        ("EGARCH",    "blue",  "--", fit_eth_egarch_t),
        ("GJR-GARCH", "red",   ":",  fit_eth_gjr_t),
    ]),
]:
    for label, color, ls, fit in fits:
        ax.plot(dates, fit.conditional_volatility,
                label=label, color=color, linestyle=ls, linewidth=1.2)
    ax.set_title(f"{coin} Conditional Volatility — All Student-t Models")
    ax.set_ylabel("Conditional Volatility")
    ax.legend()
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("output/03_Con_Vola_AllModels.png", dpi=150)
plt.close()

log.close()
out("Part II complete. Outputs saved to output/")
