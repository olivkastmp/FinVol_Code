# ============================================================
# Part I: ARMA + symmetric GARCH(1,1) for BTC and ETH
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from arch.unitroot import ADF, KPSS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product

# ============================================================
# Load cleaned data (produced by 00_stylized_facts.py)
# ============================================================

BTC_clean = pd.read_csv("output/BTC_clean.csv", parse_dates=["Date"])
ETH_clean = pd.read_csv("output/ETH_clean.csv", parse_dates=["Date"])

os.makedirs("output", exist_ok=True)
log = open("output/02_Output_Part1.txt", "w")


def out(text=""):
    print(text)
    log.write(text + "\n")


# ============================================================
# Section 3: Stationarity tests
#   ADF  : H0 = unit root   → want p < 0.05
#   KPSS : H0 = stationary  → want p > 0.05
# ============================================================

for coin, df in [("BTC", BTC_clean), ("ETH", ETH_clean)]:
    r = df["Return"].values
    adf_res  = adfuller(r, autolag="AIC")
    kpss_res = kpss(r, regression="c", nlags="auto")
    out(f"=== ADF – {coin} ===")
    out(f"  Test stat: {adf_res[0]:.4f}  p-value: {adf_res[1]:.4f}")
    out(f"=== KPSS – {coin} ===")
    out(f"  Test stat: {kpss_res[0]:.4f}  p-value: {kpss_res[1]:.4f}")
    out()

# ============================================================
# Section 4: ARMA mean equation (d=0 enforced)
# Select best ARMA(p,q) by AIC, p and q in 0..4
# ============================================================

def select_arma(series, max_p=4, max_q=4):
    """Grid search over ARMA(p,q) by AIC with d=0."""
    best_aic, best_order = np.inf, (0, 0)
    for p, q in product(range(max_p + 1), range(max_q + 1)):
        try:
            m = ARIMA(series, order=(p, 0, q)).fit(method_kwargs={"warn_convergence": False})
            if m.aic < best_aic:
                best_aic, best_order = m.aic, (p, q)
        except Exception:
            pass
    return best_order, best_aic


out("=== ARMA selection ===")
(p_btc, q_btc), aic_btc = select_arma(BTC_clean["Return"].values)
(p_eth, q_eth), aic_eth = select_arma(ETH_clean["Return"].values)
out(f"  BTC best order: ARMA({p_btc},{q_btc})  AIC={aic_btc:.2f}")
out(f"  ETH best order: ARMA({p_eth},{q_eth})  AIC={aic_eth:.2f}")
out()

arma_btc = ARIMA(BTC_clean["Return"].values, order=(p_btc, 0, q_btc)).fit()
arma_eth = ARIMA(ETH_clean["Return"].values, order=(p_eth, 0, q_eth)).fit()

# ============================================================
# Section 5: ARMA residual diagnostics
#   Residuals  : want p > 0.05 (no serial correlation)
#   Residuals² : want p < 0.05 (ARCH effects present)
# ============================================================

for coin, fit in [("BTC", arma_btc), ("ETH", arma_eth)]:
    res = fit.resid
    lb1 = acorr_ljungbox(res,    lags=[20], return_df=True)
    lb2 = acorr_ljungbox(res**2, lags=[20], return_df=True)
    out(f"=== LB on ARMA residuals – {coin} ===")
    out(f"  stat={lb1['lb_stat'].iloc[0]:.4f}  p={lb1['lb_pvalue'].iloc[0]:.4f}")
    out(f"=== LB on ARMA residuals² – {coin} ===")
    out(f"  stat={lb2['lb_stat'].iloc[0]:.4f}  p={lb2['lb_pvalue'].iloc[0]:.4f}")
    out()

# ============================================================
# Section 6: ARMA(0,0) baseline check
# GARCH(1,1)-t with constant mean; if LB on std. residuals
# passes, auto.arima orders were capturing GARCH effects.
# ============================================================

def fit_garch(series, p_arma=0, q_arma=0, model="GARCH", dist="normal"):
    """Fit ARMA(p,q)+GARCH(1,1) using the arch package."""
    am = arch_model(series, mean="ARX", lags=list(range(1, p_arma + 1)) if p_arma > 0 else 0,
                    vol=model, p=1, q=1, dist=dist)
    return am.fit(disp="off", show_warning=False)


def fit_garch_full(series, p_arma, q_arma, vol="GARCH", dist="normal"):
    """
    Fit ARMA(p,q)+GARCH(1,1). arch doesn't support MA terms directly,
    so for MA we pre-filter with ARIMA and fit GARCH on residuals.
    For ARMA(0,0) or AR-only we use arch directly.
    """
    if q_arma > 0:
        arma_fit = ARIMA(series, order=(p_arma, 0, q_arma)).fit()
        resid    = arma_fit.resid
        am       = arch_model(resid, mean="Constant", vol=vol, p=1, q=1, dist=dist)
    else:
        am = arch_model(series, mean="ARX",
                        lags=list(range(1, p_arma + 1)) if p_arma > 0 else 0,
                        vol=vol, p=1, q=1, dist=dist)
    return am.fit(disp="off", show_warning=False)


# Baseline ARMA(0,0)+GARCH(1,1)-t
fit_btc_base = fit_garch_full(BTC_clean["Return"].values, 0, 0, "GARCH", "t")
fit_eth_base = fit_garch_full(ETH_clean["Return"].values, 0, 0, "GARCH", "t")

btc_stdres_base = fit_btc_base.std_resid
eth_stdres_base = fit_eth_base.std_resid

for coin, stdres in [("BTC", btc_stdres_base), ("ETH", eth_stdres_base)]:
    lb = acorr_ljungbox(stdres.dropna(), lags=[20], return_df=True)
    out(f"=== Baseline ARMA(0,0)+GARCH(1,1) – LB on std. residuals – {coin} ===")
    out(f"  stat={lb['lb_stat'].iloc[0]:.4f}  p={lb['lb_pvalue'].iloc[0]:.4f}")
    out()

btc_lb_base_p = acorr_ljungbox(btc_stdres_base.dropna(), lags=[20], return_df=True)["lb_pvalue"].iloc[0]
eth_lb_base_p = acorr_ljungbox(eth_stdres_base.dropna(), lags=[20], return_df=True)["lb_pvalue"].iloc[0]

out(f"BTC baseline LB p = {btc_lb_base_p:.4f} → "
    f"{'no residual autocorrelation: ARMA(0,0) sufficient' if btc_lb_base_p > 0.05 else 'residual autocorrelation remains'}")
out(f"ETH baseline LB p = {eth_lb_base_p:.4f} → "
    f"{'no residual autocorrelation: ARMA(0,0) sufficient' if eth_lb_base_p > 0.05 else 'residual autocorrelation remains'}")
out()

# ============================================================
# Section 7 & 8: GARCH(1,1) — Normal and Student-t
# ============================================================

fit_btc_norm = fit_garch_full(BTC_clean["Return"].values, p_btc, q_btc, "GARCH", "normal")
fit_eth_norm = fit_garch_full(ETH_clean["Return"].values, p_eth, q_eth, "GARCH", "normal")
fit_btc_t    = fit_garch_full(BTC_clean["Return"].values, p_btc, q_btc, "GARCH", "t")
fit_eth_t    = fit_garch_full(ETH_clean["Return"].values, p_eth, q_eth, "GARCH", "t")

for label, fit in [("BTC Normal", fit_btc_norm), ("BTC Student-t", fit_btc_t),
                    ("ETH Normal", fit_eth_norm), ("ETH Student-t", fit_eth_t)]:
    out(f"=== GARCH(1,1) {label} ===")
    out(str(fit.summary()))
    out()

# ============================================================
# Section 9: Information criteria table
# ============================================================

def ic(fit):
    """Return AIC and BIC from arch fit."""
    n = fit.nobs
    k = len(fit.params)
    ll = fit.loglikelihood
    return -2 * ll / n + 2 * k / n, -2 * ll / n + k * np.log(n) / n  # AIC, BIC per obs

rows_ic = []
for label, fit in [
    (f"BTC  ARMA(0,0)+GARCH(1,1)-N", fit_btc_base),
    (f"BTC  ARMA(0,0)+GARCH(1,1)-t", fit_btc_base),
    (f"BTC  ARMA({p_btc},{q_btc})+GARCH(1,1)-N", fit_btc_norm),
    (f"BTC  ARMA({p_btc},{q_btc})+GARCH(1,1)-t", fit_btc_t),
    (f"ETH  ARMA(0,0)+GARCH(1,1)-N", fit_eth_base),
    (f"ETH  ARMA(0,0)+GARCH(1,1)-t", fit_eth_base),
    (f"ETH  ARMA({p_eth},{q_eth})+GARCH(1,1)-N", fit_eth_norm),
    (f"ETH  ARMA({p_eth},{q_eth})+GARCH(1,1)-t", fit_eth_t),
]:
    a, b = ic(fit)
    rows_ic.append({"Model": label, "AIC": round(a, 6), "BIC": round(b, 6)})

ic_table = pd.DataFrame(rows_ic)
out("=== Information criteria ===")
out(ic_table.to_string(index=False))
out()

# ============================================================
# Section 10: Post-estimation diagnostics — ARCH-LM
# ============================================================

from arch.tests.arch import TestArch  # alternative: manual Engle test


def arch_lm_test(resid, lags=10):
    """Simple ARCH-LM via OLS on squared residuals."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    sq = resid ** 2
    y  = sq[lags:]
    X  = np.column_stack([sq[lags - i - 1: len(sq) - i - 1] for i in range(lags)])
    X  = add_constant(X)
    res = OLS(y, X).fit()
    stat = len(y) * res.rsquared
    pval = 1 - stats_chi2.cdf(stat, df=lags)
    return stat, pval


from scipy.stats import chi2 as stats_chi2

for label, fit in [("BTC Normal", fit_btc_norm), ("BTC Student-t", fit_btc_t),
                    ("ETH Normal", fit_eth_norm), ("ETH Student-t", fit_eth_t)]:
    stat, pval = arch_lm_test(fit.std_resid.dropna().values, lags=10)
    out(f"=== ARCH-LM – {label} ===")
    out(f"  stat={stat:.4f}  p={pval:.4f}")
    out()

# ============================================================
# Section 11: ACF/PACF of standardised residuals
# ============================================================

def save_acf_pacf_pdf(stdres, title_prefix, filename):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_acf (stdres,    ax=axes[0, 0], title=f"{title_prefix} Std Residuals ACF")
    plot_pacf(stdres,    ax=axes[0, 1], title=f"{title_prefix} Std Residuals PACF")
    plot_acf (stdres**2, ax=axes[1, 0], title=f"{title_prefix} Std Residuals² ACF")
    plot_pacf(stdres**2, ax=axes[1, 1], title=f"{title_prefix} Std Residuals² PACF")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


btc_stdres_norm = fit_btc_norm.std_resid.dropna().values
eth_stdres_norm = fit_eth_norm.std_resid.dropna().values

save_acf_pacf_pdf(btc_stdres_norm, "BTC", "output/02_Residuals_BTC_ACF.png")
save_acf_pacf_pdf(eth_stdres_norm, "ETH", "output/02_Residuals_ETH_ACF.png")

# ============================================================
# Section 12: Conditional volatility plots
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].plot(BTC_clean["Date"], fit_btc_norm.conditional_volatility,
             linewidth=0.8, color="black")
axes[0].set_title("BTC GARCH(1,1) Conditional Volatility")
axes[0].set_ylabel("Conditional Volatility")

axes[1].plot(ETH_clean["Date"], fit_eth_norm.conditional_volatility,
             linewidth=0.8, color="black")
axes[1].set_title("ETH GARCH(1,1) Conditional Volatility")
axes[1].set_ylabel("Conditional Volatility")
plt.tight_layout()
plt.savefig("output/02_Con_Vola_BTC_ETH.png", dpi=150)
plt.close()

log.close()
out("Part I complete. Outputs saved to output/")
