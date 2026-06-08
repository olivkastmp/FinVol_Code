# ============================================================
# Stylized Facts: Summary Statistics
# ============================================================
# Data source: https://tokeninsight.com/en/cryptocurrencies
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf

# ============================================================
# Load raw data
# ============================================================

BTC_raw = pd.read_csv("BTC.csv")
ETH_raw = pd.read_csv("ETH.csv")

# ============================================================
# Helper functions
# ============================================================

def compute_returns(df, start_date="2020-01-01"):
    """Clean, filter, and compute log returns for a coin dataset."""
    df = df.copy()
    # strip time component if present
    df["Date"] = pd.to_datetime(df["Date"].str.split(" ").str[0])
    df = df[df["Date"] >= pd.Timestamp(start_date)]
    df = df.sort_values("Date").reset_index(drop=True)
    df["Return"] = np.log(df["Price"]).diff()
    df = df.dropna(subset=["Return"]).reset_index(drop=True)
    return df


def plot_returns(df, coin_name, ax, ylim=None):
    """Plot log returns on a given axes."""
    ax.plot(df["Date"], df["Return"], linewidth=0.7, color="black")
    ax.set_title(f"{coin_name} — Log Returns", fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Log Returns")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, which="major", alpha=0.3)
    ax.minorticks_off()


# ============================================================
# Process datasets (start date: 1 January 2020)
# ============================================================

BTC_clean = compute_returns(BTC_raw)
ETH_clean = compute_returns(ETH_raw)

os.makedirs("output", exist_ok=True)

BTC_clean.to_csv("output/BTC_clean.csv", index=False)
ETH_clean.to_csv("output/ETH_clean.csv", index=False)

# ============================================================
# Plot log returns side by side
# ============================================================

common_ylim = (
    min(BTC_clean["Return"].min(), ETH_clean["Return"].min()),
    max(BTC_clean["Return"].max(), ETH_clean["Return"].max()),
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_returns(BTC_clean, "BTC", axes[0], ylim=common_ylim)
plot_returns(ETH_clean, "ETH", axes[1], ylim=common_ylim)
plt.tight_layout()
plt.savefig("output/returns_plot.png", dpi=150)
plt.show()

# ============================================================
# Descriptive statistics (incl. Jarque-Bera)
# ============================================================

# crypto trades every day, so annualise with 365 trading days
ANNUALISATION_FACTOR = 365


def compute_descriptive_stats(returns, coin_name):
    """Compute summary stats including Jarque-Bera test."""
    jb_stat, jb_p, _, _ = jarque_bera(returns)
    return {
        "Symbol":        coin_name,
        "Mean":          returns.mean(),
        "Median":        np.median(returns),
        "SD":            returns.std(),
        "Annualized_SD": returns.std() * np.sqrt(ANNUALISATION_FACTOR),
        "Min":           returns.min(),
        "Max":           returns.max(),
        "Skewness":      stats.skew(returns),
        "Kurtosis":      stats.kurtosis(returns),   # excess kurtosis (Fisher)
        "Jarque_Bera":   jb_stat,
        "JB_p_value":    jb_p,
    }


summary_table = pd.DataFrame([
    compute_descriptive_stats(BTC_clean["Return"].values, "BTC"),
    compute_descriptive_stats(ETH_clean["Return"].values, "ETH"),
])

print(summary_table.to_string(index=False))

# ============================================================
# Visualizing Non-Normality: histograms with normal overlay
# ============================================================

crypto_returns = {"BTC": BTC_clean["Return"].values,
                  "ETH": ETH_clean["Return"].values}

all_rets = np.concatenate(list(crypto_returns.values()))
common_xlim = (all_rets.min(), all_rets.max())
common_breaks = np.linspace(common_xlim[0], common_xlim[1], 101)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (coin, ret) in zip(axes, crypto_returns.items()):
    ax.hist(ret, bins=common_breaks, density=True, color="steelblue",
            edgecolor="white", linewidth=0.3, alpha=0.8)
    x = np.linspace(common_xlim[0], common_xlim[1], 300)
    ax.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()),
            color="red", linewidth=2, label="Normal fit")
    ax.set_title(coin)
    ax.set_xlabel(f"{coin} Daily Returns")
    ax.set_xlim(common_xlim)
    ax.legend()
plt.tight_layout()
plt.savefig("output/histograms.png", dpi=150)
plt.show()

# ============================================================
# Autocorrelation
# ============================================================

def plot_acf_manual(series, title, ax, n_lags=40):
    """Plot ACF on a given axes (manual, no statsmodels wrapper)."""
    acf_vals = acf(series, nlags=n_lags, fft=True)
    lags = np.arange(len(acf_vals))
    conf = 1.96 / np.sqrt(len(series))
    ax.bar(lags, acf_vals, color="steelblue", width=0.4)
    ax.axhline(conf,  color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lag")


transforms = [
    (lambda x: x,    "ACF"),
    (lambda x: x**2, "Squared Returns ACF"),
    (lambda x: np.abs(x), "Absolute Returns ACF"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for row, (coin, ret) in enumerate(crypto_returns.items()):
    for col, (fn, label) in enumerate(transforms):
        plot_acf_manual(fn(ret), f"{coin} {label}", axes[row, col])
plt.suptitle("ACF Plots", fontsize=12)
plt.tight_layout()
plt.savefig("output/acf_plots.png", dpi=150)
plt.show()

# ============================================================
# Hill's Tail Index Estimate
# ============================================================

def hill_estimator(x, k):
    """
    Hill tail index estimator.
    x : array of returns
    k : number of upper-order statistics to use
    Sorts largest to smallest; Hill estimator focuses on extreme tail observations.
    """
    x = x[np.isfinite(x)]
    x = np.sort(x)[::-1]   # largest to smallest

    if k <= 1 or k >= len(x):
        raise ValueError("k must be > 1 and < sample size")

    # measures how quickly extreme values decay
    hill_gamma = np.mean(np.log(x[:k]) - np.log(x[k]))
    # lower alpha = heavier tail, higher alpha = thinner tail
    alpha = 1.0 / hill_gamma
    return {"Hill_gamma": hill_gamma, "Tail_index_alpha": alpha}


def build_tail_samples(returns):
    """Build absolute, right, and left tail samples for one coin."""
    return {
        "abs":   np.abs(returns),
        "right": returns[returns > 0],
        "left":  -returns[returns < 0],   # convert losses to positive
    }


btc_tails = build_tail_samples(BTC_clean["Return"].values)
eth_tails = build_tail_samples(ETH_clean["Return"].values)

# k = 10% and 5% of each tail sample
def choose_k(tail_dict, frac=0.10):
    return {name: int(frac * len(v)) for name, v in tail_dict.items()}

btc_k  = choose_k(btc_tails, 0.10)
eth_k  = choose_k(eth_tails, 0.10)
btc_k1 = choose_k(btc_tails, 0.05)
eth_k1 = choose_k(eth_tails, 0.05)

rows = []
for coin, tails, ks in [("BTC", btc_tails, btc_k), ("ETH", eth_tails, eth_k)]:
    for tail_name in ["abs", "right", "left"]:
        res = hill_estimator(tails[tail_name], ks[tail_name])
        rows.append({"Coin": coin, "Tail": tail_name.capitalize(), **res})

hill_results = pd.DataFrame(rows)
print("\nHill estimates (k = 10%):")
print(hill_results.to_string(index=False))

rows1 = []
for coin, tails, ks in [("BTC", btc_tails, btc_k1), ("ETH", eth_tails, eth_k1)]:
    for tail_name in ["abs", "right", "left"]:
        res = hill_estimator(tails[tail_name], ks[tail_name])
        rows1.append({"Coin": coin, "Tail": tail_name.capitalize(), **res})

hill_results1 = pd.DataFrame(rows1)
print("\nHill estimates (k = 5%):")
print(hill_results1.to_string(index=False))

# ============================================================
# Hill Plot: alpha as a function of tail fraction p = k / n_tail
# ============================================================

def hill_plot_data(tail_vec, p_min=0.01, p_max=0.10):
    tail_vec = tail_vec[np.isfinite(tail_vec)]
    tail_vec = np.sort(tail_vec)[::-1]
    n = len(tail_vec)
    k_min = max(2, int(p_min * n))
    k_max = min(int(p_max * n), n - 2)
    k_seq = np.arange(k_min, k_max + 1)
    alphas = np.array([
        1.0 / np.mean(np.log(tail_vec[:k]) - np.log(tail_vec[k]))
        for k in k_seq
    ])
    return pd.DataFrame({"k": k_seq, "p": k_seq / n, "alpha": alphas})


coins  = {"BTC": btc_tails, "ETH": eth_tails}
tail_labels = {"abs": "Absolute", "right": "Right", "left": "Left"}
colors = {"Absolute": "#2166ac", "Right": "#d6604d", "Left": "#4dac26"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (coin, tails) in zip(axes, coins.items()):
    for tail_name, label in tail_labels.items():
        df = hill_plot_data(tails[tail_name], p_min=0.01, p_max=0.10)
        ax.plot(df["p"] * 100, df["alpha"], label=label,
                color=colors[label], linewidth=0.9)
    ax.axvline(5,  color="grey", linestyle="dotted", linewidth=0.8)
    ax.axvline(10, color="grey", linestyle="dashed",  linewidth=0.8)
    ax.set_title(coin, fontweight="bold")
    ax.set_xlabel("Tail fraction k / n_tail (%)")
    ax.set_ylabel("α̂ = 1/γ̂")
    ax.legend()
    ax.grid(alpha=0.2)
plt.suptitle("Hill Plot: Tail Index α as a Function of Tail Fraction", fontsize=12)
plt.tight_layout()
plt.savefig("output/hill_plot.png", dpi=150)
plt.show()

# Hill confidence intervals (asymptotic SE)
def hill_ci(alpha_hat, k, conf=0.95):
    z  = stats.norm.ppf(1 - (1 - conf) / 2)
    se = alpha_hat / np.sqrt(k)
    return {"lower": alpha_hat - z * se, "upper": alpha_hat + z * se}

print("\nHill CI (BTC absolute, 10%):", hill_ci(2.997, btc_k["abs"]))
print("Hill CI (ETH absolute, 10%):", hill_ci(2.929, eth_k["abs"]))

# ============================================================
# Stylized Fact (v): Leverage Effects
# Cont (2001): asymmetry between past +/- returns on |r_t|
# ============================================================

def leverage_cors(returns, max_h=7):
    """Compute rho(r+_{t-h}, |r_t|) and rho(-r-_{t-h}, |r_t|) for h=1..max_h."""
    abs_ret = np.abs(returns)
    r_pos   = np.maximum(returns, 0)
    r_neg   = np.maximum(-returns, 0)
    results = []
    for h in range(1, max_h + 1):
        abs_lead = abs_ret[h:]
        pos_lag  = r_pos[:len(returns) - h]
        neg_lag  = r_neg[:len(returns) - h]
        results.append({
            "h":       h,
            "rho_pos": np.corrcoef(pos_lag, abs_lead)[0, 1],
            "rho_neg": np.corrcoef(neg_lag, abs_lead)[0, 1],
        })
    return pd.DataFrame(results)


def leverage_gw_tstat(returns, h, sign="pos", q=8):
    """Group-wise robust t-statistic for leverage correlations. H0: rho=0."""
    abs_ret   = np.abs(returns)
    predictor = np.maximum(returns, 0) if sign == "pos" else np.maximum(-returns, 0)
    y = abs_ret[h:]
    x = predictor[:len(returns) - h]
    n          = len(y)
    group_size = n // q
    n_use      = group_size * q
    y, x       = y[:n_use], x[:n_use]
    groups     = np.repeat(np.arange(q), group_size)
    rho_g      = [np.corrcoef(y[groups == g], x[groups == g])[0, 1]
                  for g in range(q)]
    t_stat, p_val = stats.ttest_1samp(rho_g, 0)
    return {"t_stat": t_stat, "p_value": p_val, "mean_rho": np.mean(rho_g)}


def run_leverage_tests(returns, coin, max_h=7, q=8):
    cors_df = leverage_cors(returns, max_h)
    rows = []
    for h in range(1, max_h + 1):
        row_cors = cors_df[cors_df["h"] == h].iloc[0]
        gw_pos   = leverage_gw_tstat(returns, h, "pos", q)
        gw_neg   = leverage_gw_tstat(returns, h, "neg", q)
        rows.append({
            "Coin":       coin,
            "h":          h,
            "rho_pos":    round(row_cors["rho_pos"], 4),
            "GW_t_pos":   round(gw_pos["t_stat"],   4),
            "GW_p_pos":   round(gw_pos["p_value"],  4),
            "rho_neg":    round(row_cors["rho_neg"], 4),
            "GW_t_neg":   round(gw_neg["t_stat"],   4),
            "GW_p_neg":   round(gw_neg["p_value"],  4),
            "asymmetry":  round(row_cors["rho_neg"] - row_cors["rho_pos"], 4),
        })
    return pd.DataFrame(rows)


lev_btc   = run_leverage_tests(BTC_clean["Return"].values, "BTC")
lev_eth   = run_leverage_tests(ETH_clean["Return"].values, "ETH")
lev_table = pd.concat([lev_btc, lev_eth], ignore_index=True)

print("\n" + "=" * 65)
print(" Stylized Fact (v) — Leverage Effects")
print("=" * 65)
for coin in ["BTC", "ETH"]:
    sub = lev_table[lev_table["Coin"] == coin]
    print(f"\n {coin}")
    print(f"  {'h:':<35}", "".join(f"{h:>7d}" for h in sub["h"]))
    print(f"  {'rho(r+_(t-h), |r_t|):':<35}", "".join(f"{v:>7.3f}" for v in sub["rho_pos"]))
    print(f"  {'  GW t-stat:':<35}", "".join(f"{v:>7.3f}" for v in sub["GW_t_pos"]))
    print(f"  {'rho(-r-_(t-h), |r_t|):':<35}", "".join(f"{v:>7.3f}" for v in sub["rho_neg"]))
    print(f"  {'  GW t-stat:':<35}", "".join(f"{v:>7.3f}" for v in sub["GW_t_neg"]))
    print(f"  {'  Asymmetry (neg - pos):':<35}", "".join(f"{v:>7.3f}" for v in sub["asymmetry"]))

# Leverage effect plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, coin in zip(axes, ["BTC", "ETH"]):
    sub = lev_table[lev_table["Coin"] == coin]
    ax.plot(sub["h"], sub["rho_pos"], marker="o", linestyle="--",
            color="#2166ac", label="Positive lag (r⁺)")
    ax.plot(sub["h"], sub["rho_neg"], marker="o", linestyle="-",
            color="#d6604d", label="Negative lag (−r⁻)")
    ax.axhline(0, color="grey", linestyle="dotted", linewidth=0.8)
    ax.set_xticks(range(1, 8))
    ax.set_title(coin, fontweight="bold")
    ax.set_xlabel("Lag h")
    ax.set_ylabel("ρ̂")
    ax.legend()
    ax.grid(alpha=0.2)
plt.suptitle("Stylized Fact (v): Leverage Effects", fontsize=12)
plt.tight_layout()
plt.savefig("output/leverage_effects.png", dpi=150)
plt.show()
