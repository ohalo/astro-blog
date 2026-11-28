# -*- coding: utf-8 -*-
"""Regime-switching jump diffusion for electricity prices and peak-call pricing."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SEED = 20261128
N_DAYS = 3 * 365
PHI = 0.72
SIGMA = 4.0
P01 = 0.008
P10 = 0.070
P_JUMP_CALM = 0.012
P_JUMP_STRESS = 0.080
WIND_PROB = 0.18
WIND_SHIFT = 30.0
JUMP_UP_P = 0.72
JUMP_UP_MU = 38.0
JUMP_UP_SD = 12.0
JUMP_DN_MU = 28.0
JUMP_DN_SD = 9.0
RATE = 0.03
N_PATHS = 6000
HORIZON = 22
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "electricity-spike-jump-pricing")


def seasonal(t):
    return 45.0 + 12.0 * np.cos(2.0 * np.pi * (t - 20.0) / 365.0) + 4.0 * np.sin(4.0 * np.pi * t / 365.0)


def simulate(rng, n, start_state=0, wind_prob=WIND_PROB, wind_shift=WIND_SHIFT,
             phi=PHI, sigma=SIGMA, calm_p=P_JUMP_CALM, stress_p=P_JUMP_STRESS,
             t0=0, track=False):
    state = start_state
    x = 0.0
    prices = np.empty(n)
    jumps = np.empty(n)
    winds = np.empty(n, dtype=bool)
    states = np.empty(n, dtype=int)
    eps = rng.standard_normal(n)
    unif = rng.random(n)
    jsz = rng.standard_normal(n)
    for i in range(n):
        t = t0 + i
        if state == 0 and unif[i] < P01:
            state = 1
        elif state == 1 and unif[i] < P10:
            state = 0
        w = rng.random() < wind_prob
        pj = stress_p if state == 1 else calm_p
        j = 0.0
        if unif[i] < pj:
            if rng.random() < JUMP_UP_P:
                j = JUMP_UP_MU + JUMP_UP_SD * jsz[i]
            else:
                j = -JUMP_DN_MU + JUMP_DN_SD * jsz[i]
        x = phi * x + sigma * eps[i] + j
        prices[i] = seasonal(t) - wind_shift * w + x
        jumps[i] = j
        winds[i] = w
        states[i] = state
    if track:
        return prices, jumps, winds, states
    return prices


def main():
    rng = np.random.default_rng(SEED)
    hist, hjump, hwind, hstate = simulate(rng, N_DAYS, track=True)

    hist_mean = float(np.mean(hist))
    hist_sd = float(np.std(hist, ddof=1))
    neg_share = float(np.mean(hist < 0.0))
    q995 = float(np.quantile(hist, 0.995))
    q999 = float(np.quantile(hist, 0.999))
    gauss_q995 = hist_mean + norm.ppf(0.995) * hist_sd
    gauss_q999 = hist_mean + norm.ppf(0.999) * hist_sd
    tail_ratio = q999 / gauss_q999

    diff = np.diff(hist)
    seas_diff = np.array([seasonal(t + 1) - seasonal(t) for t in range(N_DAYS - 1)])
    wind_diff = -WIND_SHIFT * (hwind[1:].astype(float) - hwind[:-1].astype(float))
    resid_diff = diff - seas_diff - wind_diff
    det_counts = {}
    det_rates = {}
    for thr in (12.0, 18.0, 25.0):
        cnt = int(np.sum(np.abs(resid_diff) > thr))
        det_counts[str(thr)] = cnt
        det_rates[str(thr)] = round(cnt / (N_DAYS - 1), 6)

    med = float(np.median(hist))
    mad_sd = 1.4826 * float(np.median(np.abs(hist - med)))
    robust_q999 = med + norm.ppf(0.999) * mad_sd

    # reference density: stationary stress probability from the two-state chain
    pi_stress = P01 / (P01 + P10)
    avg_jump_p = (1.0 - pi_stress) * P_JUMP_CALM + pi_stress * P_JUMP_STRESS

    # monthly peak-call: average of 22 day-ahead prices starting at t=0
    strikes = [45.0, 55.0, 65.0, 75.0, 85.0]
    disc = np.exp(-RATE * HORIZON / 365.0)
    jump_avg = np.empty(N_PATHS)
    gauss_avg = np.empty(N_PATHS)
    low_avg = np.empty(N_PATHS)
    for i in range(N_PATHS):
        p = simulate(rng, HORIZON, start_state=0, t0=0)
        jump_avg[i] = np.mean(p)
        g = np.full(HORIZON, hist_mean)
        eps = rng.standard_normal(HORIZON)
        x = np.zeros(HORIZON)
        sig_g = hist_sd * np.sqrt(1.0 - PHI ** 2)
        for k in range(HORIZON):
            x[k] = PHI * x[k - 1] + sig_g * eps[k] if k > 0 else sig_g * eps[k]
        gauss_avg[i] = np.mean(g + x)
        low_avg[i] = np.mean(simulate(rng, HORIZON, start_state=0, t0=0,
                                      calm_p=P_JUMP_CALM * 0.35, stress_p=P_JUMP_STRESS * 0.35))

    def call_price(samples, k):
        return disc * float(np.mean(np.maximum(samples - k, 0.0)))

    jump_prices = [call_price(jump_avg, k) for k in strikes]
    gauss_prices = [call_price(gauss_avg, k) for k in strikes]
    low_prices = [call_price(low_avg, k) for k in strikes]
    rel_uplift = {str(int(k)): round((jump_prices[i] / gauss_prices[i] - 1.0) * 100.0, 2)
                  for i, k in enumerate(strikes) if gauss_prices[i] > 1e-9}
    underprice_low = {str(int(k)): round((1.0 - low_prices[i] / jump_prices[i]) * 100.0, 2)
                      for i, k in enumerate(strikes) if jump_prices[i] > 1e-9}

    # high-renewable scenario: negative prices and one-day-ahead put at zero strike
    hi_rng = np.random.default_rng(SEED + 7)
    hi = simulate(hi_rng, 4000, wind_prob=0.40, wind_shift=45.0)
    hi_neg = float(np.mean(hi < 0.0))
    hi_mean = float(np.mean(hi))
    hi_sd = float(np.std(hi, ddof=1))
    true_zero_put = float(np.mean(np.maximum(-hi, 0.0))) * np.exp(-RATE / 365.0)
    pos = hi[hi > 0.0]
    mu_ln = float(np.mean(np.log(pos)))
    sd_ln = float(np.std(np.log(pos), ddof=1))
    lognormal_zero_put = 0.0
    lognormal_call10 = np.exp(-RATE / 365.0) * (
        np.exp(mu_ln + 0.5 * sd_ln ** 2) * norm.cdf((mu_ln + sd_ln ** 2 - np.log(10.0)) / sd_ln)
        - 10.0 * norm.cdf((mu_ln - np.log(10.0)) / sd_ln))
    ln_sd_mom = np.sqrt(np.log(1.0 + hi_sd ** 2 / hi_mean ** 2))
    ln_mu_mom = np.log(hi_mean) - 0.5 * ln_sd_mom ** 2
    lognormal_mom_call10 = np.exp(-RATE / 365.0) * (
        np.exp(ln_mu_mom + 0.5 * ln_sd_mom ** 2) * norm.cdf((ln_mu_mom + ln_sd_mom ** 2 - np.log(10.0)) / ln_sd_mom)
        - 10.0 * norm.cdf((ln_mu_mom - np.log(10.0)) / ln_sd_mom))
    true_call10 = float(np.mean(np.maximum(hi - 10.0, 0.0))) * np.exp(-RATE / 365.0)

    # Figure 1: sample path with jumps and wind-driven negative periods
    fig, ax = plt.subplots(figsize=(11, 4))
    show = slice(0, 420)
    idx = np.arange(*show.indices(N_DAYS))
    ax.plot(idx, hist[show], lw=0.8, color="#3b5b8c", label="day-ahead price")
    ax.plot(idx, seasonal(idx), lw=1.0, color="#555555", ls="--", label="seasonal mean")
    ax.axhline(0.0, color="#c0392b", lw=0.8)
    neg_idx = idx[hwind[show]]
    ax.scatter(neg_idx, hist[show][hwind[show]], s=6, color="#4aa3a2", label="wind state")
    jup = idx[hjump[show] > 15]
    ax.scatter(jup, hist[show][hjump[show] > 15], s=18, marker="^", color="#e07b39", label="upward jump")
    ax.set_title("Simulated day-ahead electricity price")
    ax.set_xlabel("day")
    ax.set_ylabel("price (EUR/MWh)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_price_path.png"), dpi=150)
    plt.close(fig)

    # Figure 2: tail versus Gaussian calibration
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    xgrid = np.linspace(min(-40.0, hist.min()), max(120.0, hist.max()), 240)
    axes[0].hist(hist, bins=90, density=True, color="#8fb3d9", alpha=0.85)
    axes[0].plot(xgrid, norm.pdf(xgrid, hist_mean, hist_sd), color="#c0392b", lw=1.4, label="fitted normal")
    axes[0].set_xlim(-40, 140)
    axes[0].set_title("Price distribution and normal fit")
    axes[0].set_xlabel("price")
    axes[0].legend()
    probs = np.array([0.90, 0.95, 0.99, 0.995, 0.999])
    emp_q = np.quantile(hist, probs)
    nor_q = norm.ppf(probs, hist_mean, hist_sd)
    axes[1].plot(nor_q, emp_q, "o", color="#3b5b8c")
    axes[1].plot([nor_q.min(), nor_q.max()], [nor_q.min(), nor_q.max()], "--", color="#888888")
    axes[1].set_xlabel("normal quantile")
    axes[1].set_ylabel("empirical quantile")
    axes[1].set_title("Tail QQ: spikes above the normal line")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_tail_distribution.png"), dpi=150)
    plt.close(fig)

    # Figure 3: peak-call price curves
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(strikes, jump_prices, "o-", color="#c0392b", label="jump model")
    axes[0].plot(strikes, gauss_prices, "s-", color="#3b5b8c", label="normal model")
    axes[0].plot(strikes, low_prices, "^--", color="#7a8b8b", label="underestimated intensity")
    axes[0].set_title("22-day average-price call")
    axes[0].set_xlabel("strike")
    axes[0].set_ylabel("discounted price")
    axes[0].legend()
    axes[1].bar([str(int(k)) for k in strikes], [jump_prices[i] - gauss_prices[i] for i in range(len(strikes))],
                color="#e07b39")
    axes[1].set_title("Normal-model underpricing")
    axes[1].set_xlabel("strike")
    axes[1].set_ylabel("price gap")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_peak_call_prices.png"), dpi=150)
    plt.close(fig)

    # Figure 4: negative prices under high renewables
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].hist(hi, bins=80, density=True, color="#4aa3a2", alpha=0.85)
    axes[0].axvline(0.0, color="#c0392b", ls="--", lw=1.2)
    axes[0].set_title("High-renewable price distribution")
    axes[0].set_xlabel("price")
    axes[0].set_ylabel("density")
    axes[1].bar(["zero-strike put", "10-strike call"],
                [true_zero_put, true_call10], color="#3b5b8c", label="arithmetic model")
    axes[1].bar(["zero-strike put", "10-strike call"],
                [lognormal_zero_put, lognormal_call10], color="#c0392b", alpha=0.7, label="lognormal on positives")
    axes[1].bar(["", ""], [0.0, lognormal_mom_call10], color="#e0a33e", alpha=0.8, label="lognormal on moments")
    axes[1].set_title("One-day options under negative prices")
    axes[1].set_ylabel("price")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_negative_price.png"), dpi=150)
    plt.close(fig)

    # Figure 5: jump-detection threshold and intensity error
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    thrs = [12.0, 18.0, 25.0]
    axes[0].plot(thrs, [det_rates[str(t)] for t in thrs], "o-", color="#8e6cbb")
    axes[0].axhline(avg_jump_p, color="#c0392b", ls="--", label="true mean intensity")
    axes[0].set_title("Detected jump rate versus threshold")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("rate")
    axes[0].legend()
    uplift_ks = [k for k in strikes if str(int(k)) in rel_uplift]
    axes[1].plot(uplift_ks, [rel_uplift[str(int(k))] for k in uplift_ks], "o-", color="#c0392b", label="normal model")
    low_ks = [k for k in strikes if str(int(k)) in underprice_low]
    axes[1].plot(low_ks, [underprice_low[str(int(k))] for k in low_ks], "s-", color="#7a8b8b", label="low intensity")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0.0, color="#888888", lw=0.8)
    axes[1].set_title("Pricing error from misestimated intensity")
    axes[1].set_xlabel("strike")
    axes[1].set_ylabel("percent")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_intensity_sensitivity.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "n_days": N_DAYS,
        "hist_mean": round(hist_mean, 6),
        "hist_sd": round(hist_sd, 6),
        "negative_price_share": round(neg_share, 6),
        "emp_q995": round(q995, 6),
        "emp_q999": round(q999, 6),
        "normal_q995": round(gauss_q995, 6),
        "normal_q999": round(gauss_q999, 6),
        "robust_body_sd": round(mad_sd, 6),
        "robust_q999": round(robust_q999, 6),
        "tail_ratio_q999": round(tail_ratio, 6),
        "tail_ratio_robust_q999": round(q999 / robust_q999, 6),
        "stationary_stress_prob": round(pi_stress, 6),
        "mean_jump_intensity": round(avg_jump_p, 6),
        "detected_jump_counts": det_counts,
        "detected_jump_rates": det_rates,
        "strikes": strikes,
        "jump_call_prices": [round(v, 6) for v in jump_prices],
        "normal_call_prices": [round(v, 6) for v in gauss_prices],
        "low_intensity_call_prices": [round(v, 6) for v in low_prices],
        "normal_underpricing_pct": rel_uplift,
        "low_intensity_underpricing_pct": underprice_low,
        "high_renewable_negative_share": round(hi_neg, 6),
        "high_renewable_mean": round(hi_mean, 6),
        "high_renewable_sd": round(hi_sd, 6),
        "true_zero_strike_put": round(true_zero_put, 6),
        "lognormal_zero_strike_put": round(lognormal_zero_put, 6),
        "true_10_strike_call": round(true_call10, 6),
        "lognormal_10_strike_call": round(lognormal_call10, 6),
        "lognormal_moment_10_strike_call": round(lognormal_mom_call10, 6),
        "n_paths": N_PATHS,
        "horizon_days": HORIZON,
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
