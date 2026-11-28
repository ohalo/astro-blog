# -*- coding: utf-8 -*-
"""Martingale optimal transport bounds from sparse call quotes (deterministic experiment)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.stats import lognorm

S0 = 100.0
SEED = 20261128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "martingale-transport-option-bounds")
os.makedirs(OUT, exist_ok=True)

grid = np.arange(40.0, 181.0, 1.0)
one = np.ones_like(grid)


def true_distribution():
    w = np.zeros_like(grid)
    for weight, mean, sigma in ((0.62, 100.0, 0.18), (0.38, 128.0, 0.34)):
        scale = mean * np.exp(-0.5 * sigma * sigma)
        w += weight * lognorm.pdf(grid, sigma, scale=scale)
    w /= w.sum()

    def mean_gap(eta):
        tilted = w * np.exp(eta * grid)
        tilted /= tilted.sum()
        return tilted @ grid - S0

    eta = brentq(mean_gap, -0.5, 0.5, xtol=1e-14)
    w = w * np.exp(eta * grid)
    w /= w.sum()
    return w


p_true = true_distribution()
mom1 = float(p_true @ grid)
mom2 = float(p_true @ (grid ** 2))


def call_price(strike):
    return float(p_true @ np.maximum(grid - strike, 0.0))


def digital_price(strike):
    return float(p_true @ (grid >= strike))


quote_strikes = np.arange(60.0, 141.0, 10.0)
quotes = [call_price(k) for k in quote_strikes]


def bounds(payoff, call_list=None, use_moment=False):
    rows = [one.tolist(), grid.tolist()]
    rhs = [1.0, S0]
    if call_list:
        for strike, price in call_list:
            rows.append(np.maximum(grid - strike, 0.0).tolist())
            rhs.append(price)
    if use_moment:
        rows.append((grid ** 2).tolist())
        rhs.append(mom2)
    A = np.array(rows)
    b = np.array(rhs)
    up = linprog(-payoff, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    lo = linprog(payoff, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if up.status != 0 or lo.status != 0:
        return None, None, None, None
    return float(-up.fun), float(lo.fun), up.x, lo.x


K = 120.0
payoff = (grid >= K).astype(float)
q = [(float(k), float(c)) for k, c in zip(quote_strikes, quotes)]
q3 = [(k, c) for k, c in q if k >= 120.0]

d_up, d_lo, _, _ = bounds(payoff)
m_up, m_lo, _, _ = bounds(payoff, q, use_moment=False)
b3_up, b3_lo, _, _ = bounds(payoff, q3, use_moment=False)
t_up, t_lo, p_up, p_lo = bounds(payoff, q3, use_moment=True)

# Fig 1: quote curve and the risk-neutral distribution used to generate it.
fig, ax = plt.subplots(figsize=(8, 4.8))
ks = np.arange(50.0, 151.0, 1.0)
ax.plot(ks, [call_price(k) for k in ks], color="#1764ab", lw=2, label="True call curve")
ax.scatter(quote_strikes, quotes, s=58, color="#c23b22", zorder=3, label="Observed quotes")
ax.set_xlabel("Strike")
ax.set_ylabel("Call price")
ax.set_title("Sparse call quotes sampled from a two-component distribution")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_call_quotes.png"), dpi=150)
plt.close(fig)

# Fig 2: digital bounds across strikes under three information sets.
ks2 = np.arange(60.0, 151.0, 5.0)
mart_up, mart_lo, quote_up, quote_lo, mom_up_s, mom_lo_s = [], [], [], [], [], []
for strike in ks2:
    f = (grid >= strike).astype(float)
    a, b, _, _ = bounds(f)
    c, d, _, _ = bounds(f, q, use_moment=False)
    e, g, _, _ = bounds(f, [(k, c) for k, c in q if k >= 120.0], use_moment=True)
    mart_up.append(a)
    mart_lo.append(b)
    quote_up.append(c)
    quote_lo.append(d)
    mom_up_s.append(e)
    mom_lo_s.append(g)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.fill_between(ks2, mart_lo, mart_up, color="#d9e6f2", label="Martingale only")
ax.fill_between(ks2, quote_lo, quote_up, color="#f6c9a8", label="Call quotes")
ax.plot(ks2, mom_lo_s, color="#1a7f4b", lw=1.8, label="3 high quotes + second moment")
ax.plot(ks2, mom_up_s, color="#1a7f4b", lw=1.8)
ax.plot(ks2, [digital_price(k) for k in ks2], "k--", lw=1.7, label="True digital price")
ax.set_xlabel("Digital strike")
ax.set_ylabel("Price bound")
ax.set_title("Model-free digital bounds narrow as option information grows")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_digital_bounds.png"), dpi=150)
plt.close(fig)

# Fig 3: extremal distributions that attain the K=120 bounds.
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(grid, p_true, color="#333333", lw=2, label="True implied distribution")
ax.plot(grid, p_up, color="#c23b22", lw=1.5, label="Upper-bound distribution")
ax.plot(grid, p_lo, color="#1764ab", lw=1.5, label="Lower-bound distribution")
ax.axvline(K, color="#777777", ls="--", lw=1)
ax.set_xlabel("Terminal underlying price")
ax.set_ylabel("Probability mass")
ax.set_title("Extremal martingale measures for the digital payoff")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_extremal_measures.png"), dpi=150)
plt.close(fig)

# Fig 4: how many quote strikes are needed.
quote_counts = list(range(3, 18, 2))
sparse_widths = []
sparse_lows = []
sparse_highs = []
for count in quote_counts:
    sub_strikes = np.linspace(60.0, 140.0, count)
    sub_quotes = [(float(k), call_price(k)) for k in sub_strikes]
    hi, low, _, _ = bounds(payoff, sub_quotes, use_moment=False)
    sparse_highs.append(hi)
    sparse_lows.append(low)
    sparse_widths.append(hi - low)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(quote_counts, sparse_widths, "o-", color="#7b3fa1", lw=2, label="Bound width")
ax.plot(quote_counts, sparse_highs, "s--", color="#c23b22", lw=1.3, label="Upper bound")
ax.plot(quote_counts, sparse_lows, "^--", color="#1764ab", lw=1.3, label="Lower bound")
ax.axhline(digital_price(K), color="#333333", ls=":", lw=1.5, label="True digital price")
ax.set_xlabel("Number of quoted strikes")
ax.set_ylabel("Price / width")
ax.set_title("Sparse quote sensitivity for the K=120 digital")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_quote_sparsity.png"), dpi=150)
plt.close(fig)

# Fig 5: quote noise can break feasibility or move the whole interval.
rng = np.random.default_rng(SEED)
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
noise_widths = []
noise_failures = []
noise_centers = []
for level in noise_levels:
    widths, centers, fails = [], [], 0
    for _ in range(20):
        noisy = [(float(k), float(c + rng.normal(0.0, level))) for k, c in zip(quote_strikes, quotes)]
        hi, low, _, _ = bounds(payoff, noisy, use_moment=False)
        if hi is None:
            fails += 1
            continue
        widths.append(hi - low)
        centers.append(0.5 * (hi + low))
    noise_widths.append(float(np.mean(widths)) if widths else 0.0)
    noise_centers.append(float(np.mean(np.abs(np.array(centers) - digital_price(K)))) if centers else 0.0)
    noise_failures.append(fails)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(noise_levels, noise_widths, "o-", color="#1a7f4b", lw=2, label="Mean bound width")
ax.plot(noise_levels, noise_centers, "s--", color="#c23b22", lw=1.7, label="Mean center error")
ax2 = ax.twinx()
ax2.bar(noise_levels, noise_failures, width=0.003, color="#7b3fa1", alpha=0.4, label="Infeasible runs")
ax2.set_ylabel("Infeasible count out of 20")
ax.set_xlabel("Quote noise standard deviation")
ax.set_ylabel("Price units")
ax.set_title("Noisy quotes widen and displace the no-arbitrage interval")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_quote_noise.png"), dpi=150)
plt.close(fig)

results = {
    "grid_points": len(grid),
    "S0": S0,
    "true_mean": mom1,
    "true_second_moment": mom2,
    "quote_strikes": [float(k) for k in quote_strikes],
    "quote_prices": [float(c) for c in quotes],
    "digital_strike": K,
    "true_digital_price": digital_price(K),
    "martingale_lower": d_lo,
    "martingale_upper": d_up,
    "martingale_width": d_up - d_lo,
    "quotes_lower": m_lo,
    "quotes_upper": m_up,
    "quotes_width": m_up - m_lo,
    "width_reduction_pct": 100.0 * (1.0 - (m_up - m_lo) / (d_up - d_lo)),
    "three_high_quotes_lower": b3_lo,
    "three_high_quotes_upper": b3_up,
    "three_high_quotes_width": b3_up - b3_lo,
    "moment_lower": t_lo,
    "moment_upper": t_up,
    "moment_width": t_up - t_lo,
    "extra_moment_reduction_pct": 100.0 * (1.0 - (t_up - t_lo) / (b3_up - b3_lo)),
    "quote_counts": quote_counts,
    "sparse_widths": [float(x) for x in sparse_widths],
    "sparse_lows": [float(x) for x in sparse_lows],
    "sparse_highs": [float(x) for x in sparse_highs],
    "noise_levels": noise_levels,
    "noise_mean_widths": noise_widths,
    "noise_mean_center_errors": noise_centers,
    "noise_infeasible_counts": noise_failures,
    "deep_itm_call_at_60": call_price(60.0),
    "deep_otm_call_at_140": call_price(140.0),
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    pass
