# -*- coding: utf-8 -*-
"""Wilcoxon signed-rank vs paired t-test vs sign test, on paired strategy returns.

Single source of truth for every number and figure in the article. The paired
difference d_t = r_A,t - r_B,t is tested for a zero centre. Three tests are
compared on the same data:

  * paired t: uses the mean, divides by a sample SD that a single extreme day
    can inflate without bound;
  * sign test: uses only the count of positive differences, distribution-free
    for the median but throws away all magnitudes;
  * Wilcoxon signed-rank: rank |d_t|, attach the sign, sum positive ranks.
    Its asymptotic variance carries a tie correction.

Five truth-known experiments: (1) size and power under five null distributions
including heavy tails, contamination and skewness; (2) power versus sample
size; (3) one outlier injected at growing magnitude; (4) exact small-sample
null versus the normal approximation; (5) a concrete 120-day backtest where the
three tests disagree. A cross-check against scipy.stats.wilcoxon is included.
All seeds are fixed so exec replay equals results.json exactly.
"""
import json
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261124
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "wilcoxon-signed-rank-robust-test")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------- three tests
def wilcoxon_p(d):
    """Two-sided signed-rank p-value, normal approximation with tie correction."""
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    n = nz.size
    if n < 2:
        return 1.0
    a = np.abs(nz)
    r = stats.rankdata(a)
    w_plus = r[nz > 0].sum()
    mu = n * (n + 1) / 4.0
    _, cnt = np.unique(a, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - np.sum(cnt ** 3 - cnt) / 48.0
    z = (w_plus - mu) / np.sqrt(max(var, 1e-12))
    return float(2.0 * stats.norm.sf(abs(z)))


def wilcoxon_stat(d):
    """Return (W+, n, mu, variance) used by the exact-size experiment."""
    nz = np.asarray(d, dtype=float)
    nz = nz[nz != 0]
    n = nz.size
    a = np.abs(nz)
    r = stats.rankdata(a)
    w = float(r[nz > 0].sum())
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    return w, n, mu, var


def sign_p(d):
    """Two-sided exact binomial sign test on non-zero differences."""
    nz = np.asarray(d, dtype=float)
    nz = nz[nz != 0]
    n = nz.size
    if n == 0:
        return 1.0
    k = int((nz > 0).sum())
    lower = stats.binom.cdf(k, n, 0.5)
    upper = stats.binom.sf(k - 1, n, 0.5)
    return float(min(1.0, 2.0 * min(lower, upper)))


def t_p(d):
    """Two-sided paired t-test p-value."""
    d = np.asarray(d, dtype=float)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    if se <= 0:
        return 1.0
    t = d.mean() / se
    return float(2.0 * stats.t.sf(abs(t), n - 1))


def exact_wilcoxon_crit(n, alpha=0.05):
    """Exact lower critical sum for W+ over all 2^n sign patterns (no ties)."""
    S = n * (n + 1) // 2
    dp = np.zeros(S + 1, dtype=np.float64)
    dp[0] = 1.0
    for r in range(1, n + 1):
        new = dp.copy()
        new[r:] += dp[:S + 1 - r]
        dp = new
    cdf = np.cumsum(dp) / (2.0 ** n)
    below = np.where(cdf <= alpha / 2.0 + 1e-12)[0]
    lower = int(below[-1]) if below.size else 0
    mu = S / 2.0
    return lower, float(mu - lower)


# ------------------------------------------------------------- DGP helpers
def base_noise(kind, rng, n):
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "t3":
        return stats.t.rvs(3, size=n, random_state=rng) / np.sqrt(3.0)
    if kind == "laplace":
        return stats.laplace.rvs(size=n, random_state=rng) / np.sqrt(2.0)
    if kind == "contaminated":
        z = rng.standard_normal(n)
        big = rng.standard_normal(n) * 5.0
        m = rng.random(n) < 0.01
        z[m] = big[m]
        return z / np.sqrt(0.99 + 0.01 * 25.0)
    if kind == "skewed":
        # centred exponential: median exactly zero, mean 1-log(2), right skew
        return rng.exponential(scale=1.0, size=n) - np.log(2.0)
    raise ValueError(kind)


# ------------------------------------------------------------- experiments
def exp_distributions(reps=3000, n=100):
    kinds = ["normal", "t3", "laplace", "contaminated", "skewed"]
    out = {}
    for j, kind in enumerate(kinds):
        rng = np.random.default_rng(SEED + 100 + j)
        for delta in (0.0, 0.25):
            e = base_noise(kind, rng, n * reps).reshape(reps, n)
            x = e + delta
            key = kind + ("_size" if delta == 0.0 else "_power")
            out[key] = {
                "delta": delta,
                "t": float(np.mean([t_p(row) < 0.05 for row in x])),
                "sign": float(np.mean([sign_p(row) < 0.05 for row in x])),
                "wilcoxon": float(np.mean([wilcoxon_p(row) < 0.05 for row in x])),
            }
    # independent cross-check of our p-value against scipy on one fixed sample
    rng = np.random.default_rng(SEED + 777)
    d = base_noise("normal", rng, 60) + 0.3
    sp = float(stats.wilcoxon(d).pvalue)
    out["scipy_check"] = {"ours": wilcoxon_p(d), "scipy": sp,
                          "abs_diff": abs(wilcoxon_p(d) - sp)}
    return out


def exp_sample_size(reps=2000, deltas=(0.35,)):
    out = {}
    for j, kind in enumerate(("normal", "t3")):
        rng = np.random.default_rng(SEED + 300 + j)
        grid = {}
        for n in (20, 40, 80, 160, 320):
            e = base_noise(kind, rng, n * reps).reshape(reps, n)
            x = e + deltas[0]
            grid[str(n)] = {
                "t": float(np.mean([t_p(row) < 0.05 for row in x])),
                "sign": float(np.mean([sign_p(row) < 0.05 for row in x])),
                "wilcoxon": float(np.mean([wilcoxon_p(row) < 0.05 for row in x])),
            }
        out[kind] = {"delta": deltas[0], "grid": grid}
    return out


def exp_outlier(reps=400, n=200):
    rng = np.random.default_rng(SEED + 400)
    clean = 0.20 + rng.standard_normal((reps, n))
    rows = {}
    for M in (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        x = clean.copy()
        x[:, 0] -= M
        tp = np.array([t_p(r) for r in x])
        wp = np.array([wilcoxon_p(r) for r in x])
        sp = np.array([sign_p(r) for r in x])
        rows[str(M)] = {
            "t_p_mean": float(tp.mean()),
            "t_detect": float(np.mean(tp < 0.05)),
            "wilcoxon_p_mean": float(wp.mean()),
            "wilcoxon_detect": float(np.mean(wp < 0.05)),
            "sign_detect": float(np.mean(sp < 0.05)),
        }
    return rows


def exp_exact_small_n(reps=6000):
    rows = {}
    for n in (8, 12, 16, 20, 30):
        lower, c_exact = exact_wilcoxon_crit(n)
        c_asym = 1.959963984540054 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        rng = np.random.default_rng(SEED + 500 + n)
        x = rng.standard_normal((reps, n))
        sz_exact = []
        sz_asym = []
        for row in x:
            w, nn, mu, var = wilcoxon_stat(row)
            sz_exact.append(abs(w - mu) >= c_exact - 1e-9)
            sz_asym.append(abs(w - mu) >= c_asym)
        rows[str(n)] = {
            "lower_critical_sum": lower,
            "c_exact": c_exact,
            "c_asymptotic": float(c_asym),
            "size_exact": float(np.mean(sz_exact)),
            "size_asymptotic": float(np.mean(sz_asym)),
        }
    return rows


def exp_live_backtest(n=120):
    rng = np.random.default_rng(SEED + 900)
    rb = 0.0003 + 0.012 * rng.standard_normal(n)
    ra = rb + 0.0035 + 0.011 * rng.standard_normal(n)
    d = ra - rb
    d[47] -= 0.09  # one bad day for strategy A
    cum = np.cumsum(d)
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "sd_diff": float(d.std(ddof=1)),
        "t_p": t_p(d),
        "sign_p": sign_p(d),
        "wilcoxon_p": wilcoxon_p(d),
        "positive_days": int((d > 0).sum()),
        "worst_day_index": int(np.argmin(d)),
        "worst_day": float(d.min()),
        "final_cum": float(cum[-1]),
        "cum": cum.tolist(),
        "diffs": d.tolist(),
    }


# ------------------------------------------------------------------ figures
def fig1_distributions(res):
    kinds = ["normal", "t3", "laplace", "contaminated", "skewed"]
    labels = ["normal", "t(3)", "Laplace", "contam 1%", "skewed"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    w = 0.26
    xs = np.arange(len(kinds))
    methods = [("t", "#b23a2b"), ("sign", "#2b8cbe"), ("wilcoxon", "#3a923a")]
    for panel, suffix, target in ((0, "_size", 0.05), (1, "_power", 0.05)):
        for i, (m, col) in enumerate(methods):
            vals = [res[k + suffix][m] for k in kinds]
            ax[panel].bar(xs + (i - 1) * w, vals, w, label=m, color=col, alpha=0.88)
        ax[panel].axhline(target, color="#333", ls="--", lw=1.2)
        ax[panel].set_xticks(xs)
        ax[panel].set_xticklabels(labels, fontsize=9)
        ax[panel].set_ylim(0, 1.02)
        ax[panel].set_title("size (delta=0): dashed = 5%" if panel == 0
                            else "power (delta=0.25)")
        ax[panel].legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_distributions.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_distributions.png"


def fig2_sample_size(res):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for j, (kind, title) in enumerate((("normal", "normal noise"),
                                       ("t3", "t(3) noise"))):
        grid = res[kind]["grid"]
        ns = sorted(int(k) for k in grid)
        for m, col in (("t", "#b23a2b"), ("sign", "#2b8cbe"),
                       ("wilcoxon", "#3a923a")):
            ax[j].plot(ns, [grid[str(n)][m] for n in ns], "o-",
                       label=m, color=col, lw=1.8)
        ax[j].set_title(title + "  delta=0.35")
        ax[j].set_xlabel("sample size n")
        ax[j].set_ylim(0, 1.02)
        ax[j].legend(fontsize=9)
    ax[0].set_ylabel("rejection rate at 5%")
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_sample_size.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_sample_size.png"


def fig3_outlier(rows):
    ms = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for key, lab, col in (("t_p_mean", "paired t (mean p)", "#b23a2b"),
                          ("wilcoxon_p_mean", "Wilcoxon (mean p)", "#3a923a"),
                          ("sign_p_mean", "sign (mean p)", "#2b8cbe")):
        if key in rows[str(ms[0])]:
            ax.plot(ms, [rows[str(m)][key] for m in ms], "o-",
                    label=lab, color=col, lw=1.8)
    ax.plot(ms, [rows[str(m)]["t_detect"] for m in ms], "s--",
            color="#b23a2b", alpha=0.5, label="t detection rate")
    ax.plot(ms, [rows[str(m)]["wilcoxon_detect"] for m in ms], "s--",
            color="#3a923a", alpha=0.5, label="Wilcoxon detection rate")
    ax.axhline(0.05, color="#333", ls=":", lw=1.2)
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlabel("injected outlier magnitude M (one day out of 200)")
    ax.set_ylabel("mean p-value / detection rate")
    ax.set_title("One bad day: t loses the signal, signed-rank does not")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_outlier.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_outlier.png"


def fig4_exact(rows):
    ns = sorted(int(k) for k in rows)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(ns, [rows[str(n)]["size_exact"] for n in ns], "o-",
            color="#3a923a", lw=1.8, label="exact critical value")
    ax.plot(ns, [rows[str(n)]["size_asymptotic"] for n in ns], "o-",
            color="#b23a2b", lw=1.8, label="normal approximation 1.96 sigma")
    ax.axhline(0.05, color="#333", ls="--", lw=1.2, label="nominal 5%")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("empirical size")
    ax.set_title("Small samples: exact null vs normal approximation")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_exact.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_exact.png"


def fig5_live(live):
    d = np.array(live["diffs"])
    cum = np.array(live["cum"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(np.arange(1, len(cum) + 1), cum, color="#2b8cbe", lw=1.5)
    ax[0].axvline(live["worst_day_index"] + 1, color="#b23a2b", ls="--", lw=1.3)
    ax[0].annotate("one -9% day", xy=(live["worst_day_index"] + 1,
                                      cum[live["worst_day_index"]]),
                   xytext=(18, 8), textcoords="offset points", fontsize=9,
                   color="#b23a2b")
    ax[0].set_xlabel("trading day")
    ax[0].set_ylabel("cumulative daily difference")
    ax[0].set_title("Strategy A minus B: cumulative edge")
    names = ["paired t", "sign", "Wilcoxon"]
    vals = [live["t_p"], live["sign_p"], live["wilcoxon_p"]]
    cols = ["#b23a2b", "#2b8cbe", "#3a923a"]
    ax[1].bar(names, vals, color=cols, alpha=0.88)
    ax[1].axhline(0.05, color="#333", ls="--", lw=1.2)
    ax[1].set_ylabel("two-sided p-value")
    ax[1].set_title("Same 120 days, three verdicts")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.012, "%.4f" % v, ha="center", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_live.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_live.png"


def main():
    dist = exp_distributions()
    size = exp_sample_size()
    out = exp_outlier()
    exact = exp_exact_small_n()
    live = exp_live_backtest()
    r = {
        "setup": {"seed": SEED, "alpha": 0.05, "reps_distributions": 3000,
                  "n_distributions": 100, "reps_size": 2000,
                  "reps_outlier": 400, "n_outlier": 200,
                  "reps_exact": 6000},
        "distributions": dist,
        "sample_size": size,
        "outlier": out,
        "exact": exact,
        "live": live,
    }
    r["figs"] = [fig1_distributions(dist), fig2_sample_size(size),
                 fig3_outlier(out), fig4_exact(exact), fig5_live(live)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
