# -*- coding: utf-8 -*-
"""Variance-equality tests under heavy tails: Bartlett vs robust alternatives.
Self-contained: all Monte Carlo data, five real figures and results.json from one script.
"""
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(20261127)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "variance-equality-test-robust-heavy-tails")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
NREP = 1600


def draw(rng, dist, n, scale=1.0):
    """Draw n samples from a distribution with unit scale, then apply scale."""
    if dist == "normal":
        x = rng.normal(0.0, 1.0, n)
    elif dist == "t5":
        x = rng.standard_t(5, n) / np.sqrt(5.0 / 3.0)
    elif dist == "t3":
        x = rng.standard_t(3, n) / np.sqrt(3.0)
    elif dist == "lognormal":
        x = rng.lognormal(0.0, 0.8, n) / np.exp(0.32)
    else:
        raise ValueError(dist)
    return x * scale


def run_tests(groups):
    """Return p-values of four variance-equality tests on a list of samples."""
    return (
        stats.bartlett(*groups).pvalue,
        stats.levene(*groups, center="mean").pvalue,
        stats.levene(*groups, center="median").pvalue,   # Brown-Forsythe
        stats.fligner(*groups).pvalue,
    )


TESTS = ["bartlett", "levene_mean", "brown_forsythe", "fligner"]
DISTS = ["normal", "t5", "t3", "lognormal"]


def size_experiment(rng, dist, sizes):
    """False-rejection rate under H0 (all groups share the same scale)."""
    rej = np.zeros(4, dtype=int)
    for _ in range(NREP):
        groups = [draw(rng, dist, n) for n in sizes]
        ps = run_tests(groups)
        for j, p in enumerate(ps):
            if p < ALPHA:
                rej[j] += 1
    return (rej / NREP).astype(float)


def main():
    res = {}
    rng = RNG

    # exp1: size under equal sample sizes, four distributions
    sizes_bal = [60, 60, 60]
    size_bal = {}
    for d in DISTS:
        size_bal[d] = size_experiment(rng, d, sizes_bal)
    res["exp1_tests"] = TESTS
    res["exp1_dists"] = DISTS
    res["exp1_size"] = {d: [float(v) for v in size_bal[d]] for d in DISTS}
    res["exp1_bartlett_normal"] = float(size_bal["normal"][0])
    res["exp1_bartlett_t3"] = float(size_bal["t3"][0])
    res["exp1_bartlett_lognormal"] = float(size_bal["lognormal"][0])
    res["exp1_bf_t3"] = float(size_bal["t3"][2])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(DISTS))
    w = 0.2
    colors = ["#c0392b", "#e67e22", "#2c7fb8", "#27ae60"]
    for j, (t, c) in enumerate(zip(TESTS, colors)):
        vals = [size_bal[d][j] for d in DISTS]
        ax.bar(x + (j - 1.5) * w, vals, w, label=t, color=c)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0, label="nominal 5%")
    ax.set_xticks(x); ax.set_xticklabels(["normal", "t(5)", "t(3)", "lognormal"])
    ax.set_ylabel("false rejection rate"); ax.set_ylim(0, 0.75)
    ax.set_title("Bartlett size explodes on heavy tails; robust tests hold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_size.png"), dpi=130); plt.close()

    # exp2: unbalanced sample sizes amplify the distortion
    sizes_unb = [20, 60, 300]
    size_unb = {}
    for d in ["normal", "t3"]:
        size_unb[d] = size_experiment(rng, d, sizes_unb)
    res["exp2_sizes_balanced"] = sizes_bal
    res["exp2_sizes_unbalanced"] = sizes_unb
    res["exp2_size"] = {d: [float(v) for v in size_unb[d]] for d in ["normal", "t3"]}
    res["exp2_bartlett_t3_unbalanced"] = float(size_unb["t3"][0])
    res["exp2_bartlett_t3_balanced"] = float(size_bal["t3"][0])

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(4)
    w = 0.35
    for j, (t, c) in enumerate(zip(TESTS, colors)):
        vals = [size_bal["t3"][j], size_unb["t3"][j]]
        ax.bar(x[:2] + (j - 1.5) * w, vals, w, label=t, color=c)
    for j, (t, c) in enumerate(zip(TESTS, colors)):
        vals = [size_bal["normal"][j], size_unb["normal"][j]]
        ax.bar(x[2:] + (j - 1.5) * w, vals, w, color=c, alpha=0.45)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0)
    ax.set_xticks([0.5, 2.5]); ax.set_xticklabels(["t(3)", "normal (faded)"])
    ax.set_ylabel("false rejection rate"); ax.set_ylim(0, 0.8)
    ax.set_title("Unbalanced groups: does distortion grow or shrink?")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_unbalanced.png"), dpi=130); plt.close()

    # exp3: power under a真实 variance ratio, normal vs t(3)
    ratios = [1.0, 1.2, 1.4, 1.6, 2.0]
    power = {"normal": {t: [] for t in TESTS}, "t3": {t: [] for t in TESTS}}
    for d in ["normal", "t3"]:
        for r in ratios:
            cnt = np.zeros(4, dtype=int)
            for _ in range(NREP):
                groups = [draw(rng, d, 80, 1.0), draw(rng, d, 80, r), draw(rng, d, 80, r)]
                ps = run_tests(groups)
                for j, p in enumerate(ps):
                    if p < ALPHA:
                        cnt[j] += 1
            for j, t in enumerate(TESTS):
                power[d][t].append(float(cnt[j] / NREP))
    res["exp3_ratios"] = ratios
    res["exp3_power"] = power
    res["exp3_bartlett_normal_r10"] = power["normal"]["bartlett"][0]
    res["exp3_bartlett_t3_r10"] = power["t3"]["bartlett"][0]
    res["exp3_fligner_t3_r16"] = power["t3"]["fligner"][3]
    res["exp3_bartlett_t3_r16"] = power["t3"]["bartlett"][3]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, d in zip(axes, ["normal", "t3"]):
        for t, c in zip(TESTS, colors):
            ax.plot(ratios, power[d][t], "o-", label=t, color=c)
        ax.axhline(ALPHA, color="#333", ls="--", lw=0.8)
        ax.set_xlabel("true scale ratio (group 2&3 vs 1)")
        ax.set_title(f"Power, {d}")
    axes[0].set_ylabel("rejection rate (power)")
    axes[0].legend(fontsize=7)
    fig.suptitle("At ratio=1 Bartlett already rejects on t(3): not power, but size")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_power.png"), dpi=130); plt.close()

    # exp4: two strategies with equal volatility but different tails
    mis = np.zeros(4, dtype=int)
    for _ in range(NREP):
        a = rng.normal(0.0, 1.0, 200)
        b = rng.standard_t(3, 200) / np.sqrt(3.0)
        ps = run_tests([a, b])
        for j, p in enumerate(ps):
            if p < ALPHA and np.var(b) > np.var(a):
                mis[j] += 1
    mis_rate = (mis / NREP).astype(float)
    res["exp4_tests"] = TESTS
    res["exp4_misjudge"] = [float(v) for v in mis_rate]
    res["exp4_bartlett_misjudge"] = float(mis_rate[0])
    res["exp4_bf_misjudge"] = float(mis_rate[2])

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(TESTS, mis_rate, color=colors)
    for bar, v in zip(bars, mis_rate):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0, label="true 5%")
    ax.set_ylabel('rate of claiming "B more volatile"')
    ax.set_ylim(0, max(0.3, float(mis_rate.max()) * 1.2))
    ax.set_title("Equal volatility, heavier tails: how often each test cries wolf")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_tail_misjudge.png"), dpi=130); plt.close()

    # exp5: does the size distortion vanish as n grows?
    ns = [30, 60, 120, 240, 480]
    nsize = {t: [] for t in TESTS}
    for n in ns:
        s = size_experiment(rng, "t3", [n, n, n])
        for j, t in enumerate(TESTS):
            nsize[t].append(float(s[j]))
    res["exp5_ns"] = ns
    res["exp5_size"] = nsize
    res["exp5_bartlett_n30"] = nsize["bartlett"][0]
    res["exp5_bartlett_n480"] = nsize["bartlett"][-1]
    res["exp5_fligner_n480"] = nsize["fligner"][-1]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for t, c in zip(TESTS, colors):
        ax.plot(ns, nsize[t], "o-", label=t, color=c)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0, label="nominal 5%")
    ax.set_xscale("log"); ax.set_xlabel("per-group sample size n (t(3))")
    ax.set_ylabel("false rejection rate")
    ax.set_title("More data does not cure Bartlett under heavy tails")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_n.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
