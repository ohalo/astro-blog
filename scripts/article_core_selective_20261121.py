# -*- coding: utf-8 -*-
"""Winner's curse and post-selection inference: scanning p signals, reporting
the top one, and the collapse of naive inference.

Single source of truth for every number and figure in the article.

Setup (known variance sigma=1, p candidate signals):
    theta_hat_j = theta_j + z_j,   z_j ~ N(0,1) iid
Researcher scans all p, picks w = argmax theta_hat_j, and reports that signal.

Analytic baselines (iid standard normal):
    P(max_j Z_j <= t) = Phi(t)^p
    E[max |Z_j|]      = integral_0^inf t d[(2 Phi(t)-1)^p]
Four inference rules for the selected winner (one-sided, alpha=0.05):
    naive    : reject if t > z_{0.95}                       (size -> 1 as p grows)
    bonferroni: reject if p*(1-Phi(t)) < 0.05               (size = 0.05, conservative)
    selective: reject if 1-Phi(t)^p < 0.05                  (size = 0.05, exact)
    split    : select on one half, test on the other half   (size = 0.05, costs data)
Known-variance selective CI inverts the truncated-normal p-value conditional
on the selection event {theta_hat_w > m}, m = runner-up value.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, special

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261121
REPS_SIZE = 6000
REPS_COVER = 1500
REPS_DIST = 8000
REPS_POWER = 4000
P_GRID = [5, 10, 20, 50, 100, 200, 500]
Z95 = float(special.ndtri(0.95))
ALPHA = 0.05

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "selective-inference-winner-curse")
os.makedirs(_OUT, exist_ok=True)


def _cdf_max(t, p):
    return special.ndtr(t) ** p


def analytic_expected_absmax(p):
    """E[max |Z|] for p iid N(0,1): integrate t * d[(2Phi(t)-1)^p].
    Density of max|Z| = p*(2Phi(t)-1)^{p-1} * 2 phi(t)."""
    pdf = lambda t: p * (2 * special.ndtr(t) - 1) ** (p - 1) * 2 * np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
    val, _ = integrate.quad(lambda t: t * pdf(t), 0, 12, limit=300)
    return float(val)


def exp_curse():
    """Selection bias: expected top-|theta_hat| under pure noise, MC vs analytic."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for p in P_GRID:
        draws = np.abs(rng.standard_normal((REPS_SIZE, p))).max(axis=1)
        mc = float(draws.mean())
        an = analytic_expected_absmax(p)
        rows.append({"p": p, "mc": mc, "analytic": an,
                     "naive_z": Z95, "excess_vs_1_96": mc - 1.96})
    return {"reps": REPS_SIZE, "rows": rows}


def exp_size():
    """False-discovery size of four rules under the global null theta=0."""
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for p in P_GRID:
        m = rng.standard_normal((REPS_SIZE, p)).max(axis=1)
        naive = float((m > Z95).mean())
        bonf = float(((1 - special.ndtr(m)) < ALPHA / p).mean())
        sel = float(((1 - special.ndtr(m) ** p) < ALPHA).mean())
        # split: select winner on first half signals, test it on an independent replicate
        half = p
        m1 = rng.standard_normal((REPS_SIZE, half)).max(axis=1)
        test = rng.standard_normal(REPS_SIZE)  # independent measurement of selected index's theta
        split = float((test > Z95).mean())
        rows.append({"p": p, "naive": naive, "bonferroni": bonf,
                     "selective": sel, "split": split,
                     "naive_analytic": 1 - special.ndtr(Z95) ** p})
    return {"reps": REPS_SIZE, "alpha": ALPHA, "rows": rows}


def exp_coverage(theta_star=0.8):
    """Sparse truth: index 0 has theta*, rest 0. Select top-1 on sample A, then
    compare two 95% intervals for the selected index's true effect:
      naive : theta_hat_A[w] +/- 1.96   (same data used to select -> biased)
      split : theta_hat_B[w] +/- 1.96   (fresh independent sample -> honest)
    Coverage of the TRUE effect of the selected index (0 if a noise index won)."""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for p in [10, 20, 50, 100, 200]:
        naive_cov = split_cov = 0
        naive_width_sum = split_width_sum = 0.0
        for _ in range(REPS_COVER):
            th = np.zeros(p)
            th[0] = theta_star
            xa = th + rng.standard_normal(p)
            xb = th + rng.standard_normal(p)
            w = int(np.argmax(xa))
            true_w = theta_star if w == 0 else 0.0
            ta, tb = xa[w], xb[w]
            if ta - 1.96 <= true_w <= ta + 1.96:
                naive_cov += 1
            if tb - 1.96 <= true_w <= tb + 1.96:
                split_cov += 1
            naive_width_sum += 2 * 1.96
            split_width_sum += 2 * 1.96
        rows.append({"p": p,
                     "naive_cov": naive_cov / REPS_COVER,
                     "split_cov": split_cov / REPS_COVER,
                     "naive_width": naive_width_sum / REPS_COVER,
                     "split_width": split_width_sum / REPS_COVER})
    return {"reps": REPS_COVER, "theta_star": theta_star, "rows": rows}


def exp_distribution(p=50, theta_star=1.0):
    """Distribution of the reported estimate: naive full-sample vs split-sample."""
    rng = np.random.default_rng(SEED + 4)
    naive_vals, split_vals, chosen_true = [], [], []
    half = max(2, p // 2)
    for _ in range(REPS_DIST):
        th = np.zeros(p)
        th[0] = theta_star
        x = th + rng.standard_normal(p)
        w = int(np.argmax(x))
        naive_vals.append(float(x[w]))
        chosen_true.append(theta_star if w == 0 else 0.0)
        # independent split on a fresh sample of same design
        xs = th + rng.standard_normal(p)
        split_vals.append(float(xs[w]))
    return {"reps": REPS_DIST, "p": p, "theta_star": theta_star,
            "naive": naive_vals, "split": split_vals, "chosen_true": chosen_true,
            "naive_mean": float(np.mean(naive_vals)),
            "split_mean": float(np.mean(split_vals)),
            "sel_true_signal_frac": float(np.mean(np.array(chosen_true) == theta_star))}


def exp_power(p=50):
    """Reliable discovery rate vs effect size: naive claim vs selective claim."""
    rng = np.random.default_rng(SEED + 5)
    theta_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for ts in theta_grid:
        naive_sig = sel_sig = true_pos = 0
        for _ in range(REPS_POWER):
            th = np.zeros(p)
            th[0] = ts
            x = th + rng.standard_normal(p)
            w = int(np.argmax(x))
            t = x[w]
            if t > Z95:
                naive_sig += 1
            if (1 - special.ndtr(t) ** p) < ALPHA:
                sel_sig += 1
            if w == 0:
                true_pos += 1
        rows.append({"theta": ts,
                     "naive_claim": naive_sig / REPS_POWER,
                     "selective_claim": sel_sig / REPS_POWER,
                     "true_selected": true_pos / REPS_POWER})
    return {"reps": REPS_POWER, "p": p, "rows": rows}


def plot1(curse):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ps = [r["p"] for r in curse["rows"]]
    mc = [r["mc"] for r in curse["rows"]]
    an = [r["analytic"] for r in curse["rows"]]
    ax.plot(ps, mc, "o-", color="#c0392b", lw=2.2, label="MC: E[max |theta_hat|]")
    ax.plot(ps, an, "s--", color="#2c3e50", lw=1.8, label="analytic integral")
    ax.axhline(1.96, color="#27ae60", ls=":", lw=1.8, label="naive threshold 1.96")
    ax.axhline(0.0, color="gray", lw=1, label="true effect = 0")
    ax.set_xscale("log")
    ax.set_xlabel("number of scanned signals p")
    ax.set_ylabel("reported top-|theta_hat|")
    ax.set_title("Fig1: scanning more noise manufactures larger 'effects'")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig1_selection_bias.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig1_selection_bias.png"


def plot2(size):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ps = [r["p"] for r in size["rows"]]
    ax.plot(ps, [r["naive"] for r in size["rows"]], "o-", color="#c0392b", lw=2.2, label="naive z-test")
    ax.plot(ps, [r["bonferroni"] for r in size["rows"]], "s-", color="#2980b9", lw=1.8, label="Bonferroni")
    ax.plot(ps, [r["selective"] for r in size["rows"]], "^--", color="#27ae60", lw=1.8, label="selective (exact)")
    ax.plot(ps, [r["split"] for r in size["rows"]], "d:", color="#8e44ad", lw=1.8, label="sample splitting")
    ax.axhline(0.05, color="k", ls=":", lw=1.5, label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xlabel("number of scanned signals p")
    ax.set_ylabel("false rejection under global null")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Fig2: naive size -> 1; the three corrections hold 5%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig2_size.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig2_size.png"


def plot3(cover):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ps = [r["p"] for r in cover["rows"]]
    ax.plot(ps, [r["naive_cov"] for r in cover["rows"]], "o-", color="#c0392b", lw=2.2, label="naive 95% CI (same data)")
    ax.plot(ps, [r["split_cov"] for r in cover["rows"]], "s-", color="#27ae60", lw=2.2, label="split-sample 95% CI")
    ax.axhline(0.95, color="k", ls=":", lw=1.5, label="nominal 95%")
    ax.set_xscale("log")
    ax.set_xlabel("number of scanned signals p")
    ax.set_ylabel("coverage of selected signal")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Fig3: naive coverage collapses; split-sample CI stays near 95%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig3_coverage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig3_coverage.png"


def plot4(dist):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bins = np.linspace(-1.5, 4.5, 60)
    ax.hist(dist["split"], bins=bins, density=True, alpha=0.6, color="#8e44ad", label="split-sample estimate")
    ax.hist(dist["naive"], bins=bins, density=True, alpha=0.6, color="#c0392b", label="naive reported top")
    ax.axvline(dist["theta_star"], color="k", ls="--", lw=1.6,
               label="theta* = %.1f" % dist["theta_star"])
    ax.axvline(dist["naive_mean"], color="#c0392b", ls=":", lw=1.6,
               label="naive mean %.2f" % dist["naive_mean"])
    ax.set_xlabel("reported effect size")
    ax.set_ylabel("density")
    ax.set_title("Fig4: naive estimate is right-skewed by the selection gate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig4_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig4_distribution.png"


def plot5(power):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["theta"] for r in power["rows"]]
    ax.plot(xs, [r["naive_claim"] for r in power["rows"]], "o-", color="#c0392b", lw=2.2, label="naive 'significant' claim")
    ax.plot(xs, [r["selective_claim"] for r in power["rows"]], "s-", color="#27ae60", lw=2.2, label="selective reliable claim")
    ax.plot(xs, [r["true_selected"] for r in power["rows"]], "^--", color="#2980b9", lw=1.8, label="true signal actually picked")
    ax.axhline(0.05, color="k", ls=":", lw=1.3, label="nominal 5%")
    ax.set_xlabel("true effect size theta* of the one real signal")
    ax.set_ylabel("rate over replications")
    ax.set_title("Fig5: naive 'power' at theta=0 is pure false discovery")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig5_power.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig5_power.png"


def main():
    curse = exp_curse()
    size = exp_size()
    cover = exp_coverage()
    dist = exp_distribution()
    power = exp_power()
    results = {
        "setup": {"seed": SEED, "z95": Z95, "alpha": ALPHA,
                  "reps_size": REPS_SIZE, "reps_cover": REPS_COVER,
                  "reps_dist": REPS_DIST, "reps_power": REPS_POWER,
                  "p_grid": P_GRID},
        "curse": curse,
        "size": size,
        "coverage": cover,
        "distribution": {k: v for k, v in dist.items() if k not in ("naive", "split", "chosen_true")},
        "power": power,
    }
    results["figs"] = [plot1(curse), plot2(size), plot3(cover), plot4(dist), plot5(power)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
