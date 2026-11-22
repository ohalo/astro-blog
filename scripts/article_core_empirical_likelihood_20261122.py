# -*- coding: utf-8 -*-
"""Empirical likelihood for a mean: Wilks calibration, coverage and power.

Single source of truth for every number and figure in the article.

The empirical likelihood (EL) ratio for a candidate mean mu is
    R(mu) = max prod_i p_i  s.t. sum p_i = 1, sum p_i x_i = mu,
whose Lagrange solution satisfies sum (x_i-mu)/(1+lam*(x_i-mu)) = 0.
Wilks' theorem gives  w(mu) = -2 log R(mu) -> chi2(1) at the true mu,
so a 95% region needs no variance estimate: {mu : w(mu) <= chi2_1(0.95)}.

We compare EL with the Wald t-interval and the percentile bootstrap on
three truth-known distributions (normal, shifted exponential = skew,
standardized t(5) = heavy tail), four sample sizes, and a power experiment.
All randomness uses one fixed seed; every number printed is recomputed.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261122
CONF = 0.95
Q = float(stats.chi2.ppf(CONF, 1))
Z = float(stats.norm.ppf(0.5 + CONF / 2.0))
R = 320            # Monte Carlo replications per cell
G = 61             # mu grid points for interval extraction
B_BOOT = 199       # bootstrap resamples per replication
NS = [30, 60, 120, 240]
SIZES = {"normal": 1.0, "skew": 1.0, "t5": 1.0}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "empirical-likelihood-wilks-inference")
os.makedirs(_OUT, exist_ok=True)


# ------------------------------------------------------------------ EL solver
def solve_lambda(d):
    """Solve sum(d/(1+lam*d)) = 0 with 1+lam*d > 0, vectorised on leading axes."""
    d = np.asarray(d, dtype=float)
    mx = np.max(d, axis=-1)
    mn = np.min(d, axis=-1)
    lo = -1.0 / np.where(mx > 0, mx, np.inf)
    hi = -1.0 / np.where(mn < 0, mn, -np.inf)
    lam = np.zeros_like(lo)
    for _ in range(32):
        z = 1.0 + lam[..., None] * d
        g = np.sum(d / z, axis=-1)
        gp = -np.sum((d / z) ** 2, axis=-1)
        mid = 0.5 * (lo + hi)
        cand = lam - g / gp
        bad = (~np.isfinite(cand)) | (cand <= lo) | (cand >= hi)
        nxt = np.where(bad, mid, cand)
        lo = np.where(g > 0, lam, lo)
        hi = np.where(g <= 0, lam, hi)
        eps = 1e-12 * (hi - lo)
        lam = np.minimum(np.maximum(nxt, lo + eps), hi - eps)
    return lam


def el_stat(x, mu):
    """w(mu)=-2 sum log(1+lam*(x-mu)); x (...,n), mu (...,) or scalar."""
    x = np.asarray(x, dtype=float)
    d = x - np.asarray(mu)[..., None]
    lam = solve_lambda(d)
    # R(mu)/R(muhat)=prod 1/(1+lam*d); AM-GM => prod(1+lam*d) >= 1, so w >= 0
    return 2.0 * np.sum(np.log1p(lam[..., None] * d), axis=-1)


def el_interval(x, conf=CONF, span=5.0):
    """Interval from a sample-specific mu grid; returns (lo, hi, width, missing)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    muhat = x.mean(axis=-1)
    se = x.std(axis=-1, ddof=1) / np.sqrt(n)
    grid = muhat[..., None] + np.linspace(-span, span, G)[None, :] * se[..., None]
    w = el_stat(x[:, None, :], grid)
    ok = w <= Q
    idx = np.argmin(w, axis=-1)
    lo_i = np.full(x.shape[:-1], -1, dtype=int)
    hi_i = np.full(x.shape[:-1], -1, dtype=int)
    for i in range(len(lo_i)):
        good = np.flatnonzero(ok[i])
        if good.size:
            lo_i[i] = good[0]
            hi_i[i] = good[-1]
    missing = int(np.sum(lo_i < 0))
    lo = np.where(lo_i >= 0, np.take_along_axis(grid, lo_i[..., None], -1)[..., 0], np.nan)
    hi = np.where(hi_i >= 0, np.take_along_axis(grid, hi_i[..., None], -1)[..., 0], np.nan)
    return lo, hi, hi - lo, missing


def boot_interval(x, conf=CONF):
    """Percentile bootstrap interval for the mean, x shape (R,n)."""
    rng = np.random.default_rng(SEED + 77)
    lo, hi = [], []
    for start in range(0, len(x), 80):
        chunk = x[start:start + 80]
        idx = rng.integers(0, chunk.shape[1], size=(len(chunk), B_BOOT, chunk.shape[1]))
        bm = chunk[np.arange(len(chunk))[:, None, None], idx].mean(axis=-1)
        a = (1 - conf) / 2
        lo.append(np.quantile(bm, a, axis=-1))
        hi.append(np.quantile(bm, 1 - a, axis=-1))
    lo = np.concatenate(lo)
    return lo, np.concatenate(hi)


def sample(scenario, n, rng):
    if scenario == "normal":
        return rng.standard_normal(n)
    if scenario == "skew":
        return rng.exponential(1.0, n) - 1.0   # mean 0, right-skew
    if scenario == "t5":
        z = rng.standard_t(5, size=n)
        return z / np.sqrt(5.0 / 3.0)          # mean 0, unit variance
    raise ValueError(scenario)


def truth(scenario):
    return 0.0


# ------------------------------------------------------------------ experiments
def exp_wilks():
    """Distribution of w(mu_true) against chi2(1): size and Bartlett factor."""
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for scenario in ["normal", "skew", "t5"]:
        for n in [30, 120]:
            x = np.stack([sample(scenario, n, rng) for _ in range(600)])
            w = el_stat(x, np.full(len(x), truth(scenario)))
            q95 = float(np.quantile(w, 0.95))
            out[f"{scenario}_n{n}"] = {
                "e_w": round(float(w.mean()), 6),
                "bartlett_factor": round(float(w.mean() / Q), 6),
                "q50": round(float(np.quantile(w, 0.50)), 6),
                "q90": round(float(np.quantile(w, 0.90)), 6),
                "q95": round(q95, 6),
                "q99": round(float(np.quantile(w, 0.99)), 6),
                "size_el": round(float(np.mean(w > Q)), 6),
            }
    return {"chi2_1_q95": round(Q, 6), "cells": out, "chi2_ref": {
        "q50": round(float(stats.chi2.ppf(0.5, 1)), 6),
        "q90": round(float(stats.chi2.ppf(0.9, 1)), 6),
        "q95": round(Q, 6),
        "q99": round(float(stats.chi2.ppf(0.99, 1)), 6),
    }}


def exp_coverage():
    rng = np.random.default_rng(SEED + 2)
    out = {}
    for scenario in ["normal", "skew", "t5"]:
        for n in NS:
            x = np.stack([sample(scenario, n, rng) for _ in range(R)])
            mu = truth(scenario)
            elo, ehi, ew, missing = el_interval(x)
            se = x.std(axis=-1, ddof=1) / np.sqrt(n)
            m = x.mean(axis=-1)
            wlo, whi = m - Z * se, m + Z * se
            blo, bhi = boot_interval(x)
            cov_el = np.mean((elo <= mu) & (mu <= ehi))
            cov_w = np.mean((wlo <= mu) & (mu <= whi))
            cov_b = np.mean((blo <= mu) & (mu <= bhi))
            out[f"{scenario}_n{n}"] = {
                "el": round(float(cov_el), 6),
                "wald": round(float(cov_w), 6),
                "boot": round(float(cov_b), 6),
                "el_width": round(float(np.nanmean(ew)), 6),
                "wald_width": round(float(np.mean(whi - wlo)), 6),
                "boot_width": round(float(np.mean(bhi - blo)), 6),
                "el_missing_grid": missing,
            }
    return out


def exp_shape():
    """One representative sample per distribution: the EL ratio curve is data-shaped."""
    rng = np.random.default_rng(SEED + 3)
    out = {}
    for scenario in ["normal", "skew", "bimodal"]:
        n = 60
        if scenario == "bimodal":
            x = np.concatenate([rng.normal(-1.2, 0.45, n // 2), rng.normal(1.2, 0.45, n // 2)])
        else:
            x = sample(scenario, n, rng)
        muhat = float(x.mean())
        se = float(x.std(ddof=1) / np.sqrt(n))
        grid = np.linspace(muhat - 3.2 * se, muhat + 3.2 * se, 241)
        w = el_stat(np.broadcast_to(x, (len(grid), n)), grid)
        inside = w <= Q
        if np.any(inside):
            gi = np.flatnonzero(inside)
            lo, hi = float(grid[gi[0]]), float(grid[gi[-1]])
        else:
            lo, hi = float("nan"), float("nan")
        out[scenario] = {
            "mean": round(muhat, 6), "se": round(se, 6),
            "el_lo": round(lo, 6), "el_hi": round(hi, 6),
            "wald_lo": round(muhat - Z * se, 6), "wald_hi": round(muhat + Z * se, 6),
            "left_len": round(muhat - lo, 6), "right_len": round(hi - muhat, 6),
            "grid": [round(float(v), 8) for v in grid[::6]],
            "w": [round(float(v), 8) for v in w[::6]],
        }
    return out


def exp_asymmetry():
    """Average left/right arm lengths for the right-skewed distribution."""
    rng = np.random.default_rng(SEED + 4)
    from scipy.stats import norm
    out = {}
    for n in [30, 60, 120]:
        x = np.stack([sample("skew", n, rng) for _ in range(R)])
        lo, hi, wd, missing = el_interval(x)
        m = x.mean(axis=-1)
        se = x.std(axis=-1, ddof=1) / np.sqrt(n)
        blo, bhi = boot_interval(x)
        ratio_el = float(np.nanmean((hi - m) / (m - lo)))
        ratio_b = float(np.nanmean((bhi - m) / (m - blo)))
        out[f"n{n}"] = {
            "el_left": round(float(np.nanmean(m - lo)), 6),
            "el_right": round(float(np.nanmean(hi - m)), 6),
            "el_right_left_ratio": round(ratio_el, 6),
            "boot_right_left_ratio": round(ratio_b, 6),
            "wald_symmetric": True,
            "el_missing_grid": missing,
        }
    return out


def exp_power():
    rng = np.random.default_rng(SEED + 5)
    deltas = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]
    out = {}
    for scenario in ["normal", "skew"]:
        rows = {}
        for d in deltas:
            x = np.stack([sample(scenario, 100, rng) + d for _ in range(R)])
            elo, ehi, _, _ = el_interval(x)
            m = x.mean(axis=-1)
            se = x.std(axis=-1, ddof=1) / np.sqrt(100)
            wlo, whi = m - Z * se, m + Z * se
            blo, bhi = boot_interval(x)
            rows[f"{d:+.2f}"] = {
                "el": round(float(np.nanmean((elo > 0) | (ehi < 0))), 6),
                "wald": round(float(np.mean((wlo > 0) | (whi < 0))), 6),
                "boot": round(float(np.mean((blo > 0) | (bhi < 0))), 6),
            }
        out[scenario] = rows
    return {"delta": deltas, "cells": out}


# ------------------------------------------------------------------ figures
def plot_shape(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, (name, label) in zip(ax, [("normal", "Normal"), ("skew", "Shifted exponential"),
                                     ("bimodal", "Bimodal mixture")]):
        d = e[name]
        a.plot(d["grid"], d["w"], color="#b23a2b", lw=2)
        a.axhline(Q, color="#2b5d8a", ls="--", lw=1.5, label=f"chi2(1) 95% = {Q:.3f}")
        a.axvline(0, color="0.6", lw=1)
        a.set_title(label + ": data-shaped EL ratio")
        a.set_xlabel("candidate mean mu"); a.set_ylabel("w(mu)")
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_el_shape.png"), dpi=115)
    plt.close(fig); return "fig1_el_shape.png"


def plot_wilks(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ref = e["chi2_ref"]
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    keys = [("normal_n30", "Normal n=30"), ("skew_n120", "Skew n=120"), ("t5_n120", "t(5) n=120")]
    for a, (k, label) in zip(ax, keys):
        c = e["cells"][k]
        obs = [c["q50"], c["q90"], c["q95"], c["q99"]]
        theo = [ref["q50"], ref["q90"], ref["q95"], ref["q99"]]
        a.plot(theo, theo, "k--", lw=1)
        a.plot(theo, obs, "o-", color="#b23a2b")
        a.set_title("%s\nE[w]=%.2f, factor=%.2f, size=%.3f" %
                    (label, c["e_w"], c["bartlett_factor"], c["size_el"]))
        a.set_xlabel("chi2(1) quantile"); a.set_ylabel("empirical w quantile")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_wilks_qq.png"), dpi=115)
    plt.close(fig); return "fig2_wilks_qq.png"


def plot_coverage(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, (sc, lab) in zip(ax, [("normal", "Normal"), ("skew", "Skew"), ("t5", "t(5)")]):
        for meth, col, mk in [("el", "#b23a2b", "o"), ("wald", "#2b5d8a", "s"),
                              ("boot", "#2e7d55", "^")]:
            ys = [e[f"{sc}_n{n}"][meth] for n in NS]
            a.plot(NS, ys, marker=mk, color=col, label=meth)
        a.axhline(CONF, color="0.4", ls=":")
        a.set_ylim(0.72, 1.01); a.set_title(lab + " coverage")
        a.set_xlabel("n"); a.set_ylabel("95% CI coverage"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_coverage.png"), dpi=115)
    plt.close(fig); return "fig3_coverage.png"


def plot_asym(e):
    ns = [30, 60, 120]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(ns, [e[f"n{n}"]["el_left"] for n in ns], "o-", color="#2b5d8a", label="EL left arm")
    ax[0].plot(ns, [e[f"n{n}"]["el_right"] for n in ns], "o-", color="#b23a2b", label="EL right arm")
    ax[0].set_title("Skew: EL interval is asymmetric"); ax[0].set_xlabel("n")
    ax[0].set_ylabel("mean arm length"); ax[0].legend()
    ax[1].plot(ns, [e[f"n{n}"]["el_right_left_ratio"] for n in ns], "o-",
               color="#b23a2b", label="EL")
    ax[1].plot(ns, [e[f"n{n}"]["boot_right_left_ratio"] for n in ns], "s-",
               color="#2e7d55", label="bootstrap")
    ax[1].axhline(1.0, color="0.4", ls="--", label="Wald (=1)")
    ax[1].set_title("Right / left arm length"); ax[1].set_xlabel("n")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_asymmetry.png"), dpi=115)
    plt.close(fig); return "fig4_asymmetry.png"


def plot_power(e):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ds = e["delta"]
    for a, (sc, lab) in zip(ax, [("normal", "Normal"), ("skew", "Skew")]):
        for meth, col in [("el", "#b23a2b"), ("wald", "#2b5d8a"), ("boot", "#2e7d55")]:
            a.plot(ds, [e["cells"][sc][f"{d:+.2f}"][meth] for d in ds],
                   marker="o", color=col, label=meth)
        a.axhline(0.05, color="0.4", ls=":")
        a.set_title(lab + " power, n=100"); a.set_xlabel("true shift")
        a.set_ylabel("reject H0: mu=0"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_power.png"), dpi=115)
    plt.close(fig); return "fig5_power.png"


def main():
    r = {"setup": {"seed": SEED, "conf": CONF, "chi2_q95": round(Q, 6),
                   "R": R, "G": G, "B_boot": B_BOOT, "n": NS},
         "wilks": exp_wilks(), "shape": exp_shape(),
         "coverage": exp_coverage(), "asymmetry": exp_asymmetry(),
         "power": exp_power()}
    r["figs"] = [plot_shape(r["shape"]), plot_wilks(r["wilks"]),
                 plot_coverage(r["coverage"]), plot_asym(r["asymmetry"]),
                 plot_power(r["power"])]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
