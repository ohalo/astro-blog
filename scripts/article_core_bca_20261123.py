# -*- coding: utf-8 -*-
"""BCa bootstrap confidence intervals: when the correction earns its keep.

Single source of truth for every number and figure in the article.

Four intervals are built from the SAME B bootstrap replicates so the comparison
is paired: normal approximation, percentile, basic (reverse-percentile) and
Efron's BCa (bias-corrected and accelerated, 1987). BCa fixes two things the
plain percentile interval gets wrong: median bias (through z0, the fraction of
bootstrap replicates below the point estimate) and the fact that the sampling
distribution can skew/scale with the parameter (through the jackknife
acceleration a).

    z0 = Phi^{-1}( #{theta*_b < theta_hat} / B )
    a  = sum_i (theta_bar - theta_(i))^3 / (6 [sum_i (theta_bar - theta_(i))^2]^{3/2})
    alpha1 = Phi( z0 + (z0+z_alpha) / (1 - a (z0+z_alpha)) )
    alpha2 = Phi( z0 + (z0+z_{1-alpha}) / (1 - a (z0+z_{1-alpha})) )
    CI_BCa = [ theta*_(alpha1), theta*_(alpha2) ]

We test on truth-known scenes: normal mean, exponential mean, lognormal mean,
normal variance (skewed statistic), and bivariate correlation (bounded,
nonlinear). Then we look under the hood at one sample, at the acceleration
distribution, at the small-n / boundary failure of z0, and at sample-size
sweeps. All scenes share fixed seeds so exec-replay matches results.json.
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

SEED = 20261123
B = 2000                 # bootstrap replicates per sample
ZLO, ZHI = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
Z95 = 1.959963984540054

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "bca-bootstrap-confidence-interval")
os.makedirs(_OUT, exist_ok=True)


# ---------------------------------------------------------------- estimators
def jackknife(x, theta_fn):
    n = len(x)
    return np.array([theta_fn(np.delete(x, i, axis=0)) for i in range(n)])


def bca_endpoints(tb, theta_hat, x, theta_fn):
    """Return (lo, hi, z0, a, alpha1, alpha2) from a bootstrap vector tb."""
    b = len(tb)
    tj = jackknife(x, theta_fn)
    u = tj.mean() - tj
    s2 = float(np.sum(u ** 2))
    a = float(np.sum(u ** 3) / (6.0 * s2 ** 1.5)) if s2 > 0 else 0.0
    prop = float(np.mean(tb < theta_hat))
    z0 = float(stats.norm.ppf(min(max(prop, 1.0 / b), 1.0 - 1.0 / b)))

    def adj(z):
        den = 1.0 - a * (z0 + z)
        if abs(den) < 1e-12:
            return np.nan
        return float(stats.norm.cdf(z0 + (z0 + z) / den))

    a1, a2 = adj(ZLO), adj(ZHI)
    lo = float(np.quantile(tb, a1)) if np.isfinite(a1) else np.nan
    hi = float(np.quantile(tb, a2)) if np.isfinite(a2) else np.nan
    return lo, hi, z0, a, a1, a2


def four_intervals(tb, theta_hat):
    """Paired normal / percentile / basic intervals plus percentile cutoffs."""
    se = float(np.std(tb, ddof=1))
    p_lo, p_hi = np.quantile(tb, [0.025, 0.975])
    return {
        "normal": (theta_hat - Z95 * se, theta_hat + Z95 * se),
        "percentile": (float(p_lo), float(p_hi)),
        "basic": (2 * theta_hat - float(p_hi), 2 * theta_hat - float(p_lo)),
    }


def corr_rows(xb):
    """Batch Pearson correlation for xb (B, n, 2)."""
    a = xb[:, :, 0]
    b = xb[:, :, 1]
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(ac ** 2, axis=1) * np.sum(bc ** 2, axis=1))
    return np.where(den > 0, np.sum(ac * bc, axis=1) / np.where(den > 0, den, 1.0), 0.0)


METHODS = ["normal", "percentile", "basic", "BCa"]


def coverage_scene(kind, n, true, B, R, seed):
    """Monte Carlo coverage of the four intervals on a truth-known scene."""
    rng = np.random.default_rng(seed)
    two_d = kind == "corr"

    def sample():
        if kind == "normal_mean":
            return rng.standard_normal(n)
        if kind == "exp_mean":
            return rng.exponential(1.0, n)
        if kind == "lognormal_mean":
            return rng.lognormal(0.0, 0.5, n)
        if kind == "normal_var":
            return rng.standard_normal(n)
        if kind == "corr":
            rho = 0.6
            cov = np.array([[1.0, rho], [rho, 1.0]])
            return rng.multivariate_normal([0.0, 0.0], cov, n)
        raise ValueError(kind)

    if kind == "normal_var":
        def theta_fn(x):
            return float(np.var(x, ddof=1))
        def theta_rows(xb):
            return np.var(xb, axis=1, ddof=1)
    elif two_d:
        def theta_fn(x):
            a, b = x[:, 0], x[:, 1]
            return float(np.corrcoef(a, b)[0, 1])
        def theta_rows(xb):
            return corr_rows(xb)
    else:
        def theta_fn(x):
            return float(np.mean(x))
        def theta_rows(xb):
            return xb.mean(axis=1)

    hit = {m: 0 for m in METHODS}
    left = {m: 0 for m in METHODS}
    right = {m: 0 for m in METHODS}
    width = {m: [] for m in METHODS}
    z0s, as_ = [], []
    for _ in range(R):
        x = sample()
        th = theta_fn(x)
        idx = rng.integers(0, n, (B, n))
        xb = x[idx]
        tb = theta_rows(xb)
        ints = four_intervals(tb, th)
        lo_c, hi_c, z0, a, _, _ = bca_endpoints(tb, th, x, theta_fn)
        ints["BCa"] = (lo_c, hi_c)
        z0s.append(z0)
        as_.append(a)
        for m in METHODS:
            lo, hi = ints[m]
            if np.isfinite(lo) and np.isfinite(hi):
                hit[m] += int(lo <= true <= hi)
                left[m] += int(hi < true)
                right[m] += int(lo > true)
                width[m].append(hi - lo)
    return {
        "kind": kind, "n": n, "B": B, "R": R, "true": float(true), "seed": seed,
        "cover": {m: hit[m] / R for m in METHODS},
        "left_miss": {m: left[m] / R for m in METHODS},
        "right_miss": {m: right[m] / R for m in METHODS},
        "width": {m: float(np.mean(width[m])) for m in METHODS},
        "z0_mean": float(np.mean(z0s)), "a_mean": float(np.mean(as_)),
        "a_sd": float(np.std(as_)),
    }


def exp_coverage():
    scenes = [
        coverage_scene("normal_mean", 20, 0.0, B, 1500, 1001),
        coverage_scene("exp_mean", 20, 1.0, B, 1500, 1002),
        coverage_scene("lognormal_mean", 20, float(np.exp(0.125)), B, 1500, 1003),
        coverage_scene("normal_var", 20, 1.0, B, 1500, 1004),
        coverage_scene("corr", 30, 0.6, 1200, 900, 1005),
    ]
    return scenes


def exp_mechanics():
    """One exponential sample: show the bootstrap distribution and both cuts."""
    rng = np.random.default_rng(2001)
    n = 20
    x = rng.exponential(1.0, n)
    theta_hat = float(np.mean(x))
    tb = x[rng.integers(0, n, (B, n))].mean(axis=1)
    ints = four_intervals(tb, theta_hat)
    lo_c, hi_c, z0, a, a1, a2 = bca_endpoints(tb, theta_hat, x,
                                              lambda z: float(np.mean(z)))
    return {
        "n": n, "theta_hat": theta_hat, "true": 1.0,
        "boot_skew": float(stats.skew(tb)),
        "z0": z0, "a": a, "alpha1": a1, "alpha2": a2,
        "normal": list(ints["normal"]),
        "percentile": list(ints["percentile"]),
        "basic": list(ints["basic"]),
        "bca": [lo_c, hi_c],
        "tb": tb.tolist(),
        "x": x.tolist(),
    }


def exp_acceleration():
    """Distribution of the jackknife acceleration a on a skewed statistic."""
    rng = np.random.default_rng(3001)
    n = 20
    a_all = []
    a_zero_cover, a_full_cover = 0, 0
    R = 800
    for _ in range(R):
        x = rng.exponential(1.0, n)
        th = float(np.mean(x))
        tb = x[rng.integers(0, n, (B, n))].mean(axis=1)
        lo_c, hi_c, z0, a, _, _ = bca_endpoints(tb, th, x,
                                                lambda z: float(np.mean(z)))
        a_all.append(a)
        # "z0 only" = set acceleration to zero but keep the same z0.
        def adj_zero(z):
            return float(stats.norm.cdf(z0 + (z0 + z)))
        lo_z = float(np.quantile(tb, adj_zero(ZLO)))
        hi_z = float(np.quantile(tb, adj_zero(ZHI)))
        a_zero_cover += int(lo_z <= 1.0 <= hi_z)
        a_full_cover += int(lo_c <= 1.0 <= hi_c)
    return {
        "n": n, "R": R, "a_mean": float(np.mean(a_all)),
        "a_sd": float(np.std(a_all)),
        "a_q05": float(np.quantile(a_all, 0.05)),
        "a_q95": float(np.quantile(a_all, 0.95)),
        "cover_z0_only": a_zero_cover / R,
        "cover_bca_full": a_full_cover / R,
        "a_all": a_all,
    }


def exp_boundary():
    """z0 saturation when the bootstrap replicate mass piles on one side."""
    out = {}
    for p in [0.02, 0.05, 0.10, 0.30, 0.50]:
        rng = np.random.default_rng(4000 + int(p * 100))
        n = 20
        R = 800
        hit_p, hit_c = 0, 0
        sat = 0
        widths = []
        for _ in range(R):
            x = (rng.random(n) < p).astype(float)
            th = float(np.mean(x))
            if th == 0.0:
                # no successes in the sample: every interval degenerates at 0
                hit_p += int(0.0 <= p <= 0.0)
                hit_c += int(0.0 <= p <= 0.0)
                sat += 1
                widths.append(0.0)
                continue
            tb = x[rng.integers(0, n, (B, n))].mean(axis=1)
            p_lo, p_hi = np.quantile(tb, [0.025, 0.975])
            hit_p += int(p_lo <= p <= p_hi)
            lo_c, hi_c, z0, a, _, _ = bca_endpoints(tb, th, x,
                                                    lambda z: float(np.mean(z)))
            hit_c += int(lo_c <= p <= hi_c)
            sat += int(abs(z0) >= stats.norm.ppf(1 - 1.0 / B) - 1e-9)
            widths.append(hi_c - lo_c)
        out[str(p)] = {
            "p": p, "n": n, "R": R,
            "cover_percentile": hit_p / R,
            "cover_bca": hit_c / R,
            "z0_sat_frac": sat / R,
            "bca_width": float(np.mean(widths)),
        }
    return out


def exp_sample_size():
    """Coverage vs n on the skewed exponential-mean scene."""
    out = {}
    for n in [5, 10, 20, 40, 80]:
        sc = coverage_scene("exp_mean", n, 1.0, 1200, 900, 5000 + n)
        out[str(n)] = {"n": n, "cover": sc["cover"], "width": sc["width"]}
    return out


# ------------------------------------------------------------------- figures
def fig1_coverage(scenes):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    names = [s["kind"] for s in scenes]
    x = np.arange(len(names))
    w = 0.19
    colors = {"normal": "#9aa6b2", "percentile": "#5b8ff9",
              "basic": "#61ddaa", "BCa": "#b23a2b"}
    for j, m in enumerate(METHODS):
        vals = [s["cover"][m] for s in scenes]
        ax.bar(x + (j - 1.5) * w, vals, w, label=m, color=colors[m])
    ax.axhline(0.95, color="#333", ls="--", lw=1.4, label="nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("coverage of true value")
    ax.set_title("Paired bootstrap intervals, B=2000: BCa wins only where bias+skew bite")
    ax.legend(ncol=5, fontsize=9, loc="lower right")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_coverage.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_coverage.png"


def fig2_tail_balance(scenes):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = [s["kind"] for s in scenes]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for m, color in [("left_miss", "#5b8ff9"), ("right_miss", "#b23a2b")]:
        vals = [s[m]["percentile"] for s in scenes]
        ax.bar(x, vals, 0.55, bottom=bottom, color=color, alpha=0.85,
               label=m.replace("_", " ") + " (percentile)")
        bottom += np.array(vals)
    for m, key, mk in [("BCa left", "left_miss", "o"), ("BCa right", "right_miss", "s")]:
        vals = [s[key]["BCa"] for s in scenes]
        ax.plot(x, vals, mk + "k", ms=5, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("fraction of misses")
    ax.set_title("Percentile interval misses asymmetrically; BCa rebalances the tails")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_tail_balance.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_tail_balance.png"


def fig3_mechanics(m):
    tb = np.array(m["tb"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(tb, bins=60, color="#c9d8ee", edgecolor="none", density=True)
    ax.axvline(m["true"], color="#2e7d55", lw=2, label="truth = 1.0")
    ax.axvline(m["theta_hat"], color="#333", lw=1.6, ls="--", label="theta_hat")
    p_lo, p_hi = m["percentile"]
    b_lo, b_hi = m["bca"]
    ax.axvspan(p_lo, p_hi, color="#5b8ff9", alpha=0.18, label="percentile CI")
    ax.axvspan(b_lo, b_hi, color="#b23a2b", alpha=0.16, label="BCa CI")
    ax.set_xlabel("bootstrap mean")
    ax.set_ylabel("density")
    ax.set_title("One sample: z0=%.3f, a=%.3f; percentile [%.3f,%.3f] vs BCa [%.3f,%.3f]"
                 % (m["z0"], m["a"], p_lo, p_hi, b_lo, b_hi))
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_mechanics.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_mechanics.png"


def fig4_acceleration(acc):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.hist(acc["a_all"], bins=45, color="#f2c4b4", edgecolor="none", density=True)
    ax.axvline(0, color="#333", ls="--", lw=1.3)
    ax.axvline(acc["a_mean"], color="#b23a2b", lw=2,
               label="mean a=%.3f" % acc["a_mean"])
    ax.set_xlabel("jackknife acceleration a")
    ax.set_ylabel("density")
    ax.set_title("Acceleration is positive on a right-skewed statistic: "
                 "z0-only cover %.3f -> BCa cover %.3f"
                 % (acc["cover_z0_only"], acc["cover_bca_full"]))
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_acceleration.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_acceleration.png"


def fig5_boundary(bd, ss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ps = sorted(float(k) for k in bd)
    a0 = ax[0]
    a0.plot(ps, [bd[str(p)]["cover_percentile"] for p in ps], "s-",
            color="#5b8ff9", label="percentile")
    a0.plot(ps, [bd[str(p)]["cover_bca"] for p in ps], "o-",
            color="#b23a2b", label="BCa")
    a0.plot(ps, [bd[str(p)]["z0_sat_frac"] for p in ps], "^--",
            color="#888", label="z0 saturated")
    a0.axhline(0.95, color="#333", ls="--", lw=1.2)
    a0.set_xlabel("true binomial p (n=20)")
    a0.set_ylabel("fraction")
    a0.set_title("z0 saturates at the boundary: p near 0 or 1")
    a0.legend(fontsize=8)
    ns = sorted(int(k) for k in ss)
    a1 = ax[1]
    for m, color in [("normal", "#9aa6b2"), ("percentile", "#5b8ff9"),
                     ("basic", "#61ddaa"), ("BCa", "#b23a2b")]:
        a1.plot(ns, [ss[str(n)]["cover"][m] for n in ns], "o-", color=color, label=m)
    a1.axhline(0.95, color="#333", ls="--", lw=1.2)
    a1.set_xlabel("sample size n (exponential mean)")
    a1.set_ylabel("coverage")
    a1.set_title("Small n: neither percentile nor BCa reaches nominal")
    a1.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_boundary.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_boundary.png"


def main():
    scenes = exp_coverage()
    mech = exp_mechanics()
    acc = exp_acceleration()
    bd = exp_boundary()
    ss = exp_sample_size()
    r = {
        "setup": {"seed": SEED, "B": B, "methods": METHODS,
                  "alpha": 0.05, "corr_rho": 0.6},
        "coverage": scenes,
        "mechanics": {k: v for k, v in mech.items() if k not in ("tb", "x")},
        "acceleration": {k: v for k, v in acc.items() if k != "a_all"},
        "boundary": bd,
        "sample_size": ss,
    }
    r["figs"] = [fig1_coverage(scenes), fig2_tail_balance(scenes),
                 fig3_mechanics(mech), fig4_acceleration(acc),
                 fig5_boundary(bd, ss)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
