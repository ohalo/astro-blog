# -*- coding: utf-8 -*-
"""Sieve bootstrap vs iid bootstrap vs moving-block bootstrap for a dependent
mean: coverage of a 95% CI under AR(1) dependence, block-length and sieve-order
sensitivity, AIC order selection, and ARMA misspecification.

Single source of truth for every number and figure in the article.

DGP: x_t = phi * x_{t-1} + eps_t, eps ~ N(0,1). Target = population mean mu = 0.
The iid bootstrap resamples raw x (destroys dependence) -> severe undercoverage.
The sieve bootstrap fits an AR(p), resamples centered residuals iid, and
regenerates the series -> preserves the dependence. The moving-block bootstrap
preserves local dependence with a block length L. HAC (Newey-West) is the
analytic benchmark.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261121
N = 200
BURN = 100
REPS = 500
B = 399
Z95 = 1.959963985
PHIS = [0.0, 0.3, 0.6, 0.85]
MU = 0.0

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "sieve-bootstrap-dependent-data")
os.makedirs(_OUT, exist_ok=True)


def gen_ar1(phi, rng, n=N, burn=BURN):
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x[burn:]


def gen_arma(phi, theta, rng, n=N, burn=BURN):
    x = np.zeros(n + burn)
    e = np.zeros(n + burn)
    for t in range(1, n + burn):
        e[t] = rng.standard_normal()
        x[t] = phi * x[t - 1] + e[t] + theta * e[t - 1]
    return x[burn:]


def percentile_ci(stats, lo=2.5, hi=97.5):
    return np.percentile(stats, lo), np.percentile(stats, hi)


def covers(stat_boot, target=MU):
    lo, hi = percentile_ci(stat_boot)
    return bool(lo <= target <= hi)


def iid_boot_mean(x, rng):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(axis=1)


def mbb_mean(x, L, rng):
    n = len(x)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    offs = np.arange(L)[None, None, :]
    idx = (starts[:, :, None] + offs) % n
    xs = x[idx.reshape(B, -1)[:, :n]]
    return xs.mean(axis=1)


def fit_ar(x, p):
    """OLS fit x_t = c + a1*x_{t-1} + ... + ap*x_{t-p}; return (c, coefs, resid)."""
    n = len(x)
    X = [np.ones(n - p)]
    y = x[p:]
    for j in range(1, p + 1):
        X.append(x[p - j:n - j])
    X = np.column_stack(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta[0], beta[1:], resid


def aic_order(x, pmax=8):
    n = len(x)
    best_p, best_aic = 0, np.inf
    for p in range(1, pmax + 1):
        _, _, r = fit_ar(x, p)
        aic = n * np.log(r @ r / n) + 2 * p
        if aic < best_aic:
            best_aic, best_p = aic, p
    return best_p


def sieve_boot_mean(x, p, rng):
    """Fit AR(p), resample centered residuals iid, regenerate series, mean."""
    c, coefs, resid = fit_ar(x, p)
    resid = resid - resid.mean()
    n = len(x)
    eps = rng.choice(resid, size=(B, n), replace=True)
    xs = np.empty((B, n))
    xs[:, :p] = x[:p][None, :]
    for t in range(p, n):
        val = np.full(B, c)
        for j in range(1, p + 1):
            val = val + coefs[j - 1] * xs[:, t - j]
        xs[:, t] = val + eps[:, t]
    return xs.mean(axis=1)


def hac_se(x, L=10):
    """Newey-West HAC standard error for the mean."""
    n = len(x)
    d = x - x.mean()
    g0 = d @ d / n
    var = g0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        g = float(d[:-lag] @ d[lag:] / n)
        var += 2 * w * g
    return np.sqrt(max(var, 1e-12) / n)


def exp_coverage():
    """Coverage of 95% CI for the mean across phi, four methods."""
    out = {"phi": PHIS, "iid": [], "mbb": [], "sieve": [], "hac": []}
    for phi in PHIS:
        rng = np.random.default_rng(SEED + 10 + int(phi * 100))
        ci = {k: 0 for k in ["iid", "mbb", "sieve", "hac"]}
        for _ in range(REPS):
            x = gen_ar1(phi, rng)
            ci["iid"] += covers(iid_boot_mean(x, rng))
            ci["mbb"] += covers(mbb_mean(x, 10, rng))
            p_hat = aic_order(x)
            ci["sieve"] += covers(sieve_boot_mean(x, max(p_hat, 1), rng))
            lo = x.mean() - Z95 * hac_se(x, 10)
            hi = x.mean() + Z95 * hac_se(x, 10)
            ci["hac"] += bool(lo <= MU <= hi)
        for k in out:
            if k != "phi":
                out[k].append(ci[k] / REPS)
    return out


def exp_block_length():
    """MBB coverage vs block length L at phi=0.6 and 0.85."""
    Ls = [2, 5, 10, 20, 40, 80]
    out = {"L": Ls, "phi06": [], "phi085": []}
    for phi, key in [(0.6, "phi06"), (0.85, "phi085")]:
        rng = np.random.default_rng(SEED + 30 + int(phi * 100))
        for L in Ls:
            cnt = 0
            for _ in range(REPS):
                x = gen_ar1(phi, rng)
                cnt += covers(mbb_mean(x, L, rng))
            out[key].append(cnt / REPS)
    return out


def exp_sieve_order():
    """Sieve coverage vs order p (AR(1) phi=0.6), plus AIC order distribution."""
    ps = [1, 2, 3, 5, 8]
    out = {"p": ps, "coverage": [], "aic_counts": {}, "aic_mean": None}
    rng = np.random.default_rng(SEED + 50)
    orders = []
    for _ in range(REPS):
        x = gen_ar1(0.6, rng)
        orders.append(aic_order(x))
        # coverage for each p on the same sample
    orders = np.array(orders)
    vals = {p: 0 for p in ps}
    rng2 = np.random.default_rng(SEED + 51)
    for p in ps:
        for _ in range(REPS):
            x = gen_ar1(0.6, rng2)
            vals[p] += covers(sieve_boot_mean(x, p, rng2))
        out["coverage"].append(vals[p] / REPS)
    for p in range(1, 9):
        out["aic_counts"][str(p)] = int(np.sum(orders == p))
    out["aic_mean"] = float(orders.mean())
    return out


def exp_misspec():
    """ARMA(1,1) misspecification: coverage vs MA theta, sieve(p) vs MBB vs iid."""
    thetas = [0.0, 0.3, 0.5, 0.7]
    out = {"theta": thetas, "sieve_aic": [], "mbb": [], "iid": []}
    for theta in thetas:
        rng = np.random.default_rng(SEED + 70 + int(theta * 100))
        ci = {k: 0 for k in ["sieve_aic", "mbb", "iid"]}
        for _ in range(REPS):
            x = gen_arma(0.5, theta, rng)
            ci["iid"] += covers(iid_boot_mean(x, rng))
            ci["mbb"] += covers(mbb_mean(x, 10, rng))
            ci["sieve_aic"] += covers(sieve_boot_mean(x, max(aic_order(x), 1), rng))
        for k in out:
            if k != "theta":
                out[k].append(ci[k] / REPS)
    return out


def plot1():
    rng = np.random.default_rng(SEED + 100)
    x = gen_ar1(0.6, rng)
    xb_iid = iid_boot_mean(x, rng)
    # one representative iid-resampled series and sieve-resampled series for ACF
    idx = rng.integers(0, len(x), size=len(x))
    xi = x[idx]
    p_hat = max(aic_order(x), 1)
    c, coefs, resid = fit_ar(x, p_hat)
    resid = resid - resid.mean()
    eps = rng.choice(resid, size=len(x), replace=True)
    xs = np.empty(len(x)); xs[:p_hat] = x[:p_hat]
    for t in range(p_hat, len(x)):
        xs[t] = c + sum(coefs[j - 1] * xs[t - j] for j in range(1, p_hat + 1)) + eps[t]

    def acf(v, lags):
        d = v - v.mean()
        return np.array([1.0 if k == 0 else float(d[:-k] @ d[k:] / (d @ d)) for k in lags])

    lags = np.arange(0, 11)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(x, lw=1, color="#2c3e50")
    axes[0].axhline(MU, color="k", ls="--", lw=0.8)
    axes[0].set_title("Fig1a: one AR(1) path, phi=0.6", fontsize=10)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x_t")
    w = 0.27
    for i, (label, v, col) in enumerate([("original", x, "#2c3e50"),
                                          ("iid resample", xi, "#c0392b"),
                                          ("sieve resample", xs, "#27ae60")]):
        axes[1].bar(lags + (i - 1) * w, acf(v, lags), w, label=label, color=col, alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Fig1b: iid resampling destroys the ACF; sieve preserves it", fontsize=10)
    axes[1].set_xlabel("lag"); axes[1].set_ylabel("autocorrelation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_dependence.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig1_dependence.png"


def plot2(cov):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(cov["phi"], cov["iid"], "o-", color="#c0392b", lw=2, label="iid bootstrap")
    ax.plot(cov["phi"], cov["mbb"], "s-", color="#2980b9", lw=2, label="moving block (L=10)")
    ax.plot(cov["phi"], cov["sieve"], "^-", color="#27ae60", lw=2, label="sieve bootstrap (AIC order)")
    ax.plot(cov["phi"], cov["hac"], "D--", color="#8e44ad", lw=1.5, label="HAC (Newey-West)")
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("AR(1) persistence phi")
    ax.set_ylabel("coverage of 95% CI for the mean")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig2: the iid bootstrap collapses under dependence; sieve/HAC hold ~95%")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_coverage.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig2_coverage.png"


def plot3(bl):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(bl["L"], bl["phi06"], "o-", color="#2980b9", lw=2, label="phi=0.6")
    ax.plot(bl["L"], bl["phi085"], "s-", color="#c0392b", lw=2, label="phi=0.85")
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("block length L")
    ax.set_ylabel("MBB coverage of the mean")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig3: block bootstrap needs L long enough to capture dependence")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_block_length.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig3_block_length.png"


def plot4(so):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.plot(so["p"], so["coverage"], "o-", color="#27ae60", lw=2)
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("sieve AR order p (true DGP is AR(1))")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("4a: one lag is enough; extra lags do no harm", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ps = sorted(int(k) for k in so["aic_counts"])
    cnts = [so["aic_counts"][str(p)] for p in ps]
    ax.bar(ps, cnts, color="#2980b9", alpha=0.85)
    ax.set_xlabel("AIC-selected AR order")
    ax.set_ylabel("count (of %d)" % REPS)
    ax.set_title("4b: AIC picks p=1 most often (mean %.2f)" % so["aic_mean"], fontsize=10)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_sieve_order.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig4_sieve_order.png"


def plot5(ms):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(ms["theta"], ms["iid"], "o-", color="#c0392b", lw=2, label="iid bootstrap")
    ax.plot(ms["theta"], ms["sieve_aic"], "^-", color="#27ae60", lw=2, label="sieve (AIC order)")
    ax.plot(ms["theta"], ms["mbb"], "s-", color="#2980b9", lw=2, label="moving block (L=10)")
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("MA coefficient theta in ARMA(1,1) DGP")
    ax.set_ylabel("coverage of the mean")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig5: with MA structure the sieve needs more lags; block is more robust")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_misspec.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig5_misspec.png"


def main():
    cov = exp_coverage()
    bl = exp_block_length()
    so = exp_sieve_order()
    ms = exp_misspec()
    results = {
        "setup": {"seed": SEED, "n": N, "burn": BURN, "reps": REPS, "B": B,
                  "phis": PHIS, "z95": Z95},
        "coverage": cov,
        "block_length": bl,
        "sieve_order": so,
        "misspec": ms,
    }
    results["figs"] = [plot1(), plot2(cov), plot3(bl), plot4(so), plot5(ms)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
