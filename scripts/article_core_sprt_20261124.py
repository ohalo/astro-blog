# -*- coding: utf-8 -*-
"""Sequential Probability Ratio Test (SPRT) for a Gaussian mean / strategy edge.

Single source of truth for every number and figure in the article. H0: mu=0
versus H1: mu=mu1>0 for iid daily returns with known sigma. The log likelihood
ratio is a random walk

    L_n = (mu1-mu0)/sigma^2 * sum x_t - n*(mu1^2-mu0^2)/(2*sigma^2)

with Wald boundaries A=log((1-beta)/alpha), B=log(beta/(1-alpha)). We stop at
the first crossing and measure actual size, power and expected sample size,
compare against a fixed-horizon z test of equal nominal error rates, and show
the three things the textbook boundary formula hides: overshoot at the boundary,
discrete monitoring, and design-vs-truth drift mismatch. Sigma misspecification
is also tested; common random numbers keep all comparisons paired.

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
MU0 = 0.0
MU1 = 0.15
SIGMA = 1.0
ALPHA = 0.05
BETA = 0.05
NMAX = 1200
A = np.log((1.0 - BETA) / ALPHA)
B = np.log(BETA / (1.0 - BETA))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "sprt-sequential-likelihood-ratio-test")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------- engine
def llr(x, mu0, mu1, sigma):
    """Log-likelihood ratio random walk for iid N(mu, sigma^2) observations."""
    inc = (mu1 - mu0) / sigma ** 2 * x - (mu1 ** 2 - mu0 ** 2) / (2 * sigma ** 2)
    return np.cumsum(inc, axis=1)


def crossing(path, upper=A, lower=B, every=1):
    """First crossing of a monitored path; returns (stop_n, decision, value).

    decision: +1 accept H1, -1 accept H0, 0 = hit the cap without a decision.
    """
    nrow, ncol = path.shape
    monitor = np.zeros(ncol, dtype=bool)
    monitor[every - 1::every] = True
    monitor[-1] = True
    hi = (path >= upper) & monitor[None, :]
    lo = (path <= lower) & monitor[None, :]
    hit = hi | lo
    any_hit = hit.any(axis=1)
    idx = np.where(any_hit, hit.argmax(axis=1), ncol - 1)
    rows = np.arange(nrow)
    val = path[rows, idx]
    dec = np.where(hi[rows, idx], 1, np.where(lo[rows, idx], -1, 0))
    return idx + 1, dec, val


def fixed_horizon_n(mu0, mu1, sigma, alpha=ALPHA, beta=BETA):
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta)
    return int(np.ceil((z * sigma / (mu1 - mu0)) ** 2))


def fixed_test(x, n, mu0, sigma, alpha=ALPHA):
    crit = mu0 + stats.norm.ppf(1 - alpha) * sigma / np.sqrt(n)
    return x[:, :n].mean(axis=1) > crit


# ------------------------------------------------------------- experiments
def exp_oc(reps=4000):
    rng = np.random.default_rng(SEED + 10)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    x0 = MU0 + SIGMA * z0
    x1 = MU1 + SIGMA * z1
    p0 = llr(x0, MU0, MU1, SIGMA)
    p1 = llr(x1, MU0, MU1, SIGMA)
    n0, d0, v0 = crossing(p0)
    n1, d1, v1 = crossing(p1)
    nfix = fixed_horizon_n(MU0, MU1, SIGMA)
    fsize = float(np.mean(fixed_test(x0, nfix, MU0, SIGMA)))
    fpower = float(np.mean(fixed_test(x1, nfix, MU0, SIGMA)))
    return {
        "reps": reps, "nmax": NMAX, "A": float(A), "B": float(B),
        "mu1": MU1, "sigma": SIGMA, "n_fixed": nfix,
        "sprt_size": float(np.mean(d0 == 1)),
        "sprt_power": float(np.mean(d1 == 1)),
        "sprt_e_n_h0": float(n0.mean()),
        "sprt_e_n_h1": float(n1.mean()),
        "sprt_cap_h0": float(np.mean(d0 == 0)),
        "sprt_cap_h1": float(np.mean(d1 == 0)),
        "fixed_size": fsize, "fixed_power": fpower,
        "saving_h0": 1.0 - n0.mean() / nfix,
        "saving_h1": 1.0 - n1.mean() / nfix,
        "upper_overshoot_mean": float(np.mean(v1[d1 == 1] - A)) if np.any(d1 == 1) else 0.0,
        "lower_overshoot_mean": float(np.mean(B - v0[d0 == -1])) if np.any(d0 == -1) else 0.0,
        "upper_overshoot_q90": float(np.quantile(v1[d1 == 1] - A, 0.90)),
        "lower_overshoot_q90": float(np.quantile(B - v0[d0 == -1], 0.90)),
        "upper_cross_vals": (v1[d1 == 1] - A)[:600].tolist(),
        "lower_cross_vals": (B - v0[d0 == -1])[:600].tolist(),
    }


def exp_monitor(reps=2500):
    rng = np.random.default_rng(SEED + 20)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    p0 = llr(MU0 + SIGMA * z0, MU0, MU1, SIGMA)
    p1 = llr(MU1 + SIGMA * z1, MU0, MU1, SIGMA)
    rows = {}
    for k in (1, 2, 5, 10, 20, 50):
        n0, d0, _ = crossing(p0, every=k)
        n1, d1, _ = crossing(p1, every=k)
        rows[str(k)] = {
            "size": float(np.mean(d0 == 1)),
            "power": float(np.mean(d1 == 1)),
            "e_n_h0": float(n0.mean()),
            "e_n_h1": float(n1.mean()),
        }
    return rows


def exp_sigma_mismatch(reps=3000):
    """Same underlying paths; change only the sigma used inside the LLR."""
    rng = np.random.default_rng(SEED + 30)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    x0 = MU0 + SIGMA * z0
    x1 = MU1 + SIGMA * z1
    rows = {}
    for sa in (0.7, 0.85, 1.0, 1.15, 1.3, 1.6):
        d0 = crossing(llr(x0, MU0, MU1, sa))[1]
        n1, d1, _ = crossing(llr(x1, MU0, MU1, sa))
        rows[str(sa)] = {
            "size": float(np.mean(d0 == 1)),
            "power": float(np.mean(d1 == 1)),
            "e_n_h1": float(n1.mean()),
            "cap_rate_h1": float(np.mean(d1 == 0)),
        }
    return rows


def exp_drift_mismatch(reps=2500):
    """Design the test for MU1 but let the true per-day edge vary."""
    rng = np.random.default_rng(SEED + 40)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    x0 = MU0 + SIGMA * z0
    rows = {}
    for mu in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21):
        n1, d1, _ = crossing(llr(mu + SIGMA * z1, MU0, MU1, SIGMA))
        rows[str(mu)] = {
            "power": float(np.mean(d1 == 1)),
            "e_n_h1": float(n1.mean()),
            "cap_rate": float(np.mean(d1 == 0)),
        }
    rows["h0_size"] = float(np.mean(crossing(llr(x0, MU0, MU1, SIGMA))[1] == 1))
    return rows


def exp_paths(steps=450):
    rng = np.random.default_rng(SEED + 50)
    z0 = rng.standard_normal(steps)
    z1 = rng.standard_normal(steps)
    p0 = llr((MU0 + SIGMA * z0)[None, :], MU0, MU1, SIGMA)[0]
    p1 = llr((MU1 + SIGMA * z1)[None, :], MU0, MU1, SIGMA)[0]
    for p in (p0, p1):
        hi = np.where(p >= A)[0]
        lo = np.where(p <= B)[0]
        if hi.size or lo.size:
            first = min(hi[0] if hi.size else 10 ** 9,
                        lo[0] if lo.size else 10 ** 9)
            p[first + 1:] = np.nan
    return {"h0": p0.tolist(), "h1": p1.tolist()}


# ------------------------------------------------------------------ figures
def fig1_paths(p):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    h0 = np.array(p["h0"]); h1 = np.array(p["h1"])
    ax.plot(np.arange(1, len(h0) + 1), h0, color="#2b8cbe", lw=1.4,
            label="one path under H0")
    ax.plot(np.arange(1, len(h1) + 1), h1, color="#b23a2b", lw=1.4,
            label="one path under H1")
    ax.axhline(A, color="#333", ls="--", lw=1.3, label="A = log((1-beta)/alpha)")
    ax.axhline(B, color="#333", ls=":", lw=1.3, label="B = log(beta/(1-alpha))")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("number of daily observations")
    ax.set_ylabel("log likelihood ratio L_n")
    ax.set_title("SPRT: two paths, two absorbing boundaries")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig1_paths.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig1_paths.png"


def fig2_oc(oc):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    names = ["size", "power"]
    sprt = [oc["sprt_size"], oc["sprt_power"]]
    fixed = [oc["fixed_size"], oc["fixed_power"]]
    xs = np.arange(2); w = 0.34
    ax[0].bar(xs - w / 2, sprt, w, label="SPRT (Wald)", color="#3a923a")
    ax[0].bar(xs + w / 2, fixed, w, label="fixed n=%d" % oc["n_fixed"],
              color="#2b8cbe")
    ax[0].axhline(0.05, color="#333", ls="--", lw=1.1)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names)
    ax[0].set_ylim(0, 1.03); ax[0].legend(fontsize=9)
    ax[0].set_title("Error rates at nominal alpha=beta=5%")
    ns = [oc["sprt_e_n_h0"], oc["sprt_e_n_h1"], oc["n_fixed"]]
    labels = ["SPRT under H0", "SPRT under H1", "fixed horizon"]
    ax[1].bar(labels, ns, color=["#2b8cbe", "#3a923a", "#b23a2b"], alpha=0.9)
    for i, v in enumerate(ns):
        ax[1].text(i, v + 8, "%.0f" % v, ha="center", fontsize=9)
    ax[1].set_ylabel("average observations")
    ax[1].set_title("Expected sample size, same nominal errors")
    fig.tight_layout()
    f = os.path.join(OUT, "fig2_oc.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig2_oc.png"


def fig3_overshoot(oc):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.hist(oc["upper_cross_vals"], bins=40, alpha=0.75,
            color="#b23a2b", label="upper crossing overshoot")
    ax.hist(oc["lower_cross_vals"], bins=40, alpha=0.75,
            color="#2b8cbe", label="lower crossing overshoot")
    ax.set_xlabel("distance beyond the boundary")
    ax.set_ylabel("count")
    ax.set_title("Discrete daily monitoring overshoots the Wald boundary")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig3_overshoot.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig3_overshoot.png"


def fig4_monitor(rows):
    ks = [1, 2, 5, 10, 20, 50]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for lab, col in (("size", "#b23a2b"), ("power", "#3a923a")):
        ax[0].plot(ks, [rows[str(k)][lab] for k in ks], "o-",
                   label=lab, color=col, lw=1.8)
    ax[0].axhline(0.05, color="#333", ls="--", lw=1.1)
    ax[0].set_xlabel("check the LLR every k days")
    ax[0].set_ylabel("rejection rate")
    ax[0].set_title("Rare looks shrink size but raise power")
    ax[0].legend(fontsize=9)
    for lab, col in (("e_n_h0", "#2b8cbe"), ("e_n_h1", "#3a923a")):
        ax[1].plot(ks, [rows[str(k)][lab] for k in ks], "o-",
                   label=lab, color=col, lw=1.8)
    ax[1].set_xlabel("check the LLR every k days")
    ax[1].set_ylabel("average observations")
    ax[1].set_title("Waiting costs observations")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig4_monitor.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig4_monitor.png"


def fig5_mismatch(sig, drift):
    sas = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
    mus = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(sas, [sig[str(s)]["size"] for s in sas], "o-",
               color="#b23a2b", label="size", lw=1.8)
    ax[0].plot(sas, [sig[str(s)]["power"] for s in sas], "o-",
               color="#3a923a", label="power", lw=1.8)
    ax[0].set_xlabel("sigma assumed inside the LLR (truth = 1.0)")
    ax[0].set_ylabel("rejection rate")
    ax[0].set_title("Wrong sigma: size drifts from 0.19 to 0.00")
    ax[0].legend(fontsize=9)
    axb = ax[1]
    axb.plot(mus, [drift[str(m)]["power"] for m in mus], "o-",
             color="#3a923a", label="power (left axis)", lw=1.8)
    axb.set_xlabel("true daily edge (test designed for 0.15)")
    axb.set_ylabel("power")
    axb2 = axb.twinx()
    axb2.plot(mus, [drift[str(m)]["e_n_h1"] for m in mus], "s--",
              color="#b23a2b", label="E[N] (right axis)", lw=1.8)
    axb2.set_ylabel("average observations")
    axb.set_title("Wrong edge: E[N] peaks near mu1/2 as power fades")
    axb.legend(fontsize=9, loc="upper left")
    axb2.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    f = os.path.join(OUT, "fig5_mismatch.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig5_mismatch.png"


def main():
    oc = exp_oc()
    mon = exp_monitor()
    sig = exp_sigma_mismatch()
    drift = exp_drift_mismatch()
    paths = exp_paths()
    r = {
        "setup": {"seed": SEED, "alpha": ALPHA, "beta": BETA, "mu0": MU0,
                  "mu1": MU1, "sigma": SIGMA, "A": float(A), "B": float(B),
                  "nmax": NMAX, "reps_oc": 4000, "reps_monitor": 2500,
                  "reps_sigma": 3000, "reps_drift": 2500},
        "oc": oc,
        "monitor": mon,
        "sigma_mismatch": sig,
        "drift_mismatch": drift,
    }
    r["figs"] = [fig1_paths(paths), fig2_oc(oc), fig3_overshoot(oc),
                 fig4_monitor(mon), fig5_mismatch(sig, drift)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
