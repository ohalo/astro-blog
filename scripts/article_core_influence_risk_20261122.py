# -*- coding: utf-8 -*-
"""Influence functions + delete-1 jackknife for risk-measure inference.

Single source of truth for every number and figure in the article.

Truth-known Monte-Carlo experiments on Gaussian and standardized t(3) returns.
Risk measures (LEFT tail, q=0.05):
    mean, median, VaR_q = F^{-1}(q), ES_q = E[X | X <= VaR_q].

Influence functions (IF) — the derivative of a functional T at F when a single
point x is contaminated by weight eps:
    IF_mean(x)   = x - mu
    IF_med(x)    = (0.5 - 1{x <= m}) / f(m)
    IF_VaR(x)    = (q - 1{x <= xi}) / f(xi),   xi = F^{-1}(q)
    IF_ES(x)     = (xi - ES) + (1/q) * 1{x <= xi} * (x - xi)
The last one is UNBOUNDED in the left tail (slope 1/q), while VaR's IF is
bounded — the key contrast the article documents.

Asymptotic SE from IF:  SE = sqrt(E[IF^2] / n).  We compare it with the
delete-1 jackknife SE and the iid bootstrap SE, then check finite-sample
coverage of the normal-approximation CI, and tail-heavy / contamination stress.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, special

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261122
Q = 0.05
CONF = 0.95
Z = float(special.ndtri(0.5 + CONF / 2.0))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "influence-function-risk-estimation")
os.makedirs(_OUT, exist_ok=True)


# ---------------------------------------------------------------- risk measures
def measures(x, q=Q):
    x = np.asarray(x, dtype=float)
    xm = np.sort(x)
    n = xm.size
    var = float(np.quantile(xm, q))
    tail = xm[xm <= var]
    es = float(tail.mean()) if tail.size else var
    return {
        "mean": float(xm.mean()),
        "median": float(np.median(xm)),
        "var": var,
        "es": es,
    }


def density_at(x, a):
    """Silverman-rule KDE density of x evaluated at scalar a."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = 0.9 * min(np.std(x, ddof=1), stats.iqr(x) / 1.349) * n ** (-0.2)
    h = max(h, 1e-6)
    return float(np.mean(np.exp(-0.5 * ((x - a) / h) ** 2)) / (h * np.sqrt(2 * np.pi)))


def if_values(x, q=Q):
    """Empirical IF at each observation for the four measures."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = measures(x, q)
    med, var, es = m["median"], m["var"], m["es"]
    f_med = density_at(x, med)
    f_var = density_at(x, var)
    ind_med = (x <= med).astype(float)
    ind_var = (x <= var).astype(float)
    return {
        "mean": x - m["mean"],
        "median": (0.5 - ind_med) / f_med,
        "var": (q - ind_var) / f_var,
        "es": (var - es) + (1.0 / q) * ind_var * (x - var),
    }


def se_if(x, q=Q):
    ifs = if_values(x, q)
    return {k: float(np.sqrt(np.mean(v ** 2) / x.size)) for k, v in ifs.items()}


def se_jack(x, q=Q):
    x = np.asarray(x, dtype=float)
    n = x.size
    names = ["mean", "median", "var", "es"]
    acc = {k: [] for k in names}
    for i in range(n):
        t = measures(np.delete(x, i), q)
        for k in names:
            acc[k].append(t[k])
    # delete-1 jackknife: var = (n-1)/n * sum_i (t_(i)-mean)^2 = (n-1)*var(t_(i))
    return {k: float(np.sqrt((n - 1) * np.var(acc[k], ddof=0))) for k in names}


def se_boot(x, q=Q, b=1600, rng=None):
    rng = rng or np.random.default_rng(0)
    n = x.size
    names = ["mean", "median", "var", "es"]
    acc = {k: np.empty(b) for k in names}
    for r in range(b):
        xs = x[rng.integers(0, n, n)]
        t = measures(xs, q)
        for k in names:
            acc[k][r] = t[k]
    return {k: float(np.std(acc[k], ddof=1)) for k in names}


# ---------------------------------------------------------------- exp 1: IF shape
def exp_if_shape():
    rng = np.random.default_rng(SEED)
    n = 200000
    base = rng.standard_normal(n)
    eps = 0.001
    k = int(round(eps * n))          # number of points replaced by the probe x0
    probes = np.linspace(-8.0, 4.0, 25)
    clean = measures(base)
    rows = []
    xi = float(stats.norm.ppf(Q))
    es_true = -float(stats.norm.pdf(xi)) / Q
    for x0 in probes:
        cont = base.copy()
        cont[:k] = x0
        mc = measures(cont)
        rows.append({
            "x0": float(x0),
            "mean": (mc["mean"] - clean["mean"]) / eps,
            "median": (mc["median"] - clean["median"]) / eps,
            "var": (mc["var"] - clean["var"]) / eps,
            "es": (mc["es"] - clean["es"]) / eps,
        })
    # analytic lines for mean / VaR / ES (median handled by finite diff only)
    analytic = {
        "mean_a": [float(r["x0"]) for r in rows],
        "var_a": [(Q - (1.0 if r["x0"] <= xi else 0.0)) / float(stats.norm.pdf(xi)) for r in rows],
        "es_a": [(xi - es_true) + (1.0 / Q) * (1.0 if r["x0"] <= xi else 0.0) * (r["x0"] - xi) for r in rows],
    }
    # agreement metrics (finite-diff vs analytic)
    fd_mean = np.array([r["mean"] for r in rows])
    fd_var = np.array([r["var"] for r in rows])
    fd_es = np.array([r["es"] for r in rows])
    return {
        "eps": eps,
        "xi": xi,
        "es_true": es_true,
        "rows": rows,
        "analytic": analytic,
        "max_abs_err": {
            "mean": float(np.max(np.abs(fd_mean - np.array(analytic["mean_a"])))),
            "var": float(np.max(np.abs(fd_var - np.array(analytic["var_a"])))),
            "es": float(np.max(np.abs(fd_es - np.array(analytic["es_a"])))),
        },
        "es_slope_fd": float((fd_es[0] - fd_es[6]) / (probes[0] - probes[6])),
    }


# ---------------------------------------------------------------- exp 2: single outlier
def exp_single_outlier():
    rng = np.random.default_rng(SEED + 1)
    n = 200
    base = rng.standard_normal(n)
    clean = measures(base)
    xs = np.linspace(-2.0, -20.0, 19)
    mean0 = clean["mean"]
    rows = []
    for x0 in xs:
        cont = np.append(base, x0)          # one extra extreme observation
        mc = measures(cont)
        rows.append({
            "x0": float(x0),
            "d_mean": mc["mean"] - mean0,
            "d_median": mc["median"] - clean["median"],
            "d_var": mc["var"] - clean["var"],
            "d_es": mc["es"] - clean["es"],
        })
    theo_mean = [(x - mean0) / (n + 1) for x in xs]
    return {"n": n, "rows": rows, "theo_mean": theo_mean,
            "clean": clean}


# ---------------------------------------------------------------- exp 3: SE methods
def exp_se_methods():
    rng = np.random.default_rng(SEED + 2)
    n = 500
    out = {}
    for dist, gen in [("normal", "normal"), ("t3", "t3")]:
        if gen == "normal":
            x = rng.standard_normal(n)
        else:
            t = rng.standard_t(3.0, n)
            x = t / np.sqrt(3.0 / (3.0 - 2.0))   # standardize to unit variance
        s_if = se_if(x)
        s_jk = se_jack(x)
        s_bs = se_boot(x, b=1600, rng=np.random.default_rng(SEED + 20))
        out[dist] = {"if": s_if, "jack": s_jk, "boot": s_bs}
    return {"n": n, "methods": out}


# ---------------------------------------------------------------- exp 4: coverage
def _oracle_se_if(x, dist):
    """SE from IF using the TRUE density at the true quantile (best case)."""
    n = x.size
    xs = np.sort(x)
    m = measures(xs)
    if dist == "normal":
        xi = float(stats.norm.ppf(Q))
        fx = float(stats.norm.pdf(xi))
        fmed = float(stats.norm.pdf(0.0))
        es_true = -fx / Q
    else:
        # standardized t(3): scale s = sqrt(3/(3-2)) applied to t
        s = np.sqrt(3.0 / 1.0)
        xi_t = float(stats.t.ppf(Q, 3.0))
        xi = s * xi_t
        fx = float(stats.t.pdf(xi_t, 3.0)) / s
        fmed = float(stats.t.pdf(0.0, 3.0)) / s
        es_true = -(s * (3.0 + xi_t ** 2) / (3.0 - 1.0) * float(stats.t.pdf(xi_t, 3.0))) / Q
    ind_var = (xs <= xi).astype(float)
    ind_med = (xs <= 0.0).astype(float)
    if_var = (Q - ind_var) / fx
    if_med = (0.5 - ind_med) / fmed
    if_es = (xi - es_true) + (1.0 / Q) * ind_var * (xs - xi)
    if_mean = xs - 0.0
    return {
        "mean": float(np.sqrt(np.mean(if_mean ** 2) / n)),
        "median": float(np.sqrt(np.mean(if_med ** 2) / n)),
        "var": float(np.sqrt(np.mean(if_var ** 2) / n)),
        "es": float(np.sqrt(np.mean(if_es ** 2) / n)),
    }


def exp_coverage():
    rng = np.random.default_rng(SEED + 3)
    reps = 1600
    ns = [50, 200, 1000]
    names = ["mean", "median", "var", "es"]
    out = {}
    for dist in ["normal", "t3"]:
        out[dist] = {}
        for n in ns:
            hit = {k: 0 for k in names}
            half_mean = {k: 0.0 for k in names}
            for _ in range(reps):
                if dist == "normal":
                    x = rng.standard_normal(n)
                else:
                    x = rng.standard_t(3.0, n) * np.sqrt(3.0)
                est = measures(x)
                se = _oracle_se_if(x, dist)
                for k in names:
                    half = Z * se[k]
                    half_mean[k] += half
                    if abs(est[k] - _truth(k, dist)) <= half:
                        hit[k] += 1
            out[dist][str(n)] = {
                "cover": {k: hit[k] / reps for k in names},
                "half_width": {k: half_mean[k] / reps for k in names},
            }
    return {"reps": reps, "ns": ns, "coverage": out}


def _truth(k, dist):
    if dist == "normal":
        xi = float(stats.norm.ppf(Q))
        return {
            "mean": 0.0, "median": 0.0, "var": xi,
            "es": -float(stats.norm.pdf(xi)) / Q,
        }[k]
    s = np.sqrt(3.0)
    xi_t = float(stats.t.ppf(Q, 3.0))
    xi = s * xi_t
    es = -(s * (3.0 + xi_t ** 2) / 2.0 * float(stats.t.pdf(xi_t, 3.0))) / Q
    return {"mean": 0.0, "median": 0.0, "var": xi, "es": es}[k]


# ---------------------------------------------------------------- exp 5: tail heaviness
def exp_tail_heaviness():
    rng = np.random.default_rng(SEED + 4)
    n = 400
    nus = [2.5, 3.0, 5.0, 10.0, 30.0]
    rows = []
    for nu in nus:
        x = rng.standard_t(nu, n) * np.sqrt(nu / (nu - 2.0))
        ifs = if_values(x)
        rows.append({
            "nu": nu,
            "mean_if2": float(np.mean(ifs["mean"] ** 2)),
            "es_if2": float(np.mean(ifs["es"] ** 2)),
            "ratio": float(np.mean(ifs["es"] ** 2) / np.mean(ifs["mean"] ** 2)),
        })
    return {"n": n, "rows": rows}


# ---------------------------------------------------------------- exp 6: contamination
def exp_contamination():
    rng = np.random.default_rng(SEED + 5)
    n = 1000
    base = rng.standard_normal(n)
    clean = measures(base)
    eps_grid = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
    rows = []
    for eps in eps_grid:
        k = int(round(eps * n))
        cont = base.copy()
        if k:
            cont[:k] = -10.0
        mc = measures(cont)
        rows.append({
            "eps": eps,
            "mean": mc["mean"] - clean["mean"],
            "median": mc["median"] - clean["median"],
            "var": mc["var"] - clean["var"],
            "es": mc["es"] - clean["es"],
        })
    return {"n": n, "x0": -10.0, "rows": rows}


# ---------------------------------------------------------------- plots
def plot1(r):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rows = r["rows"]
    xs = [q["x0"] for q in rows]
    ax[0].plot(xs, [q["mean"] for q in rows], "o-", color="#c0392b", label="FD mean IF")
    ax[0].plot(xs, r["analytic"]["mean_a"], "--", color="#333", label="theory x")
    ax[0].set_title("Mean IF: unbounded, linear"); ax[0].legend(fontsize=8)
    ax[1].plot(xs, [q["var"] for q in rows], "o-", color="#2980b9", label="FD VaR IF")
    ax[1].plot(xs, r["analytic"]["var_a"], "--", color="#333", label="theory")
    ax[1].set_title("VaR IF: bounded step"); ax[1].legend(fontsize=8)
    ax[2].plot(xs, [q["es"] for q in rows], "o-", color="#8e44ad", label="FD ES IF")
    ax[2].plot(xs, r["analytic"]["es_a"], "--", color="#333", label="theory")
    ax[2].set_title("ES IF: unbounded in left tail"); ax[2].legend(fontsize=8)
    for a in ax:
        a.axhline(0, color="grey", lw=0.6); a.set_xlabel("contaminating point x0")
        a.grid(alpha=0.3)
    fig.suptitle("Finite-difference empirical IF vs analytic formula")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_if_shape.png"), dpi=120)
    plt.close(fig); return "fig1_if_shape.png"


def plot2(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [q["x0"] for q in r["rows"]]
    ax.plot(xs, [q["d_mean"] for q in r["rows"]], "o-", color="#c0392b", label="mean")
    ax.plot(xs, [q["d_median"] for q in r["rows"]], "s-", color="#27ae60", label="median")
    ax.plot(xs, [q["d_var"] for q in r["rows"]], "^-", color="#2980b9", label="VaR95")
    ax.plot(xs, [q["d_es"] for q in r["rows"]], "d-", color="#8e44ad", label="ES95")
    ax.plot(xs, r["theo_mean"], "--", color="#c0392b", alpha=0.5, label="mean theory (x-mu)/(n+1)")
    ax.set_xlabel("single injected loss x0"); ax.set_ylabel("change in estimate")
    ax.set_title("One extreme trading day: who moves, who stands still")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.axhline(0, color="grey", lw=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_single_outlier.png"), dpi=120)
    plt.close(fig); return "fig2_single_outlier.png"


def plot3(r):
    names = ["mean", "median", "var", "es"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, dist in enumerate(["normal", "t3"]):
        d = r["methods"][dist]
        x = np.arange(4); w = 0.26
        for i, (m, c) in enumerate([("if", "#2c3e50"), ("jack", "#e67e22"), ("boot", "#16a085")]):
            vals = [d[m][k] for k in names]
            ax[j].bar(x + (i - 1) * w, vals, w, label=m, color=c)
        ax[j].set_xticks(x); ax[j].set_xticklabels(names)
        ax[j].set_title(f"SE comparison ({dist}, n={r['n']})")
        ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3, axis="y")
    fig.suptitle("IF-asymptotic vs delete-1 jackknife vs iid bootstrap")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_se_methods.png"), dpi=120)
    plt.close(fig); return "fig3_se_methods.png"


def plot4(r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["mean", "median", "var", "es"]
    cols = {"mean": "#c0392b", "median": "#27ae60", "var": "#2980b9", "es": "#8e44ad"}
    for j, dist in enumerate(["normal", "t3"]):
        for k in names:
            vals = [r["coverage"][dist][str(n)]["cover"][k] for n in r["ns"]]
            ax[j].plot(r["ns"], vals, "o-", color=cols[k], label=k)
        ax[j].axhline(CONF, color="grey", ls="--", lw=1, label="95% target")
        ax[j].set_ylim(0.4, 1.02); ax[j].set_xlabel("sample size n")
        ax[j].set_title(f"95% CI coverage ({dist})")
        ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3)
    fig.suptitle("Normal-approx CI built from IF standard errors")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_coverage.png"), dpi=120)
    plt.close(fig); return "fig4_coverage.png"


def plot5(r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = [q["nu"] for q in r["rows"]]
    ax[0].plot(xs, [q["mean_if2"] for q in r["rows"]], "o-", color="#c0392b", label="mean")
    ax[0].plot(xs, [q["es_if2"] for q in r["rows"]], "o-", color="#8e44ad", label="ES")
    ax[0].set_xlabel("t degrees of freedom"); ax[0].set_ylabel("E[IF^2]")
    ax[0].set_title("Asymptotic variance factor"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(xs, [q["ratio"] for q in r["rows"]], "o-", color="#e67e22")
    ax[1].set_xlabel("t degrees of freedom"); ax[1].set_ylabel("ES IF^2 / mean IF^2")
    ax[1].set_title("ES asymptotic variance blow-up vs mean")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Tail heaviness makes ES far more variable than the mean")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_tail.png"), dpi=120)
    plt.close(fig); return "fig5_tail.png"


def main():
    e1 = exp_if_shape()
    e2 = exp_single_outlier()
    e3 = exp_se_methods()
    e4 = exp_coverage()
    e5 = exp_tail_heaviness()
    e6 = exp_contamination()
    results = {
        "setup": {"seed": SEED, "q": Q, "conf": CONF, "z": Z},
        "if_shape": e1,
        "single_outlier": e2,
        "se_methods": e3,
        "coverage": e4,
        "tail": e5,
        "contamination": e6,
    }
    results["figs"] = [plot1(e1), plot2(e2), plot3(e3), plot4(e4), plot5(e5)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
