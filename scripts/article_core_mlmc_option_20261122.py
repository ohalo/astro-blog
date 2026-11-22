# -*- coding: utf-8 -*-
"""Multilevel Monte Carlo for Euler-discretised option prices.

Single source of truth for every number and figure in the article.

GBM with S0=1, r=0.05, sigma=0.2, T=1. Euler scheme at mesh h=2^-l has
first-order weak bias, so plain Monte Carlo at the finest mesh pays for both
statistical error and discretisation bias. MLMC telescoping

    E[P_L] = E[P_0] + sum_{l=1..L} E[P_l - P_{l-1}]

couples each pair through the same Brownian increments (coarse increments are
sums of fine pairs). Levels with small variance need few samples, so the total
cost grows like eps^-2 (log eps)^2 instead of eps^-3.

We measure weak bias, level variances, optimal allocation, actual estimator
RMSE, and the cost exponent for both a continuous call and a discontinuous
cash-or-nothing digital. Black-Scholes closed forms are the reference prices.
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
S0, K, R, SIG, T = 1.0, 1.0, 0.05, 0.20, 1.0
LMAX = 8
EPS_LIST = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
EPS_DEMO = 0.0025
N_REPS = 40

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "multilevel-monte-carlo-option-bias")
os.makedirs(_OUT, exist_ok=True)


# ------------------------------------------------------------------- benchmark
def bs_price(kind):
    d1 = (np.log(S0 / K) + (R + 0.5 * SIG ** 2) * T) / (SIG * np.sqrt(T))
    d2 = d1 - SIG * np.sqrt(T)
    if kind == "call":
        return float(S0 * stats.norm.cdf(d1) - K * np.exp(-R * T) * stats.norm.cdf(d2))
    if kind == "digital":
        return float(np.exp(-R * T) * stats.norm.cdf(d2))
    raise ValueError(kind)


def payoff(kind, ST):
    disc = np.exp(-R * T)
    if kind == "call":
        return disc * np.maximum(ST - K, 0.0)
    return disc * (ST > K).astype(float)


def euler_from_increments(z):
    """z (n,m) standard normals; Euler steps with m steps over T."""
    m = z.shape[1]
    dt = T / m
    s = np.full(z.shape[0], S0)
    for j in range(m):
        s = s * (1.0 + R * dt + SIG * np.sqrt(dt) * z[:, j])
    return s


def single_level(n, m, rng):
    out = []
    for start in range(0, n, 50000):
        b = min(50000, n - start)
        out.append(euler_from_increments(rng.standard_normal((b, m))))
    return np.concatenate(out)


def level_pair(n, l, rng, kind):
    m = 2 ** l
    z = rng.standard_normal((n, m))
    pf = payoff(kind, euler_from_increments(z))
    coz = (z[:, 0::2] + z[:, 1::2]) / np.sqrt(2.0)
    pc = payoff(kind, euler_from_increments(coz))
    d = pf - pc
    return float(np.mean(d)), float(np.var(d, ddof=1))


def pilot(kind, rng):
    """Weak bias and level variances from ONE shared fine Brownian path set.

    All levels are coarsenings of the same 2^LMAX-step increments, so the
    bias differences across levels are coupled and Monte-Carlo noise largely
    cancels. This is essential: independent level samples drown a O(h) call
    bias under O(n^-1/2) noise at fine meshes.
    """
    ref = bs_price(kind)
    n = 200000
    payoff_var = {}
    bias_rows, var_rows = [], []
    sums = np.zeros(LMAX + 1)       # sum of Euler - exact-GBM payoff (CV with beta=1)
    sumsq = np.zeros(LMAX + 1)
    raw_mean = np.zeros(LMAX + 1)
    for start in range(0, n, 40000):
        b = min(40000, n - start)
        z = rng.standard_normal((b, 2 ** LMAX))
        zsum = z.sum(axis=1) / np.sqrt(2 ** LMAX)
        exact_st = S0 * np.exp((R - 0.5 * SIG ** 2) * T + SIG * np.sqrt(T) * zsum)
        exact_p = payoff(kind, exact_st)
        for l in range(LMAX + 1):
            g = 2 ** (LMAX - l)
            zc = z.reshape(b, 2 ** l, g).sum(axis=2) / np.sqrt(g)
            p = payoff(kind, euler_from_increments(zc))
            diff = p - exact_p
            sums[l] += diff.sum()
            sumsq[l] += np.sum(diff ** 2)
            raw_mean[l] += p.sum()
            payoff_var.setdefault(str(l), []).append(p.var(ddof=1))
    for l in range(LMAX + 1):
        bias = sums[l] / n
        se_diff = np.sqrt(max(sumsq[l] / n - bias ** 2, 0) / n)
        bias_rows.append({"l": l, "h": round(2.0 ** -l, 8),
                          "mean": round(float(raw_mean[l] / n), 8),
                          "bias": round(float(bias), 8),
                          "se_shared": round(float(se_diff), 8),
                          "n": int(n)})
        payoff_var[str(l)] = round(float(np.mean(payoff_var[str(l)])), 10)
        v = None
        if l >= 1:
            diffs = []
            for start in range(0, n, 20000):
                z = rng.standard_normal((min(20000, n - start), 2 ** l))
                pf = payoff(kind, euler_from_increments(z))
                coz = (z[:, 0::2] + z[:, 1::2]) / np.sqrt(2.0)
                pc = payoff(kind, euler_from_increments(coz))
                diffs.append(pf - pc)
            d = np.concatenate(diffs)
            v = {"l": l, "mean_diff": round(float(d.mean()), 8),
                 "var_diff": round(float(d.var(ddof=1)), 10), "n": int(n)}
        var_rows.append(v)
    return {"ref": round(ref, 8), "payoff_var": payoff_var,
            "bias": bias_rows, "variance": var_rows}


def fit_slope(xs, ys):
    a = np.polyfit(np.log(np.asarray(xs, float)), np.log(np.abs(np.asarray(ys, float))), 1)
    return {"slope": round(float(a[0]), 6), "intercept": round(float(a[1]), 6)}


def choose_level(bias_rows, eps):
    """Smallest level whose measured bias is at most eps/2."""
    for row in bias_rows:
        if abs(row["bias"]) <= eps / 2.0:
            return row["l"]
    return None


def allocate(var_rows, level, eps, v0):
    """Standard MLMC Neyman allocation: N_l = 2/eps^2 * sqrt(V_l/C_l)*sum sqrt(V_k C_k)."""
    levels = list(range(0, level + 1))
    vv = np.array([v0 if l == 0 else var_rows[l]["var_diff"] for l in levels], float)
    cost_l = np.array([2.0 ** l for l in levels], float)
    s = float(np.sum(np.sqrt(vv * cost_l)))
    ns = (2.0 / eps ** 2) * np.sqrt(vv / cost_l) * s
    ns = np.maximum(np.ceil(ns), 100.0)
    stat_var = float(np.sum(vv / ns))
    cost = float(np.sum(ns * cost_l))
    return {"level": level, "N": {str(l): int(ns[i]) for i, l in enumerate(levels)},
            "stat_var": round(stat_var, 10), "stat_sd": round(float(np.sqrt(stat_var)), 8),
            "cost": round(cost, 2)}


def mlmc_sample(level, ns, rng, kind):
    total = 0.0
    z0 = rng.standard_normal((int(ns["0"]), 1))
    p0 = payoff(kind, euler_from_increments(z0))
    total += float(p0.mean())
    for l in range(1, level + 1):
        n = int(ns[str(l)])
        out = []
        for start in range(0, n, 40000):
            b = min(40000, n - start)
            z = rng.standard_normal((b, 2 ** l))
            pf = payoff(kind, euler_from_increments(z))
            coz = (z[:, 0::2] + z[:, 1::2]) / np.sqrt(2.0)
            pc = payoff(kind, euler_from_increments(coz))
            out.append(pf - pc)
        total += float(np.concatenate(out).mean())
    return total


def plain_sample(level, eps, var_L, rng, kind):
    n = int(np.ceil(4.0 * var_L / eps ** 2))     # sd <= eps/2
    out = []
    for start in range(0, n, 40000):
        b = min(40000, n - start)
        out.append(payoff(kind, single_level(b, 2 ** level, rng)))
    return float(np.concatenate(out).mean()), n


# ------------------------------------------------------------------ experiments
def exp_bias_variance():
    res = {}
    for kind in ["call", "digital"]:
        p = pilot(kind, rng=np.random.default_rng(SEED + (0 if kind == "call" else 1)))
        br = p["bias"]
        vr = [v for v in p["variance"] if v is not None]
        hs = [row["h"] for row in br[1:]]
        bs = [row["bias"] for row in br[1:]]
        vs = [v["var_diff"] for v in vr]
        res[kind] = {
            "ref": p["ref"], "bias": br, "variance": p["variance"],
            "payoff_var": p["payoff_var"],
            "weak_slope": fit_slope(hs, bs),
            "var_slope": fit_slope(hs, vs),
            "max_abs_bias": round(max(abs(x) for x in bs), 8),
        }
    return res


def exp_cost(pv):
    res = {}
    for kind in ["call", "digital"]:
        rows = []
        for eps in EPS_LIST:
            lev = choose_level(pv[kind]["bias"], eps)
            if lev is None or lev < 1:
                rows.append({"eps": eps, "level": lev, "mlmc_cost": None, "plain_cost": None})
                continue
            a = allocate(pv[kind]["variance"], lev, eps, pv[kind]["payoff_var"]["0"])
            # plain MC needs Var(P_L) at the same mesh; estimate from pilot sample variance
            var_payoff = pv[kind]["payoff_var"].get(str(lev))
            n_plain = int(np.ceil(4.0 * var_payoff / eps ** 2))
            plain_cost = n_plain * (2.0 ** lev)
            rows.append({"eps": eps, "level": lev, "mlmc_cost": round(a["cost"], 2),
                         "plain_cost": round(plain_cost, 2),
                         "speedup": round(plain_cost / a["cost"], 3),
                         "mlmc_stat_sd": a["stat_sd"],
                         "bias_at_level": abs(pv[kind]["bias"][lev]["bias"]),
                         "N_total": int(sum(a["N"].values()))})
        xs = [r["eps"] for r in rows if r["mlmc_cost"]]
        ym = [r["mlmc_cost"] for r in rows if r["mlmc_cost"]]
        yp = [r["plain_cost"] for r in rows if r["plain_cost"]]
        res[kind] = {"rows": rows,
                     "mlmc_cost_slope": fit_slope(xs, ym) if xs else None,
                     "plain_cost_slope": fit_slope(xs, yp) if xs else None}
    return res


def exp_validation(pv):
    out = {}
    for kind in ["call", "digital"]:
        lev = choose_level(pv[kind]["bias"], EPS_DEMO)
        rng_k = np.random.default_rng(SEED + (20 if kind == "call" else 21))
        ak = allocate(pv[kind]["variance"], lev, EPS_DEMO, pv[kind]["payoff_var"]["0"])
        ml = [mlmc_sample(lev, ak["N"], rng_k, kind) for _ in range(N_REPS)]
        var_L = pv[kind]["payoff_var"][str(lev)]
        pl, np_plain = [], None
        for _ in range(N_REPS):
            v, np_plain = plain_sample(lev, EPS_DEMO, var_L, rng_k, kind)
            pl.append(v)
        ref = pv[kind]["ref"]
        out[kind] = {
            "level": lev, "eps": EPS_DEMO, "N_plain": int(np_plain),
            "mlmc_mean": round(float(np.mean(ml)), 8),
            "plain_mean": round(float(np.mean(pl)), 8),
            "ref": ref,
            "mlmc_rmse": round(float(np.sqrt(np.mean((np.array(ml) - ref) ** 2))), 8),
            "plain_rmse": round(float(np.sqrt(np.mean((np.array(pl) - ref) ** 2))), 8),
            "mlmc_sd": round(float(np.std(ml, ddof=1)), 8),
            "plain_sd": round(float(np.std(pl, ddof=1)), 8),
            "mlmc_cost": round(ak["cost"], 2),
            "plain_cost": round(np_plain * 2.0 ** lev, 2),
            "n_reps": N_REPS,
            "estimates_mlmc": [round(float(v), 8) for v in ml],
            "estimates_plain": [round(float(v), 8) for v in pl],
        }
    return out


def exp_allocation(pv):
    lev = choose_level(pv["call"]["bias"], EPS_DEMO)
    out = {}
    for kind in ["call", "digital"]:
        a = allocate(pv[kind]["variance"], lev, EPS_DEMO, pv[kind]["payoff_var"]["0"])
        equal_n = int(a["cost"] / sum(2.0 ** l for l in range(0, lev + 1)))
        vv = [pv[kind]["payoff_var"]["0"]] + [pv[kind]["variance"][l]["var_diff"] for l in range(1, lev + 1)]
        eq_var = float(np.sum(np.array(vv, float) / equal_n))
        out[kind] = {"level": lev, "eps": EPS_DEMO, "N_opt": a["N"],
                     "N_equal": equal_n, "opt_sd": a["stat_sd"],
                     "equal_sd": round(float(np.sqrt(eq_var)), 8),
                     "opt_cost": round(a["cost"], 2),
                     "equal_cost": round(a["cost"], 2)}
    return out


# ------------------------------------------------------------------ figures
def plot_bias(pv):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = pv[kind]
        hs = [r["h"] for r in d["bias"][1:]]
        bs = [abs(r["bias"]) for r in d["bias"][1:]]
        a.loglog(hs, bs, "o-", color="#b23a2b")
        a.set_title("%s: weak bias slope %.2f" % (kind, d["weak_slope"]["slope"]))
        a.set_xlabel("mesh h = 2^-l"); a.set_ylabel("|E[P_l] - BS|")
        a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_weak_bias.png"), dpi=115)
    plt.close(fig); return "fig1_weak_bias.png"


def plot_variance(pv):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = pv[kind]
        lv = [v["l"] for v in d["variance"] if v is not None]
        vv = [v["var_diff"] for v in d["variance"] if v is not None]
        md = [abs(v["mean_diff"]) for v in d["variance"] if v is not None]
        a.loglog(lv, vv, "o-", color="#2b5d8a", label="Var(P_l-P_l-1)")
        a.loglog(lv, md, "s--", color="#2e7d55", label="|mean difference|")
        a.set_title("%s: Var slope %.2f" % (kind, d["var_slope"]["slope"]))
        a.set_xlabel("level l"); a.set_ylabel("level difference")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_level_variance.png"), dpi=115)
    plt.close(fig); return "fig2_level_variance.png"


def plot_alloc(al):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = al[kind]
        ls = sorted(int(x) for x in d["N_opt"])
        a.bar(np.array(ls) - 0.18, [d["N_opt"][str(l)] for l in ls], width=0.36,
              color="#b23a2b", label="optimal N_l")
        a.bar(np.array(ls) + 0.18, [d["N_equal"]] * len(ls), width=0.36,
              color="#2b5d8a", alpha=0.6, label="equal N_l")
        a.set_yscale("log"); a.set_title("%s: sd opt %.5f / equal %.5f" %
                                         (kind, d["opt_sd"], d["equal_sd"]))
        a.set_xlabel("level l"); a.set_ylabel("samples (log)"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_allocation.png"), dpi=115)
    plt.close(fig); return "fig3_allocation.png"


def plot_validation(va):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = va[kind]
        a.boxplot([d["estimates_mlmc"], d["estimates_plain"]],
                  tick_labels=["MLMC", "plain MC"])
        a.axhline(d["ref"], color="#2e7d55", ls="--", lw=1.5, label="BS %.5f" % d["ref"])
        a.set_title("%s: RMSE %.5f vs %.5f" % (kind, d["mlmc_rmse"], d["plain_rmse"]))
        a.set_ylabel("estimated price"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_estimator.png"), dpi=115)
    plt.close(fig); return "fig4_estimator.png"


def plot_cost(cost):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = cost[kind]
        rows = [r for r in d["rows"] if r["mlmc_cost"]]
        es = [r["eps"] for r in rows]
        a.loglog(es, [r["plain_cost"] for r in rows], "s-", color="#2b5d8a", label="plain MC")
        a.loglog(es, [r["mlmc_cost"] for r in rows], "o-", color="#b23a2b", label="MLMC")
        a.set_title("%s: slope plain %.2f / MLMC %.2f" %
                    (kind, d["plain_cost_slope"]["slope"], d["mlmc_cost_slope"]["slope"]))
        a.set_xlabel("target RMSE eps"); a.set_ylabel("normal draws")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_cost.png"), dpi=115)
    plt.close(fig); return "fig5_cost.png"


def main():
    pv = exp_bias_variance()
    cost = exp_cost(pv)
    al = exp_allocation(pv)
    va = exp_validation(pv)
    r = {"setup": {"seed": SEED, "S0": S0, "K": K, "r": R, "sigma": SIG, "T": T,
                   "LMAX": LMAX, "eps_demo": EPS_DEMO, "n_reps": N_REPS,
                   "eps_list": EPS_LIST},
         "pilot": pv, "cost": cost, "allocation": al, "validation": va}
    r["figs"] = [plot_bias(pv), plot_variance(pv), plot_alloc(al),
                 plot_validation(va), plot_cost(cost)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
