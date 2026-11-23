# -*- coding: utf-8 -*-
"""Horseshoe prior for sparse alpha: shrinkage, FDR, misspecification, and OOS.

Single source of truth for every number and figure in the article.

Normal means  y_i = alpha_i + eps_i,  eps_i ~ N(0, sigma^2),  i = 1..p.
Prior (Carvalho, Polson & Scott 2010):
    alpha_i | lambda_i, tau ~ N(0, sigma^2 lambda_i^2 tau^2)
    lambda_i ~ C+(0,1),  tau ~ C+(0, tau0)
Gibbs sampler uses auxiliary variables for the half-Cauchy priors:
    1/lambda_i^2 | a_i ~ Exp(1/a_i) ;  a_i ~ InvGamma(1/2, 1)
    1/tau^2 | a_tau ~ Exp(1/a_tau)  ;  a_tau ~ InvGamma(1/2, 1)
    alpha_i | . ~ N((1-kappa_i) y_i, sigma^2 (1-kappa_i)),
        kappa_i = 1/(1+lambda_i^2 tau^2)
Comparators: hard threshold, universal soft threshold sqrt(2 log p),
empirical-Bayes two-group normal means (spike-and-slab with EM), BH at q=0.1.

Fixed seeds keep every replay byte-identical to results.json.
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
P = 120
SIGMA = 1.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "horseshoe-prior-sparse-alpha")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------- sampler
def horseshoe_gibbs(y, sigma=1.0, tau0=None, n_iter=2500, burn=1000,
                    thin=3, seed=0, return_all=False):
    """Return posterior mean of alpha, mean shrinkage kappa, and P(|alpha|>thr).

    Standard auxiliary-variable Gibbs for the half-Cauchy prior:
        lambda_i^2 | phi_i ~ InvGamma(1/2, 1/phi_i),  phi_i ~ InvGamma(1/2, 1)
        tau^2      | psi   ~ InvGamma(1/2, 1/psi),    psi   ~ InvGamma(1/2, 1/tau0^2)
    Full conditionals (Carvalho, Polson & Scott 2010):
        alpha_i | . ~ N((1-kappa_i) y_i, sigma^2 (1-kappa_i)),
                        kappa_i = 1/(1+lambda_i^2 tau^2)
        1/lambda_i^2 | . ~ Exp(1/phi_i + alpha_i^2/(2 sigma^2 tau^2))
        1/tau^2      | . ~ Exp(1/psi   + sum(alpha_i^2)/(2 sigma^2 lambda_i^2))
        phi_i ~ InvGamma(1, 1 + 1/lambda_i^2)
        psi   ~ InvGamma(1, 1/tau0^2 + 1/tau^2)
    """
    rng = np.random.default_rng(seed)
    p = len(y)
    if tau0 is None:
        tau0 = max(1e-4, 1.0 / np.sqrt(p))
    lam2 = np.ones(p)
    phi = np.ones(p)
    tau2 = float(tau0 ** 2)
    psi = 1.0
    alpha = y.copy()
    keep, kaps = [], []
    for it in range(n_iter):
        kappa = 1.0 / (1.0 + lam2 * tau2)
        alpha = (1.0 - kappa) * y + np.sqrt(sigma ** 2 * (1.0 - kappa)) * rng.standard_normal(p)
        # xi_i = 1/lambda_i^2 ~ Exp(rate_i)
        rate_lam = 1.0 / phi + alpha ** 2 / (2.0 * sigma ** 2 * tau2)
        xi_lam = rng.exponential(scale=1.0 / np.maximum(rate_lam, 1e-12))
        lam2 = 1.0 / np.maximum(xi_lam, 1e-12)
        # phi | lambda ~ InvGamma(1, 1 + 1/lambda^2) -> 1/phi ~ Gamma(1, 1 + xi)
        inv_phi = rng.gamma(shape=1.0, scale=1.0 / (1.0 + xi_lam))
        phi = 1.0 / np.maximum(inv_phi, 1e-12)
        # xi_tau = 1/tau^2 ~ Exp(rate_tau)
        rate_tau = 1.0 / psi + np.sum(alpha ** 2 / (2.0 * sigma ** 2 * lam2))
        # tau^2 | alpha, lambda, psi: shape = (p+1)/2 (p likelihoods + 1 prior)
        xi_tau = rng.gamma(shape=(p + 1) / 2.0,
                           scale=1.0 / max(rate_tau, 1e-12))
        tau2 = 1.0 / max(xi_tau, 1e-12)
        # psi ~ InvGamma(1, 1/tau0^2 + 1/tau^2) -> 1/psi ~ Gamma(1, 1/tau0^2 + xi_tau)
        inv_psi = rng.gamma(shape=1.0, scale=1.0 / (1.0 / tau0 ** 2 + xi_tau))
        psi = 1.0 / max(inv_psi, 1e-12)
        if it >= burn and (it - burn) % thin == 0:
            keep.append(alpha.copy())
            kaps.append(1.0 / (1.0 + lam2 * tau2))
    am = np.mean(keep, axis=0)
    km = np.mean(kaps, axis=0)
    if return_all:
        return am, km, np.asarray(keep)
    return am, km, None


# ------------------------------------------------------------------ comparators
def soft_threshold(y, t):
    return np.sign(y) * np.maximum(np.abs(y) - t, 0.0)


def hard_threshold(y, t):
    return np.where(np.abs(y) > t, y, 0.0)


def eb_two_group(y, sigma=1.0, n_iter=300):
    """EM for pi ~ Beta, slab variance v estimated, spike at zero."""
    p = len(y)
    pi = 0.1
    v = max(1.0, np.var(y) - sigma ** 2)
    for _ in range(n_iter):
        # log odds of slab vs spike under N(0, sigma^2 + v)
        var = sigma ** 2 + v
        ll = -0.5 * np.log(var) - 0.5 * y ** 2 / var
        l0 = -0.5 * np.log(sigma ** 2) - 0.5 * y ** 2 / sigma ** 2
        w = 1.0 / (1.0 + np.exp(np.clip(l0 - ll + np.log((1 - pi) / pi), -500, 500)))
        pi_n = np.clip(np.mean(w), 1e-5, 0.999)
        v_n = max(1e-4, np.sum(w * (y ** 2 - sigma ** 2)) / max(np.sum(w), 1e-8))
        if max(abs(pi_n - pi), abs(v_n - v)) < 1e-9:
            pi, v = pi_n, v_n
            break
        pi, v = pi_n, v_n
    post_w = 1.0 / (1.0 + np.exp(np.clip(
        -0.5 * np.log(sigma ** 2) - 0.5 * y ** 2 / sigma ** 2
        + 0.5 * np.log(sigma ** 2 + v) + 0.5 * y ** 2 / (sigma ** 2 + v)
        + np.log((1 - pi) / pi), -500, 500)))
    return post_w * y * v / (sigma ** 2 + v), post_w


def bh_select(y, sigma=1.0, q=0.1):
    pv = 2 * stats.norm.sf(np.abs(y) / sigma)
    order = np.argsort(pv)
    crit = q * (np.arange(1, len(y) + 1) / len(y))
    ok = pv[order] <= crit
    sel = np.zeros(len(y), bool)
    if ok.any():
        sel[order[: np.where(ok)[0].max() + 1]] = True
    return sel


def mse_of(est, alpha):
    return float(np.mean((est - alpha) ** 2))


def make_alpha(p, s0, rng, mag=2.2):
    alpha = np.zeros(p)
    act = rng.choice(p, s0, replace=False)
    signs = rng.choice([-1, 1], s0)
    alpha[act] = signs * mag * rng.uniform(0.7, 1.3, s0)
    return alpha, np.sort(act)


# ------------------------------------------------------------------ experiments
def exp_shrinkage():
    """Posterior mean curves on a y grid: the horseshoe shape vs other rules."""
    grid = np.linspace(-5, 5, 201)
    a_hs, km, _ = horseshoe_gibbs(grid, n_iter=2200, burn=800, thin=2, seed=SEED + 9)
    a_eb, _ = eb_two_group(grid)
    return {"y": [float(v) for v in grid],
            "horseshoe": [float(v) for v in a_hs],
            "soft": [float(v) for v in soft_threshold(grid, np.sqrt(2 * np.log(P)))],
            "hard": [float(v) for v in hard_threshold(grid, 2.0)],
            "eb": [float(v) for v in a_eb]}


def exp_sparsity():
    rows = []
    for s0 in [2, 5, 10, 20, 40, 60]:
        acc = {k: [] for k in ["hs", "soft", "hard", "eb"]}
        rng = np.random.default_rng(SEED + 100 + s0)
        for r in range(30):
            alpha, act = make_alpha(P, s0, rng)
            y = alpha + rng.standard_normal(P)
            a_hs, _, _ = horseshoe_gibbs(y, n_iter=1800, burn=700, thin=4,
                                         seed=SEED + 4000 + s0 * 100 + r)
            a_eb, _ = eb_two_group(y)
            acc["hs"].append(mse_of(a_hs, alpha))
            acc["soft"].append(mse_of(soft_threshold(y, np.sqrt(2 * np.log(P))), alpha))
            acc["hard"].append(mse_of(hard_threshold(y, 2.0), alpha))
            acc["eb"].append(mse_of(a_eb, alpha))
        rows.append({"s0": s0,
                     "hs": float(np.mean(acc["hs"])),
                     "soft": float(np.mean(acc["soft"])),
                     "hard": float(np.mean(acc["hard"])),
                     "eb": float(np.mean(acc["eb"]))})
    return rows


def exp_detection():
    s0 = 10
    rng = np.random.default_rng(SEED + 55)
    acc = {k: {"tp": [], "fp": []} for k in ["hs_k", "bh", "hard", "eb"]}
    for r in range(40):
        alpha, act = make_alpha(P, s0, rng)
        y = alpha + rng.standard_normal(P)
        a_hs, km, _ = horseshoe_gibbs(y, n_iter=1800, burn=700, thin=4,
                                      seed=SEED + 7000 + r)
        a_eb, pw = eb_two_group(y)
        for key, sel in [("hs_k", km < 0.5), ("bh", bh_select(y)),
                         ("hard", np.abs(y) > 2.0), ("eb", pw > 0.5)]:
            tp = int(np.sum(sel[act])); fp = int(np.sum(sel) - tp)
            acc[key]["tp"].append(tp / s0)
            acc[key]["fp"].append(fp / max(1, tp + fp))
    return {k: {"power": float(np.mean(v["tp"])),
                "fdr": float(np.mean(v["fp"]))} for k, v in acc.items()}


def exp_misspec():
    rows = []
    s0 = 10
    tau0_true = s0 / P
    for tau0 in [tau0_true / 5, tau0_true, tau0_true * 5, 0.5]:
        for sig in [0.8, 1.0, 1.3]:
            rng = np.random.default_rng(SEED + 200 + int(tau0 * 1000) + int(sig * 10))
            hs_mse, bh_fd = [], []
            for r in range(18):
                alpha, act = make_alpha(P, s0, rng)
                y = alpha + rng.standard_normal(P) * sig
                a_hs, _, _ = horseshoe_gibbs(y, sigma=sig, tau0=tau0,
                                             n_iter=1600, burn=600, thin=4,
                                             seed=SEED + 9000 + r)
                hs_mse.append(mse_of(a_hs, alpha))
                sel = bh_select(y, sig)
                tp = np.sum(sel[act]); fp = np.sum(sel) - tp
                bh_fd.append(fp / max(1, tp + fp))
            rows.append({"tau0": float(tau0), "sigma": sig,
                         "hs_mse": float(np.mean(hs_mse)),
                         "bh_fdr": float(np.mean(bh_fd))})
    return rows


def exp_oos():
    s0 = 10
    rng = np.random.default_rng(SEED + 33)
    rows = []
    for method in ["hs", "hard", "soft", "bh"]:
        ins, oos, nsel = [], [], []
        for r in range(20):
            alpha, act = make_alpha(P, s0, rng)
            y = alpha + rng.standard_normal(P)
            if method == "hs":
                est, _, _ = horseshoe_gibbs(y, n_iter=1800, burn=700, thin=4,
                                            seed=SEED + 11000 + r)
            elif method == "hard":
                est = hard_threshold(y, 2.0)
            elif method == "soft":
                est = soft_threshold(y, np.sqrt(2 * np.log(P)))
            else:
                sel = bh_select(y); est = np.where(sel, y, 0.0)
            sel = np.abs(est) > 0.3
            nsel.append(int(np.sum(sel)))
            if np.sum(sel) == 0:
                oos.append(0.0); ins.append(0.0); continue
            w = np.sign(est[sel])
            w[w == 0] = 1.0
            w = w / np.sum(sel)
            r_in = alpha[sel] * w
            at = alpha + rng.standard_normal(P)
            r_out = at[sel] * w
            if np.std(r_in) < 1e-9 or np.std(r_out) < 1e-9:
                ins.append(0.0); oos.append(0.0); continue
            ins.append(float(np.mean(r_in) / np.std(r_in)))
            oos.append(float(np.mean(r_out) / np.std(r_out)))
        rows.append({"method": method,
                     "in_sharpe": float(np.mean(ins)),
                     "oos_sharpe": float(np.mean(oos)),
                     "gap": float(np.mean(ins) - np.mean(oos)),
                     "nsel": float(np.mean(nsel))})
    return rows


# ---------------------------------------------------------------------- figures
def fig1(d):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(d["y"], d["horseshoe"], lw=2.5, c="#c0392b", label="horseshoe mean")
    ax.plot(d["y"], d["soft"], c="#2980b9", label="soft threshold")
    ax.plot(d["y"], d["hard"], c="#7f8c8d", label="hard threshold")
    ax.plot(d["y"], d["eb"], ls=":", c="#8e44ad", label="EB two-group")
    ax.plot(d["y"], d["y"], ls="--", c="black", lw=0.7, alpha=0.5)
    ax.set_xlabel("observed y"); ax.set_ylabel("posterior mean / estimate")
    ax.set_title("Horseshoe: shrink noise, leave large signals alone")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig1_shrinkage.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_shrinkage.png"


def fig2(rows):
    xs = [r["s0"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, c in [("hs", "#c0392b"), ("eb", "#8e44ad"),
                 ("soft", "#2980b9"), ("hard", "#7f8c8d")]:
        ax.plot(xs, [r[k] for r in rows], "o-", c=c, label=k)
    ax.set_xlabel("number of true signals s0"); ax.set_ylabel("MSE on alpha")
    ax.set_title("Horseshoe wins when signals are sparse")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig2_sparsity.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_sparsity.png"


def fig3(d):
    names = list(d.keys())
    power = [d[k]["power"] for k in names]
    fdr = [d[k]["fdr"] for k in names]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, power, w, color="#2980b9", label="power")
    ax.bar(x + w / 2, fdr, w, color="#c0392b", label="FDR")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("kappa<0.5 is not an FDR procedure")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig3_detection.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_detection.png"


def fig4(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    tau_vals = sorted(set(r["tau0"] for r in rows))
    for sig in [0.8, 1.0, 1.3]:
        sub = [r for r in rows if r["sigma"] == sig]
        ax.plot([r["tau0"] for r in sub], [r["hs_mse"] for r in sub],
                "o-", label="sigma=%.1f" % sig)
    ax.set_xscale("log"); ax.set_xlabel("tau0 (assumed sparsity)")
    ax.set_ylabel("horseshoe MSE")
    ax.set_title("Global scale matters more than noise misspecification")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig4_misspec.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_misspec.png"


def fig5(rows):
    names = [r["method"] for r in rows]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, [r["in_sharpe"] for r in rows], w, color="#c0392b", label="in-sample")
    ax.bar(x + w / 2, [r["oos_sharpe"] for r in rows], w, color="#2980b9", label="OOS")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Sharpe-like ratio")
    ax.set_title("Selection optimism survives every shrinkage rule")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig5_oos.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_oos.png"


def main():
    sh = exp_shrinkage()
    sp = exp_sparsity()
    de = exp_detection()
    mi = exp_misspec()
    oo = exp_oos()
    result = {"setup": {"seed": SEED, "p": P, "sigma": SIGMA,
                        "sampler": "Gibbs, 1800 iters / 700 burn / thin 4",
                        "comparators": ["hard |y|>2", "soft sqrt(2log p)",
                                        "EB two-group EM", "BH q=0.1"]},
              "shrinkage": sh, "sparsity": sp, "detection": de,
              "misspec": mi, "oos": oo}
    result["figs"] = [fig1(sh), fig2(sp), fig3(de), fig4(mi), fig5(oo)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
