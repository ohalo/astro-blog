# -*- coding: utf-8 -*-
"""Heteroskedasticity-robust standard errors HC0-HC3 vs wild bootstrap: leverage breaks HC0.

Single source of truth for all numbers and figures in the article.
DGP: y_i = b0 + b1*x_i + eps_i, eps_i ~ N(0, sigma_i^2), sigma_i grows with |x_i|.
A high-leverage point (one x far out) with high variance is the killer combination.
We compare OLS classical SE, HC0, HC1, HC2, HC3 and Rademacher wild bootstrap
on size of t-test for b1, and on SE accuracy vs the Monte-Carlo truth.
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

SEED = 20261120
REPS = 2000
B_WILD = 399
Z95 = stats.norm.ppf(0.975)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "public", "images", "heteroskedasticity-robust-se-wild-bootstrap")
os.makedirs(_OUT, exist_ok=True)


def design(n, lever, rng):
    """x with one high-leverage point at `lever` (in sd units); hetero sigma=|x|-scaled."""
    x = rng.standard_normal(n)
    x[-1] = lever
    sigma = 0.5 + 0.5 * np.abs(x)
    X = np.column_stack([np.ones(n), x])
    return x, sigma, X


def fit_se(x, sigma, X, rng):
    """Return b1 and SEs from OLS / HC0-HC3 for one dataset with b1=0 true."""
    n = len(x)
    eps = rng.normal(0, sigma)
    y = eps  # b0=b1=0 under H0
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    se_ols = np.sqrt(XtX_inv[1, 1] * (resid @ resid) / (n - 2))
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat0 = (X * resid[:, None]).T @ (X * resid[:, None])
    v0 = XtX_inv @ meat0 @ XtX_inv
    se_hc0 = np.sqrt(v0[1, 1])
    se_hc1 = se_hc0 * np.sqrt(n / (n - 2))
    meat2 = (X * (resid / np.sqrt(1 - h))[:, None]).T @ (X * (resid / np.sqrt(1 - h))[:, None])
    se_hc2 = np.sqrt((XtX_inv @ meat2 @ XtX_inv)[1, 1])
    meat3 = (X * (resid / (1 - h))[:, None]).T @ (X * (resid / (1 - h))[:, None])
    se_hc3 = np.sqrt((XtX_inv @ meat3 @ XtX_inv)[1, 1])
    return beta[1], dict(ols=se_ols, hc0=se_hc0, hc1=se_hc1, hc2=se_hc2, hc3=se_hc3), \
        X, XtX_inv, resid, h


def wild_pvalue(y, X, XtX_inv, resid, B, rng):
    """Rademacher wild bootstrap, restricted residuals, two-sided p for b1."""
    n = len(X)
    Xr = X[:, [0]]  # restricted design (b1=0)
    beta_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
    rres = y - Xr @ beta_r
    t_obs = None
    # observed using HC0-ish baseline stat on full model
    beta_full = XtX_inv @ (X.T @ y)
    rf = y - X @ beta_full
    meat = (X * rf[:, None]).T @ (X * rf[:, None])
    se = np.sqrt((XtX_inv @ meat @ XtX_inv)[1, 1])
    if se < 1e-12:
        return 1.0
    t_obs = abs(beta_full[1] / se)
    count = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=n)
        yb = Xr @ beta_r + (rres * w)
        bb = XtX_inv @ (X.T @ yb)
        rb = yb - X @ bb
        meatb = (X * rb[:, None]).T @ (X * rb[:, None])
        seb = np.sqrt((XtX_inv @ meatb @ XtX_inv)[1, 1])
        if seb > 1e-12 and abs(bb[1] / seb) >= t_obs:
            count += 1
    return (count + 1) / (B + 1)


def size_experiment(lever, n, reps, wild=False, seed=SEED):
    rng = np.random.default_rng(seed)
    rej = {k: 0 for k in ["ols", "hc0", "hc1", "hc2", "hc3"]}
    if wild:
        rej["wild"] = 0
    for _ in range(reps):
        x, sigma, X = design(n, lever, rng)
        b1, ses, X, XtX_inv, resid, h = fit_se(x, sigma, X, rng)
        y = X @ np.array([0.0, b1]) + resid  # reconstruct y from fitted pieces
        for k, se in ses.items():
            rej[k] += abs(b1 / se) > Z95
        if wild:
            p = wild_pvalue(y, X, XtX_inv, resid, B_WILD, rng)
            rej["wild"] += p < 0.05
    return {k: v / reps for k, v in rej.items()}


def se_accuracy(lever, n, reps, seed=SEED + 1):
    """Compare mean estimated SE vs Monte-Carlo SD of b1 (the truth)."""
    rng = np.random.default_rng(seed)
    ses_sum = {k: 0.0 for k in ["ols", "hc0", "hc1", "hc2", "hc3"]}
    bs = []
    for _ in range(reps):
        x, sigma, X = design(n, lever, rng)
        b1, ses, _, _, _, _ = fit_se(x, sigma, X, rng)
        bs.append(b1)
        for k in ses_sum:
            ses_sum[k] += ses[k]
    true_sd = float(np.std(bs))
    return {"true_sd": true_sd,
            "mean_se": {k: v / reps for k, v in ses_sum.items()}}


def plot1_rows():
    rng = np.random.default_rng(SEED + 2)
    n = 60
    x, sigma, X = design(n, 3.0, rng)
    eps = rng.normal(0, sigma)
    y = eps
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sc = ax.scatter(x, y, c=sigma, cmap="viridis", s=35, edgecolor="k", linewidth=.3)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("x (last point = high leverage)")
    ax.set_ylabel("residual")
    ax.set_title("Fig1: heteroskedasticity - spread grows with |x|; far point has high leverage")
    fig.colorbar(sc, ax=ax, label="sigma_i")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_hetero.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_hetero.png"


def plot2_size(size_res):
    keys = ["ols", "hc0", "hc1", "hc2", "hc3"]
    labels = ["OLS", "HC0", "HC1", "HC2", "HC3"]
    levers = [float(x) for x in size_res["lever"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, lab in zip(keys, labels):
        ax.plot(levers, size_res[k], "o-", lw=2, label=lab)
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    ax.set_xlabel("leverage of the far point (sd units)")
    ax.set_ylabel("size of t-test for b1")
    ax.set_title("Fig2: high leverage inflates OLS and HC0 size; HC3 stays closest to 5%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_size_leverage.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_size_leverage.png"


def plot3_se(acc):
    keys = ["ols", "hc0", "hc1", "hc2", "hc3"]
    labels = ["OLS", "HC0", "HC1", "HC2", "HC3"]
    levers = [float(x) for x in acc["lever"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, lab in zip(keys, labels):
        ratios = [acc["rows"][i]["mean_se"][k] / acc["rows"][i]["true_sd"]
                  for i in range(len(levers))]
        ax.plot(levers, ratios, "o-", lw=2, label=lab)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="true SE")
    ax.set_xlabel("leverage of the far point")
    ax.set_ylabel("mean estimated SE / true MC SD")
    ax.set_title("Fig3: OLS/HC0 underestimate SE as leverage rises; HC3 stays near the truth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_se_ratio.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_se_ratio.png"


def plot4_wild(wild_res):
    methods = ["ols", "hc0", "hc3", "wild"]
    labels = ["OLS", "HC0", "HC3", "Wild boot"]
    xs = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = [wild_res["size"][m] for m in methods]
    bars = ax.bar(xs, vals, color=["#c0392b", "#e67e22", "#27ae60", "#2980b9"])
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    for xi, v in zip(xs, vals):
        ax.text(xi, v + .005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("size")
    ax.set_title("Fig4: wild bootstrap (0.055) beats HC3 (0.098) size with n=30 + leverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_wild.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_wild.png"


def plot5_nscan(nscan):
    ns = [int(x) for x in nscan["n"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for k, lab, c in [("ols", "OLS", "#c0392b"), ("hc0", "HC0", "#e67e22"),
                      ("hc3", "HC3", "#27ae60")]:
        ax.plot(ns, nscan[k], "o-", lw=2, color=c, label=lab)
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    ax.set_xlabel("sample size n (leverage fixed)")
    ax.set_ylabel("size")
    ax.set_title("Fig5: OLS leverage size bias persists at n=400; HC0/HC3 converge to 5%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_nscan.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_nscan.png"


def main():
    # Size vs leverage
    levers = [1.0, 2.0, 3.0, 4.0, 5.0]
    size_rows = []
    for lv in levers:
        size_rows.append(size_experiment(lv, 40, REPS))
    size_res = {"lever": levers}
    for k in ["ols", "hc0", "hc1", "hc2", "hc3"]:
        size_res[k] = [r[k] for r in size_rows]

    # SE accuracy vs leverage
    acc_rows = []
    for lv in levers:
        acc_rows.append(se_accuracy(lv, 40, 800))
    acc = {"lever": levers, "rows": acc_rows}

    # Wild bootstrap at n=30, leverage=4
    wild = {"size": size_experiment(4.0, 30, 400, wild=True, seed=SEED + 5)}

    # n scan at fixed leverage 3
    ns = [30, 50, 100, 200, 400]
    nscan = {"n": ns}
    tmp = {k: [] for k in ["ols", "hc0", "hc3"]}
    for n in ns:
        r = size_experiment(3.0, n, 800, seed=SEED + 6)
        for k in tmp:
            tmp[k].append(r[k])
    nscan.update(tmp)

    results = {
        "setup": {"seed": SEED, "reps": REPS, "B_wild": B_WILD, "z95": float(Z95)},
        "size_leverage": size_res,
        "se_accuracy": {"lever": levers,
                        "rows": [{"lever": lv,
                                  "true_sd": acc_rows[i]["true_sd"],
                                  "mean_se": acc_rows[i]["mean_se"]}
                                 for i, lv in enumerate(levers)]},
        "wild_size": wild["size"],
        "n_scan": nscan,
    }
    results["figs"] = [plot1_rows(), plot2_size(size_res), plot3_se(acc),
                       plot4_wild(wild), plot5_nscan(nscan)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
