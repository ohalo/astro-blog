# -*- coding: utf-8 -*-
"""Predictive regression: Stambaugh endogeneity bias and its correction.

Single source of truth for every number and figure in the article.

DGP (true parameters known), the Stambaugh (1999) setup:
    x_t = phi * x_{t-1} + u_t,     u_t ~ N(0,1)
    v_t = delta * u_t + sqrt(1-delta^2) * e_t     (v_t correlated with u_t)
    r_t = alpha + beta * x_{t-1} + v_t

Because x_{t-1} is independent of u_t, OLS is CONSISTENT: the bias is a
finite-sample O(1/n) effect, not an omitted-variable inconsistency. But the
bias is systematic (sign flips with delta, grows with phi) and, more
importantly, the t-test on beta=0 is size distorted when delta != 0 -- the
classic "spurious predictability" trap. We compare three slope estimates:
    OLS, Stambaugh's analytic bias correction (delta-hat plugged in),
    and a residual-bootstrap bias correction,
with sweeps over delta, phi, and n. Fixed seeds keep exec-replay identical to
results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "predictive-regression-stambaugh-bias")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.0


def simulate(n, phi, delta, beta, rng, warm=120):
    """Return x (n,), r (n,) under the Stambaugh setup (same-t u/v correlation)."""
    total = n + warm
    u = rng.standard_normal(total)
    e = rng.standard_normal(total)
    v = delta * u + np.sqrt(1.0 - delta ** 2) * e
    x = np.empty(total)
    x[0] = u[0] / np.sqrt(1.0 - phi ** 2)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + u[t]
    r = ALPHA + beta * np.concatenate([[0.0], x[:-1]]) + v
    return x[warm:], r[warm:]


def ols_fit(x_lag, r):
    """OLS slope of r on x_lag; returns (b, se, t, residual)."""
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef = np.linalg.lstsq(X, r, rcond=None)[0]
    resid = r - X @ coef
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return coef[1], se, coef[1] / se if se > 0 else 0.0, resid


def aux_resid(x, r):
    """Return u_hat (from x_t~x_{t-1}) and v_hat (from r_t~x_{t-1}), both len n-1, same-t."""
    n = len(x)
    x_lag, x_now, rr = x[:-1], x[1:], r[1:]
    Xa = np.column_stack([np.ones(n - 1), x_lag])
    ca = np.linalg.lstsq(Xa, x_now, rcond=None)[0]
    u_hat = x_now - Xa @ ca
    Xr = np.column_stack([np.ones(n - 1), x_lag])
    cr = np.linalg.lstsq(Xr, rr, rcond=None)[0]
    v_hat = rr - Xr @ cr
    return u_hat, v_hat, ca[1]


def stambaugh_corrected(x, r):
    """b_sc = b_ols - delta_hat*(sigma_v/sigma_u)*(1+phi_hat)/n."""
    x_lag, rr = x[:-1], r[1:]
    b_ols, _, _, _ = ols_fit(x_lag, rr)
    n = len(rr)
    u_hat, v_hat, phi_hat = aux_resid(x, r)
    denom = np.sqrt((u_hat @ u_hat) * (v_hat @ v_hat))
    delta_hat = float(u_hat @ v_hat / denom) if denom > 0 else 0.0
    sd_u = np.sqrt(u_hat @ u_hat / n)
    sd_v = np.sqrt(v_hat @ v_hat / n)
    phi_c = float(np.clip(phi_hat, -0.999, 0.999))
    # OLS slope bias runs opposite in sign to delta; subtract delta-hat term in that direction
    bias_hat = -delta_hat * (sd_v / sd_u) * (1.0 + phi_c) / n
    return b_ols - bias_hat, delta_hat, phi_hat


def bootstrap_corrected(x, r, n_boot=149, seed=0):
    """Residual bootstrap bias correction: b_bc = 2*b_ols - mean(b_boot*)."""
    rng = np.random.default_rng(seed)
    n = len(x) - 1
    x_lag, rr = x[:-1], r[1:]
    b_ols, _, _, _ = ols_fit(x_lag, rr)
    u_hat, v_hat, phi_hat = aux_resid(x, r)
    u_hat = u_hat - u_hat.mean()
    v_hat = v_hat - v_hat.mean()
    bs = np.empty(n_boot)
    for b in range(n_boot):
        us = rng.choice(u_hat, size=n, replace=True)
        vs = rng.choice(v_hat, size=n, replace=True)
        xs = np.empty(n + 1)
        xs[0] = 0.0
        for t in range(1, n + 1):
            xs[t] = phi_hat * xs[t - 1] + us[t - 1]
        rs = b_ols * xs[:-1] + vs
        bs[b] = ols_fit(xs[:-1], rs)[0]
    return float(2.0 * b_ols - bs.mean())


def one_rep(n, phi, delta, beta, rng, boot_seed=0):
    x, r = simulate(n, phi, delta, beta, rng)
    x_lag, rr = x[:-1], r[1:]
    b_ols, se, t_ols, _ = ols_fit(x_lag, rr)
    b_sc, dh, ph = stambaugh_corrected(x, r)
    b_bc = bootstrap_corrected(x, r, n_boot=149, seed=boot_seed)
    return b_ols, t_ols, b_sc, b_bc, dh, ph


def summarize(n, phi, delta, beta, R, seed=SEED):
    rng = np.random.default_rng(seed)
    B = np.empty((R, 6))
    for i in range(R):
        B[i] = one_rep(n, phi, delta, beta, rng, boot_seed=seed + i)
    out = {
        "n": n, "phi": phi, "delta": delta, "beta": beta, "R": R,
        "delta_hat_mean": float(B[:, 4].mean()),
        "phi_hat_mean": float(B[:, 5].mean()),
    }
    for nm, bi in zip(["ols", "stambaugh", "bootstrap"], [0, 2, 3]):
        out[nm + "_bias"] = float((B[:, bi] - beta).mean())
        out[nm + "_rmse"] = float(np.sqrt(((B[:, bi] - beta) ** 2).mean()))
    out["ols_t_reject"] = float(np.mean(np.abs(B[:, 1]) > 1.96))
    return out


def exp_delta_bias():
    return [summarize(200, 0.9, d, 0.0, 400, seed=SEED + int((d + 1) * 100))
            for d in (-0.8, -0.4, 0.0, 0.4, 0.8)]


def exp_size():
    return [summarize(200, p, dl, 0.0, 400, seed=SEED + int((p + 1) * 1000 + (dl + 1) * 10))
            for p, dl in ((0.5, 0.0), (0.9, 0.0), (0.98, 0.0),
                          (0.9, -0.8), (0.9, 0.8), (0.98, -0.8), (0.98, 0.8))]


def exp_sample_size():
    return [summarize(nn, 0.95, -0.7, 0.0, 400, seed=SEED + nn)
            for nn in (50, 100, 200, 400, 800)]


def exp_phi_bias():
    return [summarize(200, p, -0.7, 0.0, 400, seed=SEED + int(p * 1000))
            for p in (0.0, 0.3, 0.6, 0.85, 0.95, 0.98)]


# ------------------------------------------------------------------- figures
COL = ["#e8684a", "#5b8ff9", "#2ca25f"]
LAB = ["OLS", "Stambaugh", "Bootstrap"]
KEYS = ["ols", "stambaugh", "bootstrap"]


def _lines(ax, rows, xkey, field, title, xlabel, ylabel, hline=None):
    xs = [r[xkey] for r in rows]
    for k, nm in enumerate(KEYS):
        ax.plot(xs, [r[nm + field] for r in rows], "o-", color=COL[k], label=LAB[k])
    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)


def fig1_delta(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _lines(ax, rows, "delta", "_bias",
           "OLS slope bias flips sign with delta (beta=0, phi=0.9)",
           "delta = corr(u_t, v_t)", "mean beta-hat (truth=0)", hline=0.0)
    fig.tight_layout(); p = "fig1_delta_bias.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig2_size(rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = ["phi=%.2f\nd=%+.1f" % (r["phi"], r["delta"]) for r in rows]
    vals = [r["ols_t_reject"] for r in rows]
    colors = ["#888888" if r["delta"] == 0.0 else "#e8684a" for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="5% nominal size")
    ax.set_ylabel("P(|t|>1.96) under beta=0"); ax.set_title("Spurious predictability: t-test size")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = "fig2_size_distortion.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig3_sample(rows):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    _lines(ax[0], rows, "n", "_bias", "Bias shrinks as n grows (phi=0.95, delta=-0.7)",
           "sample size n", "mean beta-hat", hline=0.0)
    _lines(ax[1], rows, "n", "_rmse", "RMSE also falls with n", "sample size n", "RMSE")
    fig.tight_layout(); p = "fig3_sample_size.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig4_phi(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _lines(ax, rows, "phi", "_bias", "Persistence phi magnifies the finite-sample bias",
           "phi", "mean beta-hat (delta=-0.7)", hline=0.0)
    fig.tight_layout(); p = "fig4_phi_bias.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig5_scatter():
    rng = np.random.default_rng(SEED + 54321)
    n = 120
    x, r = simulate(n, 0.95, -0.7, 0.0, rng)
    x_lag, rr = x[:-1], r[1:]
    b_ols, se, t_ols, _ = ols_fit(x_lag, rr)
    b_sc, _, _ = stambaugh_corrected(x, r)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x_lag, rr, s=16, color="#5b8ff9", alpha=0.5)
    xs = np.linspace(x_lag.min(), x_lag.max(), 50)
    ax.plot(xs, b_ols * xs, color="#e8684a", lw=2,
            label="OLS slope=%.3f (t=%.2f)" % (b_ols, t_ols))
    ax.plot(xs, b_sc * xs, color="#2ca25f", lw=2,
            label="Stambaugh slope=%.3f" % b_sc)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("x_{t-1} (persistent predictor)")
    ax.set_ylabel("r_t")
    ax.set_title("Spurious predictability in one sample")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = "fig5_spurious_scatter.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p, [float(b_ols), float(b_sc), float(t_ols)]


def main():
    de = exp_delta_bias()
    sz = exp_size()
    nn = exp_sample_size()
    ph = exp_phi_bias()
    f5, f5vals = fig5_scatter()
    r = {
        "setup": {"seed": SEED, "model": "x_t=phi*x_{t-1}+u_t; v_t=delta*u_t+sqrt(1-delta^2)*e_t; r_t=alpha+beta*x_{t-1}+v_t",
                  "estimators": LAB},
        "delta_bias": de,
        "size": sz,
        "sample_size": nn,
        "phi_bias": ph,
        "one_sample": {"ols_slope": f5vals[0], "stambaugh_slope": f5vals[1],
                       "ols_t": f5vals[2]},
    }
    r["figs"] = [fig1_delta(de), fig2_size(sz), fig3_sample(nn),
                 fig4_phi(ph), f5]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
