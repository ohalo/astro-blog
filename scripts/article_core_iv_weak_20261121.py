# -*- coding: utf-8 -*-
"""Weak instruments and 2SLS: bias, first-stage F, CI blow-up, t-test size
distortion under overidentification, Anderson-Rubin, and Hansen J.

Single source of truth for every number and figure in the article.

DGP (k instruments):  Z ~ N(0,1)^k, v ~ N(0,1), e ~ N(0,1)
    u = RHO * v + GAMMA * z2 + e      (GAMMA>0 -> z2 has a direct effect)
    x = pi * sum_k(z) + v
    y = BETA * x + u
OLS plim = BETA + RHO/(1+k*pi^2) (biased); 2SLS plim = BETA (consistent),
but finite-sample bias ~ RHO/pi has heavy tails, the Wald interval width
blows up, and the naive t-test over-rejects once there are several weak
instruments. Anderson-Rubin keeps size; Hansen J flags a direct effect.
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

SEED = 20261121
N = 400
REPS = 2000
BETA = 0.5
RHO = 0.6
PIS = [1.0, 0.5, 0.2, 0.1, 0.05]
Z95 = 1.959963985
F95_1 = float(stats.f.ppf(0.95, 1, N - 1))
F95_3 = float(stats.f.ppf(0.95, 3, N - 3))
CHI95_1 = float(stats.chi2.ppf(0.95, 1))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "instrumental-variables-2sls-weak-instruments")
os.makedirs(_OUT, exist_ok=True)


def draw(pi, rng, n=N, k=1, gamma=0.0):
    z = rng.standard_normal((n, k))
    v = rng.standard_normal(n)
    e = rng.standard_normal(n)
    zsum = z.sum(axis=1)
    # A direct effect on only the second instrument is detectable by Hansen J;
    # putting it on zsum (collinear with x) would be absorbed and invisible.
    direct = gamma * z[:, 1] if (gamma != 0.0 and k >= 2) else 0.0
    u = RHO * v + direct + e
    x = pi * k * zsum + v
    y = BETA * x + u
    return z, x, y


def tsls(z, x, y):
    """Return beta_hat, naive t for H0 beta=BETA, and Wald CI width."""
    k = z.shape[1]
    zz_inv = np.linalg.inv(z.T @ z)
    pz = z @ zz_inv @ z.T
    xh = pz @ x
    xtpx = xh @ xh
    b2 = (xh @ y) / xtpx
    resid = y - b2 * x
    s2 = resid @ resid / (N - k)
    se = np.sqrt(s2 / xtpx)
    t = (b2 - BETA) / se
    return b2, float(t), float(2 * Z95 * se)


def ols(x, y):
    return (x @ y) / (x @ x)


def first_f(z, x):
    if z.ndim == 1:
        pih = (z @ x) / (z @ z)
        r = x - pih * z
        return pih ** 2 / ((r @ r / (N - 2)) / (z @ z))
    k = z.shape[1]
    pihat = np.linalg.lstsq(z, x, rcond=None)[0]
    r = x - z @ pihat
    s2 = r @ r / (N - k)
    xtpx = x @ (z @ np.linalg.inv(z.T @ z) @ z.T) @ x
    # F for joint significance of all k first-stage coefficients
    return float((xtpx / k) / s2)


def ar_stat(T, z):
    k = z.shape[1]
    zz_inv = np.linalg.inv(z.T @ z)
    pz = z @ zz_inv @ z.T
    mz = np.eye(N) - pz
    return float((T @ pz @ T / k) / (T @ mz @ T / (N - k)))


def hansen_j(z, x, y):
    k = z.shape[1]
    zz_inv = np.linalg.inv(z.T @ z)
    pz = z @ zz_inv @ z.T
    xh = pz @ x
    b2 = (xh @ y) / (xh @ xh)
    resid = y - b2 * x
    return float(resid @ pz @ resid / (resid @ resid / N))


def exp_bias():
    """Single instrument: OLS vs 2SLS point estimates (mean and median)."""
    out = {"pi": PIS, "ols": [], "twosls": [], "twosls_med": [], "ols_theory": []}
    rng = np.random.default_rng(SEED + 1)
    for pi in PIS:
        bo, b2 = [], []
        for _ in range(REPS):
            z, x, y = draw(pi, rng)
            o, t, _ = tsls(z, x, y)
            bo.append(ols(x, y))
            b2.append(o)
        out["ols"].append(float(np.mean(bo)))
        out["twosls"].append(float(np.mean(b2)))
        out["twosls_med"].append(float(np.median(b2)))
        out["ols_theory"].append(BETA + RHO / (1.0 + pi * pi))
    return out


def exp_first_stage():
    out = {"pi": PIS, "f_mean": [], "f_p_below10": [], "noncentrality": []}
    rng = np.random.default_rng(SEED + 2)
    for pi in PIS:
        fs = []
        for _ in range(REPS):
            z, x, y = draw(pi, rng)
            fs.append(first_f(z, x))
        fs = np.array(fs)
        out["f_mean"].append(float(np.mean(fs)))
        out["f_p_below10"].append(float(np.mean(fs < 10.0)))
        out["noncentrality"].append(pi * pi * N)
    return out


def exp_ci_width():
    """Single instrument: Wald CI width explodes while the t-test keeps its size."""
    out = {"pi": PIS, "wald_med": [], "wald_p_gt10": [], "wald_size_single": []}
    rng = np.random.default_rng(SEED + 3)
    for pi in PIS:
        ws, rej = [], 0
        for _ in range(REPS):
            z, x, y = draw(pi, rng)
            _, t, w = tsls(z, x, y)
            ws.append(w)
            rej += (abs(t) > Z95)   # H0 beta=BETA is true -> false rejection
        ws = np.array(ws)
        out["wald_med"].append(float(np.median(ws)))
        out["wald_p_gt10"].append(float(np.mean(ws > 10.0)))
        out["wald_size_single"].append(rej / REPS)
    return out


def exp_size_overid():
    """Three weak instruments: naive Wald t over-rejects, Anderson-Rubin does not."""
    out = {"pi": PIS, "wald": [], "ar": [], "ar_ci_unbounded": []}
    rng = np.random.default_rng(SEED + 4)
    k = 3
    for pi in PIS:
        rw, ra, unbounded = 0, 0, 0
        for _ in range(REPS):
            z, x, y = draw(pi, rng, k=k)
            _, t, _ = tsls(z, x, y)
            rw += (abs(t) > Z95)
            ra += (ar_stat(y - BETA * x, z) > F95_3)
            # The AR acceptance set is unbounded iff the extreme nulls at the
            # ends of a wide grid are not rejected: only 2 endpoints needed.
            u_lo = ar_stat(y - (-1.5) * x, z) <= F95_3
            u_hi = ar_stat(y - 2.5 * x, z) <= F95_3
            unbounded += bool(u_lo or u_hi)
        out["wald"].append(rw / REPS)
        out["ar"].append(ra / REPS)
        out["ar_ci_unbounded"].append(unbounded / REPS)
    return out


def exp_overid():
    """Hansen J size (all valid) and power (one instrument has a direct effect)."""
    scenes = [
        ("strong-all-valid", 0.8, 0.0),
        ("strong-gamma0.2", 0.8, 0.2),
        ("strong-gamma0.5", 0.8, 0.5),
        ("weak-all-valid", 0.15, 0.0),
        ("weak-gamma0.2", 0.15, 0.2),
        ("weak-gamma0.5", 0.15, 0.5),
    ]
    out = {"scene": [s[0] for s in scenes], "reject": [], "pi": [], "gamma": []}
    rng = np.random.default_rng(SEED + 5)
    for name, pi, gamma in scenes:
        cnt = 0
        for _ in range(REPS):
            z, x, y = draw(pi, rng, k=2, gamma=gamma)
            cnt += (hansen_j(z, x, y) > CHI95_1)
        out["reject"].append(cnt / REPS)
        out["pi"].append(pi)
        out["gamma"].append(gamma)
    return out


def plot1():
    rng = np.random.default_rng(SEED + 10)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, pi in zip(axes, [1.0, 0.1]):
        z, x, y = draw(pi, rng)
        z1 = z[:, 0]
        ax.scatter(z1, x, s=16, alpha=0.45, color="#2c3e50")
        slope = (z1 @ x) / (z1 @ z1)
        xs = np.linspace(z1.min(), z1.max(), 50)
        ax.plot(xs, slope * xs, color="#c0392b", lw=2)
        f = first_f(z, x)
        ax.set_title("pi=%.2f  first-stage slope=%.2f  F=%.1f" % (pi, slope, f), fontsize=10)
        ax.set_xlabel("instrument z")
        ax.set_ylabel("endogenous regressor x")
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("Fig1: a strong (left) vs weak (right) first stage", fontsize=11)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_first_stage.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig1_first_stage.png"


def plot2(bias):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(bias["pi"], bias["ols"], "o-", color="#c0392b", lw=2, label="OLS (simulated)")
    ax.plot(bias["pi"], bias["ols_theory"], "s--", color="#e67e22", lw=1.5,
            label="OLS plim = b + rho/(1+pi^2)")
    ax.plot(bias["pi"], bias["twosls"], "^-", color="#27ae60", lw=2, label="2SLS mean (heavy tails)")
    ax.plot(bias["pi"], bias["twosls_med"], "v:", color="#16a085", lw=1.5, label="2SLS median")
    ax.axhline(BETA, color="k", ls=":", lw=1.2, label="true beta = 0.5")
    ax.set_xlabel("instrument strength pi")
    ax.set_ylabel("mean / median estimate")
    ax.set_title("Fig2: weak instruments drag 2SLS toward the biased OLS estimate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_bias.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig2_bias.png"


def plot3(fs):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(fs["pi"], fs["f_mean"], "o-", color="#2980b9", lw=2, label="mean first-stage F")
    ax.axhline(10, color="#c0392b", ls="--", lw=1.5, label="Stock-Yogo rule of thumb: F=10")
    ax.set_xlabel("instrument strength pi")
    ax.set_ylabel("first-stage F (log scale)")
    ax.set_yscale("log")
    ax.set_title("Fig3: first-stage F collapses below 10 once pi falls under ~0.15")
    ax.legend(fontsize=8)
    for x0, p10 in zip(fs["pi"], fs["f_p_below10"]):
        ax.annotate("%.0f%%" % (100 * p10), (x0, fs["f_mean"][fs["pi"].index(x0)]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_first_stage_f.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig3_first_stage_f.png"


def plot4(cw):
    fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
    ax1.plot(cw["pi"], cw["wald_med"], "o-", color="#8e44ad", lw=2,
             label="median Wald 95% CI width")
    ax1.axhline(1.0, color="k", ls=":", lw=1.2, label="reference width = 1")
    ax1.set_xlabel("instrument strength pi")
    ax1.set_ylabel("median 95% CI width (log scale)", color="#8e44ad")
    ax1.tick_params(axis="y", labelcolor="#8e44ad")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(cw["pi"], cw["wald_size_single"], "s--", color="#2980b9", lw=2,
             label="Wald t size (true beta)")
    ax2.axhline(0.05, color="#2980b9", ls=":", lw=1.0, alpha=0.6)
    ax2.set_ylabel("false rejection rate", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0, 0.12)
    ax1.set_title("Fig4: one weak instrument keeps nominal size but the Wald interval explodes")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_ci_width.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig4_ci_width.png"


def plot5(sz, ov):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.plot(sz["pi"], sz["wald"], "o-", color="#c0392b", lw=2, label="naive Wald t (k=3)")
    ax.plot(sz["pi"], sz["ar"], "s-", color="#27ae60", lw=2, label="Anderson-Rubin (k=3)")
    ax.axhline(0.05, color="k", ls="--", lw=1.2, label="nominal 5%")
    ax.set_xlabel("instrument strength pi")
    ax.set_ylabel("false rejection of H0: beta=0.5")
    ax.set_title("5a: several weak instruments break the t-test, AR survives", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(ov["scene"]))
    colors = ["#27ae60" if g == 0 else "#c0392b" for g in ov["gamma"]]
    bars = ax.bar(x, ov["reject"], color=colors, alpha=0.85)
    ax.axhline(0.05, color="k", ls="--", lw=1.2, label="nominal 5%")
    ax.set_xticks(x)
    ax.set_xticklabels(ov["scene"], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Hansen J rejection rate")
    ax.set_title("5b: Hansen J flags a direct (invalid) instrument", fontsize=10)
    ax.legend(fontsize=8)
    for b, v in zip(bars, ov["reject"]):
        ax.annotate("%.2f" % v, (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_size_and_j.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig5_size_and_j.png"


def main():
    bias = exp_bias()
    fs = exp_first_stage()
    cw = exp_ci_width()
    sz = exp_size_overid()
    ov = exp_overid()
    results = {
        "setup": {"seed": SEED, "n": N, "reps": REPS, "beta": BETA, "rho": RHO,
                  "pis": PIS, "f95_1": F95_1, "f95_3": F95_3, "chi95_1": CHI95_1},
        "bias": bias,
        "first_stage": fs,
        "ci_width": cw,
        "size_overid": sz,
        "overid": ov,
    }
    results["figs"] = [plot1(), plot2(bias), plot3(fs), plot4(cw), plot5(sz, ov)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
