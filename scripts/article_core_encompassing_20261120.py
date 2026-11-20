# -*- coding: utf-8 -*-
"""Forecast encompassing test: does the good model absorb all information in the weak one?

This script is the single source of truth for every number and figure in the article.
Setup: y = beta*z + eta + u, where z is an AR(1) signal and u is an MA(1) noise.
Model A (restricted) may or may not use z; Model B uses an estimate of z.
Encompassing H0: regressing A's error e1 on (fB - fA) has slope 0.
We compare naive OLS-t (ignores serial correlation) with Newey-West HAC-t.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261120
N = 240
REPS = 1500
PHI_Z = 0.5       # AR(1) coefficient of the signal z (serial correlation in forecast diff)
PSI_U = 0.5       # MA(1) coefficient of noise u (serial correlation in error)
L_HAC = 8         # Newey-West lags
Z95 = 1.959963985

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "public", "images", "forecast-encompassing-test")
os.makedirs(_OUT, exist_ok=True)


def gen(beta, a1, b_bias, phi_z, psi_u, sigma_u, n, rng):
    """Return e1, x=(fB-fA). a1: A's loading on z; B uses beta+b_bias."""
    e_z = rng.standard_normal(n)
    z = np.zeros(n)
    z[0] = e_z[0]
    for t in range(1, n):
        z[t] = phi_z * z[t - 1] + np.sqrt(1 - phi_z ** 2) * e_z[t]
    xi = rng.standard_normal(n)
    u = sigma_u * (xi + psi_u * np.r_[0.0, xi[:-1]])
    eta = rng.standard_normal(n)
    y = beta * z + eta + u
    f_a = a1 * z
    f_b = (beta + b_bias) * z
    e1 = y - f_a
    x = f_b - f_a
    return e1, x


def ols_hac(e1, x):
    """Slope of e1 on x (no intercept) with naive and Newey-West HAC t stats."""
    n = len(x)
    xx = float(x @ x)
    if xx < 1e-12:
        return np.nan, np.nan, np.nan
    b = float(x @ e1) / xx
    resid = e1 - b * x
    s2 = float(resid @ resid) / (n - 1)
    se_naive = np.sqrt(s2 / xx)
    # Newey-West with Bartlett kernel
    g0 = float((resid * x) @ (resid * x))
    var = g0
    for lag in range(1, L_HAC + 1):
        w = 1.0 - lag / (L_HAC + 1.0)
        gl = float(np.sum((resid[:n - lag] * x[:n - lag]) * (resid[lag:] * x[lag:])))
        var += 2.0 * w * gl
    se_hac = np.sqrt(max(var, 1e-12)) / xx
    return b, b / se_naive, b / se_hac


def reject_rate(beta, a1, b_bias, phi_z, psi_u, sigma_u, reps, rng, kind):
    cnt = 0
    for _ in range(reps):
        e1, x = gen(beta, a1, b_bias, phi_z, psi_u, sigma_u, N, rng)
        _, tn, th = ols_hac(e1, x)
        t = tn if kind == "naive" else th
        cnt += (not np.isnan(t)) and (abs(t) > Z95)
    return cnt / reps


def exp_size_scan():
    """H0 (a1=beta): slope=0. Vary serial correlation; naive size should inflate."""
    out = {"psi": [0.0, 0.3, 0.5, 0.7, 0.9], "naive": [], "hac": []}
    rng = np.random.default_rng(SEED + 1)
    for psi in out["psi"]:
        out["naive"].append(reject_rate(0.4, 0.4, 0.25, PHI_Z, psi, 1.0, REPS, rng, "naive"))
        out["hac"].append(reject_rate(0.4, 0.4, 0.25, PHI_Z, psi, 1.0, REPS, rng, "hac"))
    return out


def exp_power_scan():
    """H1 (a1=0): B has incremental info beta; power vs beta at two noise levels."""
    out = {"beta": [0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.62],
           "hac_s1": [], "naive_s1": [], "hac_s2": []}
    rng = np.random.default_rng(SEED + 2)
    for beta in out["beta"]:
        out["hac_s1"].append(reject_rate(beta, 0.0, 0.15, PHI_Z, PSI_U, 1.0, REPS, rng, "hac"))
        out["naive_s1"].append(reject_rate(beta, 0.0, 0.15, PHI_Z, PSI_U, 1.0, REPS, rng, "naive"))
        out["hac_s2"].append(reject_rate(beta, 0.0, 0.15, PHI_Z, PSI_U, 2.0, REPS, rng, "hac"))
    return out


def exp_slope_dist():
    """Distribution of slope estimate under H0 and H1."""
    rng = np.random.default_rng(SEED + 3)
    h0, h1 = [], []
    for _ in range(REPS):
        e1, x = gen(0.4, 0.4, 0.25, PHI_Z, PSI_U, 1.0, N, rng)
        b, _, _ = ols_hac(e1, x)
        h0.append(b)
        e1b, xb = gen(0.4, 0.0, 0.15, PHI_Z, PSI_U, 1.0, N, rng)
        bb, _, _ = ols_hac(e1b, xb)
        h1.append(bb)
    return {"h0_mean": float(np.mean(h0)), "h0_sd": float(np.std(h0)),
            "h1_mean": float(np.mean(h1)), "h1_sd": float(np.std(h1)),
            "h1_theory": 0.4 / (0.4 + 0.15), "h0": h0, "h1": h1}


def exp_corr_power():
    """As B's estimate gets noisier (larger b_bias), the encompassing slope falls as
    beta/(beta+b) while HAC power stays flat (slope and its SE shrink in the same
    proportion). The economically meaningful share of incremental information shrinks
    even though the test still rejects: statistical significance != economic value."""
    out = {"b_bias": [0.0, 0.2, 0.4, 0.7, 1.0, 1.5], "power": [], "slope": []}
    rng = np.random.default_rng(SEED + 4)
    for bb in out["b_bias"]:
        out["power"].append(reject_rate(0.4, 0.0, bb, PHI_Z, PSI_U, 1.0, REPS, rng, "hac"))
        out["slope"].append(0.4 / (0.4 + bb))
    return out


def plot1():
    rng = np.random.default_rng(SEED + 5)
    e1, x = gen(0.4, 0.0, 0.15, PHI_Z, PSI_U, 1.0, N, rng)
    z = np.zeros(N); ez = rng.standard_normal(N); z[0] = ez[0]
    for t in range(1, N):
        z[t] = PHI_Z * z[t - 1] + np.sqrt(1 - PHI_Z ** 2) * ez[t]
    y = e1  # a1=0 -> e1 = y
    fb = 0.55 * z
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(y, lw=1, color="#2c3e50", label="y (target)")
    axes[0].plot(fb, lw=1, color="#c0392b", label="f_B (uses z)")
    axes[0].axhline(0, color="k", lw=.6)
    axes[0].set_title("Fig1: A ignores z, B uses it; B's forecast tracks the signal")
    axes[0].legend(fontsize=8, loc="upper right")
    e_b = e1 - fb  # a1=0 so e1=y; f_b=(beta+b_bias) z = 0.55 z
    axes[1].plot(e1, lw=1, color="#2980b9", label="e_A = y - 0")
    axes[1].plot(e_b, lw=1, color="#c0392b", label="e_B = y - f_B")
    axes[1].axhline(0, color="k", lw=.6)
    axes[1].set_title("Fig1: forecast errors - B removes the z component from A's error")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_paths.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_paths.png"


def plot2(sz):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(sz["psi"])); w = .38
    ax.bar(x - w / 2, sz["naive"], w, label="naive OLS-t", color="#c0392b")
    ax.bar(x + w / 2, sz["hac"], w, label="HAC-t (Newey-West)", color="#27ae60")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in sz["psi"]])
    ax.set_xlabel("MA(1) coefficient of error psi")
    ax.set_ylabel("size (false rejection)")
    ax.set_title("Fig2: serial correlation inflates naive size; HAC restores it")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_size.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_size.png"


def plot3(pw):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(pw["beta"], pw["hac_s1"], "o-", color="#27ae60", lw=2, label="HAC, sigma=1")
    ax.plot(pw["beta"], pw["naive_s1"], "s--", color="#c0392b", lw=1.5, label="naive, sigma=1")
    ax.plot(pw["beta"], pw["hac_s2"], "^:", color="#8e44ad", lw=2, label="HAC, sigma=2")
    ax.axhline(0.05, color="k", ls="--", lw=1)
    ax.set_xlabel("B's incremental signal beta")
    ax.set_ylabel("power")
    ax.set_title("Fig3: power rises with incremental info; noise halves detection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_power.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_power.png"


def plot4(dist):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(dist["h0"], bins=45, density=True, alpha=.6, color="#2980b9", label="H0 slope")
    ax.hist(dist["h1"], bins=45, density=True, alpha=.6, color="#c0392b", label="H1 slope")
    ax.axvline(0, color="#2980b9", ls="--", lw=1.5)
    ax.axvline(dist["h1_theory"], color="#c0392b", ls="--", lw=1.5,
               label="theory beta/(beta+b)")
    ax.set_xlabel("encompassing slope estimate")
    ax.set_ylabel("density")
    ax.set_title("Fig4: slope centers on 0 under H0, on beta/(beta+b) under H1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_slope_dist.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_slope_dist.png"


def plot5(cp):
    fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
    ax1.plot(cp["b_bias"], cp["power"], "o-", color="#c0392b", lw=2, label="HAC power")
    ax1.set_xlabel("B's estimation bias b (B drifts from A)")
    ax1.set_ylabel("power", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(cp["b_bias"], cp["slope"], "s--", color="#2980b9", lw=2,
             label="theory slope beta/(beta+b)")
    ax2.set_ylabel("encompassing slope", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_title("Fig5: a noisier B shrinks the slope but not the test power - significance != value")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_corr_power.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_corr_power.png"


def main():
    rng = np.random.default_rng(SEED)
    e1, x = gen(0.4, 0.0, 0.15, PHI_Z, PSI_U, 1.0, N, rng)
    _, tn, th = ols_hac(e1, x)
    sz = exp_size_scan()
    pw = exp_power_scan()
    dist = exp_slope_dist()
    cp = exp_corr_power()
    results = {
        "setup": {"seed": SEED, "n": N, "reps": REPS, "phi_z": PHI_Z,
                  "psi_u": PSI_U, "L_hac": L_HAC, "z95": Z95},
        "example_t": {"naive": float(tn), "hac": float(th)},
        "size": sz, "power": pw,
        "slope_dist": {k: v for k, v in dist.items() if k not in ("h0", "h1")},
        "corr_power": cp,
    }
    results["figs"] = [plot1(), plot2(sz), plot3(pw), plot4(dist), plot5(cp)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
