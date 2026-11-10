"""Block bootstrap CI coverage under serial dependence.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_block_bootstrap_20261110.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
All numbers quoted in the article come from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as student_t

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261110
ANN = 252.0

# ---------- data generating process ----------
MU = 0.0004
PHI = 0.35
OMEGA = 5.76e-6
ALPHA = 0.08
BETA = 0.88
VAR_EPS = OMEGA / (1.0 - ALPHA - BETA)          # 0.000144
VAR_U = VAR_EPS / (1.0 - PHI ** 2)              # AR(1) driven by an MDS
TRUE_SR = MU / np.sqrt(VAR_U) * np.sqrt(ANN)
TRUE_MEAN = MU * ANN


def simulate(rng, n, phi=PHI, innov="normal", burn=500):
    """AR(1) conditional mean + GARCH(1,1) variance, both with known truth."""
    tot = n + burn
    z = rng.standard_normal(tot)
    if innov == "t6":
        # Same shock ranks as the normal case, mapped through the t(6) quantile
        # and rescaled to unit variance. Keeps the two robustness runs paired.
        z = student_t.ppf(norm.cdf(z), df=6) / np.sqrt(6.0 / 4.0)
    s2 = VAR_EPS
    u = np.empty(tot)
    state = 0.0
    for t in range(tot):
        e = np.sqrt(s2) * z[t]
        state = phi * state + e
        u[t] = state
        s2 = OMEGA + ALPHA * e * e + BETA * s2
    return MU + u[burn:]


# ---------- resampling schemes (all vectorised over B replicates) ----------
def iid_idx(rng, n, B):
    return rng.integers(0, n, size=(B, n))


def mbb_idx(rng, n, B, L):
    """Moving block bootstrap: starts only in 0..n-L, no wrap, truncated to n."""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, k * L)
    return idx[:, :n]


def cbb_idx(rng, n, B, L):
    """Circular block bootstrap: starts anywhere, wrap around modulo n."""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, k * L)
    return idx[:, :n] % n


def stationary_idx(rng, n, B, L):
    """Stationary bootstrap: geometric(p=1/L) block lengths, circular starts.

    Vectorised carry-forward trick: encode (block_id * n + random_start) at
    every block start; because block_id strictly increases, a running maximum
    recovers the current block's start without any Python loop.
    """
    p = 1.0 / L
    new = rng.random((B, n)) < p
    new[:, 0] = True
    pos = np.arange(n)[None, :]
    tag = np.cumsum(new, axis=1)
    rand_start = rng.integers(0, n, size=(B, n))
    encoded = np.where(new, tag * n + rand_start, 0)
    carry = np.maximum.accumulate(encoded, axis=1)
    start = carry % n
    first_pos = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    offset = pos - first_pos
    return (start + offset) % n


def sr_rows(X):
    m = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)
    return m / s * np.sqrt(ANN)


def mean_rows(X):
    return X.mean(axis=1) * ANN


def pct_ci(stat, level=0.95):
    lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


# ---------- analytic standard errors ----------
def lrv_nw(psi, L):
    """Newey-West long-run variance with Bartlett kernel."""
    n = len(psi)
    psi = psi - psi.mean()
    out = float(np.dot(psi, psi) / n)
    for k in range(1, L + 1):
        gk = float(np.dot(psi[k:], psi[:-k]) / n)
        out += 2.0 * (1.0 - k / (L + 1.0)) * gk
    return max(out, 1e-14)


def analytic_ci(x, L=5):
    """Return (sr_ci, mean_ci) from naive iid SE and HAC/delta-method SE."""
    n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    sr = mu / sd * np.sqrt(ANN)
    # iid delta method: Var(S_daily) = (1 + S_daily^2 / 2) / n
    se_naive_sr = np.sqrt((1.0 + 0.5 * (mu / sd) ** 2) / n) * np.sqrt(ANN)
    se_naive_mean = sd / np.sqrt(n) * ANN
    # influence function of the Sharpe ratio
    psi_sr = (x - mu) / sd - mu / (2.0 * sd ** 3) * ((x - mu) ** 2 - sd ** 2)
    se_hac_sr = np.sqrt(ANN * lrv_nw(psi_sr, L) / n)
    se_hac_mean = np.sqrt(ANN ** 2 * lrv_nw(x, L) / n)
    z = 1.959963984540054
    return {
        "naive_sr": (sr - z * se_naive_sr, sr + z * se_naive_sr),
        "hac_sr": (sr - z * se_hac_sr, sr + z * se_hac_sr),
        "naive_mean": (mu * ANN - z * se_naive_mean, mu * ANN + z * se_naive_mean),
        "hac_mean": (mu * ANN - z * se_hac_mean, mu * ANN + z * se_hac_mean),
    }


def contains(ci, truth):
    return bool(ci[0] <= truth <= ci[1])


def width(ci):
    return float(ci[1] - ci[0])


# ---------- one Monte-Carlo section ----------
L_GRID = [1, 3, 5, 10, 20, 40]
METHODS = ["iid", "mbb", "cbb", "stationary"]


def run_section(section_id, M, B, n, L_values, phi=PHI, innov="normal"):
    """Empirical coverage of 95% CIs over M independent paths.

    Every method/scan uses the same M simulated paths (common random numbers),
    so paired differences between methods are measured with low noise.
    """
    keys = [("iid", 1)]
    for m in ["mbb", "cbb", "stationary"]:
        for L in L_values:
            keys.append((m, L))
    keys += [("naive", 0), ("hac", 5)]
    acc = {k: {"sr": [], "mean": [], "w_sr": [], "w_mean": [], "bias": []} for k in keys}

    for rep in range(M):
        rng_path = np.random.default_rng([SEED, section_id, 0, rep])
        x = simulate(rng_path, n, phi=phi, innov=innov)
        sr_hat = sr_rows(x[None, :])[0]
        for mi, (method, L) in enumerate(keys):
            if method in ("naive", "hac"):
                cis = analytic_ci(x, L=5)
                ci_sr = cis["naive_sr"] if method == "naive" else cis["hac_sr"]
                ci_mean = cis["naive_mean"] if method == "naive" else cis["hac_mean"]
                boot_sr = None
            else:
                rng_b = np.random.default_rng([SEED, section_id, mi + 1, rep])
                if method == "iid":
                    idx = iid_idx(rng_b, n, B)
                elif method == "mbb":
                    idx = mbb_idx(rng_b, n, B, L)
                elif method == "cbb":
                    idx = cbb_idx(rng_b, n, B, L)
                else:
                    idx = stationary_idx(rng_b, n, B, L)
                X = x[idx]
                boot_sr = sr_rows(X)
                boot_mean = mean_rows(X)
                ci_sr = pct_ci(boot_sr)
                ci_mean = pct_ci(boot_mean)
            a = acc[(method, L)]
            a["sr"].append(contains(ci_sr, TRUE_SR))
            a["mean"].append(contains(ci_mean, TRUE_MEAN))
            a["w_sr"].append(width(ci_sr))
            a["w_mean"].append(width(ci_mean))
            a["bias"].append(float(np.median(boot_sr) - sr_hat) if boot_sr is not None else 0.0)

    out = {}
    for k, a in acc.items():
        out[f"{k[0]}_{k[1]}"] = {
            "coverage_sr": round(float(np.mean(a["sr"])), 4),
            "coverage_mean": round(float(np.mean(a["mean"])), 4),
            "width_sr": round(float(np.mean(a["w_sr"])), 4),
            "width_mean": round(float(np.mean(a["w_mean"])), 4),
            "median_bootstrap_bias_sr": round(float(np.mean(a["bias"])), 4),
        }
    return out


def run_hac_scan(M=400, n=250, lags=(1, 3, 5, 10, 20, 40)):
    """The analytic alternative has a bandwidth too. Coverage vs NW lag.

    Reuses main's [SEED, 1, 0, rep] paths so lag=5 matches the main table.
    """
    acc = {L: [] for L in lags}
    naive = []
    for rep in range(M):
        rng = np.random.default_rng([SEED, 1, 0, rep])
        x = simulate(rng, n)
        cis = analytic_ci(x, L=1)
        naive.append(contains(cis["naive_sr"], TRUE_SR))
        for L in lags:
            cis = analytic_ci(x, L=L)
            acc[L].append(contains(cis["hac_sr"], TRUE_SR))
    return {"naive": round(float(np.mean(naive)), 4),
            **{str(L): round(float(np.mean(v)), 4) for L, v in acc.items()}}


def validate_stationary(B=2000, cols=4000):
    """Sanity check: geometric(1/L) blocks really have mean length L."""
    rng = np.random.default_rng([SEED, 8, 8])
    out = {}
    for L in [5, 10, 20, 40]:
        new = rng.random((B, cols)) < 1.0 / L
        new[:, 0] = True
        lens = []
        for b in range(0, B, 4):
            st = np.flatnonzero(new[b])
            lens.extend(np.diff(st).tolist())
        out[str(L)] = {"empirical_mean": round(float(np.mean(lens)), 3),
                       "theoretical_mean": L,
                       "n_blocks_measured": len(lens)}
    return out


def run_all(verbose=True):
    stats = {
        "meta": {
            "seed": SEED,
            "mu": MU, "phi": PHI, "omega": OMEGA, "alpha": ALPHA, "beta": BETA,
            "var_eps": VAR_EPS, "var_u": VAR_U,
            "true_annual_sharpe": round(float(TRUE_SR), 6),
            "true_annual_mean": round(float(TRUE_MEAN), 6),
            "L_grid": L_GRID,
            "main": {"M": 400, "B": 250, "n": 250},
            "n_scan": {"M": 400, "B": 250, "n_values": [125, 250, 500, 1000],
                       "note": "n=250 reuses the main experiment's paths"},
            "phi_scan": {"M": 400, "B": 250, "phi_values": [0.0, 0.2, 0.4, 0.6]},
            "tail_scan": {"M": 400, "B": 250, "innovations": ["normal", "t6"],
                          "note": "t6 is a rank-preserving transform of main's shocks"},
            "hac_scan": {"M": 400, "n": 250, "lags": [1, 3, 5, 10, 20, 40],
                         "note": "reuses main's paths; lag=5 matches main table"},
        }
    }

    main = run_section(1, M=400, B=250, n=250, L_values=L_GRID)
    stats["main"] = main
    if verbose:
        print("[1/5] main L-grid done")
        for k in ["iid_1", "mbb_10", "cbb_10", "stationary_10", "naive_0", "hac_5"]:
            print("   ", k, main[k])

    n_scan = {}
    for n in [125, 250, 500, 1000]:
        if n == 250:
            r = main  # identical paths, identical bootstrap draws, identical numbers
        else:
            r = run_section(2, M=400, B=250, n=n, L_values=[10])
        n_scan[str(n)] = {k: r[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "cbb_10", "stationary_10", "naive_0", "hac_5"]}
        n_scan[str(n)]["width_mbb_10"] = r["mbb_10"]["width_sr"]
    stats["n_scan"] = n_scan
    if verbose:
        print("[2/5] sample-size scan done:", n_scan)

    phi_scan = {}
    for phi in [0.0, 0.2, 0.4, 0.6]:
        r = run_section(3, M=400, B=250, n=250, L_values=[10], phi=phi)
        phi_scan[str(phi)] = {k: r[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "cbb_10", "stationary_10", "naive_0", "hac_5"]}
    stats["phi_scan"] = phi_scan
    if verbose:
        print("[3/5] phi scan done:", phi_scan)

    # t6 innovations are a rank-preserving transform of main's shocks, and use
    # the same [SEED, 1, 0, rep] path seeds: the comparison is exactly paired.
    r_t6 = run_section(1, M=400, B=250, n=250, L_values=[10], innov="t6")
    tail_scan = {
        "normal": {k: main[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "hac_5"]},
        "t6": {k: r_t6[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "hac_5"]},
    }
    stats["tail_scan"] = tail_scan
    if verbose:
        print("[4/5] innovation scan done:", tail_scan)

    hac_scan = run_hac_scan(M=400, n=250)
    stats["hac_scan"] = hac_scan
    stats["stationary_block_length_check"] = validate_stationary()
    if verbose:
        print("[5/5] HAC lag scan done:", hac_scan)
        print("      stationary block-length check:", stats["stationary_block_length_check"])

    # representative path + ACF for figure 1
    rng_rep = np.random.default_rng([SEED, 9, 9])
    rep_path = simulate(rng_rep, 750)
    def acf(v, lags):
        v = v - v.mean()
        den = float(np.dot(v, v))
        return [float(np.dot(v[k:], v[:-k]) / den) if k > 0 else 1.0 for k in lags]
    lags = list(range(0, 21))
    figs = {
        "path": rep_path.tolist(),
        "acf_ret": acf(rep_path, lags),
        "acf_sq": acf(rep_path ** 2, lags),
        "lags": lags,
        "main": main, "n_scan": n_scan, "phi_scan": phi_scan,
        "tail_scan": tail_scan, "hac_scan": hac_scan,
    }
    return stats, figs


# ---------- figures ----------
COL = {"iid": "#888888", "mbb": "#d62728", "cbb": "#1f77b4", "stationary": "#2ca02c"}


def make_figures(figs):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: representative path + ACF of returns / squared returns
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.2))
    ax[0].plot(figs["path"], lw=0.7, color="#374151")
    ax[0].axhline(MU, color="#d62728", lw=1, ls="--", label="true mean")
    ax[0].set_title("Representative path: AR(1) mean + GARCH(1,1) variance (750 days)")
    ax[0].set_ylabel("daily return")
    ax[0].legend(frameon=False)
    lags = figs["lags"]
    ax[1].plot(lags, figs["acf_ret"], "o-", ms=4, label="ACF of returns", color="#1f77b4")
    ax[1].plot(lags, figs["acf_sq"], "s-", ms=4, label="ACF of squared returns", color="#d62728")
    ax[1].axhline(0, color="#374151", lw=0.8)
    ax[1].axhline(1.96 / np.sqrt(750), color="#9ca3af", ls=":", lw=1, label="iid 95% band")
    ax[1].axhline(-1.96 / np.sqrt(750), color="#9ca3af", ls=":", lw=1)
    ax[1].set_title("Dependence lives in both the mean and the variance")
    ax[1].set_xlabel("lag (days)")
    ax[1].set_ylabel("autocorrelation")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_dependence_path.png")
    plt.close(fig)

    main = figs["main"]
    Ls = [1, 3, 5, 10, 20, 40]
    # Fig 2: coverage and width vs block length, Sharpe statistic
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in ["mbb", "cbb", "stationary"]:
        ax[0].plot(Ls, [main[f"{m}_{L}"]["coverage_sr"] for L in Ls], "o-", ms=5,
                   color=COL[m], label=m.upper())
        ax[1].plot(Ls, [main[f"{m}_{L}"]["width_sr"] for L in Ls], "o-", ms=5,
                   color=COL[m], label=m.upper())
    ax[0].axhline(main["iid_1"]["coverage_sr"], color=COL["iid"], ls="-.", lw=1.4,
                  label="iid bootstrap (no blocks)")
    ax[1].axhline(main["iid_1"]["width_sr"], color=COL["iid"], ls="-.", lw=1.4)
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2, label="nominal 0.95")
    ax[0].axhline(main["hac_5"]["coverage_sr"], color="#8b5cf6", ls=":", lw=1.4,
                  label="HAC NW(5)")
    ax[0].axhline(main["naive_0"]["coverage_sr"], color="#f59e0b", ls=":", lw=1.4,
                  label="naive iid SE")
    ax[1].axhline(main["hac_5"]["width_sr"], color="#8b5cf6", ls=":", lw=1.4)
    ax[0].set_xscale("log", base=2)
    ax[1].set_xscale("log", base=2)
    ax[0].set_xticks(Ls)
    ax[0].set_xticklabels([str(L) for L in Ls])
    ax[1].set_xticks(Ls)
    ax[1].set_xticklabels([str(L) for L in Ls])
    ax[0].set_xlabel("block length L (days)")
    ax[1].set_xlabel("block length L (days)")
    ax[0].set_ylabel("empirical coverage of 95% CI")
    ax[1].set_ylabel("mean CI width (annualised Sharpe)")
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_title("Sharpe CI: coverage vs block length")
    ax[1].set_title("Sharpe CI: the price is interval width")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_coverage_vs_block_length.png")
    plt.close(fig)

    # Fig 3: two bandwidth knobs, neither free; and mean vs Sharpe agreement
    hac = figs["hac_scan"]
    hlags = [1, 3, 5, 10, 20, 40]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(Ls, [main[f"mbb_{L}"]["coverage_sr"] for L in Ls], "o-", ms=5,
               color=COL["mbb"], label="MBB: Sharpe CI")
    ax[0].plot(Ls, [main[f"mbb_{L}"]["coverage_mean"] for L in Ls], "s--", ms=5,
               color=COL["cbb"], label="MBB: annual mean CI")
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2, label="nominal 0.95")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(Ls)
    ax[0].set_xticklabels([str(L) for L in Ls])
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_xlabel("block length L (days)")
    ax[0].set_ylabel("empirical coverage of 95% CI")
    ax[0].set_title("Sharpe and mean CIs fail together")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].plot(hlags, [hac[str(L)] for L in hlags], "o-", ms=5, color="#8b5cf6",
               label="HAC Newey-West")
    ax[1].axhline(0.95, color="#111827", ls="--", lw=1.2, label="nominal 0.95")
    ax[1].axhline(hac["naive"], color="#f59e0b", ls=":", lw=1.4, label="naive iid SE")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(hlags)
    ax[1].set_xticklabels([str(L) for L in hlags])
    ax[1].set_ylim(0.55, 1.0)
    ax[1].set_xlabel("Newey-West lag truncation L")
    ax[1].set_ylabel("coverage of 95% Sharpe CI")
    ax[1].set_title("The analytic fix has a bandwidth too")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_bandwidth_sensitivity.png")
    plt.close(fig)

    # Fig 4: sample size does not fully cure it
    ns = [125, 250, 500, 1000]
    nsc = figs["n_scan"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, c, lab in [("iid_1", "#888888", "iid bootstrap"), ("mbb_10", "#d62728", "MBB L=10"),
                      ("cbb_10", "#1f77b4", "CBB L=10"), ("hac_5", "#8b5cf6", "HAC NW(5)"),
                      ("naive_0", "#f59e0b", "naive iid SE")]:
        ax[0].plot(ns, [nsc[str(n)][k] for n in ns], "o-", ms=5, color=c, label=lab)
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(ns)
    ax[0].set_xticklabels([str(n) for n in ns])
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_xlabel("sample size n (days)")
    ax[0].set_ylabel("coverage of 95% Sharpe CI")
    ax[0].set_title("Longer samples do not rescue a wrong standard error")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].plot(ns, [nsc[str(n)]["width_mbb_10"] for n in ns], "s-", color="#d62728", ms=5, label="MBB L=10 width")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(ns)
    ax[1].set_xticklabels([str(n) for n in ns])
    ax[1].set_xlabel("sample size n (days)")
    ax[1].set_ylabel("mean CI width")
    ax[1].set_title("CI width shrinks, coverage does not converge")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_sample_size_scan.png")
    plt.close(fig)

    # Fig 5: dependence strength and fat-tailed innovations
    phis = [0.0, 0.2, 0.4, 0.6]
    ps = figs["phi_scan"]
    ts = figs["tail_scan"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, c, lab in [("iid_1", "#888888", "iid bootstrap"), ("mbb_10", "#d62728", "MBB L=10"),
                      ("cbb_10", "#1f77b4", "CBB L=10"), ("stationary_10", "#2ca02c", "stationary L=10"),
                      ("hac_5", "#8b5cf6", "HAC NW(5)")]:
        ax[0].plot(phis, [ps[str(p)][k] for p in phis], "o-", ms=5, color=c, label=lab)
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2)
    ax[0].set_xlabel("AR(1) coefficient phi")
    ax[0].set_ylabel("coverage of 95% Sharpe CI")
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_title("Stronger serial dependence, worse coverage")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    labels = ["iid", "MBB L=10", "HAC NW(5)"]
    keys = ["iid_1", "mbb_10", "hac_5"]
    xpos = np.arange(len(labels))
    for off, innov, c, hatch in [(-0.2, "normal", "#94a3b8", ""), (0.2, "t6", "#d62728", "//")]:
        ax[1].bar(xpos + off, [ts[innov][k] for k in keys], width=0.38, color=c,
                  hatch=hatch, edgecolor="white", label=f"innovations: {innov}")
    for xpos_i, k in zip(xpos, keys):
        ax[1].text(xpos_i - 0.2, ts["normal"][k] + 0.008, f"{ts['normal'][k]:.3f}", ha="center", fontsize=8)
        ax[1].text(xpos_i + 0.2, ts["t6"][k] + 0.008, f"{ts['t6'][k]:.3f}", ha="center", fontsize=8)
    ax[1].axhline(0.95, color="#111827", ls="--", lw=1.2)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylim(0.55, 1.0)
    ax[1].set_ylabel("coverage of 95% Sharpe CI")
    ax[1].set_title("t(6) innovations on top of the same GARCH")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_dependence_and_tail_scan.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nTRUE annualised Sharpe =", round(float(TRUE_SR), 6))
    print("Main results (coverage of nominal 95% CI for Sharpe):")
    for k in ["iid_1", "mbb_3", "mbb_5", "mbb_10", "mbb_20", "mbb_40",
              "cbb_10", "stationary_10", "naive_0", "hac_5"]:
        print(f"  {k:16s} coverage={stats['main'][k]['coverage_sr']:.4f}  width={stats['main'][k]['width_sr']:.4f}")
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
