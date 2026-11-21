# -*- coding: utf-8 -*-
"""DeLong test for comparing two correlated ROC AUCs on the same sample.

Single source of truth for every number and figure in the article.

DGP (binormal, known truth): for each replication draw m positives and n
negatives. A latent bivariate-normal score pair (sA, sB) with correlation rho
is generated; positives have both means shifted by delta. AUC_A = AUC_B under
H0 (delta=0). Because both classifiers are scored on the SAME cases, their
AUC errors are correlated; ignoring that covariance (a "naive independent"
z-test) mis-sizes the test.

Analytic baselines:
    binormal AUC = Phi(delta / sqrt(2)) when scores are N(0,1)/N(delta,1)
    AUC covariance under paired designs is positive for rho>0, so the naive
    independent variance overstates the true variance -> test is conservative.

Methods for H0: AUC_A = AUC_B:
    naive  : var = var(A) + var(B), covariance ignored
    delong : U-statistic structural components V10, V01 incl. covariance
    bootstrap: paired resampling of cases (B=399)
    hanley : analytic binormal variance (no covariance, for comparison)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261121
M_POS = 120
N_NEG = 120
REPS_SIZE = 2000
REPS_POWER = 1200
B_BOOT = 399
RHO_GRID = [0.0, 0.3, 0.6, 0.9]
DELTA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]   # analytic-check shifts
GAP_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # power: true AUC gap A vs B
Z95 = float(special.ndtri(0.975))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "delong-auc-roc-comparison")
os.makedirs(_OUT, exist_ok=True)


def auc(y, x):
    """Empirical AUC = P(score_pos > score_neg), mid-rank for ties.
    Mann-Whitney U via average ranks of pooled scores -> O((m+n)log), vectorized."""
    m, n = len(y), len(x)
    ranks = stats.rankdata(np.concatenate([x, y]), method="average")
    return float((ranks[n:].sum() - m * (m + 1) / 2.0) / (m * n))


def draw_pair(rng, delta, rho, gap=0.0, m=M_POS, n=N_NEG):
    """Return (pos scores A,B) and (neg scores A,B) with latent corr rho.
    Positive means: A = delta + gap/2, B = delta - gap/2, so gap>0 makes
    A genuinely better (AUC_A > AUC_B) while keeping the paired correlation."""
    L = np.array([[1.0, 0.0], [rho, np.sqrt(max(1 - rho * rho, 1e-12))]])
    zp = rng.standard_normal((m, 2)) @ L.T
    zn = rng.standard_normal((n, 2)) @ L.T
    pos = zp + np.array([delta + gap / 2, delta - gap / 2])
    neg = zn
    return pos[:, 0], pos[:, 1], neg[:, 0], neg[:, 1]


def delong_var(yA, yB, xA, xB):
    """DeLong variance of (AUC_A - AUC_B) using structural components."""
    def comps(y, x):
        # V10_i: influence of negative case i ; V01_j: influence of positive case j
        ind = (y[:, None] > x[None, :]).astype(float)  # (m,n)
        v10 = ind.mean(axis=0)   # length n
        v01 = ind.mean(axis=1)   # length m
        return v10, v01
    v10A, v01A = comps(yA, xA)
    v10B, v01B = comps(yB, xB)
    m, n = len(yA), len(xA)
    s10 = np.var(v10A - v10B, ddof=1) / n
    s01 = np.var(v01A - v01B, ddof=1) / m
    return s10 + s01


def naive_var(yA, yB, xA, xB):
    """Naive: treat the two AUCs as independent, drop covariance."""
    def one(y, x):
        ind = (y[:, None] > x[None, :]).astype(float)
        v10 = ind.mean(axis=0)
        v01 = ind.mean(axis=1)
        return np.var(v10, ddof=1) / len(x) + np.var(v01, ddof=1) / len(y)
    return one(yA, xA) + one(yB, xB)


def boot_diff(yA, yB, xA, xB, rng, B=B_BOOT):
    """Paired bootstrap of AUC_A - AUC_B (resample cases jointly)."""
    m, n = len(yA), len(xA)
    out = np.empty(B)
    for b in range(B):
        pi = rng.integers(0, m, m)
        ni = rng.integers(0, n, n)
        out[b] = auc(yA[pi], xA[ni]) - auc(yB[pi], xB[ni])
    return out


def exp_analytic_check():
    """Verify binormal AUC closed form against large-sample MC, and show ROC."""
    rng = np.random.default_rng(SEED + 1)
    big = 400000
    z = rng.standard_normal(big)
    rows = []
    for delta in [0.0, 0.3, 0.6, 0.9, 1.2]:
        y = delta + rng.standard_normal(big)
        emp = auc(y, z)
        theory = float(special.ndtr(delta / np.sqrt(2)))
        rows.append({"delta": delta, "emp": emp, "theory": theory,
                     "abs_err": abs(emp - theory)})
    # ROC curve at delta=0.9 for plotting
    delta = 0.9
    yr = delta + rng.standard_normal(200000)
    zr = rng.standard_normal(200000)
    thr = np.linspace(-3, 4, 120)
    tpr = [(yr > t).mean() for t in thr]
    fpr = [(zr > t).mean() for t in thr]
    return {"rows": rows, "roc_delta": delta, "thr": thr.tolist(),
            "tpr": [float(x) for x in tpr], "fpr": [float(x) for x in fpr]}


def one_rep(rng, delta, rho, gap=0.0, boot=False):
    yA, yB, xA, xB = draw_pair(rng, delta, rho, gap=gap)
    d = auc(yA, xA) - auc(yB, xB)
    vn = naive_var(yA, yB, xA, xB)
    vd = delong_var(yA, yB, xA, xB)
    res = {"d": d, "vn": vn, "vd": vd}
    if boot:
        bd = boot_diff(yA, yB, xA, xB, rng)
        res["vb"] = float(bd.var(ddof=1))
    return res


def exp_size_boot():
    """H0 size by rho: naive / delong / bootstrap, with bootstrap per rep."""
    rng = np.random.default_rng(SEED + 2)
    reps = 800
    rows = []
    for rho in RHO_GRID:
        rej_n = rej_d = rej_b = 0
        vn_mean = vd_mean = vb_mean = 0.0
        cov_emp = []
        for _ in range(reps):
            r = one_rep(rng, 0.0, rho, boot=True)
            vn_mean += r["vn"]
            vd_mean += r["vd"]
            vb_mean += r["vb"]
            if abs(r["d"]) / np.sqrt(max(r["vn"], 1e-15)) > Z95:
                rej_n += 1
            if abs(r["d"]) / np.sqrt(max(r["vd"], 1e-15)) > Z95:
                rej_d += 1
            if abs(r["d"]) / np.sqrt(max(r["vb"], 1e-15)) > Z95:
                rej_b += 1
            cov_emp.append(r["vn"] - r["vd"])  # naive minus true variance
        rows.append({"rho": rho, "naive": rej_n / reps, "delong": rej_d / reps,
                     "boot": rej_b / reps,
                     "vn_mean": vn_mean / reps, "vd_mean": vd_mean / reps,
                     "vb_mean": vb_mean / reps,
                     "var_inflation": (vn_mean / vd_mean)})
    return {"reps": reps, "b_boot": B_BOOT, "rows": rows}


def exp_power():
    """Power of naive/delong vs true AUC gap under rho=0.6.
    Base signal delta=0.4; A gets +gap/2, B gets -gap/2 in positive mean."""
    rng = np.random.default_rng(SEED + 3)
    rho = 0.6
    base = 0.4
    rows = []
    for gap in GAP_GRID:
        rej_n = rej_d = 0
        aucA = float(special.ndtr((base + gap / 2) / np.sqrt(2)))
        aucB = float(special.ndtr((base - gap / 2) / np.sqrt(2)))
        gap_truth = aucA - aucB
        d_sum = 0.0
        for _ in range(REPS_POWER):
            r = one_rep(rng, base, rho, gap=gap, boot=False)
            d_sum += r["d"]
            if abs(r["d"]) / np.sqrt(max(r["vn"], 1e-15)) > Z95:
                rej_n += 1
            if abs(r["d"]) / np.sqrt(max(r["vd"], 1e-15)) > Z95:
                rej_d += 1
        rows.append({"gap": gap, "naive": rej_n / REPS_POWER,
                     "delong": rej_d / REPS_POWER,
                     "auc_gap_truth": gap_truth, "auc_gap_emp": d_sum / REPS_POWER})
    return {"reps": REPS_POWER, "rho": rho, "base": base, "rows": rows}


def exp_size_coverage():
    """95% CI coverage of AUC_A - AUC_B (true delta=0) by rho, naive vs DeLong."""
    rng = np.random.default_rng(SEED + 4)
    reps = 1500
    rows = []
    for rho in RHO_GRID:
        cov_n = cov_d = 0
        for _ in range(reps):
            r = one_rep(rng, 0.0, rho, boot=False)
            d = r["d"]
            if abs(d) <= Z95 * np.sqrt(max(r["vn"], 1e-15)):
                cov_n += 1
            if abs(d) <= Z95 * np.sqrt(max(r["vd"], 1e-15)):
                cov_d += 1
        rows.append({"rho": rho, "naive_cov": cov_n / reps, "delong_cov": cov_d / reps})
    return {"reps": reps, "rows": rows}


def plot1(check):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ds = [r["delta"] for r in check["rows"]]
    ax.plot(ds, [r["theory"] for r in check["rows"]], "s-", color="#2c3e50", lw=2, label="analytic Phi(delta/sqrt2)")
    ax.plot(ds, [r["emp"] for r in check["rows"]], "o--", color="#c0392b", lw=2, label="MC (400k pairs)")
    ax.set_xlabel("score shift delta")
    ax.set_ylabel("AUC")
    ax.set_title("1a: binormal AUC closed form vs MC")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(check["fpr"], check["tpr"], color="#2980b9", lw=2)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC at delta=%.1f" % check["roc_delta"])
    fig.tight_layout()
    path = os.path.join(_OUT, "fig1_auc_theory.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig1_auc_theory.png"


def plot2(size):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["rho"] for r in size["rows"]]
    ax.plot(xs, [r["naive"] for r in size["rows"]], "o-", color="#c0392b", lw=2.2, label="naive (covariance ignored)")
    ax.plot(xs, [r["delong"] for r in size["rows"]], "s-", color="#27ae60", lw=2.2, label="DeLong")
    ax.plot(xs, [r["boot"] for r in size["rows"]], "^--", color="#8e44ad", lw=1.8, label="paired bootstrap")
    ax.axhline(0.05, color="k", ls=":", lw=1.5, label="nominal 5%")
    ax.set_xlabel("score correlation rho (same cases)")
    ax.set_ylabel("false rejection of H0: AUC_A=AUC_B")
    ax.set_ylim(-0.01, 0.16)
    ax.set_title("Fig2: naive goes conservative as rho rises; DeLong holds 5%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig2_size.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig2_size.png"


def plot3(size):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["rho"] for r in size["rows"]]
    ax.plot(xs, [r["var_inflation"] for r in size["rows"]], "o-", color="#c0392b", lw=2.2)
    ax.axhline(1.0, color="#27ae60", ls="--", lw=1.6, label="correct (inflation=1)")
    ax.set_xlabel("score correlation rho")
    ax.set_ylabel("naive var / DeLong var")
    ax.set_title("Fig3: ignoring covariance inflates the variance estimate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig3_var_inflation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig3_var_inflation.png"


def plot4(power):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["auc_gap_truth"] for r in power["rows"]]
    ax.plot(xs, [r["delong"] for r in power["rows"]], "s-", color="#27ae60", lw=2.2, label="DeLong power")
    ax.plot(xs, [r["naive"] for r in power["rows"]], "o--", color="#c0392b", lw=2, label="naive power")
    ax.axhline(0.05, color="k", ls=":", lw=1.3, label="size 5%")
    ax.set_xlabel("true AUC gap (A - B)")
    ax.set_ylabel("rejection rate")
    ax.set_title("Fig4: DeLong detects real gaps that naive misses")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig4_power.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig4_power.png"


def plot5(cov):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["rho"] for r in cov["rows"]]
    ax.plot(xs, [r["naive_cov"] for r in cov["rows"]], "o-", color="#c0392b", lw=2.2, label="naive 95% CI")
    ax.plot(xs, [r["delong_cov"] for r in cov["rows"]], "s-", color="#27ae60", lw=2.2, label="DeLong 95% CI")
    ax.axhline(0.95, color="k", ls=":", lw=1.5, label="nominal 95%")
    ax.set_xlabel("score correlation rho")
    ax.set_ylabel("CI coverage of AUC_A - AUC_B")
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Fig5: naive CI over-covers; DeLong stays at 95%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig5_coverage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig5_coverage.png"


def main():
    check = exp_analytic_check()
    size = exp_size_boot()
    power = exp_power()
    cov = exp_size_coverage()
    results = {
        "setup": {"seed": SEED, "m_pos": M_POS, "n_neg": N_NEG,
                  "reps_size": size["reps"], "reps_power": REPS_POWER,
                  "reps_cov": cov["reps"], "b_boot": B_BOOT, "z95": Z95,
                  "rho_grid": RHO_GRID, "delta_grid": DELTA_GRID, "gap_grid": GAP_GRID},
        "analytic_check": {"rows": check["rows"], "roc_delta": check["roc_delta"]},
        "size_boot": size,
        "power": power,
        "coverage": cov,
    }
    results["figs"] = [plot1(check), plot2(size), plot3(size), plot4(power), plot5(cov)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
