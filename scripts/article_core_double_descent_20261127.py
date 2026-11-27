# -*- coding: utf-8 -*-
"""Double descent around the interpolation threshold, with five real figures.
Self-contained: all synthetic data, min-norm / ridge fits, figures and results.json.
Within each experiment every parameter point shares the SAME underlying datasets
(only the parameter under study changes), so curves are directly comparable.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "double-descent-interpolation-threshold")
os.makedirs(OUT, exist_ok=True)

D = 400           # total feature pool
NTE = 3000        # test size
REPS = 6          # independent datasets averaged per point
PGRID = sorted(set([5, 10, 15, 20, 30, 40, 50, 60, 70, 75, 78,
                    80, 82, 85, 90, 100, 120, 150, 200, 250, 300, 350, 400]))


def make_data(seed, n, d, sigma2, corr=0.0):
    """y = X beta + eps. beta has unit signal energy; optional equicorrelated X.
    The random streams for beta, X and the standard-normal eps_base are FIXED by
    seed; sigma2 and corr only rescale / mix them, so curves stay comparable.
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0, d) / np.sqrt(d)
    base_tr = rng.normal(0.0, 1.0, (n, d))
    base_te = rng.normal(0.0, 1.0, (NTE, d))
    eps_base_tr = rng.normal(0.0, 1.0, n)
    eps_base_te = rng.normal(0.0, 1.0, NTE)
    if corr == 0.0:
        Xtr, Xte = base_tr, base_te
    else:
        ftr = np.sqrt(corr) * np.ones((n, 1))
        fte = np.sqrt(corr) * np.ones((NTE, 1))
        Xtr = np.sqrt(corr) * base_tr.mean(axis=1, keepdims=True) + np.sqrt(1.0 - corr) * base_tr
        Xte = np.sqrt(corr) * base_te.mean(axis=1, keepdims=True) + np.sqrt(1.0 - corr) * base_te
    ytr = Xtr @ beta + np.sqrt(sigma2) * eps_base_tr
    yte = Xte @ beta + np.sqrt(sigma2) * eps_base_te
    return Xtr, ytr, Xte, yte


def fit_pred(Xtr, ytr, Xte, p, lam=0.0):
    """Min-norm OLS (lam=0) or ridge on the first p features."""
    A = Xtr[:, :p]
    if lam == 0.0:
        w = np.linalg.pinv(A, rcond=1e-10) @ ytr
    else:
        G = A.T @ A + lam * np.eye(p)
        w = np.linalg.solve(G, A.T @ ytr)
    return Xte[:, :p] @ w


def mse_curve(n, sigma2=0.25, corr=0.0, lam=0.0, ps=None, reps=REPS, base_seed=1100):
    ps = PGRID if ps is None else ps
    tr = np.zeros(len(ps)); te = np.zeros(len(ps))
    for r in range(reps):
        Xtr, ytr, Xte, yte = make_data(base_seed + r, n, D, sigma2, corr)
        for i, p in enumerate(ps):
            tr[i] += np.mean((fit_pred(Xtr, ytr, Xtr, p, lam) - ytr) ** 2)
            te[i] += np.mean((fit_pred(Xtr, ytr, Xte, p, lam) - yte) ** 2)
    return [float(v / reps) for v in tr], [float(v / reps) for v in te]


def main():
    res = {}

    # exp1: the canonical curve, n=80, sigma^2=0.25
    tr80, te80 = mse_curve(80)
    res["exp1_ps"] = PGRID
    res["exp1_train"] = tr80
    res["exp1_test"] = te80
    peak_i = int(np.argmax(te80))
    res["exp1_peak_p"] = PGRID[peak_i]
    res["exp1_peak_test"] = te80[peak_i]
    res["exp1_test_p10"] = te80[0]
    res["exp1_test_p400"] = te80[-1]
    res["exp1_train_p400"] = tr80[-1]
    res["exp1_noise_floor"] = 0.25
    classic = [(p, v) for p, v in zip(PGRID, te80) if p <= 50]
    over = [(p, v) for p, v in zip(PGRID, te80) if p >= 150]
    res["exp1_classic_best"] = min(classic, key=lambda t: t[1])[1]
    res["exp1_over_best"] = min(over, key=lambda t: t[1])[1]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(PGRID, tr80, "s-", color="#2c7fb8", label="train MSE (min-norm)")
    ax.plot(PGRID, te80, "o-", color="#c0392b", label="test MSE (min-norm)")
    ax.axvline(80, color="#555", ls="--", lw=1.0, label="p = n (interpolation)")
    ax.axhline(0.25, color="#27ae60", ls=":", label="irreducible noise 0.25")
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("MSE")
    ax.set_title("Test error peaks at p=n then falls again")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_double_descent.png"), dpi=130); plt.close()

    # exp2: peak location tracks n (each n has its own shared datasets)
    ns = [40, 80, 160]
    peak_by_n = {}
    curves_n = {}
    for k, n in enumerate(ns):
        ps = sorted(set(int(p) for p in
                        [max(2, n // 6), n // 3, n // 2, int(n * 0.75),
                         n - 5, n, n + 5, int(n * 1.25), int(n * 1.5),
                         int(n * 2), int(n * 3), min(D, int(n * 5))] if 2 <= p <= D))
        _, te = mse_curve(n, ps=ps, base_seed=2000 + 50 * k)
        curves_n[str(n)] = {"ps": [int(p) for p in ps], "test": te}
        pi = int(np.argmax(te))
        peak_by_n[str(n)] = int(ps[pi])
    res["exp2_ns"] = ns
    res["exp2_curves"] = curves_n
    res["exp2_peak_p"] = peak_by_n

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for n, col in zip(ns, ["#8e44ad", "#c0392b", "#e67e22"]):
        d = curves_n[str(n)]
        ax.plot(d["ps"], d["test"], "o-", label=f"n={n}, peak p={peak_by_n[str(n)]}", color=col)
        ax.axvline(n, color=col, ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("The spike sits at p ≈ n, whatever n is")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_peak_location.png"), dpi=130); plt.close()

    # exp3: noise level changes spike height and the over-parameterized floor
    # Same datasets (same base_seed) across sigma2: only the noise scale changes.
    sigmas = [0.04, 0.25, 1.0]
    curves_s = {}
    for s2 in sigmas:
        _, te = mse_curve(80, sigma2=s2, base_seed=3100)
        curves_s[str(s2)] = te
        pi = int(np.argmax(te))
        curves_s["peak_" + str(s2)] = float(te[pi])
        curves_s["floor_" + str(s2)] = float(te[-1])
    res["exp3_sigmas"] = sigmas
    res["exp3_curves"] = {str(s2): curves_s[str(s2)] for s2 in sigmas}
    res["exp3_peaks"] = {str(s2): curves_s["peak_" + str(s2)] for s2 in sigmas}
    res["exp3_floors"] = {str(s2): curves_s["floor_" + str(s2)] for s2 in sigmas}

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for s2, col in zip(sigmas, ["#27ae60", "#c0392b", "#8e44ad"]):
        ax.plot(PGRID, curves_s[str(s2)], "o-", label=f"noise={s2}", color=col)
        ax.axhline(s2, color=col, ls=":", lw=0.8, alpha=0.7)
    ax.axvline(80, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("Higher noise lifts both the spike and the floor")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_noise.png"), dpi=130); plt.close()

    # exp4: ridge regularization removes the spike (same datasets across lambda)
    lams = [0.0, 0.1, 1.0, 10.0]
    curves_l = {}
    for lam in lams:
        _, te = mse_curve(80, lam=lam, base_seed=4100)
        curves_l[str(lam)] = te
        curves_l["peak_" + str(lam)] = float(max(te))
    res["exp4_lams"] = lams
    res["exp4_curves"] = {str(lam): curves_l[str(lam)] for lam in lams}
    res["exp4_peaks"] = {str(lam): curves_l["peak_" + str(lam)] for lam in lams}
    res["exp4_peak_lam0"] = curves_l["peak_0.0"]
    res["exp4_peak_lam10"] = curves_l["peak_10.0"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for lam, col in zip(lams, ["#c0392b", "#e67e22", "#2c7fb8", "#27ae60"]):
        ax.plot(PGRID, curves_l[str(lam)], "o-", label=f"ridge lambda={lam}", color=col)
    ax.axvline(80, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("Ridge penalty flattens the interpolation spike")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_ridge.png"), dpi=130); plt.close()

    # exp5: correlated features (common factor) reshape the spike
    corrs = [0.0, 0.3, 0.7]
    curves_c = {}
    for c in corrs:
        _, te = mse_curve(80, corr=c, base_seed=5100)
        curves_c[str(c)] = te
        curves_c["peak_" + str(c)] = float(max(te))
    res["exp5_corrs"] = corrs
    res["exp5_curves"] = {str(c): curves_c[str(c)] for c in corrs}
    res["exp5_peaks"] = {str(c): curves_c["peak_" + str(c)] for c in corrs}

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for c, col in zip(corrs, ["#27ae60", "#e67e22", "#c0392b"]):
        ax.plot(PGRID, curves_c[str(c)], "o-", label=f"feature corr={c}", color=col)
    ax.axvline(80, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("A common factor reshapes (but does not erase) the spike")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_correlation.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
