# -*- coding: utf-8 -*-
"""Distance concentration and kNN failure in high dimensions, with 5 real figures.
Self-contained: all data, experiments, figures and results.json from one script.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261126)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "knn-distance-concentration-high-dimension")
os.makedirs(OUT, exist_ok=True)

PMAX = 1024
PS = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])


def sqdist(A, B):
    return (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
            - 2.0 * A @ B.T)


def knn_pred(Xtr, ytr, Xte, k):
    d = sqdist(Xte, Xtr)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    return ytr[idx].mean(axis=1)


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def main():
    res = {}
    rng = RNG
    n = 1100
    Z = rng.normal(0, 1, size=(n, PMAX))

    # exp1: concentration ratio of per-coordinate RMS distance from origin
    ratios, theories = [], []
    for p in PS:
        d = np.sqrt(np.mean(Z[:, :p] ** 2, axis=1))
        ratios.append(float((d.max() - d.min()) / d.mean()))
        theories.append(float(np.sqrt(2.0 * np.log(n) / p)))
    res["exp1_ps"] = [int(p) for p in PS]
    res["exp1_ratio"] = ratios
    res["exp1_theory"] = theories
    res["exp1_ratio_p2"] = ratios[0]
    res["exp1_ratio_p1000"] = ratios[-1]
    res["exp1_theory_p1000"] = theories[-1]
    res["exp1_first_below_010"] = int(PS[int(np.argmax(np.array(ratios) < 0.1))]) if any(np.array(ratios) < 0.1) else -1

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    a1.plot(PS, ratios, "o-", color="#c0392b", label="measured (max-min)/mean")
    a1.plot(PS, theories, "s--", color="#2c7fb8", label="sqrt(2 log n / p)")
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("relative spread")
    a1.set_title("Distance spread collapses as p grows")
    a1.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig01_concentration.png"), dpi=130); plt.close()

    # exp2: is the nearest neighbour actually informative about y?
    ntr, nte = 700, 300
    y_all = Z[:, 0] + rng.normal(0, 0.6, n)
    Xtr, Xte = Z[:ntr], Z[ntr:ntr + nte]
    ytr, yte = y_all[:ntr], y_all[ntr:ntr + nte]
    cors, nn_mse, gaps = [], [], []
    for p in PS:
        d = sqdist(Xte[:, :p], Xtr[:, :p])
        nn = np.argmin(d, axis=1)
        part = np.partition(d, 1, axis=1)[:, :2]
        gaps.append(float(np.mean((part[:, 1] - part[:, 0]) / np.maximum(part[:, 1], 1e-12))))
        pred = ytr[nn]
        cors.append(float(np.corrcoef(pred, yte)[0, 1]))
        nn_mse.append(mse(pred, yte))
    baseline = mse(np.full_like(yte, ytr.mean()), yte)
    res["exp2_ps"] = [int(p) for p in PS]
    res["exp2_corr"] = cors
    res["exp2_nn_mse"] = nn_mse
    res["exp2_dn_gap"] = gaps
    res["exp2_baseline_mse"] = baseline
    res["exp2_corr_p2"] = cors[0]
    res["exp2_corr_p1000"] = cors[-1]
    res["exp2_gap_p2"] = gaps[0]
    res["exp2_gap_p1000"] = gaps[-1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(PS, cors, "o-", color="#c0392b")
    a1.axhline(0, color="#555", lw=0.8)
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("corr(y_test, y_NN)")
    a1.set_title("Nearest-neighbour label correlation dies")
    a2.plot(PS, nn_mse, "o-", color="#c0392b", label="1-NN")
    a2.axhline(baseline, color="#2c7fb8", ls="--", label="predict training mean")
    a2.set_xscale("log"); a2.set_xlabel("dimension p"); a2.set_ylabel("test MSE")
    a2.set_title("1-NN degenerates to the mean predictor")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_nn_signal.png"), dpi=130); plt.close()

    # exp3: kNN regression error vs p, compared to known-signal methods
    y2 = 0.8 * Z[:, 0] + 0.5 * Z[:, 1] ** 2 - 0.3 * Z[:, 2] + rng.normal(0, 0.5, n)
    a, b = y2[:ntr], y2[ntr:ntr + nte]
    knn1, knn10, knn_oracle = [], [], []
    for p in PS:
        knn1.append(mse(knn_pred(Xtr[:, :p], a, Xte[:, :p], 1), b))
        knn10.append(mse(knn_pred(Xtr[:, :p], a, Xte[:, :p], 10), b))
        knn_oracle.append(mse(knn_pred(Xtr[:, :3], a, Xte[:, :3], 10), b))
    D = np.column_stack([np.ones(ntr), Xtr[:, :3]])
    coef = np.linalg.lstsq(D, a, rcond=None)[0]
    ols = mse(np.column_stack([np.ones(nte), Xte[:, :3]]) @ coef, b)
    res["exp3_ps"] = [int(p) for p in PS]
    res["exp3_knn1"] = knn1
    res["exp3_knn10"] = knn10
    res["exp3_knn3d"] = knn_oracle
    res["exp3_ols3d"] = ols
    res["exp3_knn1_p2"] = knn1[0]
    res["exp3_knn10_p1000"] = knn10[-1]
    res["exp3_knn3d_mse"] = float(knn_oracle[0])

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    a1.plot(PS, knn1, "o-", color="#c0392b", label="1-NN, all p features")
    a1.plot(PS, knn10, "s-", color="#e67e22", label="10-NN, all p features")
    a1.plot(PS, knn_oracle, "^--", color="#2c7fb8", label="10-NN, 3 signal features")
    a1.axhline(ols, color="#27ae60", ls=":", label="OLS on 3 signal features")
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("test MSE")
    a1.set_title("Raw-space kNN loses to feature selection")
    a1.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig03_knn_mse.png"), dpi=130); plt.close()

    # exp4: shrinking noise variance per dimension keeps total noise energy fixed
    ps4 = np.array([3, 10, 30, 100, 300, 1000])
    curves = {}
    noise = rng.normal(0, 1, size=(n, int(ps4.max())))
    signal3 = Z[:, :3]
    y4 = 0.8 * signal3[:, 0] + 0.5 * signal3[:, 1] ** 2 - 0.3 * signal3[:, 2]
    y4 += rng.normal(0, 0.3, n)
    a4, b4 = y4[:ntr], y4[ntr:ntr + nte]
    for tag, c in [("fixed_sigma", 1.0), ("energy_1", 1.0), ("energy_4", 4.0), ("energy_16", 16.0)]:
        vals = []
        for p in ps4:
            scale = c / p if tag != "fixed_sigma" else 1.0
            full = np.column_stack([signal3, noise[:, :max(0, int(p) - 3)] * np.sqrt(scale)])[:, :int(p)]
            vals.append(mse(knn_pred(full[:ntr], a4, full[ntr:ntr + nte], 5), b4))
        curves[tag] = vals
    res["exp4_ps"] = [int(p) for p in ps4]
    res["exp4_curves"] = {k: [float(v) for v in arr] for k, arr in curves.items()}
    res["exp4_fixed_p1000"] = curves["fixed_sigma"][-1]
    res["exp4_energy1_p1000"] = curves["energy_1"][-1]
    res["exp4_energy16_p1000"] = curves["energy_16"][-1]

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    styles = [("fixed_sigma", "#c0392b", "noise var 1 per dim"),
              ("energy_1", "#2c7fb8", "total noise energy 1"),
              ("energy_4", "#e67e22", "total noise energy 4"),
              ("energy_16", "#8e44ad", "total noise energy 16")]
    for k, col, lab in styles:
        a1.plot(ps4, curves[k], "o-", color=col, label=lab)
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("5-NN test MSE")
    a1.set_title("Fixing total noise energy rescues kNN")
    a1.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig04_noise_scaling.png"), dpi=130); plt.close()

    # exp5: correlated noise directions, raw vs covariance-whitened kNN
    kfac = 5
    F = rng.normal(0, 1, size=(n, kfac))
    L = rng.normal(0, 1, size=(60, kfac))
    corr_noise = F @ L.T + rng.normal(0, 0.35, size=(n, 60))
    sig = Z[:, :3]
    X5raw = np.column_stack([sig, corr_noise])
    y5 = 0.9 * sig[:, 0] + 0.6 * sig[:, 1] - 0.4 * sig[:, 2] + rng.normal(0, 0.4, n)
    a5, b5 = y5[:ntr], y5[ntr:ntr + nte]
    mu = X5raw[:ntr].mean(axis=0)
    C = np.cov((X5raw[:ntr] - mu).T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-6)
    W = (evecs / np.sqrt(evals)) @ evecs.T
    X5w = (X5raw - mu) @ W.T
    raw_mse = mse(knn_pred(X5raw[:ntr], a5, X5raw[ntr:ntr + nte], 10), b5)
    wht_mse = mse(knn_pred(X5w[:ntr], a5, X5w[ntr:ntr + nte], 10), b5)
    ora_mse = mse(knn_pred(sig[:ntr], a5, sig[ntr:ntr + nte], 10), b5)
    D5 = np.column_stack([np.ones(ntr), sig[:ntr]])
    c5 = np.linalg.lstsq(D5, a5, rcond=None)[0]
    ols5 = mse(np.column_stack([np.ones(nte), sig[ntr:ntr + nte]]) @ c5, b5)
    res["exp5_raw_mse"] = raw_mse
    res["exp5_whitened_mse"] = wht_mse
    res["exp5_oracle3d_mse"] = ora_mse
    res["exp5_ols3d_mse"] = ols5
    res["exp5_top_eig_share"] = float(evals.max() / evals.sum())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    labels = ["raw 63d", "whitened", "signal 3d", "OLS 3d"]
    vals = [raw_mse, wht_mse, ora_mse, ols5]
    bars = a1.bar(labels, vals, color=["#c0392b", "#e67e22", "#2c7fb8", "#27ae60"])
    for bar, v in zip(bars, vals):
        a1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a1.set_ylabel("test MSE"); a1.set_title("Whitening vs raw Euclidean kNN")
    a2.plot(np.arange(1, len(evals) + 1), evals / evals.sum(), color="#8e44ad", lw=1.8)
    a2.set_xlabel("eigenvalue rank"); a2.set_ylabel("share of total variance")
    a2.set_title(f"top eigenvalue carries {res['exp5_top_eig_share']:.1%}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_whitening.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
