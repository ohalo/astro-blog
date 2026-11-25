# -*- coding: utf-8 -*-
"""Nadaraya-Watson boundary bias vs local linear regression, with MC checks.
Self-contained: data generation + experiments + 5 figures + results.json.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261126)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "kernel-local-linear-boundary-bias")
os.makedirs(OUT, exist_ok=True)

N = 300
SIGMA = 0.15
GRID = np.linspace(0.0, 1.0, 101)


def m_true(x):
    return 0.3 + 1.2 * x + 0.35 * np.sin(4.0 * np.pi * x)


def fit_nw(xq, xt, yt, h):
    d = (xq[:, None] - xt[None, :]) / h
    w = np.exp(-0.5 * d * d)
    return (w @ yt) / np.maximum(w.sum(axis=1), 1e-12)


def fit_ll(xq, xt, yt, h):
    z = xt[None, :] - xq[:, None]
    w = np.exp(-0.5 * (z / h) ** 2)
    s0 = w.sum(axis=1)
    s1 = (w * z).sum(axis=1)
    s2 = (w * z * z).sum(axis=1)
    t0 = w @ yt
    t1 = (w * z) @ yt
    det = s0 * s2 - s1 * s1
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    return (t0 * s2 - t1 * s1) / det


def loo_nw(xt, yt, h):
    d = (xt[:, None] - xt[None, :]) / h
    w = np.exp(-0.5 * d * d)
    np.fill_diagonal(w, 0.0)
    return (w @ yt) / w.sum(axis=1), w.sum(axis=1)


def loo_ll(xt, yt, h):
    z = xt[None, :] - xt[:, None]
    w = np.exp(-0.5 * (z / h) ** 2)
    np.fill_diagonal(w, 0.0)
    s0 = w.sum(axis=1); s1 = (w * z).sum(axis=1); s2 = (w * z * z).sum(axis=1)
    t0 = w @ yt; t1 = (w * z) @ yt
    det = s0 * s2 - s1 * s1
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    return (t0 * s2 - t1 * s1) / det, det


def main():
    res = {}
    rng = RNG
    xt = (np.arange(N) + 0.5) / N
    mt = m_true(xt)
    mg = m_true(GRID)
    h0 = 0.08

    # exp1: single sample curves, boundary zoom
    y1 = mt + rng.normal(0, SIGMA, N)
    nw1 = fit_nw(GRID, xt, y1, h0)
    ll1 = fit_ll(GRID, xt, y1, h0)
    res["exp1_h"] = h0
    res["exp1_m0"] = float(m_true(np.array([0.0]))[0])
    res["exp1_m1"] = float(m_true(np.array([1.0]))[0])
    res["exp1_nw_at_0"] = float(nw1[0])
    res["exp1_ll_at_0"] = float(ll1[0])
    res["exp1_nw_at_1"] = float(nw1[-1])
    res["exp1_ll_at_1"] = float(ll1[-1])
    res["exp1_nw_bias_0"] = float(nw1[0] - mg[0])
    res["exp1_ll_bias_0"] = float(ll1[0] - mg[0])
    res["exp1_nw_bias_1"] = float(nw1[-1] - mg[-1])
    res["exp1_ll_bias_1"] = float(ll1[-1] - mg[-1])
    res["exp1_mse_nw"] = float(np.mean((nw1 - mg) ** 2))
    res["exp1_mse_ll"] = float(np.mean((ll1 - mg) ** 2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(GRID, mg, color="#222", lw=2, label="true m(x)")
    a1.plot(GRID, nw1, color="#c0392b", lw=1.8, label="Nadaraya-Watson")
    a1.plot(GRID, ll1, color="#2c7fb8", lw=1.8, label="local linear")
    a1.scatter(xt[::6], y1[::6], s=8, color="#999", alpha=0.5)
    a1.set_xlabel("x"); a1.set_ylabel("m(x)")
    a1.set_title("h=0.08: whole domain")
    a1.legend(fontsize=8)
    mask = GRID <= 0.12
    a2.plot(GRID[mask], mg[mask], color="#222", lw=2, label="true")
    a2.plot(GRID[mask], nw1[mask], color="#c0392b", lw=2, label="NW")
    a2.plot(GRID[mask], ll1[mask], color="#2c7fb8", lw=2, label="LL")
    a2.axvline(0, color="#555", ls=":")
    a2.set_xlabel("x"); a2.set_ylabel("m(x)")
    a2.set_title(f"left edge: NW bias {res['exp1_nw_bias_0']:+.3f}, LL {res['exp1_ll_bias_0']:+.3f}")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_curves.png"), dpi=130); plt.close()

    # exp2: MC pointwise bias^2 / variance at h=0.08
    reps = 240
    edge = (GRID <= 0.08) | (GRID >= 0.92)
    inner = (GRID >= 0.2) & (GRID <= 0.8)
    acc_nw = np.zeros(len(GRID)); acc_ll = np.zeros(len(GRID))
    sq_nw = np.zeros(len(GRID)); sq_ll = np.zeros(len(GRID))
    for _ in range(reps):
        yy = mt + rng.normal(0, SIGMA, N)
        a = fit_nw(GRID, xt, yy, h0)
        b = fit_ll(GRID, xt, yy, h0)
        acc_nw += a; acc_ll += b
        sq_nw += a * a; sq_ll += b * b
    acc_nw /= reps; acc_ll /= reps
    sq_nw /= reps; sq_ll /= reps
    var_nw = np.maximum(sq_nw - acc_nw ** 2, 0.0)
    var_ll = np.maximum(sq_ll - acc_ll ** 2, 0.0)
    bias_nw = acc_nw - mg; bias_ll = acc_ll - mg
    mse_nw = bias_nw ** 2 + var_nw
    mse_ll = bias_ll ** 2 + var_ll
    for nm, arr in [("nw", mse_nw), ("ll", mse_ll),
                    ("nw_edge", mse_nw[edge]), ("ll_edge", mse_ll[edge]),
                    ("nw_inner", mse_nw[inner]), ("ll_inner", mse_ll[inner])]:
        res["exp2_mse_" + nm] = float(np.mean(arr))
    res["exp2_bias2_nw_edge"] = float(np.mean(bias_nw[edge] ** 2))
    res["exp2_bias2_ll_edge"] = float(np.mean(bias_ll[edge] ** 2))
    res["exp2_var_nw_inner"] = float(np.mean(var_nw[inner]))
    res["exp2_var_ll_inner"] = float(np.mean(var_ll[inner]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(GRID, bias_nw ** 2, color="#c0392b", lw=1.8, label="NW bias^2")
    a1.plot(GRID, bias_ll ** 2, color="#2c7fb8", lw=1.8, label="LL bias^2")
    a1.axvspan(0, 0.08, color="#eee"); a1.axvspan(0.92, 1, color="#eee")
    a1.set_xlabel("x"); a1.set_ylabel("squared bias")
    a1.set_title(f"edge bias^2: NW {res['exp2_bias2_nw_edge']:.5f} vs LL {res['exp2_bias2_ll_edge']:.5f}")
    a1.legend(fontsize=8)
    a2.plot(GRID, var_nw, color="#c0392b", lw=1.8, ls="--", label="NW variance")
    a2.plot(GRID, var_ll, color="#2c7fb8", lw=1.8, ls="--", label="LL variance")
    a2.set_xlabel("x"); a2.set_ylabel("pointwise variance")
    a2.set_title(f"interior variance: NW {res['exp2_var_nw_inner']:.5f} vs LL {res['exp2_var_ll_inner']:.5f}")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_bias_variance.png"), dpi=130); plt.close()

    # exp3: bandwidth scan, edge vs interior MSE
    hs = np.array([0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32])
    scan = {k: [] for k in ["nw_edge", "ll_edge", "nw_inner", "ll_inner"]}
    for h in hs:
        se_nw = se_ll = si_nw = si_ll = 0.0
        for _ in range(70):
            yy = mt + rng.normal(0, SIGMA, N)
            a = fit_nw(GRID, xt, yy, h); b = fit_ll(GRID, xt, yy, h)
            se_nw += np.mean((a[edge] - mg[edge]) ** 2)
            se_ll += np.mean((b[edge] - mg[edge]) ** 2)
            si_nw += np.mean((a[inner] - mg[inner]) ** 2)
            si_ll += np.mean((b[inner] - mg[inner]) ** 2)
        scan["nw_edge"].append(se_nw / 70); scan["ll_edge"].append(se_ll / 70)
        scan["nw_inner"].append(si_nw / 70); scan["ll_inner"].append(si_ll / 70)
    res["exp3_hs"] = [float(x) for x in hs]
    res["exp3_scan"] = {k: [float(v) for v in arr] for k, arr in scan.items()}
    res["exp3_best_nw_edge_h"] = float(hs[int(np.argmin(scan["nw_edge"]))])
    res["exp3_best_ll_edge_h"] = float(hs[int(np.argmin(scan["ll_edge"]))])
    res["exp3_nw_edge_at_h016"] = float(scan["nw_edge"][4])
    res["exp3_ll_edge_at_h016"] = float(scan["ll_edge"][4])
    res["exp3_ll_inner_at_h002"] = float(scan["ll_inner"][0])
    res["exp3_ll_inner_best"] = float(np.min(scan["ll_inner"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(hs, scan["nw_edge"], "o-", color="#c0392b", label="NW edge")
    a1.plot(hs, scan["ll_edge"], "s-", color="#2c7fb8", label="LL edge")
    a1.set_xscale("log"); a1.set_xlabel("bandwidth h"); a1.set_ylabel("MSE at edges")
    a1.set_title("NW edge error grows with h; LL has a U shape")
    a1.legend(fontsize=8)
    a2.plot(hs, scan["nw_inner"], "o-", color="#c0392b", label="NW interior")
    a2.plot(hs, scan["ll_inner"], "s-", color="#2c7fb8", label="LL interior")
    a2.set_xscale("log"); a2.set_xlabel("bandwidth h"); a2.set_ylabel("MSE in interior")
    a2.set_title("Interior: LL pays variance at small h")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_bandwidth.png"), dpi=130); plt.close()

    # exp4: LOOCV bandwidth under iid vs AR(1) errors
    hcv = np.array([0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24])
    cvres = {}
    for tag, phi in [("iid", 0.0), ("ar1", 0.75)]:
        picks = {"nw": [], "ll": []}
        realized = {"nw": [], "ll": []}
        for _ in range(60):
            eps = rng.normal(0, SIGMA, N)
            if phi > 0:
                for j in range(1, N):
                    eps[j] = phi * eps[j - 1] + np.sqrt(1 - phi ** 2) * eps[j]
            yy = mt + eps
            scores = {"nw": np.zeros(len(hcv)), "ll": np.zeros(len(hcv))}
            for i, h in enumerate(hcv):
                pn, sn = loo_nw(xt, yy, h)
                pl, _ = loo_ll(xt, yy, h)
                scores["nw"][i] = np.mean((pn - yy) ** 2)
                scores["ll"][i] = np.mean((pl - yy) ** 2)
            for k in picks:
                hsel = float(hcv[int(np.argmin(scores[k]))])
                picks[k].append(hsel)
                est = fit_nw(GRID, xt, yy, hsel) if k == "nw" else fit_ll(GRID, xt, yy, hsel)
                realized[k].append(float(np.mean((est - mg) ** 2)))
        cvres[tag] = {"picks": picks, "realized": realized}
    res["exp4_h_grid"] = [float(x) for x in hcv]
    for tag in cvres:
        for k in ["nw", "ll"]:
            res[f"exp4_{tag}_{k}_h_med"] = float(np.median(cvres[tag]["picks"][k]))
            res[f"exp4_{tag}_{k}_mse"] = float(np.mean(cvres[tag]["realized"][k]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for tag, col in [("iid", "#2c7fb8"), ("ar1", "#c0392b")]:
        a1.hist(cvres[tag]["picks"]["ll"], bins=hcv, alpha=0.55, label=f"LL {tag}", color=col)
    a1.set_xlabel("LOOCV selected h"); a1.set_ylabel("count")
    a1.set_title("LOOCV bandwidth: independent vs AR(1) errors")
    a1.legend(fontsize=8)
    labels = ["iid NW", "iid LL", "AR1 NW", "AR1 LL"]
    vals = [res["exp4_iid_nw_mse"], res["exp4_iid_ll_mse"],
            res["exp4_ar1_nw_mse"], res["exp4_ar1_ll_mse"]]
    a2.bar(labels, vals, color=["#e0a3a3", "#8fbfe0", "#c0392b", "#2c7fb8"])
    for i, v in enumerate(vals):
        a2.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("true MSE on grid")
    a2.set_title("CV-chosen h, evaluated against truth")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_loocv.png"), dpi=130); plt.close()

    # exp5: edge alpha in a score-to-return curve
    n2 = 900
    xs = rng.beta(2.0, 2.0, n2)
    def edge_alpha(s):
        return 0.001 * (s - 0.5) + 0.006 * np.exp(-((s - 0.9) / 0.07) ** 2)
    ys = edge_alpha(xs) + rng.normal(0, 0.012, n2)
    xq = np.linspace(0, 1, 201)
    hsel = 0.07
    nwq = fit_nw(xq, xs, ys, hsel)
    llq = fit_ll(xq, xs, ys, hsel)
    truth_q = edge_alpha(xq)
    edge_mask = xq >= 0.84
    res["exp5_h"] = hsel
    res["exp5_true_edge_mean"] = float(np.mean(truth_q[edge_mask]))
    res["exp5_nw_edge_mean"] = float(np.mean(nwq[edge_mask]))
    res["exp5_ll_edge_mean"] = float(np.mean(llq[edge_mask]))
    top = int(0.1 * n2)
    # evaluate the estimated top set against true alpha at those score points
    qnw = np.interp(xs, xq, nwq); qll = np.interp(xs, xq, llq)
    topn = np.argsort(qnw)[-top:]; topl = np.argsort(qll)[-top:]
    res["exp5_nw_top_true_alpha"] = float(np.mean(edge_alpha(xs[topn])))
    res["exp5_ll_top_true_alpha"] = float(np.mean(edge_alpha(xs[topl])))
    res["exp5_top_overlap"] = float(len(set(topn) & set(topl)) / top)
    res["exp5_nw_edge_bias"] = float(np.mean(nwq[edge_mask] - truth_q[edge_mask]))
    res["exp5_ll_edge_bias"] = float(np.mean(llq[edge_mask] - truth_q[edge_mask]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.scatter(xs, ys, s=5, color="#bbb", alpha=0.4)
    a1.plot(xq, truth_q, color="#222", lw=2, label="true alpha")
    a1.plot(xq, nwq, color="#c0392b", lw=1.8, label="NW fit")
    a1.plot(xq, llq, color="#2c7fb8", lw=1.8, label="LL fit")
    a1.axvspan(0.84, 1.0, color="#f5e6e6")
    a1.set_xlabel("score s"); a1.set_ylabel("expected return")
    a1.set_title("Edge alpha sits where data are sparse")
    a1.legend(fontsize=8)
    a2.bar(["NW", "LL"], [res["exp5_nw_top_true_alpha"], res["exp5_ll_top_true_alpha"]],
           color=["#c0392b", "#2c7fb8"])
    a2.axhline(res["exp5_true_edge_mean"], color="#222", ls="--", lw=1.5, label="truth edge mean")
    for i, v in enumerate([res["exp5_nw_top_true_alpha"], res["exp5_ll_top_true_alpha"]]):
        a2.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("true alpha of selected top 10%")
    a2.set_title(f"top-set overlap {res['exp5_top_overlap']:.2f}")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_edge_alpha.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
