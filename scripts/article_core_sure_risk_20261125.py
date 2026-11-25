# -*- coding: utf-8 -*-
"""SURE 无偏风险估计：Soft 阈值 Stein 引理成立、Hard 阈值边界项破功。
自包含：数据生成 + 实验 + 画图 + 结果 JSON 全部在本文件，可直接 exec 复跑。"""
import os
import json
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261125)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "stein-unbiased-risk-estimation")
os.makedirs(OUT, exist_ok=True)


def soft(y, t):
    return np.sign(y) * np.maximum(np.abs(y) - t, 0.0)


def hard(y, t):
    return np.where(np.abs(y) > t, y, 0.0)


def sure_soft(y, t, sigma2):
    """连续软阈值：Stein 引理适用，df = #{|y|>t}。"""
    return float(np.sum(np.minimum(y * y, t * t)) + 2 * sigma2 * np.sum(np.abs(y) > t) - len(y) * sigma2)


def sure_hard_naive(y, t, sigma2):
    """把软阈值自由度公式硬套在不连续硬阈值上（文章中的反例）。"""
    keep = np.abs(y) > t
    return float(np.sum(y[~keep] ** 2) + 2 * sigma2 * np.sum(keep) - len(y) * sigma2)


def boundary_term(theta, t, sigma2):
    """硬阈值跳跃的 Stein 边界修正（oracle 版）：2*sigma^2*E[deltag/dy] 的跳变贡献。"""
    s = np.sqrt(sigma2)
    return float(2 * sigma2 * np.sum(t / s * (norm.pdf((t - theta) / s) + norm.pdf((t + theta) / s))))


def risk(est, theta):
    return float(np.mean((est - theta) ** 2))


def make_theta(n, s, rng):
    th = np.zeros(n)
    if s:
        ii = rng.choice(n, s, replace=False)
        th[ii] = rng.choice([-1, 1], s) * rng.uniform(1.6, 3.0, s)
    return th


def cv_pick_soft(y, folds=5, seeds=(11, 29, 47, 83, 101)):
    """5 折 plug-in 风险选软阈值：E[(y_te-theta)^2]=(1-c)^2*theta^2 + sigma^2*c^2，
    其中保留时 c=1-t/|y|，置零时 c=0；代入 theta^2=y^2-sigma^2（边际无偏）。"""
    ts = np.linspace(0, 3.0, 151)
    err = np.zeros_like(ts)
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(len(y))
        for k in range(folds):
            te = perm[k::folds]
            for j, t in enumerate(ts):
                c = np.maximum(1 - t / np.abs(y[te]), 0.0)
                err[j] += np.sum((1 - c) ** 2 * (y[te] ** 2 - 1.0) + c ** 2)
    return float(ts[np.argmin(err)])


def main():
    rng = RNG
    n = 400
    res = {}
    tg = np.linspace(0, 3.0, 301)

    # --- exp1: 单次样本上四条曲线（软 SURE / 软真风险 / 硬误用 SURE / 硬真风险） ---
    th = make_theta(n, 24, rng)
    y = th + rng.normal(0, 1.0, n)
    s_curve = np.array([sure_soft(y, t, 1.0) for t in tg]) / n
    r_soft = np.array([risk(soft(y, t), th) for t in tg])
    h_curve = np.array([sure_hard_naive(y, t, 1.0) for t in tg]) / n
    r_hard = np.array([risk(hard(y, t), th) for t in tg])
    t_sure = float(tg[np.argmin(s_curve)])
    t_cv = cv_pick_soft(y)
    t_or = float(tg[np.argmin(r_soft)])
    res["exp1_t_sure_soft"] = t_sure
    res["exp1_t_cv_soft"] = t_cv
    res["exp1_t_oracle_soft"] = t_or
    res["exp1_risk_sure_pick"] = float(r_soft[np.argmin(s_curve)])
    res["exp1_risk_cv_pick"] = float(r_soft[np.argmin(np.abs(tg - t_cv))])
    res["exp1_risk_oracle"] = float(np.min(r_soft))
    res["exp1_risk_no_shrink"] = float(r_soft[0])
    res["exp1_sure_minus_risk_mean"] = float(np.mean(s_curve - r_soft))
    res["exp1_hardnaive_minus_risk_mean"] = float(np.mean(h_curve - r_hard))

    plt.figure(figsize=(8.4, 5))
    plt.plot(tg, s_curve, color="#c0392b", lw=2.2, label="SURE-soft (valid)")
    plt.plot(tg, r_soft, color="#2c7fb8", lw=2, ls="--", label="true risk soft")
    plt.plot(tg, h_curve, color="#e67e22", lw=1.8, label="naive SURE on hard (invalid)")
    plt.plot(tg, r_hard, color="#27ae60", lw=1.8, ls="--", label="true risk hard")
    plt.axvline(t_sure, color="#c0392b", ls=":")
    plt.axvline(t_or, color="#2c7fb8", ls=":")
    plt.xlabel("threshold t"); plt.ylabel("risk per coordinate")
    plt.title("Soft SURE tracks risk; hard-threshold df formula fails")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig01_sure_vs_risk.png"), dpi=130); plt.close()

    # --- exp2: 500 次重复验证无偏性 + 跳变边界项理论对拍 ---
    t0 = 1.4
    acc = {"ss": 0.0, "rs": 0.0, "hn": 0.0, "rh": 0.0, "corr": 0.0}
    reps = 400
    for _ in range(reps):
        th2 = make_theta(n, 24, rng)
        y2 = th2 + rng.normal(0, 1.0, n)
        acc["ss"] += sure_soft(y2, t0, 1.0) / n
        acc["rs"] += risk(soft(y2, t0), th2)
        acc["hn"] += sure_hard_naive(y2, t0, 1.0) / n
        acc["rh"] += risk(hard(y2, t0), th2)
        acc["corr"] += boundary_term(th2, t0, 1.0) / n
    for k in acc:
        acc[k] /= reps
    res["exp2_t"] = t0
    res["exp2_reps"] = reps
    res["exp2_soft_sure_mean"] = acc["ss"]
    res["exp2_soft_risk_mean"] = acc["rs"]
    res["exp2_soft_bias"] = acc["ss"] - acc["rs"]
    res["exp2_hard_naive_mean"] = acc["hn"]
    res["exp2_hard_risk_mean"] = acc["rh"]
    res["exp2_hard_bias"] = acc["hn"] - acc["rh"]
    res["exp2_boundary_theory"] = acc["corr"]
    res["exp2_hard_bias_plus_boundary"] = acc["hn"] + acc["corr"] - acc["rh"]

    labels = ["SURE-soft", "risk-soft", "naive SURE-hard", "risk-hard", "boundary term"]
    vals = [acc["ss"], acc["rs"], acc["hn"], acc["rh"], acc["corr"]]
    cols = ["#c0392b", "#2c7fb8", "#e67e22", "#27ae60", "#8e44ad"]
    plt.figure(figsize=(8.4, 4.8))
    bars = plt.bar(labels, vals, color=cols)
    plt.axhline(0, color="#555", lw=0.8)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9)
    plt.ylabel("mean risk per coordinate")
    plt.title(f"{reps} repeats: soft bias = {acc['ss']-acc['rs']:.4f}, hard bias = {acc['hn']-acc['rh']:.4f}")
    plt.xticks(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig02_unbiased_check.png"), dpi=130); plt.close()

    # --- exp3: 稀疏度扫描：soft SURE / plug-in CV / oracle / 不收缩 ---
    sparsities = [0, 4, 12, 24, 60, 400]
    names = ["sure_soft", "cv_soft", "oracle_soft", "no_shrink", "naive_sure_hard"]
    rk = {k: [] for k in names}
    for s in sparsities:
        th3 = make_theta(n, s, rng)
        y3 = th3 + rng.normal(0, 1.0, n)
        sg = np.linspace(0, 3.0, 151)
        sc = np.array([sure_soft(y3, t, 1.0) for t in sg])
        rc = np.array([risk(soft(y3, t), th3) for t in sg])
        kh = np.array([sure_hard_naive(y3, t, 1.0) for t in sg])
        rh = np.array([risk(hard(y3, t), th3) for t in sg])
        rk["sure_soft"].append(float(rc[np.argmin(sc)]))
        rk["oracle_soft"].append(float(np.min(rc)))
        rk["no_shrink"].append(float(rc[0]))
        rk["naive_sure_hard"].append(float(rh[np.argmin(kh)]))
        tc = cv_pick_soft(y3)
        rk["cv_soft"].append(float(risk(soft(y3, tc), th3)))
    res["exp3_sparsities"] = sparsities
    res["exp3_risks"] = rk
    res["exp3_sure_vs_oracle_gap"] = float(np.mean(np.array(rk["sure_soft"]) - np.array(rk["oracle_soft"])))
    res["exp3_cv_vs_oracle_gap"] = float(np.mean(np.array(rk["cv_soft"]) - np.array(rk["oracle_soft"])))
    res["exp3_hard_vs_soft_bad_cases"] = int(sum(1 for a, b in zip(rk["naive_sure_hard"], rk["sure_soft"]) if a > b))

    x = np.arange(len(sparsities)); w = 0.16
    plt.figure(figsize=(9.4, 5))
    for i, (k, col) in enumerate(zip(names, ["#c0392b", "#e67e22", "#2c7fb8", "#7f8c8d", "#8e44ad"])):
        plt.bar(x + (i - 2) * w, rk[k], w, label=k, color=col)
    plt.xticks(x, [str(s) for s in sparsities]); plt.xlabel("nonzero coordinates (n=400)")
    plt.ylabel("risk E||th^-th||^2/n"); plt.title("Soft SURE across sparsity; naive hard formula hurts")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig03_sparsity.png"), dpi=130); plt.close()

    # --- exp4: sigma 错设 ---
    ratios = np.round(np.linspace(0.6, 1.6, 21), 3)
    th4 = make_theta(n, 24, rng)
    y4 = th4 + rng.normal(0, 1.0, n)
    sg = np.linspace(0, 3.0, 151)
    rc = np.array([risk(soft(y4, t), th4) for t in sg])
    picked, rr = [], []
    for q in ratios:
        sc = np.array([sure_soft(y4, t, q * q) for t in sg])
        k = int(np.argmin(sc)); picked.append(float(sg[k])); rr.append(float(rc[k]))
    res["exp4_ratios"] = [float(q) for q in ratios]
    res["exp4_picked_t"] = picked
    res["exp4_risk"] = rr
    res["exp4_risk_0.6"] = rr[0]
    res["exp4_risk_1.0"] = rr[np.argmin(np.abs(ratios - 1.0))]
    res["exp4_risk_1.6"] = rr[-1]
    res["exp4_t_0.6"] = picked[0]
    res["exp4_t_1.6"] = picked[-1]

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    a1.plot(ratios, picked, color="#c0392b", lw=2, label="picked t")
    a1.set_xlabel("sigma_hat / sigma_true"); a1.set_ylabel("picked t", color="#c0392b")
    a2 = a1.twinx()
    a2.plot(ratios, rr, color="#8e44ad", lw=2, label="realized risk")
    a2.set_ylabel("realized risk", color="#8e44ad")
    plt.title("Wrong noise scale shifts SURE choice")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_sigma_mismatch.png"), dpi=130); plt.close()

    # --- exp5: 非高斯新息下 Stein 引理破功 + 标量收缩 c 的 SURE ---
    biases = {}
    for dist in ["gauss", "t5"]:
        bb = []
        for _ in range(60):
            th5 = make_theta(n, 24, rng)
            eps = rng.normal(0, 1, n) if dist == "gauss" else rng.standard_t(5, n) / np.sqrt(5 / 3)
            y5 = th5 + eps
            sc = np.array([sure_soft(y5, t, 1.0) for t in sg])
            rc = np.array([risk(soft(y5, t), th5) for t in sg])
            bb.append(np.mean(sc / n - rc))
        biases[dist] = float(np.mean(bb))
    res["exp5_bias_gauss"] = biases["gauss"]
    res["exp5_bias_t5"] = biases["t5"]

    th6 = make_theta(n, 24, rng)
    y6 = th6 + rng.normal(0, 1.0, n)
    cs = np.linspace(0, 1.2, 121)
    sc6 = np.array([float(np.sum((1 - c) ** 2 * y6 * y6) + 2 * c * n - n) / n for c in cs])
    rc6 = np.array([risk(c * y6, th6) for c in cs])
    c_sure = float(cs[np.argmin(sc6)])
    res["exp5_c_sure"] = c_sure
    res["exp5_c_oracle"] = float(cs[np.argmin(rc6)])
    res["exp5_risk_sure_c"] = float(rc6[np.argmin(sc6)])
    res["exp5_risk_oracle_c"] = float(np.min(rc6))
    res["exp5_risk_no_shrink"] = float(rc6[-1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.bar(["Gaussian", "t(5) scaled"], [biases["gauss"], biases["t5"]],
           color=["#2c7fb8", "#c0392b"])
    a1.axhline(0, color="#555", lw=0.8)
    a1.set_ylabel("mean SURE - true risk")
    a1.set_title("Stein identity needs Gaussian noise")
    a2.plot(cs, sc6, color="#c0392b", lw=2, label="SURE(c)")
    a2.plot(cs, rc6, color="#2c7fb8", lw=2, ls="--", label="true risk(c)")
    a2.axvline(c_sure, color="#c0392b", ls=":")
    a2.set_xlabel("shrink factor c"); a2.legend(fontsize=8)
    a2.set_title("Scalar c: SURE picks near oracle")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_limits_and_shrink.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
