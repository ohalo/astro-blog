#!/usr/bin/env python3
"""
为文章「False Discovery Rate：用 BH 程序在多因子检验里控制假阳性」
(false-discovery-rate-fdr) 生成真实配图。所有图表均由文中方法真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 模拟 m=1000 个因子 IC 检验: 前 50 个为真有预测力(IC≈0.05), 其余 950 个为纯噪声
  * 对每个因子做 t 检验得到 p 值
  * Benjamini-Hochberg (1995) 控制 FDR 于 q: 排序 p_(k), 取最大 k 使 p_(k) <= k/m*q
  * 对比: 朴素阈值(p<0.05=α) / Bonferroni(FWER) / BH(q=0.05)
  * 因合成数据已知 ground truth, 可统计假阳性/真阳性
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "false-discovery-rate-fdr")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "base": "#55A868",
     "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3",
     "null": "#999999", "tp": "#55A868", "fp": "#C44E52"}


def bh_threshold(pvals, q=0.05):
    """返回 BH 拒绝集(布尔数组)与阈值 p_(k*)。"""
    m = len(pvals)
    order = np.argsort(pvals)
    ps = pvals[order]
    # 找最大 k 使 p_(k) <= k/m * q
    k_star = 0
    for k in range(1, m + 1):
        if ps[k - 1] <= k / m * q:
            k_star = k
    thresh = k_star / m * q if k_star > 0 else 0.0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k_star]] = True
    return rej, thresh


def main():
    rng = np.random.default_rng(2026)
    m = 1000
    n_obs = 250          # 每个因子 250 个交易日 IC
    true_sig = 50        # 前 50 个因子真有预测力
    ic_true = 0.05

    # 生成每个因子的 IC 序列并做单样本 t 检验(IC 是否非零)
    ic = np.zeros((m, n_obs))
    for i in range(m):
        mu = ic_true if i < true_sig else 0.0
        ic[i] = rng.normal(mu, 0.10, n_obs)
    tstat = np.sqrt(n_obs) * ic.mean(axis=1) / (ic.std(axis=1, ddof=1) + 1e-12)
    # 双尾 p 值(正态近似)
    from scipy import stats
    pvals = 2 * (1 - stats.norm.cdf(np.abs(tstat)))

    # 三种多重检验校正
    rej_naive, _ = (pvals < 0.05), 0.05
    rej_bonf, _ = (pvals < 0.05 / m), 0.05 / m
    rej_bh, thresh_bh = bh_threshold(pvals, q=0.05)

    is_true = np.arange(m) < true_sig
    tp_naive = int((rej_naive & is_true).sum())
    fp_naive = int((rej_naive & ~is_true).sum())
    tp_bonf = int((rej_bonf & is_true).sum())
    fp_bonf = int((rej_bonf & ~is_true).sum())
    tp_bh = int((rej_bh & is_true).sum())
    fp_bh = int((rej_bh & ~is_true).sum())
    fdr_naive = fp_naive / max(tp_naive + fp_naive, 1)
    fdr_bh = fp_bh / max(tp_bh + fp_bh, 1)

    print("==== DIAGNOSTICS (FDR) ====")
    print(f"m={m}, true signals={true_sig}, n_obs={n_obs}")
    print(f"naive  : disc={tp_naive+fp_naive} TP={tp_naive} FP={fp_naive} FDR={fdr_naive:.3f}")
    print(f"bonf   : disc={tp_bonf+fp_bonf} TP={tp_bonf} FP={fp_bonf} FDR={fp_bonf/max(tp_bonf+fp_bonf,1):.3f}")
    print(f"BH(q=.05): disc={tp_bh+fp_bh} TP={tp_bh} FP={fp_bh} FDR={fdr_bh:.3f}  BH-thresh={thresh_bh:.5f}")
    missed_bh = true_sig - tp_bh
    print(f"BH missed true signals={missed_bh}")

    # ===== Fig 1: 排序 p 值 + BH 阈值线(按 ground truth 着色) =====
    order = np.argsort(pvals)
    ps = pvals[order]
    ks = np.arange(1, m + 1)
    bh_line = ks / m * 0.05
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [C["tp"] if is_true[order[i]] else C["fp"] for i in range(m)]
    ax.scatter(ks, ps, s=8, c=colors, alpha=0.6)
    ax.plot(ks, bh_line, color=C["eq"], lw=1.8, label="BH 临界线 k/m·q (q=0.05)")
    ax.axhline(0.05, color=C["null"], ls="--", lw=1.2, label="朴素阈值 α=0.05")
    ax.set_yscale("log")
    ax.set_xlabel("排序后的假设序号 k (从小到大)")
    ax.set_ylabel("p 值 (对数轴)")
    ax.set_title("BH 程序：在排序 p 值图上, 最后一个穿过临界线的点决定拒绝集", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color=C["grid"], which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "bh_sorted_pvals.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 2: 不同 q 下的发现数 / 假阳性数曲线 =====
    qs = np.linspace(0.001, 0.20, 60)
    disc_n, fp_n, tp_n = [], [], []
    for q in qs:
        rj, _ = bh_threshold(pvals, q=q)
        disc_n.append(int(rj.sum()))
        fp_n.append(int((rj & ~is_true).sum()))
        tp_n.append(int((rj & is_true).sum()))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(qs, disc_n, color=C["x"], lw=1.6, label="总发现数")
    ax.plot(qs, tp_n, color=C["tp"], lw=1.6, label="真阳性(已知 ground truth)")
    ax.plot(qs, fp_n, color=C["fp"], lw=1.6, label="假阳性")
    ax.axvline(0.05, color=C["eq"], ls="--", lw=1.2, label="q=0.05")
    ax.set_xlabel("BH 目标 FDR 水平 q")
    ax.set_ylabel("因子数量")
    ax.set_title("q 越大越宽松: 发现数上升, 但假阳性同步上升", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color=C["grid"])
    plt.tight_layout()
    fig.savefig(os.path.join(D, "fdr_vs_q_curve.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 3: 三种方法对比柱状图 =====
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["朴素 α=0.05", "Bonferroni", "BH q=0.05"]
    tps = [tp_naive, tp_bonf, tp_bh]
    fps = [fp_naive, fp_bonf, fp_bh]
    x = np.arange(len(methods))
    ax.bar(x, tps, color=C["tp"], label="真阳性")
    ax.bar(x, fps, bottom=tps, color=C["fp"], label="假阳性")
    for i, (tp, fp) in enumerate(zip(tps, fps)):
        ax.text(i, tp + fp + 3, f"{tp}+{fp}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("因子数量")
    ax.set_title("1000 个因子中 50 个真有效: 三种多重检验校正对比", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color=C["grid"], axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(D, "fdr_methods_compare.png"), dpi=130)
    plt.close(fig)

    print("images saved to", D)
    print(os.listdir(D))


if __name__ == "__main__":
    main()
