#!/usr/bin/env python3
"""
为文章「稳健统计在量化中的应用：用中位数与 MAD 替代均值方差抗极端值」
(robust-statistics-quant) 生成真实配图。所有图表均由文中方法真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 合成日收益: 正态基底 + 注入 5 个 -15%~-25% 的闪崩式离群值(单向,模拟踩雷/数据错误)
  * 经典统计: 样本均值 / 样本标准差 / 皮尔逊相关
  * 稳健统计: 中位数 / MAD(中位数绝对偏差) -> 稳健波动率 σ=1.4826·MAD
              / Winsorized-Pearson(按 MAD 裁剪边际) + Spearman 秩相关
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
D = os.path.join(BASE, "robust-statistics-quant")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "base": "#55A868",
     "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3",
     "null": "#999999", "out": "#C44E52"}


def winsorize(x, k=2.5):
    """按稳健尺度(1.4826·MAD)裁剪到 median ± k·σ_rob。"""
    med = np.median(x)
    sigma = 1.4826 * np.median(np.abs(x - med))
    lo, hi = med - k * sigma, med + k * sigma
    return np.clip(x, lo, hi)


def main():
    rng = np.random.default_rng(42)
    T = 500
    base_mu = 0.0006
    base_sd = 0.012
    base = rng.normal(base_mu, base_sd, T)
    # 注入 5 个单向闪崩(踩雷/数据错误): 全部为负, 显著拉低均值、抬高方差
    out_idx = rng.choice(T, 5, replace=False)
    shocks = -rng.uniform(0.15, 0.25, 5)
    r = base.copy()
    r[out_idx] += shocks

    # ---- 经典 vs 稳健 全样本估计 ----
    mean_cls = r.mean()
    std_cls = r.std(ddof=1)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    std_rob = 1.4826 * mad

    # ---- 滚动窗口 ----
    W = 60
    roll_std = np.full(T, np.nan)
    roll_rob = np.full(T, np.nan)
    for t in range(W, T):
        seg = r[t - W:t]
        roll_std[t] = seg.std(ddof=1)
        m2 = np.median(seg)
        roll_rob[t] = 1.4826 * np.median(np.abs(seg - m2))

    # ---- 双资产相关: 含 6 个独立 fat-finger 离群 ----
    rng2 = np.random.default_rng(7)
    n = 200
    a = rng2.normal(0, 1, n)
    b = 0.6 * a + rng2.normal(0, 0.8, n)          # 真实皮尔逊 ~0.60
    oi = rng2.choice(n, 6, replace=False)
    a[oi] += rng2.choice([-1, 1], 6) * rng2.uniform(7, 9, 6)   # 单边极端 fat-finger
    b[oi] += rng2.choice([-1, 1], 6) * rng2.uniform(7, 9, 6)
    pear = np.corrcoef(a, b)[0, 1]
    spear = np.corrcoef(np.argsort(a), np.argsort(b))[0, 1]
    aw, bw = winsorize(a), winsorize(b)
    pear_w = np.corrcoef(aw, bw)[0, 1]

    print("==== DIAGNOSTICS (robust-statistics) ====")
    print(f"T={T}, crash outliers injected={len(out_idx)} (all negative)")
    print(f"classic mean={mean_cls:.6f}  robust median={med:.6f}  gap={mean_cls-med:.6f}")
    print(f"classic std={std_cls:.6f}  robust MAD-vol={std_rob:.6f}  true base sd={base_sd}")
    print(f"classic std inflation vs base={std_cls/base_sd:.2f}x  robust ratio={std_rob/base_sd:.2f}x")
    print(f"max rolling classic std={np.nanmax(roll_std):.5f}  max rolling robust vol={np.nanmax(roll_rob):.5f}")
    print(f"pearson(classic)={pear:.3f}  spearman={spear:.3f}  winsorized-pearson={pear_w:.3f}  (true 0.60)")

    # ===== Fig 1: 含闪崩离群值的收益序列 =====
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(r, color=C["x"], lw=1.0, label="日收益序列", zorder=2)
    ax.scatter(out_idx, r[out_idx], color=C["out"], s=55, zorder=3, label="注入闪崩离群值")
    ax.axhline(mean_cls, color=C["band"], ls="--", lw=1.3, label=f"样本均值 {mean_cls:.4f}")
    ax.axhline(med, color=C["base"], ls="--", lw=1.3, label=f"中位数 {med:.4f}")
    ax.set_title("合成日收益：5 个单向闪崩把样本均值拉低, 中位数岿然不动", fontsize=12)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, color=C["grid"])
    plt.tight_layout()
    fig.savefig(os.path.join(D, "returns_with_outliers.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 2: 滚动经典波动率 vs 稳健波动率 =====
    fig, ax = plt.subplots(figsize=(10, 4.8))
    tt = np.arange(T)
    ax.plot(tt, roll_std, color=C["band"], lw=1.2, label="滚动 60 日样本标准差(经典)")
    ax.plot(tt, roll_rob, color=C["base"], lw=1.4, label="滚动 60 日 MAD 稳健波动率")
    ax.axhline(base_sd, color=C["null"], ls=":", lw=1.0, label=f"真实基底波动率 {base_sd}")
    ax.set_title("滚动波动率：闪崩尖峰只冲击经典估计, 稳健估计几乎不动", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color=C["grid"])
    plt.tight_layout()
    fig.savefig(os.path.join(D, "rolling_vol_robust_vs_classic.png"), dpi=130)
    plt.close(fig)

    # ===== Fig 3: 双资产散点 + 相关对比 =====
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(a, b, s=18, color=C["x"], alpha=0.6, label="资产日收益(含 fat-finger 离群)")
    ax.scatter(a[oi], b[oi], s=50, color=C["out"], label="离群点")
    ax.set_title(f"双资产散点：离群点摧毁皮尔逊相关\n皮尔逊={pear:.2f}  稳健(Winsorized)={pear_w:.2f}  Spearman={spear:.2f}  (真实 0.60)", fontsize=11)
    ax.set_xlabel("资产 A 标准化收益")
    ax.set_ylabel("资产 B 标准化收益")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color=C["grid"])
    plt.tight_layout()
    fig.savefig(os.path.join(D, "robust_correlation_scatter.png"), dpi=130)
    plt.close(fig)

    print("images saved to", D)
    print(os.listdir(D))


if __name__ == "__main__":
    main()
