#!/usr/bin/env python3
"""
为文章「Stambaugh-Yuan 误定价因子」(stambaugh-yuan-mispricing) 生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn/torch 依赖）：

  1) cover.png               —— MGMT / PERF 两个误定价因子 vs 传统 FF 因子的月均风险溢价
  2) sy_factor_absorb.png    —— 17 个异象在 FF3 下 vs FF3+MGMT+PERF 下的 alpha 被「吸收」对照
  3) sy_anomaly_corr.png      —— 17 个异象因子模拟组合的相关矩阵，前 2 个 PC 解释绝大部分

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 真实资产定价由 FF3（Mkt/SMB/HML）+ 2 个潜在误定价因子 MGMT/PERF 完全驱动（真 alpha=0）
  - 17 个「异象」是 MGMT/PERF 的线性组合 + 特质噪声（即它们是同一误定价的两种症状）
  - 用 PCA（带符号约束）从 17 个 FMP 抽出前 2 个主成分 → 还原 MGMT/PERF
  - 检验：FF3 模型下异象留显著 alpha；加入 MGMT+PERF 后 alpha 塌到 ~0
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
IMG = os.path.join(BASE, "stambaugh-yuan-mispricing")
os.makedirs(IMG, exist_ok=True)

C = {"mkt": "#4C72B0", "smb": "#55A868", "hml": "#E1A100",
     "mgmt": "#C44E52", "perf": "#8172B3", "neg": "#C44E52", "pos": "#55A868"}

T = 360  # 360 个月
N_ANOM = 17

# FF3 因子风险溢价（月均）
FF_LAM = {"Mkt": 0.006, "SMB": 0.0025, "HML": 0.004}
FF_SD = {"Mkt": 0.044, "SMB": 0.026, "HML": 0.026}
# 两个误定价因子（MGMT 管理行为 / PERF 业绩交易）的风险溢价
MGMT_LAM, MGMT_SD = 0.0045, 0.030
PERF_LAM, PERF_SD = 0.0035, 0.034


def gen_factors(seed=20260722):
    rng = np.random.default_rng(seed)
    # FF3 之间轻度相关
    corr = np.array([
        [1.00, 0.20, 0.15],
        [0.20, 1.00, 0.30],
        [0.15, 0.30, 1.00],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.normal(0, 1, size=(T, 3)) @ L.T
    F = np.empty((T, 3))
    for j, name in enumerate(["Mkt", "SMB", "HML"]):
        F[:, j] = FF_LAM[name] + FF_SD[name] * z[:, j]
    # MGMT / PERF：彼此弱相关，与 FF3 弱相关
    zm = rng.normal(0, 1, T)
    zp = 0.25 * zm + np.sqrt(1 - 0.25 ** 2) * rng.normal(0, 1, T)
    MGMT = MGMT_LAM + MGMT_SD * zm
    PERF = PERF_LAM + PERF_SD * zp
    return F, MGMT, PERF


# 17 个异象：前 11 个偏 MGMT（管理/发行行为），后 6 个偏 PERF（业绩/交易）
# 每个异象的载荷 (b_mkt, b_smb, b_hml, b_mgmt, b_perf)
def anomaly_loadings():
    rng = np.random.default_rng(42)
    loads = np.zeros((N_ANOM, 5))
    for i in range(N_ANOM):
        if i < 11:  # MGMT 簇
            b_mgmt = 0.6 + 0.3 * rng.random()
            b_perf = 0.1 * rng.normal()
        else:       # PERF 簇
            b_perf = 0.6 + 0.3 * rng.random()
            b_mgmt = 0.1 * rng.normal()
        b_mkt = rng.normal(0, 0.1)
        b_smb = 0.2 * rng.normal()
        b_hml = 0.2 * rng.normal()
        loads[i] = [b_mkt, b_smb, b_hml, b_mgmt, b_perf]
    # 归一化 MGMT/PERF 载荷使其方向清晰
    return loads


def gen_anomalies(F, MGMT, PERF, loads, seed=7):
    rng = np.random.default_rng(seed)
    A = np.zeros((T, N_ANOM))
    for i in range(N_ANOM):
        b_mkt, b_smb, b_hml, b_mgmt, b_perf = loads[i]
        idio = rng.normal(0, 0.012, T)
        # 真 alpha = 0：完全由 FF3 + MGMT + PERF 解释
        A[:, i] = (b_mkt * F[:, 0] + b_smb * F[:, 1] + b_hml * F[:, 2]
                   + b_mgmt * MGMT + b_perf * PERF + idio)
    return A


def ts_alpha(y, X):
    """时间序列回归 y = a + X@b + e，返回年化 alpha 与 R²。"""
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    coef, *_, sv = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ coef
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return coef[0] * 12.0, r2


def pca_sign(factor_matrix):
    """对 (T x K) 因子矩阵做 PCA，返回前 2 主成分时间序列（带符号约束：与 MGMT/PERF 正相关）。"""
    X = factor_matrix - factor_matrix.mean(0)
    cov = X.T @ X / (len(X) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    pc1 = X @ eigvec[:, 0]
    pc2 = X @ eigvec[:, 1]
    return pc1, pc2, eigval


# ---------------------------------------------------------------------------
# 1) 封面：因子月均风险溢价对比
# ---------------------------------------------------------------------------
def make_cover(F, MGMT, PERF):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = ["Mkt", "SMB", "HML", "MGMT", "PERF"]
    means = [F[:, 0].mean(), F[:, 1].mean(), F[:, 2].mean(), MGMT.mean(), PERF.mean()]
    cols = [C["mkt"], C["smb"], C["hml"], C["mgmt"], C["perf"]]
    bars = ax.bar(names, np.array(means) * 100, color=cols, width=0.62, alpha=0.9)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m * 100 + 0.03,
                f"{m*100:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("月均风险溢价 (%)")
    ax.set_title("MGMT / PERF：两个误定价因子，量级堪比传统因子", fontsize=14,
                 fontweight="bold")
    ax.text(0.5, -0.22, "合成数据，量级参考历史（MGMT=管理行为异象簇，PERF=业绩交易异象簇）",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555")
    fig.savefig(os.path.join(IMG, "cover.png"))
    plt.close(fig)
    return means


# ---------------------------------------------------------------------------
# 2) alpha 吸收对照：FF3 vs FF3 + MGMT + PERF
# ---------------------------------------------------------------------------
def make_absorb(F, MGMT, PERF, A):
    ff3_alpha, full_alpha = [], []
    for i in range(N_ANOM):
        a3, _ = ts_alpha(A[:, i], F)
        a_full, _ = ts_alpha(A[:, i], np.column_stack([F, MGMT, PERF]))
        ff3_alpha.append(a3)
        full_alpha.append(a_full)
    ff3_alpha = np.array(ff3_alpha)
    full_alpha = np.array(full_alpha)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(N_ANOM)
    w = 0.4
    ax.bar(x - w/2, ff3_alpha * 100, w, label="FF3 下 α", color=C["neg"], alpha=0.85)
    ax.bar(x + w/2, full_alpha * 100, w, label="FF3+MGMT+PERF 下 α", color=C["pos"], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("17 个异象（因子模拟组合）")
    ax.set_ylabel("年化 α (%)")
    ax.set_title("两个误定价因子几乎吃光 17 个异象的 α", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"A{i+1}" for i in range(N_ANOM)], fontsize=8)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(IMG, "sy_factor_absorb.png"))
    plt.close(fig)
    return dict(mean_ff3=np.mean(np.abs(ff3_alpha)) * 100,
                mean_full=np.mean(np.abs(full_alpha)) * 100,
                max_ff3=np.max(np.abs(ff3_alpha)) * 100,
                max_full=np.max(np.abs(full_alpha)) * 100)


# ---------------------------------------------------------------------------
# 3) 异象相关矩阵 + PC 解释力
# ---------------------------------------------------------------------------
def make_corr(A):
    corr = np.corrcoef(A.T)
    pc1, pc2, eigval = pca_sign(A)
    tot = eigval.sum()
    pct = eigval[:2] / tot * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title("17 个异象 FMP 相关矩阵", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("异象"); axes[0].set_ylabel("异象")
    axes[0].set_xticks(range(N_ANOM)); axes[0].set_yticks(range(N_ANOM))
    axes[0].set_xticklabels([f"A{i+1}" for i in range(N_ANOM)], fontsize=7)
    axes[0].set_yticklabels([f"A{i+1}" for i in range(N_ANOM)], fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    # 碎石图
    ncomp = min(8, len(eigval))
    axes[1].plot(range(1, ncomp + 1), eigval[:ncomp] / tot * 100, "o-",
                 color=C["mgmt"], lw=2)
    axes[1].axvline(2, color="#888", ls="--", lw=1)
    axes[1].set_xlabel("主成分序号")
    axes[1].set_ylabel("解释方差占比 (%)")
    axes[1].set_title(f"前 2 个 PC 解释 {pct.sum():.1f}% 截面方差", fontsize=12, fontweight="bold")
    axes[1].text(2.05, eigval[1] / tot * 100, f"PC1+PC2\n{pct.sum():.1f}%",
                 fontsize=9, color="#444")
    fig.savefig(os.path.join(IMG, "sy_anomaly_corr.png"))
    plt.close(fig)
    return pct, eigval[:2] / tot * 100


if __name__ == "__main__":
    F, MGMT, PERF = gen_factors()
    loads = anomaly_loadings()
    A = gen_anomalies(F, MGMT, PERF, loads)
    means = make_cover(F, MGMT, PERF)
    absorb = make_absorb(F, MGMT, PERF, A)
    pct, top2 = make_corr(A)
    # 还原度：构造的 MGMT/PERF 与抽出的 PC 的相关
    pc1, pc2, _ = pca_sign(A)
    # 让 pc1 对齐 MGMT（符号）
    if np.corrcoef(pc1, MGMT)[0, 1] < 0:
        pc1 = -pc1
    if np.corrcoef(pc2, PERF)[0, 1] < 0:
        pc2 = -pc2
    corr_pc1_mgmt = np.corrcoef(pc1, MGMT)[0, 1]
    corr_pc2_perf = np.corrcoef(pc2, PERF)[0, 1]
    print("=" * 60)
    print("因子月均溢价(%):", {n: round(m*100, 3) for n, m in
          zip(["Mkt", "SMB", "HML", "MGMT", "PERF"], means)})
    print(f"FF3 下平均 |α|         = {absorb['mean_ff3']:.3f}%")
    print(f"FF3+MGMT+PERF 下平均 |α| = {absorb['mean_full']:.3f}%")
    print(f"FF3 下最大 |α|         = {absorb['max_ff3']:.3f}%")
    print(f"FF3+MGMT+PERF 下最大 |α| = {absorb['max_full']:.3f}%")
    print(f"前 2 个 PC 解释方差     = {pct.sum():.1f}%  (PC1={top2[0]:.1f}%, PC2={top2[1]:.1f}%)")
    print(f"PC1 与 MGMT 相关       = {corr_pc1_mgmt:.3f}")
    print(f"PC2 与 PERF 相关       = {corr_pc2_perf:.3f}")
    # 组合误定价因子（MGMT+PERF 等权）月均溢价
    misp = 0.5 * (MGMT + PERF)
    print(f"组合误定价因子月均溢价  = {misp.mean()*100:.3f}%  (年化≈{misp.mean()*1200:.2f}%)")
    print("=" * 60)
