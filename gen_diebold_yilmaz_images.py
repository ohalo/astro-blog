#!/usr/bin/env python3
"""
为文章「溢出指数与连通性网络:用 Diebold-Yilmaz 把风险传染画成地图」
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由结构化 VAR 自洽合成，仅用于演示方法；真实落地见文末路径）：
  - DGP: r_t = A_t r_{t-1} + eps_t, 含常态/危机两态的 A（危机期股票→其他传染系数放大）
  - 估计 VAR(1) 后,用 Koop-Pesaran-Shi(KPS)广义预测误差方差分解(GFEVD)算方向性溢出
  - 连通性 d_ij(H): 行 i=被预测变量,列 j=冲击来源; 行归一使 Σ_j d_ij = 1
  - 标准 DY 口径: TO_i = Σ_{j≠i} d_ij(行离对角=净输出), FROM_i = Σ_{j≠i} d_ji(列离对角=净接收)
  - NET_i = TO_i − FROM_i; 总连通度 TCI = (1/N) Σ_{i≠j} d_ij · 100
  - 滚动窗口 TCI 展示危机期传染飙升
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "diebold-yilmaz-connectedness")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "blue": "#2F4B7C", "red": "#C44E52", "green": "#55A868",
     "purple": "#8172B3", "orange": "#FF7F0E"}
ASSETS = ["股票", "债券", "黄金", "原油", "外汇"]
N = len(ASSETS)

rng = np.random.default_rng(20260719)
T = 1600

# ---- 常态 VAR(1) 系数:股票是核心传染源(只向外传,不被其他变量滞后驱动) ----
# 行 i = 变量 i 受各变量滞后项驱动: r_i(t) = Σ_j A[i,j]·r_j(t-1) + eps_i
A_normal = np.array([
    [0.15, 0.00, 0.00, 0.00, 0.00],   # 股票: 仅自身持续,不受他人驱动
    [0.28, 0.10, 0.00, 0.00, 0.00],   # 债券 ← 股票
    [0.22, 0.00, 0.08, 0.00, 0.00],   # 黄金 ← 股票
    [0.32, 0.00, 0.00, 0.12, 0.00],   # 原油 ← 股票
    [0.20, 0.00, 0.00, 0.06, 0.05],   # 外汇 ← 股票
], dtype=float)
# 危机 A:股票→其他传染系数整体放大 ~2.4 倍(风险溢出骤升),并加入债券↔黄金避险共振
A_crisis = A_normal.copy()
A_crisis[1:, 0] = A_crisis[1:, 0] * 2.4
A_crisis[1, 2] = 0.12
A_crisis[2, 1] = 0.10

# 日度波动(对角协方差,资产特异性冲击)
vol = np.array([0.012, 0.004, 0.008, 0.018, 0.006])
Sigma_eps = np.diag(vol ** 2)
crisis_start, crisis_end = 700, 820

r = np.zeros((T, N))
r[0] = rng.normal(0, vol)
for t in range(1, T):
    in_crisis = crisis_start <= t <= crisis_end
    A = A_crisis if in_crisis else A_normal
    eps = rng.multivariate_normal(np.zeros(N), Sigma_eps)
    r[t] = A @ r[t - 1] + eps

# ---- VAR(1) 估计(OLS) ----
def fit_var1(x):
    Y = x[1:]
    Z = x[:-1]
    Ahat = np.linalg.lstsq(Z, Y, rcond=None)[0].T      # k x k
    resid = Y - Z @ Ahat
    Sigma = resid.T @ resid / len(resid)
    return Ahat, Sigma

# ---- KPS 广义预测误差方差分解(GFEVD, 非正交化,允许相关冲击) ----
def gfevd(Ahat, Sigma, H=10):
    k = Ahat.shape[0]
    Phi = [np.eye(k)]
    for h in range(1, H + 1):
        ph = np.zeros((k, k))
        for l in range(1, h + 1):
            ph += Ahat @ Phi[h - l]     # p=1
        Phi.append(ph)
    sig = np.diag(Sigma)
    denom = np.zeros(k)
    num = np.zeros((k, k))
    for h in range(H):
        Ph = Phi[h]
        for i in range(k):
            ei = np.zeros(k); ei[i] = 1
            denom[i] += ei @ Ph @ Sigma @ Ph.T @ ei
        for j in range(k):
            ej = np.zeros(k); ej[j] = 1
            for i in range(k):
                ei = np.zeros(k); ei[i] = 1
                num[i, j] += (ei @ Ph @ Sigma @ ej) ** 2 / sig[j]
    fevd = num / denom[:, None]          # k x k, 未归一
    d = fevd / fevd.sum(axis=1, keepdims=True)   # 行归一
    return d

# ---- 全样本连通性(标准 DY 口径) ----
Ahat, Sigma = fit_var1(r)
d = gfevd(Ahat, Sigma, H=10)
# 标准 DY 口径: TO_i(净输出/传染源)= 列 i 离对角(其他变量方差中被 i 冲击解释的部分)
#             FROM_i(净接收)= 行 i 离对角(变量 i 方差中被其他冲击解释的部分)
TO = (d.sum(axis=0) - np.diag(d)) * 100
FROM = (d.sum(axis=1) - np.diag(d)) * 100
NET = TO - FROM
TCI = (d.sum() - np.trace(d)) / N * 100

print("==== 全样本连通性矩阵 d_ij (行=被影响 i, 列=冲击来源 j, %) ====")
np.set_printoptions(precision=1, suppress=True)
print((d * 100).round(1))
print("\n资产   TO     FROM   NET")
for i in range(N):
    print(f"{ASSETS[i]:<5} {TO[i]:5.1f}  {FROM[i]:5.1f}  {NET[i]:+5.1f}")
print(f"\n总连通度 TCI = {TCI:.1f}%")

# ---- 滚动 TCI ----
WW = 250
tci_series = []
for end in range(WW, T):
    sub = r[end - WW:end]
    Ah, Sh = fit_var1(sub)
    dd = gfevd(Ah, Sh, H=10)
    tci_series.append((dd.sum() - np.trace(dd)) / N * 100)
tci_series = np.array(tci_series)
print(f"\n滚动 TCI: 均值 {tci_series.mean():.1f}%, 危机窗口峰值 {tci_series.max():.1f}%, "
      f"常态区均值 {tci_series[:crisis_start-WW].mean():.1f}%")

# ============================================================
# 图1:连通性热力图
# ============================================================
def fig_heatmap():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    M = d * 100
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=max(40, M.max()))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(ASSETS, fontsize=11); ax.set_yticklabels(ASSETS, fontsize=11)
    ax.set_xlabel("冲击来源 j", fontsize=11); ax.set_ylabel("被影响 i", fontsize=11)
    for i in range(N):
        for j in range(N):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > M.max() * 0.55 else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("方差贡献占比 %", fontsize=10)
    ax.set_title("Diebold-Yilmaz 连通性矩阵\n(行=被影响, 列=冲击来源, 对角=自身占比)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "connectedness_heatmap.png"), dpi=130)
    plt.close(fig)

# ============================================================
# 图2:连通性网络(红=净输出/传染源, 蓝=净接收, 边宽∝溢出强度)
# ============================================================
def fig_network():
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(N)}
    for i in range(N):
        x, y = pos[i]
        col = C["red"] if NET[i] > 0 else C["blue"]
        ax.scatter([x], [y], s=1000, c=col, edgecolors="black", zorder=3)
        ax.text(x, y, ASSETS[i], ha="center", va="center", fontsize=10,
                color="white", fontweight="bold", zorder=4)
        ax.text(x * 1.22, y * 1.22, f"NET={NET[i]:+.0f}", ha="center", va="center",
                fontsize=8.5, color=col, zorder=4)
    # 有向边 j→i (j 冲击来源, i 被影响), 宽度 ∝ d_ji, 仅画 > 6% 的强溢出
    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            w = d[i, j] * 100
            if w < 6:
                continue
            x0, y0 = pos[j]; x1, y1 = pos[i]
            dx, dy = x1 - x0, y1 - y0
            sx, sy = x0 + dx * 0.24, y0 + dy * 0.24
            ex, ey = x1 - dx * 0.24, y1 - dy * 0.24
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=C["purple"],
                                        lw=0.5 + w / 5, alpha=0.7), zorder=2)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"连通性网络 (TCI={TCI:.0f}%)\n红=净输出 蓝=净接收 边宽∝溢出强度", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "connectedness_network.png"), dpi=130)
    plt.close(fig)

# ============================================================
# 图3:滚动 TCI 时序(危机期飙升)
# ============================================================
def fig_rolling_tci():
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(tci_series))
    ax.plot(x, tci_series, color=C["blue"], lw=1.3)
    ax.axvspan(crisis_start - WW, crisis_end - WW, color=C["red"], alpha=0.15,
               label="危机窗口")
    ax.axhline(tci_series.mean(), color="gray", ls="--", lw=1, label=f"全样本均值 {tci_series.mean():.0f}%")
    ax.set_xlabel("交易日(滚动窗口 250 日)", fontsize=11)
    ax.set_ylabel("总连通度 TCI %", fontsize=11)
    ax.set_title("滚动总连通度:危机期风险传染显著飙升", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "rolling_tci.png"), dpi=130)
    plt.close(fig)

fig_heatmap()
fig_network()
fig_rolling_tci()
print("\n✅ 已生成 3 张配图:", os.listdir(D))
