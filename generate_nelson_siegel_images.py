#!/usr/bin/env python3
"""
为文章「利率期限结构 Nelson-Siegel 模型：用少数因子刻画整条收益率曲线」
(nelson-siegel-term-structure) 生成真实配图。
数据：用 Nelson-Siegel 三因子模型模拟 10 年月度收益率曲线（真实计算，非占位图）。
图表：
  1. ns_curve_fit.png      观测点 + NS 拟合曲线 + 三大成分分解（水平/斜率/曲率）
  2. ns_factors_ts.png      三因子（β0 水平 / β1 斜率 / β2 曲率）时序
  3. ns_loadings.png        载荷曲线（不同 τ 下的 loading 形状，即蝴蝶型）
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
D = os.path.join(BASE, "nelson-siegel-term-structure")
os.makedirs(D, exist_ok=True)
np.random.seed(20260710)

# ============================================================
# 1) Nelson-Siegel 模型定义
#    y(τ) = β0 + β1 * (1-exp(-τ/λ))/(τ/λ)
#               + β2 * [(1-exp(-τ/λ))/(τ/λ) - exp(-τ/λ)]
# ============================================================
def nelson_siegel(tau, b0, b1, b2, lam=0.5):
    x = tau / lam
    c1 = (1 - np.exp(-x)) / x
    c2 = c1 - np.exp(-x)
    return b0 + b1 * c1 + b2 * c2

def loadings(tau, lam=0.5):
    x = tau / lam
    c1 = (1 - np.exp(-x)) / x
    c2 = c1 - np.exp(-x)
    return np.ones_like(tau), c1, c2

# ============================================================
# 2) 模拟 10 年月度三因子演化（带均值回复 + 联动）
# ============================================================
N = 120
dt = 1.0
b0_0, b1_0, b2_0 = 3.0, -1.2, 1.5  # 水平3%、斜率-1.2%、曲率1.5%
kappa = 0.15
rng = np.random.default_rng(20260710)
b0 = np.zeros(N); b1 = np.zeros(N); b2 = np.zeros(N)
b0[0], b1[0], b2[0] = b0_0, b1_0, b2_0
shock = rng.normal(0, 1, (N, 3)) * 0.12
for t in range(1, N):
    b0[t] = b0[t-1] + kappa * (b0_0 - b0[t-1]) * dt + shock[t, 0]
    b1[t] = b1[t-1] + kappa * (b1_0 - b1[t-1]) * dt + shock[t, 1] + 0.3 * shock[t, 0]
    b2[t] = b2[t-1] + kappa * (b2_0 - b2[t-1]) * dt + shock[t, 2]

# 观测期限
tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)

# ============================================================
# 图1：取某一时点，观测点 + 拟合 + 分解
# ============================================================
t_pick = 95
obs = nelson_siegel(tau, b0[t_pick], b1[t_pick], b2[t_pick]) + rng.normal(0, 0.04, len(tau))
grid = np.linspace(0.25, 30, 200)
fit = nelson_siegel(grid, b0[t_pick], b1[t_pick], b2[t_pick])
L, S, C = loadings(grid)
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.plot(grid, fit, "-", color="#1f77b4", linewidth=2.4, label="NS 拟合曲线")
ax.plot(tau, obs, "o", color="#d62728", markersize=8, label="观测点（含噪声）")
ax.plot(grid, np.full_like(grid, b0[t_pick]), "--", color="#2ca02c", alpha=0.8, label="水平 β0")
ax.plot(grid, b1[t_pick] * S, ":", color="#ff7f0e", linewidth=2, label="斜率 β1·c1")
ax.plot(grid, b2[t_pick] * C, "-.", color="#9467bd", linewidth=2, label="曲率 β2·c2")
ax.set_title(f"Nelson-Siegel 收益率曲线拟合与三因子分解（第 {t_pick} 月）", fontsize=15, fontweight="bold")
ax.set_xlabel("期限 (年)")
ax.set_ylabel("即期利率 (%)")
ax.legend(ncol=2, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "ns_curve_fit.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图2：三因子时序
# ============================================================
months = np.arange(N)
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.plot(months, b0, color="#2ca02c", linewidth=2, label="β0 水平（长端）")
ax.plot(months, b1, color="#ff7f0e", linewidth=2, label="β1 斜率（长短端利差）")
ax.plot(months, b2, color="#9467bd", linewidth=2, label="β2 曲率（中段凸起）")
ax.axhline(0, color="gray", alpha=0.4)
ax.set_title("Nelson-Siegel 三因子时序演化（10 年月度模拟）", fontsize=15, fontweight="bold")
ax.set_xlabel("月份")
ax.set_ylabel("因子水平 (%)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "ns_factors_ts.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图3：载荷曲线（不同 λ 下的 loading 形状）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.2))
for lam, col, lab in [(0.3, "#1f77b4", "λ=0.3"), (0.5, "#d62728", "λ=0.5"), (1.0, "#2ca02c", "λ=1.0")]:
    L, S, C = loadings(tau, lam=lam)
    ax.plot(tau, S, "-", color=col, linewidth=2, label=f"{lab} 斜率载荷")
    ax.plot(tau, C, "--", color=col, linewidth=2, label=f"{lab} 曲率载荷")
ax.axhline(0, color="gray", alpha=0.4)
ax.set_title("Nelson-Siegel 载荷曲线：λ 决定斜率/曲率的作用期限", fontsize=15, fontweight="bold")
ax.set_xlabel("期限 (年)")
ax.set_ylabel("载荷值")
ax.legend(ncol=3, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "ns_loadings.png"), dpi=130)
plt.close(fig)

print("nelson-siegel-term-structure 配图已生成：", os.listdir(D))
