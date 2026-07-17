#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「因子模拟组合：用多头空头组合把抽象因子变成可交易资产」生成真实配图与统计数字。

核心机制(基于 Huberman & Kandel 1987 / Fama-French factor-mimicking portfolio):
  抽象因子 f(例如一个没法直接交易的风格因子、宏观因子、文本情绪因子)本身不可交易。
  给定 N 个可交易资产的收益矩阵 R(TxN) 与因子序列 f(T), 因子模拟组合(FMP)是满足
  「在可交易资产空间内, 找一组权重 w 使其组合收益 w'R 与 f 相关性最高(= 把 f 投影到
  资产张成空间)」的组合。解析解为(施加 sum(w)=1 约束):
      w* = Σ^{-1} γ / (1' Σ^{-1} γ),   γ = Cov(R, f)
  性质:
    (1) FMP 收益 p = w*'R 与 f 的相关系数最大;
    (2) FMP 的 Sharpe 不可能超过因子自身 Sharpe(等号成立当且仅当 f 完全落在资产张成空间内);
        二者之差 = 该因子的「不可交易成分 / 实现损耗」, 是 APT 框架里给因子定价的关键边界。
  全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib), 无占位符。

图片:
  fmp_weights.png       —— 因子模拟组合的多头/空头权重(长-短组合)
  fmp_tracking.png      —— FMP 收益 vs 原始因子收益(时间序列追踪)
  fmp_scatter.png       —— FMP 收益 vs 因子收益散点 + 回归线 + 相关系数
  fmp_sharpe_gap.png    —— 因子理论 Sharpe vs FMP 实现 Sharpe(揭示可交易损耗)
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
D = os.path.join(BASE, "factor-mimicking-portfolio")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

C_BLUE = "#2c7fb8"
C_RED = "#d7301f"
C_GREEN = "#1a9850"
C_GREY = "#7f7f7f"
C_GRID = "#dddddd"

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260718)

# ================= 1. 真实因子结构: 3 个底层因子驱动 12 个资产 =================
N = 12               # 可交易资产数
T = 240              # 20 年月度观测
# 三个底层因子(月度收益): 市场 / 规模 / 价值(本文目标因子)
fm = rng.normal(0.0080, 0.040, T)   # 市场
fs = rng.normal(0.0040, 0.030, T)   # 规模
fv = rng.normal(0.0060, 0.025, T)   # 价值(目标因子)
F = np.column_stack([fm, fs, fv])   # Tx3

# 每个资产对三个因子的载荷(结构化随机)
beta = np.zeros((N, 3))
for i in range(N):
    beta[i, 0] = rng.uniform(0.6, 1.3)          # 市场 beta
    beta[i, 1] = rng.uniform(-0.5, 0.8)         # 规模暴露
    beta[i, 2] = rng.uniform(-0.3, 0.9)         # 价值暴露(目标因子载荷, 有正有负)

# 特质收益
eig = rng.uniform(0.02, 0.05, N)
L = rng.normal(0, 1, (T, N)) * eig              # 特质噪声(对角协方差, 简化)
R = F @ beta.T + L                              # TxN 资产收益

asset_names = [f"资产{i+1:02d}" for i in range(N)]
mu_R = R.mean(axis=0)

# ================= 2. 构造目标因子的模拟组合 =================
Sigma = np.cov(R, rowvar=False)                 # NxN 资产协方差
# γ = Cov(R, fv)
C = np.cov(R, fv, rowvar=False)                 # (N+1)x(N+1)
gamma = C[:N, N]                                # N 向量 Cov(R, fv)

# 解析解: w* = Σ^{-1} γ / (1' Σ^{-1} γ)
Sigma_inv = np.linalg.inv(Sigma)
w_raw = Sigma_inv @ gamma
w = w_raw / w_raw.sum()                         # 归一使 sum(w)=1

P = R @ w                                       # 因子模拟组合收益序列

# 追踪误差 / 相关性
corr = np.corrcoef(P, fv)[0, 1]
track_err = np.std(P - fv) * np.sqrt(12)        # 年化追踪误差(用 fv 尺度近似单位风险)
R2 = corr ** 2

# 权重结构(多头/空头)
long_w = w[w > 0]
short_w = w[w < 0]

log("===== 因子模拟组合 (Factor Mimicking Portfolio) =====")
log(f"资产数 N = {N}, 观测 T = {T} (月度)")
log(f"目标因子 = 价值因子 fv, 自身年化均值 {fv.mean()*12:.2%}, 年化波动 {fv.std()*np.sqrt(12):.2%}")
log(f"模拟组合权重 sum(w) = {w.sum():.6f}")
log(f"多头权重之和 = {long_w.sum():.4f}, 空头权重之和 = {short_w.sum():.4f}")
log(f"最大多头 = {w.max():.4f} ({asset_names[int(np.argmax(w))]})")
log(f"最大空头 = {w.min():.4f} ({asset_names[int(np.argmin(w))]})")
log(f"FMP 与因子相关系数 corr = {corr:.4f}, R² = {R2:.4f}")
log(f"年化追踪误差 = {track_err:.2%}")

# ================= 3. Sharpe 边界: 因子理论 vs FMP 实现 =================
def sharpe(x):
    return x.mean() / x.std() * np.sqrt(12)

sr_factor = sharpe(fv)        # 若能直接交易因子本身的 Sharpe(理论上界)
sr_fmp = sharpe(P)            # FMP 实际实现的 Sharpe
sr_gap = sr_factor - sr_fmp
log(f"因子自身 Sharpe(理论上界) = {sr_factor:.3f}")
log(f"FMP 实现 Sharpe = {sr_fmp:.3f}")
log(f"Sharpe 损耗 = {sr_gap:.3f} ({(sr_gap/sr_factor):.1%})")

# 收敛检验: w 确实最大化相关 -> 对比朴素等权组合
w_eq = np.ones(N) / N
P_eq = R @ w_eq
corr_eq = np.corrcoef(P_eq, fv)[0, 1]
log(f"对比: 等权组合与因子 corr = {corr_eq:.4f} (FMP {corr:.4f} 更高)")

# ================= 4. 画图 =================
# 图1: 多头/空头权重
fig, ax = plt.subplots(figsize=(8.2, 4.4))
x = np.arange(N)
colors = [C_GREEN if v >= 0 else C_RED for v in w]
ax.bar(x, w, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(asset_names, fontsize=8, rotation=45)
ax.set_ylabel("权重 w")
ax.set_title("因子模拟组合：多头/空头权重（把抽象因子投影到可交易资产）", fontsize=11)
for i, v in enumerate(w):
    ax.text(i, v + (0.02 if v >= 0 else -0.03), f"{v:.2f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=7)
ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "fmp_weights.png")); plt.close(fig)

# 图2: FMP 收益 vs 因子收益(时间序列)
fig, ax = plt.subplots(figsize=(8.6, 4.0))
ax.plot(P, color=C_BLUE, lw=1.3, label="因子模拟组合 FMP")
ax.plot(fv, color=C_RED, lw=1.0, alpha=0.8, label="原始因子 fv")
ax.set_xlabel("时间 (月)")
ax.set_ylabel("月度收益")
ax.set_title(f"FMP 收益 vs 原始因子收益 (corr = {corr:.3f})", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "fmp_tracking.png")); plt.close(fig)

# 图3: 散点 + 回归
fig, ax = plt.subplots(figsize=(6.6, 5.2))
ax.scatter(fv, P, s=14, color=C_BLUE, alpha=0.55)
# OLS 拟合
A = np.vstack([fv, np.ones(T)]).T
k, b = np.linalg.lstsq(A, P, rcond=None)[0]
xs = np.linspace(fv.min(), fv.max(), 50)
ax.plot(xs, k * xs + b, color=C_RED, lw=1.5, label=f"拟合: slope={k:.2f}")
ax.set_xlabel("原始因子收益 fv")
ax.set_ylabel("FMP 收益 p")
ax.set_title(f"FMP vs 因子散点 (corr={corr:.3f}, R²={R2:.3f})", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "fmp_scatter.png")); plt.close(fig)

# 图4: Sharpe 边界
fig, ax = plt.subplots(figsize=(6.4, 4.4))
bars = ax.bar(["因子理论 Sharpe\n(上界)", "FMP 实现 Sharpe", "等权组合 Sharpe"],
              [sr_factor, sr_fmp, sharpe(P_eq)],
              color=[C_GREY, C_GREEN, C_BLUE])
for bar, val in zip(bars, [sr_factor, sr_fmp, sharpe(P_eq)]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.2f}",
            ha="center", fontsize=10)
ax.set_ylabel("年化 Sharpe")
ax.set_title("因子 Sharpe 上界 vs FMP 实现 Sharpe（损耗 = 不可交易成分）", fontsize=10.5)
ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "fmp_sharpe_gap.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(sorted(os.listdir(D))))
