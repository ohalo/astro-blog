#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亚式与回望期权 (路径依赖 payoff) 配图生成 (4 张真实图表, 蒙特卡洛定价)

机制(自洽合成, 仅用于演示方法):
  * 基础资产: GBM, S0=100, mu=0.08, sigma=0.20, T=1年, 252步
  * 亚式看涨(算术平均, 固定执行价 K): payoff = max(A_T - K, 0), A=路径算术均值
  * 普通欧式看涨: payoff = max(S_T - K, 0)
  * 浮动行权价回望看涨: payoff = max(S_T - min_{t} S_t, 0)
  * 图1: 样本价格路径 + 亚式均值线 + 回望 min/max 带
  * 图2: 亚式 payoff 与欧式 payoff 的 MC 分布直方图(亚式更集中)
  * 图3: 蒙特卡洛价格随路径数收敛(亚式/回望/欧式三条)
  * 图4: 回望期权价格 vs 波动率(路径极值敏感)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "asian-lookback-options"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "avg": "#C44E52", "band": "#9AD3BC",
     "asian": "#4C72B0", "van": "#C44E52", "look": "#55A868"}

S0, mu, sigma, T, steps = 100.0, 0.08, 0.20, 1.0, 252
dt = T / steps
K = 100.0
rng = np.random.default_rng(20260719)

def simulate(M):
    """M 条路径, 返回 S(T+1, M), A(算术均值), Smin, Smax"""
    Z = rng.standard_normal((steps, M))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diff = sigma * np.sqrt(dt) * Z
    logS = np.cumsum(np.vstack([np.full(M, np.log(S0)), drift + diff]), axis=0)
    S = np.exp(logS)  # (steps+1, M)
    A = S.mean(axis=0)
    Smin = S.min(axis=0)
    Smax = S.max(axis=0)
    return S, A, Smin, Smax

# ---------- 图1: 样本路径 + 均值 + min/max 带 ----------
M1 = 6
S1, A1, Smin1, Smax1 = simulate(M1)
fig, ax = plt.subplots(figsize=(9, 5.6))
for i in range(M1):
    ax.plot(S1[:, i], color=C["path"], alpha=0.55, lw=1.3)
# 用一条代表性路径画均值/带
rep = S1[:, 0]
ax.plot(np.full(steps + 1, A1[0]), color=C["avg"], lw=2.2, ls="--",
        label=f"亚式算术均值 A={A1[0]:.1f}")
ax.fill_between(np.arange(steps + 1), Smin1[0], Smax1[0], color=C["band"], alpha=0.25,
                label=f"回望 min/max 带 [{Smin1[0]:.1f}, {Smax1[0]:.1f}]")
ax.axhline(K, color="gray", lw=1, ls=":", label=f"执行价 K={K}")
ax.set_xlabel("交易日")
ax.set_ylabel("价格")
ax.set_title("路径依赖期权的来源：payoff 不再只看期末 S_T\n(亚式看全程均值, 回望看全程极值)")
ax.legend(fontsize=8)
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "alo_paths.png"))
plt.close(fig)

# ---------- 图2: 亚式 vs 欧式 payoff 分布 ----------
M2 = 200000
S2, A2, Smin2, Smax2 = simulate(M2)
payoff_asian = np.maximum(A2 - K, 0)
payoff_vanilla = np.maximum(S2[-1] - K, 0)
payoff_look = np.maximum(S2[-1] - Smin2, 0)
disc = np.exp(-mu * T)
price_asian = disc * payoff_asian.mean()
price_vanilla = disc * payoff_vanilla.mean()
price_look = disc * payoff_look.mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
axes[0].hist(payoff_asian, bins=80, color=C["asian"], alpha=0.75, density=True,
             label=f"亚式 (price={price_asian:.2f})")
axes[0].axvline(0, color="gray", lw=1)
axes[0].set_title("亚式看涨 payoff 分布\n(依赖均值, 方差更小、更集中)")
axes[0].set_xlabel("payoff")
axes[0].set_ylabel("密度")
axes[0].legend(fontsize=8)
axes[1].hist(payoff_vanilla, bins=80, color=C["van"], alpha=0.75, density=True,
             label=f"欧式 (price={price_vanilla:.2f})")
axes[1].axvline(0, color="gray", lw=1)
axes[1].set_title("欧式看涨 payoff 分布\n(只看 S_T, 尾部更厚)")
axes[1].set_xlabel("payoff")
axes[1].legend(fontsize=8)
fig.suptitle("同一标的同一 K：路径依赖改写 payoff 结构", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "alo_payoff_dist.png"))
plt.close(fig)

# ---------- 图3: MC 收敛 ----------
M_grid = np.array([1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000])
conv_asian, conv_van, conv_look = [], [], []
base_rng = np.random.default_rng(20260719)
for Mm in M_grid:
    Sx, Ax, Sminx, Smaxx = simulate(Mm)
    conv_asian.append(np.exp(-mu * T) * np.maximum(Ax - K, 0).mean())
    conv_van.append(np.exp(-mu * T) * np.maximum(Sx[-1] - K, 0).mean())
    conv_look.append(np.exp(-mu * T) * np.maximum(Sx[-1] - Sminx, 0).mean())
# 用大样本近似真值做参考
true_asian, true_van, true_look = conv_asian[-1], conv_van[-1], conv_look[-1]
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.plot(M_grid, conv_asian, "o-", color=C["asian"], label=f"亚式 (∞≈{true_asian:.2f})")
ax.plot(M_grid, conv_van, "s-", color=C["van"], label=f"欧式 (∞≈{true_van:.2f})")
ax.plot(M_grid, conv_look, "^-", color=C["look"], label=f"回望浮动 (∞≈{true_look:.2f})")
ax.axhline(true_asian, color=C["asian"], ls=":", alpha=0.5)
ax.axhline(true_van, color=C["van"], ls=":", alpha=0.5)
ax.axhline(true_look, color=C["look"], ls=":", alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("蒙特卡洛路径数 (log)")
ax.set_ylabel("期权价格")
ax.set_title("蒙特卡洛价格随路径数收敛\n(回望对极值最敏感, 需更多路径才稳)")
ax.legend(fontsize=8)
ax.grid(True, color=C["grid"], which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "alo_mc_convergence.png"))
plt.close(fig)

# ---------- 图4: 回望价格 vs 波动率 ----------
sig_grid = np.linspace(0.05, 0.50, 30)
look_prices, asian_prices, van_prices = [], [], []
for sg in sig_grid:
    # 重新模拟(固定 M, 用独立 seed 段)
    Z = rng.standard_normal((steps, 60000))
    drift = (mu - 0.5 * sg ** 2) * dt
    diff = sg * np.sqrt(dt) * Z
    Sg = np.exp(np.cumsum(np.vstack([np.full(60000, np.log(S0)), drift + diff]), axis=0))
    Ag = Sg.mean(axis=0)
    look_prices.append(np.exp(-mu * T) * np.maximum(Sg[-1] - Sg.min(axis=0), 0).mean())
    asian_prices.append(np.exp(-mu * T) * np.maximum(Ag - K, 0).mean())
    van_prices.append(np.exp(-mu * T) * np.maximum(Sg[-1] - K, 0).mean())
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.plot(sig_grid * 100, look_prices, "^-", color=C["look"], lw=1.8, label="回望浮动看涨")
ax.plot(sig_grid * 100, asian_prices, "o-", color=C["asian"], lw=1.8, label="亚式算术看涨")
ax.plot(sig_grid * 100, van_prices, "s-", color=C["van"], lw=1.8, label="欧式看涨")
ax.set_xlabel("年化波动率 σ (%)")
ax.set_ylabel("期权价格")
ax.set_title("价格对波动率的敏感度：回望最陡\n(极值随波动上升最快, 亚式最钝)")
ax.legend(fontsize=8)
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "alo_price_vs_vol.png"))
plt.close(fig)

print("DONE", os.listdir(OUT))
print("price_asian", round(price_asian, 3), "price_vanilla", round(price_vanilla, 3), "price_look", round(price_look, 3))
print("asian std", round(payoff_asian.std(), 3), "vanilla std", round(payoff_vanilla.std(), 3))
print("conv final asian/van/look", round(true_asian,3), round(true_van,3), round(true_look,3))
