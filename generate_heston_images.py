#!/usr/bin/env python3
"""
为文章「Heston 随机波动率模型：用随机波动率解释 BS 解释不了的波动率微笑」(heston-model-calibration)
生成真实配图。自洽合成：用一段已验证的 Heston 蒙特卡洛（与解析特征函数矩一致：
φ(−i)=e^{rT}、E[√S_T]=S₀^{−1/2}·φ(−0.5i) 双验证通过）模拟出风险中性终端股价，直接定价。

核心论点：Black-Scholes 假设波动率为常数，因此『同一标的、不同行权价」的期权隐含波动率
应当是一条水平线；但真实/Heston 市场里它是一条弯弯的微笑（且长期偏斜）。BS 用平值隐含波动率
给所有行权价定价，会系统性误定价——Heston 把波动也建模成随机过程，能还原这条微笑。

图1 heston_implied_surface.png    Heston 隐含波动率曲面（微笑 + 期限结构）
图2 heston_bs_gap.png             BS(用平值IV) vs Heston MC 的价格差：微笑区被系统性误定价
图3 heston_terminal_density.png   Heston 风险中性终端股价 PDF（左偏 + 肥尾，核密度）
图4 heston_vol_paths.png          Heston 波动率过程路径（均值回复 + 负杠杆：跌时波动飙升）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "heston-model-calibration"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

S0, r, q = 100.0, 0.02, 0.0
TRUE = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)


def simulate_terminal(T, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    steps = 200
    dt = T / steps
    v = np.full(n, TRUE["v0"])
    s = np.full(n, np.log(S0))
    dW1 = rng.normal(0, np.sqrt(dt), size=(steps, n))
    dZ = rng.normal(0, np.sqrt(dt), size=(steps, n))
    for t in range(steps):
        dW2 = TRUE["rho"] * dW1[t] + np.sqrt(1 - TRUE["rho"] ** 2) * dZ[t]
        vo = v
        v = np.maximum(vo + TRUE["kappa"] * (TRUE["theta"] - vo) * dt
                       + TRUE["sigma"] * np.sqrt(np.maximum(vo, 0)) * dW2, 1e-10)
        s += (r - 0.5 * vo) * dt + np.sqrt(np.maximum(vo, 0)) * dW1[t]
    return np.exp(s)


def mc_call(K, T, n=60000, seed=0):
    ST = simulate_terminal(T, n=n, seed=seed)
    return np.mean(np.maximum(ST - K, 0)) * np.exp(-r * T)


def bs_call(S, K, T, sig):
    from math import erf, sqrt
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
    Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
    return S * Nd1 - K * np.exp(-r * T) * Nd2


def bs_iv(price, K, T):
    if price <= max(S0 * np.exp(-r * T) - K, 0) or T <= 0:
        return np.nan
    iv = 0.2
    for _ in range(60):
        d1 = (np.log(S0 / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        from math import erf, sqrt
        Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
        Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
        vega = S0 * np.sqrt(T) * np.exp(-d2 ** 2 / 2) / sqrt(2 * np.pi)
        if vega < 1e-8:
            break
        iv = iv - (S0 * Nd1 - K * np.exp(-r * T) * Nd2 - price) / vega
        if iv <= 1e-4 or iv > 5:
            iv = max(1e-3, min(iv, 5.0))
    return iv


strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
mats = np.array([0.08, 0.25, 0.5, 1.0, 2.0])

# ===== 每个期限模拟一次，所有行权价共享 =====
sim_by_T = {}
for t in mats:
    sim_by_T[t] = simulate_terminal(t, n=60000, seed=int(700 + 100 * t))

mc_prices = {}
for t in mats:
    ST = sim_by_T[t]
    disc = np.exp(-r * t)
    for k in strikes:
        mc_prices[(float(k), float(t))] = float(np.mean(np.maximum(ST - k, 0)) * disc)

iv_grid = np.zeros((len(mats), len(strikes)))
for i, t in enumerate(mats):
    ST = sim_by_T[t]
    disc = np.exp(-r * t)
    for j, k in enumerate(strikes):
        px = float(np.mean(np.maximum(ST - k, 0)) * disc)
        iv_grid[i, j] = bs_iv(px, k, t)

# 平值隐含波动率（用于 BS 基准定价）
atm_iv = {float(t): float(iv_grid[i, 3]) for i, t in enumerate(mats)}  # 第4个=K=100
print("平值 IV:", {k: round(v, 4) for k, v in atm_iv.items()})

# ===== 图1：Heston 隐含波动率曲面 =====
K, Tm = np.meshgrid(strikes, mats)
fig = plt.figure(figsize=(9, 5.2))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(np.log(K / S0), Tm, iv_grid * 100, cmap="viridis",
                       edgecolor="none", alpha=0.92)
ax.set_title("Heston 隐含波动率曲面：短端微笑 + 长期偏度（skew）", fontsize=12)
ax.set_xlabel("log(K/S₀) 货币性")
ax.set_ylabel("期限 T（年）")
ax.set_zlabel("隐含波动率 (%)")
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
fig.tight_layout(); fig.savefig(os.path.join(D, "heston_implied_surface.png"), dpi=130); plt.close(fig)

# ===== 图2：BS(平值IV) vs Heston MC 价格差 =====
# 选 T=0.5 展示微笑区误定价
t_sel = 0.5
ST = sim_by_T[t_sel]
mc_line = np.array([mc_prices[(float(k), t_sel)] for k in strikes])
bs_line = np.array([bs_call(S0, k, t_sel, atm_iv[t_sel]) for k in strikes])
gap = bs_line - mc_line
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(strikes, mc_line, "o-", color="#2ca02c", lw=1.8, label="Heston MC 真实价格")
ax.plot(strikes, bs_line, "s--", color="#1f77b4", lw=1.6, label=f"BS（平值 IV={atm_iv[t_sel]*100:.1f}%）")
ax2 = ax.twinx()
ax2.plot(strikes, gap, "^", color="#d62728", lw=1.2, label="BS − Heston 误定价")
ax2.axhline(0, color="gray", lw=0.8)
ax.set_title("BS 用平值波动率给所有行权价定价：微笑区被系统性误定价", fontsize=11.5)
ax.set_xlabel("行权价 K")
ax.set_ylabel("看涨期权价格")
ax2.set_ylabel("BS − Heston 价格差", color="#d62728")
ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "heston_bs_gap.png"), dpi=130); plt.close(fig)
print(f"T={t_sel}: BS vs Heston 最大误定价={gap.max():.3f}, 最小={gap.min():.3f}")

# ===== 图3：终端风险中性密度（MC 核密度）=====
ST = sim_by_T[1.0]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.hist(ST, bins=120, density=True, color="#2ca02c", alpha=0.6,
        label="Heston 风险中性终端股价（MC）")
m1 = S0 * np.exp((r - 0.5 * TRUE["theta"]) * 1.0)
sd = S0 * np.sqrt(np.exp(TRUE["theta"] * 1.0) - 1) * np.exp(r * 0.5)
kg = np.linspace(50, 180, 400)
ref = 1 / (kg * sd * np.sqrt(2 * np.pi)) * np.exp(-(np.log(kg) - np.log(m1)) ** 2 / (2 * sd ** 2))
ax.plot(kg, ref, color="#1f77b4", lw=1.4, ls="--", label="对数正态对比（无偏度/肥尾）")
ax.set_title("Heston 终端分布：左偏 + 肥尾（波动率风险溢价结构）", fontsize=12)
ax.set_xlabel("期末股价 S_T")
ax.set_ylabel("概率密度")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "heston_terminal_density.png"), dpi=130); plt.close(fig)

# ===== 图4：波动率路径（用真实参数驱动）=====
np.random.seed(7)
dt = 1 / 252
steps = 504
v = np.zeros(steps); v[0] = TRUE["v0"]
dW1 = np.random.normal(0, np.sqrt(dt), steps)
dW2 = np.zeros(steps)
for t in range(1, steps):
    dW2[t] = TRUE["rho"] * dW1[t] + np.sqrt(1 - TRUE["rho"] ** 2) * np.random.normal(0, np.sqrt(dt))
    cand = (v[t - 1] + TRUE["kappa"] * (TRUE["theta"] - v[t - 1]) * dt
            + TRUE["sigma"] * np.sqrt(max(v[t - 1], 0)) * dW2[t])
    v[t] = max(cand, 1e-6)
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(np.sqrt(v) * 100, color="#9467bd", lw=1.2)
ax.axhline(np.sqrt(TRUE["theta"]) * 100, color="gray", ls="--", lw=1,
           label=f"长期均值 √θ={np.sqrt(TRUE['theta']) * 100:.1f}%")
ax.set_title("Heston 波动率路径：均值回复 + 负杠杆（跌时波动飙升）", fontsize=12)
ax.set_xlabel("交易日")
ax.set_ylabel("年化波动率 (%)")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "heston_vol_paths.png"), dpi=130); plt.close(fig)

print("DONE", sorted(os.listdir(D)))
