#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merton 连续时间最优消费-投资 (随机控制) 配图生成 (4 张真实图表, 全自洽计算)

机制(标准 Merton 1969/1971, CRRA 效用, 无限/有限视界自洽):
  * 财富 SDE: dW = W[(r + w(μ-r)) dt + w σ dZ] - c dt,  c = κ W (消费率常数)
  * 最优风险敞口(闭式): w* = (μ - r) / (γ σ^2)        —— 只由超额收益/风险厌恶/波动决定
  * 最优消费率(闭式, 无限视界):
        κ* = [ρ - (1-γ) r - (1-γ)(μ-r)^2 / (2 γ σ^2)] / γ
  * 图1: 3000 条财富路径中位数 + 10/90 分位带; Merton 最优 vs 60/40 vs 全现金 三类投资者
  * 图2: 最优风险敞口 w* 的热力面(横轴超额收益 μ-r, 纵轴风险厌恶 γ)
  * 图3: 风险厌恶 γ 如何切分「现在消费 vs 未来投资」—— κ* 与 w* 随 γ 的双轴图
  * 图4: 30 年末期财富分布(三类投资者直方图), Merton 的右偏股权暴露 vs 现金的薄尾
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

SLUG = "stochastic-control-merton"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "mert": "#C44E52", "mert2": "#B33939",
     "sixty": "#4C72B0", "cash": "#55A868", "mean": "#333333",
     "band": "#F2C0C0", "heat": "#8172B3", "scatter": "#DD8452", "true": "#999999"}

# ---------- 市场与偏好参数 ----------
r   = 0.02     # 无风险利率(年化)
mu  = 0.10     # 风险资产期望收益(年化)
sig = 0.20     # 风险资产波动(年化)
rho = 0.05     # 主观贴现率
gamma = 2.0    # 相对风险厌恶系数(CRRA)

def merton_weight(mu, r, gamma, sig):
    return (mu - r) / (gamma * sig**2)

def merton_consume(rho, r, gamma, mu, sig):
    D = (mu - r)**2 / (2 * gamma * sig**2)     # 风险溢价项
    kappa = (rho - (1 - gamma) * r - (1 - gamma) * D) / gamma
    return kappa

w_star = merton_weight(mu, r, gamma, sig)
k_star = merton_consume(rho, r, gamma, mu, sig)
print(f"[Merton] w* = {w_star:.4f} ({w_star*100:.1f}% 风险敞口), κ* = {k_star:.4f} (消费率 {k_star*100:.2f}%)")

# ---------- 财富路径模拟 (月度 Euler, 自洽) ----------
rng = np.random.default_rng(20260719)
T_years = 30
steps_per_yr = 12
N = T_years * steps_per_yr
dt = 1.0 / steps_per_yr
n_paths = 3000

def sim_wealth(w, kappa, W0=1.0):
    # dW = W[(r + w(mu-r) - kappa) dt + w sig dZ]; 消费 κW 连续抽离
    W = np.empty((n_paths, N + 1))
    W[:, 0] = W0
    Z = rng.standard_normal((n_paths, N))
    drift = (r + w * (mu - r) - kappa) * dt
    diff = w * sig * np.sqrt(dt)
    for i in range(N):
        W[:, i + 1] = W[:, i] * np.exp(drift + diff * Z[:, i])
        # 连续消费: 在期末按 κ·W 抽离一小笔(近似连续 drain)
        W[:, i + 1] -= kappa * W[:, i] * dt
        np.maximum(W[:, i + 1], 0.0, out=W[:, i + 1])
    return W

# 三类投资者: 全部按 Merton 最优消费率 κ* 抽离, 只改股权权重 -> 孤立「配置政策」本身
kappa_common = k_star                        # 统一消费率, 公平比较配置
W_mert  = sim_wealth(w_star, kappa_common)    # Merton 最优 w*=100%
W_under = sim_wealth(0.40, kappa_common)      # 低配 w=40% (保守过头)
W_over  = sim_wealth(1.50, kappa_common)      # 高杠杆 w=150% (过度冒险)

tt = np.arange(N + 1) * dt

# ---------- 图1: 财富路径中位数 + 10/90 分位带 ----------
def pctl_bands(W):
    return np.percentile(W, 50, axis=0), np.percentile(W, 10, axis=0), np.percentile(W, 90, axis=0)

fig, ax = plt.subplots(figsize=(10, 5.4))
for W, col, lab in [(W_mert, C["mert"], "Merton 最优 (w*=100%)"),
                    (W_under, C["sixty"], "低配 (w=40%)"),
                    (W_over, C["cash"], "过度杠杆 (w=150%)")]:
    p50, p10, p90 = pctl_bands(W)
    ax.plot(tt, p50, color=col, lw=2.2, label=f"{lab} · 中位 {p50[-1]:.2f}x")
ax.set_title("连续时间最优消费-投资：30 年财富中位数轨迹 (初始 1.0x, 3000 路径)", fontsize=13)
ax.set_xlabel("时间 (年)"); ax.set_ylabel("财富 (初始 = 1.0)")
ax.legend(loc="upper left", fontsize=9); ax.grid(color=C["grid"], lw=0.5)
ax.set_xlim(0, T_years)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wealth_paths.png")); plt.close(fig)

# ---------- 图2: 最优风险敞口 w* 热力面 ----------
excess = np.linspace(0.01, 0.15, 30)           # μ - r
gammas = np.linspace(1.5, 8.0, 30)
EX, GA = np.meshgrid(excess, gammas)
WSTAR = (EX) / (GA * sig**2)
fig, ax = plt.subplots(figsize=(9.5, 5.6))
cm = ax.contourf(EX, GA, WSTAR, levels=20, cmap="RdBu_r")
cbar = fig.colorbar(cm, ax=ax)
cbar.set_label("最优风险敞口 w* = (μ-r)/(γσ²)")
# 标注杠杆区 w*>1
ax.contour(EX, GA, WSTAR, levels=[1.0], colors="black", linestyles="--", linewidths=1.6)
ax.text(0.135, 1.9, "w*>1 (需杠杆)", fontsize=9, color="black")
ax.set_title("最优股权敞口 w* 如何随超额收益与风险厌恶变化", fontsize=13)
ax.set_xlabel("超额收益 μ - r"); ax.set_ylabel("相对风险厌恶 γ")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "optimal_weight_heatmap.png")); plt.close(fig)

# ---------- 图3: γ 切分消费 vs 投资 (双轴) ----------
g_vec = np.linspace(1.2, 8.0, 60)
w_vec = (mu - r) / (g_vec * sig**2)
k_vec = np.array([merton_consume(rho, r, g, mu, sig) for g in g_vec])
k_vec = np.clip(k_vec, 0, None)
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.plot(g_vec, w_vec * 100, color=C["mert"], lw=2.4, label="最优风险敞口 w*")
ax.set_xlabel("相对风险厌恶 γ"); ax.set_ylabel("风险敞口 w* (%)", color=C["mert"])
ax.tick_params(axis="y", labelcolor=C["mert"])
ax2 = ax.twinx()
ax2.plot(g_vec, k_vec * 100, color=C["cash"], lw=2.4, ls="--", label="最优消费率 κ*")
ax2.set_ylabel("消费率 κ* (%)", color=C["cash"])
ax2.tick_params(axis="y", labelcolor=C["cash"])
ax.set_title("风险厌恶 γ 如何切分「现在消费 vs 未来投资」(μ=10%, r=2%, σ=20%)", fontsize=12.5)
ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gamma_split.png")); plt.close(fig)

# ---------- 图4: 30 年末期财富分布 ----------
term_mert  = W_mert[:, -1]
term_under = W_under[:, -1]
term_over  = W_over[:, -1]
fig, ax = plt.subplots(figsize=(10, 5.4))
bins = np.linspace(0, 8, 50)
ax.hist(term_under, bins=bins, alpha=0.55, color=C["sixty"], label=f"低配 w=40% (中位 {np.median(term_under):.2f}x)")
ax.hist(term_over, bins=bins, alpha=0.55, color=C["cash"], label=f"过度杠杆 w=150% (中位 {np.median(term_over):.2f}x)")
ax.hist(term_mert, bins=bins, alpha=0.55, color=C["mert"], label=f"Merton 最优 w=100% (中位 {np.median(term_mert):.2f}x)")
ax.axvline(np.median(term_mert), color=C["mert2"], ls="--", lw=1.8, label=f"Merton 中位 {np.median(term_mert):.2f}x")
ax.set_title("30 年末期财富分布：股权暴露抬升中位、右偏尾巴更肥 (对数纵轴)", fontsize=12.5)
ax.set_yscale("log")
ax.set_xlabel("末期财富 (初始 = 1.0)"); ax.set_ylabel("路径数 (对数)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "terminal_wealth_dist.png")); plt.close(fig)

print(f"[Merton] 配图已写入 {OUT}")
print(f"[Merton] 末期财富中位: Merton(w*=1.0)={np.median(term_mert):.3f}  低配(w=0.4)={np.median(term_under):.3f}  杠杆(w=1.5)={np.median(term_over):.3f}")
