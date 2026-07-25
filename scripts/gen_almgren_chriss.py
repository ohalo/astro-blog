#!/usr/bin/env python3
"""Almgren-Chriss 最优执行文章配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/almgren-chriss-optimal-execution"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 参数 ----------
X = 1_000_000        # 总股数
T = 1.0              # 交易日（1天）
N = 60               # 60 个时间片（每片约 4 分钟）
tau = T / N
sigma = 0.02         # 日波动 2%（价格 50 元 -> 1 元）
S0 = 50.0
sigma_abs = sigma * S0          # 绝对价格波动
eta = 2.5e-7         # 临时冲击系数（元/股/(股/时间)）
gamma_perm = 2.5e-8  # 永久冲击系数

def ac_trajectory(lam):
    """给定风险厌恶 lambda，返回持仓轨迹 x_k (k=0..N)"""
    if lam <= 0:
        return np.linspace(X, 0, N + 1)
    kappa_tilde2 = lam * sigma_abs**2 / (eta * (1 - gamma_perm * tau / (2 * eta)))
    kappa = np.arccosh(kappa_tilde2 * tau**2 / 2 + 1) / tau
    k = np.arange(N + 1)
    return X * np.sinh(kappa * (T - k * tau)) / np.sinh(kappa * T)

def cost_stats(x):
    """期望冲击成本 + 方差（绝对金额，元）"""
    n = -np.diff(x)  # 每片卖出量
    v = n / tau
    temp_cost = np.sum(eta * v * n)               # 临时冲击
    perm_cost = 0.5 * gamma_perm * X**2            # 永久冲击（与轨迹无关近似）
    var = sigma_abs**2 * tau * np.sum(x[1:]**2)
    return temp_cost + perm_cost, var

# ---------- 图1：不同风险厌恶下的最优轨迹 ----------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
lams = [0, 5e-7, 2e-6, 1e-5]
labels = ["λ=0（等价 TWAP）", "λ=5e-7（温和）", "λ=2e-6（中等）", "λ=1e-5（激进前置）"]
colors = ["#888888", "#1f77b4", "#ff7f0e", "#d62728"]
tgrid = np.linspace(0, T, N + 1)
for lam, lab, c in zip(lams, labels, colors):
    x = ac_trajectory(lam)
    ax.plot(tgrid * 390, x / 1e4, label=lab, color=c, lw=2.2)
ax.set_xlabel("交易日内时间（分钟）")
ax.set_ylabel("剩余持仓（万股）")
ax.set_title("Almgren-Chriss 最优清仓轨迹：风险厌恶越高，卖得越前置")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/trajectories.png")
plt.close(fig)

# ---------- 图2：有效前沿（成本 vs 风险） ----------
lam_grid = np.concatenate([[0], np.logspace(-8, -4.5, 60)])
costs, stds = [], []
for lam in lam_grid:
    x = ac_trajectory(lam)
    c, v = cost_stats(x)
    costs.append(c / 1e4)         # 万元
    stds.append(np.sqrt(v) / 1e4)
costs = np.array(costs); stds = np.array(stds)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
ax.plot(stds, costs, color="#1f77b4", lw=2.2)
# 标注几个点
for lam, name in [(0, "TWAP（λ=0）"), (2e-6, "λ=2e-6"), (1e-5, "λ=1e-5")]:
    x = ac_trajectory(lam)
    c, v = cost_stats(x)
    ax.scatter(np.sqrt(v)/1e4, c/1e4, zorder=5, s=60)
    ax.annotate(name, (np.sqrt(v)/1e4, c/1e4), textcoords="offset points",
                xytext=(10, 6), fontsize=10)
ax.set_xlabel("执行收入标准差（万元）")
ax.set_ylabel("期望冲击成本（万元）")
ax.set_title("执行的有效前沿：省成本必须承担时间风险")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/efficient-frontier.png")
plt.close(fig)

# ---------- 图3：蒙特卡洛——TWAP vs AC 实现成本分布 ----------
def simulate(lam, n_sim=20000):
    x = ac_trajectory(lam)
    n = -np.diff(x)
    v = n / tau
    # 价格路径：算术随机游走 + 永久冲击
    dW = rng.normal(0, sigma_abs * np.sqrt(tau), size=(n_sim, N))
    perm = gamma_perm * np.cumsum(np.tile(v * tau, (n_sim, 1)), axis=1)
    S = S0 + np.cumsum(dW, axis=1) - perm
    exec_price = S - eta * v          # 临时冲击
    revenue = exec_price @ n
    shortfall = X * S0 - revenue       # 实施缺口（元）
    return shortfall / 1e4             # 万元

sf_twap = simulate(0)
sf_ac = simulate(2e-6)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
bins = np.linspace(min(sf_twap.min(), sf_ac.min()), max(sf_twap.max(), sf_ac.max()), 80)
ax.hist(sf_twap, bins=bins, alpha=0.55, label=f"TWAP：均值 {sf_twap.mean():.1f}，标准差 {sf_twap.std():.1f}", color="#888888")
ax.hist(sf_ac, bins=bins, alpha=0.55, label=f"AC(λ=2e-6)：均值 {sf_ac.mean():.1f}，标准差 {sf_ac.std():.1f}", color="#ff7f0e")
ax.set_xlabel("实施缺口（万元，越小越好）")
ax.set_ylabel("模拟次数")
ax.set_title("2 万次蒙特卡洛：AC 用少量期望成本换掉大量不确定性")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/monte-carlo.png")
plt.close(fig)

# ---------- 图4：半衰期 vs lambda ----------
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
lam_grid2 = np.logspace(-7.5, -4.5, 50)
half_life = []
for lam in lam_grid2:
    x = ac_trajectory(lam)
    idx = np.argmax(x <= X / 2)
    half_life.append(idx * tau * 390)
ax.semilogx(lam_grid2, half_life, lw=2.2, color="#2ca02c")
ax.set_xlabel("风险厌恶系数 λ（对数轴）")
ax.set_ylabel("清掉一半仓位所需时间（分钟）")
ax.set_title("交易半衰期随风险厌恶单调下降")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/half-life.png")
plt.close(fig)

print("=== 关键数字 ===")
print(f"TWAP: 期望缺口 {sf_twap.mean():.1f} 万, 标准差 {sf_twap.std():.1f} 万, 95%VaR {np.percentile(sf_twap,95):.1f} 万")
print(f"AC λ=2e-6: 期望缺口 {sf_ac.mean():.1f} 万, 标准差 {sf_ac.std():.1f} 万, 95%VaR {np.percentile(sf_ac,95):.1f} 万")
x = ac_trajectory(2e-6)
print(f"AC λ=2e-6 前 1/4 时间卖出比例: {(X - x[15]) / X * 100:.1f}%")
c_t, v_t = cost_stats(ac_trajectory(0)); c_a, v_a = cost_stats(ac_trajectory(2e-6))
print(f"TWAP 期望成本 {c_t/1e4:.1f} 万 / AC 成本 {c_a/1e4:.1f} 万; TWAP std {np.sqrt(v_t)/1e4:.1f} / AC std {np.sqrt(v_a)/1e4:.1f}")
print(f"总市值: {X*S0/1e8:.2f} 亿元")
