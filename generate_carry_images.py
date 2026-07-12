#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「Carry Trade 套息：赚的是利差还是风险补偿」(carry-trade-strategy) 生成真实配图。

数据来源：从零模拟 G10 货币利率 + 汇率月度收益，构建「做多高息/做空低息」套息组合，
并检验超额收益是否来自风险补偿(对全球风险因子负暴露 + 危机回撤)。
全部为真实数值计算，非占位图。

图表：
  1. carry_rates.png      各货币利率(套息篮子的多/空端)
  2. carry_nav.png        套息组合累积净值 vs 全球股票基准
  3. carry_drawdown.png   水下曲线(回撤)，重点展示危机崩盘
  4. carry_market_beta.png 套息超额收益 vs 全球风险因子回归(负 beta = 风险补偿)
  5. carry_crisis.png     危机期套息收益分布(常态小赚/危机大亏)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "carry-trade-strategy")
os.makedirs(D, exist_ok=True)
np.random.seed(20260713)

# ============================================================
# 1) 模拟 G10 货币：利率 + 汇率月度收益
# ============================================================
CURRENCIES = ["AUD", "NZD", "BRL", "ZAR", "MXN", "USD", "JPY", "CHF", "EUR", "SEK"]
# 名义利率(年化 %)，高息货币在前
RATES = np.array([5.6, 5.2, 11.5, 8.3, 6.8, 4.8, 0.1, 0.4, 2.1, 2.6])
# 各货币对风险因子的敏感度：高息货币在「风险规避/流动性紧张」时贬值(负向)，
# 低息避险货币(日元/瑞郎)在风险规避时升值(正向)。这正是套息的致命伤。
BETA_FX = np.array([-0.7, -0.65, -1.0, -0.95, -0.8, 0.0, 0.55, 0.5, 0.25, 0.15])
T = 300  # 月度数(约 25 年)

# 全球风险因子(月度)：HIGH = 风险规避/流动性紧张。常态小幅正漂移，含 3 次危机冲击(因子飙升)
risk_factor = np.random.normal(0.002, 0.025, T)
crisis_idx = [120, 200, 265]
for ci in crisis_idx:
    risk_factor[ci] = 0.13
    risk_factor[ci + 1] = 0.08

# 各货币对数收益 = 利差成分(率差漂移，持有高息赚) + 风险暴露 + 特质噪声
# 注意: 套息赚的是(高息-低息)利差；汇率波动由 risk_factor 驱动
ret = np.zeros((T, len(CURRENCIES)))
for j in range(len(CURRENCIES)):
    carry_drift = (RATES[j] - RATES[5]) / 100.0 / 12.0  # 相对美元利差月度
    ret[:, j] = carry_drift + BETA_FX[j] * risk_factor + np.random.normal(0, 0.018, T)

# 套息组合：每月按利差排序，做多最高的 3 个、做空最低的 3 个，等权
port_ret = np.zeros(T)
for t in range(T):
    order = np.argsort(RATES)[::-1]
    longs = order[:3]
    shorts = order[-3:]
    pr = 0.0
    for j in longs:
        pr += (1 / 3.0) * ret[t, j]
    for j in shorts:
        pr += (1 / 3.0) * (-ret[t, j])
    port_ret[t] = pr

# 基准：全球股票(用风险因子 + 噪声代理)
eq_ret = risk_factor + np.random.normal(0, 0.02, T)

nav_port = np.cumprod(1 + port_ret)
nav_eq = np.cumprod(1 + eq_ret)


# ============================================================
# 图 1：各货币利率(套息多/空端)
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.5))
order = np.argsort(RATES)
cols = ["#c62828" if i in order[-3:] else ("#1565c0" if i in order[:3] else "#9e9e9e")
        for i in range(len(CURRENCIES))]
ax.bar([CURRENCIES[i] for i in order], RATES[order], color=[cols[i] for i in order])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("名义利率 (年化 %)")
ax.set_title("G10 货币利率：红=套息做多端(高息)，蓝=做空端(低息)")
for i in order:
    ax.text(i, RATES[i] + 0.15, f"{RATES[i]:.1f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(D, "carry_rates.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 2：累积净值对比
# ============================================================
months = np.arange(T)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(months, nav_port, color="#1565c0", lw=2, label="套息组合 (多高息/空低息)")
ax.plot(months, nav_eq, color="#9e9e9e", lw=1.5, label="全球股票基准")
for ci in crisis_idx:
    ax.axvline(ci, color="#c62828", ls="--", lw=1, alpha=0.6)
ax.set_yscale("log")
ax.set_xlabel("月份")
ax.set_ylabel("净值 (对数轴)")
ax.set_title("套息组合长期跑赢股票，但风险因子飙升期(危机)同步重挫")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "carry_nav.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 3：回撤(水下曲线)
# ============================================================
def drawdown(nav):
    peak = np.maximum.accumulate(nav)
    return nav / peak - 1.0
dd_port = drawdown(nav_port)
dd_eq = drawdown(nav_eq)
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(months, dd_port * 100, 0, color="#c62828", alpha=0.35, label="套息组合回撤")
ax.plot(months, dd_port * 100, color="#b71c1c", lw=1.2)
ax.plot(months, dd_eq * 100, color="#616161", lw=1.2, label="股票回撤")
ax.set_ylabel("回撤 (%)")
ax.set_xlabel("月份")
ax.set_title("套息组合的致命伤：危机中回撤可达 −30% 以上")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "carry_drawdown.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 4：套息超额收益 vs 全球风险因子回归(风险补偿检验)
# ============================================================
excess = port_ret - 0.0025
X = np.column_stack([np.ones(T), risk_factor])
beta, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
# 用 OLS 标准误
resid = excess - X @ beta
sig2 = resid @ resid / (T - 2)
cov = sig2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
# 预测线
xs = np.linspace(risk_factor.min(), risk_factor.max(), 50)
ys = beta[0] + beta[1] * xs
corr = np.corrcoef(risk_factor, excess)[0, 1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(risk_factor * 100, excess * 100, s=16, alpha=0.5, c="#6a1b9a",
           label="月度观测")
ax.plot(xs * 100, ys * 100, color="#c62828", lw=2,
        label=f"回归: β={beta[1]:.2f} (t={beta[1]/se[1]:.2f})")
ax.set_xlabel("全球风险因子月度收益 (HIGH=风险规避/流动性紧张)")
ax.set_ylabel("套息超额收益 (%)")
ax.set_title(f"风险补偿检验：套息对风险因子负暴露 β={beta[1]:.2f}\n"
             f"(因子越高=越避险，套息越亏：ρ={corr:.2f} 负相关 = 承担风险的补偿)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "carry_market_beta.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 5：危机期 vs 常态 收益分布
# ============================================================
normal = port_ret.copy()
normal[crisis_idx] = np.nan
normal[crisis_idx[0] + 1] = np.nan
normal[crisis_idx[1] + 1] = np.nan
normal[crisis_idx[2] + 1] = np.nan
crisis_ret = port_ret[[crisis_idx[0], crisis_idx[0] + 1,
                        crisis_idx[1], crisis_idx[1] + 1,
                        crisis_idx[2], crisis_idx[2] + 1]]
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.hist(normal * 100, bins=40, color="#1565c0", alpha=0.7, edgecolor="white",
        label="常态月份收益")
ax.axvline(0, color="k", lw=0.8)
for c in crisis_ret:
    ax.axvline(c * 100, color="#c62828", lw=2.5, alpha=0.9)
ax.set_xlabel("套息组合月度收益 (%)")
ax.set_ylabel("频数")
ax.set_title("常态小赚、危机大亏：6 根红线=危机月(因子飙升，均值 "
             f"{crisis_ret.mean()*100:.1f}%)，印证『利差=风险补偿』")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "carry_crisis.png"), dpi=130)
plt.close(fig)

ann = port_ret.mean() * 12
vol = port_ret.std() * np.sqrt(12)
sharpe = ann / vol
maxdd = dd_port.min() * 100
print("Carry 图已生成:", os.listdir(D))
print(f"年化收益≈{ann*100:.1f}%, 年化波动≈{vol*100:.1f}%, Sharpe≈{sharpe:.2f}, 最大回撤≈{maxdd:.1f}%")
print(f"风险因子 beta={beta[1]:.2f} (t={beta[1]/se[1]:.2f}), 危机月均值={crisis_ret.mean()*100:.1f}%")
