#!/usr/bin/env python3
"""
为文章「日历价差与跨期套利：赚期限结构曲线的钱」(calendar-spread-arbitrage) 生成真实配图。

核心逻辑：
  - 期权日历价差：买远月 ATM 看涨 + 卖近月同价看涨，净支出(debit)；
  - 利润引擎 = 近月 Theta 衰减更快 + 波动率期限结构(contango)的 carry；
  - 最大利润出现在到期时标的≈行权价(远月 ATM 时间价值最大、近月归零)；
  - 全部用 Black-Scholes 真实定价 + GBM 模拟，非占位图。
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "calendar-spread-arbitrage")
os.makedirs(D, exist_ok=True)

# ---------- Black-Scholes ----------
def _ndf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ndf(d1) - K * math.exp(-r * T) * _ndf(d2)

def bs_call_theta(S, K, T, r, sigma):
    # 返回每单位时间的 Theta（年化），正值=时间价值流失速度
    if T <= 1e-6:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    term = -(S * pdf * sigma) / (2.0 * math.sqrt(T))
    term += r * K * math.exp(-r * T) * _ndf(d2)
    return term   # 看涨 Theta（通常为负，代表价值流失）

S0 = K = 100.0
r = 0.0
sigma_near = 0.19     # 近月隐含波动率（期限结构 contango：近低远高）
sigma_far = 0.21      # 远月隐含波动率（远月更贵）
sigma = 0.20          # 模拟标的价格用的参考已实现波动
T1 = 30 / 252         # 近月
T2 = 90 / 252         # 远月
T_remaining = T2 - T1 # 近月到期时远月剩余时间

net_debit = bs_call(S0, K, T2, r, sigma_far) - bs_call(S0, K, T1, r, sigma_near)
print(f"建仓净支出(debit) = {net_debit:.2f}（远月 {bs_call(S0,K,T2,r,sigma_far):.2f} - 近月 {bs_call(S0,K,T1,r,sigma_near):.2f}）")

# ============================================================
# 图1：近月到期时的盈亏剖面——钟形，最大利润在 ATM
# ============================================================
S_grid = np.linspace(60, 140, 200)
# 近月到期：近月=内在价值，远月还剩 T_remaining 时间
pnl_at_T1 = np.array([
    bs_call(s, K, T_remaining, r, sigma_far) - max(s - K, 0.0) - net_debit
    for s in S_grid
])
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(S_grid, pnl_at_T1, color="#1f77b4", lw=2.4, label="日历价差盈亏")
ax.axhline(0, color="#888", lw=1.0)
ax.fill_between(S_grid, 0, pnl_at_T1, where=(pnl_at_T1 > 0), color="#2ca02c", alpha=0.15)
ax.fill_between(S_grid, 0, pnl_at_T1, where=(pnl_at_T1 <= 0), color="#d62728", alpha=0.12)
ax.axvline(K, color="#555", ls="--", lw=1.2, label=f"行权价 K={K}")
ax.set_xlabel("近月到期时标的价格 S", fontsize=11)
ax.set_ylabel("组合盈亏", fontsize=11)
ax.set_title("日历价差盈亏剖面：最大利润出现在 S≈K（远月时间价值最大、近月归零）",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "calendar_payoff.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：Theta 衰减对比——近月比远月「掉血」更快
# ============================================================
days = np.arange(1, 61)            # 未来 60 个交易日
t_near = np.maximum(T1 - days / 252, 1e-6)
t_far = np.maximum(T2 - days / 252, 1e-6)
val_near = [bs_call(S0, K, t, r, sigma) for t in t_near]
val_far = [bs_call(S0, K, t, r, sigma) for t in t_far]
theta_near = [bs_call_theta(S0, K, t, r, sigma) for t in t_near]
theta_far = [bs_call_theta(S0, K, t, r, sigma) for t in t_far]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
axes[0].plot(days, val_near, color="#d62728", lw=2.2, label="近月期权价值")
axes[0].plot(days, val_far, color="#1f77b4", lw=2.2, label="远月期权价值")
axes[0].set_xlabel("距今天数", fontsize=11); axes[0].set_ylabel("期权时间价值", fontsize=11)
axes[0].set_title("时间价值衰减：近月更快见底", fontsize=11.5, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25)

axes[1].plot(days, theta_near, color="#d62728", lw=2.2, label="近月 Theta（负）")
axes[1].plot(days, theta_far, color="#1f77b4", lw=2.2, label="远月 Theta（负）")
axes[1].set_xlabel("距今天数", fontsize=11); axes[1].set_ylabel("每日时间价值流失", fontsize=11)
axes[1].set_title("Theta 差：近月「掉血」速度是远月的数倍", fontsize=11.5, fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "calendar_theta.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：P&L 分布——用 GBM 模拟近月到期标的价格
# ============================================================
npaths = 6000
rng = np.random.default_rng(2024)
Z = rng.normal(size=npaths)
S_T1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T1 + sigma * math.sqrt(T1) * Z)
pnl_sim = np.array([
    bs_call(s, K, T_remaining, r, sigma_far) - max(s - K, 0.0) - net_debit
    for s in S_T1
])
print(f"模拟 P&L：均值 {pnl_sim.mean():.2f}，盈利占比 {(pnl_sim>0).mean()*100:.1f}%，"
      f"最大亏损 {pnl_sim.min():.2f}，最大盈利 {pnl_sim.max():.2f}")

fig, ax = plt.subplots(figsize=(11, 5.6))
bins = np.linspace(-12, 12, 60)
ax.hist(pnl_sim, bins=bins, color="#1f77b4", alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", lw=1.2, ls="--")
ax.axvline(pnl_sim.mean(), color="#2ca02c", lw=1.8, label=f"均值 {pnl_sim.mean():.2f}")
ax.set_xlabel("近月到期时日历价差盈亏", fontsize=11)
ax.set_ylabel("路径数", fontsize=11)
ax.set_title("盈亏分布：标的小幅徘徊(≈K)时赚钱，大幅偏离则亏损——典型的「中性」头寸",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "calendar_pnl_dist.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：波动率期限结构——contango 下卖近买远吃 carry
# ============================================================
maturities = np.array([20, 30, 45, 60, 90, 120, 180]) / 252.0
# 构造一条向上倾斜(contango)的 IV 期限结构
iv_term = 0.16 + 0.06 * (1 - np.exp(-np.array([20,30,45,60,90,120,180]) / 90)) + 0.02 * (np.array([20,30,45,60,90,120,180]) / 180)
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(maturities * 252, iv_term * 100, "o-", color="#9467bd", lw=2.2, ms=7,
        label="隐含波动率期限结构 (contango)")
ax.axhline(sigma * 100, color="#888", ls="--", lw=1.2, label=f"平值 ATM 波动 {sigma*100:.0f}%")
# 标出日历价差两腿所在位置
ax.scatter([T1 * 252, T2 * 252],
           [iv_term[1] * 100, iv_term[4] * 100],
           color="#d62728", zorder=5, s=90,
           label=f"日历两腿：卖近月@{iv_term[1]*100:.1f}% / 买远月@{iv_term[4]*100:.1f}%")
ax.set_xlabel("到期天数", fontsize=11)
ax.set_ylabel("隐含波动率 (%)", fontsize=11)
ax.set_title("波动率期限结构：向上倾斜时，卖便宜的近月、买贵的远月，净吃正向 carry",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "calendar_iv_term.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ 日历价差与跨期套利配图生成完成：", sorted(os.listdir(D)))
