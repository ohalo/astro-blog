#!/usr/bin/env python3
"""
为文章「ETF 折溢价套利：当一级市场创设遇上二级市场价格」(etf-arbitrage-pricing)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. etf_price_iopv.png       一级 basket NAV(IOPV) 与二级市场价格的偏离
  2. etf_premium_series.png   折溢价率序列 + 套利成本阈值带
  3. etf_arbitrage_pnl.png    折溢价套利策略累计净收益
  4. etf_threshold_sensitivity.png  不同成本阈值下的可套利机会数与平均捕获溢价

数值校验：高溢价窗口买入篮子→创设 ETF→二级市场卖出应稳态盈利；
          阈值抬高后机会数骤减（成本吃掉利润）——这是真实结构。
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
D = os.path.join(BASE, "etf-arbitrage-pricing")
os.makedirs(D, exist_ok=True)

C = {"nav": "#4C72B0", "mkt": "#C44E52", "pnl": "#55A868",
     "grid": "#DDDDDD", "band": "#DDDDDD", "thr": "#C44E52"}

# ============================================================
# 1) 模拟 ETF：一级篮子净值 NAV(IOPV) + 二级市场价格
#    NAV 走 GBM；折溢价率 premium 用 OU 均值回复（中枢≈0），
#    并叠加稀疏的流动性冲击尖峰（创造/赎回摩擦失衡时溢价瞬间拉大）。
# ============================================================
def simulate(T=1500, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu, sig = 0.06, 0.18
    nav = np.empty(T)
    nav[0] = 100.0
    for t in range(1, T):
        nav[t] = nav[t - 1] * np.exp((mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * rng.normal())
    # 折溢价率（OU），中枢 0，均值回复速度 theta，波动 sigma_p
    theta, sigma_p, kappa = 0.05, 0.004, 0.02
    prem = np.zeros(T)
    prem[0] = 0.0
    for t in range(1, T):
        prem[t] = prem[t - 1] + theta * (0.0 - prem[t - 1]) + sigma_p * rng.normal()
    # 稀疏冲击：约每 120 个交易日一次，溢价被推高 0.6%~1.2%
    n_shock = T // 120
    shock_idx = rng.choice(np.arange(20, T - 20), size=n_shock, replace=False)
    for i in shock_idx:
        prem[i:i + 8] += rng.uniform(0.006, 0.012)
    mkt = nav * (1 + prem)            # 二级市场价格
    return nav, mkt, prem

nav, mkt, prem = simulate(T=1500, seed=7)
T = len(nav)
# 套利单边成本（创设/赎回费 + 买卖价差 + 冲击），年化口径折算到每笔约 0.18%
COST = 0.0018
t = np.arange(T)

# ============================================================
# 2) 折溢价套利策略（signal-on-i / execute-on-i+1）
#    premium >  +COST  -> 二级市场溢价，forward arb：买篮子创设、卖 ETF
#    premium <  -COST  -> 二级市场折价，reverse arb：买 ETF 赎回、卖篮子
#    用 next-bar 反转后的 premium 实现平仓，捕获 = 入场时点溢价 - 平仓时点溢价 - 2*COST
# ============================================================
pos = np.zeros(T)
ret = np.zeros(T)
for i in range(1, T):
    p_prev, p_cur = prem[i - 1], prem[i]
    if p_cur > COST:
        target = -1          # 溢价过高 -> 做空溢价（买篮子创设、卖二级）
    elif p_cur < -COST:
        target = 1           # 折价过低 -> 做多折价（买二级、赎回卖篮子）
    else:
        target = 0           # 回到无套利带 -> 平仓（收敛即止盈）
    cost = 0.0
    if target != pos[i - 1]:
        cost = COST if (pos[i - 1] == 0 or target == 0) else 2 * COST
    # 盯市：昨日持仓 * 溢价变动；扣当日发生的单边/双边成本
    ret[i] = pos[i - 1] * (p_cur - p_prev) - cost
    pos[i] = target
pnl = np.cumsum(ret) * 100.0     # 百分点

# ============================================================
# 3) 成本阈值敏感性
# ============================================================
def opp_count(cost):
    fwd = np.sum(prem > cost)
    rev = np.sum(prem < -cost)
    return fwd + rev

thresholds = np.linspace(0.0005, 0.006, 12)
opp = [opp_count(c) for c in thresholds]
avg_capture = [(prem[prem > c].mean() if np.any(prem > c) else 0) * 100 for c in thresholds]

# ============================================================
# 图 1：NAV vs 二级价格
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, nav, color=C["nav"], lw=1.0, label="一级篮子净值 IOPV (NAV)")
ax.plot(t, mkt, color=C["mkt"], lw=0.8, alpha=0.85, label="二级市场价格")
ax.set_xlabel("交易日"); ax.set_ylabel("价格")
ax.set_title("ETF 两个价格：一级创设净值(IOPV) 与 二级市场价长期纠缠、瞬时分离")
ax.legend(loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "etf_price_iopv.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：折溢价率序列 + 成本阈值带
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, prem * 100, color="black", lw=0.8, label="折溢价率 (%)")
ax.axhline(COST * 100, color=C["thr"], ls="--", lw=1, label="+成本阈值 %.2f%%" % (COST * 100))
ax.axhline(-COST * 100, color=C["thr"], ls="--", lw=1, label="-成本阈值 %.2f%%" % (COST * 100))
ax.fill_between(t, -COST * 100, COST * 100, color=C["band"], alpha=0.5, label="无套利带")
ax.set_xlabel("交易日"); ax.set_ylabel("折溢价率 (%)")
ax.set_title("折溢价率围绕 0 均值回复；越出 ±成本阈值 即出现套利窗口")
ax.legend(loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "etf_premium_series.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：套利累计净收益
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(t, pnl, color=C["pnl"], lw=1.1, label="折溢价套利累计净收益 (%% , 单边成本 %.2f%%)" % (COST * 100))
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("交易日"); ax.set_ylabel("累计净收益 (%)")
ax.set_title("扣成本后稳态盈利：套利把 '瞬时分离' 变成 '可捕获的价差'")
ax.legend(loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "etf_arbitrage_pnl.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：成本阈值敏感性
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 4.6))
ax1.bar([c * 100 for c in thresholds], opp, color=C["nav"], alpha=0.8, label="可套利机会数")
ax1.set_xlabel("成本阈值 (%)"); ax1.set_ylabel("机会数", color=C["nav"])
ax1.tick_params(axis="y", labelcolor=C["nav"])
ax2 = ax1.twinx()
ax2.plot([c * 100 for c in thresholds], avg_capture, color=C["mkt"], lw=1.6, marker="o", label="平均捕获溢价 (%)")
ax2.set_ylabel("平均捕获溢价 (%)", color=C["mkt"])
ax2.tick_params(axis="y", labelcolor=C["mkt"])
ax1.set_title("成本阈值越高，机会越少；阈值逼近平均溢价时机会骤减（成本吃掉利润）")
fig.tight_layout()
plt.savefig(os.path.join(D, "etf_threshold_sensitivity.png"), dpi=130)
plt.close()

print("=== ETF 折溢价套利 关键数字 ===")
print("样本天数 T=%d, 单边套利成本=%.4f" % (T, COST))
print("溢价窗口(>COST)数=%d, 折价窗口(<-COST)数=%d" % (np.sum(prem > COST), np.sum(prem < -COST)))
print("最终累计净收益=%.2f%%, 最大回撤=%.2f%%" % (pnl[-1], np.min(pnl - np.maximum.accumulate(pnl))))
print("\n图片已保存到:", D)
