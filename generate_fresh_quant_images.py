#!/usr/bin/env python3
"""
为本次两篇「非重复」量化文章生成真实配图（matplotlib 渲染，非占位图）：
  1. bond-futures-basis-trading   (国债期货基差交易与 CTD 期权)
  2. commodity-calendar-spread-roll-yield (跨期价差与展期收益：商品期货日历套利)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"


# ============================================================
# 文章 1：国债期货基差交易与 CTD 期权
# ============================================================
d1 = os.path.join(BASE, "bond-futures-basis-trading")
os.makedirs(d1, exist_ok=True)

# ---- 图1：基差 / 净基差随交割月临近的演化 ----
np.random.seed(20260710)
T = 120
t = np.arange(T)
# 现券净价温和上行（carry 累积）
bond_clean = 101.5 + 0.6 * (t / T) + np.random.normal(0, 0.08, T)
cf = 0.925  # 转换因子
# 期货价 = (现券净价 - carry) / CF 的近似，含期权溢价
carry = 0.9 * (t / T)          # 持有收益(票息-融资成本)，元
futures = (bond_clean - carry) / cf + 0.35 * (1 - t / T)  # 0.35 为初始净基差(期权价值)
gross_basis = bond_clean - futures * cf
net_basis = gross_basis - carry     # 期权价值

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(t, gross_basis, color="#1f77b4", lw=2.0, label="基差 (Gross Basis)")
ax.plot(t, net_basis, color="#d62728", lw=2.0, label="净基差 (Net Basis = 期权价值)")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.fill_between(t, net_basis, 0, where=(net_basis > 0), color="#d62728", alpha=0.15)
ax.set_xlabel("距交割天数（倒数）", fontsize=11)
ax.set_ylabel("元 / 百元面值", fontsize=11)
ax.set_title("国债期货基差随交割临近收敛：净基差=内含交割期权价值", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
ax.annotate("净基差>0：空方期权仍有价值", xy=(15, net_basis[15]),
            xytext=(30, net_basis[15] + 0.15), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#d62728"))
plt.tight_layout()
plt.savefig(os.path.join(d1, "basis_over_time.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图2：可交割券 IRR 条形图，高亮 CTD ----
candidates = ["券A 7Y", "券B 10Y", "券C 10Y", "券D 7Y", "券E 5Y", "券F 10Y"]
irr = np.array([1.82, 2.95, 2.71, 1.45, 1.10, 2.40])  # 年化隐含回购利率 %
ctd_idx = int(np.argmax(irr))
colors = ["#1f77b4"] * len(irr)
colors[ctd_idx] = "#d62728"
fig, ax = plt.subplots(figsize=(11, 5.6))
bars = ax.bar(candidates, irr, color=colors, width=0.6)
ax.axhline(0, color="gray", lw=1)
for i, v in enumerate(irr):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
ax.text(ctd_idx, irr[ctd_idx] + 0.28, "← CTD\n(IRR 最高)", ha="center",
        fontsize=9, color="#d62728", fontweight="bold")
ax.set_ylabel("隐含回购利率 IRR (%)", fontsize=11)
ax.set_title("可交割券 IRR 对比：IRR 最高者即为最便宜可交割券(CTD)", fontsize=13.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d1, "ctd_irr.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图3：做多基差 vs 做空基差 的 PnL 情景 ----
scenarios = ["期权价值归零\n(空方放弃行权)", "CTD 切换\n(券B→券C)", "资金利率上行\n(carry转负)", "逼空\n(现券稀缺)"]
# 做多基差 = 多现券 + 空期货：赚 carry 与期权收敛，亏在 CTD 切换/逼空
long_pnl = np.array([+0.30, -0.55, -0.40, -0.90])
short_pnl = -long_pnl  # 做空基差 = 空现券 + 多期货，方向相反
x = np.arange(len(scenarios))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.bar(x - w/2, long_pnl, w, label="做多基差（多现券+空期货）", color="#2ca02c")
ax.bar(x + w/2, short_pnl, w, label="做空基差（空现券+多期货）", color="#ff7f0e")
ax.axhline(0, color="gray", lw=1)
ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=9.5)
ax.set_ylabel("情景损益（元/百元面值）", fontsize=11)
ax.set_title("基差交易情景分析：多空两端的风险来源截然不同", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d1, "basis_trade_pnl.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 文章 2：跨期价差与展期收益：商品期货日历套利
# ============================================================
d2 = os.path.join(BASE, "commodity-calendar-spread-roll-yield")
os.makedirs(d2, exist_ok=True)

# ---- 图1：期限结构 contango vs back ----
months = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
contango = np.array([78.0, 79.5, 81.0, 82.3, 83.5, 84.6, 85.7, 86.8])
back = np.array([92.0, 90.8, 89.6, 88.5, 87.4, 86.4, 85.5, 84.7])
x = np.arange(len(months))
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(x, contango, "o-", color="#d62728", lw=2, label="Contango（远月升水）")
ax.plot(x, back, "s-", color="#2ca02c", lw=2, label="Backwardation（远月贴水）")
ax.set_xticks(x); ax.set_xticklabels(months, fontsize=10)
ax.set_ylabel("期货价格", fontsize=11)
ax.set_xlabel("到期月份", fontsize=11)
ax.set_title("期限结构两种形态：Contango 下展期亏损，Back 下展期盈利", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.25)
ax.annotate("Back：近月>远月\n持有近月展期赚钱", xy=(6, back[6]),
            xytext=(3.5, back[6] + 3), fontsize=8.5, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))
plt.tight_layout()
plt.savefig(os.path.join(d2, "term_structure.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图2：多商品年化展期收益（back/contango 着色）----
comms = ["原油", "黄金", "铜", "天然气", "大豆", "白银", "棉花", "白糖"]
roll = np.array([-4.8, +1.2, +2.5, -9.6, +3.1, +0.6, -6.2, +1.8])  # 年化展期收益 %
colr = ["#d62728" if v < 0 else "#2ca02c" for v in roll]
fig, ax = plt.subplots(figsize=(11, 5.6))
bars = ax.bar(comms, roll, color=colr, width=0.62)
ax.axhline(0, color="gray", lw=1)
for i, v in enumerate(roll):
    ax.text(i, v + (0.25 if v >= 0 else -0.55), f"{v:+.1f}%", ha="center",
            fontsize=9, fontweight="bold",
            color="#2ca02c" if v >= 0 else "#d62728")
ax.set_ylabel("年化展期收益 Roll Yield (%)", fontsize=11)
ax.set_title("不同商品的展期收益天差地别：Back 结构(绿)赚 carry，Contango(红)亏 carry", fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d2, "roll_yield.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图3：日历价差策略净值（back 结构下多近空远）----
np.random.seed(42)
T = 250
# 在 back 结构下，多近空远赚展期 + 均值回归
daily = np.random.normal(0.0009, 0.006, T)
spread_eq = np.cumprod(1 + daily)
# 叠加一次 regime 翻转（back->contango）造成回撤
flip = int(T * 0.7)
spread_eq[flip:] *= np.cumprod(1 + np.random.normal(-0.004, 0.008, T - flip))
peak = np.maximum.accumulate(spread_eq)
dd = spread_eq / peak - 1
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(range(T), spread_eq, color="#2ca02c", lw=2, label="日历价差净值（多近月+空远月）")
ax.fill_between(range(T), dd, 0, color="#d62728", alpha=0.25, label="回撤")
ax.axvline(flip, color="#ff7f0e", ls="--", lw=1.4, label=f"期限结构翻转 @{flip}日")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("净值", fontsize=11)
ax.set_title("商品日历价差净值：Back 结构稳定盈利，结构翻转即回撤", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d2, "calendar_spread_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ 图像生成完成")
print("   bond-futures-basis-trading:", sorted(os.listdir(d1)))
print("   commodity-calendar-spread-roll-yield:", sorted(os.listdir(d2)))
