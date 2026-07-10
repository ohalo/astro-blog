#!/usr/bin/env python3
"""
为本次两篇量化文章生成真实配图（matplotlib 渲染，非占位图）：
  1. quant-strategy-monitoring  (量化策略监控系统设计)
  2. trading-cost-control        (交易成本控制与优化)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"

# ============================================================
# 文章 1：量化策略监控系统设计
# ============================================================
d1 = os.path.join(BASE, "quant-strategy-monitoring")
os.makedirs(d1, exist_ok=True)

# 图1：单策略监控总览（净值 / 回撤 / 暴露 + 预警阈值）
np.random.seed(20260710)
T = 120
ret = np.random.normal(0.0012, 0.012, T)
ret[T // 2:] += 0.0008  # 后段策略回暖
equity = np.cumprod(1 + ret)
peak = np.maximum.accumulate(equity)
dd = equity / peak - 1
exposure = 0.55 + 0.25 * np.sin(np.linspace(0, 5 * np.pi, T)) + np.random.normal(0, 0.03, T)
exposure = np.clip(exposure, 0, 1)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax0, ax1, ax2 = axes
ax0.plot(range(T), equity, color="#1f77b4", lw=2)
ax0.axhline(equity[0], color="gray", ls=":", lw=1)
ax0.set_ylabel("累积净值", fontsize=11); ax0.set_title("策略监控总览：净值 / 回撤 / 实时暴露", fontsize=14, fontweight="bold")
ax0.grid(True, alpha=0.25)

ax1.fill_between(range(T), dd, 0, color="#d62728", alpha=0.35)
ax1.plot(range(T), dd, color="#d62728", lw=1.2)
ax1.axhline(-0.08, color="#ff7f0e", ls="--", lw=1.4, label="回撤硬止损线 -8%")
ax1.set_ylabel("回撤", fontsize=11); ax1.legend(loc="lower left", fontsize=9); ax1.grid(True, alpha=0.25)

ax2.plot(range(T), exposure, color="#2ca02c", lw=1.6)
ax2.axhline(0.9, color="#d62728", ls="--", lw=1.3, label="暴露上限 90%")
ax2.axhline(0.1, color="#9467bd", ls="--", lw=1.3, label="暴露下限 10%")
ax2.set_ylabel("多头暴露", fontsize=11); ax2.set_xlabel("交易日", fontsize=11)
ax2.legend(loc="upper right", fontsize=9, ncol=2); ax2.grid(True, alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(d1, "monitor_overview.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图2：告警升级状态机
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.set_xticks([]); ax.set_yticks([])

def box(x, y, w, h, text, fc, tc="white"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                 fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=10.5, fontweight="bold")

def arrow(x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.6))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25, text, ha="center", fontsize=8.5, color="#333333")

box(0.3, 3.0, 2.0, 1.2, "指标采集\n(行情/成交/风险)", "#1f77b4")
box(3.0, 3.0, 2.0, 1.2, "规则引擎\n阈值/漂移检测", "#17becf")
box(5.7, 4.6, 2.0, 1.2, "INFO\n(日志+看板)", "#7f7f7f")
box(5.7, 3.0, 2.0, 1.2, "WARN\n(企业微信/邮件)", "#ff7f0e")
box(5.7, 1.4, 2.0, 1.2, "CRITICAL\n(电话/短信+自动减仓)", "#d62728")
box(8.6, 3.0, 2.2, 1.2, "值班响应\n(确认/处置/复盘)", "#2ca02c")
arrow(2.3, 3.6, 3.0, 3.6)
arrow(5.0, 3.6, 5.7, 4.0, "正常")
arrow(5.0, 3.6, 5.7, 3.6, "越界")
arrow(5.0, 3.6, 5.7, 2.0, "失效")
arrow(7.7, 3.6, 8.6, 3.6)
arrow(9.7, 4.2, 7.8, 4.8, "复盘回流")
ax.text(6.0, 6.4, "告警分级与自动处置状态机", fontsize=14, fontweight="bold", ha="center")
ax.text(6.0, 0.4, "注：CRITICAL 可触发自动熔断（暂停交易/降仓），避免人工响应延迟造成的大额亏损",
        fontsize=8.5, ha="center", color="#666666")
plt.tight_layout(); plt.savefig(os.path.join(d1, "alert_flow.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图3：多策略监控状态热力图
np.random.seed(99)
strategies = ["CTA趋势", "股指套利", "可转债网格", "股票多空", "期权波动率", "国债曲线", "商品价差", "ETF套利"]
metrics = ["PnL偏离", "波动异常", "暴露超限", "数据延迟", "成交滑点", "流动性"]
# 0=正常(绿) 1=关注(黄) 2=告警(红)
state = np.array([
    [0,0,0,0,1,0],
    [1,0,0,0,0,0],
    [0,0,2,0,1,0],
    [2,1,0,0,0,1],
    [0,0,0,1,0,0],
    [0,0,0,0,0,0],
    [1,0,0,0,0,2],
    [0,0,1,0,0,0],
])
cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#ffd633", "#d62728"])
fig, ax = plt.subplots(figsize=(10.5, 6.5))
im = ax.imshow(state, cmap=cmap, vmin=0, vmax=2, aspect="auto")
ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, fontsize=10)
ax.set_yticks(range(len(strategies))); ax.set_yticklabels(strategies, fontsize=10)
for i in range(len(strategies)):
    for j in range(len(metrics)):
        labels = ["正常", "关注", "告警"]
        ax.text(j, i, labels[state[i, j]], ha="center", va="center",
                color="white" if state[i, j] != 1 else "black", fontsize=9, fontweight="bold")
ax.set_title("多策略监控矩阵：8 策略 × 6 维度实时状态", fontsize=14, fontweight="bold")
ax.set_xlabel("监控维度", fontsize=10); ax.set_ylabel("策略", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(d1, "multi_strategy_status.png"), dpi=150, bbox_inches="tight"); plt.close()

# ============================================================
# 文章 2：交易成本控制与优化
# ============================================================
d2 = os.path.join(BASE, "trading-cost-control")
os.makedirs(d2, exist_ok=True)

# 图1：交易成本拆解（不同订单规模下的成分占比）
def bp(x):
    return f"{x*1e4:.1f}bp"
trade_size = ["小单\n(<1%)", "中单\n(1-5%)", "大单\n(5-15%)", "超大单\n(>15%)"]
commission = np.array([0.0010, 0.0010, 0.0010, 0.0010])
spread = np.array([0.0008, 0.0012, 0.0020, 0.0035])
slippage = np.array([0.0005, 0.0015, 0.0045, 0.0090])
impact = np.array([0.0002, 0.0018, 0.0075, 0.0180])
opps = np.array([0.0001, 0.0006, 0.0030, 0.0080])
totals = commission + spread + slippage + impact + opps
comm_p = commission / totals * 100
spread_p = spread / totals * 100
slip_p = slippage / totals * 100
impact_p = impact / totals * 100
opps_p = opps / totals * 100
x = np.arange(len(trade_size))
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x, comm_p, 0.6, label="佣金/印花税", color="#1f77b4")
ax.bar(x, spread_p, 0.6, bottom=comm_p, label="买卖价差", color="#17becf")
ax.bar(x, slip_p, 0.6, bottom=comm_p + spread_p, label="执行滑点", color="#ff7f0e")
ax.bar(x, impact_p, 0.6, bottom=comm_p + spread_p + slip_p, label="市场冲击", color="#d62728")
ax.bar(x, opps_p, 0.6, bottom=comm_p + spread_p + slip_p + impact_p, label="机会成本", color="#7f7f7f")
ax.set_xticks(x); ax.set_xticklabels(trade_size, fontsize=10)
ax.set_ylabel("占交易成本比例 (%)", fontsize=11)
ax.set_title("交易成本拆解：订单越大，市场冲击与机会成本越主导", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, ncol=2)
for i, t in enumerate(totals):
    ax.text(i, 100.5, f"总{bp(t)}", ha="center", fontsize=9, fontweight="bold", color="#333333")
ax.set_ylim(0, 115)
plt.tight_layout(); plt.savefig(os.path.join(d2, "cost_decomposition.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图2：VWAP / TWAP 执行轨迹 vs 到达价
np.random.seed(7)
M = 78  # 分钟
minutes = np.arange(M)
# U 型成交量分布
vol_profile = np.exp(-((minutes - 0) / 18) ** 2) * 1.4 + np.exp(-((minutes - (M - 1)) / 16) ** 2) * 1.6 + 0.25
vol_profile /= vol_profile.sum()
# 价格路径（带噪声的随机游走）
drift = np.cumsum(np.random.normal(0, 0.0006, M))
price = 100 + 100 * drift
arrival = price[0]
# VWAP 执行均价（按量加权）
vwap_exec = np.sum(price * vol_profile) / np.sum(vol_profile)
# TWAP 执行均价（等权）
twap_exec = price.mean()
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(minutes, price, color="#333333", lw=1.8, label="市场价格")
ax.axhline(arrival, color="#1f77b4", ls="-", lw=1.6, label=f"到达价 {arrival:.2f}")
ax.axhline(vwap_exec, color="#2ca02c", ls="--", lw=1.6, label=f"VWAP 执行 {vwap_exec:.2f}")
ax.axhline(twap_exec, color="#ff7f0e", ls=":", lw=1.8, label=f"TWAP 执行 {twap_exec:.2f}")
ax.set_xlabel("交易时段（分钟）", fontsize=11); ax.set_ylabel("价格", fontsize=11)
ax.set_title("VWAP / TWAP 执行轨迹：被动算法跟随成交量，降低冲击成本", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(d2, "vwap_twap.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图3：实施缺口 vs 参与率（最优参与率呈 U 型）
part = np.linspace(0.02, 0.6, 40)
# 冲击成本随参与率上升；机会成本（时间风险）随参与率下降 -> U 型
impact_cost = 14 * part ** 1.5          # bp
timing_risk = 9 * (1 - part) ** 1.3 + 2  # bp
total_cost = impact_cost + timing_risk
opt_idx = int(np.argmin(total_cost))
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(part * 100, impact_cost, color="#d62728", lw=2, label="市场冲击成本")
ax.plot(part * 100, timing_risk, color="#1f77b4", lw=2, label="时间风险/机会成本")
ax.plot(part * 100, total_cost, color="#2ca02c", lw=2.6, label="总实施缺口")
ax.axvline(part[opt_idx] * 100, color="#ff7f0e", ls="--", lw=1.6,
            label=f"最优参与率 ≈ {part[opt_idx]*100:.0f}%")
ax.scatter([part[opt_idx] * 100], [total_cost[opt_idx]], color="#ff7f0e", zorder=5, s=60)
ax.set_xlabel("参与率 (% of ADV)", fontsize=11); ax.set_ylabel("成本 (bp)", fontsize=11)
ax.set_title("实施缺口 vs 参与率：存在最优参与率，而非越快越好", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper center", fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(d2, "participation_shortfall.png"), dpi=150, bbox_inches="tight"); plt.close()

print("✅ 图像生成完成")
print("   quant-strategy-monitoring:", sorted(os.listdir(d1)))
print("   trading-cost-control:", sorted(os.listdir(d2)))
