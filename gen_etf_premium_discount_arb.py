#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「ETF 折溢价套利：一二级市场价差的可交易窗口」
(etf-premium-discount-arb) 生成真实配图与核心数值。

机制（自洽合成，仅用于演示方法，固定 seed 可复现）：
  * 合成 ETF 的 IOPV（一级篮子净值）路径：带轻微正漂移的随机游走。
  * 二级市价 = IOPV * (1 + premium)，premium 是均值回复 OU 过程（中枢 0，
    缓慢回归 ~20 日）+ 偶发的流动性冲击跳变（瞬时的 ±2%~4% 折溢价尖峰，
    随后回吐）。这模拟了「一二级瞬时定价割裂」产生可交易窗口的真实来源。
  * 收敛套利：当 premium > 阈值 th 时，ETF 相对 IOPV 被高估 → 做空 ETF，
    待 premium 回落穿过 0 平仓；当 premium < -th 时做多，待回升穿 0 平仓。
    每笔毛收益 ≈ 入场时折溢价幅度，扣双边交易成本（每腿 cost_rate）。
  * 输出：①折溢价序列与 IOPV/市价对比；②逐笔收益分布与累计 PnL；
          ③阈值敏感性（年化收益 / Sharpe / 交易次数）；④收敛耗时分布。

所有数值与图表均由本脚本 numpy 真实计算生成。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

SLUG = "etf-premium-discount-arb"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"vm": "#4C72B0", "st": "#C44E52", "grid": "#DDDDDD", "lev": "#55A868",
     "dd": "#8172B3", "eq": "#DD8452", "bd": "#CCB974", "dark": "#333333",
     "gold": "#DD8452", "blue": "#4C72B0", "green": "#55A868", "red": "#C44E52"}

rng = np.random.default_rng(20261007)
T = 252 * 8                       # 8 年日度
dt = 1.0
# ---- IOPV 路径（一级篮子净值，轻微正漂移）----
iopv = np.empty(T + 1)
iopv[0] = 1.0
iopv[1:] = np.cumprod(1.0 + rng.normal(0.0003, 0.009, T))

# ---- 折溢价 premium：OU 均值回复 + 偶发冲击跳变 ----
theta, sigma = 0.04, 0.0009       # 缓慢回复（~25 日），日波动 9bps
prem = np.zeros(T)
jump_prob, jump_sd = 0.004, 0.022  # 0.4%/日 概率出现 ±2.2%(sd) 流动性冲击
for t in range(1, T):
    prem[t] = (1 - theta) * prem[t - 1] + sigma * rng.standard_normal()
    if rng.random() < jump_prob:
        prem[t] += rng.normal(0.0, jump_sd)
# 市价 = IOPV * (1 + premium)
price = iopv[1:] * (1.0 + prem)

# ============ 收敛套利回测 ============
def backtest(th, cost_rate):
    pos = 0          # +1 多 / -1 空 / 0 空仓
    entry_prem = 0.0
    entry_price = 0.0
    pnl = []         # 逐笔净收益（小数）
    days_held = []
    entry_day = 0
    eq = [1.0]
    for t in range(T):
        p = prem[t]
        if pos == 0:
            if p > th:
                pos = -1; entry_prem = p; entry_price = price[t]; entry_day = t
            elif p < -th:
                pos = 1; entry_prem = p; entry_price = price[t]; entry_day = t
        else:
            # 回到中枢（穿过 0）则平仓
            if (pos == -1 and p <= 0) or (pos == 1 and p >= 0):
                exit_price = price[t]
                gross = pos * (exit_price - entry_price) / entry_price
                net = gross - 2 * cost_rate
                pnl.append(net)
                days_held.append(t - entry_day)
                pos = 0
        # 标记到净值（简化处理：持仓期不重复计收益，只在平仓时计入）
        eq.append(eq[-1] * (1.0 + (pnl[-1] if pos == 0 and len(pnl) else 0.0)))
    return np.array(pnl), np.array(days_held), price, prem

cost_rate = 0.0008                 # 单边 8bps（含冲击+佣金）
TH = 0.005                        # 默认阈值 50bps
pnl, days_held, price, prem = backtest(TH, cost_rate)

def ann_stats(pnl_list):
    if len(pnl_list) == 0:
        return 0, 0, 0, 0
    r = np.array(pnl_list)
    n = len(r)
    ann_ret = (1 + r.mean()) ** (252.0 / (n / (T / 252.0))) - 1 if n else 0
    # 简化：用逐笔平均收益年化（按年均交易次数）
    trades_per_year = n / (T / 252.0)
    ann = (1 + r.mean()) ** trades_per_year - 1
    sharpe = r.mean() / r.std() * np.sqrt(trades_per_year) if r.std() > 0 else 0
    hit = (r > 0).mean()
    return ann, sharpe, hit, n

ann, sharpe, hit, n_trades = ann_stats(pnl)

# ============ 图1：IOPV vs 市价 + 折溢价带 ============
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(iopv[1:], color=C["blue"], lw=1.3, label="IOPV（一级篮子净值）")
ax.plot(price, color=C["red"], lw=1.1, alpha=0.85, label="二级市价")
ax.plot(iopv[1:] * (1 + TH), color=C["gold"], ls="--", lw=1.0, label=f"溢价阈值 +{TH:.1%}")
ax.plot(iopv[1:] * (1 - TH), color=C["gold"], ls="--", lw=1.0, label=f"折价阈值 -{TH:.1%}")
ax.set_xlabel("交易日")
ax.set_ylabel("净值 / 价格")
ax.set_title("IOPV 与二级市价：折溢价带界定可交易窗口")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "iopv_vs_price.png")); plt.close(fig)

# ============ 图2：折溢价序列 + 交易点 ============
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(prem * 100, color=C["dd"], lw=1.0, label="折溢价 premium (%)")
ax.axhline(TH * 100, color=C["red"], ls="--", lw=1.2, label=f"阈值 +{TH:.1%}")
ax.axhline(-TH * 100, color=C["green"], ls="--", lw=1.2, label=f"阈值 -{TH:.1%}")
ax.axhline(0, color=C["dark"], lw=0.8)
# 标注入场点
pos = 0; entry_t = []
for t in range(T):
    if pos == 0:
        if prem[t] > TH:
            pos = -1; entry_t.append(t)
        elif prem[t] < -TH:
            pos = 1; entry_t.append(t)
    else:
        if (pos == -1 and prem[t] <= 0) or (pos == 1 and prem[t] >= 0):
            pos = 0
if entry_t:
    ax.scatter(entry_t, prem[entry_t] * 100, color=C["st"], s=14, zorder=5, label="入场信号")
ax.set_xlabel("交易日")
ax.set_ylabel("折溢价 (%)")
ax.set_title("折溢价 OU 回复 + 冲击跳变：信号触发点一览")
ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "premium_series.png")); plt.close(fig)

# ============ 图3：逐笔收益分布 + 累计 PnL ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.0))
a1.hist(pnl * 100, bins=40, color=C["blue"], alpha=0.8)
a1.axvline(0, color=C["red"], lw=1.2, ls="--")
a1.set_title(f"逐笔净收益分布\n命中率 {hit*100:.1f}%  |  共 {n_trades} 笔")
a1.set_xlabel("单笔净收益 (%)")
a1.set_ylabel("笔数")
cum = np.cumprod(1 + pnl)
a2.plot(cum, color=C["green"], lw=1.5)
a2.set_title("累计净值（逐笔复利）")
a2.set_xlabel("交易序号")
a2.set_ylabel("净值")
a2.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pnl_distribution.png")); plt.close(fig)

# ============ 图4：阈值敏感性 ============
ths = np.linspace(0.001, 0.02, 20)
ann_arr, sh_arr, n_arr = [], [], []
for th in ths:
    p, d, _, _ = backtest(th, cost_rate)
    a, s, h, nt = ann_stats(p)
    ann_arr.append(a); sh_arr.append(s); n_arr.append(nt)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(ths * 100, np.array(sh_arr), color=C["blue"], lw=1.8, marker="o", ms=3, label="Sharpe（年化）")
ax.set_xlabel("入场阈值（折溢价 %）")
ax.set_ylabel("Sharpe", color=C["blue"])
ax.tick_params(axis="y", labelcolor=C["blue"])
ax2 = ax.twinx()
ax2.plot(ths * 100, np.array(n_arr), color=C["red"], lw=1.6, marker="s", ms=3, label="年交易次数")
ax2.set_ylabel("年交易次数", color=C["red"])
ax2.tick_params(axis="y", labelcolor=C["red"])
ax.set_title("阈值敏感性：阈值越高越过滤噪声，但交易次数骤降")
ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "threshold_sensitivity.png")); plt.close(fig)

# ============ 图5：收敛耗时分布 ============
fig, ax = plt.subplots(figsize=(7.4, 4.4))
if len(days_held):
    ax.hist(days_held, bins=range(0, max(days_held) + 2), color=C["lev"], alpha=0.85)
    ax.axvline(np.median(days_held), color=C["red"], ls="--", lw=1.5,
               label=f"中位数 {np.median(days_held):.0f} 日")
ax.set_xlabel("从入场到平仓的持仓天数")
ax.set_ylabel("笔数")
ax.set_title("收敛耗时分布：大多在 1~3 周内回到中枢")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence_days.png")); plt.close(fig)

print("=" * 64)
print("ARTICLE_ETF_PREMIUM_DISCOUNT_METRICS")
print(f"T={T} days; default threshold={TH:.3f}; cost_rate(one leg)={cost_rate}")
print(f"premium: mean={prem.mean()*100:.3f}% std={prem.std()*100:.3f}% "
      f"max={prem.max()*100:.2f}% min={prem.min()*100:.2f}%")
print(f"trades={n_trades}  hit_rate={hit*100:.1f}%")
print(f"avg per-trade net={pnl.mean()*100:.4f}%  std={pnl.std()*100:.4f}%")
print(f"annualized_return={ann*100:.2f}%  sharpe={sharpe:.2f}")
print(f"final cumulative NAV={cum[-1]:.3f}  (cost eaten per trade={2*cost_rate*100:.3f}%)")
print(f"median convergence days={np.median(days_held):.0f}")
# 无成本 vs 有成本对比
p_free, _, _, _ = backtest(TH, 0.0)
print(f"same strategy WITHOUT cost: ann_ret={( (1+p_free.mean())**(252/(len(p_free)/(T/252.0))) -1)*100:.2f}% "
      f"sharpe={p_free.mean()/p_free.std()*np.sqrt(252/(len(p_free)/(T/252.0))):.2f}")
print("=" * 64)
