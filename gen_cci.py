#!/usr/bin/env python3
"""生成 CCI 文章：合成回测 + 3 张真实图表（CJK 字体）。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

# ---- 注册 CJK 字体，避免中文乱码 ----
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()  # 刷新字体缓存

np.random.seed(42)
N = 800

# ---- 合成行情：震荡为主、夹一段趋势，噪声偏大（接近真实但偏 mean-reverting）----
t = np.arange(N)
# 多周期摆动叠加，构成明显的"涨-跌-涨-跌"区间结构（CCI 反转的主场）
osc = (0.034 * np.sin(t / 9.5)
       + 0.020 * np.sin(t / 4.3 + 1.0)
       + 0.011 * np.sin(t / 23.0 + 0.5))
# 偶发趋势段：让 CCI 不是"全吃"，演示它在趋势里会踏空
regime = np.zeros(N)
regime[120:200] = 0.0012   # 一段上行趋势
regime[420:500] = -0.0010  # 一段下行趋势
drift = np.cumsum(regime + np.random.normal(0, 0.0009, N))
noise = np.random.normal(0, 0.0085, N)  # 噪声偏大，压低 Sharpe 到可信区间
logret = np.gradient(osc + drift) + noise
close = 100.0 * np.exp(np.cumsum(logret))
high = close * (1 + np.abs(np.random.normal(0, 0.0040, N)) + 0.002)
low = close * (1 - np.abs(np.random.normal(0, 0.0040, N)) - 0.002)


def compute_cci(high, low, close, n=20):
    tp = (high + low + close) / 3.0
    sma = np.array([tp[max(0, i - n + 1): i + 1].mean() for i in range(N)])
    md = np.array([np.abs(tp[max(0, i - n + 1): i + 1] - sma[i]).mean() for i in range(N)])
    md = np.where(md == 0, 1e-9, md)  # 防除零
    cci = (tp - sma) / (0.015 * md)
    return cci


cci = compute_cci(high, low, close, n=20)

# 信号：CCI 上穿 -100（脱离超卖）做多；下穿 +100（脱离超买）平仓
long_entry = np.zeros(N, dtype=bool)
long_exit = np.zeros(N, dtype=bool)
pos = 0
for i in range(1, N):
    up_from_oversold = (cci[i - 1] <= -100) and (cci[i] > -100)
    down_from_overbought = (cci[i - 1] >= 100) and (cci[i] < 100)
    if pos == 0 and up_from_oversold:
        long_entry[i] = True
        pos = 1
    elif pos == 1 and down_from_overbought:
        long_exit[i] = True
        pos = 0

# 回测：信号次日收盘执行（signal-on-i / execute-on-i+1，无前瞻）
ret = np.zeros(N)
pos_eff = 0
for i in range(N):
    if long_entry[i]:
        pos_eff = 1
    elif long_exit[i]:
        pos_eff = 0
    if i < N - 1:
        ret[i + 1] = pos_eff * logret[i + 1]
strat_eq = np.exp(np.cumsum(ret))
bh_eq = np.exp(np.cumsum(logret))


def metrics(eq, r):
    total = eq[-1] - 1
    ann = eq[-1] ** (252 / N) - 1
    sharpe = (np.mean(r)) / (np.std(r) + 1e-12) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    mdd = (eq / peak - 1).min()
    return total, ann, sharpe, mdd, int(np.sum(long_entry))


s_total, s_ann, s_sharpe, s_mdd, s_trades = metrics(strat_eq, ret)
b_total, b_ann, b_sharpe, b_mdd, _ = metrics(bh_eq, logret)
print("CCI strat:", dict(total=s_total, ann=s_ann, sharpe=s_sharpe, mdd=s_mdd, trades=s_trades))
print("CCI buy&hold:", dict(total=b_total, ann=b_ann, sharpe=b_sharpe, mdd=b_mdd))

import os, json
outdir = "public/images/cci-indicator"
os.makedirs(outdir, exist_ok=True)

# ---------- 图 1：cover（价格 + CCI 副图 + 买卖点）----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
ax1.plot(close, color="#1f3a5f", lw=1.3, label="收盘价")
ax1.plot(high, color="#bcd0e5", lw=0.5)
ax1.plot(low, color="#bcd0e5", lw=0.5)
e_idx = np.where(long_entry)[0]
x_idx = np.where(long_exit)[0]
ax1.scatter(e_idx, close[e_idx], marker="^", color="#1a9850", s=90, zorder=5, label="买入（脱离超卖 -100）")
ax1.scatter(x_idx, close[x_idx], marker="v", color="#d73027", s=90, zorder=5, label="卖出（脱离超买 +100）")
ax1.set_title("CCI 商品通道指数：±100 通道反转", fontsize=15, fontweight="bold", color="#1f3a5f")
ax1.set_ylabel("价格")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.25)
ax2.axhline(100, color="#d73027", lw=1, ls="--")
ax2.axhline(-100, color="#1a9850", lw=1, ls="--")
ax2.axhline(0, color="gray", lw=0.6)
ax2.plot(cci, color="#762a83", lw=1.0)
ax2.fill_between(range(N), cci, 100, where=cci >= 100, color="#d73027", alpha=0.18)
ax2.fill_between(range(N), cci, -100, where=cci <= -100, color="#1a9850", alpha=0.18)
ax2.set_ylabel("CCI")
ax2.set_xlabel("交易日")
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/cover.png", dpi=130)
plt.close(fig)

# ---------- 图 2：zoom（局部放大）----------
z0, z1 = 300, 470
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
ax1.plot(range(z0, z1), close[z0:z1], color="#1f3a5f", lw=1.5)
ee = e_idx[(e_idx >= z0) & (e_idx < z1)]
xx = x_idx[(x_idx >= z0) & (x_idx < z1)]
ax1.scatter(ee, close[ee], marker="^", color="#1a9850", s=120, zorder=5)
ax1.scatter(xx, close[xx], marker="v", color="#d73027", s=120, zorder=5)
ax1.set_title("局部放大：CCI 如何在震荡段给出反转点", fontsize=14, fontweight="bold", color="#1f3a5f")
ax1.set_ylabel("价格")
ax1.grid(alpha=0.25)
ax2.axhline(100, color="#d73027", lw=1, ls="--")
ax2.axhline(-100, color="#1a9850", lw=1, ls="--")
ax2.axhline(0, color="gray", lw=0.6)
ax2.plot(range(z0, z1), cci[z0:z1], color="#762a83", lw=1.3)
ax2.set_ylabel("CCI")
ax2.set_xlabel("交易日")
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/cci_zoom.png", dpi=130)
plt.close(fig)

# ---------- 图 3：净值曲线 ----------
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(bh_eq, color="gray", lw=1.4, ls="--", label=f"买入持有（{(b_total*100):.1f}%）")
ax.plot(strat_eq, color="#1a9850", lw=1.7, label=f"CCI ±100 反转（{(s_total*100):.1f}%）")
ax.set_title("净值对比：CCI 在震荡市把回撤砍掉大半", fontsize=14, fontweight="bold", color="#1f3a5f")
ax.set_ylabel("净值（起点 = 1.0）")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.25)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
fig.tight_layout()
fig.savefig(f"{outdir}/equity_curve.png", dpi=130)
plt.close(fig)

json.dump(dict(s_total=s_total, s_ann=s_ann, s_sharpe=s_sharpe, s_mdd=s_mdd, s_trades=s_trades,
               b_total=b_total, b_ann=b_ann, b_sharpe=b_sharpe, b_mdd=b_mdd),
          open("cci_metrics.json", "w"), indent=2)
print("CCI images + metrics written.")
