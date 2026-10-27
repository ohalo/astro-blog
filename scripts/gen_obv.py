#!/usr/bin/env python3
"""生成 OBV 文章：合成回测 + 3 张真实图表（CJK 字体）。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

np.random.seed(7)
N = 800

# ---- 合成行情：分段 regime（涨/跌交替），成交量确认趋势（volume confirms move）----
t = np.arange(N)
regime = np.zeros(N)
# 每 ~90 天切换一次方向，制造数段清晰上行/下行
switch_points = [0, 95, 190, 285, 380, 470, 560, 650, 740]
for k in range(len(switch_points) - 1):
    a, b = switch_points[k], switch_points[k + 1]
    direction = 1 if k % 2 == 0 else -1
    regime[a:b] = direction * 0.0011
drift = np.cumsum(regime + np.random.normal(0, 0.0006, N))
noise = np.random.normal(0, 0.0065, N)
logret = np.diff(np.concatenate([[0.0], drift])) + noise
close = 100.0 * np.exp(np.cumsum(logret))
high = close * (1 + np.abs(np.random.normal(0, 0.004, N)) + 0.0015)
low = close * (1 - np.abs(np.random.normal(0, 0.004, N)) - 0.0015)

# 成交量：基础量 + 与「当日绝对波动」正相关（大波动=大成交量，真实市场特征）
base_vol = 1_000_000.0
volume = base_vol * (1.0 + 0.9 * np.abs(logret) / (np.std(logret) + 1e-12)
                     + np.random.normal(0, 0.12, N))
volume = np.abs(volume)


def compute_obv(close, volume):
    ret_sign = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(np.where(ret_sign == 0, 0.0, np.sign(np.diff(close, prepend=close[0])) * volume))
    return obv


obv = compute_obv(close, volume)


def ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


obv_ema = ema(obv, 20)

# 信号：OBV 上穿其 EMA → 吸筹（做多）；下穿 → 派发（平仓）
long_entry = np.zeros(N, dtype=bool)
long_exit = np.zeros(N, dtype=bool)
pos = 0
for i in range(1, N):
    cross_up = (obv[i - 1] <= obv_ema[i - 1]) and (obv[i] > obv_ema[i])
    cross_down = (obv[i - 1] >= obv_ema[i - 1]) and (obv[i] < obv_ema[i])
    if pos == 0 and cross_up:
        long_entry[i] = True
        pos = 1
    elif pos == 1 and cross_down:
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
print("OBV strat:", dict(total=s_total, ann=s_ann, sharpe=s_sharpe, mdd=s_mdd, trades=s_trades))
print("OBV buy&hold:", dict(total=b_total, ann=b_ann, sharpe=b_sharpe, mdd=b_mdd))

import os, json
outdir = "public/images/obv-on-balance-volume"
os.makedirs(outdir, exist_ok=True)

# 归一化用于画图（OBV 数值量级大）
obv_n = obv / base_vol
ema_n = obv_ema / base_vol

# ---------- 图 1：cover（价格 + OBV 副图 + 买卖点）----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
ax1.plot(close, color="#1f3a5f", lw=1.3, label="收盘价")
e_idx = np.where(long_entry)[0]
x_idx = np.where(long_exit)[0]
ax1.scatter(e_idx, close[e_idx], marker="^", color="#1a9850", s=90, zorder=5, label="买入（OBV 上穿信号线）")
ax1.scatter(x_idx, close[x_idx], marker="v", color="#d73027", s=90, zorder=5, label="卖出（OBV 下穿信号线）")
ax1.set_title("OBV 能量潮：成交量确认的吸筹与派发", fontsize=15, fontweight="bold", color="#1f3a5f")
ax1.set_ylabel("价格")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.25)
ax2.axhline(0, color="gray", lw=0.6)
ax2.plot(obv_n, color="#762a83", lw=1.0, label="OBV（归一化）")
ax2.plot(ema_n, color="#e08214", lw=1.2, ls="--", label="OBV 信号线（EMA20）")
ax2.fill_between(range(N), obv_n, ema_n, where=obv_n >= ema_n, color="#1a9850", alpha=0.15)
ax2.fill_between(range(N), obv_n, ema_n, where=obv_n < ema_n, color="#d73027", alpha=0.15)
ax2.set_ylabel("OBV")
ax2.set_xlabel("交易日")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/cover.png", dpi=130)
plt.close(fig)

# ---------- 图 2：zoom ----------
z0, z1 = 280, 420
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
ax1.plot(range(z0, z1), close[z0:z1], color="#1f3a5f", lw=1.5)
ee = e_idx[(e_idx >= z0) & (e_idx < z1)]
xx = x_idx[(x_idx >= z0) & (x_idx < z1)]
ax1.scatter(ee, close[ee], marker="^", color="#1a9850", s=120, zorder=5)
ax1.scatter(xx, close[xx], marker="v", color="#d73027", s=120, zorder=5)
ax1.set_title("局部放大：OBV 如何在趋势段确认方向", fontsize=14, fontweight="bold", color="#1f3a5f")
ax1.set_ylabel("价格")
ax1.grid(alpha=0.25)
ax2.axhline(0, color="gray", lw=0.6)
ax2.plot(range(z0, z1), obv_n[z0:z1], color="#762a83", lw=1.3, label="OBV")
ax2.plot(range(z0, z1), ema_n[z0:z1], color="#e08214", lw=1.2, ls="--", label="信号线")
ax2.set_ylabel("OBV")
ax2.set_xlabel("交易日")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/obv_zoom.png", dpi=130)
plt.close(fig)

# ---------- 图 3：净值曲线 ----------
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(bh_eq, color="gray", lw=1.4, ls="--", label=f"买入持有（{(b_total*100):.1f}%）")
ax.plot(strat_eq, color="#1a9850", lw=1.7, label=f"OBV 信号线交叉（{(s_total*100):.1f}%）")
ax.set_title("净值对比：OBV 靠「成交量确认」躲过下跌段", fontsize=14, fontweight="bold", color="#1f3a5f")
ax.set_ylabel("净值（起点 = 1.0）")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.25)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
fig.tight_layout()
fig.savefig(f"{outdir}/equity_curve.png", dpi=130)
plt.close(fig)

json.dump(dict(s_total=s_total, s_ann=s_ann, s_sharpe=s_sharpe, s_mdd=s_mdd, s_trades=s_trades,
               b_total=b_total, b_ann=b_ann, b_sharpe=b_sharpe, b_mdd=b_mdd),
          open("obv_metrics.json", "w"), indent=2)
print("OBV images + metrics written.")
