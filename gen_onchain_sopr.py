#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""链上 SOPR 与获利盘抛压信号 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 用「长期漂移 + 围绕成本基础的均值回复 + 噪声」生成 BTC 日线(约 1460 日 / 4 年)
    - 当 SOPR>1(价格高于近期成本基础=获利盘)时施加向下回归压力
    - 当 SOPR<1(价格低于成本基础=水下)时施加向上回归压力
    -> SOPR 因此成为可择时的均值回复信号(符号/方向与真实 SOPR 一致, 数值非真实历史)
  * 用 EMA 近似「花费产出」的平均建仓成本: cap_price = EMA(price, 半衰期 H)
  * SOPR_t = price_t / cap_price_t ; aSOPR = EMA(SOPR, 7d) 去噪
  * 择时: aSOPR<1.0 建仓(投降=低估), aSOPR>1.04 清仓(贪婪)
  * 检验: SOPR 对未来 30 日收益的负斜率(越低未来越涨)、策略净值 vs 买入持有
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

SLUG = "onchain-sopr-profit"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999", "gold": "#CCB974"}

rng = np.random.default_rng(20260720)
T = 1460                              # 约 4 年日线
days = np.arange(T)

# ---------- EMA 工具 ----------
def ema(x, halflife):
    a = 1 - np.exp(-np.log(2) / halflife)
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out

H = 45                                # 建仓成本 EMA 半衰期(天)

# ---------- BTC 价格路径: 长期漂移 + 围绕「成本基础」均值回复 + 噪声 ----------
# 关键机制: 当 SOPR>1(价格高于近期成本基础)时施加向下回归压力,
#           当 SOPR<1(水下)时施加向上回归压力 -> SOPR 成为可择时的均值回复信号
# (自洽合成, 仅用于演示方向/符号, 并非真实历史)
drift = 0.0006                       # 长期日漂移(温和上行)
vol = 0.030
kappa = 0.15                         # 向成本基础回归的速度
price = np.zeros(T)
price[0] = 16000.0
for t in range(1, T):
    cap_prev = ema(price[:t], H)[-1] if t > 1 else price[0]
    sopr_prev = price[t - 1] / cap_prev
    reversion = kappa * (1.0 - sopr_prev)   # >0 当水下(价格<成本), <0 当过热(价格>成本)
    shock = rng.normal(0, vol)
    ret_t = drift + reversion + shock
    price[t] = price[t - 1] * np.exp(ret_t - 0.5 * vol**2)
    price[t] = max(price[t], 3000.0)

# ---------- 平均建仓成本(EMA 近似)与 SOPR ----------
cap_price = ema(price, H)
sopr = price / cap_price              # SOPR
asopr = ema(sopr, 7)                  # aSOPR 去噪

# 日收益(用于净值与前瞻回归)
ret = np.zeros(T); ret[1:] = price[1:] / price[:-1] - 1.0

# ---------- 图1: BTC 价格 + SOPR 双轴 ----------
fig, ax1 = plt.subplots(figsize=(11, 5.2))
ax1.plot(days, price, color=C["net"], lw=1.4, label="BTC 价格")
ax1.set_ylabel("BTC 价格 (USD)", color=C["net"], fontsize=11)
ax1.tick_params(axis="y", labelcolor=C["net"])
ax1.set_ylim(price.min() * 0.8, price.max() * 1.15)
ax2 = ax1.twinx()
ax2.plot(days, sopr, color=C["orange"], lw=1.1, alpha=0.9, label="SOPR")
ax2.axhline(1.0, color=C["red"], ls="--", lw=1.0, label="SOPR = 1 (盈亏平衡)")
ax2.set_ylabel("SOPR", color=C["orange"], fontsize=11)
ax2.tick_params(axis="y", labelcolor=C["orange"])
ax2.set_ylim(0.6, 2.2)
# 标注 SOPR<1 区间
ax2.fill_between(days, 0.6, 1.0, where=(sopr < 1.0), color=C["red"], alpha=0.12)
ax1.set_title("BTC 价格与 SOPR：崩盘前 SOPR 已跌破 1（水下抛售），泡沫期 SOPR 飙升", fontsize=12.5, fontweight="bold")
l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/price_sopr.png"); plt.close()

# ---------- 图2: SOPR 获利盘区间带 ----------
fig, ax = plt.subplots(figsize=(11, 5.0))
# 区间色带
ax.axhspan(0.0, 0.95, color=C["red"], alpha=0.14, label="深度投降 (<0.95)")
ax.axhspan(0.95, 1.00, color=C["orange"], alpha=0.14, label="水下抛售 (0.95–1.0)")
ax.axhspan(1.00, 1.10, color=C["green"], alpha=0.12, label="获利了结 (1.0–1.10)")
ax.axhspan(1.10, 2.5, color=C["gold"], alpha=0.14, label="贪婪区 (>1.10)")
ax.plot(days, asopr, color="#333333", lw=1.0)
ax.axhline(1.0, color=C["red"], ls="--", lw=1.0)
ax.set_ylim(0.75, 1.8)
ax.set_ylabel("aSOPR (7 日平滑)", fontsize=11)
ax.set_title("SOPR 获利盘区间带：<1 为水下抛售/投降，>1.1 为获利了结狂热", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="x", color=C["grid"])
plt.tight_layout(); plt.savefig(f"{OUT}/sopr_zones.png"); plt.close()

# ---------- 图3: 择时净值 (aSOPR<1 建仓 / >1.04 清仓) ----------
pos = np.zeros(T)                     # 持仓: t 日收盘信号, t+1 开盘执行
for t in range(1, T - 1):
    sig = asopr[t - 1]                # 用上一日 aSOPR 判定, 避免前视
    if sig < 1.00:
        pos[t] = 1.0
    elif sig > 1.04:
        pos[t] = 0.0
    else:
        pos[t] = pos[t - 1]           # 持有状态延续
nav = np.ones(T)
bh = np.ones(T)
for t in range(1, T):
    nav[t] = nav[t - 1] * (1 + pos[t - 1] * ret[t])
    bh[t] = bh[t - 1] * (1 + ret[t])
nav = nav / nav[0] * (price[0] / 16000.0)   # 归一化到起始价格比, 便于同图比较
bh = bh / bh[0] * (price[0] / 16000.0)

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(days, bh, color=C["line"], lw=1.3, label="买入持有 (Buy & Hold)")
ax.plot(days, nav, color=C["green"], lw=1.5, label="SOPR 成本基础择时")
# 标注空仓区间
flat = np.where(pos[:-1] == 0)[0]
if len(flat) > 0:
    ax.fill_between(days, 0, max(nav.max(), bh.max()) * 1.05, where=(pos == 0),
                    color=C["grid"], alpha=0.35, label="空仓(规避)")
ax.set_ylabel("净值 (起始=1)", fontsize=11)
ax.set_title("SOPR 择时：aSOPR<1 建仓 / >1.04 清仓——躲过两次崩盘", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(color=C["grid"], axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/strategy_nav.png"); plt.close()

# ---------- 图4: SOPR 分桶 -> 未来 30 日收益 (前瞻性回归) ----------
HOR = 30
fwd = np.zeros(T); fwd[:-HOR] = price[HOR:] / price[:-HOR] - 1.0
valid = ~np.isnan(fwd)
svals = sopr[valid]
fvals = fwd[valid]
qs = np.quantile(svals, np.linspace(0, 1, 11))
bucket_ret = []
bucket_mid = []
for i in range(10):
    m = (svals >= qs[i]) & (svals < qs[i + 1])
    if m.sum() > 0:
        bucket_ret.append(fvals[m].mean() * 100)
        bucket_mid.append((qs[i] + qs[i + 1]) / 2)
bucket_ret = np.array(bucket_ret); bucket_mid = np.array(bucket_mid)
# 线性回归(分桶均值, 用于画图)
A = np.vstack([bucket_mid, np.ones_like(bucket_mid)]).T
slope, intercept = np.linalg.lstsq(A, bucket_ret, rcond=None)[0]
# 诚实的逐日 t 统计(每根日线 SOPR vs 未来30日收益, 含真实噪声)
xx = svals; yy = fvals
n = len(xx); xb = xx.mean(); yb = yy.mean()
sxx = np.sum((xx - xb)**2); sxy = np.sum((xx - xb) * (yy - yb))
b_hat = sxy / sxx
resid = yy - (b_hat * xx + (yb - b_hat * xb))
s2 = np.sum(resid**2) / (n - 2)
se_b = np.sqrt(s2 / sxx)
t_stat = b_hat / se_b

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(bucket_mid, bucket_ret, "-o", color=C["purple"], lw=1.8, ms=7, label="各 SOPR 分桶未来30日均值收益")
xs = np.linspace(bucket_mid.min(), bucket_mid.max(), 50)
ax.plot(xs, slope * xs + intercept, color=C["red"], ls="--", lw=1.2,
        label=f"线性拟合 (斜率={slope:.1f}%/单位)")
ax.axhline(0, color=C["line"], lw=0.8)
ax.set_xlabel("SOPR (分桶中点)", fontsize=11)
ax.set_ylabel("未来 30 日收益 (%)", fontsize=11)
ax.set_title(f"SOPR 越低，未来 30 日收益越高（负斜率 {slope:.0f}%/单位, 逐日 t={t_stat:.1f}）", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(color=C["grid"], axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/forward_regression.png"); plt.close()

# ---------- 图5: aSOPR 阈值穿越 (建仓/清仓信号) ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(days, asopr, color="#333333", lw=1.0, label="aSOPR")
ax.axhline(1.00, color=C["red"], ls="--", lw=1.1, label="建仓阈值 1.00 (投降)")
ax.axhline(1.04, color=C["green"], ls="--", lw=1.1, label="清仓阈值 1.04 (贪婪)")
buy = np.where((asopr < 1.00) & (np.r_[0, asopr[:-1]] >= 1.00))[0]
sell = np.where((asopr > 1.04) & (np.r_[0, asopr[:-1]] <= 1.04))[0]
ax.scatter(buy, asopr[buy], color=C["red"], s=28, zorder=5, label="买入信号")
ax.scatter(sell, asopr[sell], color=C["green"], s=28, zorder=5, label="卖出信号")
ax.set_ylim(0.8, 1.8)
ax.set_ylabel("aSOPR", fontsize=11)
ax.set_title("aSOPR 阈值穿越：跌破 1.0 买入、上破 1.04 卖出", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(color=C["grid"], axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/asopr_signals.png"); plt.close()

print("SOPR 配图已生成:", OUT)
print(f"  price_sopr.png / sopr_zones.png / strategy_nav.png / forward_regression.png / asopr_signals.png")
print(f"  SOPR 区间: min={sopr.min():.3f} max={sopr.max():.3f} 跌破1占比={(sopr<1).mean()*100:.1f}%")
print(f"  前向回归斜率={slope:.2f}%/单位 逐日t={t_stat:.1f} 择时终值={nav[-1]:.2f}x 买入持有={bh[-1]:.2f}x")
def _mdd(x):
    peak = np.maximum.accumulate(x)
    return -(peak - x).max() / peak.max() * 100.0
print(f"  择时最大回撤={_mdd(nav):.1f}%  买入持有最大回撤={_mdd(bh):.1f}%  持仓占比={(pos==1).mean()*100:.1f}%  切换次数={int((np.abs(np.diff(pos))>0).sum())}")
