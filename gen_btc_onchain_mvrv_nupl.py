#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比特币链上 MVRV 与 NUPL 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 用带「泡沫-崩盘」 episodess 的价格路径模拟 BTC 日线(约 1460 日 / 4 年)
  * realized_price 用 EMA 近似(每天 f 比例币被移动, 其成本基础重置为当日价)
  * MVRV_t   = price_t / realized_price_t
  * NUPL_t   = 1 - 1/MVRV_t   (两者数学等价, NUPL=1-已实现比市值)
  * 择时: MVRV<1 建仓(价格低于平均成本=投降/低估), MVRV>3.5 清仓(狂热)
  * 检验: MVRV 对未来 30 日收益的横截面负斜率、策略净值 vs 买入持有
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

SLUG = "btc-onchain-mvrv-nupl"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
T = 1460                              # 约 4 年日线
days = np.arange(T)

# ---------- BTC 价格路径: 趋势 + 泡沫/崩盘 episodes ----------
drift = 0.0011                        # 日漂移(长期上行)
vol = 0.035
price = np.zeros(T)
price[0] = 16000.0
shock = rng.normal(0, vol, T)
# 注入 2 轮泡沫-崩盘: 缓慢拉抬后急跌
bubble_start = [300, 950]
for bs in bubble_start:
    for t in range(bs, min(bs + 180, T)):
        prog = (t - bs) / 180.0
        if prog < 0.8:
            shock[t] += 0.010          # 泡沫期加速上涨
        else:
            shock[t] -= 0.022          # 崩盘期急跌
for t in range(1, T):
    price[t] = price[t - 1] * np.exp(drift + shock[t] - 0.5 * vol**2)
    price[t] = max(price[t], 3000.0)

# ---------- realized_price: EMA 近似(每日 f 比例币移动, 成本重置为当日价) ----------
f = 0.012                             # 每日链上活跃重置比例
realized = np.zeros(T)
realized[0] = price[0]
for t in range(1, T):
    realized[t] = realized[t - 1] * (1 - f) + price[t] * f

mvrv = price / realized               # 市值 / 已实现市值
nupl = 1.0 - 1.0 / mvrv               # = (市值-已实现)/市值

# ---------- 择时策略: MVRV<1 建仓, MVRV>3.5 清仓 ----------
pos = np.zeros(T)
for t in range(1, T):
    if mvrv[t] < 1.0:
        pos[t] = 1.0
    elif mvrv[t] > 3.5:
        pos[t] = 0.0
    else:
        pos[t] = pos[t - 1]           # 区间内维持上一状态
ret = np.diff(np.log(price))          # 日收益
strat_ret = pos[1:] * ret
strat_nav = np.cumprod(1 + strat_ret)
hold_nav = np.cumprod(1 + ret)

# ---------- 图 1: 价格 + MVRV 双轴 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days, price / 1000.0, color=C["net"], lw=1.6, label="BTC 价格 (千 USD)")
ax.set_ylabel("BTC 价格 (千 USD)", color=C["net"])
ax.tick_params(axis="y", labelcolor=C["net"])
ax2 = ax.twinx()
ax2.plot(days, mvrv, color=C["orange"], lw=1.8, label="MVRV")
ax2.axhline(1.0, color=C["green"], ls="--", lw=1, alpha=0.8)
ax2.axhline(3.5, color=C["red"], ls="--", lw=1, alpha=0.8)
ax2.set_ylabel("MVRV", color=C["orange"])
ax2.tick_params(axis="y", labelcolor=C["orange"])
ax.set_title("BTC 价格与 MVRV: 泡沫期 MVRV 飙升、崩盘前已见顶", fontsize=12)
ax.set_xlabel("交易日")
fig.tight_layout(); fig.savefig(f"{OUT}/price_mvrv.png"); plt.close(fig)

# ---------- 图 2: MVRV 分位/区间带 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days, mvrv, color=C["orange"], lw=1.6)
ax.axhspan(0, 1.0, color=C["green"], alpha=0.12, label="低估区 MVRV<1 (建仓)")
ax.axhspan(1.0, 3.5, color=C["net"], alpha=0.06, label="中性区 1–3.5")
ax.axhspan(3.5, mvrv.max() + 0.5, color=C["red"], alpha=0.12, label="狂热区 MVRV>3.5 (清仓)")
ax.axhline(1.0, color=C["green"], ls="--", lw=1)
ax.axhline(3.5, color=C["red"], ls="--", lw=1)
ax.set_title("MVRV 区间带: 价格相对平均成本基础的偏离", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("MVRV")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/mvrv_zones.png"); plt.close(fig)

# ---------- 图 3: NUPL 彩色分区 ----------
def nupl_color(v):
    if v < 0.0:      return C["red"]
    elif v < 0.25:   return C["orange"]
    elif v < 0.5:    return C["green"]
    elif v < 0.75:   return C["net"]
    else:            return C["purple"]
fig, ax = plt.subplots(figsize=(9.5, 4.3))
for t in range(1, T):
    ax.plot([days[t - 1], days[t]], [nupl[t - 1], nupl[t]],
            color=nupl_color(nupl[t]), lw=1.6)
ax.axhline(0.0, color=C["line"], lw=1)
ax.axhline(0.25, color=C["grid"], ls=":", lw=1)
ax.axhline(0.5, color=C["grid"], ls=":", lw=1)
ax.axhline(0.75, color=C["grid"], ls=":", lw=1)
ax.set_title("NUPL 分区: 投降(<0)→希望(0–.25)→乐观(.25–.5)→信念(.5–.75)→狂热(>.75)", fontsize=11)
ax.set_xlabel("交易日"); ax.set_ylabel("NUPL")
ax.grid(alpha=0.25, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/nupl_zones.png"); plt.close(fig)

# ---------- 图 4: 策略净值 vs 买入持有 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days[1:], strat_nav, color=C["green"], lw=2, label=f"MVRV 择时 (终值 {strat_nav[-1]:.1f}x)")
ax.plot(days[1:], hold_nav, color=C["net"], lw=1.6, label=f"买入持有 (终值 {hold_nav[-1]:.1f}x)")
ax.set_title("MVRV 成本基础择时: 躲过崩盘、净值更稳健", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("净值 (起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/strategy_nav.png"); plt.close(fig)

# ---------- 图 5: MVRV 对未来 30 日收益的横截面回归 ----------
H = 30
fwd = np.full(T, np.nan)
for t in range(T - H):
    fwd[t] = np.prod(1 + ret[t:t + H]) - 1
mask = ~np.isnan(fwd)
xs = mvrv[mask]
ys = fwd[mask] * 100
X = np.column_stack([np.ones(len(xs)), xs])
coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
resid = ys - X @ coef
dof = len(xs) - 2
se = np.sqrt((resid @ resid) / dof) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
t_slope = coef[1] / se[1]
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.scatter(xs, ys, s=14, alpha=0.4, color=C["net"])
ax.plot([xs.min(), xs.max()], [coef[0] + coef[1] * xs.min(), coef[0] + coef[1] * xs.max()],
        color=C["red"], lw=2, label=f"斜率 t={t_slope:.1f}")
ax.axvline(1.0, color=C["green"], ls="--", lw=1)
ax.axvline(3.5, color=C["red"], ls="--", lw=1)
ax.set_title("MVRV 越高, 未来 30 日收益越低 (负斜率, 择时逻辑成立)", fontsize=11)
ax.set_xlabel("MVRV (当日)"); ax.set_ylabel("未来 30 日收益 (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/forward_regression.png"); plt.close(fig)

print("IMAGES_WRITTEN:", sorted(os.listdir(OUT)))
print(f"MVRV 均值={mvrv.mean():.2f} 中位数={np.median(mvrv):.2f} 峰={mvrv.max():.2f}")
print(f"NUPL 均值={nupl.mean():.2f} 最低={nupl.min():.2f} 最高={nupl.max():.2f}")
print(f"择时终值={strat_nav[-1]:.2f}x  买入持有终值={hold_nav[-1]:.2f}x")
print(f"前向回归斜率 t={t_slope:.2f}")
