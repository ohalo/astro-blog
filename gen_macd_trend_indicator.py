#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD 趋势指示器：用双均线差分识别动能拐点 配图生成 (4 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 价格 = 分段制度: 长段上涨 / 长段下跌 / 横盘(高噪声扁平), 各持续一段
  * EMA 快线(12) / 慢线(26) / DIF = 快-慢 / DEA = EMA9(DIF) / 柱 = 2*(DIF-DEA)
  * 信号: DIF 上穿 DEA -> 金叉(看多); 下穿 -> 死叉(看空)
  * 回测: 信号确认后次日开盘交易(无 look-ahead): 金叉持有、死叉空仓(转现金)
  * 对照: 买入持有(Buy&Hold)
  * 诚实拆穿: 横盘市 MACD 频繁假金叉/假死叉来回止损; 滞后性(双 EMA 都滞后)
  * 图1: 价格 + 快/慢 EMA + 金叉/死叉标记
  * 图2: DIF / DEA / 红绿柱(动能拐点的经典图)
  * 图3: 净值(策略 vs 买入持有)
  * 图4: 假信号统计——横盘 vs 上涨趋势段的金叉「鞭锯率」(金叉后 15 日内即死叉比例)对比
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "macd-trend-indicator"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "rec": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999",
     "up": "#C44E52", "down": "#55A868"}

def ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

# ---------- 合成价格: 分段制度 ----------
rng = np.random.default_rng(20260719)
N = 1600
# 制度段: 上涨/横盘/下跌/横盘/上涨/下跌 交替
seg = (["up"] * 400 + ["side"] * 150 + ["up"] * 400 +
       ["side"] * 150 + ["down"] * 250 + ["up"] * 300)
seg = (seg + ["up"] * (N - len(seg)))[:N]   # 补齐长度
regime = np.array(seg)
mu = np.where(regime == "up", 0.0013,
      np.where(regime == "down", -0.0015, 0.0))
sigma = np.where(regime == "side", 0.014, 0.005)   # 横盘波动更大=噪声
ret = mu + rng.normal(0, 1, N) * sigma
price = np.cumprod(1 + ret)

# ---------- MACD ----------
FAST, SLOW, SIG = 12, 26, 9
ema_f = ema(price, FAST)
ema_s = ema(price, SLOW)
dif = ema_f - ema_s
dea = ema(dif, SIG)
hist = 2 * (dif - dea)

# ---------- 金叉/死叉 ----------
cross_up = (dif > dea) & (np.r_[False, (dif[:-1] <= dea[:-1])])
cross_dn = (dif < dea) & (np.r_[False, (dif[:-1] >= dea[:-1])])
idx_up = np.where(cross_up)[0]
idx_dn = np.where(cross_dn)[0]

# ---------- 回测: 信号次日开盘交易(无 look-ahead): 用 t-1 收盘信号决定 t 的持仓 ----------
# pos[t] = 截至 t 收盘的状态; 实际在 t+1 开盘按 pos[t] 建仓, 赚取 [t,t+1] 收益
pos = np.zeros(N, dtype=int)
state = 0
for t in range(1, N):
    if t in idx_up:
        state = 1
    elif t in idx_dn:
        state = 0
    pos[t] = state
# 关键: 用 t-1 信号(pos[:-1])吃 t 的收益, 避免同日信号同日执行的前瞻偏差
strat_ret = np.r_[0.0, pos[:-1] * (price[1:] / price[:-1] - 1)]
bh_ret = np.r_[0.0, price[1:] / price[:-1] - 1]
strat_nav = np.cumprod(1 + strat_ret)
bh_nav = np.cumprod(1 + bh_ret)

def metrics(nav):
    r = nav[1:] / nav[:-1] - 1
    cagr = nav[-1] ** (252.0 / (len(nav) - 1)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    return cagr, vol, sharpe, dd

m_s = metrics(strat_nav); m_b = metrics(bh_nav)
print(f"策略 : 年化 {m_s[0]:.2%} 波动 {m_s[1]:.2%} 夏普 {m_s[2]:.2f} 回撤 {m_s[3]:.2%}")
print(f"买入持有: 年化 {m_b[0]:.2%} 波动 {m_b[1]:.2%} 夏普 {m_b[2]:.2f} 回撤 {m_b[3]:.2%}")
print(f"金叉次数 {len(idx_up)}  死叉次数 {len(idx_dn)}")

# ---------- 假信号指纹: 横盘 vs 上涨趋势段的「鞭锯率」(whipsaw) ----------
# 鞭锯: 金叉后 15 交易日内即出现死叉 = 假信号来回止损
def whipsaw_rate(idx):
    dn_sorted = np.sort(idx_dn)
    cnt = 0
    for i in idx:
        # 下一个死叉是否在 15 日内
        nxt = dn_sorted[dn_sorted > i]
        if len(nxt) and (nxt[0] - i) <= 15:
            cnt += 1
    return (cnt / len(idx) if len(idx) else np.nan), len(idx)
up_reg = np.where(regime == "up")[0]
side_reg = np.where(regime == "side")[0]
up_cross = idx_up[np.isin(idx_up, up_reg)]
side_cross = idx_up[np.isin(idx_up, side_reg)]
wr_up, n_up = whipsaw_rate(up_cross)
wr_side, n_side = whipsaw_rate(side_cross)
print(f"金叉鞭锯率: 上涨趋势段 {wr_up:.1%} (n={n_up})  横盘段 {wr_side:.1%} (n={n_side})  [越低越好]")

# ============ 图1: 价格 + EMA + 叉 ============
fig, ax = plt.subplots(figsize=(13, 5.5))
tt = np.arange(N)
ax.plot(tt, price, color=C["line"], lw=1.0, label="价格", zorder=1)
ax.plot(tt, ema_f, color=C["orange"], lw=1.4, label=f"EMA{FAST} 快")
ax.plot(tt, ema_s, color=C["purple"], lw=1.4, label=f"EMA{SLOW} 慢")
ax.scatter(idx_up, price[idx_up], color=C["up"], marker="^", s=40, zorder=3, label="金叉")
ax.scatter(idx_dn, price[idx_dn], color=C["down"], marker="v", s=40, zorder=3, label="死叉")
ax.set_title("价格 + 快慢 EMA + 金叉/死叉标记", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("价格"); ax.legend(fontsize=8, ncol=4)
ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(f"{OUT}/price_ema_cross.png", dpi=130, bbox_inches="tight")
print("✓ price_ema_cross.png")

# ============ 图2: DIF/DEA/柱 ============
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(tt, dif, color=C["net"], lw=1.4, label="DIF (快-慢)")
ax.plot(tt, dea, color=C["orange"], lw=1.4, label="DEA (DIF的EMA9)")
ax.bar(tt, hist, color=np.where(hist >= 0, C["up"], C["down"]), width=1.0, alpha=0.7, label="MACD 柱 2(DIF-DEA)")
ax.axhline(0, color="k", lw=0.8)
ax.set_title("DIF / DEA / MACD 柱：动能拐点的经典表达", fontsize=13)
ax.set_xlabel("交易日"); ax.legend(fontsize=8, ncol=3)
ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(f"{OUT}/dif_dea_hist.png", dpi=130, bbox_inches="tight")
print("✓ dif_dea_hist.png")

# ============ 图3: 净值对比 ============
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(tt, strat_nav, color=C["green"], lw=2.0, label=f"MACD策略(夏普 {m_s[2]:.2f})")
ax.plot(tt, bh_nav, color=C["line"], lw=1.6, label=f"买入持有(夏普 {m_b[2]:.2f})")
ax.set_title("净值对比：趋势市吃肉、横盘市止损", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("NAV"); ax.legend(fontsize=8)
ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(f"{OUT}/nav_comparison.png", dpi=130, bbox_inches="tight")
print("✓ nav_comparison.png")

# ============ 图4: 金叉鞭锯率 横盘 vs 趋势 ============
fig, ax = plt.subplots(figsize=(7.5, 5))
cats = ["上涨趋势段金叉", "横盘段金叉"]
vals = [wr_up * 100, wr_side * 100]
bars = ax.bar(cats, vals, color=[C["green"], C["down"]], width=0.55)
ax.axhline(50, color="k", ls=":", lw=1, label="50% 基准")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=11)
ax.set_ylim(0, 100); ax.set_ylabel("金叉后 15 日内即死叉的鞭锯率 (%)")
ax.set_title("假信号指纹：横盘段金叉鞭锯率远高于趋势段", fontsize=12)
ax.legend(fontsize=8); ax.grid(color=C["grid"], lw=0.5, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/winrate_sideway.png", dpi=130, bbox_inches="tight")
print("✓ winrate_sideway.png")

np.savez(f"{OUT}/_metrics.npz",
         m_s=m_s, m_b=m_b, n_up=len(idx_up), n_dn=len(idx_dn),
         wr_side=wr_side, wr_up=wr_up, n_side=n_side, n_up_reg=n_up)
print("DONE")
