#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「CTA 多品种趋势跟踪：用 12 个低相关市场的「共同信条」把危机变成利润」生成真实配图 + 可复现指标。

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 12 个市场(股指/国债/商品/外汇)，各有不同波动与缓慢切换的趋势方向
  - 平静期：每个市场有一条持续数周的趋势漂移(±)，Donchian 通道跟随吃趋势
  - 危机段：趋势被强制转向「危机方向」——风险资产急跌、避险资产(债/金)急涨
            → 多品种 CTA 在这段「同时赚钱」，是危机 alpha 的来源
  - 各市场用 Donchian 通道(15 日)做趋势跟随：突破上轨做多、跌破下轨做空
  - 组合：等风险权重(按各市场波动归一化)
  - 对照：60/40 股债组合、买入持有(12 市场等权)

所有数字由文中 Python 真实计算（纯 numpy/matplotlib），随机种子固定。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf",
              "/System/Library/Fonts/PingFang.ttc"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti",
                                "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.bbox": "tight"})

BASE = "/Users/halo/workspace/astro-blog/public/images"
SLUG = "cta-multi-trend"
IMG = os.path.join(BASE, SLUG)
os.makedirs(IMG, exist_ok=True)
BLOG = os.path.join("/Users/halo/workspace/astro-blog/src/content/blog", SLUG)
os.makedirs(BLOG, exist_ok=True)

C = {"cta": "#1E8449", "bh": "#C0392B", "p6040": "#8E44AD",
     "grid": "#DDDDDD", "stress": "#E74C3C", "long": "#27AE60", "short": "#C0392B"}

# ---------------- 模拟参数 ----------------
rng = np.random.default_rng(20260722)
ann = 252
T = ann * 20

# 市场定义: (名称, 年化波动, 危机年化漂移方向系数, 类别)
# crisis_dir: 危机段该市场趋势方向 (+1 涨 / -1 跌)
markets = [
    ("股指A", 0.18, -1, "risk"), ("股指B", 0.20, -1, "risk"),
    ("股指C", 0.19, -1, "risk"),
    ("长久债", 0.07, +1, "safe"), ("中久债", 0.05, +1, "safe"),
    ("黄金", 0.15, +1, "safe"),
    ("原油", 0.30, +1, "com"), ("铜", 0.22, +1, "com"),
    ("农产品", 0.19, +1, "com"), ("天然气", 0.45, +1, "com"),
    ("美元指数", 0.09, +1, "fx"), ("EURUSD", 0.08, -1, "fx"),
]
N = len(markets)
SIG = np.array([m[1] for m in markets])
CRISIS_DIR = np.array([m[2] for m in markets])

# 危机 regime (contiguous episodes, ~1-2 周)
stress = np.zeros(T, dtype=bool)
state = 0
for t in range(T):
    if state == 0:
        state = 1 if rng.random() < 0.0022 else 0
    else:
        state = 0 if rng.random() < 0.03 else 1   # 平均持续 ~33 天(够长, 15 日通道能捕捉)
    stress[t] = bool(state)

# 每个市场的缓慢趋势漂移(年化), 持续数周才切换
TREND_MAG = 0.35                                  # 平静期趋势强度(年化)
trend_annual = np.zeros((N, T))
for i in range(N):
    sgn = 1 if rng.random() < 0.5 else -1
    for t in range(T):
        if rng.random() < 0.012:                  # 平均持续 ~83 天(~3 月)切换
            sgn = -sgn
        trend_annual[i, t] = sgn * TREND_MAG

# 危机段：趋势改为「危机方向」强漂移
CRISIS_MAG = 0.70                                 # 危机趋势强度(年化, 更强更持久)
for i in range(N):
    trend_annual[i, stress] = CRISIS_DIR[i] * CRISIS_MAG

# 日收益 = 趋势漂移 + 特质噪声
rets = np.zeros((N, T))
for i in range(N):
    w = rng.standard_normal(T)
    drift = trend_annual[i] / ann
    vol = SIG[i] / np.sqrt(ann)
    rets[i] = drift + vol * w

# ---------------- Donchian 通道趋势信号 (window=15) ----------------
def donchian_signal(r, window=15):
    pos = np.zeros(T, dtype=float)
    for t in range(window, T-1):
        hi = r[t-window:t].max()
        lo = r[t-window:t].min()
        if r[t] >= hi:
            pos[t] = 1.0
        elif r[t] <= lo:
            pos[t] = -1.0
        else:
            pos[t] = pos[t-1]
    return np.concatenate([[0.0], pos[:-1]])

positions = np.zeros((N, T))
for i in range(N):
    positions[i] = donchian_signal(rets[i], 15)

# 等风险权重
vols = rets.std(axis=1)
w_er = (1.0/vols) / (1.0/vols).sum()
cta_r = (positions * w_er[:, None] * rets).sum(axis=0)
cta_nav = np.cumprod(1 + cta_r)

# 对照
bh_r = rets.mean(axis=0)
bh_nav = np.cumprod(1 + bh_r)
p6040_r = 0.6*rets[0] + 0.4*rets[3]              # 股指A + 长久债
p6040_nav = np.cumprod(1 + p6040_r)

def stats(nav, r):
    yrs = T/ann
    cagr = nav[-1]**(1/yrs) - 1
    vol = r.std()*np.sqrt(ann)
    sharpe = (r.mean()*ann)/vol if vol > 0 else 0
    peak = np.maximum.accumulate(nav)
    mdd = (nav/peak - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd)

s_cta = stats(cta_nav, cta_r)
s_bh = stats(bh_nav, bh_r)
s_6040 = stats(p6040_nav, p6040_r)

# 危机 alpha：逐段 contiguous crisis episode 计算区间收益再聚合
def episode_returns(nav):
    eps = []
    t = 0
    while t < T:
        if stress[t]:
            s0 = t
            while t < T and stress[t]:
                t += 1
            s1 = t
            seg = nav[s0:s1]
            if len(seg) >= 2:
                eps.append(seg[-1]/seg[0] - 1)
        else:
            t += 1
    return np.array(eps)

cta_ep = episode_returns(cta_nav)
p6040_ep = episode_returns(p6040_nav)
bh_ep = episode_returns(bh_nav)
crisis_ret_cta = cta_ep.sum()
crisis_ret_6040 = p6040_ep.sum()
crisis_ret_bh = bh_ep.sum()
avg_ep_len = np.diff(np.where(np.diff(np.concatenate([[0], stress.astype(int), [0]])) != 0)[0])[::2].mean() \
    if stress.sum() else 0

# 各市场平均仓位（看信号是否分散）
avg_pos = positions.mean(axis=1)
# 危机段 vs 平静段 CTA 收益
cta_in_crisis = cta_r[stress].sum()
cta_in_calm = cta_r[~stress].sum()

# ---------------- 图1: 净值曲线 ----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(cta_nav, color=C["cta"], lw=1.9, label="CTA 多品种趋势（等风险）")
ax.plot(p6040_nav, color=C["p6040"], lw=1.5, label="60 / 40 股债")
ax.plot(bh_nav, color=C["bh"], lw=1.3, label="买入持有（12 市场等权）", alpha=0.8)
idx = np.where(stress)[0]
if len(idx):
    ax.axvspan(idx.min(), idx.max(), color=C["stress"], alpha=0.05)
ax.set_title("三策略 20 年净值：CTA 在危机段(红区)逆势上扬", fontsize=12)
ax.set_yscale("log")
ax.set_ylabel("净值（对数）")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "cover.png"))
plt.close(fig)

# ---------------- 图2: 危机段缩放净值 ----------------
fig, ax = plt.subplots(figsize=(9, 4.2))
if len(idx) >= 2:
    s0, s1 = max(0, idx.min()-25), min(T, idx.max()+25)
    tt = np.arange(s0, s1)
    ax.plot(tt, cta_nav[s0:s1]/cta_nav[s0], color=C["cta"], lw=1.9, label="CTA")
    ax.plot(tt, p6040_nav[s0:s1]/p6040_nav[s0], color=C["p6040"], lw=1.6, label="60/40")
    ax.axvspan(idx.min(), idx.max(), color=C["stress"], alpha=0.08, label="危机段")
ax.set_title("危机窗口缩放净值：CTA 逆势上涨，60/40 下跌", fontsize=12)
ax.set_ylabel("相对危机前净值")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "crisis_window.png"))
plt.close(fig)

# ---------------- 图3: 单市场信号示意（股指A 做空 / 长久债 做多） ----------------
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
for ax, mi, mname in [(axes[0], 0, "股指A：危机段趋势转负 → Donchian 做空赚钱"),
                       (axes[1], 3, "长久债：避险涨 → Donchian 做多赚钱")]:
    px = 100*np.cumprod(1 + rets[mi])
    ax.plot(px, color="#34495E", lw=1.2, label=f"{mname}")
    pos = positions[mi]
    longs = np.where(pos > 0)[0]
    shorts = np.where(pos < 0)[0]
    if len(longs):
        ax.scatter(longs, px[longs], color=C["long"], s=10, zorder=3, label="做多")
    if len(shorts):
        ax.scatter(shorts, px[shorts], color=C["short"], s=10, zorder=3, label="做空")
    ax.set_ylabel("价格")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
axes[1].set_xlabel("交易日")
fig.suptitle("Donchian 通道信号：突破上轨做多、跌破下轨做空", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(IMG, "signal_example.png"))
plt.close(fig)

# ---------------- 图4: 危机 alpha 条形 ----------------
labels = ["CTA 多品种", "60 / 40", "买入持有"]
vals = [crisis_ret_cta*100, crisis_ret_6040*100, crisis_ret_bh*100]
fig, ax = plt.subplots(figsize=(9, 4.0))
bars = ax.bar(labels, vals, color=[C["cta"], C["p6040"], C["bh"]])
ax.axhline(0, color="#333", lw=0.8)
ax.set_ylabel("危机段累计收益 (%)")
ax.set_title("危机段收益对比：CTA 逆势赚钱，传统组合下跌", fontsize=12)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_y()+b.get_height() +
            (0.5 if b.get_height() >= 0 else -1.2),
            f"{b.get_height():.0f}%", ha="center", fontsize=10)
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "crisis_alpha.png"))
plt.close(fig)

# ---------------- 输出统计 ----------------
out = dict(
    s_cta=s_cta, s_bh=s_bh, s_6040=s_6040,
    crisis_ret_cta=crisis_ret_cta, crisis_ret_6040=crisis_ret_6040,
    crisis_ret_bh=crisis_ret_bh, avg_ep_len=float(avg_ep_len),
    n_episodes=int(len(cta_ep)), avg_pos=avg_pos.tolist(),
    cta_in_crisis=float(cta_in_crisis), cta_in_calm=float(cta_in_calm),
)
with open(os.path.join(IMG, "stats.json"), "w") as f:
    json.dump(out, f, indent=2)

def pct(x): return f"{x*100:.2f}%"
print("====== CTA 多品种趋势 统计 ======")
print(f"CTA      : CAGR {pct(s_cta['cagr'])}  Vol {pct(s_cta['vol'])}  Sharpe {s_cta['sharpe']:.2f}  MDD {pct(s_cta['mdd'])}")
print(f"60/40    : CAGR {pct(s_6040['cagr'])}  Vol {pct(s_6040['vol'])}  Sharpe {s_6040['sharpe']:.2f}  MDD {pct(s_6040['mdd'])}")
print(f"买入持有  : CAGR {pct(s_bh['cagr'])}  Vol {pct(s_bh['vol'])}  Sharpe {s_bh['sharpe']:.2f}  MDD {pct(s_bh['mdd'])}")
print(f"危机段累计收益: CTA {pct(crisis_ret_cta)}  60/40 {pct(crisis_ret_6040)}  买入持有 {pct(crisis_ret_bh)}")
print(f"危机段CTA日收益合计 {pct(cta_in_crisis)} / 平静段 {pct(cta_in_calm)}")
print(f"危机段数 {len(cta_ep)}  平均长度 {avg_ep_len:.0f} 日")
print("OK")
