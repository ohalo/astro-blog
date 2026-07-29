#!/usr/bin/env python3
"""Tick Imbalance Bars 实验图表生成（修正版）

模拟设计：
- 平静期：买卖各半（带拆单式方向持续），每笔波动小
- 知情期：买方持续占优 p=0.68，且活动更剧烈（每笔波动大）
- TIB 用固定期望失衡阈值 K（教学版），文中另述 EWMA 动态版
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "public/images/tick-imbalance-bars"
os.makedirs(OUT, exist_ok=True)

n_ticks = 120_000
informed = np.zeros(n_ticks, dtype=bool)
informed[50_000:70_000] = True

# tick 方向
p_buy = np.where(informed, 0.68, 0.5)
b = np.where(rng.random(n_ticks) < p_buy, 1, -1)
carry = rng.random(n_ticks) < 0.3   # 拆单：30% 沿用上一笔方向
for i in range(1, n_ticks):
    if carry[i]:
        b[i] = b[i - 1]

# 每笔价格变动：知情期活动更剧烈
tick_sigma = np.where(informed, 0.020, 0.008)
price = 100 + np.cumsum(b * 0.002 + tick_sigma * rng.standard_normal(n_ticks))

# ---------- Tick Imbalance Bars（固定阈值教学版） ----------
def tick_imbalance_bars(b, K=60):
    bars = []
    theta = 0.0
    start = 0
    for i in range(len(b)):
        theta += b[i]
        if abs(theta) >= K:
            bars.append((start, i, i - start + 1, theta))
            theta = 0.0
            start = i + 1
    return bars

tib = tick_imbalance_bars(b, K=60)
tib_ends = np.array([e for (_, e, _, _) in tib])
tib_sizes = np.array([c for (_, _, c, _) in tib])

# 等 tick bar 对照（bar 数对齐）
n_bars = len(tib)
tick_bar_size = n_ticks // n_bars
tb_ends = np.arange(1, n_bars + 1) * tick_bar_size - 1
tb_ends = tb_ends[tb_ends < n_ticks]

# ---------- 图1：价格 + 两种 bar 的采样密度 ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1, 1]})
ax = axes[0]
ax.plot(price, lw=0.6, color="#1f77b4")
ax.axvspan(50_000, 70_000, color="orange", alpha=0.18, label="知情交易区段")
ax.set_ylabel("价格")
ax.set_title("模拟逐笔价格：中段为持续买方失衡的知情交易期")
ax.legend(loc="upper left")

win = 5000
edges = np.arange(0, n_ticks + win, win)
tib_density, _ = np.histogram(tib_ends, bins=edges)
tb_density, _ = np.histogram(tb_ends, bins=edges)
centers = edges[:-1] + win / 2

ax = axes[1]
ax.bar(centers, tib_density, width=win * 0.85, color="#d62728", alpha=0.8)
ax.axvspan(50_000, 70_000, color="orange", alpha=0.18)
ax.set_ylabel("TIB 根数/窗口")
ax.set_title("Tick 失衡 Bar：知情区段采样自动加密")

ax = axes[2]
ax.bar(centers, tb_density, width=win * 0.85, color="#7f7f7f", alpha=0.8)
ax.axvspan(50_000, 70_000, color="orange", alpha=0.18)
ax.set_ylabel("等tick 根数/窗口")
ax.set_xlabel("tick 序号")
ax.set_title("等 tick Bar：对信息爆发完全无感")
plt.tight_layout()
plt.savefig(f"{OUT}/tib-sampling-density.png", dpi=110)
plt.close()

# ---------- 图2：bar 长度分布 ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(tib_sizes, bins=60, color="#d62728", alpha=0.75, log=True)
ax.axvline(np.median(tib_sizes[informed[tib_ends]]), color="orange", ls="--", lw=2,
           label=f"知情期中位长度 = {np.median(tib_sizes[informed[tib_ends]]):.0f} ticks")
ax.axvline(np.median(tib_sizes[~informed[tib_ends]]), color="k", ls="--", lw=2,
           label=f"平静期中位长度 = {np.median(tib_sizes[~informed[tib_ends]]):.0f} ticks")
ax.axvline(tick_bar_size, color="#7f7f7f", ls=":", lw=2,
           label=f"等tick bar 固定 = {tick_bar_size} ticks")
ax.set_xlabel("单根 bar 包含的 tick 数")
ax.set_ylabel("频数（对数刻度）")
ax.set_title("TIB 的 bar 长度自适应：失衡强时 bar 短，均衡时 bar 长")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/tib-bar-size-dist.png", dpi=110)
plt.close()

# ---------- 图3：bar 收益的统计性质对比 ----------
def bar_returns(price, ends):
    p = price[ends]
    return np.diff(np.log(p))

r_tib = bar_returns(price, tib_ends)
r_tb = bar_returns(price, tb_ends)

def ex_kurt(x): return stats.kurtosis(x)
def acf1_sq(x):
    x2 = (x - x.mean()) ** 2
    return np.corrcoef(x2[:-1], x2[1:])[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, r, name, color in [(axes[0], r_tb, "等 tick Bar", "#7f7f7f"),
                           (axes[1], r_tib, "Tick 失衡 Bar", "#d62728")]:
    z = (r - r.mean()) / r.std()
    ax.hist(z, bins=60, density=True, color=color, alpha=0.7)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, stats.norm.pdf(xs), "k--", lw=1, label="标准正态")
    ax.set_title(f"{name}\n超额峰度={ex_kurt(r):.2f}  平方收益ACF(1)={acf1_sq(r):.3f}")
    ax.set_xlim(-4.5, 4.5)
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/tib-return-stats.png", dpi=110)
plt.close()

inf_mask = informed[tib_ends]
print(f"TIB bars: {len(tib)}, tick bar size {tick_bar_size}")
print(f"median size informed={np.median(tib_sizes[inf_mask]):.0f} quiet={np.median(tib_sizes[~inf_mask]):.0f}")
print(f"density informed={tib_density[10:14].mean():.1f}/win quiet={tib_density[:10].mean():.1f}/win")
print(f"kurt tb={ex_kurt(r_tb):.2f} tib={ex_kurt(r_tib):.2f}; acf_sq tb={acf1_sq(r_tb):.3f} tib={acf1_sq(r_tib):.3f}")
print(f"JB tb={stats.jarque_bera(r_tb).statistic:.1f} tib={stats.jarque_bera(r_tib).statistic:.1f}")
