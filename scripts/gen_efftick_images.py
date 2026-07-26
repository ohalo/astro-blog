# -*- coding: utf-8 -*-
"""有效 Tick 估计量配图生成（Holden 2009 Effective Tick）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/effective-tick-estimator"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(11)

INCR = np.array([0.01, 0.05, 0.10, 0.25, 0.50])  # 价格增量阶梯

def classify_increment(prices, incr=INCR):
    """把每个价格归到能整除它的最粗增量（分层互斥）"""
    cents = np.round(prices * 100).astype(int)
    steps = np.round(incr * 100).astype(int)
    cls = np.zeros(len(cents), dtype=int)
    for j, s in enumerate(steps):
        cls[cents % s == 0] = j
    return cls

def effective_tick(prices, incr=INCR):
    """Holden (2009) Effective Tick 估计量"""
    cls = classify_increment(prices, incr)
    J = len(incr)
    N = np.array([(cls == j).sum() for j in range(J)], dtype=float)
    F = N / N.sum()
    U = np.zeros(J)
    U[0] = 2 * F[0]
    for j in range(1, J - 1):
        U[j] = 2 * F[j] - F[j - 1]
    U[J - 1] = F[J - 1] - F[J - 2]
    g = np.zeros(J)
    rem = 1.0
    for j in range(J):
        g[j] = min(max(U[j], 0.0), rem)
        rem -= g[j]
    return (g * incr).sum(), g

def simulate_stock(true_spread, n=2000, seed=0, contam=0.0):
    """成交价被打到 true_spread 的价格网格上；contam 为整数位/角位心理聚集污染比例"""
    r = np.random.default_rng(seed)
    mid = 20 + np.cumsum(r.normal(0, 0.03, n))
    mid = np.maximum(mid, 5.0)
    q = np.where(r.random(n) < 0.5, 1, -1)
    raw = mid + q * true_spread / 2
    p = np.round(raw / true_spread) * true_spread
    if contam > 0:
        mask = r.random(n) < contam
        p[mask] = np.round(raw[mask] / 0.25) * 0.25  # 心理聚集到 0.25 网格
    return np.round(p, 2)

def roll_estimator(prices):
    d = np.diff(prices)
    cov = np.cov(d[:-1], d[1:])[0, 1]
    return 2 * np.sqrt(-cov) if cov < 0 else 0.0

# ---------------- 图1：价格聚集直方图（窄价差 vs 宽价差） ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
labels = ["0.01", "0.05", "0.10", "0.25", "0.50"]
for ax, s, title in [
    (axes[0], 0.01, "窄价差股票（真实价差 0.01 元）"),
    (axes[1], 0.25, "宽价差股票（真实价差 0.25 元）"),
]:
    p = simulate_stock(s, n=4000, seed=3)
    cls = classify_increment(p)
    F = np.array([(cls == j).mean() for j in range(len(INCR))])
    colors = ["#1f77b4"] * len(INCR)
    est, g = effective_tick(p)
    ax.bar(labels, F, color=colors, alpha=0.85)
    for i, f in enumerate(F):
        ax.text(i, f + 0.012, f"{f:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("价格落在的最粗增量网格（元）")
    ax.set_ylabel("成交价占比")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{title}\nEffective Tick 估计 = {est:.3f} 元")
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/et-clustering-histogram.png", dpi=130)
plt.close(fig)
print("图1 done")

# ---------------- 图2：估计值 vs 真实价差 + 与 Roll 对比 ----------------
true_spreads = [0.01, 0.05, 0.10, 0.25, 0.50]
n_rep = 40
et_means, et_stds, roll_means, roll_stds = [], [], [], []
for s in true_spreads:
    ets, rolls = [], []
    for k in range(n_rep):
        p = simulate_stock(s, n=2000, seed=1000 + k)
        ets.append(effective_tick(p)[0])
        rolls.append(roll_estimator(p))
    et_means.append(np.mean(ets)); et_stds.append(np.std(ets))
    roll_means.append(np.mean(rolls)); roll_stds.append(np.std(rolls))

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.array(true_spreads)
ax.plot([0, 0.55], [0, 0.55], color="black", lw=1, ls="--", label="45° 线（无偏）")
ax.errorbar(x, et_means, yerr=et_stds, fmt="o-", color="#1f77b4", lw=2,
            capsize=4, label="Effective Tick 估计")
ax.errorbar(x, roll_means, yerr=roll_stds, fmt="s--", color="#d62728", lw=2,
            capsize=4, label="Roll 序列协方差估计")
ax.set_xlabel("真实有效价差（元）")
ax.set_ylabel("估计价差（元，40 次模拟均值 ± 1σ）")
ax.set_title("五档真实价差下：Effective Tick 几乎贴住 45° 线")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/et-vs-true-spread.png", dpi=130)
plt.close(fig)
print("图2 et:", [round(m, 4) for m in et_means])
print("图2 roll:", [round(m, 4) for m in roll_means])

# ---------------- 图3：心理聚集污染的偏差 ----------------
contams = [0.0, 0.1, 0.2, 0.3, 0.5]
true_s = 0.05
biases = []
for c in contams:
    ests = [effective_tick(simulate_stock(true_s, n=2000, seed=2000 + k, contam=c))[0]
            for k in range(n_rep)]
    biases.append(np.mean(ests))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([f"{int(c*100)}%" for c in contams], biases, color="#1f77b4", alpha=0.85, width=0.55)
ax.axhline(true_s, color="#d62728", lw=1.8, ls="--", label=f"真实价差 = {true_s:.2f} 元")
for i, b in enumerate(biases):
    ax.text(i, b + 0.004, f"{b:.3f}", ha="center", fontsize=10)
ax.set_xlabel("整数位心理聚集污染比例（成交价被吸到 0.25 网格）")
ax.set_ylabel("Effective Tick 估计均值（元）")
ax.set_title("聚集不全来自价差时：估计被系统性推高")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/et-clustering-contamination.png", dpi=130)
plt.close(fig)
print("图3 bias:", [round(b, 4) for b in biases])
