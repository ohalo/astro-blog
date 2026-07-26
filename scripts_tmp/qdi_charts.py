#!/usr/bin/env python3
"""Charts for quoted depth imbalance article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/quoted-depth-imbalance"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

# ---------------- Simulate an LOB with depth-dependent price moves ----------------
# Mid price moves: prob of uptick increases with imbalance I = (B - A) / (B + A)
def simulate_lob(T, alpha=0.35, tick=0.01, rng=None, persist=0.9):
    """B, A follow log-AR(1); mid moves up with prob sigmoid(alpha_eff * I)."""
    rng = rng or np.random.default_rng(0)
    logB = np.empty(T); logA = np.empty(T)
    logB[0] = np.log(5000); logA[0] = np.log(5000)
    for t in range(1, T):
        logB[t] = persist*logB[t-1] + (1-persist)*np.log(5000) + rng.normal(0, 0.25)
        logA[t] = persist*logA[t-1] + (1-persist)*np.log(5000) + rng.normal(0, 0.25)
    B = np.exp(logB); A = np.exp(logA)
    I = (B - A) / (B + A)
    mid = np.empty(T); mid[0] = 10.0
    dirs = np.zeros(T, dtype=int)
    for t in range(T - 1):
        p_up = 1/(1 + np.exp(-6*alpha*I[t]))
        move = rng.random()
        if move < 0.25:  # only 25% of steps have a mid change
            d = 1 if rng.random() < p_up else -1
        else:
            d = 0
        dirs[t+1] = d
        mid[t+1] = mid[t] + d*tick
    return B, A, I, mid, dirs

T = 20000
B, A, I, mid, dirs = simulate_lob(T, rng=rng)

# ============ Chart 1: intuition — snapshot of book + imbalance ============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
# left: two example book snapshots
labels = ["买一量 5,000\n卖一量 1,200", "买一量 1,300\n卖一量 6,400"]
b_ex = [5000, 1300]; a_ex = [1200, 6400]
x = np.arange(2)
w = 0.35
axes[0].bar(x - w/2, b_ex, w, color="#2ca02c", alpha=0.8, label="买一挂单量 (bid depth)")
axes[0].bar(x + w/2, a_ex, w, color="#d62728", alpha=0.8, label="卖一挂单量 (ask depth)")
for i in range(2):
    imb = (b_ex[i]-a_ex[i])/(b_ex[i]+a_ex[i])
    axes[0].text(i, max(b_ex[i], a_ex[i])+250, f"失衡 I = {imb:+.2f}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(["盘口 A：买方压制", "盘口 B：卖方压制"])
axes[0].set_ylabel("挂单量（股）")
axes[0].set_title("同样的价差，两种完全不同的盘口")
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 7600)

# right: distribution of I in simulation
axes[1].hist(I, bins=60, color="#1f77b4", alpha=0.75)
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_title("模拟 LOB 中深度失衡 I 的分布（2 万个快照）")
axes[1].set_xlabel("I = (B − A) / (B + A)")
plt.tight_layout()
plt.savefig(f"{OUT}/qdi-book-snapshot.png", dpi=130)
plt.close()
print("chart1 done")

# ============ Chart 2: conditional prob of uptick vs imbalance decile ============
# use only steps where a move happened
move_mask = dirs[1:] != 0
I_pre = I[:-1][move_mask]
up = (dirs[1:][move_mask] > 0).astype(float)
deciles = np.quantile(I_pre, np.linspace(0, 1, 11))
centers, probs, counts = [], [], []
for i in range(10):
    m = (I_pre >= deciles[i]) & (I_pre <= deciles[i+1])
    centers.append(I_pre[m].mean())
    probs.append(up[m].mean())
    counts.append(m.sum())

fig, ax = plt.subplots(figsize=(7.8, 5.2))
ax.plot(centers, probs, "o-", color="#1f77b4", lw=2, ms=7)
ax.axhline(0.5, color="gray", ls=":", label="无信息基准 50%")
ax.set_xlabel("前一时刻深度失衡 I（十分位组均值）")
ax.set_ylabel("下一次中间价变动为上涨的概率")
ax.set_title(f"深度失衡对下一跳方向的预测力（{int(move_mask.sum())} 次价格变动）")
for cx, py, n in zip(centers, probs, counts):
    ax.annotate(f"{py:.0%}", (cx, py), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/qdi-uptick-prob.png", dpi=130)
plt.close()
print("chart2 done", probs[0], probs[-1])

# ============ Chart 3: simple strategy backtest w/ costs ============
# strategy: if I > thr -> long 1 tick target next K steps; if I < -thr -> short
def eval_strategy(I, mid, thr, hold=20, cost_ticks=0.0, tick=0.01):
    T = len(mid)
    pnl = []
    t = 0
    while t < T - hold - 1:
        if I[t] > thr:
            pnl.append((mid[t+hold] - mid[t]) / tick - cost_ticks)
            t += hold
        elif I[t] < -thr:
            pnl.append((mid[t] - mid[t+hold]) / tick - cost_ticks)
            t += hold
        else:
            t += 1
    return np.array(pnl)

thrs = [0.2, 0.4, 0.6, 0.8]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(thrs)))
for thr, col in zip(thrs, colors):
    pnl0 = eval_strategy(I, mid, thr, cost_ticks=0.0)
    axes[0].plot(np.cumsum(pnl0), color=col, label=f"|I|>{thr}（{len(pnl0)} 笔，均值 {pnl0.mean():.3f} tick）")
axes[0].set_title("无成本：累计盈亏（单位：tick）")
axes[0].set_xlabel("交易序号"); axes[0].set_ylabel("累计 tick")
axes[0].legend(fontsize=8)

for thr, col in zip(thrs, colors):
    pnl1 = eval_strategy(I, mid, thr, cost_ticks=1.0)
    axes[1].plot(np.cumsum(pnl1), color=col, label=f"|I|>{thr}（均值 {pnl1.mean():.3f} tick）")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("扣 1 个 tick 成本（跨价差成交）后")
axes[1].set_xlabel("交易序号")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/qdi-strategy-cost.png", dpi=130)
plt.close()
print("chart3 done")

# ============ Chart 4: signal decay ============
horizons = [1, 5, 10, 20, 50, 100, 200]
corrs = []
for h in horizons:
    fwd = mid[h:] - mid[:-h]
    c = np.corrcoef(I[:-h], fwd)[0, 1]
    corrs.append(c)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(horizons, corrs, "o-", color="#1f77b4", lw=2)
ax.set_xscale("log")
ax.set_xlabel("预测跨度（快照步数，对数轴）")
ax.set_ylabel("I 与未来中间价变动的相关系数")
ax.set_title("信号衰减：深度失衡只在很短的跨度内有预测力")
ax.axhline(0, color="gray", ls=":")
for h, c in zip(horizons, corrs):
    ax.annotate(f"{c:.3f}", (h, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/qdi-signal-decay.png", dpi=130)
plt.close()
print("chart4 done", list(zip(horizons, [round(c,3) for c in corrs])))
