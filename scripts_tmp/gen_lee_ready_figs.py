#!/usr/bin/env python3
"""Generate figures for Lee-Ready trade classification article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/lee-ready-trade-classification"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(11)

# ---------------------------------------------------------------
# Simulate a stream of trades with a true direction, quotes, and
# apply Lee-Ready (quote rule + tick rule) to classify. Measure accuracy.
# ---------------------------------------------------------------
N = 50_000
half = 0.05          # half spread
mid = 100.0
# efficient midpoint random walk
mids = mid + np.cumsum(rng.normal(0, 0.01, N))
bid = mids - half
ask = mids + half

# true direction: buyer(+1) or seller(-1) initiated
Q_true = np.where(rng.random(N) < 0.5, 1, -1)

# trade price generation with realistic hard cases:
#  - most trades at the quote (easy for quote rule)
#  - midpoint executions: exactly at mid -> quote rule fails, tick rule guesses
#  - price-improved trades: can land on the WRONG side of mid -> quote rule misleads
mid_exec_prob = 0.12    # executes exactly at midpoint (dark/midpoint match)
improve_prob = 0.18     # meaningful price improvement (can cross mid)
price = np.zeros(N)
for i in range(N):
    u = rng.random()
    if u < mid_exec_prob:
        # exactly at midpoint: genuinely ambiguous for the quote rule
        price[i] = mids[i]
    elif u < mid_exec_prob + improve_prob:
        # price improvement: start at own quote, walk toward (and sometimes past) mid
        base = ask[i] if Q_true[i] == 1 else bid[i]
        imp = rng.uniform(0, half * 1.3)   # can exceed half -> crosses midpoint
        price[i] = base - Q_true[i] * imp
    else:
        # at the quote (occasionally tiny improvement that stays on correct side)
        base = ask[i] if Q_true[i] == 1 else bid[i]
        price[i] = base - Q_true[i] * rng.uniform(0, half * 0.3) * (rng.random() < 0.3)

# ---------------------------------------------------------------
# Lee-Ready classification
# Step 1 (quote rule): above mid -> buy, below mid -> sell
# Step 2 (tick rule): at mid -> use tick test vs last different price
# ---------------------------------------------------------------
Q_est = np.zeros(N)
last_diff_price = price[0]
for i in range(N):
    m = mids[i]
    if price[i] > m + 1e-9:
        Q_est[i] = 1
    elif price[i] < m - 1e-9:
        Q_est[i] = -1
    else:
        # tick rule
        if price[i] > last_diff_price:
            Q_est[i] = 1
        elif price[i] < last_diff_price:
            Q_est[i] = -1
        else:
            Q_est[i] = Q_est[i-1] if i > 0 else 1
    if i > 0 and price[i] != price[i-1]:
        last_diff_price = price[i-1]

accuracy = np.mean(Q_est == Q_true)

# accuracy split by trade location
at_quote = ~((price > bid + 1e-9) & (price < ask - 1e-9))
inside = ~at_quote
acc_at_quote = np.mean(Q_est[at_quote] == Q_true[at_quote])
acc_inside = np.mean(Q_est[inside] == Q_true[inside])

# ---------------------------------------------------------------
# Figure 1: Decision diagram - where trades fall relative to quotes
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
sample = rng.choice(N, 400, replace=False)
rel = (price[sample] - mids[sample]) / half  # -1=bid, +1=ask
correct = Q_est[sample] == Q_true[sample]
colors = np.where(Q_est[sample] == 1, "#c0392b", "#2980b9")
ax.scatter(np.arange(len(sample))[correct], rel[correct], c=colors[correct],
           s=14, alpha=0.7, label="分类正确")
ax.scatter(np.arange(len(sample))[~correct], rel[~correct], c="black",
           marker="x", s=30, label="分类错误")
ax.axhline(1, color="#c0392b", ls="--", lw=1, label="卖价 ask")
ax.axhline(0, color="gray", ls="-", lw=1.2, label="中点 mid (报价规则失效区)")
ax.axhline(-1, color="#2980b9", ls="--", lw=1, label="买价 bid")
ax.fill_between(range(len(sample)), -0.05, 0.05, color="orange", alpha=0.2)
ax.set_ylabel("成交价相对位置 (中点=0)")
ax.set_xlabel("成交序号 (抽样 400 笔)")
ax.set_title(f"Lee-Ready 分类图谱：报价规则处理价内成交，Tick 规则救中点成交\n整体准确率 {accuracy*100:.1f}%")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_ylim(-1.6, 1.6)
plt.tight_layout()
plt.savefig(f"{OUT}/lr-classification-map.png", dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Figure 2: Accuracy by trade location (at-quote vs inside)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cats = ["报价上成交\n(quote rule)", "价差内成交\n(需 tick rule)", "整体"]
accs = [acc_at_quote * 100, acc_inside * 100, accuracy * 100]
bars = ax.bar(cats, accs, color=["#27ae60", "#e67e22", "#34495e"])
for b, v in zip(bars, accs):
    ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.axhline(50, color="red", ls="--", lw=1, label="随机猜测基线 50%")
ax.set_ylabel("分类准确率 (%)")
ax.set_ylim(0, 105)
ax.set_title("准确率的来源：难点全在价差内成交")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/lr-accuracy-breakdown.png", dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Figure 3: Order flow imbalance (OFI) built from classified trades
# vs true OFI - shows misclassification impact on downstream signal
# ---------------------------------------------------------------
win = 200
ofi_true = np.convolve(Q_true, np.ones(win), "valid") / win
ofi_est = np.convolve(Q_est, np.ones(win), "valid") / win
corr = np.corrcoef(ofi_true, ofi_est)[0, 1]

fig, ax = plt.subplots(figsize=(10, 4.4))
xx = np.arange(len(ofi_true))[:2000]
ax.plot(xx, ofi_true[:2000], color="#95a5a6", lw=1.4, label="真实订单流失衡 OFI")
ax.plot(xx, ofi_est[:2000], color="#c0392b", lw=1.0, alpha=0.8, label="Lee-Ready 估计 OFI")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("滚动窗口 (200 笔)")
ax.set_ylabel("订单流失衡 (买占比 - 卖占比)")
ax.set_title(f"下游影响：分类误差如何传导到订单流失衡信号\n估计 OFI 与真实 OFI 相关系数 = {corr:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/lr-order-flow-imbalance.png", dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Figure 4: Sensitivity to quote staleness (lag between trade and quote)
# ---------------------------------------------------------------
lags = [0, 1, 2, 3, 5, 8, 12]
acc_by_lag = []
for lag in lags:
    Qe = np.zeros(N)
    ldp = price[0]
    for i in range(N):
        # use a lagged (stale) midpoint
        j = max(i - lag, 0)
        m = mids[j]
        if price[i] > m + 1e-9:
            Qe[i] = 1
        elif price[i] < m - 1e-9:
            Qe[i] = -1
        else:
            if price[i] > ldp:
                Qe[i] = 1
            elif price[i] < ldp:
                Qe[i] = -1
            else:
                Qe[i] = Qe[i-1] if i > 0 else 1
        if i > 0 and price[i] != price[i-1]:
            ldp = price[i-1]
    acc_by_lag.append(np.mean(Qe == Q_true) * 100)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(lags, acc_by_lag, "o-", color="#8e44ad", lw=2, markersize=7)
for l, a in zip(lags, acc_by_lag):
    ax.text(l, a + 0.4, f"{a:.1f}", ha="center", fontsize=9)
ax.set_xlabel("报价滞后 (成交相对报价陈旧的 bar 数)")
ax.set_ylabel("分类准确率 (%)")
ax.set_title("报价陈旧化的代价：错配的报价快速侵蚀准确率\n(为什么 Lee 与 Ready 建议报价前移 5 秒)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/lr-quote-staleness.png", dpi=130, bbox_inches="tight")
plt.close()

print("Lee-Ready figures done")
print(f"overall acc={accuracy*100:.1f}%, at_quote={acc_at_quote*100:.1f}%, inside={acc_inside*100:.1f}%")
print(f"OFI corr={corr:.3f}")
