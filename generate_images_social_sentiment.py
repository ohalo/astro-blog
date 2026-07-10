# -*- coding: utf-8 -*-
"""为《社交媒体情绪传播与价格发现》生成 3 张配图（合成数据，仅用于示意）。
输出目录: public/images/social-media-sentiment-price-discovery/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/social-media-sentiment-price-discovery"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260710)

# 统一配色
C_S = "#4C78A8"   # 未传播(易感)
C_I = "#F58518"   # 传播中(感染)
C_R = "#54A24B"   # 已消化(移除)
C_LINE = "#333333"

# ---------------------------------------------------------------
# 图1: SIR 风格的情绪传播曲线
# ---------------------------------------------------------------
def sir_sentiment(T=120, beta=0.28, gamma=0.06, I0=0.005):
    N = 1.0
    S = np.zeros(T); I = np.zeros(T); R = np.zeros(T)
    S[0], I[0], R[0] = 1 - I0, I0, 0.0
    dt = 1.0
    for t in range(1, T):
        dS = -beta * S[t - 1] * I[t - 1] * dt
        dI = (beta * S[t - 1] * I[t - 1] - gamma * I[t - 1]) * dt
        dR = gamma * I[t - 1] * dt
        S[t] = max(S[t - 1] + dS, 0)
        I[t] = max(I[t - 1] + dI, 0)
        R[t] = min(R[t - 1] + dR, 1)
    return S, I, R

S, I, R = sir_sentiment()
t = np.arange(len(S))

fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
ax.plot(t, S, color=C_S, lw=2.6, label="未传播 S(t):尚未接触该叙事")
ax.plot(t, I, color=C_I, lw=2.6, label="传播中 I(t):情绪活跃、被反复转发")
ax.plot(t, R, color=C_R, lw=2.6, label="已消化 R(t):话题冷却、价格已反映")
ax.fill_between(t, 0, I, color=C_I, alpha=0.12)
peak = int(np.argmax(I))
ax.axvline(peak, color="red", ls="--", lw=1.4, alpha=0.7)
ax.annotate(f"情绪峰值 ≈ 第 {peak} 天", xy=(peak, I[peak]), xytext=(peak - 35, I[peak] + 0.12),
            fontsize=11, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"))
ax.set_xlabel("时间 (天)", fontsize=12, fontweight="bold")
ax.set_ylabel("人群占比", fontsize=12, fontweight="bold")
ax.set_title("情绪在社交网络中的 SIR 式传播:从话题引爆到冷却", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3, ls="--")
ax.legend(loc="center right", fontsize=10, framealpha=0.92)
fig.tight_layout()
fig.savefig(f"{OUT}/sentiment_propagation_sir.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# 图2: 网络扩散(种子节点激活 -> 邻居逐轮被感染)
# ---------------------------------------------------------------
N = 46
pos = rng.uniform(0, 1, size=(N, 2))
# 半径内连边(随机几何图)
edges = []
r0 = 0.22
for i in range(N):
    for j in range(i + 1, N):
        if np.linalg.norm(pos[i] - pos[j]) < r0:
            edges.append((i, j))
# BFS 激活时间
seed = int(np.argmax([np.linalg.norm(p - 0.5) for p in pos]))  # 选最中心的节点做种子
activated = {seed: 0}
frontier = [seed]
while frontier:
    nxt = []
    for u in frontier:
        for v in [b for a, b in edges if a == u] + [a for a, b in edges if b == u]:
            if v not in activated:
                activated[v] = activated[u] + 1
                nxt.append(v)
    frontier = nxt
maxT = max(activated.values())
# 孤立未激活节点归到最后
for i in range(N):
    if i not in activated:
        activated[i] = maxT + 1

fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
ax = axes[0]
for a, b in edges:
    ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], color="#cccccc", lw=0.8, zorder=1)
for i in range(N):
    tt = activated[i]
    color = plt.cm.viridis(min(tt, maxT + 1) / (maxT + 2))
    size = 260 if i == seed else 90
    ax.scatter(pos[i, 0], pos[i, 1], s=size, color=color, edgecolor="black", lw=0.8, zorder=2)
ax.scatter(pos[seed, 0], pos[seed, 1], s=320, facecolor="none", edgecolor="red", lw=2.2, zorder=3)
ax.set_title("情绪扩散网络(颜色=激活轮次,红圈=种子)", fontsize=13, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
rounds = np.arange(maxT + 2)
cum = [sum(1 for v in activated.values() if v <= k) for k in rounds]
ax.plot(rounds, cum, color=C_I, lw=2.6, marker="o", ms=6)
ax.fill_between(rounds, cum, alpha=0.12, color=C_I)
ax.set_xlabel("扩散轮次", fontsize=12, fontweight="bold")
ax.set_ylabel("累计被激活节点数", fontsize=12, fontweight="bold")
ax.set_title("累计触达随扩散轮次增长", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(f"{OUT}/sentiment_network_diffusion.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# 图3: 情绪对收益的领先-滞后 & 情绪动量策略净值
# ---------------------------------------------------------------
T = 500
sent_base = np.cumsum(rng.normal(0, 1, T))
sent = sent_base - np.mean(sent_base)
ret = np.zeros(T)
# 收益领先情绪 1 天(情绪变化预测次日收益)
for i in range(1, T):
    ret[i] = 0.18 * (sent[i - 1] - sent[i - 2]) + rng.normal(0, 1, 1)[0] * 0.9
ret = ret / np.std(ret)
# 互相关
lags = np.arange(-5, 6)
xcorr = np.array([np.corrcoef(np.roll(sent, lag), ret)[0, 1] for lag in lags])

fig, axes = plt.subplots(2, 1, figsize=(12, 9), facecolor="white")
ax = axes[0]
ax.bar(lags, xcorr, color=[C_I if l < 0 else (C_S if l > 0 else C_R) for l in lags])
best = lags[np.argmax(np.abs(xcorr))]
ax.axvline(best, color="red", ls="--", lw=1.4, alpha=0.8)
ax.annotate(f"最强相关在 lag={best}\n(情绪领先收益 {abs(best)} 天)", xy=(best, xcorr[np.argmax(np.abs(xcorr))]),
            xytext=(best + 0.5, 0.25), fontsize=10, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"))
ax.set_xlabel("滞后阶数 (负=情绪领先, 正=收益领先)", fontsize=11, fontweight="bold")
ax.set_ylabel("互相关系数", fontsize=11, fontweight="bold")
ax.set_title("情绪指数与未来收益的互相关:情绪领先于价格", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, ls="--", axis="y")

ax = axes[1]
signal = np.sign(np.roll(sent, 1) - np.roll(sent, 2))  # 用 t-1 情绪变化做 t 日信号
signal[0] = 0
strat = np.cumprod(1 + 0.5 * signal * ret)
mkt = np.cumprod(1 + ret)
ax.plot(np.arange(T), strat, color=C_I, lw=2.4, label="情绪动量多空策略")
ax.plot(np.arange(T), mkt, color=C_S, lw=2.0, alpha=0.8, label="买入持有(基准)")
ax.set_xlabel("交易日", fontsize=11, fontweight="bold")
ax.set_ylabel("净值 (初始=1)", fontsize=11, fontweight="bold")
ax.set_title("基于情绪动量的多空策略净值(示意)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.92)
ax.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(f"{OUT}/sentiment_lead_lag.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("saved:", os.listdir(OUT))
