#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26e-B1: 信息瓶颈配图 A —— 语义 IB 权衡曲线 + 信息阶梯
DGP: Y 均匀取 6 类, X|Y 为 24 值上的 Dirichlet 分布(部分重叠), 联合分布精确已知。
确定性 IB: 把 24 个 X 值贪心合并成 K 个状态 T, 记录 (I(X;T), I(T;Y)) 权衡曲线。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/information-bottleneck-sequence-prediction"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE, TEAL = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7", "#00cec9"
STATS = {}

Ny, Nx = 6, 24
rng = np.random.default_rng(11)
# 24 个 X 值分成 6 个"语义组"(每组 4 个值)。类 y 的质量集中在组 y 与组 (y+3)%6
# (次模式), 组内再按 Dirichlet 微扰展开 —— 类间共享模式, IB 才能"廉价买预测力"
P_xy = np.zeros((Nx, Ny))
for y in range(Ny):
    a, b = y, (y + 3) % Ny
    px_y = np.zeros(Nx)
    for g, w in ((a, 0.62), (b, 0.30)):
        base = np.full(4, w / 4)
        pert = rng.dirichlet(np.full(4, 1.2)) * w
        px_y[g * 4:(g + 1) * 4] = 0.6 * base + 0.4 * pert
    px_y += 0.02 / Nx
    P_xy[:, y] = px_y / Ny * 1.0
P_xy = P_xy / P_xy.sum()
px = P_xy.sum(axis=1)
pyx = P_xy / px[:, None]

def entropy(v):
    v = v[v > 0]
    return float(-(v * np.log2(v)).sum())

H_Y = entropy(P_xy.sum(axis=0))

def metrics_from_partition(groups):
    pT, H_YT = [], 0.0
    for g in groups:
        w = px[list(g)].sum()
        if w <= 0:
            continue
        pT.append(w)
        pty = P_xy[list(g)].sum(axis=0) / w
        H_YT += w * entropy(pty)
    H_T = entropy(np.array(pT))
    return H_T, H_Y - H_YT  # I(X;T)=H(T) (确定映射), I(T;Y)

# 贪心凝聚: 每步合并使 I(T;Y) 损失最小的状态对
states = [[x] for x in range(Nx) if px[x] > 0]
# 只保留出现概率足够高的值, 其余并入最近邻
states.sort(key=lambda g: -px[g[0]])
keepN = Nx
rest = states[keepN:]
states = states[:keepN]
for g in rest:  # 稀有值并入 KL 最近的保留状态
    p1 = pyx[g[0]]
    d = [entropy(p1 * np.log(np.maximum(p1, 1e-12) / pyx[s[0]])) for s in states]
    states[int(np.argmin(d))].extend(g)
partitions = {len(states): [tuple(s) for s in states]}
while len(states) > 1:
    best, best_loss = None, None
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            merged = states[i] + states[j]
            others = [states[k] for k in range(len(states)) if k not in (i, j)] + [merged]
            _, ity = metrics_from_partition(others)
            loss = -ity
            if best_loss is None or loss > best_loss:
                best_loss, best = loss, (i, j)
    i, j = best
    merged = sorted(states[i] + states[j])
    states = [states[k] for k in range(len(states)) if k not in (i, j)] + [merged]
    partitions[len(states)] = [tuple(s) for s in states]

Ks = sorted(partitions)
HTs, ITYs = [], []
for K in Ks:
    hT, ity = metrics_from_partition(partitions[K])
    HTs.append(hT); ITYs.append(ity)
HTs = np.array(HTs); ITYs = np.array(ITYs)

# ---------- 图1: IB 权衡曲线 ----------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot(HTs, ITYs, "o-", color=PURPLE, lw=1.6, ms=5)
for K, a, b in zip(Ks, HTs, ITYs):
    ax.annotate(f"K={K}", (a, b), textcoords="offset points", xytext=(6, 3), fontsize=7.5, color=GRAY)
full = metrics_from_partition([[x] for x in range(Nx)])
idx6 = Ks.index(6) if 6 in Ks else None
ax.axhline(H_Y, color=GRAY, ls="--", lw=1, alpha=.6, label=f"H(Y) = {H_Y:.2f} bit (无损上限)")
ax.set_xlabel("保留的输入信息  I(X;T)  [bit]")
ax.set_ylabel("预测信息  I(T;Y)  [bit]")
ax.set_title("确定性信息瓶颈权衡曲线：先买'预测力'，再买'输入细节'")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/ib-tradeoff.png"); plt.close(fig)
STATS["H_Y"] = H_Y
STATS["curve"] = {str(K): {"IXT": float(HTs[i]), "ITY": float(ITYs[i])} for i, K in enumerate(Ks)}

# ---------- 图2: 信息阶梯 + 每状态边际价值 ----------
d_ity = np.diff(ITYs)  # K->K+1 新增信息
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].plot(Ks, HTs, "o-", color=BLUE, label="I(X;T) 保留输入")
axes[0].plot(Ks, ITYs, "s-", color=PURPLE, label="I(T;Y) 预测信息")
axes[0].axhline(H_Y, color=GRAY, ls="--", lw=1, alpha=.6)
axes[0].set_xlabel("压缩后状态数 K"); axes[0].set_ylabel("bit")
axes[0].set_title("信息阶梯：状态数 vs 两路信息")
axes[0].legend(fontsize=8)
cols = [TEAL if K > 6 else GRAY for K in Ks[:-1]]
axes[1].bar([str(K) + "→" + str(K + 1) for K in Ks[:-1]], d_ity, color=cols, alpha=.85)
for i, v in enumerate(d_ity):
    axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
axes[1].set_ylabel("新增一个状态换来的 ΔI(T;Y)")
axes[1].set_title("每加一状态的'预测力售价'：K≤7 后才开始买垃圾信息")
axes[1].tick_params(axis="x", labelrotation=60, labelsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/info-staircase.png"); plt.close(fig)
STATS["d_ity"] = [float(v) for v in d_ity]
STATS["knee"] = [int(K) for K, v in zip(Ks[:-1], d_ity) if v < 0.02]

with open("/Users/halo123456/workspace/astro-blog/_ib_stats_a.json", "w") as f:
    json.dump(STATS, f, ensure_ascii=False, indent=1)
print("OK fig1/2")
for K in Ks:
    print(K, round(HTs[Ks.index(K)], 3), round(ITYs[Ks.index(K)], 3))
