#!/usr/bin/env python3
"""Generate real charts for '网络中心性与系统性重要机构识别' article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

# Use a CJK-capable system font so Chinese titles render
import matplotlib.font_manager as fm
for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Supplemental/Songti.ttc"]:
    try:
        fm.fontManager.addfont(_f)
        break
    except Exception:
        continue
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/network-centrality-sifis"
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.family": "Arial Unicode MS",
    "axes.unicode_minus": False,
})

# ---- 1. Build a core-periphery interbank exposure network ----
# 18 institutions: a tightly connected core (big banks) + a periphery (regionals)
names = [
    "ICBC", "CCB", "BOC", "ABC", "BoCom", "CMB", "SPDB", "MSB",
    "CIB", "PingAn", "CITIC", "Huaxia", "Everbright", "Minsheng",
    "BOS", "JSB-A", "JSB-B", "JSB-C",
]
n = len(names)
core = list(range(7))         # 0-6: systemically important core
peri = list(range(7, n))      # regional / smaller banks

rng = np.random.default_rng(7)
A = np.zeros((n, n))
# Core-core: dense, large exposures
for i in core:
    for j in core:
        if i < j:
            w = rng.uniform(0.6, 1.0)
            A[i, j] = A[j, i] = w
# Periphery links preferentially into the core (borrow from big banks)
for i in peri:
    k = rng.integers(2, 4)  # each regional connects to 2-3 core banks
    targets = rng.choice(core, size=k, replace=False)
    for t in targets:
        w = rng.uniform(0.15, 0.5)
        A[i, t] = A[t, i] = w
# A few periphery-periphery links (local clusters)
for i in peri:
    if rng.random() < 0.4:
        j = rng.choice(peri)
        if j != i and A[i, j] == 0:
            w = rng.uniform(0.1, 0.3)
            A[i, j] = A[j, i] = w

# ---- 2. Centrality measures ----
# Degree (strength) centrality
deg = A.sum(axis=1)
deg_c = deg / (n - 1)

# Eigenvector centrality (power iteration)
def eigenvector_centrality(A, iters=2000):
    n = A.shape[0]
    x = np.ones(n)
    for _ in range(iters):
        x = A @ x
        x = x / np.linalg.norm(x)
    return x

eig_c = eigenvector_centrality(A)
eig_c = eig_c / eig_c.max()

# Shortest-path based: closeness + betweenness (unweighted, threshold)
B = (A > 0.2).astype(float)
np.fill_diagonal(B, 0)
import scipy.sparse.csgraph as cg
dist = cg.shortest_path(cg.csgraph_from_dense(B), directed=False, unweighted=True)
clo_c = 1.0 / dist.mean(axis=1)
clo_c = clo_c / clo_c.max()

# Betweenness (Brandes) on unweighted graph
def betweenness(B):
    n = B.shape[0]
    bet = np.zeros(n)
    for s in range(n):
        S = []; P = [[] for _ in range(n)]; sigma = np.zeros(n); sigma[s] = 1
        D = np.full(n, np.inf); D[s] = 0
        Q = [s]; qh = 0
        while qh < len(Q):
            v = Q[qh]; qh += 1; S.append(v)
            for w in np.where(B[v] > 0)[0]:
                if D[w] == np.inf:
                    D[w] = D[v] + 1; Q.append(w)
                if D[w] == D[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]
    return bet / bet.max() if bet.max() > 0 else bet

bet_c = betweenness(B)

# ---- 3. Fruchterman-Reingold layout (no networkx) ----
def fruchterman_reingold(A, iters=600, seed=1):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    pos = rng.uniform(-1, 1, size=(n, 2))
    k = np.sqrt(4.0 / n)
    for it in range(iters):
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = pos[i] - pos  # (n,2)
            d = np.linalg.norm(delta, axis=1)
            d[d == 0] = 1e-6
            rep = (k * k / d)[:, None] * delta / d[:, None]
            disp[i] += rep.sum(axis=0)
            ai = A[i] > 0
            if ai.sum() > 0:
                d2 = d[ai].copy(); d2[d2 == 0] = 1e-6
                # attraction: (d2^2 / k) along unit direction
                att = (d2**2 / k)[:, None] * (delta[ai] / d2[:, None])
                disp[i] -= att.sum(axis=0) * 0.5
        length = np.linalg.norm(disp, axis=1)
        length[length == 0] = 1e-6
        step = np.minimum(length, 0.1)[:, None]
        pos += (disp / length[:, None]) * step
    return pos

pos = fruchterman_reingold(A, iters=800, seed=11)
pos -= pos.mean(axis=0)

# ---- Chart 1: network graph, node size = eigenvector centrality ----
fig, ax = plt.subplots(figsize=(9, 6.5))
# edges
for i in range(n):
    for j in range(i + 1, n):
        if A[i, j] > 0:
            x = [pos[i, 0], pos[j, 0]]; y = [pos[i, 1], pos[j, 1]]
            ax.plot(x, y, color="gray", alpha=0.25 + 0.4 * A[i, j], lw=0.5 + 2 * A[i, j])
sizes = 250 + 1600 * eig_c
colors = ["#c0392b" if i in core else "#2980b9" for i in range(n)]
ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors, alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
for i, name in enumerate(names):
    ax.annotate(name, (pos[i, 0], pos[i, 1]), fontsize=8,
                ha="center", va="center", color="white", fontweight="bold", zorder=4)
ax.set_title("银行间敞口网络：节点大小 = 特征向量中心性（红=核心行，蓝= regional）", fontsize=12)
ax.axis("off")
fig.savefig(f"{OUT}/network_graph.png"); plt.close(fig)

# ---- Chart 2: centrality bar comparison (top 10 by eigenvector) ----
order = np.argsort(eig_c)[::-1][:12]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(order)); w = 0.27
ax.bar(x - w, deg_c[order], w, label="度中心性", color="#3498db")
ax.bar(x, eig_c[order], w, label="特征向量中心性", color="#c0392b")
ax.bar(x + w, bet_c[order], w, label="介数中心性", color="#f39c12")
ax.set_xticks(x); ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("标准化中心性"); ax.set_title("三种中心性指标对比：核心行全面领先", fontsize=12)
ax.legend(fontsize=9)
fig.savefig(f"{OUT}/centrality_bars.png"); plt.close(fig)

# ---- Chart 3: adjacency heatmap ----
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(A, cmap="YlOrRd")
ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
ax.set_title("敞口邻接矩阵 A：右上角密集块即核心行的相互纠缠", fontsize=11)
fig.colorbar(im, fraction=0.046, pad=0.04, label="敞口权重")
fig.savefig(f"{OUT}/adjacency_heatmap.png"); plt.close(fig)

# ---- Chart 4: eigenvector vs betweenness scatter (SIFI ranking) ----
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(eig_c, bet_c, s=120 + 900 * deg_c, c=range(n), cmap="viridis", alpha=0.85, edgecolors="k")
for i, name in enumerate(names):
    ax.annotate(name, (eig_c[i], bet_c[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("特征向量中心性"); ax.set_ylabel("介数中心性")
ax.set_title("双维排序：右上角 = 既高度互联又掌握最短路径 → 系统重要性最高", fontsize=11)
fig.savefig(f"{OUT}/sifi_scatter.png"); plt.close(fig)

# ---- Chart 5: systemic importance ranking (composite) ----
composite = 0.4 * eig_c + 0.3 * bet_c + 0.3 * deg_c
order2 = np.argsort(composite)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(len(order2)), composite[order2][::-1], color="#8e44ad")
ax.set_yticks(range(len(order2))); ax.set_yticklabels([names[i] for i in order2][::-1], fontsize=8)
ax.set_xlabel("综合系统重要性得分")
ax.set_title("系统重要性综合排名（前高后低）", fontsize=12)
fig.savefig(f"{OUT}/sifi_ranking.png"); plt.close(fig)

print("SIFIs charts generated.")
print("Top-5 by eigenvector centrality:", [names[i] for i in np.argsort(eig_c)[::-1][:5]])
print("Top-5 by composite SIFI:", [names[i] for i in order2[:5]])
