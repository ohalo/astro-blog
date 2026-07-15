#!/usr/bin/env python3
"""
为文章「相关性网络与最小生成树：用图论看清板块联动结构」(correlation-network-mst)
生成真实配图。自洽合成：40 只股票分 4 个板块、板块内强相关、板块间弱相关，
从相关矩阵构造距离 d_ij = sqrt(0.5*(1-corr))，跑 Kruskal 求 MST（用并查集，无需 networkx），
展示「最小生成树如何把市场切成清晰的板块簇」+ MST 层级结构（同心圆布局）+ 危机期相关性飙升时
MST 如何"改道"连通原本远的板块 + 节点度分布（枢纽股）。

图1 mst_network.png           MST 网络：节点按板块着色，边按距离着色，清晰露出 4 个板块簇
图2 mst_radial.png            MST 放射状层级布局（突出枢纽节点与层级）
图3 mst_crisis_rewire.png     危机期相关性飙升：MST 改道，跨板块边增多（结构变密）
图4 mst_degree_distribution.png 节点度分布：少数枢纽股（度很大），多数叶子（度=1）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "correlation-network-mst"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

SECTORS = ["科技", "金融", "能源", "医药"]
SECT_N = [10, 10, 10, 10]
N = sum(SECT_N)
T = 250


def build_corr(within=0.55, between=0.0, seed=42):
    r = np.random.default_rng(seed)
    wv = within + 0.25 * r.random(len(SECT_N))
    bv = between + 0.10 * r.random((len(SECT_N), len(SECT_N)))
    np.fill_diagonal(bv, 0.0)
    C = np.zeros((N, N))
    idx = 0
    blocks = []
    for k, n in enumerate(SECT_N):
        sl = slice(idx, idx + n)
        C[sl, sl] = wv[k]
        blocks.append(sl)
        idx += n
    for a in range(len(SECT_N)):
        for b in range(a + 1, len(SECT_N)):
            sa, sb = blocks[a], blocks[b]
            C[sa, sb] = bv[a, b]
            C[sb, sa] = bv[a, b]
    np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-3, None)
    C = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return C, blocks


def estimate_corr(C, T, seed=123):
    r = np.random.default_rng(seed)
    L = np.linalg.cholesky(C)
    X = (r.standard_normal((T, N)) @ L.T)
    X = (X - X.mean(0)) / X.std(0)
    return np.corrcoef(X, rowvar=False)


def dist_from_corr(C):
    Cc = np.clip(C, -1.0, 1.0)
    return np.sqrt(np.clip(0.5 * (1.0 - Cc), 0.0, 1.0))


def kruskal_mst(Dmat):
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((Dmat[i, j], i, j))
    edges.sort()
    mst = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            mst.append((i, j, w))
        if len(mst) == N - 1:
            break
    return mst


def layout_ring():
    """把 40 节点按板块分成 4 段环，便于展示板块簇。"""
    pos = {}
    idx = 0
    for k, n in enumerate(SECT_N):
        base = k * 2 * np.pi / len(SECT_N)
        for s in range(n):
            ang = base + (s - n / 2) * 0.06
            rad = 1.0
            pos[idx] = (rad * np.cos(ang), rad * np.sin(ang))
            idx += 1
    return pos


def plot_network(Dmat, title, fname, pos=None, edge_alpha=0.9, note=""):
    mst = kruskal_mst(Dmat)
    if pos is None:
        pos = layout_ring()
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = np.concatenate([[c] * n for c, n in enumerate(SECT_N)])
    cmap = plt.cm.tab10
    # 边按距离（相关强弱）着色：近=强相关=暖色
    for (i, j, w) in mst:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        c = plt.cm.viridis(w)
        ax.plot(x, y, color=c, lw=1.6, alpha=edge_alpha)
    for i in range(N):
        ax.scatter(pos[i][0], pos[i][1], s=70, color=cmap(colors[i]), zorder=5,
                   edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    if note:
        ax.text(0.5, -0.08, note, transform=ax.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    # 板块图例
    for k, name in enumerate(SECTORS):
        ax.scatter([], [], c=cmap(k), label=name, s=80)
    ax.legend(scatterpoints=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(D, fname), dpi=130)
    plt.close(fig)
    return mst


def plot_radial(Dmat, fname):
    """以中心枢纽为根，放射状层级布局（按边权排序做 BFS 树）。"""
    mst = kruskal_mst(Dmat)
    # 构建邻接
    adj = {i: [] for i in range(N)}
    for (i, j, w) in mst:
        adj[i].append((j, w))
        adj[j].append((i, w))
    # 选度最大的节点为根
    deg = {i: len(adj[i]) for i in range(N)}
    root = max(deg, key=deg.get)
    # BFS 分层
    from collections import deque
    depth = {root: 0}
    parent = {root: -1}
    q = deque([root])
    order = [root]
    while q:
        u = q.popleft()
        for (v, w) in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                parent[v] = u
                q.append(v)
                order.append(v)
    maxd = max(depth.values())
    # 每层角向均分
    pos = {root: (0.0, 0.0)}
    for d_ in range(1, maxd + 1):
        nodes = [u for u in order if depth[u] == d_]
        for m, u in enumerate(nodes):
            ang = 2 * np.pi * m / len(nodes) + d_ * 0.3
            rad = 0.35 + 0.55 * d_ / maxd
            pos[u] = (rad * np.cos(ang), rad * np.sin(ang))
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = np.concatenate([[c] * n for c, n in enumerate(SECT_N)])
    cmap = plt.cm.tab10
    for (i, j, w) in mst:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, color=plt.cm.viridis(w), lw=1.4, alpha=0.85)
    for i in range(N):
        sz = 220 if i == root else 70
        ax.scatter(pos[i][0], pos[i][1], s=sz, color=cmap(colors[i]), zorder=5,
                   edgecolor="white", linewidth=1.0 if i == root else 0.6)
    ax.scatter([], [], c=cmap(root % 4), label=f"枢纽节点 S{root+1}（度={deg[root]}）", s=90)
    ax.set_title("MST 放射状层级：枢纽在中心，叶子在最外层", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(D, fname), dpi=130)
    plt.close(fig)


def plot_degree(Dmat, fname):
    mst = kruskal_mst(Dmat)
    deg = np.zeros(N, dtype=int)
    for (i, j, w) in mst:
        deg[i] += 1
        deg[j] += 1
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(deg, bins=np.arange(0.5, deg.max() + 1.5, 1), color="#1f77b4",
            edgecolor="white", rwidth=0.9)
    ax.set_xlabel("节点度（连接的边数）")
    ax.set_ylabel("节点个数")
    ax.set_title("MST 节点度分布：少数枢纽股、多数叶子股（度=1）", fontsize=13)
    ax.text(0.6, 0.7, f"最大度={deg.max()} ·\n{int((deg==1).sum())}/{N} 个叶子（度=1）",
            transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(D, fname), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    C, blocks = build_corr(42)
    S = estimate_corr(C, T, 123)
    Dmat = dist_from_corr(S)
    mst = plot_network(Dmat, "最小生成树（MST）：市场被切成清晰的 4 个板块簇",
                       "mst_network.png", note="边色=相关强弱（暖色=强相关）。每个板块内部连通紧密，跨板块只有少数桥接边。")
    plot_radial(Dmat, "mst_radial.png")
    # 危机期：板块间相关性飙升（between 升高），MST 改道
    Cc, _ = build_corr(within=0.55, between=0.45, seed=42)
    Sc = estimate_corr(Cc, T, 123)
    Dc = dist_from_corr(Sc)
    plot_network(Dc, "危机期相关性飙升：MST 跨板块边增多、结构变密",
                 "mst_crisis_rewire.png", note="板块间距离被压缩，原本远的板块被 MST 拉通——联动结构在危机中『改道』。")
    plot_degree(Dmat, "mst_degree_distribution.png")
    print("images written to", D)
    for f in sorted(os.listdir(D)):
        print("  ", f)
