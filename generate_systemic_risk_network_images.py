#!/usr/bin/env python3
"""
为文章「系统性风险网络：用关联图把传染路径画出来」(systemic-risk-network)
生成真实配图（自洽合成，非占位图）。

数据模型：
  模拟 40 家金融机构（分 4 类：银行 / 券商 / 保险 / 信托），类内强关联、类间弱关联，
  构建收益相关矩阵 → 把"相关性"翻译成"关联图"，用图论指标量化系统性风险：
    1) 节点度 / 中心度（谁是系统重要性节点 = 枢纽）
    2) 邻接结构（最小生成树 / k-NN 关联网络）
    3) 冲击传染：从某节点注入 shock，沿边传播，测传播深度与广度

图表：
  1. sr_network.png            关联网络（k-NN 图）：节点按机构类型着色，枢纽股明显
  2. sr_centrality.png         节点中心度排序：少数系统重要性机构（高中心度）
  3. sr_contagion.png          冲击传染模拟：从一个枢纽注入 shock，沿关联边扩散
  4. sr_degree_dist.png        节点度分布：少数枢纽、多数叶子
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "systemic-risk-network"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

RNG = np.random.default_rng(20260716)
TYPES = ["银行", "券商", "保险", "信托"]
TYPE_N = [14, 9, 9, 8]
N = sum(TYPE_N)
T = 250


def build_corr(within=0.6, between=0.05, seed=42):
    r = RNG if seed == 42 else np.random.default_rng(seed)
    wv = within + 0.2 * r.random(len(TYPE_N))
    bv = between + 0.12 * r.random((len(TYPE_N), len(TYPE_N)))
    np.fill_diagonal(bv, 0.0)
    C = np.zeros((N, N))
    idx = 0
    blocks = []
    for k, n in enumerate(TYPE_N):
        sl = slice(idx, idx + n)
        C[sl, sl] = wv[k]
        blocks.append(sl)
        idx += n
    for a in range(len(TYPE_N)):
        for b in range(a + 1, len(TYPE_N)):
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


def knn_graph(C, k=3):
    """每个节点连它相关最高的 k 个邻居（无向，对称化）。"""
    A = np.zeros((N, N))
    for i in range(N):
        order = np.argsort(-C[i])
        for j in order[1:k + 1]:  # 跳过自己
            A[i, j] = 1
    A = np.maximum(A, A.T)  # 对称化
    return A


def kruskal_mst(C):
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((C[i, j], i, j))
    edges.sort(reverse=True)  # 相关越高越先取
    mst = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            mst.append((i, j, w))
        if len(mst) == N - 1:
            break
    return mst


def degree(A):
    return A.sum(1)


def centrality(A):
    """用特征向量中心度（邻接矩阵主特征向量）量化系统重要性。"""
    w, V = np.linalg.eigh(A.astype(float))
    idx = np.argmax(w)
    v = np.abs(V[:, idx])
    return v / v.sum()


def layout():
    pos = {}
    idx = 0
    for k, n in enumerate(TYPE_N):
        base = k * 2 * np.pi / len(TYPE_N)
        for s in range(n):
            ang = base + (s - n / 2) * 0.08
            pos[idx] = (np.cos(ang), np.sin(ang))
            idx += 1
    return pos


def contagion(A, source, beta=0.6):
    """从 source 注入单位 shock，沿关联边按 beta 衰减传播（BFS）。"""
    infected = {source: 1.0}
    frontier = {source}
    depth = 0
    while frontier and depth < 6:
        nxt = set()
        for u in frontier:
            for v in range(N):
                if A[u, v] > 0 and v not in infected:
                    infected[v] = infected[u] * beta
                    nxt.add(v)
        frontier = nxt
        depth += 1
    load = np.zeros(N)
    for k, v in infected.items():
        load[k] = v
    return load


if __name__ == "__main__":
    C, blocks = build_corr(42)
    S = estimate_corr(C, T, 123)
    A = knn_graph(S, k=3)
    cent = centrality(A)
    deg = degree(A).astype(int)
    colors = np.concatenate([[c] * n for c, n in enumerate(TYPE_N)])
    cmap = plt.cm.tab10
    pos = layout()

    # ---------- 图1：关联网络 ----------
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                x = [pos[i][0], pos[j][0]]
                y = [pos[i][1], pos[j][1]]
                ax.plot(x, y, color="#999999", lw=1.1, alpha=0.5)
    sizes = 60 + 700 * cent
    for i in range(N):
        ax.scatter(pos[i][0], pos[i][1], s=sizes[i], color=cmap(colors[i]), zorder=5,
                   edgecolor="white", linewidth=0.9)
    for k, name in enumerate(TYPES):
        ax.scatter([], [], c=cmap(k), label=name, s=90)
    ax.legend(scatterpoints=1, loc="upper right", fontsize=11)
    ax.set_title("关联网络（k-NN，k=3）：节点大小 = 系统重要性中心度", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(D, "sr_network.png"), dpi=130)
    plt.close(fig)

    # ---------- 图2：中心度排序 ----------
    order = np.argsort(-cent)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bar_c = [cmap(colors[i]) for i in order]
    ax.bar(range(N), cent[order], color=bar_c, edgecolor="white", linewidth=0.3)
    ax.set_title("节点中心度排序：少数机构占据系统重要性高位", fontsize=14, fontweight="bold")
    ax.set_xlabel("机构（按中心度降序）")
    ax.set_ylabel("特征向量中心度")
    ax.grid(axis="y", alpha=0.3)
    top5 = order[:5]
    ax.annotate(f"Top5: {', '.join('N'+str(i+1) for i in top5)}",
                xy=(2, cent[order[0]]), xytext=(N*0.4, cent[order[0]]*0.9),
                fontsize=11, bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
    fig.tight_layout()
    fig.savefig(os.path.join(D, "sr_centrality.png"), dpi=130)
    plt.close(fig)

    # ---------- 图3：冲击传染 ----------
    source = int(np.argmax(cent))
    load = contagion(A, source, beta=0.6)
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                x = [pos[i][0], pos[j][0]]
                y = [pos[i][1], pos[j][1]]
                ax.plot(x, y, color="#cccccc", lw=0.8, alpha=0.4)
    sc = ax.scatter([pos[i][0] for i in range(N)], [pos[i][1] for i in range(N)],
                   s=80 + 500 * load, c=load, cmap="Reds", zorder=5,
                   edgecolor="black", linewidth=0.6)
    ax.scatter(pos[source][0], pos[source][1], s=380, c="darkred", marker="*",
               zorder=6, edgecolor="white", linewidth=1.2, label=f"冲击源 N{source+1}")
    ax.legend(loc="upper right")
    ax.set_title(f"冲击传染模拟：从枢纽 N{source+1} 注入 shock 的扩散负载", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("感染负载")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "sr_contagion.png"), dpi=130)
    plt.close(fig)

    # ---------- 图4：度分布 ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(deg, bins=np.arange(-0.5, deg.max() + 1.5, 1), color="#1f77b4",
            edgecolor="white", rwidth=0.9)
    ax.set_xlabel("节点度（连接的边数）")
    ax.set_ylabel("机构个数")
    ax.set_title("关联网络节点度分布：少数枢纽机构、多数叶子机构", fontsize=13, fontweight="bold")
    ax.text(0.6, 0.7, f"最大度={deg.max()} ·\n{int((deg==1).sum())}/{N} 个叶子（度=1）",
            transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "sr_degree_dist.png"), dpi=130)
    plt.close(fig)

    print("systemic-risk-network 配图已生成：", sorted(os.listdir(D)))
