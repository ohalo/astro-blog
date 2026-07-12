#!/usr/bin/env python3
"""
为文章「层次风险平价(HRP)：用聚类与递归二分超越传统风险平价」(hierarchical-risk-parity)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

数据机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 15 只资产，分 3 个板块（科技 / 金融 / 能源），板块内高相关、板块间低相关；
  - 用板块化协方差 + 多变量正态生成日收益序列，前 1000 日估计协方差、后 500 日做样本外；
  - 实现 HRP（层次风险平价：相关→距离→Ward 聚类→准对角化→递归二分等风险贡献），
    与 等权(EW) / 逆方差(IVP) / 最小方差(MinVar) / 风险平价(RP) 对比。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "hierarchical-risk-parity")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "hrp": "#2F4B7C", "ew": "#C44E52", "ivp": "#55A868",
     "minv": "#8172B3", "rp": "#E8A33D", "line": "#333333"}

# ============================================================
# 1) 板块化协方差 + 收益序列
# ============================================================
def build_data(seed=20260713):
    rng = np.random.default_rng(seed)
    sectors = ["Tech", "Fin", "Energy"]
    n_per = 5
    names = []
    for s in sectors:
        names += [f"{s}{i+1}" for i in range(n_per)]
    N = len(names)
    # 真实相关：板块内 0.7，板块间 0.2
    corr = np.ones((N, N)) * 0.2
    for k in range(3):
        a, b = k * n_per, (k + 1) * n_per
        corr[a:b, a:b] = 0.7
    np.fill_diagonal(corr, 1.0)
    corr += rng.normal(0, 0.03, (N, N)); corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    # 个别资产年化波动 16%~28%（避免极端离散导致权重塌缩到单一低波动资产）
    vols = rng.uniform(0.16, 0.28, N)
    cov = (corr * np.outer(vols, vols))
    # 日收益，T 日
    T = 1500
    daily_cov = cov / 252.0
    R = rng.multivariate_normal(np.zeros(N), daily_cov, size=T)
    return names, vols, cov, R

# ============================================================
# 2) HRP 实现（Lopez de Prado 2016）
#    相关→距离→Ward 聚类→准对角化→按聚类树递归二分（等风险贡献）
# ============================================================
def hrp_weights(cov):
    N = cov.shape[0]
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = dendrogram(Z, no_plot=True)["leaves"]

    # 每个节点（含内部节点）所包含的叶子集合
    node_leaves = {i: [i] for i in range(N)}
    for k, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        node_leaves[N + k] = node_leaves[a] + node_leaves[b]
    root = N + len(Z) - 1

    w = np.ones(N) / N

    def allocate(node):
        leaves = node_leaves[node]
        if len(leaves) <= 1:
            return
        ridx = node - N
        a, b = int(Z[ridx, 0]), int(Z[ridx, 1])
        left = node_leaves[a]
        right = node_leaves[b]

        def pv(idxs):
            sub = cov[np.ix_(idxs, idxs)]
            iv = 1.0 / np.diag(sub)            # 子簇内用 IVP 归一化权重算组合方差（尺度无关，避免孤立资产被放大）
            ww = iv / iv.sum()
            return ww @ sub @ ww

        vL, vR = pv(left) + 1e-12, pv(right) + 1e-12
        aL, aR = 1.0 / vL, 1.0 / vR
        s = aL + aR
        Wnode = w[leaves].sum()
        target_left = Wnode * aL / s
        target_right = Wnode * aR / s
        cur_left = w[left].sum()
        cur_right = w[right].sum()
        # 保持该节点总权重不变，仅按逆方差在两子簇间重新切分
        w[left] *= (target_left / cur_left)
        w[right] *= (target_right / cur_right)
        if a >= N:
            allocate(a)
        if b >= N:
            allocate(b)

    allocate(root)
    return w, Z, order, corr

# ============================================================
# 3) 对比组合权重
# ============================================================
def ivp_weights(cov):
    iv = 1.0 / np.diag(cov); return iv / iv.sum()

def ew_weights(cov):
    N = cov.shape[0]; return np.ones(N) / N

def minvar_weights(cov):
    N = cov.shape[0]
    def obj(w): return w @ cov @ w
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * N
    x0 = np.ones(N) / N
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def rp_weights(cov, iters=50):
    N = cov.shape[0]
    w = np.ones(N) / N
    for _ in range(iters):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / N
        w = w * (target / (rc + 1e-12))
        w = np.clip(w, 0, None); w /= w.sum()
    return w

def port_vol(w, cov):
    return np.sqrt(w @ cov @ w)

def risk_contrib(w, cov):
    pv = w @ cov @ w
    mrc = cov @ w
    return w * mrc / pv

# ============================================================
# 4) 图一：原始相关矩阵 vs 准对角化相关矩阵
# ============================================================
def fig_corr(corr, order):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    im1 = a1.imshow(corr, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    a1.set_title("raw correlation matrix", fontsize=10.5)
    a1.set_xticks([]); a1.set_yticks([])
    cd = corr[np.ix_(order, order)]
    a2.imshow(cd, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    a2.set_title("quasi-diagonalized (HRP leaf order)", fontsize=10.5)
    a2.set_xticks([]); a2.set_yticks([])
    fig.colorbar(im1, ax=[a1, a2], fraction=0.025, pad=0.02)
    fig.tight_layout()
    p = os.path.join(D, "hrp_corr.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 5) 图二：层次聚类树（板块着色）
# ============================================================
def fig_dendro(Z, order, names, n_per=5):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    palette = ["#2F4B7C", "#C44E52", "#55A868"]
    d = dendrogram(Z, labels=names, ax=ax, color_threshold=0,
                   above_threshold_color="#888888", leaf_rotation=70, leaf_font_size=8)
    # 给叶子按板块上色
    xl = ax.get_xticks()
    for i, lab in enumerate(d["ivl"]):
        sec = lab[:-1]
        col = palette[{"Tech": 0, "Fin": 1, "Energy": 2}[sec]]
        ax.get_xaxis().get_ticklabels()[i].set_color(col)
    ax.set_title("Ward hierarchical clustering recovers the 3 sectors", fontsize=11.5)
    ax.set_ylabel("distance"); ax.set_xlabel("asset")
    ax.grid(True, color=C["grid"], lw=0.6, axis="y"); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "hrp_dendro.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 6) 图三：五类组合权重对比
# ============================================================
def fig_weights(names, W):
    labels = ["EW", "IVP", "MinVar", "RP", "HRP"]
    cols = [C["ew"], C["ivp"], C["minv"], C["rp"], C["hrp"]]
    x = np.arange(len(names)); w = 0.16
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for i, (lab, col) in enumerate(zip(labels, cols)):
        ax.bar(x + (i - 2) * w, W[:, i], width=w, label=lab, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=70, fontsize=8)
    ax.set_title("Weights: HRP diversifies; MinVar (markwitz) concentrates", fontsize=11.0)
    ax.set_ylabel("weight"); ax.legend(frameon=False, fontsize=8.2, ncol=5)
    ax.grid(True, color=C["grid"], lw=0.6, axis="y"); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "hrp_weights.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 7) 图四：样本外累积收益对比
# ============================================================
def fig_perf(R_oos, W):
    labels = ["EW", "IVP", "MinVar", "RP", "HRP"]
    cols = [C["ew"], C["ivp"], C["minv"], C["rp"], C["hrp"]]
    cum = np.cumprod(1 + R_oos @ W, axis=0)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, (lab, col) in enumerate(zip(labels, cols)):
        ax.plot(cum[:, i], color=col, lw=1.8, label=lab)
    ax.set_title("Out-of-sample cumulative return (500 days)", fontsize=11.5)
    ax.set_xlabel("day"); ax.set_ylabel("growth of $1")
    ax.legend(frameon=False, fontsize=8.5, ncol=5)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "hrp_perf.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

if __name__ == "__main__":
    names, vols, cov, R = build_data()
    N = len(names)
    # 估计样本 / 样本外
    R_in, R_oos = R[:1000], R[1000:]
    cov_in = np.cov(R_in.T) * 252.0  # 年化协方差
    cov_oos = np.cov(R_oos.T) * 252.0

    w_hrp, Z, order, corr = hrp_weights(cov_in)
    W = np.column_stack([
        ew_weights(cov_in), ivp_weights(cov_in),
        minvar_weights(cov_in), rp_weights(cov_in), w_hrp
    ])
    # 复原质量：板块是否连续
    print("leaf order:", [names[i] for i in order])
    # 样本外波动率（年化）
    vol_oos = np.array([port_vol(W[:, i], cov_oos) for i in range(5)])
    # 样本内估计波动率（年化）
    vol_in = np.array([port_vol(W[:, i], cov_in) for i in range(5)])
    print("annualized vol (in-sample est):", np.round(vol_in, 4))
    print("annualized vol (out-of-sample):", np.round(vol_oos, 4))
    print("HRP weights sum=%.4f  max=%.3f  min=%.3f" % (w_hrp.sum(), w_hrp.max(), w_hrp.min()))
    print("MinVar max weight=%.3f (concentration)" % W[:, 2].max())
    # 风险贡献（HRP vs EW，用样本外协方差）
    rc_hrp = risk_contrib(w_hrp, cov_oos)
    rc_ew = risk_contrib(np.ones(N) / N, cov_oos)
    print("HRP risk-contrib std=%.4f  EW risk-contrib std=%.4f" %
          (rc_hrp.std(), rc_ew.std()))

    p1 = fig_corr(corr, order)
    p2 = fig_dendro(Z, order, names)
    p3 = fig_weights(names, W)
    p4 = fig_perf(R_oos, W)
    print("saved:", p1); print("saved:", p2); print("saved:", p3); print("saved:", p4)
