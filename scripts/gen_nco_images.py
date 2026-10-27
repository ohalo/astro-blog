#!/usr/bin/env python3
"""嵌套聚类优化 NCO 文章配图"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/nested-clustered-optimization"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------
# 构造块状真实协方差: 4个板块, 每板块内相关0.6, 板块间0.15
# ---------------------------------------------------------------
N = 40
K_true = 4
sizes = [12, 10, 10, 8]
labels_true = np.concatenate([[i]*s for i, s in enumerate(sizes)])
rho_in, rho_out = 0.6, 0.15
C_true = np.full((N, N), rho_out)
for k in range(K_true):
    idx = labels_true == k
    C_true[np.ix_(idx, idx)] = rho_in
np.fill_diagonal(C_true, 1.0)
vols_true = rng.uniform(0.15, 0.35, N) / np.sqrt(252)
S_true = C_true * np.outer(vols_true, vols_true)

def min_var_w(S):
    Si = np.linalg.pinv(S)
    ones = np.ones(len(S))
    return Si@ones / (ones@Si@ones)

def nco_min_var(S):
    """嵌套聚类优化: 相关距离聚类 -> 簇内minvar -> 簇间minvar"""
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    dist = np.sqrt(np.clip(0.5*(1-C), 0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    # 用简化的cluster数选择: 4 (真实结构已知; 实践中用gap/silhouette)
    lab = fcluster(Z, t=K_true, criterion="maxclust")
    W_intra = np.zeros((N, K_true))
    for k in range(1, K_true+1):
        idx = np.where(lab == k)[0]
        w = min_var_w(S[np.ix_(idx, idx)])
        W_intra[idx, k-1] = w
    S_reduced = W_intra.T @ S @ W_intra
    w_inter = min_var_w(S_reduced)
    return W_intra @ w_inter, lab, Z

# ---------------------------------------------------------------
# 1. 样本协方差 vs NCO: 不同 T 下 OOS 波动率 (Monte Carlo)
# ---------------------------------------------------------------
L = np.linalg.cholesky(S_true)
w_opt = min_var_w(S_true)
vol_opt = np.sqrt(w_opt@S_true@w_opt*252)*100

Ts = [90, 126, 252, 504, 1008]
n_mc = 60
res = {"样本协方差 + MinVar": {t: [] for t in Ts},
       "NCO（聚类两层）": {t: [] for t in Ts},
       "等权 1/N": {t: [] for t in Ts}}
w_eq = np.ones(N)/N
vol_eq = np.sqrt(w_eq@S_true@w_eq*252)*100
lev = {t: [] for t in Ts}; lev_nco = {t: [] for t in Ts}
for T in Ts:
    for _ in range(n_mc):
        R = rng.standard_normal((T, N)) @ L.T
        S_hat = np.cov(R.T)
        w1 = min_var_w(S_hat)
        w2, _, _ = nco_min_var(S_hat)
        res["样本协方差 + MinVar"][T].append(np.sqrt(w1@S_true@w1*252)*100)
        res["NCO（聚类两层）"][T].append(np.sqrt(w2@S_true@w2*252)*100)
        res["等权 1/N"][T].append(vol_eq)
        lev[T].append(np.abs(w1).sum())
        lev_nco[T].append(np.abs(w2).sum())

fig, ax = plt.subplots(figsize=(9, 4.6))
colors = {"样本协方差 + MinVar": "#1f77b4", "NCO（聚类两层）": "#d62728", "等权 1/N": "#7f7f7f"}
for name, d in res.items():
    med = [np.median(d[t]) for t in Ts]
    ax.plot(Ts, med, "o-", color=colors[name], label=name, lw=1.6)
ax.axhline(vol_opt, color="0.3", ls="--", label=f"理论最优 {vol_opt:.1f}%")
ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts)
ax.set_xlabel("估计窗口 T（天）"); ax.set_ylabel("OOS 真实年化波动率 (%)")
ax.set_title(f"最小方差组合真实波动率：N=40，块状相关结构（{n_mc} 次 MC 中位数）")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/oos-vol.png", dpi=110)
plt.close()
for T in Ts:
    print(f"T={T}: 样本={np.median(res['样本协方差 + MinVar'][T]):.2f}% NCO={np.median(res['NCO（聚类两层）'][T]):.2f}% 杠杆 {np.median(lev[T]):.2f} vs {np.median(lev_nco[T]):.2f}")
print(f"理论最优={vol_opt:.2f}%, 等权={vol_eq:.2f}%")

# ---------------------------------------------------------------
# 2. 树状图 + 排序相关矩阵
# ---------------------------------------------------------------
T_demo = 252
R_demo = rng.standard_normal((T_demo, N)) @ L.T
S_demo = np.cov(R_demo.T)
w_nco, lab_demo, Z_demo = nco_min_var(S_demo)
d = np.sqrt(np.diag(S_demo)); C_demo = S_demo/np.outer(d, d)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
dn = dendrogram(Z_demo, ax=ax, no_labels=True, color_threshold=Z_demo[-K_true+1, 2])
ax.set_title("相关距离 Ward 聚类树：4 个板块被干净切开")
ax.set_ylabel("合并距离")
order = dn["leaves"]
ax = axes[1]
im = ax.imshow(C_demo[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_title("按聚类顺序重排的样本相关矩阵（T=252）")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{OUT}/dendrogram.png", dpi=110)
plt.close()

# ---------------------------------------------------------------
# 3. 权重对比: 样本MinVar vs NCO vs 理论最优
# ---------------------------------------------------------------
w_sample = min_var_w(S_demo)
fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
for ax, w, name, c in [(axes[0], w_sample, "样本协方差 MinVar（T=252）", "#1f77b4"),
                       (axes[1], w_nco, "NCO（T=252）", "#d62728"),
                       (axes[2], w_opt, "理论最优（真实协方差）", "0.4")]:
    ax.bar(np.arange(N), w*100, color=c, width=0.8)
    ax.axhline(0, color="0.2", lw=0.6)
    ax.set_ylabel("权重 (%)")
    ax.set_title(f"{name}：总杠杆 {np.abs(w).sum():.2f}，负权重 {int((w<0).sum())} 个", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for b in np.cumsum(sizes)[:-1]:
        ax.axvline(b-0.5, color="0.75", ls=":", lw=0.8)
axes[2].set_xlabel("资产编号（虚线分隔真实板块）")
plt.tight_layout()
plt.savefig(f"{OUT}/weights.png", dpi=110)
plt.close()
print(f"权重: 样本杠杆={np.abs(w_sample).sum():.2f} 负{(w_sample<0).sum()}个; NCO杠杆={np.abs(w_nco).sum():.2f} 负{(w_nco<0).sum()}个; 最优杠杆={np.abs(w_opt).sum():.2f}")

# ---------------------------------------------------------------
# 4. 权重稳定性: bootstrap 下权重标准差
# ---------------------------------------------------------------
n_bs = 80
W_s = np.zeros((n_bs, N)); W_n = np.zeros((n_bs, N))
for b in range(n_bs):
    idx = rng.integers(0, T_demo, T_demo)
    S_b = np.cov(R_demo[idx].T)
    W_s[b] = min_var_w(S_b)
    W_n[b], _, _ = nco_min_var(S_b)
std_s = W_s.std(0)*100; std_n = W_n.std(0)*100

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(np.arange(N)-0.2, std_s, 0.4, color="#1f77b4", label=f"样本 MinVar（均值 {std_s.mean():.2f}pp）")
ax.bar(np.arange(N)+0.2, std_n, 0.4, color="#d62728", label=f"NCO（均值 {std_n.mean():.2f}pp）")
ax.set_xlabel("资产编号"); ax.set_ylabel("Bootstrap 权重标准差（百分点）")
ax.set_title("同一份数据重抽样 80 次：NCO 权重稳定性显著更好")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/weight-stability.png", dpi=110)
plt.close()
print(f"bootstrap 权重std均值: 样本={std_s.mean():.3f}pp, NCO={std_n.mean():.3f}pp")
print("done", os.listdir(OUT))
