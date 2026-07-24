import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "public/images/nmf-asset-decomposition"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# Build a non-negative "activation" data matrix:
#   3 latent themes (basis), each asset is a non-negative mix
#   V (T x N) approx  W (T x K) @ H (K x N),  all >= 0
# ---------------------------------------------------------------
T, N, K = 300, 12, 3
# latent theme time-activations (non-negative bursts)
def burst(centers, width, amp):
    t = np.arange(T)
    s = np.zeros(T)
    for c, a in zip(centers, amp):
        s += a * np.exp(-0.5*((t-c)/width)**2)
    return s
W_true = np.zeros((T, K))
W_true[:,0] = 0.4 + burst([60,180], 25, [1.2,1.0])          # theme 1
W_true[:,1] = 0.4 + burst([120,240], 30, [1.1,1.3])         # theme 2
W_true[:,2] = 0.4 + burst([30,150,270], 18, [0.9,0.9,1.0])  # theme 3
# each asset loads mostly on one theme + leakage
H_true = np.array([
    [1.0,0.9,0.8,0.7, 0.1,0.05,0.1,0.05, 0.15,0.1,0.05,0.1],   # theme1 -> assets 0-3
    [0.1,0.05,0.1,0.15, 1.0,0.9,0.85,0.8, 0.1,0.05,0.1,0.05],  # theme2 -> assets 4-7
    [0.05,0.1,0.05,0.1, 0.1,0.1,0.05,0.1, 1.0,0.95,0.85,0.9],  # theme3 -> assets 8-11
])
V_clean = W_true @ H_true
V = np.maximum(V_clean + rng.normal(0, 0.05, size=V_clean.shape), 0.0)

# ---------------------------------------------------------------
# NMF via multiplicative updates (Lee & Seung, Frobenius)
# ---------------------------------------------------------------
def nmf(V, K, iters=400, seed=0):
    r = np.random.default_rng(seed)
    W = r.uniform(0.1, 1.0, size=(V.shape[0], K))
    H = r.uniform(0.1, 1.0, size=(K, V.shape[1]))
    eps = 1e-9
    errs = []
    for _ in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)
        errs.append(np.linalg.norm(V - W @ H))
    return W, H, errs

W, H, errs = nmf(V, K, iters=400, seed=3)

# align learned components to true by correlation of H rows
from itertools import permutations
best = None
for p in permutations(range(K)):
    c = sum(np.corrcoef(H[p[i]], H_true[i])[0,1] for i in range(K))
    if best is None or c > best[0]:
        best = (c, p)
perm = best[1]
Hn = H[list(perm)]; Wn = W[:, list(perm)]
# scale normalize each component (W column * H row invariant)
for k in range(K):
    s = Hn[k].sum() + 1e-9
    Hn[k] /= s; Wn[:,k] *= s

# ---------------------------------------------------------------
# PCA comparison (SVD) -> shows negative / mixed loadings
# ---------------------------------------------------------------
Vc = V - V.mean(0)
U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
pca_load = Vt[:K]   # K x N loadings (can be negative)

# =============== FIG 1: cover — the factorization idea ===============
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2),
                         gridspec_kw={"width_ratios":[2.4,0.5,1.2]})
ax = axes[0]
im = ax.imshow(V.T, aspect="auto", cmap="magma", origin="lower")
ax.set_title("V：资产×时间 活跃度矩阵（全非负）", fontsize=12)
ax.set_xlabel("时间"); ax.set_ylabel("资产 #")
plt.colorbar(im, ax=ax, fraction=0.03)
axes[1].axis("off")
axes[1].text(0.5, 0.5, "≈\nW · H", ha="center", va="center", fontsize=26, color="#333")
ax = axes[2]
im2 = ax.imshow(Hn, aspect="auto", cmap="viridis")
ax.set_title("H：3 个基对每个资产的权重", fontsize=12)
ax.set_xlabel("资产 #"); ax.set_yticks([0,1,2]); ax.set_yticklabels(["基1","基2","基3"])
plt.colorbar(im2, ax=ax, fraction=0.045)
plt.suptitle("非负矩阵分解 NMF：把资产收益矩阵压成可解释的基与权重", fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/cover.png", dpi=120, bbox_inches="tight"); plt.close()

# =============== FIG 2: recovered basis activations ===============
fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.2), sharex=True)
cols = ["#3b6ea5", "#e0563b", "#4a9e5c"]
for k in range(K):
    ax = axes[k]
    wt = W_true[:,k] / (W_true[:,k].max()+1e-9)
    wr = Wn[:,k] / (Wn[:,k].max()+1e-9)
    ax.plot(wt, color="#999", lw=2.2, label="真实基激活")
    ax.plot(wr, color=cols[k], lw=1.4, label="NMF 还原")
    ax.set_ylabel(f"基 {k+1}")
    ax.legend(fontsize=8, loc="upper right")
axes[0].set_title("NMF 还原的时间基激活 W（归一化后对齐真实基）")
axes[-1].set_xlabel("时间")
plt.tight_layout(); plt.savefig(f"{OUT}/basis_recovery.png", dpi=120); plt.close()

# =============== FIG 3: NMF vs PCA interpretability ===============
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
im = ax.imshow(Hn, aspect="auto", cmap="viridis")
for i in range(K):
    for j in range(N):
        ax.text(j, i, f"{Hn[i,j]:.2f}", ha="center", va="center",
                color="white" if Hn[i,j]<Hn.max()*0.6 else "black", fontsize=7)
ax.set_title("NMF 权重 H：全非负，可读作『占比』")
ax.set_yticks([0,1,2]); ax.set_yticklabels(["基1","基2","基3"]); ax.set_xlabel("资产 #")
ax = axes[1]
vmax = np.abs(pca_load).max()
im2 = ax.imshow(pca_load, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
for i in range(K):
    for j in range(N):
        ax.text(j, i, f"{pca_load[i,j]:+.2f}", ha="center", va="center",
                color="black", fontsize=7)
ax.set_title("PCA 载荷：有正有负，难解释为『构成』")
ax.set_yticks([0,1,2]); ax.set_yticklabels(["PC1","PC2","PC3"]); ax.set_xlabel("资产 #")
plt.colorbar(im2, ax=ax, fraction=0.045)
plt.suptitle("非负 vs 有符号：为什么 NMF 的成分更像『主题构成』", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/nmf_vs_pca.png", dpi=120, bbox_inches="tight"); plt.close()

# =============== FIG 4: convergence + rank selection ===============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
ax.plot(errs, color="#3b6ea5")
ax.set_title("乘法更新的重构误差单调下降")
ax.set_xlabel("迭代"); ax.set_ylabel("‖V − WH‖_F"); ax.grid(alpha=0.3)
# rank scan
ax = axes[1]
ranks = range(1, 8); rec = []
for kk in ranks:
    _, _, e = nmf(V, kk, iters=250, seed=5)
    rec.append(e[-1])
ax.plot(list(ranks), rec, "o-", color="#e0563b")
ax.axvline(3, color="#4a9e5c", ls="--", label="真实秩 K=3")
ax.set_title("秩选择：误差在 K=3 后拐平（肘部）")
ax.set_xlabel("成分数 K"); ax.set_ylabel("最终 ‖V − WH‖_F")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/convergence_rank.png", dpi=120); plt.close()

# print some numbers for the article
print("RECON err final:", round(errs[-1],3))
print("H recovered corr per comp:",
      [round(np.corrcoef(Hn[k], H_true[k]/H_true[k].sum())[0,1],3) for k in range(K)])
print("W recovered corr per comp:",
      [round(np.corrcoef(Wn[:,k], W_true[:,k])[0,1],3) for k in range(K)])
print("PCA neg loading fraction:", round((pca_load<0).mean(),3))
print("rank scan errs:", [round(x,2) for x in rec])
print("done", os.listdir(OUT))
