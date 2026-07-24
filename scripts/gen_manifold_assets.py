import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(11)
OUT = "public/images/manifold-learning-assets"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# Build high-D asset feature vectors whose TRUE structure lives on
# a 1-D curved manifold (a "sector ring") embedded in high-D.
# We want to show: PCA (linear) tears the curve; Isomap (geodesic)
# unrolls it correctly.
# ---------------------------------------------------------------
n = 200
# latent 1-D coordinate = position around a sector cycle
t = np.sort(rng.uniform(0, 1.6*np.pi, n))
# swiss-roll style 3D embedding of the 1-D curve
x = t*np.cos(t)
y = 12*rng.uniform(0,1,n)          # a second spread dim (thickness)
z = t*np.sin(t)
X3 = np.c_[x, y, z]
# lift to high-D (D=30) with a random linear map + small noise
D = 30
Rmap = rng.normal(size=(3, D))
Xhd = X3 @ Rmap + rng.normal(0, 0.05, size=(n, D))
color = t   # true 1-D ordering (e.g. sector rotation phase)

# ---------------------------------------------------------------
# PCA (SVD)
# ---------------------------------------------------------------
Xc = Xhd - Xhd.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pca2 = Xc @ Vt[:2].T

# ---------------------------------------------------------------
# Isomap from scratch:
#  1. kNN graph on Euclidean dist
#  2. graph shortest paths = geodesic dist  (Floyd-Warshall, small n)
#  3. classical MDS on geodesic dist matrix
# ---------------------------------------------------------------
def pairwise(A):
    s = (A**2).sum(1)
    d2 = s[:,None] + s[None,:] - 2*A@A.T
    return np.sqrt(np.maximum(d2, 0))

def isomap(A, k=8, dim=2):
    Dm = pairwise(A)
    nn = np.argsort(Dm, 1)[:, 1:k+1]
    G = np.full_like(Dm, np.inf)
    np.fill_diagonal(G, 0)
    for i in range(len(A)):
        for j in nn[i]:
            G[i,j] = Dm[i,j]; G[j,i] = Dm[i,j]   # symmetric
    # Floyd-Warshall
    for kk in range(len(A)):
        G = np.minimum(G, G[:,kk][:,None] + G[kk,:][None,:])
    # classical MDS
    n = len(A)
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ (G**2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    L = np.sqrt(np.maximum(w[idx], 0))
    Y = V[:, idx] * L
    return Y, G

iso2, Gdist = isomap(Xhd, k=8, dim=2)

# recovered 1-D coordinate quality: correlation of embedding axis with true t
def best_axis_corr(emb, truth):
    c1 = abs(np.corrcoef(emb[:,0], truth)[0,1])
    c2 = abs(np.corrcoef(emb[:,1], truth)[0,1])
    return max(c1, c2)
pca_c = best_axis_corr(pca2, color)
iso_c = best_axis_corr(iso2, color)

# neighborhood preservation (trustworthiness-lite): fraction of local
# kNN preserved after embedding
def knn_preserved(hd, emb, k=10):
    Dh = pairwise(hd); De = pairwise(emb)
    nh = np.argsort(Dh,1)[:,1:k+1]
    ne = np.argsort(De,1)[:,1:k+1]
    keep = [len(set(nh[i]) & set(ne[i]))/k for i in range(len(hd))]
    return np.mean(keep)
pca_keep = knn_preserved(Xhd, pca2)
iso_keep = knn_preserved(Xhd, iso2)

# =============== FIG 1 cover: the manifold + two embeddings ===============
fig = plt.figure(figsize=(13, 4.3))
ax = fig.add_subplot(1,3,1, projection="3d")
ax.scatter(x, y, z, c=color, cmap="Spectral", s=16)
ax.set_title("高维资产特征的真实结构\n（1 维流形卷在高维里）", fontsize=11)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax2 = fig.add_subplot(1,3,2)
ax2.scatter(pca2[:,0], pca2[:,1], c=color, cmap="Spectral", s=16)
ax2.set_title(f"PCA（线性）：把卷曲的流形拍扁、撕裂\n轴相关={pca_c:.2f}", fontsize=11)
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
ax3 = fig.add_subplot(1,3,3)
ax3.scatter(iso2[:,0], iso2[:,1], c=color, cmap="Spectral", s=16)
ax3.set_title(f"Isomap（测地）：把流形展开成有序带\n轴相关={iso_c:.2f}", fontsize=11)
ax3.set_xlabel("维1"); ax3.set_ylabel("维2")
plt.suptitle("流形学习资产表征：用降维把高维因子铺成可看的地图", y=1.03, fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/cover.png", dpi=120, bbox_inches="tight"); plt.close()

# =============== FIG 2: euclidean vs geodesic distance ===============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
Deuc = pairwise(Xhd)
# pick a reference point at one end
ref = np.argmin(color)
ax = axes[0]
sc = ax.scatter(pca2[:,0], pca2[:,1], c=Deuc[ref], cmap="viridis", s=18)
ax.scatter(*pca2[ref], s=120, edgecolors="red", facecolors="none", lw=2)
ax.set_title("欧氏距离：直接跨越流形空洞\n（近邻假象）")
plt.colorbar(sc, ax=ax, fraction=0.045); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax = axes[1]
sc = ax.scatter(iso2[:,0], iso2[:,1], c=Gdist[ref], cmap="viridis", s=18)
ax.scatter(*iso2[ref], s=120, edgecolors="red", facecolors="none", lw=2)
ax.set_title("测地距离：沿流形爬行\n（真实的『远近』）")
plt.colorbar(sc, ax=ax, fraction=0.045); ax.set_xlabel("维1"); ax.set_ylabel("维2")
plt.suptitle("欧氏 vs 测地：为什么线性距离在弯曲流形上说谎", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/geodesic.png", dpi=120, bbox_inches="tight"); plt.close()

# =============== FIG 3: k sensitivity of isomap ===============
fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
for ax, k in zip(axes, [3, 6, 12, 40]):
    emb, _ = isomap(Xhd, k=k, dim=2)
    c = best_axis_corr(emb, color)
    ax.scatter(emb[:,0], emb[:,1], c=color, cmap="Spectral", s=12)
    ax.set_title(f"k={k}  轴相关={c:.2f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Isomap 对近邻数 k 极敏感：太小图断裂、太大短路穿洞", y=1.03, fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/k_sensitivity.png", dpi=120, bbox_inches="tight"); plt.close()

# =============== FIG 4: neighborhood preservation bar ===============
fig, ax = plt.subplots(figsize=(6.6, 4.2))
methods = ["PCA", "Isomap"]
vals = [pca_keep, iso_keep]
bars = ax.bar(methods, vals, color=["#3b6ea5", "#e0563b"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=11)
ax.set_ylim(0, 1); ax.set_ylabel("局部近邻保持率 (k=10)")
ax.set_title("谁保住了『谁挨着谁』的局部结构")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/neighbor_preserve.png", dpi=120); plt.close()

print("PCA axis corr:", round(pca_c,3), "Isomap axis corr:", round(iso_c,3))
print("PCA knn keep:", round(pca_keep,3), "Isomap knn keep:", round(iso_keep,3))
for k in [3,6,12,40]:
    emb,_ = isomap(Xhd,k=k,dim=2)
    print("k=",k,"axis corr", round(best_axis_corr(emb,color),3))
print("done", os.listdir(OUT))
