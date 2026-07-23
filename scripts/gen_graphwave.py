import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(11)
OUT = "public/images/graph-wave-net-financial"
os.makedirs(OUT, exist_ok=True)

# ---- build a 2-block asset graph (two sectors) ----
N = 24
block = np.array([0]*12 + [1]*12)
pos = np.zeros((N, 2))
ang = np.linspace(0, 2*np.pi, 12, endpoint=False)
pos[:12] = np.c_[np.cos(ang)-1.6, np.sin(ang)]
pos[12:] = np.c_[np.cos(ang)+1.6, np.sin(ang)]
# adjacency: strong within block, weak across
A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j: continue
        p = 0.55 if block[i]==block[j] else 0.06
        if rng.random() < p:
            A[i, j] = A[j, i] = 1
D = np.diag(A.sum(1))
Dinv = np.diag(1/np.sqrt(np.maximum(A.sum(1),1e-9)))
L = np.eye(N) - Dinv @ A @ Dinv     # normalized laplacian
lam, U = np.linalg.eigh(L)

# 1. the graph
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for i in range(N):
    for j in range(i+1, N):
        if A[i, j]:
            ax.plot(*zip(pos[i], pos[j]), color="#bbb", lw=0.6, zorder=1)
for b, c in [(0, "#3b6ea5"), (1, "#e0563b")]:
    m = block == b
    ax.scatter(pos[m,0], pos[m,1], s=120, color=c, edgecolors="white", zorder=2,
               label=f"板块 {b+1}")
ax.set_title("24 资产两板块关联图（块内密、块间疏）")
ax.axis("off"); ax.legend(fontsize=9, loc="upper center", ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/asset_graph.png", dpi=120); plt.close()

# 2. graph wavelet vs spectral eigenbasis: heat-kernel wavelet g(s*lam)=exp(-s*lam)
def wavelet(s):  # returns Psi_s = U diag(exp(-s lam)) U^T
    return U @ np.diag(np.exp(-s*lam)) @ U.T
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
node = 3
for ax, s in zip(axes, [1.0, 3.0, 8.0]):
    psi = wavelet(s)[node]
    sc = ax.scatter(pos[:,0], pos[:,1], c=psi, cmap="coolwarm", s=90,
                    vmin=-abs(psi).max(), vmax=abs(psi).max(), edgecolors="k", lw=0.3)
    for i in range(N):
        for j in range(i+1, N):
            if A[i,j]: ax.plot(*zip(pos[i],pos[j]), color="#ddd", lw=0.4, zorder=0)
    ax.scatter(*pos[node], s=180, facecolors="none", edgecolors="lime", lw=2)
    ax.set_title(f"尺度 s={s}"); ax.axis("off")
fig.suptitle("图小波（热核）以节点3为中心：尺度越大，能量扩散越广")
plt.tight_layout(); plt.savefig(f"{OUT}/graph_wavelet_scales.png", dpi=120); plt.close()

# 3. spectrum: eigenvalues + wavelet filters in spectral domain
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(lam, "o-", color="#444", ms=3, label="拉普拉斯特征值 λ")
for s, c in [(1.0,"#3b6ea5"), (3.0,"#e0a13b"), (8.0,"#e0563b")]:
    ax.plot(np.exp(-s*lam), "--", color=c, lw=1.4, label=f"小波核 exp(-{s}λ)")
ax.set_title("谱域视角：图小波=对每个图频率 λ 施加带通/低通增益")
ax.set_xlabel("特征值序号（图频率由低到高）"); ax.set_ylabel("增益 / λ 值")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/spectral_filters.png", dpi=120); plt.close()

# ---- denoising task: signal on graph = block-constant + noise; wavelet smoothing ----
true = block.astype(float)*2 - 1        # +1 / -1 per block
obs = true + 0.9*rng.standard_normal(N)
# graph-wavelet low-pass: keep low frequencies (small lam)
def gwave_denoise(sig, s=4.0):
    hs = U.T @ sig
    hs = hs * np.exp(-s*lam)            # attenuate high graph-freq
    return U @ hs
den = gwave_denoise(obs, s=4.0)
# local GCN-style smoothing for comparison
Ahat = A + np.eye(N); Dh = np.diag(1/np.sqrt(Ahat.sum(1)))
S = Dh @ Ahat @ Dh
gcn = S @ S @ obs
def acc(p): return (np.sign(p) == np.sign(true)).mean()
def rmse(p): return np.sqrt(((p-true)**2).mean())

fig, ax = plt.subplots(figsize=(9, 4.0))
idx = np.arange(N)
ax.plot(idx, true, "k-", lw=2, label="真实板块信号")
ax.plot(idx, obs, "o", color="#bbb", ms=5, label=f"观测(含噪) RMSE={rmse(obs):.2f}")
ax.plot(idx, den, "s-", color="#e0563b", ms=4, label=f"图小波去噪 RMSE={rmse(den):.2f}")
ax.plot(idx, gcn, "^--", color="#3b6ea5", ms=4, label=f"局部GCN RMSE={rmse(gcn):.2f}")
ax.axvline(11.5, color="#888", ls=":")
ax.set_title(f"图信号去噪对比（方向准确率：小波 {acc(den):.2f} / GCN {acc(gcn):.2f}）")
ax.set_xlabel("资产编号"); ax.set_ylabel("信号值"); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/denoise_compare.png", dpi=120); plt.close()

print("graphwave images done", rmse(obs), rmse(den), rmse(gcn), acc(den), acc(gcn))
