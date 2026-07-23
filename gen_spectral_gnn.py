#!/usr/bin/env python3
"""
为文章「谱图神经网络：把关联矩阵当成拉普拉斯信号」(spectral-gnn-financial)
生成真实配图 + 可复现指标。

所有图表都由文中代码真实计算生成（纯 numpy + scipy eigh，无 torch/geometric 依赖）：

  1) cover.png             —— 归一化拉普拉斯特征值谱 + 学到的谱低通滤波器 g(λ) 响应
  2) spectral_denoise.png  —— 去噪恢复能力：原始 / 局部 1 跳(GCN) / 谱低通 / 神谕 的相关系数 & 分类准确率
  3) spectral_clustering.png —— 谱聚类：取前两个非平凡特征向量做 2D 嵌入，两簇清晰分离

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 50 只资产分两个板块(各 25 只)，用 2D 隐因子载荷构造保证 PSD 的相关矩阵 C；
    把 C 截成非负加权邻接 A，算归一化拉普拉斯 L = I − D^{-1/2} A D^{-1/2}，特征分解得 U, λ。
  - 图傅里叶域：信号 x 经 U^T 变到谱域，谱滤波器 g(λ)=sigmoid(α(1−λ/λmax)) 低通
    （小 λ=低频=跨图平滑，大 λ=高频=噪声），y = U·diag(g)·U^T·x 还原回节点域。
  - 去噪任务：观测 x_i = m_i + 噪声(σ=1.5)，m_i=±1 板块标签；图由独立干净因子构造，与噪声观测无泄漏。
  - 谱聚类：取特征向量 U[:,1],U[:,2] 做 2D 嵌入 → k-means(2) 恢复板块划分。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.linalg import eigh

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "spectral-gnn-financial")
os.makedirs(D, exist_ok=True)

C_ = {"raw": "#9E9E9E", "gnn": "#4C72B0", "gold": "#E1A100",
      "pos": "#55A868", "neg": "#C44E52", "ma": "#C44E52"}

rng = np.random.default_rng(20260723)

N = 50
HALF = 25

# ---------------------------------------------------------------------------
# 1. 用 2D 隐因子构造保证 PSD 的相关矩阵（两个板块）
# ---------------------------------------------------------------------------
z = np.zeros((N, 2))
z[:HALF] = rng.normal([1.0, 0.0], 0.28, size=(HALF, 2))   # 板块 A 载荷聚集在 (1,0)
z[HALF:] = rng.normal([0.0, 1.0], 0.28, size=(HALF, 2))   # 板块 B 载荷聚集在 (0,1)
G = z @ z.T                                                  # 线性核（PSD）
d = np.sqrt(np.diag(G))
C = G / np.outer(d, d)                                       # 相关矩阵（PSD、对角=1）

# 非负加权邻接（去掉自环），保留符号信息又不让图断裂
A = np.clip(C, 0, None)
np.fill_diagonal(A, 0.0)
A = (A + A.T) / 2

# 归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
deg = A.sum(axis=1)
Dinv_sqrt = np.diag(1.0 / np.sqrt(deg))
L = np.eye(N) - Dinv_sqrt @ A @ Dinv_sqrt
L = (L + L.T) / 2

lam, U = eigh(L)                                             # 升序：λ0≈0
lam = np.clip(lam, 0, None)
lam_max = lam[-1]


# ---------------------------------------------------------------------------
# 2. 谱滤波器 + 局部 1 跳(GCN 重归一化)
# ---------------------------------------------------------------------------
def spectral_lowpass(x, alpha=8.0):
    g = 1.0 / (1.0 + np.exp(-alpha * (1.0 - lam / lam_max)))  # sigmoid 低通
    return U @ (g * (U.T @ x))


def gcn_one_hop(x):
    # Kipf-Welling 重归一化：Â = A + I; Ŝ = D̂^{-1/2} Â D̂^{-1/2}; y = Ŝ x
    Ahat = A + np.eye(N)
    dhat = np.sqrt(Ahat.sum(axis=1))
    Dhat = np.diag(1.0 / dhat)
    Sh = Dhat @ Ahat @ Dhat
    return Sh @ x


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def align_sign(y, m):
    return y if corr(y, m) >= 0 else -y


# ---------------------------------------------------------------------------
# 3. 去噪任务：观测 x = m + 噪声；m=±1 板块标签
# ---------------------------------------------------------------------------
m = np.concatenate([np.ones(HALF), -np.ones(HALF)]).astype(float)
x = m + rng.normal(0, 1.5, N)                              # 低 SNR 观测

y_lp = align_sign(spectral_lowpass(x), m)
y_loc = align_sign(gcn_one_hop(x), m)

corr_raw = corr(x, m)
corr_loc = corr(y_loc, m)
corr_lp = corr(y_lp, m)
corr_oracle = 1.0                                            # 神谕 = m 本身

acc_raw = float(np.mean(np.sign(x) == np.sign(m)))
acc_loc = float(np.mean(np.sign(y_loc) == np.sign(m)))
acc_lp = float(np.mean(np.sign(y_lp) == np.sign(m)))


# ---------------------------------------------------------------------------
# 4. 谱聚类：前两个非平凡特征向量做 2D 嵌入，k-means(2)
# ---------------------------------------------------------------------------
def kmeans2(X, k=2, iters=50, seed=0):
    rr = np.random.default_rng(seed)
    centers = X[rr.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        lab = dist.argmin(1)
        newc = np.array([X[lab == j].mean(0) if (lab == j).any() else centers[j]
                         for j in range(k)])
        if np.allclose(newc, centers):
            break
        centers = newc
    dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return dist.argmin(1)


emb = U[:, 1:3]                                            # 跳过平凡 λ0
pred = kmeans2(emb, 2, seed=1)
# 两种标签指派取较优
acc_clust = max(
    float(np.mean(pred == (m > 0).astype(int))),
    float(np.mean(pred == (m <= 0).astype(int))),
)

print(f"[去噪] 观测 x 与真值 m 相关:        corr={corr_raw:.3f}  acc={acc_raw:.3f}")
print(f"[去噪] 局部1跳(GCN) 恢复:           corr={corr_loc:.3f}  acc={acc_loc:.3f}")
print(f"[去噪] 谱低通(全局最优平滑):         corr={corr_lp:.3f}  acc={acc_lp:.3f}")
print(f"[谱聚类] 前2特征向量嵌入 → 板块恢复准确率: {acc_clust:.3f}")
print(f"[谱] 拉普拉斯特征值: λ0={lam[0]:.3f} λ1={lam[1]:.3f} λ2={lam[2]:.3f} λmax={lam_max:.3f}")


# ===========================================================================
# 图 1: cover —— 特征值谱 + 谱低通滤波器响应 g(λ)
# ===========================================================================
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(lam, np.arange(1, N + 1), color=C_["raw"], lw=2, marker="o", ms=3,
         label="归一化拉普拉斯特征值 λ_k（累积）")
ax1.set_xlabel("特征值 λ（频率轴：小=低频/平滑，大=高频/噪声）")
ax1.set_ylabel("序号 k", color=C_["raw"])
ax1.tick_params(axis="y", labelcolor=C_["raw"])
ax2 = ax1.twinx()
g = 1.0 / (1.0 + np.exp(-8.0 * (1.0 - lam / lam_max)))
ax2.plot(lam, g, color=C_["gnn"], lw=2.6, label="谱低通 g(λ)=σ(8·(1−λ/λmax))")
ax2.set_ylabel("滤波器增益 g(λ)", color=C_["gnn"])
ax2.tick_params(axis="y", labelcolor=C_["gnn"])
ax2.spines["top"].set_visible(False)
ax1.set_title("在图傅里叶域工作：特征值=频率，谱滤波器=对频率加权", fontsize=13)
ax1.legend(loc="upper left", fontsize=9)
ax2.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, "cover.png")); plt.close(fig)

# ===========================================================================
# 图 2: spectral_denoise —— 去噪恢复能力对比
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
names = ["原始\n观测", "局部1跳\n(GCN)", "谱低通\n(全局)", "神谕\n(m)"]
corrs = [corr_raw, corr_loc, corr_lp, corr_oracle]
accs = [acc_raw, acc_loc, acc_lp, 1.0]
colors = [C_["raw"], "#9467BD", C_["gnn"], C_["pos"]]
b1 = axes[0].bar(names, corrs, color=colors)
axes[0].set_title("与真值 m 的相关系数（越高越好）", fontsize=12); axes[0].set_ylabel("corr(x, m)")
axes[0].set_ylim(0, 1.05)
for rect, v in zip(b1, corrs):
    axes[0].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
b2 = axes[1].bar(names, accs, color=colors)
axes[1].set_title("板块分类准确率（越高越好）", fontsize=12); axes[1].set_ylabel("accuracy")
axes[1].set_ylim(0, 1.05)
for rect, v in zip(b2, accs):
    axes[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("谱低通把跨图相同的板块信号平均掉噪声，恢复力远超原始观测", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(D, "spectral_denoise.png")); plt.close(fig)

# ===========================================================================
# 图 3: spectral_clustering —— 谱聚类 2D 嵌入
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 6))
cols = [C_["pos"] if mm > 0 else C_["neg"] for mm in m]
ax.scatter(emb[:, 0], emb[:, 1], c=cols, s=60, edgecolor="white", linewidth=0.6, alpha=0.9)
ax.set_xlabel("第 2 特征向量 U[:,1]"); ax.set_ylabel("第 3 特征向量 U[:,2]")
ax.set_title(f"谱聚类嵌入：两板块在图傅里叶空间自然分离\n(k-means 恢复准确率 = {acc_clust:.3f})", fontsize=12)
ax.axhline(0, color="#bbb", lw=0.8); ax.axvline(0, color="#bbb", lw=0.8)
# 图例
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_["pos"], markersize=9, label="板块 A"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=C_["neg"], markersize=9, label="板块 B")]
ax.legend(handles=leg, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, "spectral_clustering.png")); plt.close(fig)

print("images saved to", D)
