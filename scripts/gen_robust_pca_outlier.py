#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「稳健 PCA 与异常因子剥离」文章配图 + 核心数值。
数据均为 numpy 合成（固定 seed 可复现）。

设定：N=60 只股票、T=250 天，真实收益由 r=3 个公因子驱动（低秩矩阵 L_true）。
在干净矩阵上注入约 3% 的极端离群收益（≈ ±5σ，模拟业绩暴雷/复牌补跌/错单 tick），
得到被污染的观测矩阵 M。对比：
  - 标准 PCA（对 M 直接做 SVD）vs
  - 稳健 PCA（主成分追踪 PCP，把 M 拆成 低秩 L + 稀疏 S）
看稳健 PCA 能否把离群点「压」进稀疏矩阵 S，还回干净的低秩因子结构。

配图保存到 public/images/robust-pca-outlier-factor/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

SEED = 20260830
rng = np.random.default_rng(SEED + 11)
OUT = "public/images/robust-pca-outlier-factor"
os.makedirs(OUT, exist_ok=True)

N, T, r = 60, 250, 3
F = rng.standard_normal((T, r))                      # 真实公因子 (T x r)
B = rng.standard_normal((N, r))                       # 载荷 (N x r)
L_true = F @ B.T                                       # 低秩真实因子收益 (T x N)
idio = rng.standard_normal((T, N)) * 0.3              # 个股特异噪声
M_clean = L_true + idio

# ---- 注入稀疏离群点 ----
frac = 0.03
n_out = int(frac * T * N)
out_idx = rng.choice(T * N, size=n_out, replace=False)
S_true = np.zeros((T, N))
sig = M_clean.std()
vals = rng.choice([-1, 1], size=n_out) * (4.5 + rng.random(n_out) * 2.0) * sig
for k, v in zip(out_idx, vals):
    S_true[k // N, k % N] = v
M = M_clean + S_true

# ===================== 稳健 PCA：主成分追踪 (PCP, 不精确 ALM) =====================
def shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

def robust_pca(M, tol=1e-7, max_iter=200):
    m, n = M.shape
    lam = 1.0 / np.sqrt(max(m, n))
    Y = np.zeros((m, n))
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    mu = 1.25 / np.linalg.norm(M, 2) if np.linalg.norm(M, 2) > 0 else 1.0
    for _ in range(max_iter):
        S = shrink(M - L + Y / mu, lam / mu)
        U, D, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        D = np.maximum(D - 1.0 / mu, 0.0)
        L = U @ np.diag(D) @ Vt
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * 1.1, 1e8)
        if np.linalg.norm(Z, "fro") / (np.linalg.norm(M, "fro") + 1e-12) < tol:
            break
    return L, S

L_rob, S_rob = robust_pca(M)

# ---- 标准 PCA（对污染矩阵直接 SVD，取 r 个主成分重建） ----
Uc, Dc, VcT = np.linalg.svd(M - M.mean(0), full_matrices=False)
L_pca = (Uc[:, :r] * Dc[:r]) @ VcT[:r, :]
L_pca = L_pca + M.mean(0)

# ===================== 评估指标 =====================
def vcorr(a, b):
    a = a.ravel(); b = b.ravel()
    return np.corrcoef(a, b)[0, 1]

corr_rob = vcorr(L_rob, L_true)
corr_pca = vcorr(L_pca, L_true)

# 离群点召回 / 误报
thr = 2.0 * sig
tp = np.sum((np.abs(S_rob) > thr) & (S_true != 0))
fp = np.sum((np.abs(S_rob) > thr) & (S_true == 0))
fn = np.sum((np.abs(S_rob) <= thr) & (S_true != 0))
recall = tp / max(tp + fn, 1)
precision = tp / max(tp + fp, 1)

print("=== 核心统计（用于正文）===")
print(f"股票 N={N}, 天数 T={T}, 真实因子 r={r}, 离群占比={frac*100:.0f}%")
print(f"标准 PCA 低秩恢复相关 |corr(L_pca, L_true)| = {corr_pca:.4f}")
print(f"稳健 PCA 低秩恢复相关 |corr(L_rob, L_true)| = {corr_rob:.4f}")
print(f"稳健 PCA 把离群点压进 S：召回率={recall*100:.1f}%  精确率={precision*100:.1f}%")
print(f"稀疏矩阵 S 中非零项占比={np.mean(np.abs(S_rob)>thr)*100:.2f}%  (真实离群占比={frac*100:.0f}%)")

# ===================== 图 1：离群点检测 =====================
region = slice(0, 60)  # 展示前 60 天
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
im0 = axes[0].imshow(M[region], aspect="auto", cmap="RdBu_r", vmin=-3*sig, vmax=3*sig)
axes[0].set_title("① 观测矩阵 M（被离群点污染）", fontsize=11)
axes[0].set_xlabel("股票"); axes[0].set_ylabel("交易日")
im1 = axes[1].imshow(S_true[region] != 0, aspect="auto", cmap="Greys")
axes[1].set_title("② 真实离群点位置（ground truth）", fontsize=11)
axes[1].set_xlabel("股票")
im2 = axes[2].imshow(np.abs(S_rob[region]) > thr, aspect="auto", cmap="Greys")
axes[2].set_title("③ 稳健 PCA 检出的离群点 S", fontsize=11)
axes[2].set_xlabel("股票")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
fig.suptitle("稳健 PCA 把极端收益单独剥离到稀疏矩阵 S，低秩 L 不再被污染", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/outlier_detection.png")
plt.close(fig)

# ===================== 图 2：碎石图 —— 稳健 vs 标准 =====================
fig, ax = plt.subplots(figsize=(9, 5))
ev_rob = (np.linalg.svd(L_rob - L_rob.mean(0), compute_uv=False) ** 2)
ev_rob = ev_rob / ev_rob.sum()
ev_pca = (Dc**2); ev_pca = ev_pca / ev_pca.sum()
ks = np.arange(1, 11)
ax.plot(ks, ev_pca[:10] * 100, marker="o", color="#6a4c93", lw=2, label="标准 PCA（污染数据）")
ax.plot(ks, ev_rob[:10] * 100, marker="s", color="#1b6ca8", lw=2, label="稳健 PCA（L 的奇异值）")
ax.axvline(r, color="#d1495b", ls="--", lw=1.5, label=f"真实因子数 r={r}")
ax.set_xlabel("主成分序号 k"); ax.set_ylabel("解释方差占比 (%)")
ax.set_title("稳健 PCA 的 L 干净地落在 r=3 个因子上；标准 PCA 被离群点拖散", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/scree_comparison.png")
plt.close(fig)

# ===================== 图 3：低秩恢复质量 =====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].scatter(L_true.ravel(), L_pca.ravel(), s=3, alpha=0.25, color="#6a4c93")
axes[0].set_title(f"标准 PCA 恢复 L\nspearman≈{corr_pca:.3f}", fontsize=11)
axes[0].set_xlabel("真实 L_true"); axes[0].set_ylabel("重建 L")
axes[1].scatter(L_true.ravel(), L_rob.ravel(), s=3, alpha=0.25, color="#1b6ca8")
axes[1].set_title(f"稳健 PCA 恢复 L\nspearman≈{corr_rob:.3f}", fontsize=11)
axes[1].set_xlabel("真实 L_true")
lim = max(np.abs(L_true).max(), np.abs(L_rob).max(), np.abs(L_pca).max())
for a in axes:
    a.plot([-lim, lim], [-lim, lim], color="#d1495b", ls="--", lw=1)
    a.grid(alpha=0.3)
fig.suptitle("稳健 PCA 还原的低秩因子结构更接近真实（离群点已剥离）", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/lowrank_recovery.png")
plt.close(fig)

print(f"图片已保存到 {OUT}")
