#!/usr/bin/env python3
"""
为文章「动态 Nelson-Siegel：用三因子状态空间跟踪收益率曲线形变」(dynamic-nelson-siegel) 生成真实配图。

核心逻辑（Dynamic Nelson-Siegel / Diebold & Li 2006, 状态空间实现）：
  yield(τ, t) = β0(t) + β1(t)·(−(τ/λ)·e^{−τ/λ}) + β2(t)·(1 − e^{−τ/λ})·(τ/λ) ... 标准 DNS 用线性短率因子形式：
  本文采用 Diebold-Li 经典三因子：
    yield(τ) = L + S·((1−e^{−τ/λ})/(τ/λ)) + C·((1−e^{−τ/λ})/(τ/λ) − e^{−τ/λ})
    L=水平(长期), S=斜率(短长利差, 负=正常向上), C=曲率(中段隆起)
  - 把 L,S,C 当隐状态，用 AR(1) 状态方程驱动，卡尔曼滤波做一步预测 + 平滑；
  - 因子载荷固定 λ(决定曲率峰值在 2.5 年附近)；只让 L,S,C 随时间演化；
  - 演示：模拟 L/S/C 的 AR(1) 路径 → 生成整条期限结构 → 卡尔曼滤波还原 L/S/C → 对比真实曲线。
  全部为合成但结构贴合真实收益率曲线形态（水平/斜率/曲率 + 随机游走演化），非占位图。
"""
import os
import numpy as np
from scipy.linalg import inv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "dynamic-nelson-siegel")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

# ---------- 期限点（年）----------
TAU = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
LAMBDA = 2.5          # 决定曲率峰值在 τ≈1.5λ=... 实际峰值在 τ=λ 附近（本实现曲率项峰在 τ=2λ）

def loadings(tau, lam=LAMBDA):
    x = tau / lam
    lvl = np.ones_like(tau)                                   # 水平
    slp = (1.0 - np.exp(-x)) / x                             # 斜率（短端高、长端→0）
    cur = slp - np.exp(-x)                                   # 曲率（中段隆起）
    return np.vstack([lvl, slp, cur])                       # (3, n_tau)

B = loadings(TAU)                                            # 固定载荷 (3, 11)

def yield_curve(beta):
    """beta = (L, S, C) → 收益率向量（%）。"""
    return B.T @ beta

# ---------- 模拟 L/S/C 的 AR(1) 演化 ----------
T = 250
beta_true = np.zeros((T, 3))
a = np.array([3.2, -1.0, -0.5])            # 中枢（水平3.2%、斜率-1%、曲率-0.5%）
phi = np.array([0.985, 0.90, 0.85])        # AR(1) 持续性：水平最黏
sig = np.array([0.18, 0.45, 0.35])         # 扰动标准差
b = a * (1 - phi)
b[0] = 3.2
beta = a.copy()
for t in range(T):
    beta = b + phi * beta + rng.normal(0, sig, 3)
    beta_true[t] = beta
# 注入一段「斜率倒挂→修复」（熊平→牛陡）的可见形变，让图更真实
beta_true[120:160, 1] -= np.linspace(0, 1.4, 40)      # 斜率下压（倒挂加深）
beta_true[160:200, 1] += np.linspace(0, 1.6, 40)      # 斜率回升（修复）

Y = np.array([yield_curve(bt) for bt in beta_true])    # (T, 11) 含噪观测前
Y = Y + rng.normal(0, 0.04, Y.shape)                   # 观测噪声（bp 级）

# ---------- 卡尔曼滤波（状态空间：量测=Y, 状态=β, 状态方程 AR(1)）----------
# 状态 x_t = β_t (3,), 量测 Z_t = Y_t = Bᵀ β_t + ε
Fmat = np.diag(phi)                                    # 状态转移
Hmat = B.T                                             # 量测矩阵 (11,3)
Qmat = np.diag(sig**2)                                 # 状态噪声
Rmat = np.eye(len(TAU)) * (0.04**2)                    # 量测噪声

beta_filt = np.zeros((T, 3))
P_filt = np.zeros((T, 3, 3))
x_pred = a.copy(); P_pred = np.diag(np.var(beta_true, axis=0) + 1e-6)
for t in range(T):
    # 预测步（基于上一状态 → 当前先验）
    x_pred = b + Fmat @ x_pred
    P_pred = Fmat @ P_pred @ Fmat.T + Qmat
    # 更新步
    y = Y[t]
    e = y - Hmat @ x_pred
    S = Hmat @ P_pred @ Hmat.T + Rmat
    K = P_pred @ Hmat.T @ inv(S)
    x_upd = x_pred + K @ e
    P_upd = (np.eye(3) - K @ Hmat) @ P_pred
    beta_filt[t] = x_upd
    P_filt[t] = P_upd
    x_pred, P_pred = x_upd, P_upd

# 平滑（RTS，简单实现）
beta_sm = beta_filt.copy()
P_sm = P_filt.copy()
for t in range(T - 2, -1, -1):
    C = P_filt[t] @ Fmat.T @ inv(P_pred if False else (Fmat @ P_filt[t] @ Fmat.T + Qmat))
    beta_sm[t] = beta_filt[t] + C @ (beta_sm[t + 1] - (b + Fmat @ beta_filt[t]))
    P_sm[t] = P_filt[t] + C @ (P_sm[t + 1] - (Fmat @ P_filt[t] @ Fmat.T + Qmat)) @ C.T
# 注意：上面 C 的分母用预测协方差，重算
for t in range(T - 2, -1, -1):
    Pp = Fmat @ P_filt[t] @ Fmat.T + Qmat
    C = P_filt[t] @ Fmat.T @ inv(Pp)
    beta_sm[t] = beta_filt[t] + C @ (beta_sm[t + 1] - (b + Fmat @ beta_filt[t]))
    P_sm[t] = P_filt[t] + C @ (P_sm[t + 1] - Pp) @ C.T

# ===================== 图1：期限结构截面（含卡尔曼还原）=====================
pick = [0, 120, 140, 200]                              # 正常、倒挂最深、过渡、修复
fig, ax = plt.subplots(figsize=(11, 5.8))
colors = ["#4c72b0", "#c44e52", "#dd8452", "#55a868"]
labels = ["t=0 正常向上", "t=120 倒挂加深", "t=140 过渡", "t=200 斜率修复"]
for p, c, lab in zip(pick, colors, labels):
    ax.plot(TAU, Y[p], "o-", color=c, lw=1.8, ms=5, label=f"{lab}")
ax.set_xlabel("期限（年）", fontsize=11)
ax.set_ylabel("收益率（%）", fontsize=11)
ax.set_title("收益率曲线形变：从正常向上到深度倒挂再到修复",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "dns_curve_evolution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 图2：三因子 L/S/C 的真实轨迹 vs 卡尔曼平滑 =====================
fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True)
names = ["水平 β0 (长期)", "斜率 β1 (短长利差)", "曲率 β2 (中段隆起)"]
cols = ["#4c72b0", "#c44e52", "#55a868"]
for k in range(3):
    axes[k].plot(beta_true[:, k], color=cols[k], lw=1.2, alpha=0.45, label="真实 L/S/C")
    axes[k].plot(beta_sm[:, k], color="#333", lw=2.0, label="卡尔曼平滑还原")
    axes[k].set_ylabel("因子值", fontsize=10)
    axes[k].set_title(names[k], fontsize=11, fontweight="bold")
    axes[k].legend(loc="upper right", fontsize=8.5)
    axes[k].grid(True, alpha=0.25)
axes[2].set_xlabel("交易日", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(D, "dns_factors_kalman.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 图3：因子载荷（三因子形状）=====================
fig, ax = plt.subplots(figsize=(11, 5.2))
xs = np.linspace(0.25, 30, 300)
Bl = loadings(xs)
ax.plot(xs, Bl[0], color="#4c72b0", lw=2.2, label="水平载荷（常 1）")
ax.plot(xs, Bl[1], color="#c44e52", lw=2.2, label="斜率载荷（短高长零）")
ax.plot(xs, Bl[2], color="#55a868", lw=2.2, label="曲率载荷（中段隆起）")
ax.axvline(LAMBDA, color="#888", ls=":", lw=1.4, label=f"λ={LAMBDA}（曲率峰值附近）")
ax.set_xlabel("期限（年）", fontsize=11)
ax.set_ylabel("载荷值", fontsize=11)
ax.set_title("DNS 三因子载荷：固定形状、只让系数 L/S/C 随时间游走",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "dns_factor_loadings.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 图4：卡尔曼还原误差（RMSE 随期限）=====================
err = (Y - np.array([yield_curve(bt) for bt in beta_sm]))      # (T,11)
rmse = np.sqrt((err**2).mean(axis=0))
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.bar(range(len(TAU)), rmse, color="#4c72b0", alpha=0.8)
ax.set_xticks(range(len(TAU)))
ax.set_xticklabels([f"{t:g}" for t in TAU])
ax.set_xlabel("期限（年）", fontsize=11)
ax.set_ylabel("卡尔曼还原 RMSE（%）", fontsize=11)
ax.set_title("卡尔曼滤波还原整条曲线的误差：bp 级、期限间平稳",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "dns_rmse.png"), dpi=150, bbox_inches="tight")
plt.close()

# ===================== 诊断 =====================
print("=== DNS 诊断 ===")
print(f"期限点: {TAU}")
print(f"λ={LAMBDA}  因子载荷 B.shape={B.shape}")
print(f"真实 L/S/C 中枢≈ {a}  AR(1) φ={phi}")
print(f"卡尔曼平滑 vs 真实 因子 RMSE: L={np.sqrt(((beta_sm[:,0]-beta_true[:,0])**2).mean()):.4f} "
      f"S={np.sqrt(((beta_sm[:,1]-beta_true[:,1])**2).mean()):.4f} "
      f"C={np.sqrt(((beta_sm[:,2]-beta_true[:,2])**2).mean()):.4f}")
print(f"曲线整体还原 RMSE（bp 级）: 均值={rmse.mean()*100:.2f}bps 最大={rmse.max()*100:.2f}bps")
print(f"生成图片: {sorted(os.listdir(D))}")
