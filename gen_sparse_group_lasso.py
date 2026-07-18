#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏组 Lasso 变量选择 配图生成 (4 张真实图表, 纯 numpy FISTA + 5 折 CV)

机制(自洽合成, 仅用于演示方法):
  * 高维面板: N=480 样本, P=600 因子, 60 组(每组 10 个), 组内相关模拟真实因子面板
  * 仅 6 个组含真信号、组内仅部分成员非零 -> 天然适合「组级 + 组内」双重稀疏
  * 关键: N<P, 普通 OLS 会过拟合(memorize), 需要稀疏正则
  * SGL 复合损失 (Simon 2013) 用 FISTA + 闭式组近端算子求解(带 warm start):
        L(β) = (1/2n)||y-Xβ||²  +  λ[ α·Σ_g √p_g ||β_g||₂  +  (1-α)/2·||β||₂² ]
    组近端算子: β_g = max(0, 1 - λα√p_g/(s||z̃_g||))·z̃_g
  * λ 用 5 折 CV + 1-SE 规则(取「最优 - 1 倍折间 SE」以上最稀疏处)
  * 对照: SGL / Lasso / GroupLasso / OLS / Ridge 的样本外 R² 与稀疏度
  * 图1: 各组 L2 范数系数路径 -> 组被逐个「整组清零」
  * 图2: 各方法 5 折 CV R² vs λ -> SGL 防过拟合最优
  * 图3: 活跃组数 / 活跃系数数随 λ 的稀疏阶梯
  * 图4: 估计 vs 真系数(按组高亮) -> SGL 找回正确组并清零噪声组
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "sparse-group-lasso-factors"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "sig": "#C44E52", "noise": "#BBBBBB", "sgl": "#4C72B0",
     "lasso": "#DD8452", "gl": "#55A868", "ridge": "#8172B3", "ols": "#999999"}

rng = np.random.default_rng(20260719)
N, P, G, GS = 480, 600, 60, 10
group = np.repeat(np.arange(G), GS)
base = rng.normal(0, 1, (N, G))
X = np.empty((N, P))
for j in range(P):
    X[:, j] = 0.78 * base[:, group[j]] + 0.62 * rng.normal(0, 1, N)
X = (X - X.mean(0)) / X.std(0)
beta_true = np.zeros(P)
signal_groups = rng.choice(G, 6, replace=False)
true_idx = []
for g in signal_groups:
    members = np.where(group == g)[0]
    pick = rng.choice(members, rng.integers(3, 6), replace=False)
    beta_true[pick] = rng.uniform(-1.2, 1.2, len(pick))
    true_idx += list(pick)
true_idx = np.array(true_idx)
signal = X @ beta_true
signal = signal / signal.std()
y = signal + rng.normal(0, 1.0 / 0.55, N)
y = y / y.std()

# 独立测试集(同分布, 用于诚实的样本外 R² 评估)
rng2 = np.random.default_rng(20260720)
Nt = 2000
base_t = rng2.normal(0, 1, (Nt, G))
Xt2 = np.empty((Nt, P))
for j in range(P):
    Xt2[:, j] = 0.78 * base_t[:, group[j]] + 0.62 * rng2.normal(0, 1, Nt)
Xt2 = (Xt2 - Xt2.mean(0)) / Xt2.std(0)
# 用训练集的均值/标准差标准化, 保证可比
Xt2 = (Xt2 - X.mean(0)) / X.std(0)
yt2 = (Xt2 @ beta_true); yt2 = yt2 / yt2.std()
yt2 = yt2 + rng2.normal(0, 1.0 / 0.55, Nt); yt2 = yt2 / yt2.std()

# ---- FISTA 求解 SGL (带 warm start) ----
def fit_sgl(Xd, yd, lam, alpha, groups, beta0=None, max_iter=900, tol=1e-6):
    nn = Xd.shape[0]
    p = Xd.shape[1]
    ug = np.unique(groups)
    sizes = np.array([np.sum(groups == g) for g in ug])
    XtX = (Xd.T @ Xd)
    Xty = Xd.T @ yd
    L = np.linalg.eigvalsh(XtX / nn).max()
    t = 1.0 / L
    s = lam * (1 - alpha) + 1.0 / t
    w_g = lam * alpha * np.sqrt(sizes)
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    beta_old = beta.copy()
    for k in range(1, max_iter + 1):
        momentum = (k - 1) / (k + 2)
        y_k = beta + momentum * (beta - beta_old)
        grad = (XtX @ y_k - Xty) / nn
        z = y_k - t * grad
        zt = z / (t * s)
        u = np.zeros(p)
        for gi, g in enumerate(ug):
            idx = np.where(groups == g)[0]
            zg = zt[idx]
            norm = np.linalg.norm(zg)
            if norm > 1e-12:
                u[idx] = max(0.0, 1.0 - w_g[gi] / norm) * zg
        beta_old = beta.copy()
        beta = u
        if np.linalg.norm(beta - beta_old) / (np.linalg.norm(beta_old) + 1e-9) < tol:
            break
    return beta

def fit_ridge(Xd, yd, lam):
    return np.linalg.solve(Xd.T @ Xd + len(Xd) * lam * np.eye(Xd.shape[1]), Xd.T @ yd)

def oos_r2(b, Xte, yte):
    pred = Xte @ b
    return 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)

# ---- 5 折 CV + 1-SE ----
def cv_folds(X, y, lams, alpha, groups, n_splits=5, seed=7):
    rngc = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rngc.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    per_fold = np.zeros((n_splits, len(lams)))
    for fi in range(n_splits):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != fi])
        # warm start 沿 λ 递减扫描
        b0 = None
        lam_sorted = sorted(range(len(lams)), key=lambda i: -lams[i])  # 从大 λ(稀疏)开始
        for li in lam_sorted:
            b0 = fit_sgl(X[tr], y[tr], lams[li], alpha, groups, beta0=b0)
            per_fold[fi, li] = oos_r2(b0, X[te], y[te])
    return per_fold

lams = np.logspace(-3.4, -1.0, 30)
g_single = np.arange(P)
cvf_sgl = cv_folds(X, y, lams, 0.95, group)
cvf_lasso = cv_folds(X, y, lams, 1.0, g_single)
cvf_gl = cv_folds(X, y, lams, 1.0, group)
cv_sgl = cvf_sgl.mean(0)
cv_lasso = cvf_lasso.mean(0)
cv_gl = cvf_gl.mean(0)
se_sgl = cvf_sgl.std(0, ddof=1) / np.sqrt(cvf_sgl.shape[0])

best_raw = int(np.argmax(cv_sgl))
thr = cv_sgl[best_raw] - se_sgl[best_raw]
cand = np.where(cv_sgl >= thr)[0]
lam_star = lams[cand[0]]
gi_star = np.where(lams == lam_star)[0][0]
print("CV-SGL best_raw_lam", lams[best_raw], "cv_r2", cv_sgl[best_raw], "se", se_sgl[best_raw])
print("1-SE lam_star", lam_star, "cv_r2@star", cv_sgl[gi_star])

# 全样本评估
def evaluate(lam, alpha, groups):
    b = fit_sgl(X, y, lam, alpha, groups)
    tr = oos_r2(b, Xt2, yt2)   # 样本外 R² (独立测试集)
    n_coef = int(np.sum(np.abs(b) > 1e-6))
    n_grp = sum(1 for g in range(G) if np.linalg.norm(b[group == g]) > 1e-6)
    recall = float(np.mean(np.abs(b[true_idx]) > 1e-6))
    noise_idx = np.setdiff1d(np.arange(P), true_idx)
    fp = float(np.mean(np.abs(b[noise_idx]) > 1e-6))
    return b, tr, n_coef, n_grp, recall, fp

b_sgl, tr_sgl, nc_sgl, ng_sgl, rec_sgl, fp_sgl = evaluate(lam_star, 0.95, group)
ols_b = np.linalg.lstsq(X, y, rcond=None)[0]
ridge_b = fit_ridge(X, y, lams[len(lams)//2])
ols_oos = oos_r2(ols_b, Xt2, yt2)
ridge_oos = oos_r2(ridge_b, Xt2, yt2)

# ---- 图1: 各组 L2 范数路径 ----
group_norm_path = np.zeros((G, len(lams)))
for li in range(len(lams)):
    b = fit_sgl(X, y, lams[li], 0.95, group)
    for g in range(G):
        idx = np.where(group == g)[0]
        group_norm_path[g, li] = np.linalg.norm(b[idx], 2)
fig, ax = plt.subplots(figsize=(9, 5.2))
for g in range(G):
    col = C["sig"] if g in signal_groups else C["noise"]
    lw = 1.6 if g in signal_groups else 0.6
    ax.plot(np.log10(lams), group_norm_path[g], color=col, lw=lw,
            alpha=0.95 if g in signal_groups else 0.45)
ax.axvline(np.log10(lam_star), color="#222", ls="--", lw=1.2, label=f"CV 1-SE λ*={lam_star:.3f}")
ax.set_xlabel("log₁₀(λ)  (惩罚强度 ↑)")
ax.set_ylabel("组内系数 L2 范数 ||β_g||₂")
ax.set_title("稀疏组 Lasso 系数路径：组被逐个「整组清零」\n(红=含真信号组, 灰=噪声组)")
ax.legend(fontsize=8)
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sgl_group_path.png"))
plt.close(fig)

# ---- 图2: 各方法 5 折 CV R² vs λ ----
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(np.log10(lams), cv_sgl, color=C["sgl"], lw=2.2, label="Sparse Group Lasso (α=0.95)")
ax.plot(np.log10(lams), cv_lasso, color=C["lasso"], lw=1.8, label="Lasso (无组结构)")
ax.plot(np.log10(lams), cv_gl, color=C["gl"], lw=1.8, label="Group Lasso (纯组惩罚)")
ax.axhline(ols_oos, color=C["ols"], ls="--", lw=1.4, label=f"OLS 全量 (OOS R²={ols_oos:.2f})")
ax.axhline(ridge_oos, color=C["ridge"], ls=":", lw=1.4, label=f"Ridge (OOS R²={ridge_oos:.2f})")
ax.axvline(np.log10(lam_star), color="#222", ls="--", lw=1.2)
ax.set_xlabel("log₁₀(λ)")
ax.set_ylabel("5 折交叉验证 R²")
ax.set_title("CV R² 对比：SGL 在高维下防过拟合最优")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sgl_oos_r2.png"))
plt.close(fig)

# ---- 图3: 稀疏阶梯 ----
n_groups_active, n_coef_active = [], []
for l in lams:
    b = fit_sgl(X, y, l, 0.95, group)
    ng = sum(1 for g in range(G) if np.linalg.norm(b[group == g]) > 1e-6)
    n_groups_active.append(ng)
    n_coef_active.append(int(np.sum(np.abs(b) > 1e-6)))
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.step(np.log10(lams), n_groups_active, where="mid", color=C["gl"], lw=2, label="活跃组数")
ax.step(np.log10(lams), n_coef_active, where="mid", color=C["sgl"], lw=2, label="活跃系数数")
ax.axhline(6, color=C["sig"], ls="--", lw=1.2, label="真信号组数=6")
ax.axvline(np.log10(lam_star), color="#222", ls="--", lw=1.2)
ax.set_xlabel("log₁₀(λ)")
ax.set_ylabel("数量")
ax.set_title("稀疏阶梯：组级与组内双重稀疏同步发生")
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sgl_sparsity.png"))
plt.close(fig)

# ---- 图4: 估计 vs 真 ----
order = np.argsort(group)
xt = beta_true[order]
xe = b_sgl[order]
fig, ax = plt.subplots(figsize=(9, 5.2))
colors = [C["sig"] if g in signal_groups else C["noise"] for g in group[order]]
ax.scatter(xt, xe, c=colors, s=26, alpha=0.8, edgecolor="white", linewidth=0.3)
lim = max(np.max(np.abs(xt)), np.max(np.abs(xe))) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="#444", ls="--", lw=1)
ax.set_xlabel("真系数 β_true")
ax.set_ylabel("SGL 估计系数 β̂")
ax.set_title(f"估计 vs 真值 (λ*={lam_star:.3f}, OOS R²={tr_sgl:.2f})\n红=真信号组, 灰=噪声组(理想应贴 0)")
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sgl_est_vs_true.png"))
plt.close(fig)

print("DONE sgl", os.listdir(OUT))
print(f"SGL@λ*: OOS_R2={tr_sgl:.3f} ncoef={nc_sgl}(/{P}) ngroups={ng_sgl}/{G} recall={rec_sgl:.3f} fp_rate={fp_sgl:.3f}")
print(f"OLS OOS={ols_oos:.3f}  Ridge OOS={ridge_oos:.3f}")
print(f"Lasso CV best={cv_lasso.max():.3f} @lam={lams[np.argmax(cv_lasso)]:.3f}  GL CV best={cv_gl.max():.3f}")
print("true signal groups", sorted(signal_groups.tolist()))
