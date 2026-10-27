#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「主成分套期保值：用少数公因子把组合暴露降到最低方差对冲」文章配图 + 核心数值。
数据均为 numpy 合成（固定 seed 可复现）。
核心设定：被套保资产由 3 个真实公因子驱动；候选对冲工具共 K=200 个，
但只有前 3 个主成分承载真实信息，其余 197 个都是噪声 —— 这正是 PCA 降维
比 OLS 全解更稳健的场景（高维 + 噪声 + 小样本 T=150）。
配图保存到 public/images/pca-minimum-variance-hedge/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang TC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

SEED = 20260830
rng = np.random.default_rng(SEED + 1)
OUT = "public/images/pca-minimum-variance-hedge"
os.makedirs(OUT, exist_ok=True)

T, K = 150, 200
F = rng.standard_normal((T, 3))                       # 3 个真实公因子
B_hedge = rng.standard_normal((K, 3)) * 0.7          # 工具对公因子的载荷
B_target = rng.standard_normal((1, 3)) * 0.7
eps_t = rng.standard_normal((T, K)) * 0.6             # 工具特异性噪声（较强）
eps_y = rng.standard_normal((T, 1)) * 0.2
X = F @ B_hedge.T + eps_t                            # T x K  候选对冲工具（高维噪声）
y = (F @ B_target.T + eps_y).ravel()                 # 被套保资产

# ---- 最小方差对冲（OLS 全解，无约束，K=200 个工具） ----
XtX = X.T @ X
Xty = X.T @ y
beta_ols = np.linalg.solve(XtX, Xty)
var_ols = np.var(y - X @ beta_ols)
var_y = np.var(y)

# ---- PCA 降维：只取前 p 个主成分 ----
U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
var_expl = (S**2) / (S**2).sum()
cum_var = np.cumsum(var_expl)
p95 = int(np.searchsorted(cum_var, 0.95) + 1)
pcs = U[:, :p95] * S[:p95]
P = np.linalg.solve(pcs.T @ pcs, pcs.T @ y)
var_pca = np.var(y - pcs @ P)

print("=== 核心统计（用于正文）===")
print(f"工具数 K={K}, 样本 T={T}, 真实因子数=3")
print(f"OLS 全工具对冲: 残余方差={var_ols:.4f} 对冲率={(1-var_ols/var_y)*100:.1f}%  非零权重={int((np.abs(beta_ols)>1e-4).sum())}/{K}")
print(f"PCA 前 {p95} 主成分对冲: 残余方差={var_pca:.4f} 对冲率={(1-var_pca/var_y)*100:.1f}%")

# ---- 不同 p 下的残余方差 ----
ps = list(range(1, K + 1, 5))
resid_vars, r2s = [], []
for p in ps:
    pc = U[:, :p] * S[:p]
    Pb = np.linalg.solve(pc.T @ pc, pc.T @ y)
    resid_vars.append(np.var(y - pc @ Pb))
    r2s.append(1 - np.var(y - pc @ Pb) / var_y)
resid_vars, r2s = np.array(resid_vars), np.array(r2s)

# ---- 样本内 vs 样本外（关键：OLS 过拟合，PCA 不） ----
split = T // 2
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
U_tr, S_tr, Vt_tr = np.linalg.svd(X_tr - X_tr.mean(0), full_matrices=False)
# OLS 全工具
b_tr = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ y_tr)
r2_ols_is = 1 - np.var(y_tr - X_tr @ b_tr) / np.var(y_tr)
r2_ols_oos = 1 - np.var(y_te - X_te @ b_tr) / np.var(y_te)
# PCA p95
pc_tr = U_tr[:, :p95] * S_tr[:p95]
P_tr = np.linalg.solve(pc_tr.T @ pc_tr, pc_tr.T @ y_tr)
pc_te = (X_te - X_tr.mean(0)) @ Vt_tr[:p95].T
r2_pca_is = 1 - np.var(y_tr - pc_tr @ P_tr) / np.var(y_tr)
r2_pca_oos = 1 - np.var(y_te - pc_te @ P_tr) / np.var(y_te)

print(f"OLS   样本内R²={r2_ols_is:.4f}  样本外R²={r2_ols_oos:.4f}  过拟合差={r2_ols_is-r2_ols_oos:.4f}")
print(f"PCA   样本内R²={r2_pca_is:.4f}  样本外R²={r2_pca_oos:.4f}  过拟合差={r2_pca_is-r2_pca_oos:.4f}")

# ============ 图 1：残余方差 vs p + 累计解释方差 ============
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(ps, resid_vars, marker="o", color="#1b6ca8", lw=2, label="对冲后残余方差")
ax1.axvline(p95, color="#d1495b", ls="--", lw=1.5, label=f"95% 解释度 (p={p95})")
ax1.set_xlabel("使用的主成分个数 p")
ax1.set_ylabel("对冲后残余方差", color="#1b6ca8")
ax1.tick_params(axis="y", labelcolor="#1b6ca8")
ax1.set_title("残余方差在 p 越过真实因子数后几乎不再下降", fontsize=12.5)
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(range(1, len(cum_var) + 1), cum_var * 100, color="#edae49", lw=1.8, ls=":", label="累计解释方差%")
ax2.set_ylabel("累计解释方差 (%)", color="#edae49")
ax2.tick_params(axis="y", labelcolor="#edae49")
fig.tight_layout()
fig.savefig(f"{OUT}/residual_vs_pcs.png")
plt.close(fig)

# ============ 图 2：样本内 vs 样本外 R² ============
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["OLS 全工具", f"PCA 降维 (p={p95})"]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [r2_ols_is, r2_pca_is], w, label="样本内 R²", color="#6a4c93")
ax.bar(x + w/2, [r2_ols_oos, r2_pca_oos], w, label="样本外 R²", color="#1b6ca8")
for i, (a, b) in enumerate(zip([r2_ols_is, r2_pca_is], [r2_ols_oos, r2_pca_oos])):
    ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("对冲拟合优度 R²")
ax.set_title("高维下 OLS 过拟合塌方，PCA 降维保住样本外", fontsize=12.5)
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/is_oos_r2.png")
plt.close(fig)

# ============ 图 3：被套保资产 + 对冲残余（一段样本） ============
seg = slice(0, 60)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(y[seg], color="#d1495b", lw=1.6, label="被套保资产 y")
ax.plot(y[seg] - X[seg] @ beta_ols, color="#1b6ca8", lw=1.4, label=f"OLS 对冲残余")
ax.plot(y[seg] - pcs[seg] @ P, color="#2a9d8f", lw=1.4, ls="--", label=f"PCA 对冲残余")
ax.set_title("对冲滤掉大部分波动；OLS 残余在小样本上反而更抖", fontsize=12)
ax.set_xlabel("时间（样本段）"); ax.set_ylabel("收益")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/hedged_residual.png")
plt.close(fig)

print(f"图片已保存到 {OUT}")
