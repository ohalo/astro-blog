#!/usr/bin/env python3
"""
为文章「DCC-GARCH 多元波动率模型：让相关矩阵随时间起舞」(dcc-garch-multivariate)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. dcc_returns_true_corr.png   两资产收益 + 真实时变相关（危机窗口相关性飙升）
  2. dcc_fitted_vs_true_corr.png DCC 拟合相关 vs 真实相关（危机被精准捕捉）
  3. dcc_corr_heatmap.png        平静期 vs 危机期 相关矩阵热力图（DCC 估计）
  4. dcc_portfolio_risk.png      等权组合风险：DCC 动态协方差 vs 常数相关协方差
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "dcc-garch-multivariate")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============================================================
# 0) 模拟：两资产，时变相关 + 各自 GARCH 波动聚集
#    思路：z_it = sqrt(rho_t)*m_t + sqrt(1-rho_t)*e_it  => corr(z1,z2)=rho_t
# ============================================================
N = 2500
t = np.arange(N)
# 真实时变相关：基线 0.30，危机窗口平滑飙升到 ~0.85
center, width = int(N * 0.55), int(N * 0.10)
bump = np.exp(-((t - center) / width) ** 2)
rho_true = 0.30 + 0.55 * bump

m = rng.standard_normal(N)
e1 = rng.standard_normal(N)
e2 = rng.standard_normal(N)
z1 = np.sqrt(rho_true) * m + np.sqrt(1 - rho_true) * e1
z2 = np.sqrt(rho_true) * m + np.sqrt(1 - rho_true) * e2


def simulate_garch(n, omega=1e-5, alpha=0.08, beta=0.90, mu=0.0002):
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for i in range(1, n):
        z = rng.standard_normal()
        s2[i] = omega + alpha * r[i - 1] ** 2 + beta * s2[i - 1]
        r[i] = mu + np.sqrt(s2[i]) * z
    return r, np.sqrt(s2)


r1, sig1 = simulate_garch(N)
r2, sig2 = simulate_garch(N)
# 把标准化因子 z 带上各自 GARCH 波动率
R1 = sig1 * z1
R2 = sig2 * z2

# ============================================================
# 1) 对每只资产拟合 GARCH(1,1)，得到条件标准差与标准化残差
# ============================================================
def garch_mle(r):
    def negll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        s2 = np.empty(len(r))
        s2[0] = np.var(r)
        for i in range(1, len(r)):
            s2[i] = w + a * r[i - 1] ** 2 + b * s2[i - 1]
        ll = -0.5 * (np.log(2 * np.pi) + np.log(s2) + r ** 2 / s2)
        return -ll.sum()

    x0 = [1e-5, 0.08, 0.90]
    bnds = [(1e-9, 1e-3), (1e-6, 0.6), (1e-6, 0.99)]
    res = minimize(negll, x0, bounds=bnds, method="L-BFGS-B")
    w, a, b = res.x

    s2 = np.empty(len(r))
    s2[0] = np.var(r)
    for i in range(1, len(r)):
        s2[i] = w + a * r[i - 1] ** 2 + b * s2[i - 1]
    return np.sqrt(s2), r / np.sqrt(s2)


sig1_hat, u1 = garch_mle(R1)
sig2_hat, u2 = garch_mle(R2)
U = np.column_stack([u1, u2])

# ============================================================
# 2) DCC：Q_t = (1-a-b)Qbar + a u'u' + b Q_{t-1}; R_t 标准化
# ============================================================
a, b = 0.03, 0.92
Qbar = np.cov(U.T)
Q = Qbar.copy()
R_series = np.zeros(N)
R_mat_series = np.zeros((N, 2, 2))
for i in range(N):
    if i == 0:
        Q = Qbar.copy()
    else:
        u = U[i - 1]
        Q = (1 - a - b) * Qbar + a * np.outer(u, u) + b * Q
    d = np.sqrt(np.diag(Q))
    R = Q / np.outer(d, d)
    R_mat_series[i] = R
    R_series[i] = R[0, 1]

# 常数相关的“朴素”协方差估计（用全样本相关系数）
R_const = np.corrcoef(U.T)

# ============================================================
# 图 1：收益 + 真实时变相关
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, R1, lw=0.6, color="#3b6ea5", label="资产 1 收益")
ax[0].plot(t, R2, lw=0.6, color="#c0504d", label="资产 2 收益", alpha=0.8)
ax[0].axvspan(center - 2 * width, center + 2 * width, color="red", alpha=0.08)
ax[0].set_ylabel("日收益")
ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_title("两资产日收益（各自带 GARCH 波动聚集）")
ax[1].plot(t, rho_true, lw=1.8, color="#2c7a3d")
ax[1].axvspan(center - 2 * width, center + 2 * width, color="red", alpha=0.08)
ax[1].set_ylabel("真实相关 ρ_t")
ax[1].set_xlabel("交易日")
ax[1].set_title("真实时变相关：危机窗口相关性从 0.30 飙升到 ~0.85")
ax[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(D, "dcc_returns_true_corr.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：DCC 拟合相关 vs 真实相关
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(t, rho_true, lw=1.4, color="gray", ls="--", label="真实 ρ_t")
ax.plot(t, R_series, lw=1.4, color="#1f77b4", label="DCC 拟合 R_t[0,1]")
ax.axvspan(center - 2 * width, center + 2 * width, color="red", alpha=0.08)
ax.set_ylabel("相关系数")
ax.set_xlabel("交易日")
ax.set_title("DCC 相关动态：危机飙升被精准追踪（并非简单平滑掉）")
ax.legend(fontsize=9)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(D, "dcc_fitted_vs_true_corr.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：平静期 vs 危机期 相关矩阵热力图
# ============================================================
calm_idx = 300
crisis_idx = center
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
labels = ["资产1", "资产2"]
for ax, idx, title in zip(
    axes, [calm_idx, crisis_idx], ["平静期相关矩阵", "危机期相关矩阵"]
):
    M = R_mat_series[idx]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    ax.set_title(f"{title}\n(t={idx})")
fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(D, "dcc_corr_heatmap.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：组合风险对比（DCC 动态 vs 常数相关）
# ============================================================
w = np.array([0.5, 0.5])
port_vol_dcc = np.zeros(N)
port_vol_const = np.zeros(N)
for i in range(N):
    Dmat = np.diag([sig1_hat[i], sig2_hat[i]])
    Cov_dcc = Dmat @ R_mat_series[i] @ Dmat
    Cov_const = Dmat @ R_const @ Dmat
    port_vol_dcc[i] = np.sqrt(w @ Cov_dcc @ w)
    port_vol_const[i] = np.sqrt(w @ Cov_const @ w)
# 年化（×sqrt(252)）
scale = np.sqrt(252)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(t, port_vol_dcc * scale, lw=1.6, color="#d62728", label="DCC 动态协方差")
ax.plot(t, port_vol_const * scale, lw=1.6, color="#7f7f7f", label="常数相关协方差", alpha=0.85)
ax.axvspan(center - 2 * width, center + 2 * width, color="red", alpha=0.08)
ax.set_ylabel("组合年化波动率")
ax.set_xlabel("交易日")
ax.set_title("等权组合风险：DCC 在危机段如实放大风险，常数相关严重低估")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "dcc_portfolio_risk.png"), dpi=130)
plt.close()

print("DCC images done:", os.listdir(D))
print(f"rho_true peak={rho_true.max():.3f}, DCC corr peak={R_series.max():.3f}, "
      f"const corr={R_const[0,1]:.3f}")
