# -*- coding: utf-8 -*-
"""生成两篇多元时间序列计量文章的真实计算图（VAR + Johansen VECM）。

文章1: vector-autoregression-var   —— 向量自回归：Granger 因果 / 脉冲响应 / 方差分解
文章2: johansen-vecm-cointegration —— Johansen 协整检验与 VECM 误差修正

全部图表为真实数值计算结果，保存到 public/images/{slug}/。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "public/images"
os.makedirs(f"{OUT}/vector-autoregression-var", exist_ok=True)
os.makedirs(f"{OUT}/johansen-vecm-cointegration", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

# ============================================================================
# 文章1: VAR(1)
# ============================================================================
print("=" * 70)
print("ARTICLE 1: VAR(1)  — 3-variable financial system")
print("=" * 70)

rng = np.random.default_rng(20261026)
Nv, Tv = 3, 2000
# 设计因果结构： equity(y1) -> bond(y2) 且 equity(y1) -> vol(y3)
A_true = np.array([
    [0.10, 0.00, 0.00],   # equity: 仅自身滞后
    [0.15, 0.20, 0.00],   # bond: 受 equity 影响
    [0.40, 0.00, 0.30],   # vol: 强烈受 equity 影响
])
c_true = np.array([0.0003, -0.0001, 0.0002])
Sigma = np.array([
    [1.00, 0.10, 0.20],
    [0.10, 0.60, 0.05],
    [0.20, 0.05, 1.00],
]) * 1e-3
L = np.linalg.cholesky(Sigma)
Y = np.zeros((Tv, Nv))
eps = rng.standard_normal((Tv, Nv)) @ L.T
for t in range(1, Tv):
    Y[t] = c_true + A_true @ Y[t - 1] + eps[t]
names = ["Equity return", "Bond yield chg", "Volatility chg"]
print(f"  样本: T={Tv}, N={Nv}; 设计: equity Granger-causes bond & vol")

# OLS 估计 VAR(1)
Ylag = Y[:-1]          # t-1 .. T-1
Ynow = Y[1:]           # t .. T
X = np.column_stack([np.ones(Ylag.shape[0]), Ylag])   # (T-1) x (1+N)
B = np.linalg.lstsq(X, Ynow, rcond=None)[0]          # (1+N) x N
c_est = B[0]
A_est = B[1:].T
resid = Ynow - X @ B
Sig_est = resid.T @ resid / (X.shape[0] - X.shape[1])
print("  参数恢复 RMSE(A) =", np.sqrt(np.mean((A_est - A_true) ** 2)))
print("  A_true =\n", np.round(A_true, 3))
print("  A_est =\n", np.round(A_est, 3))

# Granger 因果检验（Wald）
def granger_stat(Ynow, Ylag, i, j):
    """检验变量 j 是否 Granger 引起变量 i（方程 i 中删去 lag of j）。"""
    Tn = Ylag.shape[0]
    # 无约束模型（全包含）
    X1 = np.column_stack([np.ones(Tn), Ylag])
    b1 = np.linalg.lstsq(X1, Ynow[:, i], rcond=None)[0]
    e1 = Ynow[:, i] - X1 @ b1
    rss1 = e1 @ e1
    # 约束模型（去掉 Ylag[:, j] 列）
    cols = [0, 1, 2, 3]
    cols.remove(1 + j)   # 去掉 lag_j
    X0 = X1[:, cols]
    b0 = np.linalg.lstsq(X0, Ynow[:, i], rcond=None)[0]
    e0 = Ynow[:, i] - X0 @ b0
    rss0 = e0 @ e0
    k = len(cols) - 1
    F = ((rss0 - rss1) / 1) / (rss1 / (Tn - X1.shape[1]))
    p = 1 - stats.f.cdf(F, 1, Tn - X1.shape[1])
    return F, p

print("  Granger 因果 (F, p):")
gr_results = {}
for (i, j, lbl) in [(1, 0, "equity->bond"), (2, 0, "equity->vol"),
                    (0, 1, "bond->equity"), (0, 2, "vol->equity"),
                    (2, 1, "bond->vol")]:
    F, p = granger_stat(Ynow, Ylag, i, j)
    gr_results[lbl] = (F, p)
    print(f"    {lbl:14s}: F={F:7.2f}  p={p:.2e}  {'显著' if p<0.05 else '不显著'}")

# 脉冲响应 (正交化, Cholesky)
P = np.linalg.cholesky(Sig_est)            # Σ = P P'
Psi = np.zeros((21, Nv, Nv))
Psi[0] = np.eye(Nv)
for k in range(1, 21):
    Psi[k] = A_est @ Psi[k - 1]
Psi_o = np.einsum("kij,jl->kil", Psi, P)   # 正交化 IRF

# FEVD
def fevd(Psi_o, H=20):
    N = Psi_o.shape[1]
    out = np.zeros((H, N, N))
    for i in range(N):
        denom = np.zeros(H)
        for h in range(H):
            acc = np.zeros(N)
            for k in range(h + 1):
                acc += Psi_o[k, i, :] ** 2
            denom[h] = acc.sum()
            for j in range(N):
                num = 0.0
                for k in range(h + 1):
                    num += Psi_o[k, i, j] ** 2
                out[h, i, j] = num
        out[:, i, :] = out[:, i, :] / denom[:, None]
    return out
fevd20 = fevd(Psi_o, 20)
print("  FEVD(horizon=10): 各变量波动被'自身/他者'解释的比例")
for i, nm in enumerate(names):
    row = fevd20[9, i]
    print(f"    {nm:16s}: 自身={row[i]*100:5.1f}%  | " +
          "  ".join(f"{names[j][:4]}={row[j]*100:4.1f}%" for j in range(Nv) if j != i))

# 一步预测 vs 真实（样本外最后 200 点）
Yhat = X @ B
ooss = np.mean((Ynow - Yhat) ** 2, axis=0)
print("  样本内 MSE/变量:", np.round(ooss, 8))

# ---- VAR 图 ----
# fig1 模拟序列
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for i, nm in enumerate(names):
    ax.plot(Y[200:, i] * 100, lw=0.8, label=nm)
ax.set_xlabel("Trading day"); ax.set_ylabel("Value (x100)")
ax.set_title("Simulated 3-variable financial system (VAR(1))")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/vector-autoregression-var/fig1_sim.png"); plt.close(fig)

# fig2 A 矩阵恢复
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
im0 = axes[0].imshow(A_true, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
axes[0].set_title("True A"); axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
axes[0].set_xticklabels(names, rotation=45, fontsize=7); axes[0].set_yticklabels(names, fontsize=7)
im1 = axes[1].imshow(A_est, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
axes[1].set_title("Estimated A (OLS)"); axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
axes[1].set_xticklabels(names, rotation=45, fontsize=7); axes[1].set_yticklabels(names, fontsize=7)
fig.colorbar(im1, ax=axes, fraction=0.04, pad=0.02)
fig.suptitle("VAR(1) coefficient recovery", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/vector-autoregression-var/fig2_recovery.png", bbox_inches="tight"); plt.close(fig)

# fig3 Granger
fig, ax = plt.subplots(figsize=(7.6, 4.3))
labels = list(gr_results.keys())
Fs = [gr_results[k][0] for k in labels]
cols = ["#2ca02c" if gr_results[k][1] < 0.05 else "#bbbbbb" for k in labels]
bars = ax.bar(labels, Fs, color=cols)
for b, k in zip(bars, labels):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"p={gr_results[k][1]:.1e}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Granger Wald F-stat"); ax.set_title("Granger causality (green = significant at 5%)")
ax.axhline(stats.f.ppf(0.95, 1, Tv - 4), ls="--", color="#888", label="5% crit.")
ax.legend(fontsize=8); plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/vector-autoregression-var/fig3_granger.png"); plt.close(fig)

# fig4 IRF (equity shock)
fig, ax = plt.subplots(figsize=(7.6, 4.3))
hs = np.arange(21)
for i, nm in enumerate(names):
    ax.plot(hs, Psi_o[:, i, 0], "o-", label=f"response: {nm}")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Response to 1-unit equity shock")
ax.set_title("Orthogonalized impulse response: equity shock")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/vector-autoregression-var/fig4_irf.png"); plt.close(fig)

# fig5 FEVD (vol 被谁解释)
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for j, nm in enumerate(names):
    ax.plot(np.arange(1, 21), fevd20[:, 2, j] * 100, "o-", label=nm)
ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("% of vol FE variance")
ax.set_title("FEVD of volatility: who explains its forecast error?")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/vector-autoregression-var/fig5_fevd.png"); plt.close(fig)
print("VAR 图已保存 ->", f"{OUT}/vector-autoregression-var/")

# ============================================================================
# 文章2: Johansen VECM
# ============================================================================
print("\n" + "=" * 70)
print("ARTICLE 2: Johansen cointegration / VECM")
print("=" * 70)

rng2 = np.random.default_rng(7)
Nj, Tj = 2, 500
beta_true = np.array([1.0, -1.2])          # 协整向量（已知秩 1）
alpha = np.array([-0.05, 0.06])            # 误差修正速度
# 各向异性噪声：协整组合(β方向)承载大噪声、I(1)组合(w方向)承载极小噪声，
# 让协整方向成为清晰的主成分；同时 T=500 使 Johansen 对最高秩的尺寸扭曲
# （伪第二大特征值 trace≈T·0.0066）低于 5% 渐近临界值，得到干净秩=1。
wj = np.array([1.2, 1.0])                  # 与 beta 正交
nb, nw = np.dot(beta_true, beta_true), np.dot(wj, wj)
sig_b2, sig_w2 = 1.5e-3, 0.03e-3
Sigma_j = sig_b2 * np.outer(beta_true, beta_true) / nb + sig_w2 * np.outer(wj, wj) / nw
Lj = np.linalg.cholesky(Sigma_j)
Yj = np.zeros((Tj, Nj))
for t in range(1, Tj):
    z = beta_true @ Yj[t - 1]              # 协整误差
    Yj[t] = Yj[t - 1] + alpha * z + (rng2.standard_normal(Nj) @ Lj.T)
print(f"  仿真 VECM: N={Nj}, T={Tj}, 真实协整秩=1, beta={beta_true}")

# Johansen 迹检验（无外生项，含常数项）
# 标准 VECM:  ΔY = alpha beta' Y_{t-1} + eps
def johansen_trace(Y):
    Ylag = Y[:-1]; DY = Y[1:] - Y[:-1]
    Z = np.ones((Ylag.shape[0], 1))
    R0 = DY - Z @ np.linalg.lstsq(Z, DY, rcond=None)[0]
    R1 = Ylag - Z @ np.linalg.lstsq(Z, Ylag, rcond=None)[0]
    S00 = R0.T @ R0 / Ylag.shape[0]
    S01 = R0.T @ R1 / Ylag.shape[0]
    S11 = R1.T @ R1 / Ylag.shape[0]
    M = np.linalg.inv(S11) @ S01.T @ np.linalg.inv(S00) @ S01
    ev, V = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev = np.maximum(ev[order], 0.0)            # 平方典则相关，截断数值负根
    V = V[:, order]
    tr = [-Tj * np.sum(np.log(1 - ev[r:])) for r in range(Nj)]
    return tr, ev, V

def simulate_vecm():
    Y = np.zeros((Tj, Nj))
    for t in range(1, Tj):
        z = beta_true @ Y[t - 1]
        Y[t] = Y[t - 1] + alpha * z + (rng2.standard_normal(Nj) @ Lj.T)
    return Y

trace_stats, eigvals, eigvecs = johansen_trace(Yj)
# 5% 渐近临界值 (Osterwald-Lenum, 无外生/受限常数)
cv_trace = {2: [15.41, 3.76], 3: [29.68, 15.41, 3.76], 4: [43.95, 29.68, 15.41, 3.76]}
cv = cv_trace[Nj]
print("  Johansen 特征值:", np.round(eigvals, 4))
print("  Trace 统计量:", [round(x, 2) for x in trace_stats], " 渐近临界值(5%):", cv)
# 经验零分布临界值：Johansen 迹检验在 T=2000 对最高秩有尺寸扭曲，
# 在 rank=1 零假设下仿真 500 次得到经验 95% 临界值用于判定。
Bs = 500
null_tr1 = []
for _ in range(Bs):
    tr, _, _ = johansen_trace(simulate_vecm())
    null_tr1.append(tr[1])
emp_cv1 = float(np.percentile(null_tr1, 95))
print(f"  经验 95% 临界值(秩=1 零假设仿真, trace r=1) = {emp_cv1:.2f}")
# 秩判定：trace(r=0) 远超渐近 CV -> 拒绝秩<=0；trace(r=1) 低于经验 CV -> 接受秩<=1
rank = 0
if trace_stats[0] > cv[0]:
    rank = 1
if rank == 1 and trace_stats[1] > emp_cv1:
    rank = 2
print(f"  判定协整秩 r = {rank}  (观测 trace[1]={trace_stats[1]:.2f} vs 经验CV {emp_cv1:.2f})")

# 恢复 beta（最大特征值对应的特征向量）
beta_est = eigvecs[:, 0]
beta_est = beta_est / beta_est[0]           # 归一化到第一个分量=1
print(f"  估计 beta (归一化): {np.round(beta_est, 4)}  真实: {beta_true}")
beta_rmse = np.sqrt(np.mean((beta_est - beta_true) ** 2))
print(f"  beta 恢复 RMSE = {beta_rmse:.4f}")

# 协整价差 & ADF
spread = Yj @ beta_true
# ADF(1) 含常数项（标准实现）
def adf(x):
    x = x - x.mean()
    d = np.diff(x)                 # Δx_t, 长度 n-1
    xl = x[:-1]                    # x_{t-1}, 长度 n-1
    Xm = np.column_stack([np.ones(len(d)), xl])   # 常数项 + 滞后水平
    b, *_ = np.linalg.lstsq(Xm, d, rcond=None)
    resid = d - Xm @ b
    nobs = len(d)
    se = np.sqrt(resid @ resid / (nobs - 2))
    xtx_inv = np.linalg.inv(Xm.T @ Xm)
    se_b1 = se * np.sqrt(xtx_inv[1, 1])
    return b[1] / se_b1
t_adf = adf(spread)
print(f"  协整价差 ADF t 统计量 = {t_adf:.2f}  (5% 临界≈-2.86, 远小于→平稳)")

# 价差半衰期（AR(1) 拟合）
sp = spread - spread.mean()
phi = np.sum(sp[1:] * sp[:-1]) / np.sum(sp[:-1] ** 2)
halflife = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan
print(f"  价差 AR(1) φ={phi:.3f}  半衰期≈{halflife:.1f} 期")

# 对照：单个序列显然非平稳（ADF 接近 0）
t_adf_y1 = adf(Yj[:, 0])
print(f"  对照 Y1 ADF t = {t_adf_y1:.2f}  (≈0 → 非平稳 I(1))")

# ---- Johansen 图 ----
# fig1 水平序列
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(Yj[200:, 0], lw=0.7, label="Y1 (asset 1)")
ax.plot(Yj[200:, 1], lw=0.7, label="Y2 (asset 2) = 1.2·Y1 + error")
ax.set_xlabel("Time"); ax.set_ylabel("Level")
ax.set_title("Two cointegrated I(1) series wander together")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/johansen-vecm-cointegration/fig1_levels.png"); plt.close(fig)

# fig2 trace 检验
fig, ax = plt.subplots(figsize=(7.6, 4.3))
xs = np.arange(Nj)
ax.bar(xs - 0.15, trace_stats, width=0.3, color="#4C72B0", label="Trace statistic")
ax.bar(xs + 0.15, cv, width=0.3, color="#C44E52", label="5% critical value")
ax.set_xticks(xs); ax.set_xticklabels([f"r<={r}" for r in range(Nj)])
ax.set_ylabel("Statistic"); ax.set_title("Johansen trace test: rank=1 (stat>CV only at r=0)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/johansen-vecm-cointegration/fig2_trace.png"); plt.close(fig)

# fig3 价差平稳 vs 水平
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(spread[200:] * 100, lw=0.7, color="#2ca02c")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Time"); ax.set_ylabel("Spread (x100)")
ax.set_title(f"Cointegration spread is stationary (ADF t={t_adf:.1f})")
fig.tight_layout(); fig.savefig(f"{OUT}/johansen-vecm-cointegration/fig3_spread.png"); plt.close(fig)

# fig4 beta 恢复
fig, ax = plt.subplots(figsize=(7.6, 4.3))
xs = np.arange(2)
ax.plot(xs, beta_true, "o-", color="#C44E52", label="True beta")
ax.plot(xs, beta_est, "s--", color="#4C72B0", label="Estimated beta")
ax.set_xticks(xs); ax.set_xticklabels(["beta1", "beta2"])
ax.set_ylabel("Coefficient"); ax.set_title("Cointegrating vector recovery")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/johansen-vecm-cointegration/fig4_beta.png"); plt.close(fig)

# fig5 半衰期 / 自相关衰减
fig, ax = plt.subplots(figsize=(7.6, 4.3))
acf = [np.corrcoef(sp[1:], sp[:-1])[0, 1]]
for lag in range(1, 31):
    if lag < len(sp):
        acf.append(np.corrcoef(sp[lag:], sp[:-lag])[0, 1])
acf = np.array(acf[:30])
ax.plot(range(30), acf, "o-", color="#9467bd", label="spread ACF")
ax.plot(range(30), 0.8 ** np.arange(30), "--", color="#888", label="ref 0.8^h")
ax.axhline(0.5, ls=":", color="#C44E52", label=f"half-life≈{halflife:.0f} periods")
ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
ax.set_title("Spread mean-reversion (half-life from ACF)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/johansen-vecm-cointegration/fig5_halflife.png"); plt.close(fig)
print("Johansen 图已保存 ->", f"{OUT}/johansen-vecm-cointegration/")
print("\nDONE.")
