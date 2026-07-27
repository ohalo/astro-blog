#!/usr/bin/env python3
"""POET 因子结构协方差配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "images", "factor-covariance-poet")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(21)

# ---------- 市场构造：N=100 高维, 3 公共因子 + 稀疏残差块 ----------
N, K_true = 100, 3
n_sec = 10  # 10 个行业，每行业 10 只
sec = np.repeat(np.arange(n_sec), N // n_sec)

B = np.zeros((N, K_true))
B[:, 0] = rng.uniform(0.8, 1.2, N)        # 市场
B[:, 1] = rng.normal(0, 0.5, N)           # 规模
B[:, 2] = rng.normal(0, 0.4, N)           # 价值
F_cov = np.diag([0.16**2, 0.08**2, 0.06**2])

# 稀疏残差协方差：行业内小相关 + 对角
vol_idio = 0.20 * rng.uniform(0.7, 1.4, N)
Su_true = np.diag(vol_idio**2)
for s in range(n_sec):
    idx = np.where(sec == s)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            rho = 0.25
            Su_true[i, j] = Su_true[j, i] = rho * vol_idio[i] * vol_idio[j]

Sigma_true = B @ F_cov @ B.T + Su_true
L_chol = np.linalg.cholesky(Sigma_true / 252.0)

def sim_returns(T, seed=None):
    r = np.random.default_rng(seed)
    return r.standard_normal((T, N)) @ L_chol.T

def poet(X, K, tau_mult=1.0):
    """POET：PCA 前 K 个主成分 + 残差软阈值"""
    T = X.shape[0]
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    w, V = np.linalg.eigh(S)
    w, V = w[::-1], V[:, ::-1]
    Lam = V[:, :K] * np.sqrt(np.maximum(w[:K], 0))
    S_lowrank = Lam @ Lam.T
    R = S - S_lowrank                              # 残差协方差
    # 自适应阈值：tau_ij = tau_mult * sqrt(theta_ij * log N / T)
    U = Xc - Xc @ V[:, :K] @ V[:, :K].T            # 残差收益
    theta = (U[:, :, None] * U[:, None, :]).var(axis=0) if N <= 60 else None
    if theta is None:
        # 大 N 用近似：theta_ij ~ (R_ii R_jj)
        dR = np.diag(R)
        theta = np.outer(dR, dR)
    tau = tau_mult * np.sqrt(theta * np.log(N) / T)
    R_th = np.sign(R) * np.maximum(np.abs(R) - tau, 0)
    np.fill_diagonal(R_th, np.diag(R))
    return S_lowrank + R_th, S_lowrank, R, R_th

def frob(A, B):
    return np.linalg.norm(A - B, "fro")

def gmv_vol(Sig, ridge=0.0):
    ones = np.ones(N)
    Sg = Sig + ridge * np.eye(N)
    w = np.linalg.solve(Sg, ones)
    w /= w.sum()
    return np.sqrt(w @ (Sigma_true / 252) @ w * 252) * 100

# ---------- 图1：特征值谱 + 尖峰 ----------
T = 252
X = sim_returns(T, seed=5)
S = np.cov(X.T)
w_s = np.linalg.eigvalsh(S)[::-1]
w_true = np.linalg.eigvalsh(Sigma_true / 252)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].semilogy(range(1, N + 1), w_s * 252, "o-", ms=3, color="#1f77b4", label="样本特征值")
axes[0].semilogy(range(1, N + 1), w_true * 252, "s--", ms=3, color="#7f7f7f", alpha=0.7, label="真实特征值")
axes[0].axvline(K_true + 0.5, color="#d62728", ls=":", label=f"K={K_true} 分界")
axes[0].set_xlabel("特征值序号"); axes[0].set_ylabel("特征值（年化方差口径，log）")
axes[0].set_title("尖峰谱：3 个因子特征值远离本体")
axes[0].legend()
ratios = w_s[:-1] / w_s[1:]
axes[1].plot(range(1, 16), ratios[:15], "o-", color="#2ca02c")
k_hat = int(np.argmax(ratios[:15]) + 1)
k_hat2 = int(np.argmax(ratios[1:15]) + 2)
axes[1].axvline(k_hat, color="#d62728", ls=":", label=f"全局最大 K̂={k_hat}（市场因子霸屏）")
axes[1].axvline(k_hat2, color="#1f77b4", ls=":", label=f"排除 k=1 后次峰 K̂={k_hat2}")
axes[1].set_xlabel("k"); axes[1].set_ylabel("λ_k / λ_{k+1}")
axes[1].set_title("特征值比值法选 K：注意市场因子霸屏问题")
axes[1].legend(fontsize=8)
fig.suptitle(f"N=100, T=252：c=N/T={N/T:.2f} 的高维场景", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/poet-spectrum.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("图1 done, K_hat =", k_hat)

# ---------- 图2：残差矩阵阈值前后热图 ----------
_, S_lr, R_raw, R_th = poet(X, 3, tau_mult=1.0)
d = np.sqrt(np.diag(R_raw))
Rc_raw = R_raw / np.outer(d, d)
d2 = np.sqrt(np.diag(R_th))
Rc_th = R_th / np.outer(d2, d2)
d_true = np.sqrt(np.diag(Su_true))
Rc_true = Su_true / np.outer(d_true, d_true)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
for ax, M, title in [(axes[0], Rc_true, "真实残差相关（块对角）"),
                     (axes[1], Rc_raw, "PCA 残差相关（噪声弥漫）"),
                     (axes[2], Rc_th, "软阈值后（结构恢复）")]:
    im = ax.imshow(M, vmin=-0.4, vmax=0.6, cmap="RdBu_r")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
fig.suptitle("POET 第二步：残差协方差的自适应软阈值（N=100, T=252, K=3）", y=1.00)
fig.savefig(f"{OUT}/poet-threshold-heatmap.png", dpi=110, bbox_inches="tight")
plt.close(fig)
nz_raw = (np.abs(Rc_raw[np.triu_indices(N,1)]) > 0.01).mean()
nz_th = (np.abs(Rc_th[np.triu_indices(N,1)]) > 0.01).mean()
nz_true = (np.abs(Rc_true[np.triu_indices(N,1)]) > 0.01).mean()
print(f"图2 done 非零占比 true={nz_true:.3f} raw={nz_raw:.3f} thresholded={nz_th:.3f}")

# ---------- 图3：估计误差 vs T（多方法） ----------
Ts = [126, 252, 504, 1008]
n_mc = 30
methods = ["样本协方差", "纯因子 (K=3, 残差取对角)", "POET (K=3)", "POET (K=1 欠拟合)", "POET (K=8 过拟合)"]
err = {m: {t: [] for t in Ts} for m in methods}
for mc in range(n_mc):
    for T_ in Ts:
        Xm = sim_returns(T_, seed=3000 + mc * 7 + Ts.index(T_))
        S_m = np.cov(Xm.T) * 252
        P3, S_lr3, R3, _ = poet(Xm, 3); P3 *= 252
        P1 = poet(Xm, 1)[0] * 252
        P8 = poet(Xm, 8)[0] * 252
        Fonly = (S_lr3 + np.diag(np.diag(R3))) * 252
        for name, Est in [("样本协方差", S_m), ("纯因子 (K=3, 残差取对角)", Fonly),
                          ("POET (K=3)", P3), ("POET (K=1 欠拟合)", P1), ("POET (K=8 过拟合)", P8)]:
            err[name][T_].append(frob(Est, Sigma_true))

# 同时算逆矩阵误差（真正区分方法的口径）
err_inv = {m: {t: [] for t in Ts} for m in methods}
Prec_true = np.linalg.inv(Sigma_true)
for mc in range(n_mc):
    for T_ in Ts:
        Xm = sim_returns(T_, seed=3000 + mc * 7 + Ts.index(T_))
        S_m = np.cov(Xm.T) * 252
        P3, S_lr3, R3, _ = poet(Xm, 3); P3 *= 252
        P1 = poet(Xm, 1)[0] * 252
        P8 = poet(Xm, 8)[0] * 252
        Fonly = (S_lr3 + np.diag(np.diag(R3))) * 252
        for name, Est in [("样本协方差", S_m), ("纯因子 (K=3, 残差取对角)", Fonly),
                          ("POET (K=3)", P3), ("POET (K=1 欠拟合)", P1), ("POET (K=8 过拟合)", P8)]:
            try:
                err_inv[name][T_].append(np.linalg.norm(np.linalg.inv(Est) - Prec_true, 2))
            except np.linalg.LinAlgError:
                pass

fig, axes3 = plt.subplots(1, 2, figsize=(12.5, 5))
colors = dict(zip(methods, ["#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]))
for m in methods:
    axes3[0].plot(Ts, [np.mean(err[m][t]) for t in Ts], marker="o", label=m, color=colors[m])
    axes3[1].plot(Ts, [np.mean(err_inv[m][t]) for t in Ts], marker="o", label=m, color=colors[m])
for ax_ in axes3:
    ax_.set_xscale("log"); ax_.set_xticks(Ts); ax_.set_xticklabels(Ts)
    ax_.set_xlabel("样本长度 T（天）")
axes3[0].set_ylabel("‖Σ̂ − Σ‖F"); axes3[0].set_title("协方差口径：各方法几乎分不出胜负")
axes3[1].set_yscale("log")
axes3[1].set_ylabel("‖Σ̂⁻¹ − Σ⁻¹‖₂（log）"); axes3[1].set_title("精度矩阵口径：样本协方差灾难现场")
axes3[0].legend(fontsize=8)
fig.suptitle("POET 的主战场在 Σ⁻¹，不在 Σ（N=100）", y=1.00)
fig.tight_layout()
fig.savefig(f"{OUT}/poet-error-vs-t.png", dpi=110, bbox_inches="tight")
plt.close(fig)
for m in methods:
    print(m, "Frob:", {t: round(np.mean(err[m][t]), 2) for t in Ts},
          "Inv:", {t: round(np.mean(err_inv[m][t]), 1) for t in Ts})

# ---------- 图4：GMV 组合真实波动率 ----------
res = {m: {t: [] for t in Ts} for m in ["样本协方差", "纯因子 (残差对角)", "POET (K=3)"]}
for mc in range(n_mc):
    for T_ in Ts:
        Xm = sim_returns(T_, seed=6000 + mc * 7 + Ts.index(T_))
        S_m = np.cov(Xm.T)
        P3, S_lr3, R3, _ = poet(Xm, 3)
        Fonly = S_lr3 + np.diag(np.diag(R3))
        for name, Est in [("样本协方差", S_m), ("纯因子 (残差对角)", Fonly), ("POET (K=3)", P3)]:
            try:
                res[name][T_].append(gmv_vol(Est, ridge=1e-10))
            except np.linalg.LinAlgError:
                pass

fig, ax = plt.subplots(figsize=(9, 5))
cmap2 = {"样本协方差": "#d62728", "纯因子 (残差对角)": "#ff7f0e", "POET (K=3)": "#1f77b4"}
for m in res:
    ax.plot(Ts, [np.mean(res[m][t]) for t in Ts], marker="o", label=m, color=cmap2[m])
oracle = gmv_vol(Sigma_true / 252)
ax.axhline(oracle, color="#7f7f7f", ls="--", label=f"真实协方差（{oracle:.1f}%）")
ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts)
ax.set_xlabel("估计窗口 T（天）"); ax.set_ylabel("GMV 组合真实年化波动率 (%)")
ax.set_title("GMV 判决：N=100 高维下 POET 的组合表现")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/poet-gmv-vol.png", dpi=110, bbox_inches="tight")
plt.close(fig)
for m in res:
    print(m, {t: round(np.mean(res[m][t]), 2) for t in Ts})
print("oracle", round(oracle, 2))
print("ALL DONE")
