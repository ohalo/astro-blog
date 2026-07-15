#!/usr/bin/env python3
"""
为文章「Ledoit-Wolf 协方差收缩：把样本协方差的噪声压回信号」(covariance-shrinkage-ledoit-wolf)
生成真实配图。自洽合成：一个含少数共同因子的真实协方差 Σ0，抽样估计样本协方差 S，
用 Ledoit-Wolf(2004) 解析收缩把 S 拉向目标 F=mean(diag)·I，展示 OOS 组合方差被压低、
最优收缩强度 δ̂ 随维数/噪声上升、特征值被降噪、有效前沿更贴近 Oracle。

图1 lw_oos_variance.png      OOS 组合方差 vs 收缩强度 δ（精确扫描，标注真实最低点 δ* 与 LW 解析 δ̂）
图2 lw_delta_vs_nt.png       最优收缩 δ* 与条件数随 p/T（维数/样本比）上升
图3 lw_eigen_denoise.png     特征值谱：样本(噪声) vs 最优收缩 vs 真实
图4 lw_frontier.png          OOS 最小方差：收缩后单名权重更分散（不再过拟合）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "covariance-shrinkage-ledoit-wolf"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)


def true_cov(p, k=4, dmin=0.4, dmax=0.7, fscale=0.4, seed=42):
    """构造含 k 个共同因子的真实协方差 Σ0（主导因子 + 适中特异性方差）。"""
    r = np.random.default_rng(seed)
    B = r.normal(0, fscale, (p, k))
    d = r.uniform(dmin, dmax, p)
    Sig = B @ B.T + np.diag(d)
    return Sig


def lw(R):
    """Ledoit-Wolf(2004) 解析线性收缩，目标 F = diag(S)（清掉相关性、保留各自方差）。"""
    X = R - R.mean(0)
    T, p = X.shape
    Sb = X.T @ X / T
    F = np.diag(np.diag(Sb))                    # 标准 LW 目标：对角=S 的对角，非对角=0
    M = X[:, :, None] * X[:, None, :]            # (T,p,p) 逐 t 外积
    pi = ((M - Sb[None]) ** 2).mean(0).sum()
    rho_ij = ((M - Sb[None]) * (M - F[None])).mean(0)
    rho = ((Sb - F) * rho_ij).sum()
    gamma = ((Sb - F) ** 2).sum()
    delta = min(1.0, max(0.0, (pi - rho) / gamma))
    return float(delta), (1.0 - delta) * Sb + delta * F


def mv_weights(Sig):
    """全局最小方差权重 w = Σ⁻¹1 / (1'Σ⁻¹1)。"""
    inv = np.linalg.inv(Sig)
    ones = np.ones(Sig.shape[0])
    return inv @ ones / (ones @ inv @ ones)


def frontier(Sig, mu, lam_vec):
    """固定风险厌恶 λ：w ∝ Σ⁻¹ μ，年化均值/方差。"""
    inv = np.linalg.inv(Sig)
    ones = np.ones(Sig.shape[0])
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    out = []
    for lam in lam_vec:
        w = inv @ (lam * mu + (1 - lam * B / A) * ones / A)
        m = float(w @ mu) * 252
        v = float(w @ Sig @ w) * 252
        out.append((m, v))
    return np.array(out)


# ===================== 主实验 =====================
p, T = 30, 250
Sig0 = true_cov(p, k=4, dmin=0.4, dmax=0.7, fscale=0.4, seed=42)
rng = np.random.default_rng(20260715)
R_train = rng.multivariate_normal(np.zeros(p), Sig0, size=T)
R_test = rng.multivariate_normal(np.zeros(p), Sig0, size=T)

X = R_train - R_train.mean(0)
Sb = X.T @ X / T
delta_hat, Slw = lw(R_train)

S_sample = Sb
S_target = np.diag(np.diag(Sb))                 # 与 LW 一致的目标：清相关性、留方差

w_samp, w_lw, w_oracle, w_target = (mv_weights(s) for s in (S_sample, Slw, Sig0, S_target))
risk_samp = float(w_samp @ Sig0 @ w_samp)
risk_lw = float(w_lw @ Sig0 @ w_lw)
risk_oracle = float(w_oracle @ Sig0 @ w_oracle)
risk_target = float(w_target @ Sig0 @ w_target)

print(f"delta_hat(LW解析) = {delta_hat:.3f}")
print(f"OOS 组合方差: sample={risk_samp:.5f}  LW={risk_lw:.5f}  "
      f"target(δ=1)={risk_target:.5f}  oracle(真)={risk_oracle:.5f}")

# ===== 图1：OOS 方差 vs δ（精确扫描，找真实最低点）=====
deltas = np.linspace(0, 1, 101)
risks = []
for dlt in deltas:
    Sig_d = (1 - dlt) * S_sample + dlt * S_target
    w = mv_weights(Sig_d)
    risks.append(float(w @ Sig0 @ w))
risks = np.array(risks)
d_argmin = float(deltas[np.argmin(risks)])
risk_argmin = float(risks.min())
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(deltas, risks, color="#1f77b4", lw=1.8, label="OOS 组合方差 w'Σ₀w（精确扫描）")
ax.axvline(d_argmin, color="#2ca02c", ls="-.", lw=1.3,
           label=f"精确扫描最低点 δ*={d_argmin:.2f}")
ax.axvline(delta_hat, color="#d62728", ls="--", lw=1.2,
           label=f"Ledoit-Wolf 解析 δ̂={delta_hat:.2f}（强相关下有偏保守）")
ax.scatter([0, 1], [risk_samp, risk_target], color="#ff7f0e", zorder=5,
           label="样本(δ=0) / 目标(δ=1)")
ax.set_title("OOS 方差随收缩强度 δ 呈 U 形：收缩不足或过度都更差", fontsize=12)
ax.set_xlabel("收缩强度 δ  (Σ̂=(1−δ)S + δF)")
ax.set_ylabel("OOS 组合方差")
ax.legend(loc="upper center", fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "lw_oos_variance.png"), dpi=130); plt.close(fig)

# ===== 图2：δ* vs p/T（精确扫描最低点）与条件数改善 =====
ps = np.array([10, 20, 30, 40, 50, 60, 75, 90])
T_fix = 250
delta_curve = []
cond_s = []; cond_o = []
ds = np.linspace(0, 1, 21)
for pp in ps:
    vals = []
    for sd in [1, 2, 3]:
        S0i = true_cov(pp, k=4, dmin=0.4, dmax=0.7, fscale=0.4, seed=sd)
        Ri = np.random.default_rng(1000 + sd).multivariate_normal(np.zeros(pp), S0i, size=T_fix)
        X = Ri - Ri.mean(0); Sbi = X.T @ X / T_fix; F0 = np.diag(np.diag(Sbi))
        r = []
        for dlt in ds:
            w = mv_weights((1 - dlt) * Sbi + dlt * F0)
            r.append(float(w @ S0i @ w))
        vals.append(float(np.argmin(r) / 20))
    delta_curve.append(np.mean(vals))
    S0x = true_cov(pp, k=4, dmin=0.4, dmax=0.7, fscale=0.4, seed=42)
    Rx = np.random.default_rng(20260715).multivariate_normal(np.zeros(pp), S0x, size=T_fix)
    Xx = Rx - Rx.mean(0); Sx = Xx.T @ Xx / T_fix; Fx = np.diag(np.diag(Sx))
    rs = []
    for dlt in ds:
        w = mv_weights((1 - dlt) * Sx + dlt * Fx)
        rs.append(float(w @ S0x @ w))
    dopt = ds[int(np.argmin(rs))]
    cond_s.append(np.linalg.cond(Sx))
    cond_o.append(np.linalg.cond((1 - dopt) * Sx + dopt * Fx))
delta_curve = np.array(delta_curve); cond_s = np.array(cond_s); cond_o = np.array(cond_o)

fig, ax1 = plt.subplots(figsize=(9, 4.3))
ax1.plot(ps / T_fix, delta_curve, "-o", color="#2ca02c", lw=1.8, label="最优收缩 δ*（精确扫描）")
ax1.set_xlabel("p / T（维数 ÷ 样本数）")
ax1.set_ylabel("最优收缩强度 δ*", color="#2ca02c")
ax1.tick_params(axis="y", labelcolor="#2ca02c")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(ps / T_fix, cond_s, "-s", color="#d62728", lw=1.6, label="样本 Σ 条件数")
ax2.plot(ps / T_fix, cond_o, "-^", color="#1f77b4", lw=1.6, label="最优收缩 Σ̂ 条件数")
ax2.set_ylabel("条件数 κ(Σ)", color="#444444")
ax1.set_title("p/T 越大：最优收缩越强，样本 Σ 越病态、收缩后越良态", fontsize=11.5)
l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(D, "lw_delta_vs_nt.png"), dpi=130); plt.close(fig)

# ===== 图3：特征值降噪（用精确最优 δ* 收缩）=====
S_opt = (1 - d_argmin) * S_sample + d_argmin * S_target
eig_sample = np.sort(np.linalg.eigvalsh(S_sample))[::-1]
eig_opt = np.sort(np.linalg.eigvalsh(S_opt))[::-1]
eig_true = np.sort(np.linalg.eigvalsh(Sig0))[::-1]
n = len(eig_sample)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.stem(np.arange(n), eig_sample, linefmt="#d62728", markerfmt="o",
        basefmt=" ", label="样本协方差 S（噪声大、谱散开）")
ax.stem(np.arange(n), eig_opt, linefmt="#2ca02c", markerfmt="",
        basefmt=" ", label=f"最优收缩 Σ̂(δ*={d_argmin:.2f})")
ax.stem(np.arange(n), eig_true, linefmt="#1f77b4", markerfmt="",
        basefmt=" ", label="真实 Σ₀（Oracle）")
ax.set_title("收缩把特征值谱从『样本噪声』拉回『真实结构』", fontsize=12)
ax.set_xlabel("特征值序号（按大小排序）")
ax.set_ylabel("特征值")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "lw_eigen_denoise.png"), dpi=130); plt.close(fig)

# ===== 图4：OOS 最小方差——收缩后权重不再爆炸 =====
w_s = mv_weights(S_sample); w_o = mv_weights(S_opt)
top_s = np.sort(np.abs(w_s))[::-1][:5]
top_o = np.sort(np.abs(w_o))[::-1][:5]
fig, ax = plt.subplots(figsize=(9, 4.3))
x = np.arange(5)
ax.bar(x - 0.2, top_s, width=0.4, color="#d62728", label=f"样本 Σ 前5大权重(和={top_s.sum():.2f})")
ax.bar(x + 0.2, top_o, width=0.4, color="#2ca02c", label=f"最优收缩 Σ̂ 前5大权重(和={top_o.sum():.2f})")
ax.set_xticks(x); ax.set_xticklabels([f"第{i+1}大" for i in range(5)])
ax.set_title("收缩让权重更分散：样本 Σ 把仓位压进少数名（过拟合信号）", fontsize=11.5)
ax.set_ylabel("单名权重绝对值")
ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "lw_frontier.png"), dpi=130); plt.close(fig)

print("DONE", sorted(os.listdir(D)))
