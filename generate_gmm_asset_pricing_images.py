#!/usr/bin/env python3
"""
为文章「GMM 广义矩估计在资产定价中的应用：用矩条件替代强假设」(gmm-asset-pricing)
生成真实配图。所有图表均由脚本内自洽合成数据 + 文中方法真实计算生成。

机制：
  资产定价核心矩条件（欧拉方程）：E[ m_{t+1} * R_{t+1} ] = 1
  线性 SDF：m = 1 - b'(f - E[f])，f 为因子。
  GMM 用样本矩 g_T(θ) = (1/T) Σ [ m_t(θ) * R_t - 1 ] 逼近 0，
  最小化 g_T' W g_T。两步 GMM 用最优权重 W = S^{-1}（矩条件协方差逆）。
  过度识别检验：J = T * g_T' S^{-1} g_T ~ χ²(矩数 - 参数数)。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "gmm-asset-pricing")
os.makedirs(D, exist_ok=True)

C = {"gmm": "#C44E52", "ols": "#2F4B7C", "true": "#55A868", "grid": "#DDDDDD",
     "j": "#8172B3", "chi": "#DD8452", "err": "#B22222"}

rng = np.random.default_rng(20260712)

# ========== 构造一个单因子线性 SDF 世界 ==========
T = 600
N = 10          # 10 个测试资产
b_true = 3.0    # 真 SDF 因子载荷
f_std = 0.045   # 因子波动
var_f = f_std ** 2
# 为使无套利 E[m R]=1 与 b_true 自洽：mu_i = b_true * beta_i * Var(f)
lam_f = b_true * var_f    # 因子风险溢价（月）

# 因子（如市场超额收益）
f = 0.005 + f_std * rng.standard_normal(T)
beta = np.linspace(0.4, 1.6, N)  # 各资产对因子的 beta
# 由无套利：E[R_i] = beta_i * lam_f （超额收益）
mu_R = beta * lam_f
eps = 0.03 * rng.standard_normal((T, N))
R_excess = mu_R[None, :] + np.outer(f - f.mean(), beta) + eps  # 超额收益
Rg = 1.0 + R_excess  # 毛收益（相对无风险）

def sdf(b, f):
    return 1.0 - b * (f - f.mean())

def moments(b, f, Rg):
    m = sdf(b, f)[:, None]
    return m * Rg - 1.0   # T x N，每列一个矩条件

def gbar(b, f, Rg):
    return moments(b, f, Rg).mean(axis=0)

# ---------- 一步 GMM（单位权重）----------
bs = np.linspace(0.0, 8.0, 500)
obj_I = np.array([gbar(b, f, Rg) @ gbar(b, f, Rg) for b in bs])
b_hat1 = bs[np.argmin(obj_I)]

# ---------- 两步 GMM（最优权重 S^{-1}）----------
g0 = moments(b_hat1, f, Rg)
S = (g0.T @ g0) / T
Sinv = np.linalg.pinv(S)
obj_W = np.array([gbar(b, f, Rg) @ Sinv @ gbar(b, f, Rg) for b in bs])
b_hat2 = bs[np.argmin(obj_W)]

# ---------- OLS 横截面回归（Fama-MacBeth 风格对照）----------
X = np.column_stack([np.ones(N), beta])
coef = np.linalg.lstsq(X, mu_R, rcond=None)[0]
lam_ols = coef[1]

# ========== 图1：GMM 目标函数（一步 vs 两步）==========
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(bs, obj_I / obj_I.max(), color=C["ols"], lw=2, label="一步 GMM（单位权重 W=I）")
ax.plot(bs, obj_W / obj_W.max(), color=C["gmm"], lw=2, label="两步 GMM（最优权重 W=S⁻¹）")
ax.axvline(b_true, color=C["true"], ls="--", lw=1.6, label=f"真值 b={b_true}")
ax.axvline(b_hat2, color=C["gmm"], ls=":", lw=1.4, label=f"两步估计 b̂={b_hat2:.2f}")
ax.set_title("GMM 目标函数：最小化加权矩条件二次型 g'Wg", fontsize=12)
ax.set_xlabel("SDF 因子载荷 b"); ax.set_ylabel("归一化目标函数")
ax.grid(color=C["grid"], alpha=0.6); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "gmm_objective.png"), dpi=130)
plt.close(fig)

# ========== 图2：过度识别检验 J 统计量 vs χ² 分布 ==========
# 蒙特卡洛：重复抽样算 J
J_samples = []
for _ in range(2000):
    fi = 0.005 + f_std * rng.standard_normal(T)
    epsi = 0.03 * rng.standard_normal((T, N))
    Ri = 1.0 + (mu_R[None, :] + np.outer(fi - fi.mean(), beta) + epsi)
    # 两步
    o1 = np.array([gbar(b, fi, Ri) @ gbar(b, fi, Ri) for b in bs])
    bh1 = bs[np.argmin(o1)]
    g0i = moments(bh1, fi, Ri)
    Si = (g0i.T @ g0i) / T
    Sinvi = np.linalg.pinv(Si)
    o2 = np.array([gbar(b, fi, Ri) @ Sinvi @ gbar(b, fi, Ri) for b in bs])
    bh2 = bs[np.argmin(o2)]
    gb = gbar(bh2, fi, Ri)
    J = T * (gb @ Sinvi @ gb)
    J_samples.append(J)
J_samples = np.array(J_samples)
dof = N - 1  # 矩数 N 减参数数 1

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.hist(J_samples, bins=50, density=True, color=C["j"], alpha=0.55,
        label=f"J 统计量经验分布（2000 次模拟）")
xx = np.linspace(0, max(J_samples.max(), 30), 300)
ax.plot(xx, stats.chi2.pdf(xx, dof), color=C["chi"], lw=2.4,
        label=f"理论 χ²(df={dof})")
crit = stats.chi2.ppf(0.95, dof)
ax.axvline(crit, color=C["err"], ls="--", lw=1.6, label=f"5% 临界值 {crit:.1f}")
ax.set_title("过度识别检验 J≈χ²(N−k)：模型设定正确则 J 落在分布内", fontsize=12)
ax.set_xlabel("J 统计量"); ax.set_ylabel("密度")
ax.grid(color=C["grid"], alpha=0.6); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "j_test_chi2.png"), dpi=130)
plt.close(fig)

# ========== 图3：定价误差 —— 正确模型 vs 遗漏因子的错误模型 ==========
# 正确模型的定价误差（拟合值 vs 实际平均超额收益）
m_fit = sdf(b_hat2, f)[:, None]
pricing_err_ok = (m_fit * Rg - 1.0).mean(axis=0)  # 每资产矩条件残差

# 错误模型：SDF 常数（b=0），无法解释横截面
pricing_err_bad = (np.ones((T, 1)) * Rg - 1.0).mean(axis=0)

xpos = np.arange(N)
fig, ax = plt.subplots(figsize=(9, 4.8))
wbar = 0.38
ax.bar(xpos - wbar/2, pricing_err_ok * 1e2, wbar, color=C["true"], alpha=0.85,
       label="GMM 拟合 SDF（矩条件≈0）")
ax.bar(xpos + wbar/2, pricing_err_bad * 1e2, wbar, color=C["err"], alpha=0.8,
       label="错误模型（常数 SDF，b=0）")
ax.axhline(0, color="#333", lw=0.8)
ax.set_title("定价误差对比：正确 SDF 把横截面误差压到≈0，错误模型系统性偏离", fontsize=12)
ax.set_xlabel("测试资产编号（按 β 递增）"); ax.set_ylabel("平均定价误差 (×10⁻²)")
ax.set_xticks(xpos); ax.set_xticklabels([f"A{i+1}" for i in range(N)], fontsize=8)
ax.grid(color=C["grid"], alpha=0.6, axis="y"); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "pricing_errors.png"), dpi=130)
plt.close(fig)

print("gmm-asset-pricing images done:", os.listdir(D))
print(f"b_true={b_true} b_hat1={b_hat1:.2f} b_hat2={b_hat2:.2f} lam_ols={lam_ols:.4f}")
print(f"J均值={J_samples.mean():.1f} 理论df={dof} 拒绝率={(J_samples>crit).mean():.3f}")
