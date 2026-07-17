#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「极值理论尾部依赖：用广义帕累托分布给极端联动定价」生成真实配图与统计数字。

核心逻辑(极值理论 EVT / 尾部依赖 Tail Dependence):
  - 阈值超限(Peaks-Over-Threshold, POT): 超过阈值 u 的超出量 x-u 近似服从
    广义帕累托分布 GPD:
        G(x) = 1 - (1 + ξ (x-u)/σ)^(-1/ξ),   x > u, 1+ξ(x-u)/σ > 0
    用 MLE 拟合 shape(ξ) 与 scale(σ)。ξ>0 厚尾; ξ=0 指数尾。
  - 均值超出函数(mean excess) e(u)=E(X-u|X>u): GPD 下 e(u)=σ/(1-ξ)+ξ u/(1-ξ),
    近似线性 -> 用 POT 图选阈值 u。
  - 尾部依赖系数(tail dependence) λ = lim_{q→1} P(X>F_X⁻¹(q) | Y>F_Y⁻¹(q)):
        * 高斯/椭圆正态: 渐近独立, λ=0 (极端不联动)。
        * 多元 t(ν) 相关 ρ: λ = 2·t_{ν+1}( -√((ν+1)(1-ρ)/(1+ρ)) ) > 0。
    实证估计: λ(q)= #{X>Q_x(q) 且 Y>Q_y(q)} / #{Y>Q_y(q)}。
  - 用蒙特卡洛生成「同相关 ρ 的高斯」与「同相关 ρ 的 t(ν=4)」两组双变量收益,
    对比两者尾部依赖: 高斯→0, t→理论 λ。

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  evt_gpd_fit.png      —— 单资产超限超出量直方图 vs 拟合 GPD 密度
  evt_pot.png          —— 均值超出函数(POT 图)选阈值
  evt_scatter.png      —— 高斯 vs t 双变量散点, 高亮联合上尾极端
  evt_taildep.png      —— 实证尾部依赖 λ(q) vs q: 高斯→0, t→理论值
"""
import os
import json
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "evt-tail-dependence")
os.makedirs(D, exist_ok=True)

C = {"gauss": "#2F4F8F", "t": "#C0392B", "gpd": "#27AE60", "band": "#C9D8F0",
     "grid": "#DDDDDD", "accent": "#E67E22", "hi": "#E67E22"}

# ----------------- 模拟参数 -----------------
rng = np.random.default_rng(20260717)
N = 40000                 # 蒙特卡洛样本量
rho = 0.6                 # 相关系数
nu = 4.0                  # t 分布自由度
Cov = np.array([[1.0, rho], [rho, 1.0]])

# 高斯双变量
z_g = rng.multivariate_normal([0, 0], Cov, size=N)
# 多元 t 双变量(用标准正态 / sqrt(χ²/ν) 构造, 再标准化边际)
z_n = rng.multivariate_normal([0, 0], Cov, size=N)
chi = rng.chisquare(nu, size=N)
t_scale = np.sqrt(nu / chi)[:, None]
z_t = z_n * t_scale       # 边际为标准化 t(ν)

# ---------------- GPD 拟合(用 t 边际 X 的超限) ----------------
x = z_t[:, 0]
u_frac = 0.90
u = np.quantile(x, u_frac)
exceed = x[x > u] - u
n_ex = len(exceed)

def gpd_nll(params, data):
    xi, sigma = params
    if sigma <= 0:
        return 1e10
    z = 1.0 + xi * data / sigma
    if np.any(z <= 0):
        return 1e10
    return len(data) * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(z))

res = optimize.minimize(gpd_nll, [0.1, np.std(exceed)], args=(exceed,),
                        method="Nelder-Mead",
                        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
xi_hat, sigma_hat = res.x
xi_hat = float(xi_hat)
sigma_hat = float(sigma_hat)

# GPD 理论均值超出 e(u)=σ/(1-ξ) + ξ u/(1-ξ)
def gpd_pdf(y, xi, sigma):
    z = 1.0 + xi * y / sigma
    z = np.clip(z, 1e-12, None)
    return (1.0 / sigma) * (z) ** (-(1.0 + 1.0 / xi))

# ---------------- 均值超出函数(POT) ----------------
qs = np.linspace(0.80, 0.98, 31)
mean_excess = []
for q in qs:
    uu = np.quantile(x, q)
    ex = x[x > uu] - uu
    mean_excess.append(np.mean(ex))
mean_excess = np.array(mean_excess)

# ---------------- 尾部依赖 λ(q) ----------------
def tail_dependence(a, b, q_grid):
    Qa = np.quantile(a, q_grid)
    Qb = np.quantile(b, q_grid)
    lam = np.array([np.mean((a > Qa[i]) & (b > Qb[i])) / np.mean(b > Qb[i])
                    for i in range(len(q_grid))])
    return lam

qg = np.linspace(0.90, 0.995, 21)
lam_g = tail_dependence(z_g[:, 0], z_g[:, 1], qg)
lam_t = tail_dependence(z_t[:, 0], z_t[:, 1], qg)
lam_t_last = float(lam_t[-1])

# 理论 t 尾部依赖
def t_taildep(rho, nu):
    z = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * stats.t.cdf(z, df=nu + 1.0)
lam_t_theory = float(t_taildep(rho, nu))

# ---------------- 图1: GPD 拟合 ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hist(exceed, bins=45, density=True, color=C["band"], edgecolor="white",
        label=f"超限超出量 (u=Q{int(u_frac*100)}, n={n_ex})")
yy = np.linspace(0, exceed.max(), 200)
ax.plot(yy, gpd_pdf(yy, xi_hat, sigma_hat), color=C["gpd"], lw=2.2,
        label=f"拟合 GPD (ξ={xi_hat:.3f}, σ={sigma_hat:.3f})")
ax.set_title("阈值超限的超出量服从广义帕累托分布(GPD)")
ax.set_xlabel("超出量 y = x − u")
ax.set_ylabel("密度")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "evt_gpd_fit.png"), dpi=130)
plt.close(fig)

# ---------------- 图2: POT 均值超出 ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(qs, mean_excess, "o-", color=C["accent"], lw=1.8, ms=4,
        label="实证均值超出 e(u)")
# GPD 线性外推(用 u 处拟合)
uu = np.quantile(x, u_frac)
qline = np.linspace(qs[0], 0.99, 100)
eline = sigma_hat / (1 - xi_hat) + xi_hat * (np.quantile(x, qline) - uu) / (1 - xi_hat)
ax.plot(qline, eline, "--", color=C["gpd"], lw=1.8, label="GPD 线性拟合 (ξ>0 厚尾)")
ax.axvline(u_frac, color=C["gauss"], ls=":", lw=1.4, label=f"选定阈值 u=Q{int(u_frac*100)}")
ax.set_title("POT 均值超出图：右端近似线性 → 厚尾(GPD 适用)")
ax.set_xlabel("阈值分位 u")
ax.set_ylabel("均值超出 E(X−u | X>u)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "evt_pot.png"), dpi=130)
plt.close(fig)

# ---------------- 图3: 散点(高斯 vs t) ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, data, title, col in [
        (axes[0], z_g, "高斯双变量 (ρ=0.6)", C["gauss"]),
        (axes[1], z_t, f"多元 t(ν={int(nu)}) (ρ=0.6)", C["t"])]:
    ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.25, color=col)
    qh = np.quantile(data, 0.95)
    mask = (data[:, 0] > qh) & (data[:, 1] > qh)
    ax.scatter(data[mask, 0], data[mask, 1], s=14, color=C["hi"],
               edgecolor="k", lw=0.3, label=f"联合上尾 (n={int(mask.sum())})")
    ax.set_title(title)
    ax.set_xlabel("资产 A 收益")
    ax.set_ylabel("资产 B 收益")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
fig.suptitle("同一相关系数下：高斯极端不联动，t 分布极端高度联动", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(D, "evt_scatter.png"), dpi=130)
plt.close(fig)

# ---------------- 图4: 尾部依赖曲线 ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(qg, lam_g, "o-", color=C["gauss"], lw=1.8, ms=4, label="高斯(渐近独立, λ→0)")
ax.plot(qg, lam_t, "s-", color=C["t"], lw=1.8, ms=4,
        label=f"t(ν={int(nu)}) 实证 λ")
ax.axhline(lam_t_theory, color=C["t"], ls="--", lw=1.4,
           label=f"t 理论 λ={lam_t_theory:.3f}")
ax.set_ylim(0, max(lam_t_theory * 1.25, 0.1))
ax.set_title("尾部依赖系数 λ(q)：极端联动程度")
ax.set_xlabel("分位 q（极端阈值）")
ax.set_ylabel("λ(q) = P(A>Q_A | B>Q_B)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "evt_taildep.png"), dpi=130)
plt.close(fig)

# ===================== 指标 =====================
metrics = {
    "n_sim": N, "rho": rho, "nu": nu, "u_frac": u_frac,
    "threshold_u": round(float(u), 4),
    "n_exceed": int(n_ex),
    "gpd_xi": round(xi_hat, 4),
    "gpd_sigma": round(sigma_hat, 4),
    "gpd_mean_excess_theory": round(float(sigma_hat/(1-xi_hat) + xi_hat*(u-uu)/(1-xi_hat)), 4),
    "lam_gauss_last": round(float(lam_g[-1]), 4),
    "lam_t_last": round(lam_t_last, 4),
    "lam_t_theory": round(lam_t_theory, 4),
    "taildep_ratio_t_over_gauss": round(float(lam_t_last / (lam_g[-1] + 1e-9)), 1),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("=== EVT TAIL DEPENDENCE METRICS ===")
for k_, v_ in metrics.items():
    print(f"{k_}: {v_}")
print("done.")
