#!/usr/bin/env python3
"""
为文章「GARCH 波动率模型族：从 ARCH 到 EGARCH 的实战预测」(garch-volatility-family)
生成真实配图。所有图表均由文中 Python 代码真实计算生成（GARCH 模拟 + MLE 拟合）。

图表：
  1. garch_returns_clustering.png   模拟收益与真实条件波动（波动率聚集可视化）
  2. garch_fit_condvol.png          GARCH(1,1) 拟合的条件波动率 vs |收益|
  3. garch_forecast_fan.png         样本外多步波动率预测扇形带 vs 实现波动
  4. garch_news_impact.png          新闻冲击曲线：对称 GARCH vs 非对称 EGARCH/GJR 的杠杆效应
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
D = os.path.join(BASE, "garch-volatility-family")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============================================================
# 0) 模拟一个带杠杆效应的 GARCH(1,1) 过程
#    σ_t^2 = ω + α r_{t-1}^2 + β σ_{t-1}^2
#    叠加非对称：下跌对波动的冲击 > 上涨
# ============================================================
def simulate_garch(n, omega=1e-5, alpha=0.08, beta=0.90, leverage=0.05, mu=0.0002):
    r = np.zeros(n)
    sig2 = np.zeros(n)
    sig2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        z = rng.standard_normal()
        # 非对称冲击：前一日下跌(负收益)额外放大波动
        asy = leverage * (-r[t - 1]) if r[t - 1] < 0 else 0.0
        sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1] + asy * (r[t - 1] < 0)
        r[t] = mu + np.sqrt(sig2[t]) * z
    return r, np.sqrt(sig2)

N = 3000
r, true_sig = simulate_garch(N)

# 拆分：前 2500 拟合，后 500 做样本外
n_fit = 2500
r_fit = r[:n_fit]
r_oos = r[n_fit:]

# ============================================================
# 1) GARCH(1,1) 负对数似然 + MLE 拟合
# ============================================================
def garch11_nll(params, r):
    omega, alpha, beta = params
    n = len(r)
    sig2 = np.zeros(n)
    sig2[0] = np.var(r)
    ll = 0.0
    for t in range(1, n):
        sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
        if sig2[t] <= 1e-12:
            return 1e10
        ll += 0.5 * (np.log(2 * np.pi) + np.log(sig2[t]) + r[t] ** 2 / sig2[t])
    return ll

def fit_garch11(r):
    x0 = [1e-5, 0.1, 0.85]
    bnds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]
    res = minimize(garch11_nll, x0, args=(r,), bounds=bnds, method="L-BFGS-B")
    return res.x, res.fun

def garch11_filter(params, r):
    omega, alpha, beta = params
    n = len(r)
    sig2 = np.zeros(n)
    sig2[0] = np.var(r)
    for t in range(1, n):
        sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
    return sig2

omega_hat, alpha_hat, beta_hat = fit_garch11(r_fit)[0]
print(f"GARCH(1,1) 拟合: omega={omega_hat:.2e}, alpha={alpha_hat:.3f}, beta={beta_hat:.3f}, "
      f"persistence(alpha+beta)={alpha_hat+beta_hat:.3f}")

# ============================================================
# 2) ARCH(1) 与 GJR-GARCH(1,1) 拟合（用于对比与新闻冲击）
# ============================================================
def arch1_nll(params, r):
    omega, alpha = params
    n = len(r); sig2 = np.zeros(n); sig2[0] = np.var(r); ll = 0.0
    for t in range(1, n):
        sig2[t] = omega + alpha * r[t - 1] ** 2
        if sig2[t] <= 1e-12:
            return 1e10
        ll += 0.5 * (np.log(2 * np.pi) + np.log(sig2[t]) + r[t] ** 2 / sig2[t])
    return ll

def gjr_nll(params, r):
    omega, alpha, gamma, beta = params
    n = len(r); sig2 = np.zeros(n); sig2[0] = np.var(r); ll = 0.0
    for t in range(1, n):
        sig2[t] = omega + (alpha + gamma * (r[t - 1] < 0)) * r[t - 1] ** 2 + beta * sig2[t - 1]
        if sig2[t] <= 1e-12:
            return 1e10
        ll += 0.5 * (np.log(2 * np.pi) + np.log(sig2[t]) + r[t] ** 2 / sig2[t])
    return ll

res_arch = minimize(arch1_nll, [1e-5, 0.1], args=(r_fit,), bounds=[(1e-8, None), (1e-6, 0.8)], method="L-BFGS-B")
res_gjr = minimize(gjr_nll, [1e-5, 0.05, 0.05, 0.85], args=(r_fit,),
                   bounds=[(1e-8, None), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.999)], method="L-BFGS-B")
gjr_params = res_gjr.x
print(f"GJR-GARCH 拟合: omega={gjr_params[0]:.2e}, alpha={gjr_params[1]:.3f}, "
      f"gamma={gjr_params[2]:.3f}, beta={gjr_params[3]:.3f}")

# ============================================================
# 图 1：模拟收益 + 真实条件波动（波动率聚集）
# ============================================================
fig, ax1 = plt.subplots(figsize=(11, 5.4))
ax1.fill_between(np.arange(N), r * 100, color="#9ecae1", lw=0, alpha=0.7, label="日收益 (%)")
ax1.set_ylabel("日收益 (%)", fontsize=11, color="#1f77b4")
ax1.set_xlabel("交易日", fontsize=11)
ax2 = ax1.twinx()
ax2.plot(np.arange(N), true_sig * 100, color="#d62728", lw=1.3, label="真实条件波动 σ_t (%)")
ax2.set_ylabel("真实条件波动 (%)", fontsize=11, color="#d62728")
ax1.set_title("波动率聚集：平静期与暴风雨期成串出现（GARCH 模拟）", fontsize=12.5, fontweight="bold")
ax1.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "garch_returns_clustering.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 2：GARCH(1,1) 拟合条件波动 vs |收益|
# ============================================================
sig2_fit = garch11_filter([omega_hat, alpha_hat, beta_hat], r_fit)
sig_fit = np.sqrt(sig2_fit)
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(np.abs(r_fit) * 100, color="#9ecae1", lw=0.8, alpha=0.8, label="|日收益| (%)")
ax.plot(sig_fit * 100, color="#d62728", lw=1.6, label="GARCH(1,1) 条件波动 σ̂_t (%)")
ax.set_xlabel("交易日（样本内）", fontsize=11)
ax.set_ylabel("波动 (%)", fontsize=11)
ax.set_title("GARCH(1,1) 拟合：条件波动率紧跟 |收益| 的起伏", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "garch_fit_condvol.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 3：样本外多步波动率预测扇形带
# ============================================================
# 递归预测：σ̂_{T+h|T}^2 = ω + (α+β) σ̂_{T+h-1|T}^2 + α r_{T+h-1}^2 (已知 r 时)
# 对未知 r，用 E[r^2|σ]=σ^2，故不变分量递推；给出 1σ 扇形（假设正态）
h = len(r_oos)
sig2_last = sig_fit[-1] ** 2
fore_var = np.zeros(h)          # 条件方差的点预测
fore_var[0] = omega_hat + alpha_hat * r_fit[-1] ** 2 + beta_hat * sig2_last
for i in range(1, h):
    fore_var[i] = omega_hat + (alpha_hat + beta_hat) * fore_var[i - 1]
fore_sig = np.sqrt(fore_var)
realized = np.abs(r_oos) * 100
fc = fore_sig * 100
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(realized, color="#9ecae1", lw=1.0, alpha=0.9, label="样本外 |实现收益| (%)")
ax.plot(fc, color="#d62728", lw=1.8, label="GARCH 多步预测 σ̂ (%)")
ax.fill_between(np.arange(h), (fc * 0.7), (fc * 1.3), color="#d62728", alpha=0.10,
                label="预测区间 (≈±30%)")
ax.set_xlabel("样本外交易日", fontsize=11)
ax.set_ylabel("波动 (%)", fontsize=11)
ax.set_title(f"样本外预测：GARCH 把波动拉回长期水平（persistence={alpha_hat+beta_hat:.3f}）",
             fontsize=12.0, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "garch_forecast_fan.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 4：新闻冲击曲线（非对称）
#   对称 GARCH: σ_t^2 只依赖 |r_{t-1}|（这里用 r^2 表达，曲线左右对称）
#   GJR: σ_t^2 = ω + (α+γ·I_{r<0}) r_{t-1}^2 + β σ_{t-1}^2 → 负冲击更高
#   设定前一期波动 σ_{t-1}=某基准，横轴为前一日收益 e=r_{t-1}
# ============================================================
e = np.linspace(-0.06, 0.06, 200)
sig_prev = 0.01  # 基准前一期波动
# 对称 GARCH 条件（用拟合的 ω,α,β，且把非对称项置 0）
garch_next = omega_hat + alpha_hat * e ** 2 + beta_hat * sig_prev ** 2
# GJR 非对称
gjr_next = omega_hat + (alpha_hat + gjr_params[2] * (e < 0)) * e ** 2 + gjr_params[3] * sig_prev ** 2
# 以对称 GARCH 在 e=0 的值为基准，画增量（相对冲击）
base = omega_hat + beta_hat * sig_prev ** 2
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(e * 100, (garch_next - base) * 1e4, color="#1f77b4", lw=2.2, label="对称 GARCH(1,1)：冲击仅看幅度（左右对称）")
ax.plot(e * 100, (gjr_next - base) * 1e4, color="#d62728", lw=2.2, label="GJR-GARCH：下跌冲击 > 上涨（杠杆效应）")
ax.axvline(0, color="gray", lw=1.0, ls=":")
ax.set_xlabel("前一日收益 r_{t-1} (%)", fontsize=11)
ax.set_ylabel("次日方差增量 (×1e-4)", fontsize=11)
ax.set_title("新闻冲击曲线：坏消息比好消息更能推高波动", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "garch_news_impact.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ GARCH 配图生成完成：{sorted(os.listdir(D))}")
