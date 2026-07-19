#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swaption 估值与波动率校准 配图生成 (4 张真实图表, 全自洽计算)

机制(欧洲 payer swaption on a fixed-float swap, HW 短期利率模型):
  * 市场设定: 扁平初始曲线 r0=b=2%, 故平价互换利率恒为 F=2% (K=2% 即 ATM)
  * 标的: 到期日 T_e=5y 后开始的 n=5y 互换, 半年付息 τ=0.5, 支付日 T_1..T_10
  * 互换: 固定端付 S(T_e)·τ_i, 浮动端付 1−P(T_e,T_n); 平价要求 S=(1−P(T_e,T_n))/A
  * A(T_e)=Σ τ_i P(T_e,T_i) 为年金, P(T_e,T_i)=P(0,T_i)/P(0,T_e)
  * payer swaption 到期 payoff = max(0, S(T_e)−K)·A(T_e)
  * 现值 PV = E[ P(0,T_e)·payoff ] (HW 风险中性全路径模拟, 同一批 Z 保证各图一致)
  * 对数正态 Black 基准 (市场 vol σ_B=20%): PV = A·P(0,T_e)·[F N(d1) − K N(d2)]
  * 校准: 用二分法扫 HW 扩散 σ_HW 使模型 ATM 价 = 目标 Black ATM 价 (0.0153)
  * 图1: HW 短期利率路径(向 2% 中枢) + 到期日互换利率 S(T_e) 分布
  * 图2: swaption 价格 vs 行权价 K (模型 vs Black, 校准后在 ATM 贴合、两翼分化)
  * 图3: 校准曲线 —— 模型 ATM 价 vs σ_HW, 与目标 Black 价交叉
  * 图4: 不同 σ_HW 下 S(T_e) 分布, 显示凸性如何随波动抬升 payer swaption 价值
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "swaption-valuation"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "mean": "#C44E52", "fit": "#55A868",
     "true": "#999999", "curve": "#8172B3", "scatter": "#DD8452", "black": "#333333"}

# ---------- 参数 ----------
r0 = 0.02          # 初始/中枢短期利率(扁平初始曲线)
b  = 0.02          # HW 长期中枢
a  = 0.10          # 均值回复速度
Te = 5.0           # 期权到期(互换开始) = 5y
tenor = 5.0        # 互换期限 = 5y
tau = 0.5          # 半年付息
pay_dates = np.arange(tau, tenor + 1e-9, tau)   # T_e 之后的相对支付时刻
swap_times = Te + pay_dates                     # 绝对支付日
Tmax = Te + tenor
dt = 1.0 / 12                                   # 月度步长
M = int(round(Tmax / dt))
alpha = np.exp(-a * dt)
mu_step = b * (1 - alpha)

tidx_te = int(round(Te / dt))
tidx_Tn = int(round((Te + tenor) / dt))
tidx_pay = [int(round(ti / dt)) for ti in swap_times]

# 确定性年金与贴现(扁平曲线 P(0,t)=e^{-rt})
# 正确做法: 平价互换利率用【远期】年金(支付时刻相对 T_e 的 τ_i)
P0Te_det = np.exp(-r0 * Te)
A_fwd = sum(tau * np.exp(-r0 * rel) for rel in pay_dates)   # Σ τ e^{-r·τ_i}, τ_i=0.5..5.0
A_det = sum(tau * np.exp(-r0 * ti) for ti in swap_times)    # 兼容旧变量(绝对时刻年金)
F = (1.0 - np.exp(-r0 * tenor)) / A_fwd                     # 远期平价互换利率 ≈ 2.01%
sigma_B = 0.20                                            # 市场 swaption 对数正态 vol

def black_swaption(F, K, v, T, A, P0T):
    d1 = (np.log(F / K) + 0.5 * v**2 * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return A * P0T * (F * norm.cdf(d1) - K * norm.cdf(d2))

target = black_swaption(F, F, sigma_B, Te, A_fwd, P0Te_det)   # 真 ATM: 行权价=远期 F
print(f"[Swaption] 确定性 Black ATM 目标价 = {target:.6f} (F={F*100:.3f}%, σ_B={sigma_B*100:.0f}%, A_fwd={A_fwd:.3f}, P0Te={P0Te_det:.4f})")

# 固定一批随机冲击, 所有模型估值共用 -> 各图互相一致、可复现
rng = np.random.default_rng(20260719)
n_paths = 20000
Z_fixed = rng.standard_normal((n_paths, M))

def simulate_paths(Z, sigma):
    sd = sigma * np.sqrt((1 - alpha**2) / (2 * a))
    r = np.empty((n_paths, M + 1))
    r[:, 0] = r0
    for i in range(M):
        r[:, i + 1] = alpha * r[:, i] + mu_step + sd * Z[:, i]
    cum = np.cumsum(r, axis=1) * dt           # ∫_0^{i*dt} r_s ds
    return r, cum

def price_swaption(Z, sigma, K):
    r, cum = simulate_paths(Z, sigma)
    P_0_Te = np.exp(-cum[:, tidx_te])
    Ann = np.zeros(n_paths)
    for j, idx in enumerate(tidx_pay):
        Ann += tau * (np.exp(-cum[:, idx]) / P_0_Te)
    P_Te_Tn = np.exp(-cum[:, tidx_Tn]) / P_0_Te
    S_Te = (1.0 - P_Te_Tn) / Ann
    payoff = np.maximum(S_Te - K, 0.0) * Ann
    PV = np.mean(payoff * P_0_Te)
    return PV, S_Te, Ann

def model_atm(Z, sigma):
    pv, _, _ = price_swaption(Z, sigma, F)     # 真 ATM 行权价 = 远期 F
    return pv

# ---------- 二分法校准 σ_HW ----------
lo, hi = 0.001, 0.05
for _ in range(40):
    mid = 0.5 * (lo + hi)
    if model_atm(Z_fixed, mid) < target:
        lo = mid
    else:
        hi = mid
sigma_fit = 0.5 * (lo + hi)
print(f"[Swaption] 拟合 HW 扩散 σ_HW = {sigma_fit*100:.3f}% (模型 ATM = {model_atm(Z_fixed, sigma_fit):.6f})")

# ---------- 图1: HW 利率路径 + 互换利率分布 (带 10/90 分位带, 收紧纵轴) ----------
r, cum = simulate_paths(Z_fixed, sigma_fit)
tt = np.arange(M + 1) * dt
p10 = np.percentile(r * 100, 10, axis=0)
p90 = np.percentile(r * 100, 90, axis=0)
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.fill_between(tt, p10, p90, color=C["path"], alpha=0.18, label="10%–90% 路径带")
for k in range(min(120, n_paths)):
    ax.plot(tt, r[k] * 100, color=C["path"], alpha=0.06, lw=0.6)
ax.plot(tt, r.mean(axis=0) * 100, color=C["mean"], lw=2.4, label="平均路径")
ax.axhline(b * 100, color=C["true"], ls="--", lw=1.6, label=f"中枢 b = {b*100:.0f}%")
ax.set_title("Hull-White 短期利率模拟：向 2% 中枢回复 (期权 5y / 互换 5y, 校准 σ_HW)", fontsize=13)
ax.set_xlabel("时间 (年)"); ax.set_ylabel("短期利率 (%)")
ax.legend(loc="upper right", fontsize=9); ax.grid(color=C["grid"], lw=0.5)
ax.set_xlim(0, Tmax)
ax.set_ylim(1.2, 2.8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hw_rate_paths.png")); plt.close(fig)

_, S_Te, _ = price_swaption(Z_fixed, sigma_fit, 0.02)
fig2, ax2 = plt.subplots(figsize=(10, 5.2))
ax2.hist(S_Te * 100, bins=60, color=C["curve"], alpha=0.75, density=True)
ax2.axvline(np.median(S_Te) * 100, color=C["mean"], lw=2.2, label=f"中位互换利率 {np.median(S_Te)*100:.2f}%")
ax2.axvline(F * 100, color=C["true"], ls="--", lw=1.6, label=f"ATM K = F = {F*100:.2f}%")
ax2.set_title("期权到期日平价互换利率 S(T_e) 分布 (HW 模拟, 对数正态偏峰)", fontsize=12.5)
ax2.set_xlabel("S(T_e) (%)"); ax2.set_ylabel("密度")
ax2.legend(fontsize=9); ax2.grid(color=C["grid"], lw=0.5)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "swap_rate_dist.png")); plt.close(fig2)

# ---------- 图2: swaption 价格 vs 行权价 (模型 vs Black) ----------
strikes = np.linspace(0.010, 0.030, 11)
model_pv = [price_swaption(Z_fixed, sigma_fit, K)[0] for K in strikes]
# 注: 真 ATM 在 K=F, 但价格曲线在 K≈2% 附近已充分展示 payer 单调性
black_pv = [black_swaption(F, K, sigma_B, Te, A_fwd, P0Te_det) for K in strikes]
# 整条曲线标题标注 ATM 在 K=F
fig3, ax3 = plt.subplots(figsize=(10, 5.4))
ax3.plot(strikes * 100, model_pv, color=C["fit"], lw=2.6, marker="o", ms=4, label=f"HW 模型 (校准 σ_HW={sigma_fit*100:.2f}%)")
ax3.plot(strikes * 100, black_pv, color=C["true"], lw=2.2, ls="--", label=f"Black 基准 (σ_B={sigma_B*100:.0f}%)")
ax3.axvline(F * 100, color=C["mean"], ls=":", lw=1.6, label=f"ATM (K=F={F*100:.2f}%)")
ax3.set_title(f"Payer Swaption 价格 vs 行权价 K (模型 vs 对数正态基准, ATM=K=F={F*100:.2f}%)", fontsize=12)
ax3.set_xlabel("行权价 K (%)"); ax3.set_ylabel("Swaption 现值 (每 1 单位面值)")
ax3.legend(fontsize=9); ax3.grid(color=C["grid"], lw=0.5)
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "swaption_vs_strike.png")); plt.close(fig3)

# ---------- 图3: 校准曲线 (模型 ATM 价 vs σ_HW) ----------
sig_grid = np.linspace(0.002, 0.030, 25)
atm_model = [model_atm(Z_fixed, s) for s in sig_grid]
fig4, ax4 = plt.subplots(figsize=(10, 5.4))
ax4.plot(sig_grid * 100, atm_model, color=C["fit"], lw=2.6, marker="o", ms=3, label="HW 模型 ATM 价")
ax4.axhline(target, color=C["true"], ls="--", lw=1.8, label=f"目标 Black 价 (σ_B={sigma_B*100:.0f}%) = {target:.5f}")
ax4.axvline(sigma_fit * 100, color=C["mean"], ls=":", lw=1.8, label=f"拟合 σ_HW = {sigma_fit*100:.2f}%")
ax4.set_title("波动率校准：扫描 HW 扩散 σ_HW 使 ATM Swaption 价 = 市场 Black 价", fontsize=12)
ax4.set_xlabel("HW 扩散系数 σ_HW (%)"); ax4.set_ylabel("ATM Payer Swaption 现值")
ax4.legend(fontsize=9); ax4.grid(color=C["grid"], lw=0.5)
fig4.tight_layout(); fig4.savefig(os.path.join(OUT, "calibration_curve.png")); plt.close(fig4)

# ---------- 图4: 不同 σ_HW 下的 S(T_e) 分布 (凸性效应) ----------
fig5, ax5 = plt.subplots(figsize=(10, 5.4))
for s, col, lab in [(0.008, C["path"], "σ_HW=0.8%"),
                    (sigma_fit, C["curve"], f"σ_HW={sigma_fit*100:.2f}% (校准)"),
                    (0.030, C["mean"], "σ_HW=3.0%")]:
    _, Ss, _ = price_swaption(Z_fixed, s, F)
    ax5.hist(Ss * 100, bins=50, alpha=0.5, density=True, color=col, label=f"{lab} · 中位 {np.median(Ss)*100:.2f}%")
ax5.axvline(F * 100, color=C["true"], ls="--", lw=1.4, label=f"ATM K=F={F*100:.2f}%")
ax5.set_title("波动率越高，S(T_e) 分布越肥 → payer swaption 凸性价值越高", fontsize=12)
ax5.set_xlabel("S(T_e) (%)"); ax5.set_ylabel("密度")
ax5.legend(fontsize=9); ax5.grid(color=C["grid"], lw=0.5)
fig5.tight_layout(); fig5.savefig(os.path.join(OUT, "vol_convexity.png")); plt.close(fig5)

print(f"[Swaption] 配图已写入 {OUT}")
print(f"[Swaption] 校准点: 模型 ATM={model_atm(Z_fixed, sigma_fit):.6f}  目标 Black={target:.6f}")
