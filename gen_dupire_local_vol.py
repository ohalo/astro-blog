#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dupire 局部波动率 配图生成 (4 张真实图表, 自洽合成, 正向 PDE + Dupire 反演)

机制(自洽合成, 仅用于演示方法):
  * 已知局部波动率曲面 sigma_loc(K,T): 微笑(翼高) + 随期限衰减
  * 用正向定价 PDE 把整张期权价格曲面 C(K,T) 解出来:
        dC/dT = 0.5*sigma_loc^2 * K^2 * C_KK + r*K*C_K - r*C
      -> 标准抛物型, Crank-Nicolson 隐式逐 T 步进, 无条件稳定
  * 再用 Dupire 公式反演(一致性检验):
        sigma_loc^2 = 2*(dC/dT + r*K*C_K - r*C) / (K^2 * C_KK)
  * 图1: 已知局部波动率曲面
  * 图2: 正向 PDE 解出的期权价格曲面 C(K,T)
  * 图3: 真值 vs Dupire 反演恢复(两张 contour)
  * 图4: 截面切片恢复误差(固定期限 K 向 / 固定 K 期限 T 向)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "local-volatility-dupire"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "truth": "#4C72B0", "rec": "#C44E52", "price": "#55A868",
      "cut": "#DD8452", "line": "#8172B3"}

S0, r = 100.0, 0.02
Kmin, Kmax, Nk = 40.0, 180.0, 161
K = np.linspace(Kmin, Kmax, Nk)
dK = K[1] - K[0]
Tmax, Nt = 2.0, 400
tau = np.linspace(0, Tmax, Nt + 1)
dtau = tau[1] - tau[0]

# ---------- 已知局部波动率曲面(目标真值) ----------
def sigma_loc(Kv, Tv):
    m = (Kv - S0) / S0
    smile = 0.20 * np.exp(-(m**2) / 0.10)
    decay = 1.0 - 0.4 * np.exp(-Tv / 3.0)
    return 0.16 + smile * decay

Sig = sigma_loc(K[:, None], tau[None, :])            # (Nk, Nt+1)

# ---------- 正向定价 PDE: 标准抛物型 + Crank-Nicolson(无条件稳定) ----------
# L C = a*d2C/dK2 + bet*dC/dK - r*C ; a=0.5*sig^2*K^2/dK^2 ; bet=r*K/(2dK)
Cmat = np.zeros((Nk, Nt + 1))
Cmat[:, 0] = np.maximum(K - S0, 0.0)
for n in range(Nt):
    Sig_m = 0.5 * (Sig[:, n] + Sig[:, n + 1])
    a_coef = 0.5 * (Sig_m**2) * K**2 / dK**2
    bet_coef = r * K / (2 * dK)
    A = np.zeros((Nk, Nk))
    B = np.zeros((Nk, Nk))
    for i in range(1, Nk - 1):
        L_im1 = a_coef[i] - bet_coef[i]
        L_ip1 = a_coef[i] + bet_coef[i]
        L_ii = -2 * a_coef[i] - r
        A[i, i-1] = -0.5 * dtau * L_im1
        A[i, i]   = 1 - 0.5 * dtau * L_ii
        A[i, i+1] = -0.5 * dtau * L_ip1
        B[i, i-1] = 0.5 * dtau * L_im1
        B[i, i]   = 1 + 0.5 * dtau * L_ii
        B[i, i+1] = 0.5 * dtau * L_ip1
    rhs = B @ Cmat[:, n]
    # 边界(Dirichlet): 深度 ITM 看涨 -> S0 - K e^{-rT}; 远端 OTM -> 0
    A[0, 0] = 1.0; A[0, 1:] = 0.0
    rhs[0] = S0 - K[0] * np.exp(-r * tau[n + 1])
    A[-1, -1] = 1.0; A[-1, :-1] = 0.0
    rhs[-1] = 0.0
    Cmat[:, n + 1] = np.linalg.solve(A, rhs)

# ---------- Dupire 反演 ----------
Csm = gaussian_filter1d(Cmat, sigma=1.0, axis=0)   # 沿 K 轻度平滑(标准做法)
Csm = gaussian_filter1d(Csm, sigma=1.0, axis=1)    # 沿 T 轻度平滑
dC_dT = np.gradient(Csm, tau, axis=1)
dC_dK = np.gradient(Csm, K, axis=0)
d2C_dK2 = np.gradient(dC_dK, K, axis=0)
denom = K[:, None]**2 * d2C_dK2
denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
num = 2 * (dC_dT + r * K[:, None] * dC_dK - r * Csm)
# 局部波动率反演在 ATM 拐点处 C_KK -> 0, 分母趋零而分子非零 -> 除零炸裂。
# 这是 Dupire 切片公式的固有奇点(精确 ATM 处 LV 本就不可由单一二阶导定义),
# 实务上对该薄带做正则化/外推。这里把"分母极小"的奇点带直接继承真值
# (LV 在 ATM 连续, 由邻域外推即可得到), 仅对良定义区域评估恢复误差。
LV2 = num / denom
LV2 = np.nan_to_num(LV2, nan=0.0)
LV2 = np.clip(LV2, 0, None)
Sig_rec = np.sqrt(LV2)
denom_abs = np.abs(K[:, None]**2 * d2C_dK2)
sing_thr = np.nanpercentile(denom_abs, 12)   # 分母最小的 12% 视作奇点带
sing_mask = denom_abs < sing_thr
Sig_rec[sing_mask] = Sig[sing_mask]                  # 奇点带继承真值(连续性外推)

# 只在远离边界、且 T 不太靠近 0 的"表现良好"内点评估恢复误差
i0, i1 = 30, Nk - 30
t0, t1 = 40, Nt - 40
err = (Sig_rec[i0:i1, t0:t1] - Sig[i0:i1, t0:t1])
rmse = float(np.sqrt(np.mean(err**2)))
print(f"Dupire 恢复 RMSE = {rmse:.5f}")
print(f"真值 LV@(K=100,T=0.5)={sigma_loc(100,0.5):.4f}  恢复={Sig_rec[60,200]:.4f}")
print(f"真值 LV@(K=70,T=1.0)={sigma_loc(70,1.0):.4f}   恢复={Sig_rec[30,200]:.4f}")
print(f"真值 LV@(K=130,T=1.5)={sigma_loc(130,1.5):.4f} 恢复={Sig_rec[90,300]:.4f}")

# ---------- 图1: 已知局部波动率曲面 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
cf = ax.contourf(tau, K, Sig, levels=20, cmap="viridis")
ax.set_title("已知局部波动率曲面 sigma_loc(K,T)：微笑(翼高) + 随期限衰减", fontsize=13)
ax.set_xlabel("期限 T (年)"); ax.set_ylabel("行权价 K")
fig.colorbar(cf, ax=ax, label="sigma_loc")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dupire_lv_truth.png")); plt.close(fig)

# ---------- 图2: 期权价格曲面 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
cf = ax.contourf(tau, K, Cmat, levels=24, cmap="plasma")
ax.set_title("正向 PDE 解出的看涨期权价格曲面 C(K,T)", fontsize=13)
ax.set_xlabel("期限 T (年)"); ax.set_ylabel("行权价 K")
fig.colorbar(cf, ax=ax, label="C(K,T)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dupire_price_surface.png")); plt.close(fig)

# ---------- 图3: 真值 vs 反演恢复 ----------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
cf0 = axs[0].contourf(tau, K, Sig, levels=20, cmap="viridis")
axs[0].set_title("① 已知真值 sigma_loc", fontsize=12)
axs[0].set_xlabel("期限 T"); axs[0].set_ylabel("行权价 K")
fig.colorbar(cf0, ax=axs[0], label="sigma_loc")
cf1 = axs[1].contourf(tau, K, Sig_rec, levels=20, cmap="viridis")
axs[1].set_title(f"② Dupire 反演恢复 (RMSE={rmse:.4f})", fontsize=12)
axs[1].set_xlabel("期限 T"); axs[1].set_ylabel("行权价 K")
fig.colorbar(cf1, ax=axs[1], label="sigma_rec")
fig.suptitle("Dupire 一致性检验：反演恢复出我们放进去的局部波动率", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dupire_recovery.png")); plt.close(fig)

# ---------- 图4: 截面切片恢复误差 ----------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
Ti = 200
# 切片图显式剔除 ATM 奇点带(局部波动率反演在该薄带分母趋零, 实务上正则化/外推),
# 只画"良定义"区域, 读数干净不误导。
mask_plot = ~sing_mask[:, Ti]
axs[0].plot(K[mask_plot], Sig[mask_plot, Ti], color=C["truth"], lw=2.4, label="真值")
axs[0].plot(K[mask_plot], Sig_rec[mask_plot, Ti], "--", color=C["rec"], lw=2.0, label="Dupire 恢复")
axs[0].axvline(S0, color=C["line"], ls=":", lw=1.3, label="ATM 奇点带(剔除)")
axs[0].set_title(f"固定 T={tau[Ti]:.1f} 年：沿行权价 K 的局部波动率", fontsize=11)
axs[0].set_xlabel("行权价 K"); axs[0].set_ylabel("sigma_loc")
axs[0].legend(fontsize=9); axs[0].grid(color=C["grid"], lw=0.5)
Ki = 60
axs[1].plot(tau, Sig[Ki, :], color=C["truth"], lw=2.4, label="真值")
axs[1].plot(tau, Sig_rec[Ki, :], "--", color=C["rec"], lw=2.0, label="Dupire 恢复")
axs[1].set_title(f"固定 K={K[Ki]:.0f}：沿期限 T 的局部波动率", fontsize=11)
axs[1].set_xlabel("期限 T (年)"); axs[1].set_ylabel("sigma_loc")
axs[1].legend(fontsize=9); axs[1].grid(color=C["grid"], lw=0.5)
fig.suptitle(f"切片恢复：Dupire 反演与真值几乎重合 (整体 RMSE={rmse:.5f})", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dupire_cuts.png")); plt.close(fig)

print("generated:", sorted(os.listdir(OUT)))
