#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vanna-Volga 校准 配图生成 (4 张真实图表, 自洽合成, 半解析微笑校正)

机制(自洽合成, 仅用于演示方法):
  * 已知真实市场微笑 sigma_mkt(K) = sigma_atm + a*ln(K/F) + b*ln(K/F)^2
  * 三个支柱期权(put wing / ATM forward / call wing)的市场价 = BS(sigma_mkt(Ki))
  * 用 ATM vol 做 BS 基准价 V_BS(K) = BS(sigma_atm)
  * Vanna-Volga 三支柱插值校正:
        V(K) = V_BS(K) + w1*(C_mkt(K1)-V_BS(K1)) + w3*(C_mkt(K3)-V_BS(K3))
        (ATM 处 sigma_mkt=sigma_atm 故 delta2=0)
        其中 w1,w3 为对数moneyness的 Lagrange 基函数
  * 关键自洽检验: VV 在三个支柱上精确还原市场价(IV 残差=0)
  * 图1: 市场真实微笑 + 三个支柱 + 平值参考线
  * 图2: VV 反推 IV vs 真实微笑(支柱处完全重合, 中间平滑插值)
  * 图3: 校正项 dV(K) 分解为 put 翼 / call 翼贡献
  * 图4: 三支柱基函数 w1/w2/w3 (partition of unity)
"""
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "vanna-volga-smile"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"truth": "#4C72B0", "rec": "#C44E52", "atm": "#55A868",
     "put": "#8172B3", "call": "#DD8452", "grid": "#DDDDDD", "line": "#999999"}

# ---------- 参数 ----------
S = 100.0; r = 0.0; q = 0.0; T = 1.0
F = S * np.exp((r - q) * T)          # 远期 = ATM
sigma_atm = 0.20
a_skew = -0.15                       # 斜率(偏度, 左尾肥)
b_conv = 0.30                        # 凸度(微笑)

def sigma_mkt(K):
    m = np.log(K / F)
    return sigma_atm + a_skew * m + b_conv * m**2

# ---------- BS 工具 ----------
def bs_call(S, K, T, sigma):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, T, sigma):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(S, K, T, price, lo=1e-4, hi=5.0, tol=1e-9):
    lo, hi = lo, hi
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        p = bs_call(S, K, T, mid)
        if p > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

# ---------- 三支柱 ----------
K1, K2, K3 = 0.90 * F, F, 1.10 * F
C_mkt = {K1: bs_call(S, K1, T, sigma_mkt(K1)),
         K2: bs_call(S, K2, T, sigma_mkt(K2)),
         K3: bs_call(S, K3, T, sigma_mkt(K3))}
V_BS = {K: bs_call(S, K, T, sigma_atm) for K in (K1, K2, K3)}
print(f"put wing : mkt={C_mkt[K1]:.4f}  BS(atm)={V_BS[K1]:.4f}  d={C_mkt[K1]-V_BS[K1]:+.4f}")
print(f"ATM      : mkt={C_mkt[K2]:.4f}  BS(atm)={V_BS[K2]:.4f}  d={C_mkt[K2]-V_BS[K2]:+.4f}")
print(f"call wing: mkt={C_mkt[K3]:.4f}  BS(atm)={V_BS[K3]:.4f}  d={C_mkt[K3]-V_BS[K3]:+.4f}")

# ---------- Vanna-Volga 校正 (对数moneyness 基函数) ----------
def vv_price(K):
    x1, x2, x3 = np.log(K1 / K), np.log(K2 / K), np.log(K3 / K)
    w1 = x2 * x3 / ((x2 - x1) * (x3 - x1))
    w2 = x1 * x3 / ((x1 - x2) * (x3 - x2))
    w3 = x1 * x2 / ((x1 - x3) * (x2 - x3))
    d1 = C_mkt[K1] - V_BS[K1]
    d2 = C_mkt[K2] - V_BS[K2]
    d3 = C_mkt[K3] - V_BS[K3]
    return bs_call(S, K, T, sigma_atm) + w1 * d1 + w2 * d2 + w3 * d3, (w1, w2, w3)

def vv_weights(K):
    x1, x2, x3 = np.log(K1 / K), np.log(K2 / K), np.log(K3 / K)
    w1 = x2 * x3 / ((x2 - x1) * (x3 - x1))
    w2 = x1 * x3 / ((x1 - x2) * (x3 - x2))
    w3 = x1 * x2 / ((x1 - x3) * (x2 - x3))
    return w1, w2, w3

# ---------- 扫描 ----------
Ks = np.linspace(80, 120, 161)
iv_true = sigma_mkt(Ks)
iv_vv = np.array([implied_vol(S, K, T, vv_price(K)[0]) for K in Ks])
dV = np.array([vv_price(K)[0] - bs_call(S, K, T, sigma_atm) for K in Ks])
# 分解: put翼贡献 w1*d1, call翼贡献 w3*d3 (ATM d2=0)
d1 = C_mkt[K1] - V_BS[K1]
d3 = C_mkt[K3] - V_BS[K3]
comp_put = np.array([vv_weights(K)[0] * d1 for K in Ks])
comp_call = np.array([vv_weights(K)[2] * d3 for K in Ks])
resid = iv_vv - iv_true
print(f"VV 恢复最大 |IV残差| = {np.max(np.abs(resid)):.5f}  (支柱处理论=0)")
print(f"支柱1 K={K1:.1f}: 真值IV={sigma_mkt(K1):.4f} VV={implied_vol(S,K1,T,vv_price(K1)[0]):.4f}")
print(f"支柱3 K={K3:.1f}: 真值IV={sigma_mkt(K3):.4f} VV={implied_vol(S,K3,T,vv_price(K3)[0]):.4f}")

# ---------- 图1: 市场真实微笑 + 三支柱 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(Ks, iv_true * 100, color=C["truth"], lw=2.6, label="真实市场微笑 σ_mkt(K)")
ax.axhline(sigma_atm * 100, color=C["atm"], ls="--", lw=1.8, label=f"ATM 平值 vol = {sigma_atm:.0%}")
ax.plot(K1, sigma_mkt(K1) * 100, "o", ms=9, color=C["put"], label="put 翼支柱 K₁=90")
ax.plot(K2, sigma_mkt(K2) * 100, "s", ms=9, color=C["atm"], label="ATM 远期支柱 K₂=100")
ax.plot(K3, sigma_mkt(K3) * 100, "^", ms=9, color=C["call"], label="call 翼支柱 K₃=110")
ax.set_title("市场真实波动率微笑：左尾偏斜(smirk) + 两翼抬升", fontsize=13)
ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 (%)")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vv_smile.png")); plt.close(fig)

# ---------- 图2: VV 反推 IV vs 真实微笑 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(Ks, iv_true * 100, color=C["truth"], lw=2.6, label="真实微笑 (目标)")
ax.plot(Ks, iv_vv * 100, "--", color=C["rec"], lw=2.2, label="Vanna-Volga 反推 IV")
ax.scatter([K1, K2, K3], np.array([sigma_mkt(K1), sigma_mkt(K2), sigma_mkt(K3)]) * 100,
           s=90, facecolors="none", edgecolors="k", zorder=5, label="三支柱(精确还原)")
ax.set_title("VV 校准一致性：在三个支柱处精确还原市场 IV", fontsize=13)
ax.set_xlabel("行权价 K"); ax.set_ylabel("隐含波动率 (%)")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vv_recovery.png")); plt.close(fig)

# ---------- 图3: 校正项 dV 分解 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(Ks, dV, color=C["line"], lw=2.4, label="总校正 ΔV(K) = VV − BS(atm)")
ax.plot(Ks, comp_put, color=C["put"], lw=1.8, ls=":", label="put 翼贡献 w₁·Δ₁")
ax.plot(Ks, comp_call, color=C["call"], lw=1.8, ls=":", label="call 翼贡献 w₃·Δ₃")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(K2, color=C["atm"], ls="--", lw=1.0, alpha=0.6)
ax.set_title("半解析校正项：把微笑从 BS 常数里补回去", fontsize=13)
ax.set_xlabel("行权价 K"); ax.set_ylabel("价格校正 ΔV")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vv_correction.png")); plt.close(fig)

# ---------- 图4: 三支柱基函数 ----------
Ks4 = np.linspace(60, 140, 401)
W = np.array([vv_weights(K) for K in Ks4])
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(Ks4, W[:, 0], color=C["put"], lw=2.2, label="w₁(K) — put 翼基函数")
ax.plot(Ks4, W[:, 1], color=C["atm"], lw=2.2, label="w₂(K) — ATM 基函数")
ax.plot(Ks4, W[:, 2], color=C["call"], lw=2.2, label="w₃(K) — call 翼基函数")
ax.axvline(K1, color=C["put"], ls=":", lw=1.0); ax.axvline(K2, color=C["atm"], ls=":", lw=1.0)
ax.axvline(K3, color=C["call"], ls=":", lw=1.0)
ax.set_title("三支柱 Lagrange 基函数：在支点处取 1、其余取 0", fontsize=13)
ax.set_xlabel("行权价 K"); ax.set_ylabel("基函数权重 w")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vv_basis.png")); plt.close(fig)

print("generated:", sorted(os.listdir(OUT)))
