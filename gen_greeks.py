#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Greeks 全景与对冲 配图生成 (4 张真实图表, 全自洽计算)

机制 (欧式看涨 BS):
  * price = S·N(d1) − K e^{−rT} N(d2)
  * Delta = N(d1); Gamma = N'(d1)/(S·σ·√T); Vega = S·N'(d1)·√T (每 1.00 vol)
  * Theta = −S·N'(d1)·σ/(2√T) − r·K·e^{−rT}·N(d2) (每年); Rho = K·T·e^{−rT}·N(d2)
  * 参数: S0=100, K=100(ATM), T=0.5, r=2%, σ=25%
  * 图1: 看涨价格 / Delta / Gamma vs 标的价格 S (三面板)
  * 图2: Vega / Theta / Rho vs 标的价格 S (三面板)
  * 图3: Greeks 风险地图 —— Delta/Gamma/Vega/Theta 在 (标的价格冲击%, 波动率冲击%) 网格上的热力图 (2×2)
  * 图4: 对冲 P&L 分布 —— 未对冲 / 仅 Delta 对冲 / Delta+Gamma 对冲 的 1 日 P&L 直方图, 展示方差递减
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

SLUG = "greeks-hedging-map"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "mean": "#C44E52", "fit": "#55A868",
     "true": "#999999", "curve": "#8172B3", "scatter": "#DD8452", "black": "#333333"}

# ---------- 参数 ----------
S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.02, 0.25

def bs(S, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0      # 每 1% vol
    theta = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0   # 每天
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0          # 每 1% r
    return price, delta, gamma, vega, theta, rho

# ---------- 图1: price / delta / gamma vs S ----------
S_grid = np.linspace(60, 140, 200)
P_ = [bs(s, sigma)[0] for s in S_grid]
D_ = [bs(s, sigma)[1] for s in S_grid]
G_ = [bs(s, sigma)[2] for s in S_grid]
fig1, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.6))
a1.plot(S_grid, P_, color=C["fit"], lw=2.4); a1.axvline(K, color=C["true"], ls=":", lw=1.4)
a1.set_title("看涨期权价格 vs S"); a1.set_xlabel("S"); a1.set_ylabel("Price"); a1.grid(color=C["grid"], lw=0.5)
a2.plot(S_grid, D_, color=C["path"], lw=2.4); a2.axvline(K, color=C["true"], ls=":", lw=1.4)
a2.set_title("Delta vs S"); a2.set_xlabel("S"); a2.set_ylabel("Δ = N(d1)"); a2.grid(color=C["grid"], lw=0.5)
a3.plot(S_grid, G_, color=C["mean"], lw=2.4); a3.axvline(K, color=C["true"], ls=":", lw=1.4)
a3.set_title("Gamma vs S (ATM 处峰值)"); a3.set_xlabel("S"); a3.set_ylabel("Γ"); a3.grid(color=C["grid"], lw=0.5)
fig1.suptitle("第一阶 Greeks：价格 / Delta / Gamma (K=100, T=0.5, σ=25%)", fontsize=12.5)
fig1.tight_layout(); fig1.savefig(os.path.join(OUT, "price_delta_gamma.png")); plt.close(fig1)

# ---------- 图2: vega / theta / rho vs S ----------
V_ = [bs(s, sigma)[3] for s in S_grid]
Th_ = [bs(s, sigma)[4] for s in S_grid]
Rh_ = [bs(s, sigma)[5] for s in S_grid]
fig2, (b1, b2, b3) = plt.subplots(1, 3, figsize=(14, 4.6))
b1.plot(S_grid, V_, color=C["curve"], lw=2.4); b1.axvline(K, color=C["true"], ls=":", lw=1.4)
b1.set_title("Vega vs S (每 1% vol)"); b1.set_xlabel("S"); b1.set_ylabel("ν"); b1.grid(color=C["grid"], lw=0.5)
b2.plot(S_grid, Th_, color=C["scatter"], lw=2.4); b2.axvline(K, color=C["true"], ls=":", lw=1.4)
b2.set_title("Theta vs S (每天, 为负)"); b2.set_xlabel("S"); b2.set_ylabel("Θ"); b2.grid(color=C["grid"], lw=0.5)
b3.plot(S_grid, Rh_, color=C["fit"], lw=2.4); b3.axvline(K, color=C["true"], ls=":", lw=1.4)
b3.set_title("Rho vs S (每 1% r)"); b3.set_xlabel("S"); b3.set_ylabel("ρ"); b3.grid(color=C["grid"], lw=0.5)
fig2.suptitle("第二阶 / 跨变量 Greeks：Vega / Theta / Rho (K=100, T=0.5, σ=25%)", fontsize=12.5)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "vega_theta_rho.png")); plt.close(fig2)

# ---------- 图3: Greeks 风险地图 (2×2 heatmap) ----------
s_shock = np.linspace(-0.20, 0.20, 41)     # 标的价格冲击 ±20%
v_shock = np.linspace(-0.50, 0.50, 41)     # 波动率冲击 ±50%
SS, VV = np.meshgrid(s_shock, v_shock)
G_delta = np.zeros_like(SS); G_gamma = np.zeros_like(SS)
G_vega = np.zeros_like(SS); G_theta = np.zeros_like(SS)
for i in range(SS.shape[0]):
    for j in range(SS.shape[1]):
        s = S0 * (1 + SS[i, j]); vol = sigma * (1 + VV[i, j])
        _, d, g, vv, th, _ = bs(s, vol)
        G_delta[i, j] = d; G_gamma[i, j] = g; G_vega[i, j] = vv; G_theta[i, j] = th

fig3, axes = plt.subplots(2, 2, figsize=(12, 9))
titles = [("Delta 风险地图", G_delta, "Δ"), ("Gamma 风险地图", G_gamma, "Γ"),
          ("Vega 风险地图", G_vega, "ν"), ("Theta 风险地图 (每天)", G_theta, "Θ")]
for ax, (ti, mat, lab) in zip(axes.flat, titles):
    im = ax.contourf(SS * 100, VV * 100, mat, levels=20, cmap="RdYlBu_r")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_title(ti); ax.set_xlabel("标的价格冲击 (%)"); ax.set_ylabel("波动率冲击 (%)")
    fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
fig3.suptitle("Greeks 风险地图：横轴标的价格冲击、纵轴波动率冲击，中心(0,0)=当前持仓", fontsize=12.5)
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "greeks_heatmap.png")); plt.close(fig3)

# ---------- 图4: 对冲 P&L 分布 ----------
h = 1.0 / 52.0                 # 1 周对冲 horizon
rng = np.random.default_rng(20260719)
N = 200000
Z = rng.standard_normal(N)
S_h = S0 * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Z)

# 初始 Greeks
V0, D0, G0, _, _, _ = bs(S0, sigma)
# 第二个期权 (K2=110) 用于 Gamma 对冲
K2 = 110.0
def bs_K2(S, vol, Kk):
    d1 = (np.log(S / Kk) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    price = S * norm.cdf(d1) - Kk * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1); gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    return price, delta, gamma
V2_0, D2_0, G2_0 = bs_K2(S0, sigma, K2)
# Gamma 对冲: 持仓 w2 使总 Gamma=0 → w2 = −G0/G2_0; 再用股票中和 Delta: w1 = −(D0 + w2·D2_0)
w2 = -G0 / G2_0            # 持 w2 份期权2使组合 Gamma=0
w1 = -(D0 + w2 * D2_0)   # 空头 w1 份股票使组合 Delta=0 (short 即 +w1 份贡献, w1<0)

# 期末重估 (同 σ, 期限 T−h)
def reprice(S, Kk, Tk):
    d1 = (np.log(S / Kk) + (r + 0.5 * sigma**2) * Tk) / (sigma * np.sqrt(Tk))
    d2 = d1 - sigma * np.sqrt(Tk)
    return S * norm.cdf(d1) - Kk * np.exp(-r * Tk) * norm.cdf(d2)

Tk = T - h
V_h = reprice(S_h, K, Tk)
V2_h = reprice(S_h, K2, Tk)

pnl_unhedged = V_h - V0
pnl_delta = (V_h - V0) - D0 * (S_h - S0)
# 组合: 1份期权 - Δ0 股票 + w2 份期权2 - (Δ0+w2Δ2) 股票 → 自融资 P&L 中股票贡献 = +w1·dS (空头 w1<0)
pnl_delta_gamma = (V_h - V0) + w2 * (V2_h - V2_0) + w1 * (S_h - S0)

def astats(x):
    return x.mean(), x.std()

mu_u, sd_u = astats(pnl_unhedged)
mu_d, sd_d = astats(pnl_delta)
mu_dg, sd_dg = astats(pnl_delta_gamma)
print(f"[Greeks] P&L std: 未对冲={sd_u:.4f}  Delta={sd_d:.4f}  Delta+Gamma={sd_dg:.4f}")

fig4, ax4 = plt.subplots(figsize=(10, 5.4))
bins = np.linspace(-12, 12, 80)
ax4.hist(pnl_unhedged, bins=bins, alpha=0.45, color=C["scatter"], label=f"未对冲 (σ={sd_u:.2f})")
ax4.hist(pnl_delta, bins=bins, alpha=0.55, color=C["path"], label=f"仅 Delta 对冲 (σ={sd_d:.2f})")
ax4.hist(pnl_delta_gamma, bins=bins, alpha=0.6, color=C["fit"], label=f"Delta+Gamma 对冲 (σ={sd_dg:.2f})")
ax4.axvline(0, color="k", ls="--", lw=1.0)
ax4.set_title("对冲 P&L 分布 (1 周, 已实现 vol=隐含 vol)：Gamma 对冲进一步压低方差", fontsize=12)
ax4.set_xlabel("组合 P&L (指数点)"); ax4.set_ylabel("频次")
ax4.legend(fontsize=9); ax4.grid(color=C["grid"], lw=0.5)
fig4.tight_layout(); fig4.savefig(os.path.join(OUT, "hedge_pnl.png")); plt.close(fig4)

print(f"[Greeks] 配图已写入 {OUT}")
print(f"[Greeks] 初始 Delta={D0:.4f} Gamma={G0:.5f} Vega={bs(S0,sigma)[3]:.4f} Theta={bs(S0,sigma)[4]:.4f}")
