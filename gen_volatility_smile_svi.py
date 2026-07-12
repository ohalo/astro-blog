#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「波动率微笑的 SVI 参数化：用一条曲线吃下整张隐含波动率曲面」生成真实配图与真实数字。

核心概念(J. Gatheral, "The Volatility Surface", 2004/2006):
  - 隐含波动率曲面: IV 是 log-资金费率 k=ln(K/F) 与到期 T 的函数
  - 股票微笑常是『smirk』: 左偏(虚值看跌 IV 偏高) + 凸性
  - 原始 SVI 参数化(对总方差 w(k)=σ_imp^2·T):
        w(k) = a + b·[ ρ·(k - m) + sqrt((k - m)^2 + σ^2) ]
        参数约束: a∈R, b>0, |ρ|<1, m∈R, σ>0
  - 风险中性密度(Breeden-Litzenberger / Gatheral):
        f_K(K) = (1/√w)·φ(k/√w)·[ 1 - (k/2w)w'(k) + (1/4)(1/w - 1/4w^2)(w')^2 + (1/2)w''(k) ]
        蝶式套利 ⇔ f_K(K)<0; 坏的 σ 会让密度变负

数据: 在若干到期 T 上合成带 smirk 的『市场』IV 微笑(含轻噪声), 用最小二乘逐月拟合 SVI,
      恢复平滑曲面, 并用解析公式构造风险中性密度, 对比『好参数』与『坏参数(蝶式套利)』。

图片:
  svi_market_smiles.png —— 合成的 4 条不同到期『市场』IV 微笑(资金费率 vs IV)
  svi_fit.png           —— SVI 逐月拟合: 市场点 vs 拟合曲线, 看误差
  svi_surface.png       —— 拟合得到的 IV 曲面(资金费率 × 到期)
  svi_density.png       —— 同到期下: 好参数密度(正) vs 坏参数密度(出现负值=蝶式套利)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import least_squares

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "volatility-smile-svi")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "acc": "#DD8452", "mem": "#4C72B0", "stat": "#8172B3", "gold": "#CCB974"}

RES = {}

# ---------------------------------------------------------------------
# SVI 原始参数化(对总方差 w = (σ_imp·√T)^2)
# ---------------------------------------------------------------------
def svi_w(k, a, b, rho, m, sig):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))

def svi_w_derivs(k, a, b, rho, m, sig):
    d = np.sqrt((k - m) ** 2 + sig ** 2)
    w = a + b * (rho * (k - m) + d)
    wp = b * (rho + (k - m) / d)
    wpp = b * (sig ** 2) / (d ** 3)
    return w, wp, wpp

def risk_neutral_density(k, a, b, rho, m, sig):
    """Gatheral 风险中性密度(以 k=ln(K/F) 为变量, 未归一化但形状正确)"""
    w, wp, wpp = svi_w_derivs(k, a, b, rho, m, sig)
    sq = np.sqrt(w)
    phi = np.exp(-k ** 2 / (2 * w)) / np.sqrt(2 * np.pi * w)
    g = 1.0 - (k / (2 * w)) * wp + (1.0 / 4.0) * (1.0 / w - 1.0 / (4 * w ** 2)) * wp ** 2 + 0.5 * wpp
    return phi * g

# ---------------------------------------------------------------------
# 合成『市场』IV 微笑: smirk 形状
#   σ_imp(k,T) = σ0(T) + skew(T)*k + curv(T)*k^2   (轻噪声)
# ---------------------------------------------------------------------
rng = np.random.default_rng(20260712)
maturities = np.array([0.1, 0.25, 0.5, 1.0])      # 年
ks = np.linspace(-0.4, 0.4, 41)                  # log-moneyness 网格

def market_iv(k, T):
    base = 0.20 + 0.05 * np.sqrt(T)              # 期限结构: 越远越高
    sk = -0.25 - 0.10 * np.sqrt(T)               # 负偏: 越虚值看跌 IV 越高(smirk)
    cv = 0.60 + 0.30 * np.sqrt(T)                # 凸性
    return base + sk * k + cv * k ** 2

market = {}
for T in maturities:
    iv = market_iv(ks, T) + rng.normal(0, 0.002, size=ks.shape)  # 轻噪声
    market[T] = np.clip(iv, 0.05, 2.0)

# 图 1: 市场微笑
fig, ax = plt.subplots(figsize=(9.2, 4.8))
for T in maturities:
    ax.plot(ks, market[T], lw=1.8, marker="o", ms=3,
            label="T=%.2f 年" % T)
ax.set_xlabel("log-资金费率 k = ln(K/F)")
ax.set_ylabel("隐含波动率 IV")
ax.set_title("合成的『市场』IV 微笑: 股票式 smirk(左偏 + 凸)")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "svi_market_smiles.png"), dpi=130); plt.close()

# ---------------------------------------------------------------------
# 逐月最小二乘拟合 SVI(约束: b>0, |ρ|<1, σ>0)
# ---------------------------------------------------------------------
def fit_svi(k, w_market):
    """带约束的最小二乘拟合: 限制 |ρ|<0.95, b≤3, σ∈[0.05,1.2, 并优先选无套利(总方差>0 且密度>0)的解."""
    def resid(p):
        a, b, rho, m, sig = p
        return svi_w(k, a, b, rho, m, sig) - w_market
    def arbitrage_free(p):
        a, b, rho, m, sig = p
        d = np.sqrt((k - m) ** 2 + sig ** 2)
        w = a + b * (rho * (k - m) + d)
        if np.any(w <= 0):
            return False
        wp = b * (rho + (k - m) / d)
        wpp = b * sig ** 2 / d ** 3
        g = 1.0 - (k / (2 * w)) * wp + (1.0 / 4.0) * (1.0 / w - 1.0 / (4 * w ** 2)) * wp ** 2 + 0.5 * wpp
        return bool(np.all(g > 0))
    inits = [
        [w_market.mean() * 0.9, 0.5, -0.5, 0.0, 0.3],
        [w_market.mean() * 0.8, 1.0, -0.7, -0.3, 0.5],
        [w_market.mean() * 1.0, 0.3, -0.6, -0.2, 0.4],
        [w_market.mean() * 0.7, 1.5, -0.8, -0.5, 0.6],
    ]
    lb = [-np.inf, 0.05, -0.95, -np.inf, 0.05]
    ub = [np.inf, 3.0, 0.95, np.inf, 1.2]
    best = None
    for p0 in inits:
        try:
            sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=20000)
        except Exception:
            continue
        r = float(np.sum(sol.fun ** 2))
        af = arbitrage_free(sol.x)
        # 优先无套利; 同档下取残差最小
        key = (0 if af else 1, r)
        if best is None or key < best[0]:
            best = (key, sol.x, af)
    return best[1]

fits = {}
for T in maturities:
    iv = market[T]
    w_mkt = (iv ** 2) * T
    a, b, rho, m, sig = fit_svi(ks, w_mkt)
    fits[T] = (a, b, rho, m, sig)
    # 还原 IV 看拟合误差
    w_fit = svi_w(ks, a, b, rho, m, sig)
    iv_fit = np.sqrt(np.maximum(w_fit, 1e-8) / T)

RES["fit_summary"] = {}
print("=== SVI 逐月拟合 ===")
for T in maturities:
    a, b, rho, m, sig = fits[T]
    iv = market[T]; w_mkt = (iv ** 2) * T
    w_fit = svi_w(ks, a, b, rho, m, sig)
    iv_fit = np.sqrt(np.maximum(w_fit, 1e-8) / T)
    rmse = np.sqrt(np.mean((iv_fit - iv) ** 2))
    RES["fit_summary"]["T=%.2f" % T] = dict(a=round(a,4), b=round(b,4),
        rho=round(rho,3), m=round(m,3), sig=round(sig,3), iv_rmse=round(float(rmse),5))
    print("T=%.2f  a=%.4f b=%.4f ρ=%.3f m=%.3f σ=%.3f  IV-RMSE=%.5f"
          % (T, a, b, rho, m, sig, rmse))

# 图 2: 拟合对比(逐月)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
for ax, T in zip(axes.ravel(), maturities):
    iv = market[T]; w_mkt = (iv ** 2) * T
    a, b, rho, m, sig = fits[T]
    w_fit = svi_w(ks, a, b, rho, m, sig)
    iv_fit = np.sqrt(np.maximum(w_fit, 1e-8) / T)
    ax.plot(ks, iv, "o", ms=3, color=C["dn"], label="市场 IV")
    ax.plot(ks, iv_fit, color=C["eq"], lw=2.0, label="SVI 拟合")
    ax.set_title("T=%.2f 年" % T)
    ax.set_xlabel("k = ln(K/F)"); ax.set_ylabel("IV")
    ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
fig.suptitle("SVI 逐月拟合: 一条 5 参数曲线吃下整条微笑", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(D, "svi_fit.png"), dpi=130); plt.close()

# 图 3: 拟合得到的 IV 曲面
KK, TT = np.meshgrid(ks, maturities)
IV_surf = np.zeros_like(KK)
for i, T in enumerate(maturities):
    a, b, rho, m, sig = fits[T]
    w_fit = svi_w(ks, a, b, rho, m, sig)
    IV_surf[i] = np.sqrt(np.maximum(w_fit, 1e-8) / T)
fig = plt.figure(figsize=(9.5, 5.6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(KK, TT, IV_surf, cmap="viridis", alpha=0.9, edgecolor="none")
ax.set_xlabel("k = ln(K/F)"); ax.set_ylabel("到期 T(年)"); ax.set_zlabel("IV")
ax.set_title("SVI 恢复的隐含波动率曲面(平滑、无噪点)")
plt.tight_layout(); plt.savefig(os.path.join(D, "svi_surface.png"), dpi=130); plt.close()

# ---------------------------------------------------------------------
# 图 4: 风险中性密度 —— 好参数(正) vs 坏参数(出现负值=蝶式套利)
# ---------------------------------------------------------------------
T_show = 0.5
a, b, rho, m, sig = fits[T_show]
k_d = np.linspace(-0.35, 0.35, 400)
# 好参数: 选出的无套利拟合 -> 密度处处为正
# 坏参数: 同到期下一个『误校准』SVI(mis-calibrated), 总方差仍>0 但密度出现负值(蝶式套利)
#   —— 这正是无约束/稀疏报价拟合最容易踩的坑
a_b, b_b, rho_b, m_b, sig_b = -0.30, 0.60, -0.50, -0.60, 0.60
g_good = risk_neutral_density(k_d, a, b, rho, m, sig)
g_bad = risk_neutral_density(k_d, a_b, b_b, rho_b, m_b, sig_b)
RES["density"] = dict(T_show=T_show,
                      good=dict(a=round(a,3), b=round(b,3), rho=round(rho,3),
                                m=round(m,3), sig=round(sig,3),
                                min_good=round(float(g_good.min()),5)),
                      bad=dict(a=a_b, b=b_b, rho=rho_b, m=m_b, sig=sig_b,
                               min_bad=round(float(g_bad.min()),5),
                               neg_frac_bad=round(float((g_bad < 0).mean()),3)))
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.plot(k_d, g_good, color=C["up"], lw=2.2,
        label="无套利校准 SVI: 密度处处为正 (min=%.3f)" % g_good.min())
ax.plot(k_d, g_bad, color=C["dn"], lw=2.2,
        label="误校准 SVI: 总方差>0 但密度跌为负 (蝶式套利)")
ax.axhline(0, color="#444", ls="--", lw=1.0)
ax.fill_between(k_d, g_bad, 0, where=g_bad < 0, color=C["dn"], alpha=0.25)
ax.set_xlabel("k = ln(K/F)"); ax.set_ylabel("风险中性密度 f_K(K)")
ax.set_title("同一到期: 误校准 SVI 给出负密度 —— 无套利被打破的红旗")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "svi_density.png"), dpi=130); plt.close()

print("\n=== 关键数字 ===")
print(json.dumps(RES, ensure_ascii=False, indent=2))
print("\n图片已保存到:", D)
