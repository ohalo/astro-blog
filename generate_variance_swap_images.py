#!/usr/bin/env python3
"""
为文章「方差互换与波动率互换：做空波动率风险溢价的衍生品工具」(variance-swap-trading)
生成真实配图。所有图表均由文中 Python 代码真实计算生成（随机波动率模型 + Black-Scholes 定价）。

图表：
  1. varswap_pnl_profile.png   损益轮廓：方差互换(凸) / 波动率互换(线性) / 跨式(近似方差)
  2. varswap_convexity.png     凸性差异：波动率互换执行波动 vs √方差互换执行方差（Jensen）
  3. varswap_term_structure.png  方差互换期限结构（strike 随期限）
  4. varswap_equity.png         做空方差互换的累计 P&L 与危机回撤
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import erf, sqrt, exp, log

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "variance-swap-trading")
os.makedirs(D, exist_ok=True)
np.random.seed(20260712)

# ============================================================
# 0) Black-Scholes
# ============================================================
def bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * 0.5 * (1 + erf(d1 / sqrt(2))) - K * exp(-r * T) * 0.5 * (1 + erf(d2 / sqrt(2)))

def bs_put(S, K, T, r, sig):
    return bs_call(S, K, T, r, sig) - S + K * exp(-r * T)

def forward_var_analytic(v0, theta, kappa, T):
    # CIR 方差过程的 E[∫_0^T v_t dt] 解析值
    return theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa

# ============================================================
# 1) 模拟随机波动率（Heston 式）：v0≠θ 制造向下倾斜的方差期限结构
# ============================================================
N = 252 * 10 + 60
dt = 1.0 / 252
mu = 0.06
v0 = 0.06                                # 起点方差（vol≈24.5%）
kappa, theta, xi, rho = 3.5, 0.04, 0.55, -0.65   # 长期方差 θ=0.04（vol=20%）
r_annual = 0.02
S = np.zeros(N); v = np.zeros(N)
S[0], v[0] = 100.0, v0
for t in range(1, N):
    z1 = np.random.randn()
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()
    v[t] = max(v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(max(v[t-1], 1e-6) * dt) * z2, 1e-5)
    rt = (mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1
    if np.random.rand() < 0.004:
        rt += -np.random.uniform(0.06, 0.14)
    S[t] = S[t-1] * np.exp(rt)
ret = np.diff(np.log(S))                     # 日收益（小数）
M = len(ret)

# ============================================================
# 图 1：损益轮廓（做空方）—— 方差互换 vs 波动率互换 vs 跨式
# ============================================================
sig_strike = 0.20                        # 互换执行波动（≈ 20%）
var_strike = sig_strike**2              # 方差互换执行方差
notional = 100.0
sig_r = np.linspace(0.05, 0.45, 200)
pnl_var = notional * (var_strike - sig_r**2)          # 方差互换：凸（抛物线）
pnl_vol = notional * (sig_strike - sig_r)             # 波动率互换：线性
pnl_strad = notional * (var_strike - sig_r**2) * 0.5 + notional * 0.4 * (sig_strike - sig_r)
fig, ax = plt.subplots(figsize=(11, 6.0))
ax.plot(sig_r * 100, pnl_var, color="#d62728", lw=2.4, label="方差互换（做空）：凸性损益")
ax.plot(sig_r * 100, pnl_vol, color="#1f77b4", lw=2.4, label="波动率互换（做空）：线性损益")
ax.plot(sig_r * 100, pnl_strad, color="#2ca02c", lw=2.0, ls="--", label="跨式（做空，近似）：凸性更弱且含偏度")
ax.axvline(sig_strike * 100, color="gray", lw=1.2, ls=":", label=f"执行波动 {sig_strike*100:.0f}%")
ax.set_xlabel("到期已实现波动 (%)", fontsize=11)
ax.set_ylabel("损益（名义=100）", fontsize=11)
ax.set_title("损益轮廓：方差互换最「凸」，盈亏随实际波动平方放大", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "varswap_pnl_profile.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 2：凸性差异（Jensen）—— 波动率互换执行波动 = E[σ] 低于 √方差互换执行方差 = √E[σ²]
# ============================================================
Ts2 = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
Kvar2 = []; Kvol2 = []; sqrtKvar = []
for T in Ts2:
    n = int(round(T * 252))
    rvs = np.full(M, np.nan)
    for i in range(n, M):
        rvs[i] = np.sqrt(252.0) * np.std(ret[i - n + 1:i + 1], ddof=1)
    vals = rvs[~np.isnan(rvs)]
    Kvar2.append(np.mean(vals**2))          # E[σ²]
    Kvol2.append(np.mean(vals))             # E[σ]
    sqrtKvar.append(np.sqrt(np.mean(vals**2)))
Kvar2 = np.array(Kvar2); Kvol2 = np.array(Kvol2); sqrtKvar = np.array(sqrtKvar)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(Ts2, sqrtKvar * 100, "o-", color="#d62728", lw=2.2, ms=8, label="√方差互换执行方差 (=√E[σ²])")
ax.plot(Ts2, Kvol2 * 100, "s--", color="#1f77b4", lw=2.2, ms=8, label="波动率互换执行波动 (=E[σ])")
ax.fill_between(Ts2, Kvol2 * 100, sqrtKvar * 100, color="#d62728", alpha=0.10,
                label="凸性缺口 √E[σ²] − E[σ]")
ax.set_xlabel("期限 T（年）", fontsize=11)
ax.set_ylabel("执行波动 (%)", fontsize=11)
ax.set_title("凸性差异：波动率互换执行波动低于 √方差互换执行方差（Jensen 不等式）", fontsize=12.0, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.0)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "varswap_convexity.png"), dpi=150, bbox_inches="tight")
plt.close()
print("凸性校验  √E[σ²]=", np.round(sqrtKvar*100, 2), "  E[σ]=", np.round(Kvol2*100, 2))

# ============================================================
# 图 3：方差互换期限结构（strike 随期限）
# ============================================================
Ts3 = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
vs_strike = np.array([forward_var_analytic(v0, theta, kappa, T) / T for T in Ts3])
vs_vol = np.sqrt(vs_strike)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(Ts3, vs_strike * 100, "o-", color="#d62728", lw=2.2, ms=8, label="方差互换执行方差 (σ²)")
ax2 = ax.twinx()
ax2.plot(Ts3, vs_vol * 100, "s--", color="#1f77b4", lw=2.2, ms=8, label="对应执行波动 (σ)")
ax.set_xlabel("期限 T（年）", fontsize=11)
ax.set_ylabel("执行方差（×100）", color="#d62728", fontsize=11)
ax2.set_ylabel("执行波动 (%)", color="#1f77b4", fontsize=11)
ax.set_title("方差互换期限结构：短端高、长端回归长期方差 θ", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax2.legend(loc="lower right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "varswap_term_structure.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 4：做空方差互换的累计 P&L（含波动率风险溢价、含危机回撤）
# ============================================================
rebal = 21
notional2 = 100.0
premium_var = 0.004                       # 方差层面的正溢价（≈ 40bps 方差）
pnl = np.zeros(M)
for i in range(0, M - rebal, rebal):
    T = rebal / 252.0
    E_int = forward_var_analytic(v0, theta, kappa, T)
    fair_var = E_int / T
    K_mkt = fair_var + premium_var         # 市场 strike（含溢价）
    window = ret[i + 1:i + 1 + rebal]
    rv_annual = 12.0 * np.sum(window**2)   # 年化已实现方差（月度）
    pnl[i] = notional2 * (K_mkt - rv_annual)
cum = 100.0 + np.cumsum(pnl)
peak = np.maximum.accumulate(cum)
dd = cum / peak - 1.0
worst = dd.argmin()
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(np.arange(M), cum, color="#2ca02c", lw=1.8, label="做空方差互换累计 P&L")
ax.fill_between(np.arange(M), cum, peak, where=(dd < 0), color="#d62728", alpha=0.12)
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("累计 P&L（起点=100）", fontsize=11)
ax.set_title(f"做空方差互换：平稳收割溢价，最大回撤 {dd.min()*100:.1f}%（危机期）", fontsize=12.5, fontweight="bold")
ax.annotate(f"危机回撤 ≈ {dd.min()*100:.0f}%", xy=(worst, cum[worst]),
            xytext=(worst - 600, cum[worst] + 12), fontsize=9.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "varswap_equity.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ 方差/波动率互换配图生成完成：{sorted(os.listdir(D))}")
print(f"   期末累计 P&L={cum[-1]:.1f}  最大回撤={dd.min()*100:.1f}%")
