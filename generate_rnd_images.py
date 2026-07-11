#!/usr/bin/env python3
"""
为文章「风险中性密度提取：从期权价格反推市场隐含的回报分布」(rnd-option-implied-density)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  给定一个「真实」风险中性密度（两对数正态混合 → 左偏、左尾厚，对应股指崩盘恐惧），
  用数值积分定价一组看涨期权 C(K)；再用 Breeden-Litzenberger：
    q(K) = e^{rT} * d^2 C / dK^2
  通过「期权价格 → 隐含波动率微笑 → 平滑 C(K) → 在行权价网格上二阶有限差分」反推密度，
  与真实密度对比，展示隐含分布如何把左偏/厚尾如实还原。
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "rnd-option-implied-density")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "pos": "#2F4B7C", "neg": "#C44E52", "ls": "#55A868",
     "mk": "#8172B3", "gold": "#E1A100"}

S0 = 100.0
r = 0.02

# 宽价网格（覆盖深度 ITM/OTM，避免边界截断伪影）
S_grid = np.linspace(10.0, 250.0, 1800)
# 中央行权价网格（定价精确、无边界失真）
K_grid = np.linspace(70.0, 130.0, 41)

# ------------------------------------------------------------
# 数学工具
# ------------------------------------------------------------
def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, r, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)

# ------------------------------------------------------------
# 真实风险中性密度（两对数正态混合；用 T 缩放波动，保持左偏比例）
# ------------------------------------------------------------
def true_rnd(S, T):
    F = S0 * math.exp(r * T)
    w1, m1, s1 = 0.90, math.log(S0) + (r - 0.5 * 0.10 ** 2) * T, 0.10 * math.sqrt(T)
    w2, m2, s2 = 0.10, math.log(S0) + (r - 0.5 * 0.22 ** 2) * T - 0.10 * math.sqrt(T), 0.22 * math.sqrt(T)
    f1 = w1 * np.exp(-0.5 * ((np.log(S) - m1) / s1) ** 2) / (S * s1 * math.sqrt(2 * math.pi))
    f2 = w2 * np.exp(-0.5 * ((np.log(S) - m2) / s2) ** 2) / (S * s2 * math.sqrt(2 * math.pi))
    f = f1 + f2
    meanS = np.trapezoid(f * S, S)
    return f * (F / meanS)            # 无套利归一：E^Q[S_T] = 远期

# ------------------------------------------------------------
# 由真实密度数值积分定价看涨期权 C(K)
# ------------------------------------------------------------
def price_calls(f, S, K):
    Cm = np.array([np.trapezoid(f * np.maximum(S - kk, 0.0), S) for kk in K])
    return Cm * math.exp(-r * T_global)

# ------------------------------------------------------------
# 由期权价格反推隐含波动率（BS 反解，二分法）
# ------------------------------------------------------------
def implied_vol_call(S, K, r, T, Cm, lo=1e-4, hi=3.0):
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if Cm <= intrinsic:
        return lo
    a, b = lo, hi
    for _ in range(80):
        mid = 0.5 * (a + b)
        if bs_call(S, K, r, T, mid) > Cm:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)

# ------------------------------------------------------------
# Breeden-Litzenberger：平滑 IV → 平滑 C(K) → 在行权价网格上二阶有限差分
# ------------------------------------------------------------
T_global = 0.25

def recover_density(K, Cmkt):
    # 1) 反推 IV 微笑
    iv = np.array([implied_vol_call(S0, kk, r, T_global, cc) for kk, cc in zip(K, Cmkt)])
    # 2) 用三次多项式在归一化行权价上平滑 IV（避免高频抖动）
    Kn = (K - K.mean()) / K.std()
    coef = np.polyfit(Kn, iv, 3)
    iv_smooth = np.clip(np.polyval(coef, Kn), 1e-3, 3.0)
    # 3) 用平滑 IV 重估 C(K)
    Csmooth = np.array([bs_call(S0, kk, r, T_global, iv_) for kk, iv_ in zip(K, iv_smooth)])
    # 4) 在行权价网格上二阶有限差分（dK 为网格步长）
    dK = K[1] - K[0]
    d2 = (Csmooth[2:] - 2 * Csmooth[1:-1] + Csmooth[:-2]) / dK ** 2
    Kmid = K[1:-1]
    q = math.exp(r * T_global) * d2          # 风险中性密度
    return Kmid, q, iv, Csmooth, iv_smooth

# ============================================================
# 图一：期权价格曲线 C(K)
# ============================================================
def fig_option_prices(K, Cmkt):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(K, Cmkt, color=C["pos"], lw=2.4, marker="o", ms=3)
    ax.set_title("Call option prices C(K) vs strike (T = 3 months)", fontsize=12)
    ax.set_xlabel("strike K")
    ax.set_ylabel("call price C(K)")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "rnd_option_prices.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

# ============================================================
# 图二：隐含波动率微笑（左偏 smirk）
# ============================================================
def fig_iv_smile(K, iv):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(K, iv * 100, color=C["mk"], lw=2.4, marker="o", ms=3)
    ax.set_title("Implied volatility skew (smirk): IV rises as K falls", fontsize=12)
    ax.set_xlabel("strike K")
    ax.set_ylabel("implied volatility (%)")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "rnd_iv_smile.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

# ============================================================
# 图三：反推 RND vs 真实 RND
# ============================================================
def fig_recovered(K, Cmkt, T):
    Kmid, q, iv, Csmooth, iv_smooth = recover_density(K, Cmkt)
    f_true = true_rnd(Kmid, T)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(Kmid, f_true, color=C["pos"], lw=2.4, label="true RND")
    ax.plot(Kmid, q, color=C["neg"], lw=2.2, ls="--", label="recovered (Breeden-Litzenberger)")
    ax.set_title("Recovered risk-neutral density vs true", fontsize=12)
    ax.set_xlabel("terminal stock price S_T  (=$K)")
    ax.set_ylabel("risk-neutral density q(S_T)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "rnd_recovered.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, q, f_true

# ============================================================
# 图四：不同期限的 RND（近月更左偏，远月趋对称）
# ============================================================
def fig_term_structure():
    global T_global
    out = {}
    for TT, col, lab in [(0.08, C["neg"], "T=1M"), (0.50, C["pos"], "T=6M")]:
        T_global = TT
        f = true_rnd(S_grid, TT)
        Cm = price_calls(f, S_grid, K_grid)
        Kmid, q, _, _, _ = recover_density(K_grid, Cm)
        out[lab] = (Kmid, q, col)
    T_global = 0.25
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for lab, (Kmid, q, col) in out.items():
        ax.plot(Kmid, q, color=col, lw=2.2, label=lab)
    ax.set_title("RND by maturity: near-term skew vs longer-term symmetry", fontsize=12)
    ax.set_xlabel("terminal stock price S_T  (=$K)")
    ax.set_ylabel("risk-neutral density q(S_T)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "rnd_term_structure.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

if __name__ == "__main__":
    T = 0.25
    T_global = T
    f_true_full = true_rnd(S_grid, T)
    C_mkt = price_calls(f_true_full, S_grid, K_grid)
    iv = np.array([implied_vol_call(S0, kk, r, T, cc) for kk, cc in zip(K_grid, C_mkt)])
    p1 = fig_option_prices(K_grid, C_mkt)
    p2 = fig_iv_smile(K_grid, iv)
    p3, q, f_true_mid = fig_recovered(K_grid, C_mkt, T)
    p4 = fig_term_structure()
    Kmid = K_grid[1:-1]
    int_true = np.trapezoid(f_true_mid, Kmid)
    int_rec = np.trapezoid(q, Kmid)
    print("saved:", p1); print("saved:", p2); print("saved:", p3); print("saved:", p4)
    print(f"true RND integrates to {int_true:.4f}; recovered integrates to {int_rec:.4f}")
    print(f"IV range %: [{iv.min()*100:.2f}, {iv.max()*100:.2f}]  skew = {(iv[0]-iv[-1])*100:.2f} vol pts")
    print(f"F = {S0*math.exp(r*T):.2f}; true RND mean S_T = {np.trapezoid(f_true_full*S_grid, S_grid):.2f}")
