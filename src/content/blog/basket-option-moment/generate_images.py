#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「篮子期权与矩匹配：用相关结构给一篮子资产定价」
(basket-option-moment) 生成真实配图与真实统计数字。

核心模型：篮子（算术平均/加权和）期权定价。
  - 算术篮子 B(T) = Σ_i w_i S_i(T)，单个资产 S_i 为风险中性 GBM，
    彼此相关（Corr(W_i, W_j)=ρ_ij）。B 本身不是对数正态——没有闭式。
  - 矩匹配（Levy 1992 / Milevsky-Posner 1998）：解析算出 E[B]、E[B²]，
    把 B 的分布用"前两矩"配成对数正态 ln B ~ N(m, v)，再用 BS 型公式定价。
  - 对照：把篮子当"单一对数正态资产 + 组合方差"（朴素法）会系统性低估
    ——因为它忽略了"算术和"比"几何均值"更高的凸性，且错估相关结构的作用。

所有图表与数字均由文中逻辑真实计算生成：
  1) basket_corr_weights.png   —— 4 资产相关系数矩阵热图 + 权重
  2) basket_mc_paths.png       —— 单条典型 MC 篮子路径（含成分贡献）
  3) basket_terminal_dist.png  —— 终端篮子分布：MC 直方图 vs 矩匹配对数正态
  4) basket_price_compare.png  —— 三法对比（MC真值 / 矩匹配 / 朴素）跨行权价
  5) basket_corr_sensitivity.png —— 篮子价 vs 共同相关系数 ρ（凸性 + 相关效应）

参数：S0=[100,80,120,60], σ=[0.25,0.30,0.20,0.35], w=[0.30,0.25,0.25,0.20],
      r=3%, T=1y, ρ 基准=0.4（equicorrelation），M=200000 条 MC。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "basket-option-moment")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E6E6E6",
     "ink": "#2b2b2b", "acc": "#E8C04B", "pur": "#8172B3", "cya": "#4C72B0"}
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 参数
# =====================================================================
S0 = np.array([100.0, 80.0, 120.0, 60.0])
SIG = np.array([0.25, 0.30, 0.20, 0.35])
W = np.array([0.30, 0.25, 0.25, 0.20])
r, T = 0.03, 1.0
M = 200_000

def corr_matrix(rho):
    d = len(S0)
    R = np.full((d, d), rho)
    np.fill_diagonal(R, 1.0)
    return R

# =====================================================================
# 解析：E[B]、E[B²]、矩匹配对数正态
# =====================================================================
def basket_moments(rho):
    """解析算 E[B]、E[B²]（风险中性、资产为相关 GBM）。"""
    d = len(S0)
    R = corr_matrix(rho)
    # E[S_i(T)] = S0_i * exp(rT)
    ES = S0 * np.exp(r * T)
    # E[S_i S_j] = S0_i S0_j * exp(2 rT + ρ_ij σ_i σ_j T)
    ES2 = np.outer(S0, S0) * np.exp(2 * r * T + R * np.outer(SIG, SIG) * T)
    EB = W @ ES
    EB2 = W @ ES2 @ W
    return EB, EB2

def moment_match_lognormal(EB, EB2):
    """把 B 配成对数正态 ln B ~ N(m, v)，匹配前两矩。"""
    m = 2.0 * np.log(EB) - 0.5 * np.log(EB2)
    v = np.log(EB2) - 2.0 * np.log(EB)
    v = max(v, 1e-12)
    return m, v

def bs_lognormal_call(m, v, K):
    """对数正态 X~LN(m,v) 的看涨期望 E[max(X-K,0)]。"""
    sq = np.sqrt(v)
    F = np.exp(m + 0.5 * v)            # E[X]
    d1 = (m + v - np.log(K)) / sq if sq > 0 else np.inf
    d2 = d1 - sq
    if np.isinf(d1):
        return max(F - K, 0.0)
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def naive_basket_call(rho, K):
    """朴素基准 = 几何篮子（教科书下界）：把篮子当作单一对数正态资产，
    用*几何均值*做 forward、用组合方差 σ_b²=w'Σw 定价。
    AM ≥ GM 逐点成立 => 几何篮子看涨价 ≤ 算术篮子真值（低估，作为下界）。"""
    d = len(S0)
    R = corr_matrix(rho)
    cov = np.outer(SIG, SIG) * R
    sig_b = np.sqrt(W @ cov @ W)
    Fb = np.exp(np.sum(W * np.log(S0))) * np.exp(r * T)   # 几何均值 forward
    sig_eff = sig_b * np.sqrt(T)
    d1 = (np.log(Fb / K) + 0.5 * sig_eff ** 2) / sig_eff
    d2 = d1 - sig_eff
    return Fb * norm.cdf(d1) - K * norm.cdf(d2)

def basket_call(rho, K):
    """矩匹配法篮子看涨价。"""
    EB, EB2 = basket_moments(rho)
    m, v = moment_match_lognormal(EB, EB2)
    return bs_lognormal_call(m, v, K)

# =====================================================================
# Monte Carlo 真值（算术篮子）
# =====================================================================
def mc_basket_call(rho, K, seed=20240719):
    rng = np.random.default_rng(seed)
    d = len(S0)
    R = corr_matrix(rho)
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((M, d))
    eps = z @ L.T
    ST = S0[None, :] * np.exp((r - 0.5 * SIG ** 2) * T + SIG[None, :] * np.sqrt(T) * eps)
    B = ST @ W
    payoff = np.maximum(B - K, 0.0)
    return np.exp(-r * T) * payoff.mean()

# =====================================================================
# 图 1：相关系数矩阵 + 权重
# =====================================================================
def fig_corr_weights():
    R = corr_matrix(0.4)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(S0))); ax.set_yticks(range(len(S0)))
    ax.set_xticklabels([f"资产{i+1}" for i in range(len(S0))])
    ax.set_yticklabels([f"资产{i+1}" for i in range(len(S0))])
    for i in range(len(S0)):
        for j in range(len(S0)):
            ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(R[i, j]) > 0.6 else "black", fontsize=9)
    ax.set_title("4 资产相关系数矩阵（基准 ρ=0.4 等相关）", fontsize=11)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("ρ")
    # 权重文本
    fig.text(0.5, 0.02, "权重 w = [0.30, 0.25, 0.25, 0.20]    "
                        "S0=[100,80,120,60]    σ=[0.25,0.30,0.20,0.35]",
             ha="center", fontsize=9, color=C["ink"])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(D, "basket_corr_weights.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 2：MC 篮子路径
# =====================================================================
def fig_mc_paths():
    rng = np.random.default_rng(7)
    d = len(S0)
    R = corr_matrix(0.4); L = np.linalg.cholesky(R)
    N = 250
    dt = T / N
    steps = rng.standard_normal((N, d)) @ L.T
    # 单条路径
    S = np.zeros((N + 1, d)); S[0] = S0
    for t in range(N):
        S[t + 1] = S[t] * np.exp((r - 0.5 * SIG ** 2) * dt + SIG * np.sqrt(dt) * steps[t])
    B = S @ W
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i in range(d):
        ax.plot(np.linspace(0, T, N + 1), S[:, i], lw=1.0, alpha=0.55,
                label=f"资产{i+1} (σ={SIG[i]:.2f})")
    ax.plot(np.linspace(0, T, N + 1), B, c=C["eq"], lw=2.2, label="篮子 B(t)=Σ w_i S_i")
    ax.set_xlabel("时间 t (年)"); ax.set_ylabel("价格")
    ax.set_title("单条典型 MC 路径：4 成分 + 加权篮子（算术和，非对数正态）", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper left"); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "basket_mc_paths.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 3：终端分布 MC vs 矩匹配对数正态
# =====================================================================
def fig_terminal_dist():
    rng = np.random.default_rng(20240719)
    d = len(S0); R = corr_matrix(0.4); L = np.linalg.cholesky(R)
    z = rng.standard_normal((M, d)); eps = z @ L.T
    ST = S0[None, :] * np.exp((r - 0.5 * SIG ** 2) * T + SIG[None, :] * np.sqrt(T) * eps)
    B = ST @ W
    EB, EB2 = basket_moments(0.4)
    m, v = moment_match_lognormal(EB, EB2)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.hist(B, bins=120, density=True, color=C["eq"], alpha=0.55,
            label="MC 终端篮子分布（真实算术和）")
    xs = np.linspace(B.min(), B.max(), 400)
    pdf = np.exp(-(np.log(xs) - m) ** 2 / (2 * v)) / (xs * np.sqrt(2 * np.pi * v))
    ax.plot(xs, pdf, c=C["dn"], lw=2.2, label="矩匹配对数正态 LN(m,v)")
    ax.axvline(EB, c=C["acc"], ls="--", lw=1.5, label=f"E[B]={EB:.2f}")
    ax.set_xlabel("终端篮子价 B(T)"); ax.set_ylabel("密度")
    ax.set_title("终端篮子分布：矩匹配对数正态贴合 MC 真实算术和（重心一致）", fontsize=11)
    ax.legend(fontsize=9); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "basket_terminal_dist.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 4：三法跨行权价对比
# =====================================================================
def fig_price_compare():
    rho = 0.4
    strikes = np.linspace(70, 130, 13)
    mc = [mc_basket_call(rho, k) for k in strikes]
    mm = [basket_call(rho, k) for k in strikes]
    nv = [naive_basket_call(rho, k) for k in strikes]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(strikes, mc, "o-", c=C["dn"], lw=1.8, ms=5, label="MC 真值（算术篮子）")
    ax.plot(strikes, mm, "s--", c=C["eq"], lw=1.6, ms=4, label="矩匹配对数正态")
    ax.plot(strikes, nv, "^:", c=C["pur"], lw=1.4, ms=4, label="朴素组合方差法")
    ax.set_xlabel("行权价 K"); ax.set_ylabel("看涨期权价格")
    ax.set_title("三法对比（ρ=0.4）：矩匹配贴合 MC，朴素法系统性偏低", fontsize=11)
    ax.legend(fontsize=9); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "basket_price_compare.png"), dpi=130)
    plt.close(fig)

# =====================================================================
# 图 5：价格 vs 相关系数
# =====================================================================
def fig_corr_sensitivity():
    K = 100.0
    rhos = np.linspace(-0.2, 0.95, 20)
    mc = [mc_basket_call(rho, K) for rho in rhos]
    mm = [basket_call(rho, K) for rho in rhos]
    nv = [naive_basket_call(rho, K) for rho in rhos]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(rhos, mc, "o-", c=C["dn"], lw=1.6, ms=4, label="MC 真值")
    ax.plot(rhos, mm, "s--", c=C["eq"], lw=1.5, ms=4, label="矩匹配")
    ax.plot(rhos, nv, "^:", c=C["pur"], lw=1.4, ms=4, label="朴素法")
    ax.set_xlabel("共同相关系数 ρ"); ax.set_ylabel("篮子看涨价 (K=100)")
    ax.set_title("相关结构效应：ρ 越高篮子方差越大、价越高；矩匹配全程紧跟 MC", fontsize=10.5)
    ax.legend(fontsize=9); ax.grid(c=C["grid"], lw=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "basket_corr_sensitivity.png"), dpi=130)
    plt.close(fig)

# =====================================================================
if __name__ == "__main__":
    fig_corr_weights()
    fig_mc_paths()
    fig_terminal_dist()
    fig_price_compare()
    fig_corr_sensitivity()

    # ---- 数值验证 ----
    rho = 0.4
    K = 100.0
    mc = mc_basket_call(rho, K)
    mm = basket_call(rho, K)
    nv = naive_basket_call(rho, K)
    EB, EB2 = basket_moments(rho)
    m, v = moment_match_lognormal(EB, EB2)
    print(f"[base ρ={rho}, K={K}]")
    print(f"  MC 真值(算术) = {mc:.4f}")
    print(f"  矩匹配法      = {mm:.4f}   (vs MC 偏差 {mm-mc:+.4f}, {(mm-mc)/mc*100:+.2f}%)")
    print(f"  朴素组合方差  = {nv:.4f}   (vs MC 偏差 {nv-mc:+.4f}, {(nv-mc)/mc*100:+.2f}%)")
    print(f"  E[B]={EB:.3f}  Var(B)={EB2-EB**2:.3f}  匹配 m={m:.4f} v={v:.4f}")
    # 极端相关对比
    for rr in (-0.2, 0.95):
        print(f"  ρ={rr:+.2f}: MC={mc_basket_call(rr,K):.4f}  矩匹配={basket_call(rr,K):.4f}  "
              f"朴素={naive_basket_call(rr,K):.4f}")
    print("✅ 全部配图生成完毕 ->", D)
