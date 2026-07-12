#!/usr/bin/env python3
"""
为文章「阈值 GARCH 与杠杆效应：坏消息为什么比好消息更搅动波动」(threshold-garch-leverage)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

数据机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 用 GJR/TGARCH(1,1) 制造「杠杆效应」：负冲击比同幅度正冲击推高更多方差；
    σ_t² = ω + α·ε_{t-1}² + γ·ε_{t-1}²·I(ε_{t-1}<0) + β·σ_{t-1}²，γ>0；
  - 用高斯 MLE 分别拟合「非对称 GJR」与「对称 GARCH(γ≡0)」，对比似然与残差非对称；
  - 画出「新闻 impact curve」展示非对称 V 形，以及历史路径的聚集 + 危机放大。
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
D = os.path.join(BASE, "threshold-garch-leverage")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "ret": "#2F4B7C", "vol": "#C44E52", "gjr": "#55A868",
     "sym": "#E8A33D", "neg": "#C44E52", "pos": "#55A868", "line": "#333333"}

# ============================================================
# 1) 合成 GJR/TGARCH(1,1) 数据
# ============================================================
def simulate_gjr(T=4000, omega=1e-6, alpha=0.06, gamma=0.09, beta=0.88, seed=407):
    rng = np.random.default_rng(seed)
    eps = np.zeros(T); sig = np.zeros(T)
    sig[0] = np.sqrt(omega / (1 - alpha - 0.5 * gamma - beta))  # 平稳方差初值
    for t in range(1, T):
        z = rng.normal()
        eps[t] = sig[t - 1] * z
        shock = eps[t - 1] ** 2
        ind = 1.0 if eps[t - 1] < 0 else 0.0
        sig[t] = np.sqrt(max(omega + alpha * shock + gamma * shock * ind + beta * sig[t - 1] ** 2, 1e-12))
    return eps, sig

# ============================================================
# 2) MLE 拟合（GJR 与对称 GARCH）
# ============================================================
def garch_filter(eps, omega, alpha, gamma, beta):
    T = len(eps)
    sig2 = np.empty(T)
    sig2[0] = np.mean(eps ** 2)
    for t in range(1, T):
        ind = 1.0 if eps[t - 1] < 0 else 0.0
        sig2[t] = omega + alpha * eps[t - 1] ** 2 + gamma * eps[t - 1] ** 2 * ind + beta * sig2[t - 1]
    return np.maximum(sig2, 1e-12)

def neg_ll(params, eps, fit_gamma=True):
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or (fit_gamma and gamma < 0):
        return 1e10
    sig2 = garch_filter(eps, omega, alpha, gamma if fit_gamma else 0.0, beta)
    ll = -0.5 * (np.log(2 * np.pi) + np.log(sig2) + eps ** 2 / sig2)
    return -ll.sum()

def fit(eps, fit_gamma=True):
    x0 = [1e-6, 0.06, 0.09, 0.88] if fit_gamma else [1e-6, 0.06, 0.0, 0.88]
    bnds = [(1e-9, None), (0, 0.5), (0 if fit_gamma else 0, 0.5 if fit_gamma else 0), (0, 0.99)]
    res = minimize(neg_ll, x0, args=(eps, fit_gamma), bounds=bnds, method="L-BFGS-B")
    return res.x, -res.fun

# ============================================================
# 3) 图一：收益路径 + 条件波动（聚集 + 负冲击放大）
# ============================================================
def fig_paths(eps, sig):
    last = 600
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True)
    a1.plot(np.arange(last), eps[-last:], color=C["ret"], lw=0.8, alpha=0.85)
    a1.axhline(0, color="#777", lw=0.8)
    a1.set_title("GJR-TGARCH returns: volatility clusters and negative shocks bite harder", fontsize=11)
    a1.set_ylabel("return")
    a2.plot(np.arange(last), sig[-last:], color=C["vol"], lw=1.2)
    a2.set_ylabel("conditional vol")
    a2.set_xlabel("day")
    for ax in (a1, a2):
        ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "tg_paths.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 4) 图二：news impact curve（非对称 V 形）
# ============================================================
def fig_news_impact(omega, alpha, gamma, beta):
    e = np.linspace(-0.06, 0.06, 200)
    sig2_gjr = omega + alpha * e ** 2 + gamma * e ** 2 * (e < 0) + beta * 1.0
    sig2_sym = omega + (alpha + 0.5 * gamma) * e ** 2 + beta * 1.0  # 对称等价
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(e, sig2_gjr, color=C["gjr"], lw=2.4, label="GJR / TGARCH (asymmetric)")
    ax.plot(e, sig2_sym, color=C["sym"], lw=2.0, ls="--", label="symmetric GARCH")
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_title("News impact curve: negative shocks raise variance more", fontsize=11.5)
    ax.set_xlabel(r"past shock $\varepsilon_{t-1}$")
    ax.set_ylabel(r"next variance $\sigma_t^2$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "tg_news_impact.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 5) 图三：杠杆散点（过去收益 x 未来方差 y）
# ============================================================
def fig_leverage_scatter(eps):
    # 用 t-1 收益对 t 的平方收益，按符号着色
    x = eps[:-1]; y = eps[1:] ** 2
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pos = x >= 0
    ax.scatter(x[pos], y[pos], s=5, color=C["pos"], alpha=0.25, label="positive past return")
    ax.scatter(x[~pos], y[~pos], s=5, color=C["neg"], alpha=0.25, label="negative past return")
    # 分箱均值
    edges = np.linspace(-0.05, 0.05, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = [y[(x >= edges[i]) & (x < edges[i + 1])].mean() for i in range(len(edges) - 1)]
    ax.plot(centers, means, color=C["line"], lw=2.2, marker="o", ms=3, label="binned mean")
    ax.set_title("Leverage effect: |return next day| rises faster after bad news", fontsize=11)
    ax.set_xlabel(r"past return $\varepsilon_{t-1}$")
    ax.set_ylabel(r"next squared return $\varepsilon_t^2$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "tg_leverage_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 6) 图四：模型对比（似然 + 残差非对称比）
# ============================================================
def fig_fit_compare(gjr_ll, sym_ll, asym_gjr, asym_sym):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 4.0))
    a1.bar(["symmetric\nGARCH", "GJR/TGARCH"], [sym_ll / 1000, gjr_ll / 1000],
           color=[C["sym"], C["gjr"]], alpha=0.9)
    a1.set_title("Log-likelihood (×1000)", fontsize=10.5)
    a1.grid(True, color=C["grid"], lw=0.6, axis="y"); a1.set_axisbelow(True)
    a2.bar(["symmetric\nGARCH", "GJR/TGARCH"], [asym_sym, asym_gjr],
           color=[C["sym"], C["gjr"]], alpha=0.9)
    a2.axhline(1.0, color="#777", lw=0.9, ls=":")
    a2.set_title("Residual asymmetry ratio\n(mean vol after − / after +)", fontsize=10.5)
    a2.grid(True, color=C["grid"], lw=0.6, axis="y"); a2.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "tg_fit_compare.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

if __name__ == "__main__":
    eps, sig = simulate_gjr(T=4000, omega=1e-6, alpha=0.06, gamma=0.09, beta=0.88)
    gjr_p, gjr_ll = fit(eps, fit_gamma=True)
    sym_p, sym_ll = fit(eps, fit_gamma=False)
    print("True GJR:  ω=1e-6 α=0.060 γ=0.090 β=0.880")
    print("Fitted GJR:        ω=%.2e α=%.3f γ=%.3f β=%.3f  LL=%.1f"
          % (gjr_p[0], gjr_p[1], gjr_p[2], gjr_p[3], gjr_ll))
    print("Fitted symmetric:  ω=%.2e α=%.3f γ=%.3f β=%.3f  LL=%.1f"
          % (sym_p[0], sym_p[1], sym_p[2], sym_p[3], sym_ll))
    print("LL gain (GJR−sym)=%.1f" % (gjr_ll - sym_ll))

    # 残差非对称比：用模型过滤出的「条件方差」按过去收益符号分组
    # GJR 对负冲击给更大方差 → 比 >1；对称 GARCH 不看符号 → 比 ≈1
    def model_asym(params):
        s2 = garch_filter(eps, *params)
        neg = s2[1:][eps[:-1] < 0].mean()
        pos = s2[1:][eps[:-1] >= 0].mean()
        return neg / pos
    asym_gjr = model_asym(gjr_p)
    asym_sym = model_asym(sym_p)
    print("asymmetry ratio (GJR fit)=%.3f  (sym fit)=%.3f  (model-implied)" % (asym_gjr, asym_sym))

    p1 = fig_paths(eps, sig)
    p2 = fig_news_impact(gjr_p[0], gjr_p[1], gjr_p[2], gjr_p[3])
    p3 = fig_leverage_scatter(eps)
    p4 = fig_fit_compare(gjr_ll, sym_ll, asym_gjr, asym_sym)
    print("saved:", p1); print("saved:", p2); print("saved:", p3); print("saved:", p4)
