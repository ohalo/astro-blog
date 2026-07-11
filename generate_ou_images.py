#!/usr/bin/env python3
"""
为文章「Ornstein-Uhlenbeck 过程与均值回归半衰期：给均值复归定个时标」(ou-mean-reversion-halflife)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由 OU 离散化自洽合成，仅用于演示方法；真实落地见文末路径）：
  dx_t = κ(θ−x_t)dt + σ dW_t ；离散化 x_t = θ + b(x_{t-1}−θ) + σ√Δt·ε，b=e^{−κΔt}；
  半衰期 half-life = ln2 / κ = −Δt·ln2 / ln(b) —— 这正是用 AR(1) 斜率反推均值回归速度的方法。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "ou-mean-reversion-halflife")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "ou": "#2F4B7C", "rw": "#C44E52", "fit": "#55A868", "mk": "#8172B3"}

def simulate_ou(kappa, theta=0.0, sigma=0.10, dt=1.0, T=300, seed=None):
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa * dt)
    x = np.zeros(T)
    x[0] = theta + sigma * rng.normal()
    for t in range(1, T):
        x[t] = theta + b * (x[t - 1] - theta) + sigma * np.sqrt(dt) * rng.normal()
    return x, b

def simulate_rw(sigma=0.10, dt=1.0, T=300, seed=None):
    rng = np.random.default_rng(seed)
    return np.cumsum(sigma * np.sqrt(dt) * rng.normal(size=T))

def ar1_slope(x):
    """用 OLS 估计 AR(1) 斜率 b：x_t = a + b·x_{t-1} + e"""
    y = x[1:]
    X = np.column_stack([np.ones_like(x[:-1]), x[:-1]])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta
    resid = y - X @ beta
    # 标准误用于报告
    dof = len(y) - 2
    seb = np.sqrt(resid @ resid / dof / (np.sum((x[:-1] - x[:-1].mean()) ** 2)))
    return a, b, seb

# ============================================================
# 1) 图一：不同 κ 的 OU 路径 vs 随机游走
# ============================================================
def fig_paths():
    T = 300
    rng_seed = 11
    x_slow, _ = simulate_ou(0.02, T=T, seed=rng_seed)
    x_med, _ = simulate_ou(0.08, T=T, seed=rng_seed + 1)
    x_fast, _ = simulate_ou(0.30, T=T, seed=rng_seed + 2)
    x_rw = simulate_rw(T=T, seed=rng_seed + 3)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    tt = np.arange(T)
    ax.plot(tt, x_slow, color=C["ou"], lw=1.4, alpha=0.85, label=r"OU  $\kappa=0.02$ (slow)")
    ax.plot(tt, x_med, color="#3C8C6E", lw=1.4, alpha=0.9, label=r"OU  $\kappa=0.08$ (med)")
    ax.plot(tt, x_fast, color="#1F6F8C", lw=1.4, alpha=0.95, label=r"OU  $\kappa=0.30$ (fast)")
    ax.plot(tt, x_rw, color=C["rw"], lw=1.3, alpha=0.8, label="random walk")
    ax.axhline(0.0, color="#555555", lw=1, ls="--")
    ax.set_title("Ornstein-Uhlenbeck paths revert to mean; random walk does not", fontsize=11.5)
    ax.set_xlabel("time step t")
    ax.set_ylabel("x_t")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    p = os.path.join(D, "ou_paths.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

# ============================================================
# 2) 图二：AR(1) 拟合散点 x_{t-1} vs x_t，反推 κ 与半衰期
# ============================================================
def fig_ar1_fit():
    x, b_true = simulate_ou(0.15, T=600, seed=21)
    a, b, seb = ar1_slope(x)
    kappa = -np.log(b)            # dt = 1
    half = np.log(2) / kappa
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x[:-1], x[1:], s=8, color=C["mk"], alpha=0.35, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, a + b * xs, color=C["fit"], lw=2.4,
            label=f"OLS fit:  $x_t = {a:.3f} + {b:.3f}\\, x_{{t-1}}$")
    # 45° 参考线（随机游走时斜率 = 1）
    ax.plot([x.min(), x.max()], [x.min(), x.max()], color="#999999", lw=1, ls=":", label="45° (random walk, b=1)")
    ax.set_title("AR(1) regression recovers mean-reversion speed", fontsize=11.5)
    ax.set_xlabel(r"$x_{t-1}$")
    ax.set_ylabel(r"$x_t$")
    txt = (f"b = {b:.3f} ± {seb:.3f}\n"
            f"κ = −ln(b) = {kappa:.3f}\n"
            f"half-life = ln(2)/κ = {half:.1f} steps")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec=C["fit"], alpha=0.9))
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "ou_ar1_fit.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, kappa, half

# ============================================================
# 3) 图三：半衰期随 κ 变化曲线 + 各 κ 标注
# ============================================================
def fig_halflife_curve():
    ks = np.linspace(0.01, 0.6, 200)
    hl = np.log(2) / ks
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, hl, color=C["ou"], lw=2.4)
    for kk, lab in [(0.02, "κ=0.02"), (0.08, "κ=0.08"), (0.30, "κ=0.30")]:
        h = np.log(2) / kk
        ax.scatter([kk], [h], color=C["fit"], zorder=5, s=40)
        ax.annotate(f"{lab}\nHL={h:.1f}", xy=(kk, h), xytext=(kk + 0.02, h + 1.5),
                   fontsize=8.5, color=C["fit"])
    ax.set_title("Half-life = ln(2)/κ  shrinks as reversion speeds up", fontsize=11.5)
    ax.set_xlabel(r"mean-reversion speed $\kappa$")
    ax.set_ylabel("half-life (steps)")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "ou_halflife_curve.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

if __name__ == "__main__":
    p1 = fig_paths()
    p2, kappa, half = fig_ar1_fit()
    p3 = fig_halflife_curve()
    print("saved:", p1)
    print("saved:", p2)
    print("saved:", p3)
    print(f"estimated kappa={kappa:.3f}  half-life={half:.2f} steps")
