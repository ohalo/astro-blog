#!/usr/bin/env python3
"""Generate figures for Kyle continuous auction article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/kyle-continuous-auction"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ---- Model parameters ----
T = 1.0
N = 1000
dt = T / N
t = np.linspace(0, T, N + 1)
Sigma0 = 1.0          # prior variance of v
sigma_u = 1.0         # noise trader vol
p0 = 0.0              # prior mean

lam = np.sqrt(Sigma0) / sigma_u   # constant lambda in continuous-time Kyle equilibrium
# beta_t = sigma_u^2 / (Sigma_t) adjusted; use beta(t) = sigma_u / sqrt(Sigma0) * 1/(T-t)
# Standard result: beta(t) = sigma_u / (sqrt(Sigma0) * (T - t)) ... trading rate x_dot = beta(t)(v - p_t)

def simulate_path(v, seed):
    r = np.random.default_rng(seed)
    p = np.zeros(N + 1); p[0] = p0
    X = np.zeros(N + 1)   # informed cumulative position
    profit = np.zeros(N + 1)
    for i in range(N):
        remaining = T - t[i]
        beta = sigma_u / (np.sqrt(Sigma0) * max(remaining, dt))
        dx = beta * (v - p[i]) * dt
        du = sigma_u * np.sqrt(dt) * r.standard_normal()
        dy = dx + du
        p[i + 1] = p[i] + lam * dy
        X[i + 1] = X[i] + dx
        profit[i + 1] = profit[i] + dx * (v - p[i + 1])
    return p, X, profit

# Fig 1: price discovery paths
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
v_true = 1.2
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for k in range(5):
    p, X, _ = simulate_path(v_true, 100 + k)
    ax[0].plot(t, p, lw=1.1, color=colors[k], alpha=0.85)
    ax[1].plot(t, X, lw=1.1, color=colors[k], alpha=0.85)
ax[0].axhline(v_true, color="crimson", ls="--", lw=1.4, label="真实价值 v = 1.2")
ax[0].set_title("价格向真实价值收敛（5 条模拟路径）")
ax[0].set_xlabel("时间 t"); ax[0].set_ylabel("价格 p(t)")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].set_title("知情交易者累计持仓 X(t)")
ax[1].set_xlabel("时间 t"); ax[1].set_ylabel("累计持仓")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kyle-price-discovery.png", dpi=130)
plt.close(fig)

# Fig 2: posterior variance decay + trading rate explosion
Sigma_t = Sigma0 * (1 - t / T)
beta_t = sigma_u / (np.sqrt(Sigma0) * np.maximum(T - t, dt))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t, Sigma_t, color="steelblue", lw=2)
ax[0].set_title("剩余信息量 Σ(t) 线性衰减")
ax[0].set_xlabel("时间 t"); ax[0].set_ylabel("后验方差 Σ(t)")
ax[0].grid(alpha=0.3)
ax[1].plot(t[:-5], beta_t[:-5], color="darkorange", lw=2)
ax[1].set_yscale("log")
ax[1].set_title("交易强度 β(t) 在收盘前发散（对数轴）")
ax[1].set_xlabel("时间 t"); ax[1].set_ylabel("β(t)")
ax[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/kyle-variance-beta.png", dpi=130)
plt.close(fig)

# Fig 3: lambda vs sigma_u and vs Sigma0
sig_u_grid = np.linspace(0.3, 3.0, 60)
lam_u = np.sqrt(Sigma0) / sig_u_grid
Sig0_grid = np.linspace(0.2, 4.0, 60)
lam_s = np.sqrt(Sig0_grid) / sigma_u
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(sig_u_grid, lam_u, color="seagreen", lw=2)
ax[0].set_title("噪声越多，市场越深：λ 随 σ_u 下降")
ax[0].set_xlabel("噪声交易强度 σ_u"); ax[0].set_ylabel("价格冲击系数 λ")
ax[0].grid(alpha=0.3)
ax[1].plot(Sig0_grid, lam_s, color="purple", lw=2)
ax[1].set_title("信息优势越大，冲击越陡：λ 随 Σ₀ 上升")
ax[1].set_xlabel("先验不确定性 Σ₀"); ax[1].set_ylabel("价格冲击系数 λ")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kyle-lambda-comparative.png", dpi=130)
plt.close(fig)

# Fig 4: profit distribution Monte Carlo
profits = []
for k in range(400):
    r = np.random.default_rng(2000 + k)
    v = np.sqrt(Sigma0) * r.standard_normal()
    p, X, prof = simulate_path(v, 5000 + k)
    profits.append(prof[-1])
profits = np.array(profits)
theory = sigma_u * np.sqrt(Sigma0)  # continuous Kyle: E[profit] = sigma_u * sqrt(Sigma0)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.hist(profits, bins=40, color="steelblue", alpha=0.75, edgecolor="white")
ax.axvline(profits.mean(), color="crimson", ls="--", lw=1.6,
           label=f"模拟均值 = {profits.mean():.3f}")
ax.axvline(theory, color="darkorange", ls=":", lw=1.8,
           label=f"理论期望 σ_u·√Σ₀ = {theory:.3f}")
ax.set_title("知情交易者期末利润分布（400 次蒙特卡洛）")
ax.set_xlabel("期末利润"); ax.set_ylabel("频数")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kyle-profit-dist.png", dpi=130)
plt.close(fig)

print("kyle figs done", profits.mean(), theory)
