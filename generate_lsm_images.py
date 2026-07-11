#!/usr/bin/env python3
"""
为文章「Longstaff-Schwartz 最小二乘蒙特卡洛：给美式期权一个可落地的定价引擎」(longstaff-schwartz-lsm)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. lsm_paths.png           标的 GBM 模拟路径（美式看跌）
  2. lsm_value_vs_spot.png   某行权日：延续价值 vs 内在价值，标出提前行权区
  3. lsm_boundary.png        提前行权边界 S*(t) 随时间变化
  4. lsm_convergence.png     LSM 价格随模拟路径数收敛（对照欧式价）
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
D = os.path.join(BASE, "longstaff-schwartz-lsm")
os.makedirs(D, exist_ok=True)


# ============================================================
# 全局参数（经典 Longstaff-Schwartz 设定）
# ============================================================
S0, K, r, sigma, T = 36.0, 40.0, 0.06, 0.20, 1.0
M = 50                # 时间步
dt = T / M


def simulate_paths(n_paths, seed=20260711):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, M))
    S = np.zeros((n_paths, M + 1))
    S[:, 0] = S0
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    for i in range(1, M + 1):
        S[:, i] = S[:, i - 1] * np.exp(drift + vol * Z[:, i - 1])
    return S


def basis(x):
    """三次多项式基（用 moneyness x = S/K 提升数值条件）"""
    return np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])


def lsm_price(n_paths, seed=20260711):
    S = simulate_paths(n_paths, seed)
    CF = np.maximum(K - S[:, -1], 0.0) * np.exp(-r * T)   # 到期 payoff，贴现到 t=0
    for t in range(M - 1, 0, -1):
        itm = S[:, t] < K
        if not itm.any():
            continue
        x = S[itm, t] / K
        X = basis(x)
        Y = CF[itm]                       # 已是 t=0 美元计价的延续价值
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        cont = X @ beta                   # 延续价值（t=0 美元）
        exercise_t0 = (K - S[itm, t]) * np.exp(-r * t * dt)
        do_ex = itm.copy()
        do_ex[itm] = exercise_t0 > cont
        CF[do_ex] = (K - S[do_ex, t]) * np.exp(-r * t * dt)
    return CF.mean()


# 欧式看跌 BS 价（下界参考）
def bs_european_put():
    from math import log, sqrt, exp, erf
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    return K * exp(-r * T) * N(-d2) - S0 * N(-d1)


# ============================================================
# 图 1：模拟路径
# ============================================================
S = simulate_paths(40, seed=7)
fig, ax = plt.subplots(figsize=(11, 4.4))
for i in range(40):
    ax.plot(np.arange(M + 1), S[i], lw=0.7, alpha=0.6, color="#3b6ea5")
ax.axhline(K, color="red", ls="--", lw=1, label=f"行权价 K={K:.0f}")
ax.set_xlabel("时间步（共 %d 步，T=1 年）" % M)
ax.set_ylabel("标的价格 S")
ax.set_title("GBM 模拟的标的路径：美式看跌在 S<K 时进入价内")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "lsm_paths.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：某行权日 延续价值 vs 内在价值
# ============================================================
t0 = M - 4
S_big = simulate_paths(30000, seed=20260711)
CF = np.maximum(K - S_big[:, -1], 0.0) * np.exp(-r * T)
exercise_dates = list(range(M - 1, 0, -1))
# 先跑完整 LSM 拿到最终 CF（用于延续价值）
for t in exercise_dates:
    itm = S_big[:, t] < K
    if not itm.any():
        continue
    x = S_big[itm, t] / K
    X = basis(x)
    Y = CF[itm]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    cont = X @ beta
    exercise_t0 = (K - S_big[itm, t]) * np.exp(-r * t * dt)
    do_ex = itm.copy()
    do_ex[itm] = exercise_t0 > cont
    CF[do_ex] = (K - S_big[do_ex, t]) * np.exp(-r * t * dt)

# 在 t0 重新回归，延续价值用 t0 美元计价以便与内在价值同口径比较
itm = S_big[:, t0] < K
x = S_big[itm, t0] / K
X = basis(x)
Y_t0 = CF[itm] * np.exp(r * t0 * dt)        # 还原到 t0 美元
beta, *_ = np.linalg.lstsq(X, Y_t0, rcond=None)
sgrid = np.linspace(0.5 * K, K, 200)
cont_grid = basis(sgrid / K) @ beta
intr_grid = K - sgrid
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sgrid, cont_grid, lw=2, color="#1f77b4", label="延续价值 (t0 美元)")
ax.plot(sgrid, intr_grid, lw=2, color="#d62728", label="内在价值 K−S")
ax.axvline(K, color="gray", ls=":", lw=1)
ax.fill_between(sgrid, intr_grid, cont_grid,
                where=(intr_grid > cont_grid), color="#d62728", alpha=0.15,
                label="提前行权区")
ax.set_xlabel("标的价格 S（t0 时刻）")
ax.set_ylabel("价值（美元）")
ax.set_title(f"第 {t0} 步：内在价值超过延续价值时提前行权（红色区域）")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "lsm_value_vs_spot.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：提前行权边界 S*(t)
# ============================================================
steps = np.arange(1, M)
S_star = np.zeros(M - 1)
for t in steps:
    itm = S_big[:, t] < K
    if not itm.any():
        S_star[t - 1] = np.nan
        continue
    x = S_big[itm, t] / K
    X = basis(x)
    Y_t0 = CF[itm] * np.exp(r * t * dt)
    beta, *_ = np.linalg.lstsq(X, Y_t0, rcond=None)
    sgrid = np.linspace(0.4 * K, K, 400)
    cont = basis(sgrid / K) @ beta
    intr = K - sgrid
    cross = sgrid[intr > cont]
    S_star[t - 1] = cross.max() if len(cross) else np.nan

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(steps, S_star, lw=2, color="#2ca02c", marker=".", ms=3)
ax.axhline(K, color="red", ls="--", lw=1, label="K")
ax.set_xlabel("时间步 t（离到期越近，t 越小）")
ax.set_ylabel("临界股价 S*(t)")
ax.set_title("提前行权边界：越临近到期，越愿意在更高股价行权")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "lsm_boundary.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：收敛性
# ============================================================
eu = bs_european_put()
path_list = [2000, 5000, 10000, 20000, 40000, 80000]
prices = [lsm_price(n) for n in path_list]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(path_list, prices, "o-", color="#1f77b4", label="LSM 美式看跌价")
ax.axhline(eu, color="#d62728", ls="--", lw=1.5, label=f"欧式看跌价 {eu:.3f}（下界）")
ax.set_xlabel("模拟路径数")
ax.set_ylabel("期权价格（美元）")
ax.set_title("LSM 价格随路径数收敛，且始终 ≥ 欧式价（提前行权溢价为正）")
ax.legend(fontsize=9)
ax.set_xscale("log")
plt.tight_layout()
plt.savefig(os.path.join(D, "lsm_convergence.png"), dpi=130)
plt.close()

print("LSM images done:", os.listdir(D))
print(f"American(LSM) @80k paths = {lsm_price(80000):.4f}, European = {eu:.4f}")
