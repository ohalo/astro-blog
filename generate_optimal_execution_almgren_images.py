#!/usr/bin/env python3
"""
为文章「最优执行 Almgren-Chriss：把大单拆成最优执行流」(optimal-execution-almgren)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

与已发布 almgren-chriss-execution(卖出/TWAP/Pareto/蒙特卡洛)的区别:
  本篇聚焦 (a) 执行期限 T 的权衡 (b) 最优"交易速率"形态 (c) VWAP 基准比较 (d) 调度误差代价。

设定(自洽合成, 仅用于演示方法):
  * 需在期限 T 天内卖出 X=1,000,000 股, 分 N=50 步(连续时间近似)
  * 日波动 2%, 但波动率集中在开盘收盘(U 形): 每步噪声 std = daily_vol*S0*shape(k)*sqrt(dt)
  * 永久冲击 g(v)=eta*v, 临时冲击 h(v)=tau*v (与 almgren-chriss-execution 同参数家族)
  * 连续时间最优速率(源自主文推导):
        v*(t) = (X/T) * (cosh(kappa*(T-t))) / sinh(kappa*T),  kappa = sqrt(gamma*sigma^2/(2*tau))
  * 离散化 v*(t_k) -> 库存 x_k = X - Σ_{j<=k} v*_j*dt, 每步卖 n_k = v*_k*dt
  * VWAP 基准: 假设市场成交量 U 形分布, VWAP 策略卖出速率 ∝ 日内成交量份额 -> 其 IS 与 AC 最优比较
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "optimal-execution-almgren")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "rob": "#9467bd", "nom": "#C44E52", "thr": "#888888",
     "green": "#2ca02c", "orange": "#FF7F0E", "blue": "#1f77b4", "purple": "#8c564b"}

# ---------- 参数 ----------
S0 = 100.0
X = 1_000_000.0
daily_vol = 0.02
N = 50
eta = 8e-8
tau = 5e-7
gamma = 2e-7

# U 形日内波动形态(归一化, 中段低、两端高)
def intraday_shape(N):
    k = np.arange(N)
    u = 0.5 - 0.5 * np.cos(2 * np.pi * (k + 0.5) / N)   # U 形, 0~1
    u = 0.4 + 0.6 * u                                    # 避免中段为 0
    return u / u.mean()

SHAPE = intraday_shape(N)

def ac_optimal(T):
    """连续时间最优速率 v*(t); 返回离散库存轨迹 x[0..N] 与卖出 n[1..N]。
    正确公式(Almgren-Chriss 2001):
        x*(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)        库存
        v*(t) = -(dx*/dt) = X*kappa*cosh(kappa*(T-t))/sinh(kappa*T)  卖出速率
        kappa = sqrt(gamma*sigma^2 / (2*tau))
    离散化后严格归一: sum(n)=X, x 单调从 X 降到 0。
    """
    dt = T / N
    sigma = daily_vol * S0 / np.sqrt(T)          # 期内日波动
    kappa = np.sqrt(gamma * sigma ** 2 / (2 * tau)) + 1e-12
    t = np.arange(1, N + 1) * dt
    # 中间库存 x_k = X * sinh(kappa*(T - t_k)) / sinh(kappa*T)
    x_mid = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    x = np.concatenate([[X], x_mid])            # 长度 N+1, 末尾已≈0
    # 卖出 n_k = x_{k-1} - x_k, 严格保证 sum=X 且 x 单调>=0
    n = x[:-1] - x[1:]
    n = np.clip(n, 0, None)
    n = n / n.sum() * X
    x = np.concatenate([[X], X - np.cumsum(n)])
    return x, n, dt, sigma, kappa

def cost_stats(x, dt, sigma):
    n = x[:-1] - x[1:]
    E = 0.5 * eta * X ** 2 + tau * np.sum(n ** 2)
    Var = (sigma ** 2) * dt * np.sum(x[1:] ** 2)
    return E, Var

def vwap_schedule(T):
    """VWAP 策略: 卖出速率 ∝ 日内成交量(U 形), 与 AC 的波动形状对比."""
    dt = T / N
    w = SHAPE / SHAPE.sum()
    n = X * w
    x = np.concatenate([[X], X - np.cumsum(n)])
    return x, n, dt

# ---------- 图 1: 期限 T 的权衡 ----------
Ts = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
res = []
for T in Ts:
    x_a, _, dt_a, sigma_a, _ = ac_optimal(T)
    res.append(cost_stats(x_a, dt_a, sigma_a))
E_T = np.array([float(E) / (X * S0) * 1e4 for E, _ in res])
S_T = np.array([float(np.sqrt(V)) / (X * S0) * 1e4 for _, V in res])
fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
ax1.plot(Ts, E_T, "o-", color=C["nom"], lw=2, label="期望冲击成本 (bps)")
ax1.set_xscale("log"); ax1.set_xlabel("执行期限 T (交易日, 对数)", fontsize=11)
ax1.set_ylabel("期望冲击成本 (bps)", color=C["nom"], fontsize=11)
ax1.tick_params(axis="y", labelcolor=C["nom"]); ax1.grid(True, color=C["grid"])
ax2 = ax1.twinx()
ax2.plot(Ts, S_T, "s-", color=C["rob"], lw=2, label="波动风险 (bps)")
ax2.set_ylabel("波动风险 (bps)", color=C["rob"], fontsize=11)
ax2.tick_params(axis="y", labelcolor=C["rob"])
ax1.set_title("期限 T 的权衡: 拖久→冲击成本↓但波动风险↑", fontsize=12.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "ac_horizon_tradeoff.png"), dpi=150)
plt.close(fig)

# ---------- 图 2: 最优交易速率形态(不同 T) ----------
fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.8))
for T, col in [(0.5, C["orange"]), (2.0, C["eq"]), (8.0, C["fv"]), (32.0, C["purple"])]:
    x, n, dt, _, _ = ac_optimal(T)
    rate = n / dt
    axs[0].plot(np.arange(1, N + 1) / N, rate / 1e3, color=col, lw=1.8,
                label=f"T={T}d")
axs[0].set_xlabel("执行进度 (0→1)", fontsize=11); axs[0].set_ylabel("交易速率 (千股/步)", fontsize=11)
axs[0].set_title("最优交易速率 v*(t): 越怕波动越前置(早卖)", fontsize=11.5)
axs[0].legend(fontsize=9); axs[0].grid(True, color=C["grid"])
for T, col in [(0.5, C["orange"]), (2.0, C["eq"]), (8.0, C["fv"]), (32.0, C["purple"])]:
    x, n, dt, _, _ = ac_optimal(T)
    axs[1].plot(np.arange(N + 1) / N, x / 1e3, color=col, lw=1.8, label=f"T={T}d")
axs[1].set_xlabel("执行进度 (0→1)", fontsize=11); axs[1].set_ylabel("剩余库存 (千股)", fontsize=11)
axs[1].set_title("库存轨迹 x(t): 期限越短越陡(越快清仓)", fontsize=11.5)
axs[1].legend(fontsize=9); axs[1].grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "ac_rate_and_inventory.png"), dpi=150)
plt.close(fig)

# ---------- 图 3: VWAP 基准 vs AC 最优 ----------
x_ac, n_ac, dt_ac, sigma_ac, _ = ac_optimal(2.0)
x_vw, n_vw, dt_vw = vwap_schedule(2.0)
E_ac, V_ac = cost_stats(x_ac, dt_ac, sigma_ac)
E_vw, V_vw = cost_stats(x_vw, dt_vw, daily_vol * S0 / np.sqrt(2.0))
E_ac_b = E_ac / (X * S0) * 1e4
S_ac_b = np.sqrt(V_ac) / (X * S0) * 1e4
E_vw_b = E_vw / (X * S0) * 1e4
S_vw_b = np.sqrt(V_vw) / (X * S0) * 1e4

fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.8))
axs[0].plot(np.arange(1, N + 1) / N, n_ac / dt_ac / 1e3, "-", color=C["eq"], lw=2, label="AC 最优速率")
axs[0].plot(np.arange(1, N + 1) / N, n_vw / dt_vw / 1e3, "--", color=C["orange"], lw=2, label="VWAP 速率(∝成交量)")
axs[0].set_xlabel("执行进度", fontsize=11); axs[0].set_ylabel("交易速率 (千股/步)", fontsize=11)
axs[0].set_title("AC 最优 vs VWAP: 形状不同", fontsize=11.5)
axs[0].legend(fontsize=9); axs[0].grid(True, color=C["grid"])
labels = ["期望冲击", "波动风险"]
ac_v = [E_ac_b / 1e4, S_ac_b / 1e4]
vw_v = [E_vw_b / 1e4, S_vw_b / 1e4]
xp = np.arange(2)
axs[1].bar(xp - 0.2, ac_v, width=0.4, color=C["eq"], label="AC 最优")
axs[1].bar(xp + 0.2, vw_v, width=0.4, color=C["orange"], label="VWAP")
axs[1].set_xticks(xp); axs[1].set_xticklabels(labels, fontsize=10.5)
axs[1].set_ylabel("成本 ($ 万)", fontsize=11)
axs[1].set_title("成本拆解: VWAP 风险更低, AC 冲击更优", fontsize=11.5)
axs[1].legend(fontsize=9); axs[1].grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "ac_vs_vwap.png"), dpi=150)
plt.close(fig)

# ---------- 图 4: 偏离最优的代价(调度误差) ----------
# 用单参数家族 n_k ∝ exp(λ·k) 归一化; λ=0 即匀速, 存在某 λ*=最优前置/后置
# 扫描 λ, 画目标函数, 标出 AC 最优 -> 偏离即代价
lams = np.linspace(-2.0, 2.0, 41)
obj_curve = []
T = 2.0
dt = T / N
sigma = daily_vol * S0 / np.sqrt(T)
for lam in lams:
    w = np.exp(lam * np.arange(N))
    n_f = X * w / w.sum()
    x_f = np.concatenate([[X], X - np.cumsum(n_f)])
    E_f, V_f = cost_stats(x_f, dt, sigma)
    obj_curve.append(E_f + 0.5 * gamma * V_f)
obj_curve = np.array(obj_curve)
# 找到 AC 近似最优 tilt: 用速率 v*(t) 的 log 斜率估计
x_ac, n_ac, dt_ac, sigma_ac, _ = ac_optimal(T)
slope = np.polyfit(np.arange(N), np.log(n_ac + 1e-9), 1)[0]
idx_best = int(np.argmin(np.abs(lams - slope)))

fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.plot(lams, obj_curve / 1e4, "o-", color=C["rob"], lw=2)
ax.axvline(lams[idx_best], color=C["fv"], ls="--", lw=1.5, label=f"AC 最优点(λ≈{slope:.2f})")
ax.set_xlabel("调度倾斜 λ (负=后置, 正=前置)", fontsize=11)
ax.set_ylabel("目标函数 (期望冲击 + γ·风险, 万元)", fontsize=11)
ax.set_title("偏离 AC 最优的代价: 调度偏了, 目标函数立刻抬升", fontsize=12)
ax.legend(fontsize=9.5); ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "ac_schedule_error.png"), dpi=150)
plt.close(fig)

print(f"[REF] T 权衡: T=0.5 E={float(E_T[1]):.1f}bps S={float(S_T[1]):.1f}bps | T=8 E={float(E_T[6]):.1f}bps S={float(S_T[6]):.1f}bps")
print(f"[REF] VWAP vs AC (T=2): E_ac={E_ac_b:.1f} S_ac={S_ac_b:.1f} | E_vw={E_vw_b:.1f} S_vw={S_vw_b:.1f}")
print(f"[REF] 目标基准={float(obj_curve[idx_best]):.1f}; tilt 扫描范围[{float(lams[0]):.1f},{float(lams[-1]):.1f}]")
print("DONE optimal-execution-almgren images")
