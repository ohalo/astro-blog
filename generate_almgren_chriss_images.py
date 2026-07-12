#!/usr/bin/env python3
"""
为文章「Almgren-Chriss 最优执行：把交易成本写进下单算法」(almgren-chriss-execution)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

设定(自洽合成, 仅用于演示方法):
  * 需在 1 个交易日内(T=1 天)卖出 X=1,000,000 股, 分 N=20 笔执行
  * 初始价 S0=100, 日波动 daily_vol=2%, 每步噪声 std = daily_vol*S0/sqrt(N)
  * 永久冲击 g(v)=eta*v: 每卖 v 股, 之后所有价格永久下移 eta*v  (E 永久成本 = 0.5*eta*X^2)
  * 临时冲击 h(v)=tau*v: 仅影响本笔成交价 eta*v  (E 临时成本 = tau*sum(n_k^2))
  * 价格风险: 持有库存 x_j 暴露于噪声, Var[cost] = sigma_step^2 * sum_{j=1}^{N-1} x_j^2
  * 目标: min_{x}  E[cost] + (gamma/2) Var[cost]
        -> 三对角系统  -tau*x_{k-1} + (2*tau + b)*x_k - tau*x_{k+1} = 0,  b=(gamma/2)*sigma_step^2
        x_0=X, x_N=0; 每步卖出 n_k = x_{k-1} - x_k
  * gamma=0(风险中性) -> 匀速 TWAP; gamma 越大 -> 越前置(早卖早脱手库存)
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
D = os.path.join(BASE, "almgren-chriss-execution")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "rob": "#9467bd", "nom": "#C44E52", "thr": "#888888",
     "green": "#2ca02c", "orange": "#FF7F0E", "blue": "#1f77b4", "purple": "#8c564b"}

# ---------- 参数 ----------
S0 = 100.0
X = 1_000_000.0
notional = X * S0
daily_vol = 0.02
N = 20
dt = 1.0 / N
sigma_step = daily_vol * S0 * np.sqrt(dt)     # 每步价格噪声 std ($)
eta = 8e-8     # 永久冲击系数 ($/股^2): 0.5*eta*X^2 = $40,000
tau = 5e-7     # 临时冲击系数 ($/股^2): 匀速时 tau*X^2/N = $25,000

def solve_ac(gamma):
    a = tau
    b = 0.5 * gamma * sigma_step ** 2
    M = np.zeros((N - 1, N - 1))
    rhs = np.zeros(N - 1)
    for k in range(N - 1):
        M[k, k] = 2 * a + b
        if k > 0:
            M[k, k - 1] = -a
        if k < N - 2:
            M[k, k + 1] = -a
    rhs[0] = a * X
    y = np.linalg.solve(M, rhs)
    x = np.concatenate([[X], y, [0.0]])
    return x

def n_of(x):
    return x[:-1] - x[1:]          # n_k = x_{k-1} - x_k (>=0 表示卖出)

def cost_stats(x):
    n = n_of(x)
    E = 0.5 * eta * X ** 2 + tau * np.sum(n ** 2)
    Var = sigma_step ** 2 * np.sum(x[1:N] ** 2)   # x_1..x_{N-1}
    return E, Var

def e_bps(x):
    return cost_stats(x)[0] / notional * 1e4

def std_bps(x):
    return np.sqrt(cost_stats(x)[1]) / notional * 1e4

# ---------------------------------------------------------------- 图1: 最优执行轨迹(前置效应)
def fig_schedule():
    gammas = [0.0, 1e-8, 5e-8, 2e-7, 1e-6]
    cols = [C["thr"], C["blue"], C["rv"], C["orange"], C["vix"]]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for g, col in zip(gammas, cols):
        x = solve_ac(g)
        y = x / X * 100.0
        lbl = "TWAP (γ=0)" if g == 0 else (fr"γ={g:.0e}")
        ax.plot(range(N + 1), y, marker="o", ms=4, lw=1.8, color=col, label=lbl)
    ax.set_xlabel("执行步数 k (0 = 开始时, N = 全部卖完)", fontsize=11)
    ax.set_ylabel("剩余库存占总量 (%)", fontsize=11)
    ax.set_title(r"风险厌恶越大, 卖出越前置(早卖早脱手库存)", fontsize=12.5)
    ax.legend(fontsize=9.5)
    ax.grid(True, color=C["grid"])
    fig.tight_layout()
    fig.savefig(os.path.join(D, "ac_schedule.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- 图2: 执行权衡前沿(期望成本 vs 风险)
def fig_tradeoff():
    gs = [0.0, 1e-8, 5e-8, 2e-7, 5e-7, 1e-6, 3e-6, 1e-5, 3e-5]
    Es, Vs = [], []
    for g in gs:
        x = solve_ac(g)
        Es.append(e_bps(x)); Vs.append(std_bps(x))
    # 基准策略
    x_twap = X * (1 - np.arange(N + 1) / N)
    x_imm = np.concatenate([[X], np.zeros(N)])          # 一开盘全卖
    x_end = np.concatenate([[X], np.full(N - 1, X), [0.0]])  # 收盘全卖
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(Vs, Es, "o-", color=C["eq"], lw=1.8, label="Almgren-Chriss 最优轨迹")
    ax.scatter([std_bps(x_twap)], [e_bps(x_twap)], color=C["fv"], zorder=5, s=90, label="TWAP 匀速")
    ax.scatter([std_bps(x_imm)], [e_bps(x_imm)], color=C["vix"], zorder=5, s=90, label="开盘全抛")
    ax.scatter([std_bps(x_end)], [e_bps(x_end)], color=C["orange"], zorder=5, s=90, label="收盘全抛")
    ax.set_xlabel("执行风险  std(冲击成本) [bps]", fontsize=11)
    ax.set_ylabel("期望冲击成本  E[cost] [bps]", fontsize=11)
    ax.set_title(r"执行权衡前沿: 快=低风险高冲击, 慢=高波动低风险", fontsize=12.5)
    ax.legend(fontsize=9.5)
    ax.grid(True, color=C["grid"])
    fig.tight_layout()
    fig.savefig(os.path.join(D, "ac_tradeoff.png"), dpi=150)
    plt.close(fig)
    return e_bps(x_twap), std_bps(x_twap), e_bps(x_imm), std_bps(x_imm)

# ---------------------------------------------------------------- 图3: 实现短差(IS)分布 MC
def fig_is_dist(gamma=2e-7, trials=4000):
    rng = np.random.default_rng(20260713)
    def simulate(x):
        n = n_of(x)
        S = S0
        cash = 0.0
        for k in range(N):
            exec_price = S - tau * n[k]          # 临时冲击: 每股少 tau*n_k, 本笔成本 = tau*n_k^2
            cash += n[k] * exec_price
            S = S - eta * n[k] + sigma_step * rng.standard_normal()
        return (S0 * X - cash) / notional * 1e4   # 实现短差 bps
    x_ac = solve_ac(gamma)
    x_twap = X * (1 - np.arange(N + 1) / N)
    is_ac = np.array([simulate(x_ac) for _ in range(trials)])
    is_tw = np.array([simulate(x_twap) for _ in range(trials)])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bins = np.linspace(-250, 250, 60)
    ax.hist(is_ac, bins=bins, density=True, alpha=0.45, color=C["rob"], label=fr"AC 最优 (γ={gamma:.0e})")
    ax.hist(is_tw, bins=bins, density=True, alpha=0.45, color=C["nom"], label="TWAP 匀速")
    ax.axvline(is_ac.mean(), color=C["rob"], lw=1.8)
    ax.axvline(is_tw.mean(), color=C["nom"], lw=1.8)
    ax.set_xlabel("实现短差 Implementation Shortfall [bps]", fontsize=11)
    ax.set_ylabel("密度", fontsize=11)
    ax.set_title(r"蒙特卡洛: AC 最优把成本分布压窄(波动风险更低)", fontsize=12.5)
    ax.legend(fontsize=10)
    ax.grid(True, color=C["grid"], axis="y")
    print(f"[IS] AC  mean={is_ac.mean():.1f}bps std={is_ac.std():.1f}bps | TWAP mean={is_tw.mean():.1f}bps std={is_tw.std():.1f}bps")
    print(f"[IS] AC std 比 TWAP 窄 {1-is_ac.std()/is_tw.std():.1%}")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "ac_is_distribution.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- 图4: 冲击模型 + 成本拆解
def fig_impact_model():
    rng = np.random.default_rng(7)
    x = solve_ac(2e-7)
    n = n_of(x)
    # 单条价格路径: 中间价(含永久冲击) + 成交价(再减临时冲击)
    S = S0
    mid = [S0]
    exec_p = []
    for k in range(N):
        exec_p.append(S - tau * n[k])                     # 本笔成交价 = 当前中间价 - 临时冲击
        cash_k = n[k] * (S - tau * n[k])
        S = S - eta * n[k] + sigma_step * rng.standard_normal()
        mid.append(S)
    steps_mid = np.arange(N + 1)
    steps_exec = np.arange(N)
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0))
    axs[0].plot(steps_mid, mid, "-", color=C["eq"], lw=2, label="中间价(含永久冲击, 阶梯下行)")
    axs[0].plot(steps_exec, exec_p, "v", color=C["vix"], ms=7, label="成交价(再减临时冲击)")
    axs[0].set_xlabel("执行步数 k", fontsize=11)
    axs[0].set_ylabel("价格 ($)", fontsize=11)
    axs[0].set_title(r"永久冲击=阶梯下移; 临时冲击=每笔独立凹陷(不愈后)", fontsize=11.5)
    axs[0].legend(fontsize=9.5)
    axs[0].grid(True, color=C["grid"])
    # 成本拆解: AC vs TWAP
    x_twap = X * (1 - np.arange(N + 1) / N)
    def comp(x):
        n = n_of(x)
        perm = 0.5 * eta * X ** 2
        temp = tau * np.sum(n ** 2)
        risk = np.sqrt(sigma_step ** 2 * np.sum(x[1:N] ** 2))
        return perm, temp, risk
    p_a, t_a, r_a = comp(x)
    p_t, t_t, r_t = comp(x_twap)
    labels = ["永久冲击", "临时冲击", "波动风险(std)"]
    ac_v = [p_a / 1e4, t_a / 1e4, r_a / 1e4]
    tw_v = [p_t / 1e4, t_t / 1e4, r_t / 1e4]
    xpos = np.arange(3)
    axs[1].bar(xpos - 0.2, ac_v, width=0.4, color=C["rob"], label="AC 最优")
    axs[1].bar(xpos + 0.2, tw_v, width=0.4, color=C["nom"], label="TWAP")
    axs[1].set_xticks(xpos); axs[1].set_xticklabels(labels, fontsize=10.5)
    axs[1].set_ylabel("成本 ($ 万)", fontsize=11)
    axs[1].set_title(r"成本拆解: 更快的 AC 把波动风险(std)压下来", fontsize=11.5)
    axs[1].legend(fontsize=9.5)
    axs[1].grid(True, color=C["grid"], axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "ac_impact_model.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    fig_schedule()
    tw_e, tw_s, imm_e, imm_s = fig_tradeoff()
    fig_is_dist()
    fig_impact_model()
    print(f"[REF] TWAP E={tw_e:.1f}bps std={tw_s:.1f}bps | 开盘全抛 E={imm_e:.1f}bps std={imm_s:.1f}bps")
    print("DONE almgren-chriss-execution images")
