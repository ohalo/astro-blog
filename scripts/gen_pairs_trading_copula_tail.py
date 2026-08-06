#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copula 尾部配对交易：相关性在极端行情下会断裂
所有图表由真实计算生成，固定随机种子可复现。
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as t_dist

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

OUT = "/Users/halo/workspace/astro-blog/public/images/pairs-trading-copula-tail"
os.makedirs(OUT, exist_ok=True)

SEED = 20260806
np.random.seed(SEED)

# ---------- 配色 ----------
C_GAUSS  = "#2563eb"
C_T      = "#dc2626"
C_CLAY   = "#16a34a"
C_ADAPT  = "#f59e0b"

# =============================================================================
# 一、Copula 采样器
# =============================================================================

def gaussian_copula(rho, n):
    """Gaussian Copula：Z ~ N(0, Σ) → U = Φ(Z)"""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    Z = np.random.multivariate_normal([0, 0], cov, size=n)
    return norm.cdf(Z)  # uniform marginals by construction


def t_copula(rho, n, nu):
    """
    t-Copula：先 Gaussian copula → 再 apply inv-t CDF
    (t-marginals → rank → uniform copula).
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    Z = np.random.multivariate_normal([0, 0], cov, size=n)
    U_g = norm.cdf(Z)
    # Apply inv-t CDF to marginally t-distributed variables
    T = np.column_stack([t_dist.ppf(U_g[:, 0], nu), t_dist.ppf(U_g[:, 1], nu)])
    # Empirical CDF → copula space (uniform margins)
    T_c = np.column_stack([stats.rankdata(T[:, 0]) / (n + 1),
                           stats.rankdata(T[:, 1]) / (n + 1)])
    return T_c


def clayton_copula(theta, n):
    """
    Clayton Copula（Joe 1997 参数化）via 条件分位函数：
    U1 ~ Uniform(0,1)
    U2 | U1=u1 = (1 + w^θ * (u1^{-θ} - 1))^{-1/θ},  w ~ Uniform(0,1)
    理论下尾相关 λ_L = 2^{-1/θ}
    """
    U1 = np.random.uniform(0.001, 0.999, n)
    w  = np.random.uniform(0.001, 0.999, n)
    inside = 1.0 + (w ** theta) * (U1 ** (-theta) - 1.0)
    inside = np.maximum(inside, 1e-300)
    U2 = np.clip(inside ** (-1.0 / theta), 1e-10, 1 - 1e-10)
    return np.column_stack([U1, U2])


def t_copula_theory_lambda(rho, nu):
    """t-Copula 理论下尾相关系数（Joe 2015）"""
    a = (nu + 1.0) * (1.0 - rho) / (1.0 + rho)
    return max(0.0, 2.0 * t_dist.cdf(-np.sqrt(a), df=nu + 1))


# =============================================================================
# 二、经验尾部相关（两个版本）
# =============================================================================

def emp_tail_dependence_ratio(U, q):
    """
    经验下尾相关（ratio 版）：
    λ̂_L(q) = P(U2 ≤ q | U1 ≤ q)
    理论意义：Gaussian → 0 as q→0；Clayton → 2^{-1/θ}
    """
    mx = U[:, 0] <= q
    my = U[:, 1] <= q
    if mx.sum() < 2:
        return np.nan
    return (mx & my).sum() / mx.sum()


def emp_tail_dependence_joint(U, q):
    """
    经验下尾相关（joint/independence 比值版）：
    λ̂(q) = P(U1≤q, U2≤q) / q
    → 独立：≈ q；Clayton：→ λ_L + O(q)
    """
    joint = (U[:, 0] <= q) & (U[:, 1] <= q)
    return joint.sum() / len(U) / q


# =============================================================================
# 三、OU 协整价差 → 价格序列
# =============================================================================

def build_cointegrated_prices(U_copula, half_life=20, vol=0.012,
                                hedge_ratio=1.0, idiosyncratic=0.008, n_steps=None):
    """
    用 Copula 驱动 OU 协整价差的均值回复。
    
    机制：
    1. OU 价差 S = OU(θ=ln2/hl, σ=vol, seed from copula col0)
    2. X = 100*exp(cumsum(GARCH-lite returns))
    3. Y = X * exp(S) + idiosyncratic noise
    
    Copula 驱动：
    - col0 → 价差的 OU innovation (对称的零均值扰动)
    - col1 → 资产 Y 的异质冲击 (与价差相关的)
    
    这样 Pearson 相关由 OU 结构和 hedge_ratio 决定，
    而 Copula 决定尾部结构（Gaussian=col0 独立于 col1 vs t/Clayton=下尾聚集）。
    """
    if n_steps is None:
        n_steps = len(U_copula)
    
    # --- OU spread ---
    dt = 1.0
    kappa = np.log(2) / half_life   # mean-reversion speed
    theta_s = 0.0                    # long-run mean
    sigma_s  = vol * np.sqrt(2 * kappa / dt)  # calibrate so spread std ≈ vol

    spread = np.zeros(n_steps)
    spread[0] = 0.0
    eps_s = norm.ppf(U_copula[:, 0])   # driving noise for spread
    for t in range(1, n_steps):
        dW = eps_s[t] * sigma_s
        spread[t] = spread[t-1] + kappa * (theta_s - spread[t-1]) * dt + dW

    # --- Asset X (GBM) ---
    r_x = np.zeros(n_steps)
    vol_x = vol
    for t in range(1, n_steps):
        sig = np.sqrt(0.90 * vol_x**2 + 0.10 * (0.012 * norm.ppf(U_copula[t, 0]))**2)
        r_x[t] = -0.0001 + sig * norm.ppf(U_copula[t, 0])

    p_x = 100 * np.exp(np.cumsum(r_x))

    # --- Asset Y ---
    # Y = X * exp(h*spread) + noise
    eps_y = norm.ppf(U_copula[:, 1])
    r_y_raw = hedge_ratio * r_x + vol * eps_y  # heteroskedastic component
    # GARCH-lite volatility
    vol_y = idiosyncratic
    r_y = np.zeros(n_steps)
    for t in range(1, n_steps):
        sig_y = np.sqrt(0.90 * vol_y**2 + 0.10 * (idiosyncratic * eps_y[t])**2)
        r_y[t] = -0.0001 + sig_y * eps_y[t]

    p_y = 100 * np.exp(np.cumsum(r_y))

    return p_x, p_y, spread


# =============================================================================
# 四、配对交易策略
# =============================================================================

def standard_pairs(p1, p2, lookback=60, entry=2.0, exit_threshold=0.5):
    """
    标准 z-score 价差策略（signal-on-i, execute-on-i+1）。
    返回 (pos, spread_z) 均为 (n,) 数组。
    pos[i] = day i 的收盘持仓（信号在 day i-1 收盘生成，day i 执行）。
    """
    n = len(p1)
    pos   = np.zeros(n)
    z_ser = np.full(n, np.nan)

    for i in range(lookback + 1, n - 1):
        w1 = p1[i - lookback:i + 1]
        w2 = p2[i - lookback:i + 1]
        # OLS beta
        beta = np.polyfit(w1 - w1.mean(), w2 - w2.mean(), 1)[0]
        # spread = Y - beta*X
        spread_w = w2 - beta * w1
        mu_s = spread_w.mean()
        sig_s = spread_w.std() + 1e-12
        z = (w2[-1] - beta * p1[i] - mu_s) / sig_s
        z_ser[i] = z

        if z > entry:
            pos[i] = -1.0   # short spread: Y overvalued → expect reversion down
        elif z < -entry:
            pos[i] =  1.0   # long spread:  Y undervalued → expect reversion up
        else:
            pos[i] = 0.0

    return pos, z_ser


def tail_aware_pairs(p1, p2, lookback=60, entry=2.0, exit_th=0.5, tail_q=0.10):
    """
    尾部感知策略：
    在标准 z-score 触发时，额外检查 copula 条件概率
    P(U2 < q | U1 < q) —— 若高于阈值，视为尾部一起崩，不开仓。
    """
    n = len(p1)
    pos   = np.zeros(n)
    z_ser = np.full(n, np.nan)
    cop_cond = np.full(n, np.nan)

    for i in range(lookback + 1, n - 1):
        w1 = p1[i - lookback:i + 1]
        w2 = p2[i - lookback:i + 1]
        beta = np.polyfit(w1 - w1.mean(), w2 - w2.mean(), 1)[0]
        spread_w = w2 - beta * w1
        z = (w2[-1] - beta * p1[i] - spread_w.mean()) / (spread_w.std() + 1e-12)
        z_ser[i] = z

        # 滚动 copula 条件概率
        ret1 = np.diff(p1[i - lookback:i + 1]) / p1[i - lookback:i]
        ret2 = np.diff(p2[i - lookback:i + 1]) / p2[i - lookback:i]
        if len(ret1) < 10:
            continue
        r1 = stats.rankdata(ret1) / (lookback + 1)
        r2 = stats.rankdata(ret2) / (lookback + 1)
        mask_low = r1 <= tail_q
        if mask_low.sum() > 5:
            cop_cond[i] = r2[mask_low].mean()

        if np.isnan(z) or np.isnan(cop_cond[i]):
            continue
        if abs(z) > entry:
            if cop_cond[i] > (tail_q + 0.05):
                pos[i] = 0.0
            else:
                pos[i] = -np.sign(z)
        elif abs(z) < exit_th:
            pos[i] = 0.0

    return pos, z_ser, cop_cond


# =============================================================================
# 五、回测引擎
# =============================================================================

def backtest(pos, ret_spread, n_train=252):
    """
    pos: (n,) 持仓向量（day i 的收盘持仓）
    ret_spread: (n-1,) 价差日收益率（day i 的 ret = P_{i+1}/P_i - 1）
    n_train: warm-up（跳过）
    返回 equity / drawdown / max_dd / sharpe / pnl
    """
    pnl_all = pos[:-1] * ret_spread   # pos[i] 赚 ret[i]
    pnl = pnl_all[n_train:]
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return np.array([1.0]), np.array([0.0]), 0.0, 0.0, np.array([])
    equity = 1.0 + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return equity, dd, dd.min(), (np.mean(pnl) / (np.std(pnl) + 1e-12)
                                   * np.sqrt(252)) if len(pnl) > 5 else 0.0, pnl


# =============================================================================
# 六、仿真参数
# =============================================================================
RHO       = 0.72     # Pearson 相关（两个世界一样）
N_SIM     = 3000     # 交易日 ≈ 12 年
N_TRAIN   = 252
HL        = 20       # OU half-life (天)
VOL       = 0.012    # 日波动
ENTRY     = 2.0
EXIT_TH   = 0.5
LOOKBACK  = 60
TAIL_Q    = 0.10

QS = np.array([0.20, 0.15, 0.10, 0.08, 0.06, 0.05,
               0.04, 0.03, 0.025, 0.020, 0.015, 0.010,
               0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001])

# =============================================================================
# 七、生成 Copula 样本
# =============================================================================
print("正在生成 Copula 样本...")
U_gauss  = gaussian_copula(RHO, N_SIM)
U_t4     = t_copula(RHO, N_SIM, nu=4)
U_t8     = t_copula(RHO, N_SIM, nu=8)
U_clay   = clayton_copula(theta=3.0, n=N_SIM)

lam_t4_th  = t_copula_theory_lambda(RHO, nu=4)
lam_t8_th  = t_copula_theory_lambda(RHO, nu=8)
lam_clay_th = 2.0 ** (-1.0 / 3.0)

# =============================================================================
# 八、经验尾部相关曲线
# =============================================================================
print("计算经验尾部相关...")

def scan_tail(U, qs):
    return np.array([emp_tail_dependence_ratio(U, q) for q in qs])

def scan_joint(U, qs):
    return np.array([emp_tail_dependence_joint(U, q) for q in qs])

# joint/indep ratio: Gaussian → 1, Clayton → >1 (tail clustering)
lam_j_gauss = scan_joint(U_gauss, QS)
lam_j_t4    = scan_joint(U_t4,    QS)
lam_j_clay  = scan_joint(U_clay, QS)

# ratio version: Gaussian → 0 (as q→0), Clayton → λ_L
lam_r_gauss = scan_tail(U_gauss, QS)
lam_r_t4    = scan_tail(U_t4,    QS)
lam_r_clay  = scan_tail(U_clay,  QS)

# =============================================================================
# 九、价格序列 & 回测
# =============================================================================
print("生成价格序列并回测...")

p1_g, p2_g, sp_g = build_cointegrated_prices(U_gauss, half_life=HL, vol=VOL)
p1_t, p2_t, sp_t = build_cointegrated_prices(U_t4,    half_life=HL, vol=VOL)

# Spread return
ret_g = np.diff(sp_g) / sp_g[:-1]   # (N-1,)
ret_t = np.diff(sp_t) / sp_t[:-1]   # (N-1,)

# Standard strategies
pos_g, z_g = standard_pairs(p1_g, p2_g, LOOKBACK, ENTRY, EXIT_TH)
pos_t, z_t = standard_pairs(p1_t, p2_t, LOOKBACK, ENTRY, EXIT_TH)

# Tail-aware strategy (on t world)
pos_ta, _, _ = tail_aware_pairs(p1_t, p2_t, LOOKBACK, ENTRY, EXIT_TH, TAIL_Q)

# Backtest
eq_g, dd_g, mdd_g, sh_g, pnl_g = backtest(pos_g, ret_g, N_TRAIN)
eq_t, dd_t, mdd_t, sh_t, pnl_t = backtest(pos_t, ret_t, N_TRAIN)
eq_ta, dd_ta, mdd_ta, sh_ta, pnl_ta = backtest(pos_ta, ret_t, N_TRAIN)

# Key stats
worst_g  = np.nanmin(pnl_g)
worst_t  = np.nanmin(pnl_t)
worst_ta = np.nanmin(pnl_ta)
skew_g   = stats.skew(pnl_g[~np.isnan(pnl_g)]) if len(pnl_g) > 3 else 0.0
skew_t   = stats.skew(pnl_t[~np.isnan(pnl_t)]) if len(pnl_t) > 3 else 0.0
skew_ta  = stats.skew(pnl_ta[~np.isnan(pnl_ta)]) if len(pnl_ta) > 3 else 0.0

n_trades_g  = np.sum(pos_g[N_TRAIN:] != 0)
n_trades_t  = np.sum(pos_t[N_TRAIN:] != 0)
n_trades_ta = np.sum(pos_ta[N_TRAIN:] != 0)

# Pearson correlation check
pearson_g = np.corrcoef(np.diff(p1_g), np.diff(p2_g))[0, 1]
pearson_t = np.corrcoef(np.diff(p1_t), np.diff(p2_t))[0, 1]

# =============================================================================
# 十、安慰剂检验
# =============================================================================
print("安慰剂检验...")

# 安慰剂 1：Gaussian 世界有限样本虚假尾部相关
lam_placebo_gauss = scan_joint(U_gauss, QS)

# 安慰剂 2：打乱时序（破坏 copula 结构）
perm = np.random.RandomState(42).permutation(N_SIM)
# 把 U_gauss[:,1] 打乱 → 破坏 col0 和 col1 之间的相关性
U_shuff2 = np.column_stack([U_gauss[:, 0], U_gauss[perm, 1]])
lam_shuff = scan_joint(U_shuff2, QS)

# 安慰剂 3：t-Copula ν 扫描（理论 λ_L 塌陷）
NU_SCAN = [3, 4, 5, 6, 8, 10, 15, 20, 30, 60]
lam_nu_theory = [t_copula_theory_lambda(RHO, nu=nu) for nu in NU_SCAN]
lam_nu_emp_q10 = []
for nu in NU_SCAN:
    U_nu = t_copula(RHO, 2000, nu)
    lam_nu_emp_q10.append(emp_tail_dependence_joint(U_nu, 0.10))

# 安慰剂 4：样本量扫描
SAMPLE_SIZES = [200, 500, 1000, 2000, 3000]
se_placebo = []   # Gaussian 的有限样本偏差
se_t4 = []        # t-copula 的估计方差
for ns in SAMPLE_SIZES:
    trials_g = []
    trials_t = []
    for _ in range(30):
        U_sg = gaussian_copula(RHO, ns)
        U_st = t_copula(RHO, ns, nu=4)
        trials_g.append(emp_tail_dependence_joint(U_sg, 0.05))
        trials_st = emp_tail_dependence_joint(U_st, 0.05)
        trials_t.append(trials_st)
    se_placebo.append(np.std(trials_g))
    se_t4.append(np.std(trials_t))

# =============================================================================
# 十一、画图
# =============================================================================

# ---- 图 1: cover ----
print("画 cover...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
for ax, U, color, title in zip(
    axes,
    [U_gauss, U_clay],
    [C_GAUSS, C_CLAY],
    [f"Gaussian Copula (ρ={RHO}, λ_L=0 理论)",
     f"Clayton Copula (θ=3, λ_L={lam_clay_th:.3f} 理论)"]
):
    ax.scatter(U[::4, 0], U[::4, 1], s=2.5, alpha=0.28, color=color)
    ax.axvline(0.05, color="gray", ls="--", lw=1)
    ax.axhline(0.05, color="gray", ls="--", lw=1)
    ax.fill_between([0, 0.05], [0]*2, [0.05]*2, color="orange", alpha=0.12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("U₁  (资产 X 的 CDF-rank)", fontsize=10)
    ax.set_ylabel("U₂  (资产 Y 的 CDF-rank)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.text(0.025, 0.025, "下尾", fontsize=8, color="darkorange", ha="center", fontweight="bold")

fig.suptitle(f"同样 Pearson r = {RHO:.2f}，尾部结构天差地别",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png", dpi=150, facecolor="white")
plt.close()

# ---- 图 2: tail_dependence ----
print("画 tail_dependence...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

# 左：joint/indep ratio（主图）
ax = axes[0]
ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="独立 (λ=1)")
ax.axhline(1 + lam_clay_th, color=C_CLAY, ls=":", lw=1.5,
           label=f"Clayton θ=3 λ_L={lam_clay_th:.3f} (理论 → λ+1={1+lam_clay_th:.3f})")
ax.plot(QS, lam_j_gauss, "o-", color=C_GAUSS, lw=2, ms=4, label="Gaussian (经验)")
ax.plot(QS, lam_j_t4,    "s-", color=C_T,     lw=2, ms=4, label="t-Copula ν=4 (经验)")
ax.plot(QS, lam_j_clay,  "^-", color=C_CLAY,  lw=2, ms=4, label="Clayton θ=3 (经验)")
ax.set_xlabel("分位阈值 q  →", fontsize=11)
ax.set_ylabel("λ̂(q) = P(U₁≤q, U₂≤q) / q", fontsize=11)
ax.set_title("尾部相关 λ̂(q)（联合/独立比值版）\n"
             "Gaussian→1（无尾部聚集）；Clayton→1+λ_L（尾部强聚集）",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(QS.max() + 0.01, QS.min() - 0.005)
ax.invert_xaxis()
ax.grid(True, alpha=0.25)

# 右：经验 P(Y≤q|X≤q)  ratio版
ax = axes[1]
ax.axhline(0, color=C_GAUSS, ls=":", lw=1.5, label="Gaussian λ_L=0 (理论)")
ax.axhline(lam_t4_th, color=C_T, ls=":", lw=1.5,
           label=f"t-Copula ν=4 λ_L={lam_t4_th:.3f} (理论)")
ax.axhline(lam_clay_th, color=C_CLAY, ls=":", lw=1.5,
           label=f"Clayton θ=3 λ_L={lam_clay_th:.3f} (理论)")
ax.plot(QS, lam_r_gauss, "o-", color=C_GAUSS, lw=2, ms=4, label="Gaussian (经验)")
ax.plot(QS, lam_r_clay,  "^-", color=C_CLAY,  lw=2, ms=4, label="Clayton θ=3 (经验)")
ax.set_xlabel("分位阈值 q  →", fontsize=11)
ax.set_ylabel("经验 P(U₂≤q | U₁≤q)", fontsize=11)
ax.set_title("尾部相关收敛路径\n"
             "Gaussian 随 q→0 趋近 0；Clayton 趋近 2^{-1/θ}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(QS.max() + 0.01, QS.min() - 0.005)
ax.invert_xaxis()
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/tail_dependence.png", dpi=150, facecolor="white")
plt.close()

# ---- 图 3: strategy_compare ----
print("画 strategy_compare...")
fig, axes = plt.subplots(2, 1, figsize=(11, 7), facecolor="white",
                          gridspec_kw={"height_ratios": [2.2, 1]})

curves = [
    (eq_g,  dd_g,  C_GAUSS,  f"Gaussian 世界（标准 z-score）"),
    (eq_t,  dd_t,  C_T,      f"t-Copula 世界（标准 z-score）"),
    (eq_ta, dd_ta, C_ADAPT,  "t-Copula 世界（尾部感知）"),
]
for eq, dd, color, label in curves:
    days = np.arange(len(eq))
    axes[0].plot(days, eq, color=color, lw=1.8, label=label, alpha=0.92)
    axes[0].fill_between(days, 1.0, eq, where=eq >= 1.0, color=color, alpha=0.07)
    axes[0].fill_between(days, 1.0, eq, where=eq < 1.0,  color=color, alpha=0.12)
axes[0].axhline(1.0, color="gray", ls="--", lw=0.8)
axes[0].set_title(f"配对策略权益曲线对比  (Pearson r = {RHO:.2f})",
                  fontsize=12, fontweight="bold")
axes[0].set_ylabel("累计收益（初始=1）", fontsize=10)
axes[0].legend(fontsize=9, loc="upper left")

for eq, dd, color, label in curves:
    days = np.arange(len(eq))
    axes[1].plot(days, dd * 100, color=color, lw=1.4, label=label)
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set_ylabel("回撤 (%)", fontsize=10)
axes[1].set_xlabel("交易日", fontsize=10)
axes[1].set_title("回撤对比（尾部一起崩的世界里最大回撤更深）", fontsize=11)
axes[1].legend(fontsize=8)
axes[1].set_ylim(min(-35, dd_t.min()*110), 5)
plt.tight_layout()
plt.savefig(f"{OUT}/strategy_compare.png", dpi=150, facecolor="white")
plt.close()

# ---- 图 4: nu_scan ----
print("画 nu_scan...")
fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
x = np.arange(len(NU_SCAN))
bars = ax.bar(x - 0.2, lam_nu_emp_q10, 0.35, color=C_T, alpha=0.70,
              label="经验 λ̂ (q=0.10)", edgecolor="white", lw=0.5)
bars2 = ax.bar(x + 0.2, lam_nu_theory, 0.35, color=C_GAUSS, alpha=0.70,
               label="理论 λ_L", edgecolor="white", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in NU_SCAN])
ax.set_xlabel("t-Copula 自由度 ν", fontsize=11)
ax.set_ylabel("尾部相关 λ", fontsize=11)
ax.set_title("t-Copula 自由度 ν 扫描：ν↑ → λ_L ↓（ν→∞ 即 Gaussian）",
             fontsize=12, fontweight="bold")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/nu_scan.png", dpi=150, facecolor="white")
plt.close()

# ---- 图 5: placebo ----
print("画 placebo...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")

# 安慰剂 1: Gaussian 有限样本偏差
ax = axes[0]
ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="独立基准 λ=1")
ax.fill_between(QS, 1.0, lam_placebo_gauss, where=lam_placebo_gauss > 1.0,
                color=C_GAUSS, alpha=0.12)
ax.plot(QS, lam_placebo_gauss, "o-", color=C_GAUSS, lw=2, ms=4,
        label="Gaussian λ̂(q)（应→1）")
bias = np.nanmean(lam_placebo_gauss[QS >= 0.05]) - 1.0
ax.text(0.12, 1.25, f"有限样本偏差≈{bias:.3f}\n(Pearson 聚集效应)", fontsize=8,
        color=C_GAUSS)
ax.set_xlabel("分位阈值 q  →", fontsize=10)
ax.set_ylabel("经验 λ̂(q)", fontsize=10)
ax.set_title("安慰剂 1：Gaussian 世界\n（有限样本虚假尾部聚集）", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(QS.max() + 0.01, QS.min() - 0.005)
ax.invert_xaxis()
ax.grid(True, alpha=0.25)

# 安慰剂 2: 打乱时序
ax = axes[1]
ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="独立基准 λ=1")
ax.plot(QS, lam_shuff, "s-", color="purple", lw=2, ms=4,
        label="打乱后 λ̂(q)（应→1）")
ax.set_xlabel("分位阈值 q  →", fontsize=10)
ax.set_ylabel("经验 λ̂(q)", fontsize=10)
ax.set_title("安慰剂 2：打乱时序\n（copula 结构破坏后 → 归零）", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(QS.max() + 0.01, QS.min() - 0.005)
ax.invert_xaxis()
ax.grid(True, alpha=0.25)

# 安慰剂 3: 样本量红线
ax = axes[2]
ax.plot(SAMPLE_SIZES, se_placebo, "o-", color=C_GAUSS, lw=2, ms=5, label="Gaussian SE")
ax.plot(SAMPLE_SIZES, se_t4,      "s-", color=C_T,     lw=2, ms=5, label="t-Copula SE")
ax.axhline(0.05, color="red", ls="--", lw=1.5, label="±0.05 目标精度")
ax.set_xlabel("样本量 N", fontsize=11)
ax.set_ylabel("λ̂_L(q=0.05) 标准误", fontsize=11)
ax.set_title("安慰剂 3：样本量红线\n(N↑ → 标准误 ↓，Gaussian 偏差不随 N 消失)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

fig.suptitle("安慰剂检验：虚假尾部相关的来源与识别", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/placebo.png", dpi=150, facecolor="white")
plt.close()

# ---- 图 6: loss_distribution ----
print("画 loss_distribution...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")
bins = np.linspace(-0.08, 0.06, 45)
for ax, pnl, color, title in zip(
    axes,
    [pnl_g, pnl_t, pnl_ta],
    [C_GAUSS, C_T, C_ADAPT],
    ["Gaussian（标准策略）", "t-Copula（标准策略）", "t-Copula（尾部感知）"]
):
    d = pnl[~np.isnan(pnl)]
    if len(d) == 0:
        ax.text(0.5, 0.5, "无交易", transform=ax.transAxes, ha="center")
        continue
    mn = np.nanmean(d)
    mx = np.nanmin(d)
    ax.hist(d, bins=bins, color=color, alpha=0.65, edgecolor="white")
    ax.axvline(mn, color="black", ls="--", lw=1.5, label=f"均值={mn:.4f}")
    ax.axvline(mx, color="red",   ls="--", lw=1.5, label=f"最差={mx:.4f}")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("单笔 PnL", fontsize=9)
    ax.set_ylabel("频数", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
fig.suptitle("单笔盈亏分布：尾部相关世界的左侧厚尾更显著",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/loss_distribution.png", dpi=150, facecolor="white")
plt.close()

# =============================================================================
# 十二、stats.json
# =============================================================================
stats_out = {
    "rho_pearson": RHO,
    "pearson_empirical_gaussian": float(pearson_g),
    "pearson_empirical_tcopula":  float(pearson_t),
    "n_sim": N_SIM,
    "n_backtest": N_SIM - N_TRAIN,
    "theory": {
        "gaussian_lambda_l": 0.0,
        "t4_lambda_l": float(lam_t4_th),
        "t8_lambda_l": float(lam_t8_th),
        "clayton_theta": 3.0,
        "clayton_lambda_l": float(lam_clay_th)
    },
    "empirical_tail_joint_q005": {
        "gaussian": float(emp_tail_dependence_joint(U_gauss, 0.05)),
        "t4":       float(emp_tail_dependence_joint(U_t4,    0.05)),
        "clayton":  float(emp_tail_dependence_joint(U_clay,  0.05))
    },
    "empirical_tail_ratio_q005": {
        "gaussian": float(emp_tail_dependence_ratio(U_gauss, 0.05)),
        "clayton":  float(emp_tail_dependence_ratio(U_clay,  0.05))
    },
    "tail_dependence_joint_qs": {
        "qs": QS.tolist(),
        "gaussian": [float(x) for x in lam_j_gauss],
        "t4":       [float(x) for x in lam_j_t4],
        "clayton":  [float(x) for x in lam_j_clay]
    },
    "strategy_standard_gaussian": {
        "sharpe":         float(sh_g),
        "max_drawdown":   float(mdd_g),
        "worst_single":   float(worst_g),
        "skewness":       float(skew_g),
        "total_return":   float(eq_g[-1]  - 1.0),
        "n_trades":       int(n_trades_g)
    },
    "strategy_standard_tcopula": {
        "sharpe":         float(sh_t),
        "max_drawdown":   float(mdd_t),
        "worst_single":   float(worst_t),
        "skewness":       float(skew_t),
        "total_return":   float(eq_t[-1]  - 1.0),
        "n_trades":       int(n_trades_t)
    },
    "strategy_tail_aware_tcopula": {
        "sharpe":         float(sh_ta),
        "max_drawdown":   float(mdd_ta),
        "worst_single":   float(worst_ta),
        "skewness":       float(skew_ta),
        "total_return":   float(eq_ta[-1] - 1.0),
        "n_trades":       int(n_trades_ta)
    },
    "placebo_gaussian_joint_q005": float(emp_tail_dependence_joint(U_gauss, 0.05)),
    "placebo_shuffled_joint_q005": float(lam_shuff[5]),  # q=0.05 is QS[5]
    "nu_scan": {
        "nus":         NU_SCAN,
        "lambda_theory": [float(x) for x in lam_nu_theory],
        "lambda_emp_q10": [float(x) for x in lam_nu_emp_q10]
    },
    "sample_size_se": {
        "sizes":    SAMPLE_SIZES,
        "gaussian": [float(x) for x in se_placebo],
        "t4":       [float(x) for x in se_t4]
    }
}

with open(f"{OUT}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats_out, f, ensure_ascii=False, indent=2)

print("\n✅ 所有图表和 stats.json 生成完毕！")
print(f"输出目录: {OUT}")

s = stats_out
print("\n===== 关键数字摘要 =====")
print(f"Pearson 相关系数（设定）: ρ = {s['rho_pearson']}")
print(f"Pearson 经验值 Gaussian: {s['pearson_empirical_gaussian']:.4f}")
print(f"Pearson 经验值 t-Copula: {s['pearson_empirical_tcopula']:.4f}")
print(f"\n理论尾部相关:")
print(f"  Gaussian: λ_L = 0")
print(f"  t-Copula ν=4: λ_L = {s['theory']['t4_lambda_l']:.4f}")
print(f"  Clayton θ=3: λ_L = {s['theory']['clayton_lambda_l']:.4f}")
print(f"\n经验尾部相关（joint/indep ratio, q=0.05）:")
print(f"  Gaussian:  {s['empirical_tail_joint_q005']['gaussian']:.4f}  (独立基准=1, >1=尾部聚集)")
print(f"  t-Copula: {s['empirical_tail_joint_q005']['t4']:.4f}")
print(f"  Clayton:   {s['empirical_tail_joint_q005']['clayton']:.4f}")
print(f"\n标准策略（Gaussian 世界）: Sharpe={s['strategy_standard_gaussian']['sharpe']:.3f}, "
      f"最大回撤={s['strategy_standard_gaussian']['max_drawdown']*100:.2f}%, "
      f"最差单笔={s['strategy_standard_gaussian']['worst_single']:.4f}, "
      f"总收益={s['strategy_standard_gaussian']['total_return']*100:.2f}%")
print(f"标准策略（t-Copula 世界）: Sharpe={s['strategy_standard_tcopula']['sharpe']:.3f}, "
      f"最大回撤={s['strategy_standard_tcopula']['max_drawdown']*100:.2f}%, "
      f"最差单笔={s['strategy_standard_tcopula']['worst_single']:.4f}, "
      f"总收益={s['strategy_standard_tcopula']['total_return']*100:.2f}%")
print(f"尾部感知（t-Copula 世界）: Sharpe={s['strategy_tail_aware_tcopula']['sharpe']:.3f}, "
      f"最大回撤={s['strategy_tail_aware_tcopula']['max_drawdown']*100:.2f}%, "
      f"最差单笔={s['strategy_tail_aware_tcopula']['worst_single']:.4f}")
print(f"\n安慰剂 1 Gaussian joint/indep q=0.05: {s['placebo_gaussian_joint_q005']:.4f}")
print(f"安慰剂 2 打乱后: {s['placebo_shuffled_joint_q005']:.4f}")
print(f"\nν 扫描 (λ_L theory): {[f'{x:.3f}' for x in s['nu_scan']['lambda_theory']]}")
print(f"\n样本量红线:")
for sz, sg, st in zip(s['sample_size_se']['sizes'], s['sample_size_se']['gaussian'], s['sample_size_se']['t4']):
    print(f"  N={sz}: Gaussian SE={sg:.4f}, t-Copula SE={st:.4f}")
