#!/usr/bin/env python3
"""Generate real charts + live stats for the Kalman-filter pairs-trading article.
All numbers printed are real outputs used in the blog text."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pykalman import KalmanFilter

CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
fm.fontManager.addfont(CJK)
CJK_NAME = fm.FontProperties(fname=CJK).get_name()

BASE = "/Users/halo/workspace/astro-blog/public/images/kalman-pairs-trading"
os.makedirs(BASE, exist_ok=True)

plt.rcParams.update({
    "font.family": CJK_NAME,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})

C1, C2, C3, C4, GREY = "#2c7fb8", "#d95f0e", "#1b9e77", "#7570b3", "#666666"

# ---------------------------------------------------------------------------
# 1. Simulate a cointegrated pair with a *slowly time-varying* hedge ratio
#    x_t = f_t + w_t                     (leg X, the hedged leg)
#    y_t = alpha_t + beta_t * f_t + u_t  (leg Y, the main leg)
#    f_t : common random-walk factor -> x and y share the same driver (cointegration)
#    beta_t : TRUE time-varying hedge ratio (what we want to track)
#    u_t : stationary OU mean-reverting error -> y - (alpha+beta*x) is stationary
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260717)
T = 1500
f = np.cumsum(rng.normal(0, 0.35, T))                 # common factor
beta_true = 1.05 + 0.18 * np.sin(2 * np.pi * np.arange(T) / 360.0)   # in [0.87, 1.23]
alpha_true = 8.0 + 0.6 * np.sin(2 * np.pi * np.arange(T) / 520.0)
ou = np.zeros(T)
eps = rng.normal(0, 0.45, T)
for i in range(1, T):
    ou[i] = 0.94 * ou[i - 1] + eps[i]

x = f + rng.normal(0, 0.6, T)                         # leg X
y = alpha_true + beta_true * f + ou                   # leg Y
dates = np.arange(T)

# ---------------------------------------------------------------------------
# 2. Kalman filter to TRACK the hedge ratio online
#    state theta_t = [alpha_t, beta_t];  scalar obs y_t = alpha_t + beta_t*x_t + nu
# ---------------------------------------------------------------------------
obs = np.vstack([x, np.ones(T)]).T[:, None, :]        # (T, n_dim_obs=1, n_dim_state=2)
kf = KalmanFilter(
    transition_matrices=np.eye(2),
    observation_matrices=obs,
    observation_covariance=1.0,
    transition_covariance=1e-3 * np.eye(2),
    initial_state_mean=np.array([8.0, 1.1]),
    initial_state_covariance=5.0 * np.eye(2),
)
state_means, _ = kf.filter(np.array(y))
alpha_kf = state_means[:, 1]
beta_kf = state_means[:, 0]
state_smooth, _ = kf.smooth(np.array(y))
beta_kf_s = state_smooth[:, 0]
alpha_kf_s = state_smooth[:, 1]

# ---------------------------------------------------------------------------
# 3. Dynamic spread + z-score trading
# ---------------------------------------------------------------------------
spread_kf = y - (alpha_kf_s + beta_kf_s * x)
mu, sd = spread_kf.mean(), spread_kf.std()
z = (spread_kf - mu) / sd

entry, exit_z = 2.0, 0.5
pos = np.zeros(T)
for t in range(1, T):
    if pos[t - 1] == 0:
        if z[t - 1] > entry:
            pos[t] = -1          # spread too high -> short spread (short Y, long beta*X)
        elif z[t - 1] < -entry:
            pos[t] = 1
    else:
        if abs(z[t - 1]) < exit_z:
            pos[t] = 0
        else:
            pos[t] = pos[t - 1]
pnl = pos * (np.diff(spread_kf, prepend=spread_kf[0]))
eq_kf = np.cumsum(pnl)

# ---------------------------------------------------------------------------
# 4. Fixed-hedge (full-sample OLS) benchmark
# ---------------------------------------------------------------------------
b_fixed, a_fixed = np.polyfit(x, y, 1)
spread_fixed = y - (a_fixed + b_fixed * x)
zf = (spread_fixed - spread_fixed.mean()) / spread_fixed.std()
posf = np.zeros(T)
for t in range(1, T):
    if posf[t - 1] == 0:
        if zf[t - 1] > entry:
            posf[t] = -1
        elif zf[t - 1] < -entry:
            posf[t] = 1
    else:
        if abs(zf[t - 1]) < exit_z:
            posf[t] = 0
        else:
            posf[t] = posf[t - 1]
pnlf = posf * (np.diff(spread_fixed, prepend=spread_fixed[0]))
eq_fixed = np.cumsum(pnlf)


def sharpe(eq):
    r = np.diff(eq)
    return r.mean() / (r.std() + 1e-9) * np.sqrt(252)


def mdd(eq):
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, np.nan)
    return float(np.nanmin(dd))


n_trades_kf = int(np.sum(np.diff(pos) != 0) / 2)
n_trades_fix = int(np.sum(np.diff(posf) != 0) / 2)

print("=== KALMAN PAIRS TRADING STATS ===")
print(f"T={T}  true beta range=({beta_true.min():.3f},{beta_true.max():.3f})  mean={beta_true.mean():.3f}")
print(f"KF smoothed beta range=({beta_kf_s.min():.3f},{beta_kf_s.max():.3f})  mean={beta_kf_s.mean():.3f}")
print(f"fixed OLS beta={b_fixed:.3f}  intercept={a_fixed:.3f}")
print(f"tracking RMSE(beta) vs true = {np.sqrt(np.mean((beta_kf_s-beta_true)**2)):.4f}")
print(f"spread stationary std: kf={sd:.3f}  fixed={spread_fixed.std():.3f}")
print(f"KF  eq final={eq_kf[-1]:.2f}  Sharpe={sharpe(eq_kf):.2f}  MDD={mdd(eq_kf):.2%}  trades={n_trades_kf}")
print(f"FIX eq final={eq_fixed[-1]:.2f}  Sharpe={sharpe(eq_fixed):.2f}  MDD={mdd(eq_fixed):.2%}  trades={n_trades_fix}")

# ---------------------------------------------------------------------------
# FIG 1: prices + dynamic spread
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
a1.plot(dates, x, color=C1, lw=1.6, label="标的 X（被对冲腿）")
a1.plot(dates, y, color=C2, lw=1.6, label="标的 Y（主仓腿）")
a1.set_title("协整配对：两只同走的价格（共享因子驱动）")
a1.set_ylabel("价格")
a1.legend(loc="upper left")
a2.plot(dates, spread_kf, color=C4, lw=1.3)
a2.axhline(mu, color="k", lw=1)
a2.axhline(mu + 2 * sd, color="r", ls="--", lw=1.2, label="+2 SD")
a2.axhline(mu - 2 * sd, color="g", ls="--", lw=1.2, label="−2 SD")
a2.set_title("卡尔曼动态对冲后的价差序列（平稳、均值回复）")
a2.set_ylabel("价差")
a2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{BASE}/kalman_prices_spread.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# FIG 2: tracked hedge ratio vs true
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(dates, beta_true, color=GREY, lw=2.2, label="真实 beta(t)（缓慢变化）")
ax.plot(dates, beta_kf_s, color=C1, lw=1.4, label="卡尔曼滤波估计 beta_hat(t)")
ax.axhline(b_fixed, color=C2, ls="--", lw=1.6, label=f"固定 OLS beta={b_fixed:.3f}")
ax.set_title("动态对冲比例：卡尔曼在线追踪缓慢变化的 beta(t)")
ax.set_xlabel("交易日")
ax.set_ylabel("对冲比例 beta")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{BASE}/kalman_hedge_ratio.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# FIG 3: signals + equity (KF)
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[1.4, 1])
a1.plot(dates, z, color=C4, lw=1.2, label="价差 z-score")
a1.axhline(entry, color="r", ls="--", lw=1.1, label="+2 SD 入场")
a1.axhline(-entry, color="r", ls="--", lw=1.1, label="−2 SD 入场")
a1.axhline(exit_z, color="g", ls=":", lw=1, label="+0.5 SD 离场")
a1.axhline(-exit_z, color="g", ls=":", lw=1, label="−0.5 SD 离场")
longs = np.where(pos == 1)[0]
shorts = np.where(pos == -1)[0]
a1.scatter(longs, z[longs], color=C3, s=14, zorder=3, label="持多价差")
a1.scatter(shorts, z[shorts], color=C2, s=14, zorder=3, label="持空价差")
a1.set_title("z-score 交易信号（±2 倍标准差入场，±0.5 倍标准差离场）")
a1.legend(loc="upper right", ncol=3, fontsize=9)
a2.plot(dates, eq_kf, color=C1, lw=1.8, label="卡尔曼动态对冲净值")
a2.set_title("卡尔曼配对策略累计净值")
a2.set_ylabel("净值")
a2.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{BASE}/kalman_signals.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# FIG 4: KF vs fixed hedge equity
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(dates, eq_kf, color=C1, lw=1.8, label=f"卡尔曼动态 beta (Sharpe {sharpe(eq_kf):.2f})")
ax.plot(dates, eq_fixed, color=C2, lw=1.6, label=f"固定 OLS beta (Sharpe {sharpe(eq_fixed):.2f})")
ax.set_title("动态对冲 vs 固定对冲：谁更贴近真实均值回复？")
ax.set_xlabel("交易日")
ax.set_ylabel("净值")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{BASE}/kalman_compare.png", dpi=130)
plt.close()

print("SAVED figures to", BASE)
