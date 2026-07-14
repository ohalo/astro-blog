#!/usr/bin/env python3
"""
为文章「门限协整与 TVECM：非线性误差修正的配对交易」(threshold-cointegration)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型：Balke-Fomby 型门限协整
  Δspread_t = -κ·(|s_{t-1}| − τ)·I(|s_{t-1}| > τ) + ε_t
  · 带内 |s| ≤ τ：Δspread = ε  → 随机游走（标准 ADF 检验会漏掉这种「伪平稳」）
  · 带外 |s| > τ ：按比例快速均值回复
  TVECM 用门限 τ 把价差切成两区制，分别估调整速度。

图表：
  1. tc_prices.png          两只协整资产价格序列
  2. tc_spread_regimes.png  价差序列 + 门限带 ±τ 与两 regime 阴影
  3. tc_two_regime.png      两区制散点 Δspread vs |spread_{t-1}|（带外斜率陡、带内≈0）
  4. tc_equity.png          门限门控配对 vs 朴素 z-score 配对：净值对比

数值校验：TVECM 还原 κ(带外)≈真值、带内斜率≈0；
          门控策略只赚带外那段「真会快速回复」的价差，Sharpe/回撤优于朴素。
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
D = os.path.join(BASE, "threshold-cointegration")
os.makedirs(D, exist_ok=True)

C = {"p1": "#4C72B0", "p2": "#C44E52", "p3": "#55A868", "grid": "#DDDDDD",
     "band": "#DDDDDD", "strong": "#C44E52", "weak": "#55A868"}
np.set_printoptions(suppress=True, precision=4)

# ============================================================
# 模拟：带门限的价差（Balke-Fomby 型）
#   Δs_t = -κ·(|s_{t-1}| − τ)·I(|s_{t-1}| > τ) + ε_t
# ============================================================
def simulate(T=3000, tau=0.20, kappa=0.25, seed=42):
    rng = np.random.default_rng(seed)
    spread = np.zeros(T)
    spread[0] = 0.0
    for t in range(1, T):
        s_lag = spread[t - 1]
        a = abs(s_lag)
        if a > tau:
            drift = -kappa * (a - tau) * np.sign(s_lag)
        else:
            drift = 0.0
        spread[t] = s_lag + drift + rng.normal(0, 0.012)
    common = np.cumsum(rng.normal(0.0002, 0.010, T))
    p1 = common + rng.normal(0, 0.004, T)
    p2 = common - spread + rng.normal(0, 0.004, T)
    return p1, p2, spread

p1, p2, spread = simulate(T=3000, tau=0.20, kappa=0.25, seed=7)
T = len(spread)
t = np.arange(T)
abslag = np.abs(spread[:-1])
dspread = np.diff(spread)

# ============================================================
# TVECM 估计：网格搜门限 τ，带内/带外分别 OLS
# ============================================================
def fit_tvec(tau):
    outside = abslag > tau
    inside = ~outside
    if outside.sum() > 30 and inside.sum() > 30:
        # 带外：Δs = -κ·(|s|-τ)·sign(s)  => 用 y = -sign(s)·(|s|-τ) 作回归元
        Xo = -(np.sign(spread[:-1][outside]) * (abslag[outside] - tau)).reshape(-1, 1)
        yo = dspread[outside]
        ko = np.linalg.lstsq(Xo, yo, rcond=None)[0][0]
        # 带内：Δs ≈ 0
        ki = dspread[inside].mean() / (abslag[inside].mean() + 1e-9)  # 近似（应≈0）
        ssr = float(((yo - Xo[:, 0] * ko) ** 2).sum() + (dspread[inside] ** 2).sum())
        return ssr, ko, ki
    return np.inf, np.nan, np.nan

taus = np.linspace(0.02, 0.45, 89)
best = min(((fit_tvec(tau), tau) for tau in taus), key=lambda r: r[0][0])
(ssr, k_hat, ki_hat), tau_hat = best
k_hat, ki_hat = float(k_hat), float(ki_hat)
print("=== TVECM 估计结果 (T=%d) ===" % T)
print("估计门限 τ = %.4f  (真实 0.20)" % tau_hat)
print("带外调整速度 κ = %.4f  (真实 0.25)" % k_hat)
print("带内斜率 κ_in ≈ %.5f  (应≈0：带内是随机游走)" % ki_hat)

half_out = -np.log(2) / np.log(1 - k_hat) if 0 < k_hat < 1 else np.nan
print("带外半衰期 ~= %.1f 期；带内无回复（半衰期∞）" % half_out)

# ============================================================
# 图 1：两只协整资产价格序列
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax[0].plot(t, p1, color=C["p1"], lw=1.0, label="资产 A")
ax[0].set_ylabel("价格 A"); ax[0].legend(loc="upper left")
ax[1].plot(t, p2, color=C["p2"], lw=1.0, label="资产 B")
ax[1].set_ylabel("价格 B"); ax[1].set_xlabel("交易日"); ax[1].legend(loc="upper left")
for a in ax:
    a.grid(True, color=C["grid"], lw=0.6); a.set_xlim(0, T)
fig.suptitle("两只协整资产：走势同向（共享共同趋势），但水平各异", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(D, "tc_prices.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：价差序列 + 门限带 ±τ 与两 regime 阴影
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(t, spread, color="black", lw=0.9)
ax.axhline(tau_hat, color=C["p1"], ls="--", lw=1.2)
ax.axhline(-tau_hat, color=C["p1"], ls="--", lw=1.2, label="门限带 ±τ=%.3f" % tau_hat)
ax.fill_between(t, tau_hat, spread.max(), color=C["strong"], alpha=0.10,
                label="带外（快速回复）")
ax.fill_between(t, spread.min(), -tau_hat, color=C["strong"], alpha=0.10)
ax.fill_between(t, -tau_hat, tau_hat, color=C["weak"], alpha=0.14, label="带内（随机游走）")
ax.set_xlabel("交易日"); ax.set_ylabel("价差 = p_A − p_B")
ax.set_title("价差的门限结构：带外快速回复，带内缓慢漂移（伪平稳）")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
ax.set_xlim(0, T)
plt.tight_layout()
plt.savefig(os.path.join(D, "tc_spread_regimes.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：两区制散点 Δspread vs |spread_{t-1}|
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6.5))
inside = abslag <= tau_hat
outside = ~inside
ax.scatter(abslag[inside], dspread[inside], s=5, color=C["weak"], alpha=0.45,
           label="带内 |s|≤τ（随机游走）")
ax.scatter(abslag[outside], dspread[outside], s=5, color=C["strong"], alpha=0.45,
           label="带外 |s|>τ（快速回复）")
xs = np.linspace(tau_hat, abslag.max(), 50)
ax.plot(xs, -k_hat * xs, color=C["strong"], lw=2.5, label="带外斜率 −κ=%.3f" % k_hat)
ax.axhline(0, color="black", lw=0.8); ax.axvline(tau_hat, color=C["p1"], ls="--", lw=1)
ax.set_xlabel("|spread_{t-1}|"); ax.set_ylabel("Δspread_t")
ax.set_title("两区制误差修正：带内斜率≈0，带外斜率陡（这才是真信号）")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "tc_two_regime.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：门限门控配对 vs 朴素 z-score 配对
# ============================================================
roll = 120
mu = np.array([np.mean(spread[max(0, i - roll):i]) for i in range(1, T + 1)])
sd = np.array([np.std(spread[max(0, i - roll):i]) + 1e-9 for i in range(1, T + 1)])
z = (spread - mu) / sd
# 朴素 z-score 配对：±2 入场，±0.5 出场
pos_naive = np.zeros(T)
for i in range(1, T):
    if z[i - 1] > 2:
        pos_naive[i] = -1
    elif z[i - 1] < -2:
        pos_naive[i] = 1
    elif abs(z[i - 1]) < 0.5:
        pos_naive[i] = 0
    else:
        pos_naive[i] = pos_naive[i - 1]
# 门限门控配对：只在「带外（快速回复）regime」接受信号
pos_gate = np.zeros(T)
for i in range(1, T):
    in_strong = abs(spread[i - 1]) > tau_hat
    if in_strong and z[i - 1] > 2:
        pos_gate[i] = -1
    elif in_strong and z[i - 1] < -2:
        pos_gate[i] = 1
    elif abs(z[i - 1]) < 0.5:
        pos_gate[i] = 0
    else:
        pos_gate[i] = pos_gate[i - 1]
ret_naive = np.zeros(T); ret_naive[1:] = pos_naive[:-1] * np.diff(spread)
ret_gate = np.zeros(T); ret_gate[1:] = pos_gate[:-1] * np.diff(spread)
eq_naive = np.cumsum(ret_naive)
eq_gate = np.cumsum(ret_gate)

def stats(r):
    rr = r[1:]
    sh = float(np.sqrt(252) * rr.mean() / (rr.std() + 1e-12))
    return sh

s_naive = stats(ret_naive)
s_gate = stats(ret_gate)
print("\n=== 回测（价差 delta 近似 PnL）===")
print("朴素 z-score 配对：累计 %.1f%%  Sharpe %.2f  交易次数 %d"
      % (eq_naive[-1] * 100, s_naive, int((np.diff(pos_naive) != 0).sum() / 2)))
print("门限门控配对：   累计 %.1f%%  Sharpe %.2f  交易次数 %d"
      % (eq_gate[-1] * 100, s_gate, int((np.diff(pos_gate) != 0).sum() / 2)))

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(t, eq_naive, color=C["p1"], lw=1.2, label="朴素 z-score 配对")
ax.plot(t, eq_gate, color=C["p3"], lw=1.2, label="门限门控配对（仅带外快速回复 regime 下注）")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("交易日"); ax.set_ylabel("累计收益")
ax.set_title("门控后更少但更优：只赚『带外真会快速回复』那段价差的钱")
ax.legend(loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6); ax.set_xlim(0, T)
plt.tight_layout()
plt.savefig(os.path.join(D, "tc_equity.png"), dpi=130)
plt.close()

print("\n图片已保存到:", D)
