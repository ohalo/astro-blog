#!/usr/bin/env python3
"""
为文章「OU 过程均值回复：用 Ornstein-Uhlenbeck 给配对价差做统计套利」
(mean-reversion-ou) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 协整对：Y_t = beta * X_t + spread_t，其中 spread_t 服从 OU 过程
        dx_t = kappa*(theta - x_t)*dt + sigma*dW_t
    离散化：x_t = theta + b*(x_{t-1} - theta) + sigma*sqrt(dt)*eps, b = exp(-kappa*dt)
  * 半衰期：HL = ln2 / kappa = -dt*ln2 / ln(b)
  * 信号：z = (spread - mean) / std；|z| > entry 开仓，回到 |z| < exit 平仓
  * 执行：signal-on-i, execute-on-(i+1)，避免前视
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
D = os.path.join(BASE, "mean-reversion-ou")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "spread": "#55A868",
      "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3"}

rng = np.random.default_rng(20260718)


def simulate_pair(n=600, beta=1.0, kappa_x=0.02, kappa_s=0.05, theta_s=0.0,
                  sigma_x=0.006, sigma_s=0.004, dt=1.0, seed=0):
    r = np.random.default_rng(seed)
    # X：平稳的慢均值回复（共同驱动），避免价格发散
    bx = np.exp(-kappa_x * dt)
    x = np.zeros(n)
    x[0] = sigma_x * r.normal()
    for t in range(1, n):
        x[t] = bx * x[t - 1] + sigma_x * np.sqrt(dt) * r.normal()
    # 价差 = OU 过程（快均值回复）
    bs = np.exp(-kappa_s * dt)
    spread = np.zeros(n)
    spread[0] = theta_s + sigma_s * r.normal()
    for t in range(1, n):
        spread[t] = theta_s + bs * (spread[t - 1] - theta_s) + sigma_s * np.sqrt(dt) * r.normal()
    y = beta * x + spread
    return x, y, spread, bs


def estimate_ou(x):
    """AR(1) 斜率法估计 OU：x_t = a + b x_{t-1} + e，kappa = -ln(b)/dt"""
    xlag = x[:-1]
    xnow = x[1:]
    # OLS y = a + b*x
    b_hat, a_hat = np.polyfit(xlag, xnow, 1)
    resid = xnow - (a_hat + b_hat * xlag)
    dt = 1.0
    kappa = -np.log(b_hat) / dt
    theta = a_hat / (1.0 - b_hat)
    sigma = np.std(resid) * np.sqrt((1.0 - b_hat ** 2))  # 近似
    hl = np.log(2.0) / kappa if kappa > 0 else np.inf
    return dict(b=b_hat, a=a_hat, kappa=kappa, theta=theta, sigma=sigma, hl=hl)


def backtest(spread, entry=2.0, exit_z=0.5, fee=0.0008):
    """信号在 i 生成，i+1 以 mark-to-market 结算；多空价差均值回复。
    持仓 pos 在 { -1, 0, +1 }，每日收益 = pos * (spread_t - spread_{t-1})，
    即空价差在价差回落时赚钱、多价差在价差回升时赚钱。
    """
    z = (spread - np.mean(spread)) / np.std(spread)
    n = len(z)
    pos = np.zeros(n)
    ret = np.zeros(n)
    cur = 0
    for t in range(1, n):
        target = 0
        if cur == 0 and z[t - 1] > entry:
            target = -1   # 价差过高 -> 做空价差
        elif cur == 0 and z[t - 1] < -entry:
            target = 1
        elif cur != 0 and abs(z[t - 1]) < exit_z:
            target = 0
        else:
            target = cur
        if target != cur:
            ret[t] -= fee * abs(target - cur)
        cur = target
        pos[t] = cur
        ret[t] += cur * (spread[t] - spread[t - 1])  # 当日盯市收益
    eq = np.cumprod(1.0 + ret)
    return pos, eq


# ---------- 生成数据 ----------
x, y, spread, b = simulate_pair(seed=7)
ou = estimate_ou(spread)
pos, strat_eq = backtest(spread, entry=2.0, exit_z=0.5)
bh = np.ones_like(strat_eq)   # 持有现金基准（价差本身零漂移，持有它无收益）

print("OU 估计: b=%.4f kappa=%.4f theta=%.3f HL=%.1f 步" % (ou["b"], ou["kappa"], ou["theta"], ou["hl"]))
print("策略终值=%.3f  Buy&Hold(Y)=%.3f" % (strat_eq[-1], bh[-1]))

# ============ 图 1：价格路径 + 价差 ============
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(x, color=C["x"], lw=1.4, label="X (共同驱动)")
ax[0].plot(y, color=C["y"], lw=1.4, label="Y (beta*X + spread)")
ax[0].set_ylabel("价格", fontsize=11)
ax[0].legend(loc="upper left", fontsize=9)
ax[0].set_title("协整对：Y 长期锁定在 beta*X 附近，偏离即价差", fontsize=12)
ax[0].grid(True, color=C["grid"])

ax[1].plot(spread, color=C["spread"], lw=1.4)
ax[1].axhline(ou["theta"], color=C["band"], ls="--", lw=1.0, label="OU 中枢 theta")
ax[1].axhline(ou["theta"] + 2 * np.std(spread), color=C["band"], ls=":", lw=1.0)
ax[1].axhline(ou["theta"] - 2 * np.std(spread), color=C["band"], ls=":", lw=1.0)
ax[1].set_ylabel("价差 spread", fontsize=11)
ax[1].legend(loc="upper right", fontsize=9)
ax[1].grid(True, color=C["grid"])

z = (spread - np.mean(spread)) / np.std(spread)
ax[2].plot(z, color=C["mk"], lw=1.2)
ax[2].axhline(2.0, color=C["band"], ls="--", lw=1.0, label="开仓阈值 ±2")
ax[2].axhline(-2.0, color=C["band"], ls="--", lw=1.0)
ax[2].axhline(0.5, color=C["grid"], ls="-", lw=0.8)
ax[2].axhline(-0.5, color=C["grid"], ls="-", lw=0.8)
ax[2].set_ylabel("z-score", fontsize=11)
ax[2].set_xlabel("交易日", fontsize=11)
ax[2].legend(loc="upper right", fontsize=9)
ax[2].grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "pair_price_spread_z.png"), dpi=130)
plt.close(fig)

# ============ 图 2：价差 z 与开平仓信号 ============
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(z, color=C["mk"], lw=1.2, label="价差 z-score")
ax.axhline(2.0, color=C["band"], ls="--", lw=1.0, label="开仓 ±2 / 平仓 ±0.5")
ax.axhline(-2.0, color=C["band"], ls="--", lw=1.0)
ax.axhline(0.5, color="#999999", ls=":", lw=0.9)
ax.axhline(-0.5, color="#999999", ls=":", lw=0.9)
# 标注开平仓点
opens = np.where((pos == 1) & (np.r_[0, pos[:-1]] == 0))[0]
closes = np.where((pos == 0) & (np.r_[0, pos[:-1]] != 0))[0]
ax.scatter(opens, z[opens], color=C["spread"], s=36, zorder=5, label="开仓")
ax.scatter(closes, z[closes], color=C["x"], s=36, marker="v", zorder=5, label="平仓")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("z-score", fontsize=11)
ax.set_title("OU 均值回复交易：|z|>2 开仓、|z|<0.5 平仓", fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "ou_trading_signals.png"), dpi=130)
plt.close(fig)

# ============ 图 3：净值曲线 策略 vs 买入持有 ============
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(strat_eq, color=C["eq"], lw=1.6, label="OU 配对策略（净值 %.3f）" % strat_eq[-1])
ax.plot(bh, color=C["bh"], lw=1.3, ls="--", label="持有现金基准（净值 1.000）")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_title("净值对比：从零漂移价差里榨出 %.1f%% 收益，而持有现金不动" % ((strat_eq[-1]-1)*100), fontsize=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "ou_equity_vs_buyhold.png"), dpi=130)
plt.close(fig)

print("saved 3 figures to", D)
