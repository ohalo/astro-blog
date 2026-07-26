#!/usr/bin/env python3
"""Generate figures for Stoll realized spread decomposition article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/stoll-realized-spread"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---- simulate trade sequence with three cost components ----
def simulate(n, S, w_info, w_inv, seed):
    """Stoll-style: quoted half spread S/2. After a buy at ask:
    - information component: mid permanently moves up by w_info*S/2
    - inventory component: quotes shift up by w_inv*S/2 then mean-revert via subsequent trades
    price reversal magnitude determines realized spread."""
    r = np.random.default_rng(seed)
    half = S / 2
    mid = 100.0
    inv_shift = 0.0
    trades = np.zeros(n)
    prices = np.zeros(n)
    mids = np.zeros(n)      # efficient mid (post-trade)
    qmids = np.zeros(n)     # quote midpoint just BEFORE the trade
    for i in range(n):
        qm = mid + inv_shift
        # trade direction: serial correlation from inventory-induced quote shifts
        p_buy = 0.5 - 0.35 * np.tanh(inv_shift / half) if half > 0 else 0.5
        q = 1 if r.random() < p_buy else -1
        price = qm + q * half
        # permanent info update
        mid += q * w_info * half
        # inventory shift (transient)
        inv_shift += q * w_inv * half
        inv_shift *= 0.7  # mean reversion of inventory pressure
        mid += 0.01 * r.standard_normal()  # public news
        trades[i] = q
        prices[i] = price
        mids[i] = mid
        qmids[i] = qm
    return trades, prices, mids, qmids

S = 0.10
n = 200_000

# Fig 1: covariance-based decomposition across three regimes
regimes = {
    "纯订单处理\n(θ=0, φ=0)": (0.0, 0.0),
    "信息为主\n(θ=0.6, φ=0.1)": (0.6, 0.1),
    "库存为主\n(θ=0.1, φ=0.6)": (0.1, 0.6),
}
fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
for ax, (name, (wi, wv)) in zip(axes, regimes.items()):
    trades, prices, mids, qmids = simulate(30_000, S, wi, wv, seed=11)
    dp = np.diff(prices)
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    ax.plot(prices[:300] , lw=0.9, color="steelblue")
    ax.set_title(f"{name}\ncov(Δp(t),Δp(t-1))={cov:.5f}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("成交价")
for ax in axes: ax.set_xlabel("成交序号")
fig.suptitle("三种成本结构下的成交价路径与串行协方差", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/stoll-price-paths.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# Fig 2: realized vs quoted spread as function of info share
info_grid = np.linspace(0, 0.9, 19)
realized = []
for wi in info_grid:
    trades, prices, mids, qmids = simulate(60_000, S, wi, 0.05, seed=23)
    # realized spread: 2*q_t*(p_t - qm_{t+5})
    horizon = 5
    rs = 2 * trades[:-horizon] * (prices[:-horizon] - qmids[horizon:])
    realized.append(rs.mean())
realized = np.array(realized)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(info_grid, realized, "o-", color="seagreen", lw=1.8, label="已实现价差（模拟）")
ax.axhline(S, color="crimson", ls="--", lw=1.4, label=f"报价价差 S = {S:.2f}")
ax.set_title("信息成分占比越高，做市商真正赚到的越少")
ax.set_xlabel("信息成分占比 θ"); ax.set_ylabel("价差 (元)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/stoll-realized-vs-quoted.png", dpi=130)
plt.close(fig)

# Fig 3: decomposition bar chart (estimated on simulated data)
trades, prices, mids, qmids = simulate(100_000, S, 0.45, 0.25, seed=31)
horizon = 5
rs = (2 * trades[:-horizon] * (prices[:-horizon] - qmids[horizon:])).mean()
# effective spread: against pre-trade quote midpoint
es = (2 * trades * (prices - qmids)).mean()
# price impact = effective - realized
pi = es - rs
fig, ax = plt.subplots(figsize=(8.5, 4.4))
bars = ax.bar(["报价价差 S", "有效价差", "价格冲击\n(信息成分)", "已实现价差\n(库存+订单处理)"],
              [S, es, pi, rs],
              color=["#888", "steelblue", "crimson", "seagreen"], alpha=0.85)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
            f"{b.get_height():.4f}", ha="center", fontsize=10)
ax.set_title("模拟数据上的价差分解（θ=0.45, φ=0.25）")
ax.set_ylabel("元")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/stoll-decomposition-bar.png", dpi=130)
plt.close(fig)

# Fig 4: serial covariance of price changes vs quote changes (Stoll's key identification)
w_info, w_inv = 0.45, 0.25
trades, prices, mids, qmids = simulate(100_000, S, w_info, w_inv, seed=41)
dp = np.diff(prices)
dm = np.diff(qmids)
cov_p = np.cov(dp[:-1], dp[1:])[0, 1]
cov_m = np.cov(dm[:-1], dm[1:])[0, 1]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.bar(["cov(Δp(t), Δp(t+1))\n成交价", "cov(Δq(t), Δq(t+1))\n报价中点"],
       [cov_p, cov_m], color=["steelblue", "darkorange"], alpha=0.85, width=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.text(0, cov_p - 0.00003, f"{cov_p:.6f}", ha="center", fontsize=11)
ax.text(1, cov_m + 0.000005, f"{cov_m:.6f}", ha="center", fontsize=11)
ax.set_title("Stoll 的识别核心：成交价与报价中点的串行协方差携带不同信息")
ax.set_ylabel("串行协方差")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/stoll-serial-cov.png", dpi=130)
plt.close(fig)

print("stoll figs done", rs, es, pi, cov_p, cov_m)
