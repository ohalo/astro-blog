#!/usr/bin/env python3
"""Generate figures for Huang-Stoll spread decomposition article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/huang-stoll-decomposition"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---------------------------------------------------------------
# Simulate a trade-price series under the Huang-Stoll two-way model
# m_t : efficient (unobserved) price, follows random walk with info shocks
# Q_t : trade direction (+1 buy, -1 sell), serially correlated
# p_t = m_t + (s/2) Q_t  (observed transaction price)
# Spread s decomposed into: alpha (adverse selection), beta (inventory),
# 1 - alpha - beta (order processing)
# ---------------------------------------------------------------

N = 100_000
s = 0.10            # constant quoted (half-spread * 2) spread
half = s / 2.0

# true component fractions we bake in
alpha_true = 0.30   # adverse selection share
beta_true = 0.15    # inventory share
op_true = 1 - alpha_true - beta_true  # order processing

# trade direction with serial correlation (order splitting -> continuation)
# Huang-Stoll: E[Q_t | Q_{t-1}] = (1 - 2*pi) Q_{t-1}, pi = prob of reversal
pi = 0.35           # probability of direction reversal
Q = np.zeros(N)
Q[0] = 1 if rng.random() < 0.5 else -1
for i in range(1, N):
    if rng.random() < pi:
        Q[i] = -Q[i-1]
    else:
        Q[i] = Q[i-1]

# efficient price innovations: info component reacts to order flow surprise
# plus public noise
sig_pub = half * 0.6
m = np.zeros(N)
for i in range(1, N):
    # adverse selection: efficient price moves in direction of trade by alpha*half
    info_move = alpha_true * half * Q[i-1]
    pub = rng.normal(0, sig_pub)
    m[i] = m[i-1] + info_move + pub

# inventory: transitory quote shift proportional to beta, mean-reverting
# realized price = m + half*Q + inventory_term
inv = np.zeros(N)
for i in range(1, N):
    inv[i] = beta_true * half * Q[i-1]

p = m + half * Q + inv

# observed price changes
dp = np.diff(p)
dQ = np.diff(Q)

# ---------------------------------------------------------------
# Figure 1: The three cost components as stacked bar (identity)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
comps = [alpha_true, beta_true, op_true]
labels = ["逆向选择成分\n(信息, 永久)", "库存成分\n(暂时, 均值回复)", "订单处理成分\n(暂时, bid-ask bounce)"]
colors = ["#c0392b", "#e67e22", "#2980b9"]
bottom = 0
for c, lab, col in zip(comps, labels, colors):
    ax.bar(["报价价差 s = 0.10"], [c * s], bottom=bottom, color=col, label=lab, width=0.5)
    ax.text(0, bottom + c * s / 2, f"{c*s:.3f}\n({c*100:.0f}%)",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    bottom += c * s
ax.set_ylabel("价差绝对值 (元)")
ax.set_title("Huang-Stoll 三成分分解：0.10 元价差被谁拿走")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
ax.set_ylim(0, s * 1.05)
plt.tight_layout()
plt.savefig(f"{OUT}/hs-three-components.png", dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Figure 2: Serial covariance of price changes -> identifies spread
# Roll vs Huang-Stoll intuition. Show autocovariance structure.
# ---------------------------------------------------------------
# Roll estimator: s_roll = 2*sqrt(-Cov(dp_t, dp_{t-1}))
cov_dp = np.cov(dp[1:], dp[:-1])[0, 1]
s_roll = 2 * np.sqrt(max(-cov_dp, 0))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
# left: scatter dp_t vs dp_{t-1}
idx = rng.choice(N-2, size=3000, replace=False)
axes[0].scatter(dp[:-1][idx], dp[1:][idx], s=4, alpha=0.25, color="#34495e")
axes[0].axhline(0, color="gray", lw=0.6)
axes[0].axvline(0, color="gray", lw=0.6)
axes[0].set_xlabel("Δp$_{t-1}$")
axes[0].set_ylabel("Δp$_t$")
axes[0].set_title(f"价格变化的负串行相关\nCov = {cov_dp:.5f} → Roll 价差 = {s_roll:.3f}")

# right: autocovariance by lag
lags = range(1, 11)
acov = [np.cov(dp[k:], dp[:-k])[0, 1] for k in lags]
axes[1].bar(list(lags), acov, color=["#c0392b" if a < 0 else "#2980b9" for a in acov])
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xlabel("滞后阶数 k")
axes[1].set_ylabel("Cov(Δp$_t$, Δp$_{t-k}$)")
axes[1].set_title("自协方差结构：一阶负、高阶迅速衰减\n(bid-ask bounce 的指纹)")
plt.tight_layout()
plt.savefig(f"{OUT}/hs-serial-covariance.png", dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Figure 3: Two-stage regression recovery of alpha (info share)
# Stage 1: estimate trade direction unexpected component
# Stage 2: regress price change on Q and dQ
# Recover alpha (info), and the transitory piece
# ---------------------------------------------------------------
# HS regression: dp_t = (s/2)(Q_t - Q_{t-1}) + alpha*(s/2)*Q_{t-1} + e
# Build design matrix
Y = dp
X1 = half * (Q[1:] - Q[:-1])   # this coefficient ~ 1 in theory but we estimate scaling
# Simpler: regress dp on Q_t and Q_{t-1}
Qt = Q[1:]
Qtm1 = Q[:-1]
A = np.column_stack([Qt, Qtm1, np.ones(len(Qt))])
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
b_Qt, b_Qtm1, _ = coef
# implied half spread and info fraction
# p_t = m_t + half*Q_t ; dp = half*dQ + info_move
# coefficient on Q_t ~ half*(1) ; on Q_{t-1} ~ -half + alpha*half + beta*half
implied_half = b_Qt
implied_alpha_beta = (b_Qtm1 + implied_half) / implied_half  # (alpha+beta)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
est = [b_Qt, b_Qtm1]
true = [half, -half + (alpha_true + beta_true) * half]
xpos = np.arange(2)
w = 0.35
ax.bar(xpos - w/2, est, w, label="两步回归估计值", color="#2980b9")
ax.bar(xpos + w/2, true, w, label="模型真值", color="#95a5a6")
ax.set_xticks(xpos)
ax.set_xticklabels(["Q$_t$ 系数\n(即时价差半幅)", "Q$_{t-1}$ 系数\n(含信息+库存留存)"])
ax.axhline(0, color="black", lw=0.7)
ax.set_ylabel("回归系数 (元)")
ax.set_title(f"两步回归还原价差结构\n隐含半价差={implied_half:.4f}(真值{half:.4f}), 暂时+永久占比≈{implied_alpha_beta:.2f}")
ax.legend()
for i, (e, t) in enumerate(zip(est, true)):
    ax.text(i - w/2, e + (0.002 if e >= 0 else -0.004), f"{e:.4f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/hs-regression-recovery.png", dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Figure 4: Effective vs realized spread & price impact identity
# Effective spread = 2|p_t - m_t| ; realized spread = 2 Q_t (p_t - m_{t+tau})
# price impact = effective - realized
# ---------------------------------------------------------------
tau = 30  # bars ahead for "permanent" midpoint
eff = 2 * np.abs(p - m)                       # effective spread proxy
# realized: use future efficient price
realized = np.zeros(N - tau)
for i in range(N - tau):
    realized[i] = 2 * Q[i] * (p[i] - m[i + tau])
eff_mean = np.mean(eff)
real_mean = np.mean(realized)
impact_mean = eff_mean - real_mean

fig, ax = plt.subplots(figsize=(7.6, 4.6))
bars = ax.bar(["有效价差\n(投资者付出)", "价格冲击\n(信息, 永久)", "已实现价差\n(做市商赚到)"],
              [eff_mean, impact_mean, real_mean],
              color=["#8e44ad", "#c0392b", "#27ae60"])
for b, v in zip(bars, [eff_mean, impact_mean, real_mean]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("价差 (元)")
ax.set_title(f"价差恒等式：有效价差 = 价格冲击 + 已实现价差\n{eff_mean:.3f} ≈ {impact_mean:.3f} + {real_mean:.3f}")
plt.tight_layout()
plt.savefig(f"{OUT}/hs-effective-realized.png", dpi=130, bbox_inches="tight")
plt.close()

print("Huang-Stoll figures done")
print(f"Roll spread={s_roll:.4f}, implied_half={implied_half:.4f}")
print(f"eff={eff_mean:.4f} impact={impact_mean:.4f} realized={real_mean:.4f}")
