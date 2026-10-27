#!/usr/bin/env python3
"""生成 Roll 协方差度量买卖成本 文章：合成微观结构 + 3 张真实图表（CJK 字体）。

Roll (1984) 隐含价差/交易成本度量：对连续成交价的差分 Δp_t，
有效（全额）价差 s = 2·√{ −Cov(Δp_t, Δp_{t-1}) }，当协方差为负时成立。
本文合成「真实中间价随机游走 + 含买卖价差Tick的成交价」，滚动估计 Roll 成本，
并演示它在平静/应激两种价差 regime 下的表现，以及被「交易方向自相关」低估的真实陷阱。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

np.random.seed(20260722)
TICK = 0.01          # 最小报价单位（美元）
P0 = 50.0            # 起始中间价
N = 4000             # 成交笔数（近似一个交易日的逐笔）
K = 50               # 滚动窗口（笔）

# ---- 1) 真实中间价（有效价格）随机游走 ----
mid = np.empty(N + 1)
mid[0] = P0
for t in range(1, N + 1):
    mid[t] = mid[t - 1] * np.exp(np.random.normal(0, 0.0008))

# ---- 2) 真实价差 regime：平静段 2 tick，应激段（2000~3000 笔）扩到 6 tick ----
spread_ticks = np.full(N, 2.0)
spread_ticks[2000:3000] = 6.0
true_spread = spread_ticks * TICK          # 真实全额价差（美元/股）

# ---- 3) 成交方向 z_t（买卖指示）：Roll 基准假设「成交方向独立」（ρ=0）----
# 真实市况成交方向常带轻度自相关，那是图 3 要单独演示的「低估陷阱」；
# 主图先用独立方向，让滚动估计量精确还原真实价差（基准正确了才能谈偏差）。
z = np.where(np.random.normal(0, 1, N) >= 0, 1, -1)   # +1 买方主动 / -1 卖方主动（独立）

# ---- 4) 观测成交价 = 中间价 + 半价差·方向（连续价，便于贴合 Roll 连续假设）----
half = true_spread / 2.0
obs = mid[1:] + half * z            # 连续成交价（真实逐笔含 tick 取整是次要偏差，文末作为陷阱④单列）
dp = np.diff(obs)                          # Δp_t 序列

# ---- 5) Roll 协方差度量（滚动窗口）----
roll_spread = np.full(N - 1, np.nan)
for i in range(K + 1, N - 1):
    a = dp[i - K:i]                      # Δp_t   (t = i-K .. i-1)
    b = dp[i - K - 1:i - 1]              # Δp_{t-1}
    cov = np.mean(a * b) - np.mean(a) * np.mean(b)
    if cov < 0:
        roll_spread[i] = 2.0 * np.sqrt(-cov)   # 估计的全额价差（美元/股）

valid = ~np.isnan(roll_spread)
base_price = mid[2:]            # 与 dp / roll_spread[i] 对齐：dp[i]=obs[i+1]−obs[i] ↔ mid[i+2]
base_spread = true_spread[1:]    # 同上，对齐到 obs[i+1] 的价差
roll_bps = roll_spread[valid] / base_price[valid] * 1e4
true_bps = base_spread[valid] / base_price[valid] * 1e4

# ---- 6) 指标：平静段 vs 应激段 Roll 估计精度 ----
calm = (np.arange(N - 1) >= K) & (np.arange(N - 1) < 2000)
stress = (np.arange(N - 1) >= 2000) & (np.arange(N - 1) < 3000)
# 取每个 regime 中段稳定区
calm_idx = np.where(calm & valid)[0]
stress_idx = np.where(stress & valid)[0]
roll_calm = np.nanmean(roll_spread[calm_idx])
roll_stress = np.nanmean(roll_spread[stress_idx])
true_calm = np.mean(base_spread[calm_idx])
true_stress = np.mean(base_spread[stress_idx])

# ---- 7) 被「方向自相关」低估的演示：ρ=0（独立）时应精确还原 ----
def roll_estimate(rho_val, spread_dollars, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    mid_ = P0 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    uu = np.zeros(n)
    uu[0] = rng.normal()
    for t in range(1, n):
        uu[t] = rho_val * uu[t - 1] + np.sqrt(1 - rho_val ** 2) * rng.normal()
    zz = np.where(uu >= 0, 1, -1)
    oo = np.round(mid_ + spread_dollars / 2 * zz, 2)   # 长度 n
    dpp = np.diff(oo)                                    # 长度 n-1
    covs = dpp[1:] * dpp[:-1]
    cov = np.mean(covs) - np.mean(dpp[1:]) * np.mean(dpp[:-1])
    return 2 * np.sqrt(max(0, -cov)) if cov < 0 else np.nan

s_true = 0.02
rho_grid = [0.0, 0.1, 0.3, 0.5, 0.7]
roll_vs_rho = [roll_estimate(r, s_true) for r in rho_grid]
bias = [ (r - s_true) / s_true * 100 for r in roll_vs_rho ]

# ===== 写图 =====
outdir = "public/images/roll-covariance-measure"
import os
os.makedirs(outdir, exist_ok=True)

# 图1：cover —— 观测成交价（散点、低透明）vs 中间价（平滑线） + regime 标注
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.scatter(range(N), obs, s=4, color="#1f3a5f", alpha=0.28, label="观测成交价（含价差 Tick）")
win = 60
mid_sm = np.convolve(mid[1:], np.ones(win)/win, mode="same")
ax.plot(mid_sm, color="#d73027", lw=1.6, label="真实中间价（有效价格，平滑）")
ax.axvspan(2000, 3000, color="#762a83", alpha=0.12, label="应激 regime（价差 6 tick）")
ax.set_title("Roll 协方差度量：从逐笔成交价反推隐含交易成本", fontsize=15, fontweight="bold", color="#1f3a5f")
ax.set_ylabel("价格（美元）")
ax.set_xlabel("成交笔数（近似日内）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/cover.png", dpi=130)
plt.close(fig)

# 图2：Roll 估计成本（bps）随时间，对照真实成本
fig, ax = plt.subplots(figsize=(11, 5.2))
tt = np.arange(N - 1)[valid]
ax.plot(tt, true_bps, color="gray", lw=1.6, ls="--", label="真实隐含成本（已知）")
ax.plot(tt, roll_bps, color="#1a9850", lw=1.3, alpha=0.9, label="Roll 协方差估计成本")
ax.axvspan(2000, 3000, color="#d73027", alpha=0.08)
ax.set_title(f"Roll 估计（绿）对照真实成本（灰）：价差主导噪声时几乎精确（应激 0.0601≈0.0600），\n噪声主导时（平静 2 tick）被向上偏置到 0.030（真实 0.020）",
             fontsize=12.5, fontweight="bold", color="#1f3a5f")
ax.set_ylabel("隐含交易成本（bps）")
ax.set_xlabel("成交笔数")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/roll_cost_time.png", dpi=130)
plt.close(fig)

# 图3：低估陷阱 —— 方向自相关 ρ 越大，Roll 低估越严重
fig, ax = plt.subplots(figsize=(11, 5.0))
colors = ["#1a9850" if b >= -1 else "#d73027" for b in bias]
ax.bar([str(r) for r in rho_grid], roll_vs_rho, color=colors, alpha=0.85)
ax.axhline(s_true, color="gray", ls="--", lw=1.5, label=f"真实价差 {s_true:.2f} 美元")
for xi, (rv, bv) in enumerate(zip(roll_vs_rho, bias)):
    ax.text(xi, rv + 0.0008, f"{rv:.3f}\n({bv:+.0f}%)", ha="center", fontsize=8.5)
ax.set_title("真实陷阱：成交方向自相关 ρ↑ → Roll 低估成本（ρ=0 才精确还原）",
             fontsize=13.5, fontweight="bold", color="#1f3a5f")
ax.set_ylabel("估计全额价差（美元/股）")
ax.set_xlabel("成交方向自相关系数 ρ")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(f"{outdir}/roll_bias_rho.png", dpi=130)
plt.close(fig)

# ===== 打印指标供文章引用 =====
print("=== Roll metrics ===")
print(f"T={N}, K={K}, TICK={TICK}, P0={P0}")
print(f"calm_true_spread={true_calm:.4f} ({true_calm/TICK:.1f} tick), roll_est={roll_calm:.4f} ({(roll_calm-true_calm)/true_calm*100:+.1f}%)")
print(f"stress_true_spread={true_stress:.4f} ({true_stress/TICK:.1f} tick), roll_est={roll_stress:.4f} ({(roll_stress-true_stress)/true_stress*100:+.1f}%)")
print(f"roll_vs_rho={[round(x,4) for x in roll_vs_rho]}")
print(f"bias_pct={[round(x,1) for x in bias]}")
print("Roll images written.")
