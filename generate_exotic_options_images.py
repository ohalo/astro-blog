#!/usr/bin/env python3
"""
为文章「障碍期权与奇异期权在结构化策略中的应用」(exotic-options-structures) 生成真实配图。
数据：GBM 模拟标的路径 + 蒙特卡洛障碍期权定价 + 自动赎回票据(Autocallable)收益结构。
图表：
  1. barrier_paths.png      GBM 路径 + 障碍线，触发敲出的路径标红
  2. barrier_vs_vanilla.png 向下敲出看涨(DO Call)vs 普通看涨价格随障碍/执行价变化
  3. autocall_payoff.png    自动赎回票据收益结构（赎回观察 + 到期保本/亏损区）
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
D = os.path.join(BASE, "exotic-options-structures")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

# ============================================================
# 1) 标的路径：GBM
# ============================================================
def gbm_paths(S0, mu, sigma, T, M, N):
    dt = T / M
    # 返回 (N, M+1) 价格路径
    Z = np.random.randn(N, M)
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    logS = np.log(S0) + np.cumsum(increments, axis=1)
    S = np.exp(np.column_stack([np.full(N, np.log(S0)), logS]))
    return S

S0, mu, sigma, T, M = 100.0, 0.05, 0.25, 1.0, 252
paths = gbm_paths(S0, mu, sigma, T, M, 4000)

# ============================================================
# 图1：GBM 路径 + 向下敲出障碍（Down-and-Out）
# ============================================================
barrier = 80.0      # 向下敲出障碍
K_show = 100.0
np.random.seed(7)
show = gbm_paths(S0, mu, sigma, T, M, 12)   # 12 条展示路径
fig, ax = plt.subplots(figsize=(11, 5.6))
t = np.arange(M + 1)
knocked = 0
for i in range(show.shape[0]):
    knocked_out = np.any(show[i] <= barrier)
    if knocked_out:
        ko_idx = np.argmax(show[i] <= barrier)
        ax.plot(t[:ko_idx + 1], show[i][:ko_idx + 1], color="#d62728", lw=1.2, alpha=0.85)
        ax.scatter(ko_idx, show[i][ko_idx], color="#d62728", s=28, zorder=5)
        knocked += 1
    else:
        ax.plot(t, show[i], color="#1f77b4", lw=1.1, alpha=0.7)
ax.axhline(barrier, color="#d62728", ls="--", lw=1.6, label=f"向下敲出障碍 H={barrier}")
ax.axhline(S0, color="#444", ls=":", lw=1.1, label=f"起始价 S0={S0}")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("标的价格", fontsize=11)
ax.set_title(f"向下敲出看涨期权 (Down-and-Out Call)：触及 H 即作废（演示 {knocked} 条路径敲出）",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "barrier_paths.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 2) 蒙特卡洛定价：向下敲出看涨 vs 普通看涨
# ============================================================
def mc_down_out_call(S0, K, H, r, sigma, T, M, N):
    """向下敲出看涨，障碍在期初即生效（美式障碍，路径依赖）。"""
    dt = T / M
    Z = np.random.randn(N, M)
    logS = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.exp(np.column_stack([np.full(N, np.log(S0)), logS]))  # (N, M+1)
    touched = np.any(S <= H, axis=1)
    payoff = np.where(touched, 0.0, np.maximum(S[:, -1] - K, 0.0))
    return np.exp(-r * T) * payoff.mean()

def mc_vanilla_call(S0, K, r, sigma, T, M, N):
    dt = T / M
    Z = np.random.randn(N, M)
    logS = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.exp(np.column_stack([np.full(N, np.log(S0)), logS]))
    payoff = np.maximum(S[:, -1] - K, 0.0)
    return np.exp(-r * T) * payoff.mean()

r = 0.03
K_grid = np.linspace(80, 120, 17)
do_prices = [mc_down_out_call(S0, K, barrier, r, sigma, T, M, 60000) for K in K_grid]
van_prices = [mc_vanilla_call(S0, K, r, sigma, T, M, 60000) for K in K_grid]

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(K_grid, van_prices, color="#1f77b4", lw=2.0, marker="o", ms=4, label="普通看涨 (Vanilla Call)")
ax.plot(K_grid, do_prices, color="#d62728", lw=2.0, marker="s", ms=4, label=f"向下敲出看涨 (H={barrier})")
ax.set_xlabel("执行价 K", fontsize=11)
ax.set_ylabel("期权价格", fontsize=11)
ax.set_title("障碍折价明显：同执行价下敲出看涨比普通看涨便宜（蒙特卡洛 6 万路径）",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "barrier_vs_vanilla.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 3) 自动赎回票据 (Autocallable Note) 收益结构
# ============================================================
# 设定：1 年期，每季度观察；敲出价 100%（=S0），派息 4%/季；
# 到期保本障碍 70%；若到期低于 70% 则按跌幅承担本金损失。
coupon = 0.04
ko_barrier = 1.00      # 敲出价（相对 S0）
pp_barrier = 0.70      # 保本障碍（相对 S0）
obs_dates = [0.25, 0.50, 0.75, 1.00]

# 单路径收益函数
def autocall_payoff(final_ret, ko_hit_quarter):
    """返回票据收益率（含已派息），ko_hit_quarter: 第几季敲出(1..4)，None=未敲出。"""
    if ko_hit_quarter is not None:
        return coupon * ko_hit_quarter           # 敲出：拿对应季度票息，本金返还
    if final_ret >= pp_barrier:
        return coupon * 4                          # 未敲出但高于保本线：拿满票息
    else:
        return final_ret                           # 跌破保本线：承担跌幅损失（票息也亏掉）

# 绘制到期收益曲线（假设未敲出情景，按期末相对收益）
x = np.linspace(-0.40, 0.30, 200)
# 票息累积层：>=70% 时锁定 4 季票息；<70% 时按跌幅
y = np.where(x >= pp_barrier, coupon * 4, x)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(x * 100, y * 100, color="#2ca02c", lw=2.2, label="到期票据收益（未敲出情形）")
ax.axhline(coupon * 4 * 100, color="#1f77b4", ls="--", lw=1.3, label="票息封顶 4×4%=16%")
ax.axhline(0, color="#444", ls=":", lw=1.2)
ax.axvline(pp_barrier * 100, color="#d62728", ls="--", lw=1.5, label=f"保本障碍 70%")
ax.fill_between(x * 100, y * 100, 0, where=(x < pp_barrier), color="#d62728", alpha=0.18)
ax.set_xlabel("标的期末相对收益 (%)", fontsize=11)
ax.set_ylabel("票据收益率 (%)", fontsize=11)
ax.set_title("自动赎回票据：高于保本线吃票息，跌破则承担本金损失（敲出另算）",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=9.5)
ax.grid(True, alpha=0.25)
ax.set_xlim(-40, 30)
plt.tight_layout()
plt.savefig(os.path.join(D, "autocall_payoff.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ exotic-options-structures 配图生成完成：", sorted(os.listdir(D)))
