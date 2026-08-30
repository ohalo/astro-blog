#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成动量崩溃（momentum crash）文章配图（合成数据）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "STHeiti"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/momentum-crash-tail-risk"
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c6fbb"
RED = "#d1495b"
GREEN = "#2a9d8f"
GRAY = "#6c757d"
ORANGE = "#e09f3e"
PURPLE = "#8e6bbf"

rng = np.random.default_rng(2026)

# ============================================================
# 1. 合成数据：高动量（此前赢家）组合 + 低动量（此前输家）组合
# ============================================================
T = 800
n_stock = 100

# 基础因子结构：动量因子 m_t，市场因子 f_mkt
f_mkt = rng.normal(0.0003, 0.01, T)
m_t = rng.normal(0.0008, 0.008, T)

# 高动量组合：正暴露于动量因子，低动量组合负暴露
beta_winner_mom = 1.0
beta_loser_mom = -1.0
# 两者都对市场有暴露
beta_mkt = 1.0

# 特质噪声
id_winner = rng.normal(0, 0.012, T)
id_loser = rng.normal(0, 0.012, T)

r_winner = beta_mkt * f_mkt + beta_winner_mom * m_t + id_winner
r_loser = beta_mkt * f_mkt + beta_loser_mom * m_t + id_loser

# 动量组合 = 多赢家 / 空输家
r_wml = r_winner - r_loser

# 市场崩盘窗口：第 550-590 天，市场暴跌 + 赢家/输家暴露反转（输家大幅反弹）
crash_start, crash_end = 550, 590
crash_mask = np.zeros(T, dtype=bool)
crash_mask[crash_start:crash_end] = True

# 市场崩盘期间：市场大跌，且此前输家（低动量）因为估值低/空头回补而暴力反弹
f_mkt[crash_mask] = rng.normal(-0.006, 0.02, crash_end - crash_start)
# 输家在崩盘期大幅反弹（short squeeze）
r_loser[crash_mask] = r_loser[crash_mask] + rng.normal(0.006, 0.015, crash_end - crash_start)
# 赢家在崩盘期回调
r_winner[crash_mask] = r_winner[crash_mask] + rng.normal(-0.003, 0.015, crash_end - crash_start)
# 重新计算 WML
r_wml = r_winner - r_loser

# ============================================================
# 图1：WML 累计收益 + 崩盘窗口标注
# ============================================================
cum_wml = np.cumprod(1 + r_wml) - 1

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cum_wml, color=BLUE, lw=1.2, label="WML 多空组合累计收益")
ax.axvspan(crash_start, crash_end, color=RED, alpha=0.18, label="市场崩盘窗口")
ax.set_xlabel("交易日")
ax.set_ylabel("累计收益")
ax.set_title("动量组合：长期上行，但崩盘窗口出现陡峭回撤")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/wml_cumulative.png", dpi=110)
plt.close()

# ============================================================
# 图2：崩盘期赢家 vs 输家日收益对比
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
axes[0].plot(r_winner, color=GREEN, lw=0.7, label="赢家组合（高动量）")
axes[0].axvspan(crash_start, crash_end, color=RED, alpha=0.15)
axes[0].set_ylabel("日收益")
axes[0].set_title("崩盘期：此前赢家回调，此前输家暴力反弹（空头回补）")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(r_loser, color=ORANGE, lw=0.7, label="输家组合（低动量）")
axes[1].axvspan(crash_start, crash_end, color=RED, alpha=0.15)
axes[1].set_xlabel("交易日")
axes[1].set_ylabel("日收益")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/winner_loser_crash.png", dpi=110)
plt.close()

# ============================================================
# 图3：崩盘窗口 vs 正常窗口的 WML 日收益分布
# ============================================================
normal = r_wml[~crash_mask]
crash = r_wml[crash_mask]

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-0.06, 0.06, 40)
ax.hist(normal, bins=bins, alpha=0.6, color=BLUE, label="正常窗口", density=True)
ax.hist(crash, bins=bins, alpha=0.75, color=RED, label="崩盘窗口", density=True)
ax.axvline(0, color="black", lw=0.8, alpha=0.4)
ax.set_xlabel("WML 日收益")
ax.set_ylabel("密度")
ax.set_title("崩盘窗口的 WML 分布明显左移、左尾更肥")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/distribution_shift.png", dpi=110)
plt.close()

# ============================================================
# 图4：滚动 60 日波动率 + 左尾 VaR/ES
# ============================================================
win = 60
roll_vol = np.array([r_wml[i:i+win].std() for i in range(T - win + 1)])
roll_vol_full = np.full(T, np.nan)
roll_vol_full[win-1:] = roll_vol

# 崩盘窗口的 VaR(5%) 与 ES(5%)
crash_sorted = np.sort(crash)
var5 = np.percentile(crash, 5)
es5 = crash_sorted[:int(0.05*len(crash_sorted))].mean()

normal_sorted = np.sort(normal)
var5_normal = np.percentile(normal, 5)
es5_normal = normal_sorted[:int(0.05*len(normal_sorted))].mean()

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(roll_vol_full, color=PURPLE, lw=1.2)
ax.axvspan(crash_start, crash_end, color=RED, alpha=0.15)
ax.set_xlabel("交易日")
ax.set_ylabel("滚动 60 日波动率")
ax.set_title("崩盘窗口滚动波动率急剧上升（左尾风险集中释放）")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/rolling_vol.png", dpi=110)
plt.close()

print("=== 崩盘窗口统计 ===")
print(f"崩盘期 WML 累计回撤: {np.prod(1 + crash) - 1:.2%}")
print(f"正常窗口 5% VaR: {var5_normal:.3%}  ES: {es5_normal:.3%}")
print(f"崩盘窗口 5% VaR: {var5:.3%}  ES: {es5:.3%}")
print(f"崩盘窗口 WML 平均日收益: {crash.mean():.4%} vs 正常: {normal.mean():.4%}")
print("图片已保存到", OUT)
