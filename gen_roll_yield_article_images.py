#!/usr/bin/env python3
"""展期收益与期货结构文章配图生成（真实 matplotlib 图表，非占位符）"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

mpl_fonts = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = mpl_fonts
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "public/images/commodity-roll-yield-contango-backwardation"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260828)

# ============ 图1: Contango vs Backwardation 期限结构 ============
print("生成图1: 期限结构 Contango / Backwardation ...")
months = ["当月", "下月", "3月", "6月", "12月", "18月"]
x = np.arange(len(months))

# Contango
contango = np.array([80, 81.5, 83, 85, 88, 91])
# Backwardation
back = np.array([95, 93.5, 92, 90, 87, 84])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
ax1.plot(x, contango, "o-", color="#c0392b", lw=2, ms=7)
ax1.fill_between(x, contango, contango[0], alpha=0.08, color="#c0392b")
ax1.set_title("Contango（升水）：远月 > 近月\n展期 = 高价卖近、低价买远 → 负展期收益", fontsize=10.5)
ax1.set_xticks(x); ax1.set_xticklabels(months); ax1.set_ylabel("期货价格")
ax1.grid(alpha=0.3); ax1.annotate("持续失血", xy=(3, 85), color="#c0392b", fontsize=10, fontweight="bold")

ax2.plot(x, back, "o-", color="#27ae60", lw=2, ms=7)
ax2.fill_between(x, back, back[0], alpha=0.08, color="#27ae60")
ax2.set_title("Backwardation（贴水）：近月 > 远月\n展期 = 高价卖近、低价买远 → 正展期收益", fontsize=10.5)
ax2.set_xticks(x); ax2.set_xticklabels(months); ax2.set_ylabel("期货价格")
ax2.grid(alpha=0.3); ax2.annotate("持续造血", xy=(3, 90), color="#1e7e34", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/contango_backwardation.png", dpi=160, bbox_inches="tight")
plt.close()

# ============ 图2: 展期收益累计（原油 vs 黄金两类结构） ============
print("生成图2: 展期收益累计曲线 ...")
T = 252 * 4
# 原油：长期 contango，现货年化 5%，但展期每年 -8%
spot_oil = np.cumprod(1 + rng.normal(0.05/252, 0.03, T))
roll_oil = np.cumprod(1 + (-0.08/252) * np.ones(T))
oil_total = 1.0 * spot_oil * roll_oil

# 黄金：长期 backwardation，现货年化 4%，展期每年 +3%
spot_gold = np.cumprod(1 + rng.normal(0.04/252, 0.012, T))
roll_gold = np.cumprod(1 + (0.03/252) * np.ones(T))
gold_total = 1.0 * spot_gold * roll_gold

t = np.arange(T)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
ax1.plot(t, spot_oil, color="#888", lw=1.4, label="WTI 现货（假设）")
ax1.plot(t, oil_total, color="#c0392b", lw=2.0, label="原油期货（含展期）")
ax1.set_title("WTI 原油：现货涨，但 Contango 展期把收益吃光", fontsize=11.5)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.plot(t, spot_gold, color="#888", lw=1.4, label="黄金现货（假设）")
ax2.plot(t, gold_total, color="#27ae60", lw=2.0, label="黄金期货（含展期）")
ax2.set_title("黄金：Backwardation 展期额外贡献正收益", fontsize=11.5)
ax2.set_xlabel("交易日"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/roll_yield_paths.png", dpi=160, bbox_inches="tight")
plt.close()

# ============ 图3: 展期收益年化 vs 持有期限（不同结构成本） ============
print("生成图3: 结构成本敏感性 ...")
fig, ax = plt.subplots(figsize=(11, 6))
hold_years = np.array([1, 2, 3, 5, 7, 10])
# 不同年化展期收益
for ann, color, label in [(-8.0, "#c0392b", "强 Contango −8%/年"),
                          (-3.0, "#e67e22", "弱 Contango −3%/年"),
                          (0.0, "#999", "平价 0%/年"),
                          (3.0, "#27ae60", "Backwardation +3%/年")]:
    cum = (1 + ann/100) ** hold_years - 1
    ax.plot(hold_years, cum * 100, "o-", color=color, lw=2, label=label)

ax.axhline(0, color="black", lw=0.8)
ax.set_title("展期收益（年化）随持有年限的累计拖累 / 贡献", fontsize=12.5, fontweight="bold")
ax.set_xlabel("持有年限（年）")
ax.set_ylabel("累计展期收益（%）")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/roll_yield_sensitivity.png", dpi=160, bbox_inches="tight")
plt.close()

print("展期收益配图完成:", os.listdir(OUT))
