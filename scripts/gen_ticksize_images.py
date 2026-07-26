# -*- coding: utf-8 -*-
"""Tick Size 与流动性配图生成（最小报价单位对价差/深度的影响）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/tick-size-liquidity"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(2026)

# ---------------------------------------------------------------
# 简化做市模型：
# 无摩擦均衡半价差 s* 由逆选择+库存成本决定（连续值）
# 实际报价必须取整到 tick 网格：s_quoted = max(tick, ceil(s*/tick)*tick)
# 深度：做市商在约束价差 > 均衡价差时赚超额租金 → 愿意挂更多量
#       depth ∝ base * (s_quoted / s*)^gamma
# ---------------------------------------------------------------

def quoted_spread(s_star, tick):
    return np.maximum(tick, np.ceil(s_star / tick) * tick)

def depth(s_star, tick, base=1000, gamma=1.6):
    sq = quoted_spread(s_star, tick)
    return base * (sq / s_star) ** gamma

# ---------------- 图1：约束价差 vs 均衡价差 ----------------
s_star = np.linspace(0.001, 0.05, 400)  # 均衡半价差（元）
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax = axes[0]
for tick, c in [(0.01, "#3a7ca5"), (0.005, "#e07b39"), (0.001, "#5b9e5b")]:
    ax.plot(s_star * 100, quoted_spread(s_star, tick) * 100, lw=1.8,
            label=f"tick = {tick*100:.1f} 分")
ax.plot(s_star * 100, s_star * 100, "k--", lw=1, label="无摩擦均衡价差")
ax.set_xlabel("均衡半价差（分）")
ax.set_ylabel("实际报价半价差（分）")
ax.set_title("tick 是价差的地板：均衡价差越小，约束越狠")
ax.legend(fontsize=9)

ax = axes[1]
for tick, c in [(0.01, "#3a7ca5"), (0.005, "#e07b39"), (0.001, "#5b9e5b")]:
    binding = quoted_spread(s_star, tick) / s_star
    ax.plot(s_star * 100, binding, lw=1.8, label=f"tick = {tick*100:.1f} 分")
ax.axhline(1, color="k", ls="--", lw=1)
ax.set_xlabel("均衡半价差（分）")
ax.set_ylabel("报价价差 / 均衡价差")
ax.set_ylim(0.9, 6)
ax.set_title("约束倍数：低价差股票被 1 分 tick 撑到均衡的数倍")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/tick-floor-spread.png", dpi=110)
plt.close(fig)

# ---------------- 图2：价差-深度的跷跷板 ----------------
# 横截面：500 只股票，均衡价差分布对数正态
n_stk = 500
s_cross = np.exp(rng.normal(np.log(0.008), 0.7, n_stk))
s_cross = np.clip(s_cross, 0.0008, 0.06)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
ticks_list = [0.001, 0.002, 0.005, 0.01, 0.02]
mean_sq, mean_dep = [], []
for tk in ticks_list:
    mean_sq.append(quoted_spread(s_cross, tk).mean() * 100)
    mean_dep.append(depth(s_cross, tk).mean())
ax = axes[0]
ax.plot([t * 100 for t in ticks_list], mean_sq, "o-", color="#c0504d", lw=1.8)
ax.set_xlabel("tick size（分）")
ax.set_ylabel("横截面平均报价半价差（分）")
ax.set_title("tick 越大，平均价差越宽（对小价差股立竿见影）")
ax2 = axes[1]
ax2.plot([t * 100 for t in ticks_list], mean_dep, "s-", color="#3a7ca5", lw=1.8)
ax2.set_xlabel("tick size（分）")
ax2.set_ylabel("横截面平均最优档深度（股）")
ax2.set_title("但 tick 越大，挂单租金越厚 → 深度越深")
fig.tight_layout()
fig.savefig(f"{OUT}/tick-seesaw.png", dpi=110)
plt.close(fig)

# ---------------- 图3：成本 U 型曲线（小单 vs 大单） ----------------
# 小单成本 ≈ 半价差；大单成本 ≈ 半价差 + 冲击成本 η·Q/depth
# （tick 小 → 深度薄 → 冲击大；tick 大 → 价差宽。两头都贵）
fig, ax = plt.subplots(figsize=(9, 4.6))
tick_grid = np.linspace(0.0005, 0.025, 60)
small_cost, big_cost = [], []
ORDER_SMALL, ORDER_BIG = 500, 20000
ETA = 0.001  # 冲击系数：每单位 Q/depth 的价格让步（元）
for tk in tick_grid:
    sq = quoted_spread(s_cross, tk)
    dep = depth(s_cross, tk)
    # 小单：一档内解决
    small = sq
    # 大单：可见深度越薄，冲击越大
    big = sq + ETA * ORDER_BIG / dep
    small_cost.append(small.mean() * 100)
    big_cost.append(big.mean() * 100)
ax.plot(tick_grid * 100, small_cost, lw=1.8, color="#5b9e5b", label=f"小单（{ORDER_SMALL} 股）成本")
ax.plot(tick_grid * 100, big_cost, lw=1.8, color="#c0504d", label=f"大单（{ORDER_BIG:,} 股）成本")
imin = int(np.argmin(big_cost))
ax.axvline(tick_grid[imin] * 100, color="k", ls=":", lw=1.1)
ax.annotate(f"大单最优 tick ≈ {tick_grid[imin]*100:.2f} 分",
            xy=(tick_grid[imin] * 100, big_cost[imin]),
            xytext=(tick_grid[imin] * 100 + 0.35, big_cost[imin] + 0.35),
            fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_xlabel("tick size（分）")
ax.set_ylabel("平均单边成本（分/股）")
ax.set_title("小单希望 tick 越小越好，大单的成本是 U 型：深度也值钱")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/tick-cost-ushape.png", dpi=110)
plt.close(fig)

# ---------------- 图4：Tick 缩小实验（事件前后对比） ----------------
# 模拟一次「tick 从 1 分降到 0.1 分」的制度变化：
# 高价差股（不受约束）几乎无变化；低价差股价差大降、深度大降、报价更新频率上升
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
groups = ["受约束组\n(均衡价差 < 1 tick)", "不受约束组\n(均衡价差 > 3 tick)"]
constrained = s_cross < 0.01
uncon = s_cross > 0.03

sq_before = quoted_spread(s_cross, 0.01)
sq_after = quoted_spread(s_cross, 0.001)
dep_before = depth(s_cross, 0.01)
dep_after = depth(s_cross, 0.001)

ax = axes[0]
x = np.arange(2)
w = 0.32
sp_b = [sq_before[constrained].mean() * 100, sq_before[uncon].mean() * 100]
sp_a = [sq_after[constrained].mean() * 100, sq_after[uncon].mean() * 100]
ax.bar(x - w / 2, sp_b, w, label="tick=1 分", color="#3a7ca5")
ax.bar(x + w / 2, sp_a, w, label="tick=0.1 分", color="#e07b39")
for i in range(2):
    chg = (sp_a[i] / sp_b[i] - 1) * 100
    ax.text(i + w / 2, sp_a[i] + 0.02, f"{chg:+.0f}%", ha="center", fontsize=10)
chg_con = (sp_a[0] / sp_b[0] - 1) * 100
chg_unc = (sp_a[1] / sp_b[1] - 1) * 100
ax.set_xticks(x, groups)
ax.set_ylabel("平均报价半价差（分）")
ax.set_title(f"缩小 tick：受约束组价差 {chg_con:.0f}%，不受约束组仅 {chg_unc:.0f}%")
ax.legend(fontsize=9)

ax = axes[1]
dp_b = [dep_before[constrained].mean(), dep_before[uncon].mean()]
dp_a = [dep_after[constrained].mean(), dep_after[uncon].mean()]
ax.bar(x - w / 2, dp_b, w, label="tick=1 分", color="#3a7ca5")
ax.bar(x + w / 2, dp_a, w, label="tick=0.1 分", color="#e07b39")
for i in range(2):
    chg = (dp_a[i] / dp_b[i] - 1) * 100
    ax.text(i + w / 2, dp_a[i] + 30, f"{chg:+.0f}%", ha="center", fontsize=10)
dchg_con = (dp_a[0] / dp_b[0] - 1) * 100
ax.set_xticks(x, groups)
ax.set_ylabel("平均最优档深度（股）")
ax.set_title(f"代价：受约束组深度坍缩 {dchg_con:.0f}%，大单更难成交")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/tick-pilot-experiment.png", dpi=110)
plt.close(fig)

print("done:", os.listdir(OUT))
