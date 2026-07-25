#!/usr/bin/env python3
"""融券余额因子文章配图：short interest 横截面分组回测（合成面板）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/short-interest-factor"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(526)

# ---------- 合成面板：600 股 x 120 月 ----------
n, T = 600, 120
# 每股基础特征
size = rng.normal(0, 1, n)                 # 规模因子暴露
# short interest ratio (融券余额/流通市值)，右偏
si = np.exp(rng.normal(-3.2, 0.9, n))      # 中位数 ~4%
si_rank = (np.argsort(np.argsort(si))) / (n - 1)  # 越高越被做空

# 真实设定：高 short interest 预示未来负超额（做空者是聪明钱），但效应集中在极端组
# 年化 alpha：从 +1%（低SI）到 -5%（高SI），非线性——只有最高分位显著负
alpha_annual = 0.01 - 0.06 * (si_rank ** 2)
mu_month = alpha_annual / 12

mkt = rng.normal(0.008, 0.042, T)
beta = 1.0 + 0.2 * rng.normal(0, 1, n)
idio_sd = 0.05

# short interest 随时间缓慢变化
si_t = np.zeros((T, n))
si_t[0] = si
for t in range(1, T):
    si_t[t] = np.clip(si_t[t-1] * np.exp(rng.normal(0, 0.15, n)), 0.001, 0.5)

ret = np.zeros((T, n))
for t in range(T):
    rank_t = (np.argsort(np.argsort(si_t[t]))) / (n - 1)
    a_t = 0.02 - 0.09 * (rank_t ** 1.6)
    ret[t] = a_t/12 + beta * mkt[t] + rng.normal(0, idio_sd, n)

# ---------- 分组回测 ----------
Q = 5
grp_ret = np.zeros((T, Q))
for t in range(T):
    order = np.argsort(si_t[t])   # 低 SI 到 高 SI
    bins = np.array_split(order, Q)
    for q in range(Q):
        grp_ret[t, q] = ret[t, bins[q]].mean()

ann = (1 + grp_ret.mean(axis=0)) ** 12 - 1

# ---------- 图1：short interest 分布 ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(si * 100, bins=50, color="#4c72b0", alpha=0.8, edgecolor="white")
ax.axvline(np.median(si)*100, color="#d62728", ls="--", lw=2,
           label=f"中位数 {np.median(si)*100:.1f}%")
ax.set_title("融券余额占流通市值比（SI ratio）的横截面分布", fontsize=13, fontweight="bold")
ax.set_xlabel("融券余额 / 流通市值 (%)")
ax.set_ylabel("股票数")
ax.legend(fontsize=11)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/si-distribution.png", dpi=130)
plt.close()

# ---------- 图2：五分组年化收益 ----------
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2ca02c", "#8bc34a", "#c9c9c9", "#ff9800", "#d62728"]
bars = ax.bar([f"Q{q+1}" for q in range(Q)], ann*100, color=colors, alpha=0.85)
for i, b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15,
            f"{ann[i]*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("融券余额五分组年化收益：低SI(Q1) → 高SI(Q5)", fontsize=13, fontweight="bold")
ax.set_ylabel("年化收益 (%)")
ax.set_xlabel("← 融券余额低    融券余额高 →")
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/quintile-returns.png", dpi=130)
plt.close()

# ---------- 图3：多空组合净值（做多低SI，做空高SI） ----------
ls_ret = grp_ret[:, 0] - grp_ret[:, Q-1]  # Q1 - Q5
nav_ls = np.cumprod(1 + ls_ret)
nav_mkt = np.cumprod(1 + mkt)
short_leg = np.cumprod(1 - grp_ret[:, Q-1])  # 纯做空高SI
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(nav_ls, color="#d62728", lw=2, label="多低SI/空高SI 多空组合")
ax.plot(nav_mkt, color="gray", lw=1.5, ls="--", label="市场基准")
ax.set_title("融券余额多空组合净值曲线", fontsize=13, fontweight="bold")
ax.set_xlabel("月份")
ax.set_ylabel("累计净值")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
sharpe = ls_ret.mean()/ls_ret.std()*np.sqrt(12)
ax.text(0.02, 0.95, f"多空 Sharpe = {sharpe:.2f}\n月均 = {ls_ret.mean()*100:.2f}%",
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", fc="#fff3cd", alpha=0.9))
plt.tight_layout()
plt.savefig(f"{OUT}/long-short-nav.png", dpi=130)
plt.close()

# ---------- 图4：牛熊市分解——空头腿在下跌市更有效 ----------
up_mask = mkt > 0
down_mask = ~up_mask
q5_up = grp_ret[up_mask, Q-1].mean() - grp_ret[up_mask, 0].mean()
q5_down = grp_ret[down_mask, Q-1].mean() - grp_ret[down_mask, 0].mean()
fig, ax = plt.subplots(figsize=(9, 5))
xs = ["上涨月", "下跌月"]
vals = [(grp_ret[up_mask,0].mean()-grp_ret[up_mask,Q-1].mean())*100,
        (grp_ret[down_mask,0].mean()-grp_ret[down_mask,Q-1].mean())*100]
b = ax.bar(xs, vals, color=["#2ca02c", "#d62728"], alpha=0.85, width=0.5)
for i, bb in enumerate(b):
    ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+0.02,
            f"{vals[i]:.2f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_title("多空组合月均收益：上涨市 vs 下跌市", fontsize=13, fontweight="bold")
ax.set_ylabel("多空月均收益 (%)")
ax.axhline(0, color="black", lw=0.8)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/regime-split.png", dpi=130)
plt.close()

print("annual by quintile (%):", np.round(ann*100, 2))
print("LS Sharpe=%.2f monthly=%.3f%%" % (sharpe, ls_ret.mean()*100))
print("up-month LS=%.3f%% down-month LS=%.3f%%" % (vals[0], vals[1]))
print("DONE short-interest images ->", OUT)
