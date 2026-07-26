#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为两篇文章生成真实配图与统计数字：
1) 零收益天数流动性度量（Lesmond-Ogden-Trzcinka 1999 Zeros measure）
2) 换手率流动性因子（Datar-Naik-Radcliffe 1998 截面选股）

全部数字由本脚本真实计算（numpy/matplotlib），无占位符。
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
D1 = os.path.join(BASE, "zeros-liquidity-measure")
D2 = os.path.join(BASE, "turnover-liquidity-factor")
os.makedirs(D1, exist_ok=True)
os.makedirs(D2, exist_ok=True)

rng = np.random.default_rng(42)

# ============================================================
# Part 1: Zeros 零收益天数度量
# ============================================================
# 模拟三类股票的日收益：真实"有效收益"~N(0, sigma)，但存在交易成本带 [ -c, c ]，
# LOT 模型：|真实收益| < c 时价格不动 -> 观测收益为 0（做市商/投资者不交易）
T = 756  # 三年日线

def simulate_stock(sigma, cost, T, rng):
    """LOT 式模拟：真实信息收益 r*, 交易成本带 c；|r*|<c -> 观测 0"""
    r_true = rng.normal(0, sigma, T)
    r_obs = np.where(np.abs(r_true) < cost, 0.0, r_true - np.sign(r_true) * cost)
    return r_true, r_obs

groups = {
    "大盘蓝筹": dict(sigma=0.015, cost=0.0005, n=30),
    "中盘": dict(sigma=0.020, cost=0.0030, n=30),
    "小盘/微盘": dict(sigma=0.025, cost=0.0110, n=30),
}

zeros_by_group = {}
all_zeros = []
all_amihud = []
all_cost = []
for gname, p in groups.items():
    zs = []
    for i in range(p["n"]):
        c = p["cost"] * rng.uniform(0.6, 1.6)
        sig = p["sigma"] * rng.uniform(0.8, 1.2)
        r_true, r_obs = simulate_stock(sig, c, T, rng)
        zero_ratio = np.mean(r_obs == 0)
        zs.append(zero_ratio)
        # 成交额: 与成本负相关
        base_vol = 1e9 / (1 + c * 3000)
        vol = base_vol * rng.lognormal(0, 0.5, T)
        vol = np.where(r_obs == 0, vol * 0.25, vol)  # 零收益日成交萎缩
        with np.errstate(divide="ignore"):
            illiq = np.abs(r_obs) / vol
        amihud = np.mean(illiq[np.abs(r_obs) > 0]) * 1e9
        all_zeros.append(zero_ratio)
        all_amihud.append(amihud)
        all_cost.append(c)
    zeros_by_group[gname] = np.array(zs)

print("=" * 60)
print("【文章1】Zeros 零收益天数度量")
for g, zs in zeros_by_group.items():
    print(f"  {g}: 平均 Zeros = {zs.mean()*100:.1f}%  (范围 {zs.min()*100:.1f}%~{zs.max()*100:.1f}%)")

# Spearman 相关: zeros vs Amihud vs 真实成本
from scipy import stats
rho_za, _ = stats.spearmanr(all_zeros, all_amihud)
rho_zc, _ = stats.spearmanr(all_zeros, all_cost)
rho_ac, _ = stats.spearmanr(all_amihud, all_cost)
print(f"  Spearman(Zeros, Amihud) = {rho_za:.3f}")
print(f"  Spearman(Zeros, 真实成本) = {rho_zc:.3f}")
print(f"  Spearman(Amihud, 真实成本) = {rho_ac:.3f}")

# 图1: 三类股票 Zeros 分布（箱线图）
fig, ax = plt.subplots(figsize=(9, 6))
data = [zeros_by_group[g] * 100 for g in groups]
bp = ax.boxplot(data, tick_labels=list(groups.keys()), patch_artist=True, widths=0.5)
colors = ["#4C72B0", "#DD8452", "#C44E52"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("零收益天数占比 Zeros (%)")
ax.set_title("三类股票的 Zeros 度量：流动性越差，价格越常「装死」")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D1, "zeros-cross-section.png"), dpi=110)
plt.close()

# 图2: LOT 机制示意 —— 真实收益 vs 观测收益
sig_demo, c_demo = 0.02, 0.012
r_true_d = rng.normal(0, sig_demo, 3000)
r_obs_d = np.where(np.abs(r_true_d) < c_demo, 0.0, r_true_d - np.sign(r_true_d) * c_demo)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(r_true_d * 100, bins=80, color="#4C72B0", alpha=0.8)
axes[0].axvspan(-c_demo * 100, c_demo * 100, color="red", alpha=0.15, label="交易成本带 ±c")
axes[0].set_title("真实信息收益：连续分布")
axes[0].set_xlabel("收益 (%)")
axes[0].legend()
axes[1].hist(r_obs_d * 100, bins=80, color="#C44E52", alpha=0.8)
axes[1].set_title(f"观测收益：{np.mean(r_obs_d==0)*100:.0f}% 的天数被压成 0")
axes[1].set_xlabel("收益 (%)")
for a in axes:
    a.grid(alpha=0.3)
plt.suptitle("LOT 机制：交易成本带把小幅信息「吃掉」，制造零收益日", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(D1, "zeros-lot-mechanism.png"), dpi=110, bbox_inches="tight")
plt.close()
print(f"  演示股票: 成本带 ±{c_demo*100:.1f}%, 零收益日占比 {np.mean(r_obs_d==0)*100:.1f}%")

# 图3: Zeros vs Amihud 散点（log）
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(np.array(all_zeros) * 100, all_amihud, c="#55A868", alpha=0.7, s=45, edgecolors="k", linewidths=0.4)
ax.set_yscale("log")
ax.set_xlabel("Zeros (%)")
ax.set_ylabel("Amihud 非流动性 (log 轴)")
ax.set_title(f"Zeros 与 Amihud：Spearman 相关 {rho_za:.2f}，量的不完全是一件事")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D1, "zeros-vs-amihud.png"), dpi=110)
plt.close()

# 图4: 危机情景 —— 滚动 Zeros 上升
T2 = 504
r_normal = rng.normal(0, 0.018, T2)
cost_path = np.full(T2, 0.004)
cost_path[250:330] = np.linspace(0.004, 0.020, 80)  # 危机中成本带扩大
cost_path[330:420] = np.linspace(0.020, 0.006, 90)
r_obs2 = np.where(np.abs(r_normal) < cost_path, 0.0, r_normal)
roll_zeros = np.array([np.mean(r_obs2[max(0, i - 60):i + 1] == 0) for i in range(T2)])
price = 100 * np.cumprod(1 + np.where(r_obs2 == 0, 0, r_obs2) - (cost_path - 0.004) * 0.35)
fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.plot(roll_zeros * 100, color="#C44E52", lw=2, label="60日滚动 Zeros (%)")
ax1.set_ylabel("Zeros (%)", color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax2 = ax1.twinx()
ax2.plot(price, color="#4C72B0", lw=1.5, alpha=0.8, label="价格")
ax2.set_ylabel("价格", color="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#4C72B0")
ax1.axvspan(250, 420, color="gray", alpha=0.12)
ax1.set_title("流动性危机情景：成本带扩大 → Zeros 从 {:.0f}% 飙到 {:.0f}%".format(
    roll_zeros[:250].mean() * 100, roll_zeros[300:400].max() * 100))
ax1.grid(alpha=0.3)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
plt.tight_layout()
plt.savefig(os.path.join(D1, "zeros-crisis.png"), dpi=110)
plt.close()
print(f"  危机情景: 平时滚动Zeros均值 {roll_zeros[:250].mean()*100:.1f}%, 危机峰值 {roll_zeros[300:400].max()*100:.1f}%")

# ============================================================
# Part 2: 换手率流动性因子（截面选股）
# ============================================================
# 模拟 200 只股票 x 60 个月。低换手率股票有流动性溢价（Datar-Naik-Radcliffe）。
N, M = 200, 60
# 每只股票的"特征换手率"（月换手，%）
log_turn = rng.normal(np.log(15), 0.9, N)  # 月换手中位数 ~15%
turnover = np.exp(log_turn)
# 流动性溢价：低换手组年化多 ~4-6%；加上噪声
liq_premium_annual = 0.05
# 每只股票的预期月收益 = 基准 + 溢价 * (换手率分位的负向暴露)
turn_rank = stats.rankdata(turnover) / N  # 0~1
mu_month = 0.007 + liq_premium_annual / 12 * (1 - 2 * turn_rank)  # 低换手 -> 高 mu
beta = rng.normal(1.0, 0.25, N)
mkt = rng.normal(0.007, 0.045, M)
rets = np.zeros((M, N))
for t in range(M):
    rets[t] = mu_month - 0.007 + beta * mkt[t] + rng.normal(0, 0.08, N)

# 十分位组合
deciles = np.digitize(turn_rank, np.linspace(0, 1, 11)[1:-1])
dec_ret = np.array([rets[:, deciles == d].mean(axis=1) for d in range(10)])  # 10 x M
dec_ann = dec_ret.mean(axis=1) * 12 * 100
ls = dec_ret[0] - dec_ret[9]  # 低换手 - 高换手
ls_ann = ls.mean() * 12 * 100
ls_sharpe = ls.mean() / ls.std() * np.sqrt(12)
ls_t = ls.mean() / (ls.std() / np.sqrt(M))
print("=" * 60)
print("【文章2】换手率流动性因子")
print(f"  D1(最低换手) 年化 {dec_ann[0]:.1f}% vs D10(最高换手) 年化 {dec_ann[9]:.1f}%")
print(f"  多空组合: 年化 {ls_ann:.1f}%, Sharpe {ls_sharpe:.2f}, t-stat {ls_t:.2f}")
print(f"  换手率分布: 中位数 {np.median(turnover):.1f}%/月, P10 {np.percentile(turnover,10):.1f}%, P90 {np.percentile(turnover,90):.1f}%")

# 图1: 换手率十分位年化收益
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(1, 11), dec_ann, color=plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 10)), edgecolor="k", linewidth=0.5)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
ax.set_xlabel("换手率十分位（D1 最低 → D10 最高）")
ax.set_ylabel("年化收益 (%)")
ax.set_title(f"换手率分组：低换手组年化 {dec_ann[0]:.1f}%，高换手组 {dec_ann[9]:.1f}%")
ax.axhline(0, color="k", lw=0.8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D2, "turnover-deciles.png"), dpi=110)
plt.close()

# 图2: 多空净值曲线
eq_ls = np.cumprod(1 + ls)
eq_d1 = np.cumprod(1 + dec_ret[0])
eq_d10 = np.cumprod(1 + dec_ret[9])
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(eq_d1, label=f"D1 低换手（多头）", color="#55A868", lw=2)
ax.plot(eq_d10, label=f"D10 高换手（空头腿）", color="#C44E52", lw=2)
ax.plot(eq_ls, label=f"多空组合 (年化 {ls_ann:.1f}%, Sharpe {ls_sharpe:.2f})", color="#4C72B0", lw=2.5)
ax.set_xlabel("月份")
ax.set_ylabel("净值")
ax.set_title("低换手 − 高换手 多空组合：流动性溢价的截面收割")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D2, "turnover-longshort.png"), dpi=110)
plt.close()

# 图3: 换手率 vs 市值混淆 —— 控制市值前后
log_cap = -0.55 * (log_turn - log_turn.mean()) + rng.normal(0, 0.8, N)  # 换手与市值负相关? 实际小盘换手高
# 现实：A股小盘换手更高 -> 正相关性反转。这里做"换手与市值负相关于收益"演示
# 用回归剥离市值后的换手率残差再分组
resid_turn = log_turn - np.polyval(np.polyfit(log_cap, log_turn, 1), log_cap)
rr = stats.rankdata(resid_turn) / N
dec2 = np.digitize(rr, np.linspace(0, 1, 11)[1:-1])
dec2_ret = np.array([rets[:, dec2 == d].mean(axis=1) for d in range(10)])
dec2_ann = dec2_ret.mean(axis=1) * 12 * 100
ls2_ann = (dec2_ret[0] - dec2_ret[9]).mean() * 12 * 100
print(f"  市值中性化后: D1-D10 多空年化 {ls2_ann:.1f}% (原始 {ls_ann:.1f}%)")
fig, ax = plt.subplots(figsize=(10, 6))
w = 0.38
ax.bar(np.arange(1, 11) - w / 2, dec_ann, w, label="原始换手率分组", color="#4C72B0", alpha=0.85)
ax.bar(np.arange(1, 11) + w / 2, dec2_ann, w, label="市值中性化后分组", color="#DD8452", alpha=0.85)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
ax.set_ylabel("年化收益 (%)")
ax.set_title(f"中性化检验：剥离市值后多空收益 {ls_ann:.1f}% → {ls2_ann:.1f}%")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D2, "turnover-size-neutral.png"), dpi=110)
plt.close()

# 图4: 成本侵蚀 —— 空头腿高换手股的做空成本
costs_bps = [0, 10, 20, 30, 50]
net_ann = []
monthly_turnover_pct = 0.35  # 每月组合调仓换手 35%
for cb in costs_bps:
    net = ls - monthly_turnover_pct * 2 * cb / 1e4
    net_ann.append(net.mean() * 12 * 100)
print("  成本敏感性: " + ", ".join(f"{c}bp→{v:.1f}%" for c, v in zip(costs_bps, net_ann)))
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(costs_bps, net_ann, "o-", color="#C44E52", lw=2.5, markersize=9)
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xlabel("单边交易成本 (bp)")
ax.set_ylabel("多空组合净年化收益 (%)")
ax.set_title("成本侵蚀：流动性因子的悖论——赚的就是难交易的钱")
for c, v in zip(costs_bps, net_ann):
    ax.annotate(f"{v:.1f}%", (c, v), textcoords="offset points", xytext=(8, 8))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D2, "turnover-cost-erosion.png"), dpi=110)
plt.close()

print("完成，全部图片已生成")
print("D1:", os.listdir(D1))
print("D2:", os.listdir(D2))
