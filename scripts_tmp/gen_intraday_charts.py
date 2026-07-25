#!/usr/bin/env python3
"""日内动量最后一小时效应 配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/intraday-momentum-last-hour"
os.makedirs(OUT, exist_ok=True)

# ---------- 合成日内数据 ----------
# 每天 48 个 5 分钟 bar（4 小时交易），前 6 个 bar = 开盘半小时，后 12 个 bar = 最后一小时
DAYS = 1250  # ~5年
BARS = 48
FIRST = 6    # 开盘 30 分钟
LAST = 12    # 收盘前 60 分钟

daily_vol = 0.012
bar_vol = daily_vol / np.sqrt(BARS)

# U 型日内波动
u_shape = 1.0 + 0.8 * (np.cos(np.linspace(0, 2 * np.pi, BARS)) + 1) / 2
u_shape = u_shape / np.sqrt((u_shape ** 2).mean())

BETA = 0.10  # 开盘收益对尾盘收益的传导强度（植入的效应）

first_rets, last_rets, all_bars = [], [], []
for d in range(DAYS):
    # 当天信息冲击：开盘段承载隔夜信息
    info = rng.standard_normal() * 0.006
    bars = rng.standard_normal(BARS) * bar_vol * u_shape
    bars[:FIRST] += info / FIRST
    r_first = bars[:FIRST].sum()
    # 植入日内动量：尾盘方向部分延续开盘方向（对冲者/被动资金尾盘同向交易）
    drift = BETA * r_first
    bars[-LAST:] += drift / LAST
    r_last = bars[-LAST:].sum()
    first_rets.append(r_first)
    last_rets.append(r_last)
    all_bars.append(bars)

first_rets = np.array(first_rets)
last_rets = np.array(last_rets)
all_bars = np.array(all_bars)

corr = np.corrcoef(first_rets, last_rets)[0, 1]
slope = np.polyfit(first_rets, last_rets, 1)[0]
tstat = corr * np.sqrt((DAYS - 2) / (1 - corr ** 2))
print(f"corr(first30, last60) = {corr:.3f}, slope = {slope:.3f}, t = {tstat:.2f}")

# ---------- 策略：开盘半小时方向 → 尾盘一小时持仓 ----------
COST = 0.0002  # 单边 2bp
sig = np.sign(first_rets)
strat = sig * last_rets - 2 * COST
bh_last = last_rets.copy()

def stats(r, per_year=252):
    ann = r.mean() * per_year
    sh = r.mean() / (r.std() + 1e-12) * np.sqrt(per_year)
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    win = (r > 0).mean()
    return ann, sh, dd, win

a1, s1, d1, w1 = stats(strat)
a2, s2, d2, w2 = stats(bh_last)
print(f"策略: 年化 {a1:.2%}, Sharpe {s1:.2f}, MDD {d1:.2%}, 胜率 {w1:.1%}")
print(f"被动持有尾盘: 年化 {a2:.2%}, Sharpe {s2:.2f}, MDD {d2:.2%}, 胜率 {w2:.1%}")

# 无效应对照
first0, last0 = [], []
for d in range(DAYS):
    bars = rng.standard_normal(BARS) * bar_vol * u_shape
    bars[:FIRST] += rng.standard_normal() * 0.006 / FIRST
    first0.append(bars[:FIRST].sum())
    last0.append(bars[-LAST:].sum())
first0, last0 = np.array(first0), np.array(last0)
strat0 = np.sign(first0) * last0 - 2 * COST
a0, s0, d0, w0 = stats(strat0)
print(f"无效应对照: 年化 {a0:.2%}, Sharpe {s0:.2f}, 胜率 {w0:.1%}")

# ---------- 图1：散点 + 回归 ----------
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(first_rets * 100, last_rets * 100, s=8, alpha=0.4, color="#1f77b4")
xs = np.linspace(first_rets.min(), first_rets.max(), 50)
b, a = np.polyfit(first_rets, last_rets, 1), None
ax.plot(xs * 100, (b[0] * xs + b[1]) * 100, color="#d62728", lw=2,
        label=f"回归斜率 {b[0]:.3f}，corr={corr:.3f}，t={tstat:.1f}")
ax.axhline(0, color="gray", lw=0.6)
ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("开盘前 30 分钟收益 (%)")
ax.set_ylabel("收盘前 60 分钟收益 (%)")
ax.set_title("日内动量：开盘半小时的方向被尾盘部分延续")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/intraday_scatter.png", dpi=110)
plt.close()

# ---------- 图2：条件收益柱状 ----------
qs = np.quantile(first_rets, [0.2, 0.4, 0.6, 0.8])
groups = np.digitize(first_rets, qs)
means = [last_rets[groups == g].mean() * 1e4 for g in range(5)]
fig, ax = plt.subplots(figsize=(9, 5))
cols = ["#d62728" if m < 0 else "#2ca02c" for m in means]
ax.bar(range(5), means, color=cols, alpha=0.85)
ax.set_xticks(range(5))
ax.set_xticklabels(["最弱开盘\nQ1", "Q2", "Q3", "Q4", "最强开盘\nQ5"])
ax.set_ylabel("尾盘一小时平均收益 (bp)")
ax.set_title("按开盘半小时收益分五组：尾盘平均收益单调递增")
ax.axhline(0, color="gray", lw=0.8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/intraday_quintile_bars.png", dpi=110)
plt.close()

# ---------- 图3：策略净值对比 ----------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.cumprod(1 + strat), label=f"日内动量策略（Sharpe {s1:.2f}）", color="#1f77b4", lw=1.5)
ax.plot(np.cumprod(1 + bh_last), label=f"每天被动持有尾盘（Sharpe {s2:.2f}）", color="#7f7f7f", lw=1.2)
ax.plot(np.cumprod(1 + strat0), label=f"无效应对照组（Sharpe {s0:.2f}）", color="#d62728", lw=1.2, ls="--")
ax.set_xlabel("交易日")
ax.set_ylabel("净值")
ax.set_title("尾盘动量策略净值：信号真实存在时赚钱，效应不存在时被成本磨平")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/intraday_strategy_equity.png", dpi=110)
plt.close()

# ---------- 图4：传导强度 & 成本敏感性 ----------
betas = [0.0, 0.03, 0.06, 0.10, 0.15]
costs = [0.0, 0.0002, 0.0005, 0.001]
heat = np.zeros((len(betas), len(costs)))
for i, bta in enumerate(betas):
    rng2 = np.random.default_rng(100 + i)
    fr, lr = [], []
    for d in range(DAYS):
        bars = rng2.standard_normal(BARS) * bar_vol * u_shape
        bars[:FIRST] += rng2.standard_normal() * 0.006 / FIRST
        rf = bars[:FIRST].sum()
        bars[-LAST:] += bta * rf / LAST
        fr.append(rf)
        lr.append(bars[-LAST:].sum())
    fr, lr = np.array(fr), np.array(lr)
    for j, c in enumerate(costs):
        r = np.sign(fr) * lr - 2 * c
        heat[i, j] = r.mean() / (r.std() + 1e-12) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=-1.5, vmax=1.5)
ax.set_xticks(range(len(costs)))
ax.set_xticklabels([f"{c*1e4:.0f}bp" for c in costs])
ax.set_yticks(range(len(betas)))
ax.set_yticklabels([f"β={b}" for b in betas])
ax.set_xlabel("单边交易成本")
ax.set_ylabel("传导强度 β")
ax.set_title("Sharpe 热力图：效应强度与成本的生死线")
for i in range(len(betas)):
    for j in range(len(costs)):
        ax.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, label="年化 Sharpe")
plt.tight_layout()
plt.savefig(f"{OUT}/intraday_beta_cost_heatmap.png", dpi=110)
plt.close()
print("saved:", os.listdir(OUT))
