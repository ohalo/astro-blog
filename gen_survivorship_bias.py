#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「存活者偏差：你的回测收益被系统性高估了多少」生成真实配图与统计数字。

核心逻辑:
  - 合成面板 N=300 股票 x T=180 月。
  - 每股月收益 r_{i,t} = mu_i + beta*mkt + eps; mu_i 有持久差异。
  - 退市机制: 每年有基础退市概率, 且"过去 12 月累计收益越差"退市概率越高
    (亏损、ST、破产退市), 退市当月额外 -50% 冲击收益, 次月从样本消失。
  - 两个股票池:
      PIT (point-in-time) 池: 含全部股票, 退市股保留到退市当月(含 -50% 冲击), 之后剔除。
      Survivor (存活者) 池: 只保留活到期末的股票, 全程等权回测。
  - 对比等权组合: 年化 / Sharpe / 最大回撤; 存活者池系统性高估。
  - 额外: 逐年退市率 + 退市前 12 月累计收益(负向恶化)。

图片:
  survivor_universe_nav.png  —— PIT 池 vs 存活者池 等权累计净值
  survivor_bias_bar.png      —— 年化/Sharpe/最大回撤 的偏差量柱状图
  survivor_delist.png        —— 逐年退市率 + 退市前 12 月累计收益
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
D = os.path.join(BASE, "survivorship-bias-point-in-time-backtest")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260901)
N, T = 300, 180
beta, sig_eps, sig_idio = 0.95, 0.02, 0.06
base_delist = 0.06   # 每年基础退市率

mkt = rng.normal(0.006, 0.045, T)
mu = rng.normal(0.004, 0.008, N)   # 持久 alpha 差异
eps = rng.normal(0, sig_eps, (T, N)) + rng.normal(0, sig_idio, (T, N))
r = np.zeros((T, N))
for t in range(T):
    r[t] = mu + beta * mkt[t] + eps[t]

# 退市: 每年末根据过去12月收益决定是否退市
alive = np.ones((T, N), dtype=bool)   # alive[t, i] = 股票 i 在 t 月末是否还活着
delist_ret = np.zeros((T, N))
delist_months = {}
past12 = np.zeros(N)
for t in range(12, T, 12):   # 每年
    for i in range(N):
        if not alive[t - 1, i]:
            continue
        # 过去12月累计收益
        past12[i] = np.prod(1 + r[max(0, t - 12):t, i]) - 1
    # 退市概率: 收益越差越可能退市
    p = np.clip(base_delist + np.where(past12 < -0.2, 0.10, 0) + np.where(past12 < -0.4, 0.15, 0), 0, 0.7)
    die = rng.random(N) < p
    for i in range(N):
        if alive[t - 1, i] and die[i]:
            delist_months[i] = t
            r[t, i] += -0.60  # 退市当月 -60% 冲击
            # 之后标记死亡
            alive[t:, i] = False

# 修正: 退市冲击只加在退市当月, 且该月之后不再有收益
r_delisted = r.copy()
# 对每个退市股, 退市月之后收益清零(不参与后续)
for i, tm in delist_months.items():
    r_delisted[tm + 1:, i] = 0.0

# 存活者池: 活到期末的股票
survived = alive[T - 1].copy()

def ew_portfolio(returns, mask):
    """等权组合月收益序列; mask[t,i] = 该股票 t 月是否纳入"""
    rets = []
    for t in range(1, T):
        m = mask[t]
        if m.sum() == 0:
            rets.append(0.0); continue
        rets.append(returns[t, m].mean())
    return np.array(rets)

# PIT: 含全部存续股票(退市月含 -50%)
mask_pit = alive.copy()
ret_pit = ew_portfolio(r_delisted, mask_pit)

# Survivor: 只含存活股
mask_surv = np.zeros((T, N), dtype=bool)
for i in range(N):
    if survived[i]:
        mask_surv[:, i] = True
ret_surv = ew_portfolio(r_delisted, mask_surv)

def stats(rets):
    ann = rets.mean() * 12
    vol = rets.std() * np.sqrt(12)
    sharpe = ann / (vol + 1e-9)
    nav = np.cumprod(1 + rets)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ann, sharpe, mdd

a_pit, s_pit, mdd_pit = stats(ret_pit)
a_surv, s_surv, mdd_surv = stats(ret_surv)

print(f"PIT      ann={a_pit:.2%}  sharpe={s_pit:.2f}  mdd={mdd_pit:.2%}")
print(f"SURVIVOR ann={a_surv:.2%}  sharpe={s_surv:.2f}  mdd={mdd_surv:.2%}")
print(f"Bias(ann): {a_surv-a_pit:.2%}  Bias(sharpe): {s_surv-s_pit:.2f}")

# ===== 图1: 累计净值 =====
nav_pit = np.cumprod(1 + ret_pit); nav_surv = np.cumprod(1 + ret_surv)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(nav_surv, color="#C44E52", lw=1.8, label=f"存活者池 (Sharpe {s_surv:.2f})")
ax.plot(nav_pit, color="#2F4B7C", lw=1.8, label=f"PIT 全历史池 (Sharpe {s_pit:.2f})")
ax.axhline(1, color="black", lw=0.7)
ax.set_title("等权组合累计净值：存活者偏差让曲线系统性抬高", fontsize=13, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("净值（起始=1）")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "survivor_universe_nav.png"), dpi=150, bbox_inches="tight"); plt.close()

# ===== 图2: 偏差量柱状图 =====
labels = ["年化收益", "Sharpe", "最大回撤"]
bias = [a_surv - a_pit, s_surv - s_pit, mdd_surv - mdd_pit]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ["#C44E52" if v > 0 else "#2F4B7C" for v in bias]
b = ax.bar(x, bias, 0.5, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("存活者偏差量：年化与 Sharpe 被高估、回撤被低估", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for bb, vv in zip(b, bias):
    off = 0.0015 if vv >= 0 else -0.002
    ax.text(bb.get_x() + bb.get_width()/2, vv + off,
            f"{vv:+.3f}", ha="center", va="bottom" if vv >= 0 else "top", fontsize=10)
ax.set_ylim(min(bias) - 0.015, max(bias) + 0.02)
plt.tight_layout(); plt.savefig(os.path.join(D, "survivor_bias_bar.png"), dpi=150, bbox_inches="tight"); plt.close()

# ===== 图3: 退市率 + 退市前收益 =====
years = list(range(12, T, 12))
delist_rates = []
pre12_returns = []
for t in years:
    cnt = 0
    for i in delist_months.values():
        if i == t:
            cnt += 1
    delist_rates.append(cnt / N)
# 退市股退市前12月累计收益
pre = []
for i, tm in delist_months.items():
    pre.append(np.prod(1 + r[max(0, tm - 12):tm, i]) - 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
ax1.bar([f"Y{y//12+1}" for y in years], delist_rates, color="#DD8452", width=0.6)
ax1.set_title("逐年退市率（含 ST/破产）", fontsize=12, fontweight="bold")
ax1.set_ylabel("退市率"); ax1.grid(True, alpha=0.3, axis="y")
ax2.hist(pre, bins=20, color="#2F4B7C", alpha=0.85, edgecolor="white")
ax2.axvline(np.mean(pre), color="#C44E52", ls="--", lw=1.5,
            label=f"均值 {np.mean(pre):.1%}")
ax2.set_title("退市股退市前 12 月累计收益分布", fontsize=12, fontweight="bold")
ax2.set_xlabel("退市前 12 月累计收益"); ax2.set_ylabel("股票数")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "survivor_delist.png"), dpi=150, bbox_inches="tight"); plt.close()

print("DONE survivorship-bias images")
