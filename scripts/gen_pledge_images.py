#!/usr/bin/env python3
"""股权质押风险因子 - 配图生成"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
OUT = "/Users/halo/workspace/astro-blog/public/images/equity-pledge-risk-factor"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---------- 横截面面板: 500 股 x 36 月 ----------
N, T = 500, 36
# 质押比例(控股股东质押股数/持股数): 双峰 —— 一堆不质押, 一堆重度质押
no_pledge = rng.random(N) < 0.35
pledge_ratio = np.where(no_pledge, 0.0, np.clip(rng.beta(2, 2.2, N), 0, 1))

# 市场月收益(含一段下跌 regime 触发平仓螺旋)
mkt = rng.normal(0.026, 0.040, T)
mkt[14:19] = np.array([-0.04, -0.07, -0.09, -0.04, -0.03])  # 5个月熊市

# 个股收益: beta*mkt + 质押拖累(非线性: 熊市中高质押被放大) + 噪声
beta = rng.normal(1.0, 0.25, N)
ret = np.zeros((N, T))
for t in range(T):
    bear = mkt[t] < -0.03
    # 质押惩罚: 平时小幅折价, 熊市中平仓风险非线性爆发
    drag = -0.002 * pledge_ratio + (mkt[t] * 0.9 * pledge_ratio**2 if bear else 0)
    ret[:, t] = beta * mkt[t] + drag + rng.normal(0, 0.075, N)

# ---------- 图1: 质押比例分布 ----------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
ax.hist(pledge_ratio[pledge_ratio > 0] * 100, bins=36, color="#4C72B0", alpha=0.85, edgecolor="white")
ax.axvline(50, color="#C0392B", ls="--", lw=1.5)
ax.text(51, ax.get_ylim()[1]*0.85, "50%：预警关注线", color="#C0392B", fontsize=10)
ax.axvline(80, color="#8E44AD", ls="--", lw=1.5)
ax.text(81, ax.get_ylim()[1]*0.7, "80%：几乎无补仓能力", color="#8E44AD", fontsize=10)
ax.set_xlabel("控股股东质押比例 (%)")
ax.set_ylabel("公司数")
ax.set_title(f"质押比例分布（{int((~no_pledge).sum())} 家有质押 / 全样本 {N} 家）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pledge_ratio_distribution.png")
plt.close(fig)

# ---------- 图2: 五分组月均收益 (全期 vs 熊市) ----------
q = pd.qcut(pledge_ratio + rng.normal(0, 1e-6, N), 5, labels=False)  # 加微噪声打破0堆叠
bear_idx = mkt < -0.03
labels = ["Q1 最低", "Q2", "Q3", "Q4", "Q5 最高"]
full_m = [ret[q == g].mean() * 100 for g in range(5)]
bear_m = [ret[q == g][:, bear_idx].mean() * 100 for g in range(5)]

x = np.arange(5)
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
ax.bar(x - 0.19, full_m, width=0.38, label="全期月均收益", color="#4C72B0", alpha=0.9)
ax.bar(x + 0.19, bear_m, width=0.38, label="熊市月均收益", color="#C0392B", alpha=0.9)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(x, labels)
ax.set_xlabel("按质押比例五分组")
ax.set_ylabel("月均收益 (%)")
ax.set_title("质押因子的不对称性：平时差别不大，熊市里高质押组被碾压")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/pledge_quintile_returns.png")
plt.close(fig)

# ---------- 图3: 平仓螺旋示意 (单股价格 vs 平仓线) ----------
days = 250
p = np.zeros(days); p[0] = 20.0
warn_line = 20 * 0.55 / 1.6 * 1.6  # 质押价20, 预警线160%, 平仓线140% (质押率50%)
pledge_price = 20.0
loan = pledge_price * 0.5          # 质押率50%
warn = loan * 1.6                  # 预警线 16.0? -> 20*0.5*1.6=16
close_line = loan * 1.4            # 平仓线 14
shock = rng.normal(0, 0.018, days)
trend = np.linspace(0, -0.45, days) / days
extra = np.zeros(days)
for t in range(1, days):
    r = trend[t] + shock[t] + extra[t-1]
    p[t] = p[t-1] * (1 + r)
    # 跌破预警线后, 市场担忧引发额外抛压; 跌破平仓线后强平
    if p[t] < close_line:
        extra[t] = -0.012
    elif p[t] < warn:
        extra[t] = -0.005
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
ax.plot(p, color="#2C3E50", lw=1.8, label="股价")
ax.axhline(warn, color="#E8A33D", ls="--", lw=1.5, label=f"预警线 {warn:.1f} 元（履约保障比例 160%）")
ax.axhline(close_line, color="#C0392B", ls="--", lw=1.5, label=f"平仓线 {close_line:.1f} 元（140%）")
first_warn = int(np.argmax(p < warn)); first_close = int(np.argmax(p < close_line))
if first_close > 0:
    ax.axvspan(first_close, days, color="#C0392B", alpha=0.08)
    ax.annotate("强平抛压加速下跌", xy=(first_close, close_line), xytext=(first_close-95, close_line-3.5),
                arrowprops=dict(arrowstyle="->", color="#C0392B"), color="#C0392B", fontsize=10)
ax.set_xlabel("交易日")
ax.set_ylabel("股价 (元)")
ax.set_title("平仓螺旋：跌破预警线→质押方抛售担忧→跌破平仓线→强平→加速下跌")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pledge_liquidation_spiral.png")
plt.close(fig)

# ---------- 图4: 排除高质押的组合 vs 基准 ----------
# 月度组合: 等权全样本 vs 剔除 Q5(质押>阈值)
port_all = ret.mean(axis=0)
mask_low = q < 4
port_screen = ret[mask_low].mean(axis=0)
nav_all = np.cumprod(1 + port_all)
nav_scr = np.cumprod(1 + port_screen)

def stats(r, per=12):
    ar = (1 + r).prod() ** (per / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(per)
    nav = np.cumprod(1 + r)
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ar, sh, dd

a1, s1, d1 = stats(port_all)
a2, s2, d2 = stats(port_screen)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
mo = np.arange(T)
ax.plot(mo, nav_all, color="#888", lw=1.8, label=f"全样本等权  年化{a1*100:.1f}%  Sharpe {s1:.2f}  MDD {d1*100:.0f}%")
ax.plot(mo, nav_scr, color="#C0392B", lw=1.8, label=f"剔除高质押 Q5  年化{a2*100:.1f}%  Sharpe {s2:.2f}  MDD {d2*100:.0f}%")
ax.axvspan(14, 19, color="#C0392B", alpha=0.06)
ax.text(14.3, ax.get_ylim()[0]*1.02 + 0.02, "熊市段", color="#C0392B", fontsize=10)
ax.set_xlabel("月份")
ax.set_ylabel("净值")
ax.set_title("负面清单用法：不指望它选牛股，指望它躲开尾部")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pledge_screen_nav.png")
plt.close(fig)

# 正文引用数字
print(f"有质押公司数: {int((~no_pledge).sum())}, 中位质押比例: {np.median(pledge_ratio[pledge_ratio>0])*100:.1f}%")
print(f"Q5 全期月均: {full_m[4]:.2f}%, Q1 全期: {full_m[0]:.2f}%")
print(f"Q5 熊市月均: {bear_m[4]:.2f}%, Q1 熊市: {bear_m[0]:.2f}%")
print(f"全样本: 年化{a1*100:.1f}% Sharpe {s1:.2f} MDD {d1*100:.1f}%")
print(f"剔除Q5: 年化{a2*100:.1f}% Sharpe {s2:.2f} MDD {d2*100:.1f}%")
# 熊市月的组合差
bear_diff = (ret[mask_low][:, bear_idx].mean() - ret[:, bear_idx].mean()) * 100
print(f"熊市月剔除组相对全样本月均改善: {bear_diff:.2f}pp")
print(f"spiral: 预警线 {warn:.1f}, 平仓线 {close_line:.1f}, 最低价 {p.min():.2f}")
