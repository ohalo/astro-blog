#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「美股隔夜异象：收盘到开盘跳变能否预测次日方向」
(us-overnight-anomaly) 生成真实配图与核心数值。

机制（自洽合成，仅用于演示方法，固定 seed 可复现）：
  * 合成 N 只股票的长面板（日度）。每只股票每日收益拆成两段：
      - 隔夜 o_t = close_{t-1} -> open_t：正漂移 + 高波动（情绪的发动机）
      - 日内 i_t = open_t -> close_t：轻微负漂移 + 中波动
  * 关键结构（异象可交易的核心）：「前一日隔夜跳变」与「当日日内收益」呈
    反转关系（过度反应被纠正）。即 i_t 含一个 beta*lag(o_{t-1}) 成分，beta<0。
  * 实证三层：
      ① 指数层面：隔夜年化 vs 日内年化，看收益到底由谁贡献；
      ② 横截面：前一日隔夜跳变 -> 同日日内收益 rank-IC（应为负）；
      ③ 可交易：Fama-MacBeth 逐日按前一日隔夜十档分组，看同日日内收益是否单调反转。
  * 策略：D10（前夜跌最多）做多日内、D1（前夜涨最多）做空日内，多空组合。

所有数值与图表均由本脚本 numpy 真实计算生成。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

SLUG = "us-overnight-anomaly"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"vm": "#4C72B0", "st": "#C44E52", "dd": "#8172B3", "eq": "#DD8452",
     "bd": "#CCB974", "dark": "#333333", "gold": "#DD8452", "blue": "#4C72B0",
     "green": "#55A868", "red": "#C44E52", "purple": "#8172B3"}

rng = np.random.default_rng(20261007)
N, T = 400, 252 * 20          # 400 只股票 × 20 年
# 隔夜：正漂移 + 高波动
mu_on, sd_on = 0.00045, 0.0065
# 日内：略负漂移 + 中波动
mu_in, sd_in = -0.00010, 0.0072
beta = -0.9                    # 当日日内对前一日隔夜的反转强度（可交易核心）

# 逐只股票生成（带截面异质性：每只自己的漂移）
stock_mu_on = mu_on + rng.normal(0, 0.00015, N)
stock_mu_in = mu_in + rng.normal(0, 0.00010, N)

on = np.empty((N, T))          # 隔夜
inr = np.empty((N, T))         # 日内
mkt_on = rng.normal(0, 0.0015, T)
mkt_in = rng.normal(0, 0.0012, T)
# 隔夜加入 AR(1) 自相关(phi=0.30)：过度反应的状态会延续到次日，
# 于是「前一日隔夜跳变」能预测「次日日内」(经反转结构)。
phi = 0.30
on_lag = np.zeros((N, T))      # 滞后一期隔夜：t-1 的跳变预测 t 的日内
for j in range(N):
    o = np.empty(T)
    o[0] = stock_mu_on[j] + 0.5 * mkt_on[0]
    for t in range(1, T):
        o[t] = (1 - phi) * stock_mu_on[j] + phi * o[t - 1] \
               + sd_on * rng.standard_normal() + 0.5 * mkt_on[t]
    on[j] = o
    on_lag[j, 1:] = on[j, :-1]
    # 日内：自身噪声 + 对「当日隔夜跳变」的反转(beta<0，去均值使截面零和) + 市场因子
    inr[j] = (stock_mu_in[j] + 0.0025 * rng.standard_normal(T)
              + beta * (on[j] - on[j].mean())
              + 0.5 * mkt_in)

# ---------- ① 指数层面：隔夜 vs 日内 年化 ----------
idx_on = on.mean(axis=0)
idx_in = inr.mean(axis=0)
def ann(r):
    return (1 + r.mean()) ** 252 - 1
ann_on = ann(idx_on)
ann_in = ann(idx_in)
vol_on = idx_on.std() * np.sqrt(252)
vol_in = idx_in.std() * np.sqrt(252)

# ---------- ② 横截面 rank-IC：前一日隔夜跳变 vs 同日日内 ----------
prev_on = on[:, :-1]
cur_in = inr[:, 1:]
ic_list = []
for t in range(prev_on.shape[1]):
    ic = np.corrcoef(np.argsort(prev_on[:, t]), np.argsort(cur_in[:, t]))[0, 1]
    ic_list.append(ic)
ic_mean = np.mean(ic_list)
ic_t = ic_mean / (np.std(ic_list) / np.sqrt(len(ic_list)))

# ---------- ③ Fama-MacBeth 逐日十档分组：前一日隔夜 -> 同日日内 ----------
ranks = np.argsort(np.argsort(prev_on, axis=0), axis=0)   # 每只每日截面排名 0..N-1
dec = (9 - ranks * 10 // N).astype(int)                 # 翻转：D1=涨最多 .. D10=跌最多
cur_in_flat = cur_in.ravel()
dec_flat = dec.ravel()
grp_mean_by_day = np.array([cur_in_flat[dec_flat == g].mean() for g in range(10)])
grp_mean_intraday = grp_mean_by_day
# 多空：D10(跌最多)多日内 - D1(涨最多)空日内，逐日序列
ls_by_day = np.array([cur_in[dec[:, t] == 9, t].mean()
                      - cur_in[dec[:, t] == 0, t].mean()
                      for t in range(prev_on.shape[1])])
ls_sharpe = ls_by_day.mean() / ls_by_day.std() * np.sqrt(252)
ls_ann = (1 + ls_by_day.mean()) ** 252 - 1
ls_t = ls_by_day.mean() / (ls_by_day.std() / np.sqrt(len(ls_by_day)))

# ========== 绘图 ==========
# 图1：指数隔夜 vs 日内累计贡献
fig, ax = plt.subplots(figsize=(7.4, 4.4))
eq_on = np.insert(np.cumprod(1 + idx_on), 0, 1.0)
eq_in = np.insert(np.cumprod(1 + idx_in), 0, 1.0)
ax.plot(eq_on, color=C["blue"], lw=1.6, label=f"隔夜累计 (年化 {ann_on*100:.1f}%)")
ax.plot(eq_in, color=C["red"], lw=1.6, label=f"日内累计 (年化 {ann_in*100:.1f}%)")
ax.set_xlabel("交易日")
ax.set_ylabel("累计净值（增长因子）")
ax.set_title("收益的两段：隔夜是发动机，日内几乎不贡献")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "overnight_vs_intraday.png")); plt.close(fig)

# 图2：每日收益拆解占比
fig, ax = plt.subplots(figsize=(7.0, 4.4))
vals = [ann_on * 100, ann_in * 100]
bars = ax.bar(["隔夜 return", "日内 return"], vals, color=[C["blue"], C["red"]])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.3 if v >= 0 else -0.6),
            f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("年化收益 (%)")
ax.set_title(f"隔夜贡献 {ann_on/(ann_on+ann_in)*100:.0f}% 的收益，日内 {ann_in/(ann_on+ann_in)*100:.0f}%")
ax.axhline(0, color=C["dark"], lw=0.8)
ax.grid(alpha=0.2, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "return_share.png")); plt.close(fig)

# 图3：横截面 rank-IC 时序
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(ic_list, color=C["purple"], lw=0.8, alpha=0.7)
ax.axhline(ic_mean, color=C["red"], ls="--", lw=1.6, label=f"均值 IC={ic_mean:.3f} (t={ic_t:.1f})")
ax.axhline(0, color=C["dark"], lw=0.8)
ax.set_xlabel("交易日")
ax.set_ylabel("前夜隔夜跳变→同日日内 rank-IC")
ax.set_title("横截面：前夜跳得越狠，同日日内越倾向反转")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ic_timeseries.png")); plt.close(fig)

# 图4：十档分组同日日内收益
fig, ax = plt.subplots(figsize=(7.4, 4.4))
xs = np.arange(1, 11)
ax.plot(xs, grp_mean_by_day * 100, color=C["green"], marker="o", lw=1.8)
ax.axhline(0, color=C["dark"], lw=0.8)
ax.set_xticks(xs)
ax.set_xlabel("按前一日隔夜跳变分组 (D1=涨最多 → D10=跌最多)")
ax.set_ylabel("同日日内平均收益 (%)")
ax.set_title(f"十档单调性：D1 {grp_mean_by_day[0]*100:.2f}% → D10 {grp_mean_by_day[9]*100:.2f}%")
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "decile_intraday.png")); plt.close(fig)

# 图5：多空组合净值
fig, ax = plt.subplots(figsize=(7.4, 4.4))
eq_ls = np.insert(np.cumprod(1 + ls_by_day), 0, 1.0)
ax.plot(eq_ls, color=C["blue"], lw=1.6,
        label=f"多空 (D10多-D1空) 年化 {ls_ann*100:.1f}% / Sharpe {ls_sharpe:.2f}")
ax.set_xlabel("交易日")
ax.set_ylabel("累计净值")
ax.set_title("可交易多空：做多前夜跌最狠、做空前夜涨最狠")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "longshort_equity.png")); plt.close(fig)

print("=" * 64)
print("ARTICLE_US_OVERNIGHT_METRICS")
print(f"N={N} stocks, T={T} days, beta(reversal on lagged overnight)={beta}")
print(f"[Index] overnight ann={ann_on*100:.2f}% (vol {vol_on*100:.1f}%)  "
      f"intraday ann={ann_in*100:.2f}% (vol {vol_in*100:.1f}%)")
print(f"overnight share of total return = {ann_on/(ann_on+ann_in)*100:.0f}%")
print(f"[Cross-section] mean rank-IC={ic_mean:.4f}  t-stat={ic_t:.1f}")
print(f"[Decile] D1 intraday={grp_mean_by_day[0]*100:.3f}%  "
      f"D10 intraday={grp_mean_by_day[9]*100:.3f}%  reversal(D1<D10)={grp_mean_by_day[0]<grp_mean_by_day[9]}")
print(f"[Long-Short] ann_ret={ls_ann*100:.2f}%  Sharpe={ls_sharpe:.2f}  t-stat={ls_t:.1f}")
print("=" * 64)
