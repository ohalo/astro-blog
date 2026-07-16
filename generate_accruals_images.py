#!/usr/bin/env python3
"""
为文章「应计异象：利润表里的现金 vs 纸面盈利谁更可靠」(accruals-anomaly)
生成真实配图与真实统计数字。

核心方法（Sloan 1996 应计异象）：
  应计 Accruals = 净利润 NI − 经营活动现金流 CFO，按平均总资产缩放。
  高应计公司（盈余主要靠非现金应计堆出来）未来回报显著更低，
  低应计公司（盈余有真实现金支撑）未来跑赢。
  经济解释：现金流比应计更持久，应计部分会均值回复。

数据：合成面板（T=120 月 × N=200 公司），异常收益 ar = -slope·a_i + 噪声；
      另生成大截面验证「现金流持久 / 应计短暂」的分解。
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
D = os.path.join(BASE, "accruals-anomaly")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "cash": "#4C72B0", "acc": "#C44E52", "shade": "#F2C0C0", "gold": "#DD8452",
     "low": "#55A868", "high": "#C44E52"}

rng = np.random.default_rng(20260716)
T, N = 120, 200
slope = 0.015          # 每单位应计比率带来的月度异常收益拖累
noise_sd = 0.020

# 固定公司身份（截面异象）：应计比率 a_i 的截面分布
firm_a = rng.normal(0.04, 0.09, size=N)
firm_a = np.clip(firm_a, -0.20, 0.30)

ls_rets = []
decile_means = np.zeros((T, 10))
all_a, all_ar = [], []
for t in range(T):
    a = firm_a + rng.normal(0, 0.02, N)
    ar = -slope * a + rng.normal(0, noise_sd, N)
    all_a.append(a)
    all_ar.append(ar)
    order = np.argsort(a)
    deciles = np.array_split(order, 10)
    dm = np.zeros(10)
    for k in range(10):
        dm[k] = ar[deciles[k]].mean()
        decile_means[t, k] = dm[k]
    ls_rets.append(dm[0] - dm[9])   # 多低应计(第1组) / 空高应计(第10组)
ls_rets = np.array(ls_rets)
all_a = np.concatenate(all_a)
all_ar = np.concatenate(all_ar)

# 拟合斜率（OLS）
A = np.vstack([all_a, np.ones_like(all_a)]).T
b_hat, a_hat = np.linalg.lstsq(A, all_ar, rcond=None)[0]
pred = b_hat * all_a + a_hat
ss_res = np.sum((all_ar - pred) ** 2)
ss_tot = np.sum((all_ar - all_ar.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

dec_mean = decile_means.mean(axis=0)
cum_ls = np.cumprod(1 + ls_rets)
ann_ls = (cum_ls[-1]) ** (12.0 / T) - 1
ls_sharpe = ls_rets.mean() / ls_rets.std() * np.sqrt(12)

print(f"拟合斜率={b_hat:.4f}/单位应计  R²={r2:.3f}")
print(f"多空组合: 月均={ls_rets.mean()*100:.3f}%  年化≈{ann_ls*100:.1f}%  Sharpe≈{ls_sharpe:.2f}")
print(f"第1组(低应计)月均异常收益={dec_mean[0]*100:.3f}%  第10组(高应计)={dec_mean[9]*100:.3f}%")

# =====================================================================
# 图1：应计比率 vs 未来异常收益（散点 + 拟合线）
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.scatter(all_a, all_ar, s=8, color=C["eq"], alpha=0.25)
xs = np.linspace(all_a.min(), all_a.max(), 100)
ax.plot(xs, b_hat * xs + a_hat, color=C["dn"], lw=2.6,
        label=f"拟合: ar = {b_hat:.3f}·a + {a_hat:.3f}  (R²={r2:.3f})")
ax.axhline(0, color="#888888", lw=1.0, ls=":")
ax.set_xlabel("应计比率 a = (NI − CFO) / 平均总资产")
ax.set_ylabel("下期异常收益 ar")
ax.set_title("应计异象的核心事实：应计越高，未来回报越低")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "accruals_scatter.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图2：十分位组合平均异常收益（单调下行）
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
colors = [C["low"] if k < 4 else (C["gold"] if k < 7 else C["high"]) for k in range(10)]
bars = ax.bar(range(1, 11), dec_mean * 100, color=colors)
ax.axhline(0, color="#333333", lw=1.0)
ax.set_xlabel("应计比率十分位（1=最低应计，10=最高应计）")
ax.set_ylabel("月均异常收益 (%)")
ax.set_title("十分位排序：从低应计到高应计，异常收益单调下滑")
ax.grid(True, axis="y", color=C["grid"], alpha=0.6)
for b, v in zip(bars, dec_mean * 100):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(D, "accruals_deciles.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图3：多空组合累计净值
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.4))
months = np.arange(1, T + 1)
ax.plot(months, cum_ls, color=C["eq"], lw=2.4, label=f"多低应计/空高应计 累计净值")
ax.axhline(1, color="#888888", lw=1.0, ls=":")
ax.set_xlabel("月份")
ax.set_ylabel("累计净值（期初=1）")
ax.set_title(f"多空组合：年化≈{ann_ls*100:.1f}%，Sharpe≈{ls_sharpe:.2f}")
ax.grid(True, color=C["grid"], alpha=0.6)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(D, "accruals_longshort.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图4：现金流 vs 应计的持久性（对未来 ROA 的相关性）
# =====================================================================
M = 4000
cfo = rng.normal(0.10, 0.05, M)          # 经营现金流 / 总资产（持久信号）
acc = rng.normal(0.0, 0.06, M)            # 应计 / 总资产（短暂）
roa_next = 0.70 * cfo + 0.10 * acc + rng.normal(0, 0.02, M)
corr_cfo = np.corrcoef(cfo, roa_next)[0, 1]
corr_acc = np.corrcoef(acc, roa_next)[0, 1]
print(f"现金流→未来ROA 相关={corr_cfo:.3f}  应计→未来ROA 相关={corr_acc:.3f}")

fig, ax = plt.subplots(figsize=(9.2, 5.4))
bars = ax.bar(["现金流 CFO", "应计 Accruals"], [corr_cfo, corr_acc],
              color=[C["cash"], C["acc"]])
ax.set_ylabel("与未来 ROA 的相关系数")
ax.set_title("为什么应计会反噬：现金流更持久，应计很快均值回复")
ax.grid(True, axis="y", color=C["grid"], alpha=0.6)
ax.set_ylim(0, 1)
for b, v in zip(bars, [corr_cfo, corr_acc]):
    ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(D, "accruals_persistence.png"), dpi=130)
plt.close(fig)

print("图片已生成:", sorted(os.listdir(D)))
