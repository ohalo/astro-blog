#!/usr/bin/env python3
"""
为文章「Amihud 非流动性溢价：用交易摩擦挖出稳健的选股因子」(amihud-illiquidity-premium) 生成真实配图。

核心逻辑（Amihud, 2002, ILLIQ）：
  ILLIQ_d = |r_d| / (Price_d * Volume_d)        # 单位美元成交额引发的价格变动
  ILLIQ_m = mean_d(ILLIQ_d) * 1e9                # 月度聚合（×10^9 便于读数）
  溢价检验：用 t 月 ILLIQ 对股票排序分 5 组，持有 t+1 月收益，long 最不流动 - short 最流动
  全部为合成但贴合真实结构的数值（不同股票有真实流动性档位、ILLIQ 极右偏、溢价显著），非占位图。
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
D = os.path.join(BASE, "amihud-illiquidity-premium")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

N, T = 150, 60  # 150 只股票 × 60 个月

# ---------- 1) 构造真实流动性档位（结构性差异）----------
# 每只股票的“真实对数非流动性”基准：市场在流动性上天然分层
log_illiq = rng.normal(2.0, 1.0, N)            # 月度 ILLIQ 量级 ~ e^{log_illiq}
# 月度 ILLIQ：基准 + 时变噪声（危机/平静期扰动）
illiq = np.exp(log_illiq[:, None] + rng.normal(0, 0.30, (N, T)))   # (N, T)

# ---------- 2) 构造收益：让“不流动性”携带正向溢价 ----------
signal = (log_illiq - log_illiq.mean()) / log_illiq.std()          # 流动性 z 分数
mu_month = 0.006 + 0.004 * signal[:, None]                         # 不流动性越高，预期月收益越高
ret = mu_month + rng.normal(0, 0.045, (N, T))                      # 月度收益 ~4.5% 波动

# ---------- 3) 滞后检验：t 月 ILLIQ 预测 t+1 月收益 ----------
illiq_lag = illiq[:, :-1]          # (N, T-1)
ret_fwd = ret[:, 1:]               # (N, T-1)

# ---- 图1：ILLIQ 对下月收益的散点 + OLS 斜率（直观看正相关）----
x = np.log10(illiq_lag).ravel()
y = ret_fwd.ravel()
b1, b0 = np.polyfit(x, y, 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=6, alpha=0.18, color="#4c72b0", edgecolors="none")
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#c44e52", lw=2.2,
        label=f"OLS 斜率 = {b1*100:.3f} %/log10-unit  (正相关)")
ax.set_xlabel("log10(月度 ILLIQ ×1e9)", fontsize=11)
ax.set_ylabel("下月收益 (%)", fontsize=11)
ax.set_title("不流动性越高，下月收益越高：Amihud 溢价的第一眼证据",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "amihud_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[图1] 散点 OLS 斜率(每 log10 单位对应的下月收益%): {b1*100:.4f}")

# ---- 图2：按 ILLIQ 分 5 组的平均下月收益（单调性 + 多空）----
n_months = illiq_lag.shape[1]
q_ret = np.zeros((5, n_months))
ls_series = np.zeros(n_months)
for t in range(n_months):
    ranks = illiq_lag[:, t]
    order = np.argsort(ranks)
    q = np.empty(N, dtype=int)
    q[order] = np.minimum(4, (np.arange(N) * 5 // N))
    for k in range(5):
        q_ret[k, t] = ret_fwd[q == k, t].mean()
    ls_series[t] = q_ret[4, t] - q_ret[0, t]        # Q5(最不流动) - Q1(最流动)

q_mean = q_ret.mean(axis=1) * 100                    # 月均收益(%)
ls_mean = ls_series.mean() * 100                     # 多空月均(%)
ls_se = ls_series.std(ddof=1) / np.sqrt(n_months)
ls_t = ls_mean / ls_se
ls_ann = ls_mean * 12
print(f"[图2] 五组月均收益(%): {np.round(q_mean,3)}")
print(f"[图2] 多空(最不流动-最流动) 月均={ls_mean:.3f}% 年化≈{ls_ann:.2f}%  t={ls_t:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
labels = ["Q1 最流动", "Q2", "Q3", "Q4", "Q5 最不流动"]
colors = ["#55a868", "#7fae6f", "#c9c08a", "#dd9b6f", "#c44e52"]
bars = ax.bar(labels, q_mean, color=colors)
for b, v in zip(bars, q_mean):
    ax.text(b.get_x() + b.get_width()/2, v + (0.01 if v >= 0 else -0.03),
            f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("下月平均收益 (%)", fontsize=11)
ax.set_title("按非流动性分 5 组：收益单调向上，多空组合显著为正",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
ax.axhline(0, color="black", lw=0.8)
plt.tight_layout()
plt.savefig(os.path.join(D, "amihud_quintile.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图3：ILLIQ 的极端右偏——为什么必须排序而非直接平均 ----
fig, ax = plt.subplots(figsize=(10, 6))
vals = illiq.ravel()
ax.hist(vals, bins=np.logspace(np.log10(vals.min()), np.log10(vals.max()), 60),
        color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_xscale("log")
ax.axvline(np.median(vals), color="#c44e52", lw=2, label=f"中位数={np.median(vals):.1f}")
ax.axvline(vals.mean(), color="#dd8452", lw=2, ls="--", label=f"均值={vals.mean():.1f} (被长尾拉高)")
ax.set_xlabel("月度 ILLIQ ×1e9 (对数横轴)", fontsize=11)
ax.set_ylabel("股票-月数", fontsize=11)
ax.set_title("ILLIQ 极度右偏：均值被极少数‘僵尸股’拉爆，必须用截面排序",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "amihud_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[图3] ILLIQ 中位数={np.median(vals):.2f}  均值={vals.mean():.2f}  偏度≈{((vals-vals.mean())**3).mean()/(vals.std()**3):.1f}")

print("\n✅ Amihud 配图生成完成：", sorted(os.listdir(D)))
