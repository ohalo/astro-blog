#!/usr/bin/env python3
"""
为文章「波动率锥：用历史分位给期权定价与择时」(volatility-cone) 生成真实配图。
数据：用 Heston 式随机波动率模型模拟 10 年日度指数序列（真实计算，非占位图）。
图表：
  1. vol_cone.png          波动率锥：各期限滚动已实现波动的历史分位带 + 当前点位
  2. vol_percentile_series.png  60日已实现波动的滚动分位时序 + 极端区带（买卖信号区）
  3. vol_iv_vs_rv.png      隐含 vs 已实现（期限结构对照）
  4. vol_strategy_equity.png    波动率择时策略净值 vs 买入持有波动率溢价
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "volatility-cone")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

# ============================================================
# 1) 模拟随机波动率价格序列（Heston 式 Euler 离散）
# ============================================================
N = 252 * 10                      # 10 年日度
dt = 1.0 / 252
mu = 0.07                         # 漂移
v0 = 0.04
kappa, theta, xi, rho = 4.0, 0.04, 0.30, -0.60
S = np.zeros(N); v = np.zeros(N)
S[0], v[0] = 100.0, v0
for t in range(1, N):
    z1 = np.random.randn()
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()
    v[t] = max(v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(max(v[t-1], 1e-6) * dt) * z2, 1e-5)
    r = (mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1
    # 偶发崩盘冲击（市场跌得比涨快，给短波动制造真实尾部亏损）
    if np.random.rand() < 0.004:
        r += -np.random.uniform(0.06, 0.14)
    S[t] = S[t-1] * np.exp(r)
ret = np.diff(np.log(S)) * 100.0          # 日收益（%）

# ============================================================
# 2) 波动率锥：各期限滚动已实现波动 + 历史分位带
# ============================================================
windows = [10, 20, 30, 60, 90, 120, 180]
pcts = [5, 25, 50, 75, 95]
cone = {}                              # window -> array of rolling annualized vol
for w in windows:
    rv = np.full(N - 1, np.nan)
    for i in range(w, N - 1):
        rv[i] = np.std(ret[i - w + 1:i + 1], ddof=1) * np.sqrt(252)
    cone[w] = rv

band = {p: [] for p in pcts}
current = []                            # 当前（最后一点）各期限已实现波动
for w in windows:
    series = cone[w]
    valid = series[~np.isnan(series)]
    for p in pcts:
        band[p].append(np.percentile(valid, p))
    current.append(valid[-1])

fig, ax = plt.subplots(figsize=(11, 6.0))
colors = {"5": "#d62728", "25": "#ff7f0e", "50": "#1f77b4", "75": "#2ca02c", "95": "#9467bd"}
labels = {"5": "5% 分位", "25": "25% 分位", "50": "中位数", "75": "75% 分位", "95": "95% 分位"}
for p in pcts:
    ax.plot(windows, band[p], color=colors[str(p)], lw=2.0 if p in (50,) else 1.5,
            label=labels[str(p)])
# 填充 25-75 分位带
ax.fill_between(windows, band[25], band[75], color="#1f77b4", alpha=0.10)
# 当前点位
ax.plot(windows, current, "ko", ms=8, label="当前已实现波动")
ax.plot(windows, current, "k--", lw=1.0, alpha=0.5)
ax.set_xlabel("滚动窗口（交易日）", fontsize=11)
ax.set_ylabel("年化波动率 (%)", fontsize=11)
ax.set_title("波动率锥：把「正常波动区间」画成随期限展开的喇叭", fontsize=12.5, fontweight="bold")
ax.set_xticks(windows)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vol_cone.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 3) 60日已实现波动的滚动分位时序 + 极端区带
# ============================================================
w = 60
rv60 = np.full(N - 1, np.nan)
for i in range(w, N - 1):
    rv60[i] = np.std(ret[i - w + 1:i + 1], ddof=1) * np.sqrt(252)
look = 252                              # 用过去 1 年估计分布
pct_series = np.full(N - 1, np.nan)
for i in range(w + look, N - 1):
    hist = rv60[i - look:i + 1]
    hist = hist[~np.isnan(hist)]
    pct_series[i] = (hist < rv60[i]).mean() * 100.0     # 当前波动在历史中的分位

idx = np.arange(N - 1)
mask = ~np.isnan(pct_series)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(idx[mask], pct_series[mask], color="#1f77b4", lw=1.3, label="60日已实现波动分位")
ax.axhline(90, color="#d62728", ls="--", lw=1.4, label="90 分位（卖波动区）")
ax.axhline(10, color="#2ca02c", ls="--", lw=1.4, label="10 分位（买波动区）")
# 着色极端区
ax.fill_between(idx[mask], 90, 100, color="#d62728", alpha=0.08)
ax.fill_between(idx[mask], 0, 10, color="#2ca02c", alpha=0.08)
# 标记信号点
sig_hi = mask & (pct_series >= 90)
sig_lo = mask & (pct_series <= 10)
ax.scatter(idx[sig_hi], pct_series[sig_hi], color="#d62728", s=14, zorder=5)
ax.scatter(idx[sig_lo], pct_series[sig_lo], color="#2ca02c", s=14, zorder=5)
ax.set_xlabel("交易日（近 10 年）", fontsize=11)
ax.set_ylabel("分位 (%)", fontsize=11)
ax.set_title("60日已实现波动的历史分位：突破 90 / 跌破 10 即触发极值信号", fontsize=12.5, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vol_percentile_series.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 4) 隐含 vs 已实现（期限结构对照）
# ============================================================
# 模拟隐含波动率：在 30 日已实现波动上加一个均值回复的正溢价（波动率风险溢价）
rv30 = np.full(N - 1, np.nan)
for i in range(30, N - 1):
    rv30[i] = np.std(ret[i - 30 + 1:i + 1], ddof=1) * np.sqrt(252)
premium = 0.030 + 0.010 * np.sin(np.arange(N - 1) / 80.0)   # 结构溢价（均值回复）
iv30 = rv30 + premium + np.random.randn(N - 1) * 0.003
iv30 = np.clip(iv30, 0.01, None)

# 取末段 252 天做期限结构对照
term_windows = [10, 20, 30, 60, 90, 120, 180]
rv_term, iv_term = [], []
rv_end = np.full(N - 1, np.nan)
for tw in term_windows:
    s = np.full(N - 1, np.nan)
    for i in range(tw, N - 1):
        s[i] = np.std(ret[i - tw + 1:i + 1], ddof=1) * np.sqrt(252)
    rv_term.append(s[-1])
iv_term.append(iv30[-1]) if False else None
# IV 期限结构：用不同窗口的「已实现+溢价」近似
iv_term = [rv_term[i] + premium[-1] * (1.0 + 0.3 * np.sin(term_windows[i] / 40.0)) for i in range(len(term_windows))]

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(term_windows, rv_term, "o-", color="#1f77b4", lw=2, ms=7, label="已实现波动（期限结构）")
ax.plot(term_windows, iv_term, "s-", color="#d62728", lw=2, ms=7, label="隐含波动（期限结构）")
ax.set_xlabel("期限（交易日）", fontsize=11)
ax.set_ylabel("波动率 (%)", fontsize=11)
ax.set_title("隐含 vs 已实现：IV 整体高于 RV 即「波动率风险溢价」", fontsize=12.5, fontweight="bold")
ax.set_xticks(term_windows)
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vol_iv_vs_rv.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 5) 波动率锥作为风险叠加层（vol overlay）的择时回测
# ============================================================
# 最稳妥的用法：把波动率锥当「风险叠加层」。
#   规则：60 日已实现波动分位 > 90（恐慌/风暴区）→ 降仓至 0（持现金）；否则满仓。
#   对比基准：始终满仓持有底层资产。
#   逻辑：极端高波动区往往对应崩盘与剧烈回撤，锥把「现在是否危险」量化出来。
r = ret / 100.0                       # 日收益（小数）
start = w + look
strat_nav = [1.0]; bench_nav = [1.0]
for t in range(N - 1):
    danger = (t >= start) and (not np.isnan(pct_series[t])) and (pct_series[t] >= 90)
    x = 0.0 if danger else 1.0
    strat_nav.append(strat_nav[-1] * (1 + x * r[t]))
    bench_nav.append(bench_nav[-1] * (1 + r[t]))
strat_eq = np.array(strat_nav); bench_eq = np.array(bench_nav)

def perf(eq):
    eq = np.array(eq)
    rets = np.diff(eq) / eq[:-1]
    cagr = eq[-1] ** (252.0 / (len(eq) - 1)) - 1
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr, sharpe, mdd

c_s, sh_s, mdd_s = perf(strat_eq)
c_b, sh_b, mdd_b = perf(bench_eq)
print(f"锥叠加层(去危): CAGR={c_s:.2%} Sharpe={sh_s:.2f} MDD={mdd_s:.2%}")
print(f"始终满仓基准: CAGR={c_b:.2%} Sharpe={sh_b:.2f} MDD={mdd_b:.2%}")

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(strat_eq, color="#1f77b4", lw=1.8, label=f"波动率锥叠加层（CAGR={c_s:.1%}, Sharpe={sh_s:.2f}, MDD={mdd_s:.1%}）")
ax.plot(bench_eq, color="#ff7f0e", lw=1.4, alpha=0.8, label=f"始终满仓基准（CAGR={c_b:.1%}, Sharpe={sh_b:.2f}, MDD={mdd_b:.1%}）")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_title("波动率锥风险叠加层：风暴区降仓，显著压低最大回撤", fontsize=12.5, fontweight="bold")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vol_strategy_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ 波动率锥配图生成完成：", sorted(os.listdir(D)))
