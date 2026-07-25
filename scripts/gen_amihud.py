#!/usr/bin/env python3
"""Amihud 非流动性因子文章配图生成：合成横截面 + 分组回测"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/amihud-illiquidity-factor"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- 合成市场：800 只股票 x 15 年月度 ----------
n_stocks, n_years = 800, 15
n_months = n_years * 12
days_per_month = 21

# 每只股票的"真实流动性水平"（对数正态分布的日均成交额，百万元）
log_dollar_vol = rng.normal(4.0, 1.5, n_stocks)   # ln(百万元)，中位数 ~55 百万
dollar_vol = np.exp(log_dollar_vol)               # 日均成交额（百万）

# 流动性溢价设定：非流动性股票有更高期望收益（年化溢价随 ILLIQ 排名线性上升 0~6%）
illiq_rank = (np.argsort(np.argsort(-log_dollar_vol))) / (n_stocks - 1)  # 成交额越低 rank 越高
annual_premium = 0.02 + 0.10 * illiq_rank         # 年化 2%~12%（含下行拖累前）
mu_month = annual_premium / 12

# 市场因子
mkt = rng.normal(0.007, 0.045, n_months)          # 月度市场收益
beta = rng.normal(1.0, 0.25, n_stocks)
# 非流动性股票在市场下跌时跌更多（流动性 beta）
liq_beta = 0.20 * illiq_rank                       # 下行放大

idio = rng.normal(0, 0.08, (n_months, n_stocks))
ret = np.empty((n_months, n_stocks))
for t in range(n_months):
    down_amp = liq_beta * min(mkt[t], 0)           # 只在下跌月放大
    ret[t] = mu_month + beta * mkt[t] + down_amp + idio[t]

# 日度数据用于计算 ILLIQ：日收益 & 日成交额（成交额有噪声 + 与波动正相关）
# ILLIQ_i = mean( |r_daily| / dollar_volume )，这里直接用月度近似模拟日度
def compute_illiq(month_idx, window=12):
    """滚动 12 个月的 Amihud ILLIQ（合成日度）"""
    lo = max(0, month_idx - window)
    n_days = (month_idx - lo) * days_per_month
    if n_days == 0:
        return None
    # 合成日收益: 月收益缩放 + 噪声
    sig_d = 0.08 / np.sqrt(days_per_month)
    r_d = rng.normal(0, sig_d, (n_days, n_stocks)) + \
          np.repeat(ret[lo:month_idx], days_per_month, axis=0) / days_per_month
    dv_d = dollar_vol[None, :] * np.exp(rng.normal(0, 0.4, (n_days, n_stocks)))
    return np.mean(np.abs(r_d) / dv_d, axis=0) * 1e2   # x100 便于阅读（|r|% per 百万元 x1e2）

# ---------- 分组回测：每月按 ILLIQ 分 5 组 ----------
warmup = 12
n_groups = 5
group_rets = np.full((n_months, n_groups), np.nan)
ls_ret = np.full(n_months, np.nan)
illiq_snapshot = None
for t in range(warmup, n_months):
    illiq = compute_illiq(t)
    if t == warmup:
        illiq_snapshot = illiq.copy()
    q = np.quantile(illiq, np.linspace(0, 1, n_groups + 1))
    for g in range(n_groups):
        mask = (illiq >= q[g]) & (illiq <= q[g + 1]) if g == n_groups - 1 else (illiq >= q[g]) & (illiq < q[g + 1])
        group_rets[t, g] = ret[t, mask].mean()
    ls_ret[t] = group_rets[t, -1] - group_rets[t, 0]

valid = slice(warmup, n_months)
gr = group_rets[valid]
ls = ls_ret[valid]
mk = mkt[valid]

# ---------- 图1：ILLIQ 与市值/成交额的横截面散点 ----------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
ax.scatter(dollar_vol, illiq_snapshot, s=8, alpha=0.5, color="#1f77b4")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("日均成交额（百万元，对数轴）")
ax.set_ylabel("Amihud ILLIQ（对数轴）")
ax.set_title("横截面：ILLIQ 与成交额近似对数线性负相关")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/illiq-cross-section.png"); plt.close(fig)

# ---------- 图2：五组累计净值 ----------
fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
colors = plt.cm.viridis(np.linspace(0.15, 0.9, n_groups))
labels = ["G1 最流动", "G2", "G3", "G4", "G5 最不流动"]
tgrid = np.arange(gr.shape[0]) / 12
ann = []
for g in range(n_groups):
    nav = np.cumprod(1 + gr[:, g])
    a = nav[-1] ** (1 / (len(nav) / 12)) - 1
    ann.append(a)
    ax.plot(tgrid, nav, color=colors[g], lw=1.9, label=f"{labels[g]}（年化 {a*100:.1f}%）")
ax.set_yscale("log")
ax.set_xlabel("年"); ax.set_ylabel("累计净值（对数轴）")
ax.set_title("按 ILLIQ 五分组：非流动组跑赢流动组")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/quintile-nav.png"); plt.close(fig)

# ---------- 图3：多空组合净值 + 回撤 ----------
nav_ls = np.cumprod(1 + ls)
peak = np.maximum.accumulate(nav_ls)
dd = nav_ls / peak - 1
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=150, sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
axes[0].plot(tgrid, nav_ls, color="#d62728", lw=2)
ls_ann = nav_ls[-1] ** (1/(len(ls)/12)) - 1
ls_sharpe = ls.mean() / ls.std() * np.sqrt(12)
axes[0].set_title(f"G5-G1 多空组合：年化 {ls_ann*100:.1f}%，Sharpe {ls_sharpe:.2f}")
axes[0].set_ylabel("累计净值"); axes[0].grid(alpha=0.3)
axes[1].fill_between(tgrid, dd * 100, 0, color="#d62728", alpha=0.4)
axes[1].set_ylabel("回撤（%）"); axes[1].set_xlabel("年"); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/long-short-nav.png"); plt.close(fig)

# ---------- 图4：多空收益 vs 市场收益（流动性风险暴露） ----------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
down = mk < 0
ax.scatter(mk[~down]*100, ls[~down]*100, s=18, alpha=0.6, label="市场上涨月", color="#2ca02c")
ax.scatter(mk[down]*100, ls[down]*100, s=18, alpha=0.6, label="市场下跌月", color="#d62728")
# 分段回归线
for mask, c in [(~down, "#2ca02c"), (down, "#d62728")]:
    b, a = np.polyfit(mk[mask], ls[mask], 1)
    xs = np.linspace(mk[mask].min(), mk[mask].max(), 10)
    ax.plot(xs*100, (a + b*xs)*100, color=c, lw=2, ls="--")
ax.set_xlabel("市场月度收益（%）"); ax.set_ylabel("G5-G1 多空月度收益（%）")
ax.set_title("流动性溢价的阴暗面：市场下跌时多空组合跟着跌")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/downside-exposure.png"); plt.close(fig)

# ---------- 数字 ----------
print("=== 关键数字 ===")
for g in range(n_groups):
    print(f"{labels[g]}: 年化 {ann[g]*100:.1f}%")
print(f"多空: 年化 {ls_ann*100:.1f}%, Sharpe {ls_sharpe:.2f}, 最大回撤 {dd.min()*100:.1f}%")
b_dn, _ = np.polyfit(mk[down], ls[down], 1)
b_up, _ = np.polyfit(mk[~down], ls[~down], 1)
print(f"下跌月 beta: {b_dn:.2f}, 上涨月 beta: {b_up:.2f}")
t_stat = ls.mean() / (ls.std() / np.sqrt(len(ls)))
print(f"多空月均 {ls.mean()*100:.2f}%, t={t_stat:.2f}, N={len(ls)}")
print(f"ILLIQ 中位数 {np.median(illiq_snapshot):.4f}, 90分位/10分位比 {np.percentile(illiq_snapshot,90)/np.percentile(illiq_snapshot,10):.0f}x")
