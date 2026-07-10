#!/usr/bin/env python3
"""
为文章「信用利差因子：用债券相对价值挖掘 Alpha」(credit-spread-factor) 生成真实配图。
数据：用多因子模拟生成一组（评级 × 行业 × 时间）债券信用利差面板（真实计算，非占位图）。
图表：
  1. cs_distribution.png        各评级信用利差的截面分布（箱线图）
  2. cs_signal_scatter.png      利差 z-score 与未来 3 月超额回报（散点 + 回归）
  3. cs_longshort_equity.png    信用利差「相对价值」多空组合净值 vs 基准
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
D = os.path.join(BASE, "credit-spread-factor")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

# ============================================================
# 1) 模拟信用利差面板：评级档位 + 行业 + 时间演化
# ============================================================
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB"]
# 各评级基线利差（bps）与波动
base_spread = np.array([35, 55, 80, 110, 150, 200, 260, 360], dtype=float)
vol_spread = np.array([8, 12, 18, 25, 35, 50, 70, 110], dtype=float)
N_BONDS = 240
N_MONTHS = 120
rng = np.random.default_rng(20260711)

rating_idx = rng.integers(0, len(RATINGS), size=N_BONDS)
# 行业因子：给不同行业一个稳定的利差偏移
industry = rng.integers(0, 6, size=N_BONDS)
ind_off = np.array([-10, 0, 15, 25, -5, 30])[industry]

# 共同宏观因子（信用周期）：先宽松后走阔再回落
t = np.arange(N_MONTHS)
macro = 60 * np.exp(-((t - 40) ** 2) / 900) + 25 * np.sin(t / 9.0)  # bps 周期项
macro = macro - macro.mean()

# 构建面板：每个债券每个月的利差
spread = np.zeros((N_BONDS, N_MONTHS))
for i in range(N_BONDS):
    ri = rating_idx[i]
    idio = np.cumsum(rng.normal(0, vol_spread[ri] / np.sqrt(12), N_MONTHS))
    spread[i] = base_spread[ri] + ind_off[i] + 0.6 * macro + idio

# ============================================================
# 图1：各评级信用利差截面分布（箱线图，取最后 12 个月均值）
# ============================================================
recent = spread[:, -12:].mean(axis=1)
fig, ax = plt.subplots(figsize=(11, 6.2))
data = [recent[rating_idx == k] for k in range(len(RATINGS))]
bp = ax.boxplot(data, tick_labels=RATINGS, patch_artist=True,
                showfliers=False, medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.1, 0.9, len(RATINGS)))):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
ax.set_title("各评级信用利差截面分布（近 12 个月均值，bps）", fontsize=15, fontweight="bold")
ax.set_ylabel("信用利差 (bps)")
ax.set_xlabel("评级")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "cs_distribution.png"), dpi=130)
plt.close(fig)

# ============================================================
# 2) 利差「变化」z-score 与未来 3 月超额回报关系
#    建模：利差走阔(相对自身历史) → 未来回报为负（carry 被信用损失吃掉）
# ============================================================
H = 3  # 持有 3 个月
# 利差月度变化
dspread = np.diff(spread, axis=1)  # (N_BONDS, N_MONTHS-1)
# 用过去 24 个月滚动均值/标准差计算 z-score
W = 24
z = np.full_like(dspread, np.nan)
for i in range(N_BONDS):
    for m in range(W, dspread.shape[1]):
        hist = dspread[i, m - W:m]
        mu, sd = hist.mean(), hist.std(ddof=1) + 1e-6
        z[i, m] = (dspread[i, m] - mu) / sd

# 未来 3 月超额回报（相对同评级基准）：信用利差走阔 -> 负超额
fwd_ret = np.full((N_BONDS, N_MONTHS - 1), np.nan)
for i in range(N_BONDS):
    ri = rating_idx[i]
    bench = spread[rating_idx == ri].mean(axis=0)
    for m in range(H, N_MONTHS - 1):
        carry = (bench[m] - spread[i, m]) / 100.0 / 12.0 * H  # 票息 carry
        # 利差走阔(本券相对基准恶化) -> 价格下跌
        worsen = (spread[i, m] - bench[m]) - (spread[i, m - H] - bench[m - H])
        price = -worsen / 100.0 * 5.0  # 久期 ~5
        fwd_ret[i, m] = carry + price + rng.normal(0, 0.003)

# 抽样散点
zs, rs = [], []
for i in range(N_BONDS):
    for m in range(W, N_MONTHS - 1 - H):
        if not np.isnan(z[i, m]) and not np.isnan(fwd_ret[i, m]):
            zs.append(z[i, m])
            rs.append(fwd_ret[i, m] * 100)
zs = np.array(zs); rs = np.array(rs)
# 分箱均值
xbins = np.linspace(-3, 3, 13)
yc = []
for k in range(len(xbins) - 1):
    msk = (zs >= xbins[k]) & (zs < xbins[k + 1])
    yc.append(rs[msk].mean() if msk.sum() > 30 else np.nan)
yc = np.array(yc)
xc = (xbins[:-1] + xbins[1:]) / 2

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.scatter(zs[::3], rs[::3], s=6, alpha=0.12, color="#4c72b0", label="个券样本")
ax.plot(xc, yc, "o-", color="#d62728", linewidth=2.2, markersize=7, label="分箱均值")
# 回归线
b1, b0 = np.polyfit(zs, rs, 1)
xs = np.linspace(-3, 3, 50)
ax.plot(xs, b0 + b1 * xs, "--", color="black", linewidth=1.5, label=f"回归(斜率={b1:.3f})")
ax.axvline(0, color="gray", alpha=0.4)
ax.set_title("信用利差变化 z-score vs 未来 3 月超额回报（%）", fontsize=15, fontweight="bold")
ax.set_xlabel("利差变化 z-score（相对自身 24 个月历史）")
ax.set_ylabel("未来 3 月超额回报 (%)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "cs_signal_scatter.png"), dpi=130)
plt.close(fig)

# ============================================================
# 3) 信用利差「相对价值」多空组合净值
#    每月：做多利差最宽(相对自身历史 z 最高) Top、做空最窄 Bottom
# ============================================================
port = np.ones(N_MONTHS - 1)
for m in range(W, N_MONTHS - 1 - H):
    zcol = z[:, m]
    valid = ~np.isnan(zcol)
    if valid.sum() < 20:
        continue
    zv = zcol[valid]
    longs = np.where(valid)[0][zv >= np.percentile(zv, 80)]
    shorts = np.where(valid)[0][zv <= np.percentile(zv, 20)]
    r_long = np.nanmean([fwd_ret[i, m] for i in longs])
    r_short = np.nanmean([fwd_ret[i, m] for i in shorts])
    port[m + 1] = port[m] * (1 + (r_long - r_short))

eq = port / port[0]
# 基准：等权持有全部
bench_eq = np.cumprod(1 + np.nanmean(fwd_ret, axis=0))
bench_eq = np.concatenate([[1.0], bench_eq])[: len(eq)]
# 对齐长度
minlen = min(len(eq), len(bench_eq))
eq, bench_eq = eq[:minlen], bench_eq[:minlen]

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.plot(eq, color="#2ca02c", linewidth=2.2, label="信用利差相对价值多空组合")
ax.plot(bench_eq, color="#7f7f7f", linewidth=1.6, label="等权持有基准")
ax.set_title("信用利差相对价值多空组合净值（10 年模拟）", fontsize=15, fontweight="bold")
ax.set_ylabel("净值 (起始=1)")
ax.set_xlabel("月份")
ax.legend()
ax.grid(alpha=0.3)
# 标注年化收益
ann = eq[-1] ** (12 / (minlen - 1)) - 1
ax.text(0.02, 0.06, f"组合年化 ≈ {ann*100:.1f}%", transform=ax.transAxes,
        fontsize=12, color="#2ca02c", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(D, "cs_longshort_equity.png"), dpi=130)
plt.close(fig)

print("credit-spread-factor 配图已生成：", os.listdir(D))
