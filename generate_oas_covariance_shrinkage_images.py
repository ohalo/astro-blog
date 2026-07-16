#!/usr/bin/env python3
"""
为文章「OAS 信用利差收缩：把债券相对价值做成可交易信号」(oas-covariance-shrinkage)
生成真实配图（自洽合成，非占位图）。

数据模型：
  模拟 240 只信用债（评级 × 行业 × 时间）的 OAS（期权调整利差）面板。
  OAS 相对自身历史的 z-score 是相对价值信号：利差走阔(变宽) → 未来超额回报为负，
  利差收窄(变窄) → 未来超额回报为正（carry + 信用利差收敛）。

图表：
  1. oas_distribution.png       各评级 OAS 截面分布（箱线图）
  2. oas_zscore_signal.png      OAS 变化 z-score vs 未来 3 月超额回报（散点+回归）
  3. oas_shrinkage_weights.png  Ledoit-Wolf 收缩把利差变动协方差的噪声压掉，
                                 相对价值篮子权重从尖峰变分散
  4. oas_longshort_equity.png   信用 OAS 相对价值多空组合净值 vs 等权基准
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "oas-covariance-shrinkage"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

RNG = np.random.default_rng(20260716)
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB"]
base_oas = np.array([45, 70, 100, 140, 190, 250, 320, 430], dtype=float)
vol_oas = np.array([10, 15, 22, 30, 42, 58, 80, 120], dtype=float)
N_BONDS = 240
N_MONTHS = 120

rating_idx = RNG.integers(0, len(RATINGS), size=N_BONDS)
industry = RNG.integers(0, 6, size=N_BONDS)
ind_off = np.array([-12, 0, 18, 28, -6, 35])[industry]

# 共同信用周期宏观因子
t = np.arange(N_MONTHS)
macro = 70 * np.exp(-((t - 45) ** 2) / 1100) + 30 * np.sin(t / 8.0)
macro = macro - macro.mean()

oas = np.zeros((N_BONDS, N_MONTHS))
for i in range(N_BONDS):
    ri = rating_idx[i]
    idio = np.cumsum(RNG.normal(0, vol_oas[ri] / np.sqrt(12), N_MONTHS))
    oas[i] = base_oas[ri] + ind_off[i] + 0.6 * macro + idio


# ---------- 图1：各评级 OAS 截面分布 ----------
recent = oas[:, -12:].mean(axis=1)
fig, ax = plt.subplots(figsize=(11, 6.2))
data = [recent[rating_idx == k] for k in range(len(RATINGS))]
bp = ax.boxplot(data, tick_labels=RATINGS, patch_artist=True,
                showfliers=False, medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.1, 0.9, len(RATINGS)))):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
ax.set_title("各评级 OAS（期权调整利差）截面分布（近 12 月均值，bps）", fontsize=15, fontweight="bold")
ax.set_ylabel("OAS (bps)")
ax.set_xlabel("评级")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "oas_distribution.png"), dpi=130)
plt.close(fig)


# ---------- 构建信号与未来收益 ----------
H = 3
doas = np.diff(oas, axis=1)  # (N_BONDS, N_MONTHS-1)
W = 24
z = np.full_like(doas, np.nan)
for i in range(N_BONDS):
    for m in range(W, doas.shape[1]):
        hist = doas[i, m - W:m]
        mu, sd = hist.mean(), hist.std(ddof=1) + 1e-6
        z[i, m] = (doas[i, m] - mu) / sd

fwd_ret = np.full((N_BONDS, N_MONTHS - 1), np.nan)
for i in range(N_BONDS):
    ri = rating_idx[i]
    bench = oas[rating_idx == ri].mean(axis=0)
    for m in range(H, N_MONTHS - 1):
        carry = (bench[m] - oas[i, m]) / 100.0 / 12.0 * H
        worsen = (oas[i, m] - bench[m]) - (oas[i, m - H] - bench[m - H])
        price = -worsen / 100.0 * 6.0  # 久期 ~6
        fwd_ret[i, m] = carry + price + RNG.normal(0, 0.0035)


# ---------- 图2：z-score vs 未来超额回报 ----------
zs, rs = [], []
for i in range(N_BONDS):
    for m in range(W, N_MONTHS - 1 - H):
        if not np.isnan(z[i, m]) and not np.isnan(fwd_ret[i, m]):
            zs.append(z[i, m])
            rs.append(fwd_ret[i, m] * 100)
zs = np.array(zs); rs = np.array(rs)
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
b1, b0 = np.polyfit(zs, rs, 1)
xs = np.linspace(-3, 3, 50)
ax.plot(xs, b0 + b1 * xs, "--", color="black", linewidth=1.5, label=f"回归(斜率={b1:.3f})")
ax.axvline(0, color="gray", alpha=0.4)
ax.set_title("OAS 变化 z-score vs 未来 3 月超额回报（%）", fontsize=15, fontweight="bold")
ax.set_xlabel("OAS 变化 z-score（相对自身 24 个月历史）")
ax.set_ylabel("未来 3 月超额回报 (%)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "oas_zscore_signal.png"), dpi=130)
plt.close(fig)


# ---------- 图3：Ledoit-Wolf 收缩对相对价值篮子权重的影响 ----------
# 取某月截面：用每只债券过去 24 月 OAS 变化序列估协方差，
# 目标 = 做多最宽 OAS / 做空最窄 OAS（相对价值），权重用协方差逆。
# 样本协方差 vs LW 收缩协方差 → 权重稳定性对比。
def ledoit_wolf(S, xs):
    """解析 Ledoit-Wolf 线性收缩，目标 F = 对角（清相关性、留方差）。"""
    p = S.shape[0]
    F = np.diag(np.diag(S))
    d2 = np.sum((S - F) ** 2) / p
    # pi：误差交叉项（简化解析估计）
    T = xs.shape[0]
    # 经验估计 pi（样本叉积二阶矩与协方差平方之差）
    Xc = xs - xs.mean(0)
    cov_err = np.zeros((p, p))
    for t_ in range(T):
        xt = Xc[t_]
        outer = np.outer(xt, xt)
        cov_err += (outer - S) ** 2
    cov_err /= T
    pi = np.sum(cov_err) / p
    rho = np.minimum(pi / d2, 1.0)
    delta = np.minimum(np.maximum(rho, 0.0), 1.0)
    return (1 - delta) * S + delta * F, delta


m = 80  # 取某个横截面月
sel = np.where(~np.isnan(z[:, m]))[0][:40]
# 每只债券过去 24 月 OAS 变化序列
Xs = doas[np.ix_(sel, range(m - W, m))].T  # (T, p)
S = np.cov(Xs, rowvar=False)
Slw, delta = ledoit_wolf(S, Xs)

# 相对价值信号：OAS z-score（该月），做多高 z（最宽）、做空低 z（最窄）
zcol = z[sel, m]
signal = zcol - zcol.mean()
# 最小方差型权重（用协方差逆），并做多空中性
def mv_weights(cov):
    inv = np.linalg.inv(cov)
    w = inv @ signal
    w = w - w.mean()  # 多空中性
    return w
w_samp = mv_weights(S)
w_lw = mv_weights(Slw)

p = len(sel)
fig, ax = plt.subplots(figsize=(12, 5.6))
xpos = np.arange(p)
ax.bar(xpos - 0.2, w_samp, width=0.4, color="#d62728", alpha=0.8, label=f"样本协方差权重（尖峰）")
ax.bar(xpos + 0.2, w_lw, width=0.4, color="#2ca02c", alpha=0.8, label=f"LW 收缩权重 δ={delta:.2f}（分散）")
ax.set_title("Ledoit-Wolf 收缩把相对价值篮子权重从『尖峰』压回『分散』", fontsize=14, fontweight="bold")
ax.set_xlabel("债券序号（按 OAS 信号排序）")
ax.set_ylabel("权重（多空中性）")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "oas_shrinkage_weights.png"), dpi=130)
plt.close(fig)


# ---------- 图4：相对价值多空组合净值 ----------
port = np.ones(N_MONTHS - 1)
for mm in range(W, N_MONTHS - 1 - H):
    zcol = z[:, mm]
    valid = ~np.isnan(zcol)
    if valid.sum() < 20:
        continue
    zv = zcol[valid]
    longs = np.where(valid)[0][zv >= np.percentile(zv, 80)]
    shorts = np.where(valid)[0][zv <= np.percentile(zv, 20)]
    r_long = np.nanmean([fwd_ret[i, mm] for i in longs])
    r_short = np.nanmean([fwd_ret[i, mm] for i in shorts])
    port[mm + 1] = port[mm] * (1 + (r_long - r_short))

eq = port / port[0]
bench_eq = np.cumprod(1 + np.nanmean(fwd_ret, axis=0))
bench_eq = np.concatenate([[1.0], bench_eq])[: len(eq)]
minlen = min(len(eq), len(bench_eq))
eq, bench_eq = eq[:minlen], bench_eq[:minlen]

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.plot(eq, color="#2ca02c", linewidth=2.2, label="OAS 相对价值多空组合")
ax.plot(bench_eq, color="#7f7f7f", linewidth=1.6, label="等权持有基准")
ax.set_title("OAS 相对价值多空组合净值（10 年模拟）", fontsize=15, fontweight="bold")
ax.set_ylabel("净值 (起始=1)")
ax.set_xlabel("月份")
ax.legend()
ax.grid(alpha=0.3)
ann = eq[-1] ** (12 / (minlen - 1)) - 1
ax.text(0.02, 0.06, f"组合年化 ≈ {ann*100:.1f}%", transform=ax.transAxes,
        fontsize=12, color="#2ca02c", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(D, "oas_longshort_equity.png"), dpi=130)
plt.close(fig)

print("oas-covariance-shrinkage 配图已生成：", sorted(os.listdir(D)))
