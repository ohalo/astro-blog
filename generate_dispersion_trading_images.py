#!/usr/bin/env python3
"""
为文章「离散度交易：用个股 IV 与指数 IV 的背离做波动率套利」(dispersion-trading-correlation)
生成真实配图。模拟：实现相关性(rho_real) 由 OU 过程驱动，隐含相关性(rho_impl) 在
高相关 regime 被指数期权市场超额定价；当 rho_impl 高于阈值时做空相关性(long dispersion)。

图1 dispersion_rho_series.png     rho_impl vs rho_real 时间序列（背离可见）
图2 dispersion_pnl_scatter.png    每期 PnL vs (rho_impl_entry - rho_real_held) 散点
图3 dispersion_equity.png         策略累积权益（择时做空相关性）vs 一直做空（无择时）
图4 dispersion_signal.png         入场信号区（rho_impl>阈值）高亮 + 对应 PnL 柱
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "dispersion-trading-correlation"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

# ===== 1) 模拟相关性序列 =====
T = 750
theta, kappa, sig = 0.45, 0.05, 0.06
rho_real = np.zeros(T); rho_real[0] = theta
for t in range(1, T):
    rho_real[t] = rho_real[t - 1] + kappa * (theta - rho_real[t - 1]) + sig * rng.standard_normal()
    rho_real[t] = min(max(rho_real[t], 0.05), 0.95)
# 隐含相关性：高相关时溢价放大（危机/泡沫期指数 IV 被推高）
premium = 0.18 * (rho_real - 0.40) + 0.02 * np.sin(np.arange(T) / 25.0)
rho_impl = np.clip(rho_real + premium + rng.normal(0, 0.02, T), 0.05, 0.95)

# ===== 2) 月度调仓信号（做空相关性 = long dispersion）=====
months = 60
idx_month = np.linspace(0, T - 1, months + 1).astype(int)
thr = float(np.quantile(rho_impl, 0.70))
scale = 0.20
pnl = np.zeros(months)
pnl_always = np.zeros(months)
signal = np.zeros(months, dtype=bool)
for j in range(months):
    t0 = idx_month[j]; t1 = max(idx_month[j + 1], t0 + 1)
    entry_impl = rho_impl[t0]
    held_real = rho_real[t0:t1].mean()
    diff = entry_impl - held_real
    pnl_always[j] = diff * scale
    if entry_impl > thr:
        signal[j] = True
        pnl[j] = diff * scale

eq = np.cumprod(1 + pnl)
eq_always = np.cumprod(1 + pnl_always)
print(f"threshold rho_impl={thr:.3f}  交易期数={signal.sum()}/{months}")
print(f"择时策略 终值={eq[-1]:.3f}  年化={100*(eq[-1]**(12/months)-1):.1f}%")
print(f"一直做空 终值={eq_always[-1]:.3f}  年化={100*(eq_always[-1]**(12/months)-1):.1f}%")

# ===== 图1：相关性序列 =====
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(rho_real, color="#1f77b4", lw=1.0, label="实现相关性 ρ_real（成分股实际联动）")
ax.plot(rho_impl, color="#d62728", lw=1.0, alpha=0.85, label="隐含相关性 ρ_impl（指数期权推算）")
ax.axhline(thr, color="#7f7f7f", ls="--", lw=1, label=f"入场阈值 {thr:.2f}")
ax.set_title("隐含相关性在危机/泡沫期被指数期权超额定价（背离）", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("相关性")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "dispersion_rho_series.png"), dpi=130); plt.close(fig)

# ===== 图2：PnL vs diff 散点 =====
fig, ax = plt.subplots(figsize=(9, 4.4))
diffs = (rho_impl[idx_month[:-1]] - np.array(
    [rho_real[idx_month[j]:max(idx_month[j+1], idx_month[j]+1)].mean() for j in range(months)]))
ax.scatter(diffs[signal], pnl[signal], color="#2ca02c", s=28, label="做空相关性（已入场）")
ax.scatter(diffs[~signal], np.zeros((~signal).sum()), color="#cccccc", s=18,
           marker="x", label="未入场（空仓）")
# 拟合线（仅交易期）
z = np.polyfit(diffs[signal], pnl[signal], 1)
xx = np.linspace(diffs.min(), diffs.max(), 50)
ax.plot(xx, np.polyval(z, xx), "--", color="#ff7f0e", label=f"斜率≈{z[0]:.1f}")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("入场隐含 − 持有期实现相关性 (ρ_impl − ρ_real)")
ax.set_ylabel("该期策略收益 (%)")
ax.set_title("收益正比于『隐含被高估的幅度』：背离越大赚越多", fontsize=12)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "dispersion_pnl_scatter.png"), dpi=130); plt.close(fig)

# ===== 图3：权益曲线 =====
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(eq, color="#2ca02c", lw=1.6, label="择时做空相关性（long dispersion）")
ax.plot(eq_always, color="#d62728", lw=1.2, alpha=0.7, label="一直做空相关性（无择时）")
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_title("择时过滤掉『低隐含却做空』的亏损期，权益更稳", fontsize=12)
ax.set_xlabel("调仓期（月）"); ax.set_ylabel("累积净值（起始=1）")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "dispersion_equity.png"), dpi=130); plt.close(fig)

# ===== 图4：信号区高亮 + PnL 柱 =====
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(np.arange(months), pnl, color=np.where(signal, "#2ca02c", "#dddddd"),
       label="逐期收益（绿=已入场）")
for j in range(months):
    if signal[j]:
        ax.axvspan(j - 0.5, j + 0.5, color="#2ca02c", alpha=0.08)
ax.axhline(0, color="gray", lw=0.8)
ax.set_title("入场信号区（ρ_impl 高于阈值）高亮：集中贡献正收益", fontsize=12)
ax.set_xlabel("调仓期（月）"); ax.set_ylabel("该期收益 (%)")
ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "dispersion_signal.png"), dpi=130); plt.close(fig)

print("DONE", sorted(os.listdir(D)))
