#!/usr/bin/env python3
"""
为文章「隐含相关性指数与相关性风险溢价：相关也能成为因子」(implied-correlation)
生成真实配图。所有图表均由脚本内自洽合成数据 + 文中方法真实计算生成，仅用于演示方法论。

机制：
  指数方差恒等式 σ_I^2 = Σ w_i^2 σ_i^2 + Σ_{i≠j} w_i w_j σ_i σ_j ρ_ij
  在「等相关」近似下解出隐含相关性：
    ρ_impl = (σ_I^2 - Σ w_i^2 σ_i^2) / (Σ_{i≠j} w_i w_j σ_i σ_j)
  用指数期权 IV 得 σ_I（隐含），用成分股期权 IV 得 σ_i（隐含），即得隐含相关性指数。
  相关性风险溢价 CRP = 隐含相关性 - 已实现相关性（通常 > 0，是可收割的风险溢价）。
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
D = os.path.join(BASE, "implied-correlation")
os.makedirs(D, exist_ok=True)

C = {"impl": "#C44E52", "real": "#2F4B7C", "crp": "#55A868", "grid": "#DDDDDD",
     "idx": "#8172B3", "comp": "#DD8452", "crash": "#B22222"}

rng = np.random.default_rng(20260712)
T = 756  # 3 年日度
t = np.arange(T)

# ---------- 合成一个隐含相关性 / 已实现相关性时间序列 ----------
# 已实现相关性：均值回归 + 两次危机尖峰
real_corr = 0.35 + 0.05 * np.sin(t / 90)
real_corr += 0.02 * rng.standard_normal(T).cumsum() / np.sqrt(T)
# 两次危机：相关性飙升
for c0, w, amp in [(220, 40, 0.42), (520, 50, 0.35)]:
    real_corr += amp * np.exp(-((t - c0) ** 2) / (2 * w ** 2))
real_corr = np.clip(real_corr, 0.05, 0.95)

# 隐含相关性 = 已实现 + 正的风险溢价（恐慌前抬升，危机中被压缩甚至倒挂）
premium = 0.12 + 0.03 * np.sin(t / 70 + 1.0)
# 危机爆发瞬间隐含被已实现追上（溢价收窄/倒挂）
for c0, w in [(225, 35), (525, 45)]:
    premium -= 0.10 * np.exp(-((t - c0) ** 2) / (2 * w ** 2))
impl_corr = np.clip(real_corr + premium + 0.01 * rng.standard_normal(T), 0.05, 0.98)
crp = impl_corr - real_corr

# ========== 图1：隐含相关性 vs 已实现相关性 ==========
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(t, impl_corr, color=C["impl"], lw=1.8, label="隐含相关性（期权 IV 反推）")
ax.plot(t, real_corr, color=C["real"], lw=1.8, label="已实现相关性（历史回看）")
ax.fill_between(t, real_corr, impl_corr, where=(impl_corr >= real_corr),
                color=C["crp"], alpha=0.25, label="相关性风险溢价 CRP > 0")
ax.fill_between(t, real_corr, impl_corr, where=(impl_corr < real_corr),
                color=C["crash"], alpha=0.25, label="溢价倒挂（危机中）")
for c0 in [222, 522]:
    ax.axvline(c0, color=C["crash"], ls="--", lw=1, alpha=0.6)
ax.set_title("隐含相关性长期高于已实现相关性——溢价可收割，危机中倒挂", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("平均成分相关性 ρ")
ax.set_ylim(0, 1); ax.grid(color=C["grid"], alpha=0.6)
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(D, "implied_vs_realized_corr.png"), dpi=130)
plt.close(fig)

# ========== 图2：分散度交易 —— 指数波动率 vs 成分股加权波动率 ==========
# 指数方差恒等式：卖指数波动、买成分波动 = 做空相关性
n = 50
w = np.ones(n) / n
sig_i = 0.25 + 0.05 * rng.standard_normal(n)  # 成分股年化波动
sig_i = np.clip(sig_i, 0.12, 0.45)
rho_grid = np.linspace(0.05, 0.95, 100)
# 指数波动率随相关性上升而上升
avg_var_comp = np.sum(w ** 2 * sig_i ** 2)
cross = np.sum(np.outer(w, w) * np.outer(sig_i, sig_i)) - avg_var_comp
idx_vol = np.sqrt(avg_var_comp + cross * rho_grid)
comp_weighted_vol = np.full_like(rho_grid, np.sqrt(np.sum(w * sig_i ** 2)))  # 加权成分波动参考线

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(rho_grid, idx_vol * 100, color=C["idx"], lw=2.2, label="指数隐含波动率 σ_I")
ax.plot(rho_grid, comp_weighted_vol * 100, color=C["comp"], lw=2.2, ls="--",
        label="成分股加权波动率（相关性无关）")
ax.fill_between(rho_grid, idx_vol * 100, comp_weighted_vol * 100,
                color=C["crp"], alpha=0.2, label="分散度价差（做空相关性收益源）")
ax.axvline(0.35, color=C["real"], ls=":", lw=1.4, label="已实现 ρ≈0.35")
ax.axvline(0.50, color=C["impl"], ls=":", lw=1.4, label="隐含 ρ≈0.50")
ax.set_title("分散度交易：指数波动率随相关性上升，成分波动不变——价差即溢价", fontsize=12)
ax.set_xlabel("平均成分相关性 ρ"); ax.set_ylabel("年化波动率 (%)")
ax.grid(color=C["grid"], alpha=0.6); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(D, "dispersion_trade.png"), dpi=130)
plt.close(fig)

# ========== 图3：做空相关性策略净值 + 危机回撤 ==========
# 每日收 CRP（正 carry），危机中相关性飙升导致巨亏（左尾）
daily_pnl = crp * 0.010 - 0.00015  # 正常日子薄利（正 carry）
# 危机日：已实现相关性>隐含 → 巨亏
crisis_loss = np.zeros(T)
for c0, w2, amp in [(222, 8, 0.014), (522, 8, 0.010)]:
    crisis_loss -= amp * np.exp(-((t - c0) ** 2) / (2 * w2 ** 2))
daily_pnl = daily_pnl + crisis_loss
equity = np.cumprod(1 + daily_pnl)
peak = np.maximum.accumulate(equity)
dd = equity / peak - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})
ax1.plot(t, equity, color=C["crp"], lw=1.8)
ax1.axhline(1.0, color="#999", lw=0.8, ls="--")
for c0 in [222, 522]:
    ax1.axvline(c0, color=C["crash"], ls="--", lw=1, alpha=0.6)
ax1.set_title("做空相关性策略：平日薄利、危机中被相关性飙升重创（左尾风险）", fontsize=12)
ax1.set_ylabel("策略净值"); ax1.grid(color=C["grid"], alpha=0.6)
ax1.text(0.02, 0.9, f"末值 {equity[-1]:.2f}", transform=ax1.transAxes, fontsize=10)

ax2.fill_between(t, dd * 100, 0, color=C["crash"], alpha=0.4)
ax2.set_ylabel("回撤 (%)"); ax2.set_xlabel("交易日")
ax2.grid(color=C["grid"], alpha=0.6)
ax2.text(0.02, 0.15, f"最大回撤 {dd.min()*100:.1f}%", transform=ax2.transAxes, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(D, "short_corr_strategy.png"), dpi=130)
plt.close(fig)

print("implied-correlation images done:", os.listdir(D))
print(f"末值={equity[-1]:.3f} 最大回撤={dd.min()*100:.1f}% 平均CRP={crp.mean():.3f}")
