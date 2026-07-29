"""GEX 伽马敞口模拟：做市商对冲流对波动的压制与放大"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/gamma-exposure-gex"
import os
os.makedirs(OUT, exist_ok=True)

# ---------- 1. BS gamma 与 GEX 剖面 ----------
def bs_gamma(S, K, T, sigma, r=0.0):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

S0 = 4000.0
strikes = np.arange(3600, 4401, 50.0)
T = 21 / 252
sigma = 0.18

# 合成持仓：做市商 long call gamma（客户卖 call）、short put gamma（客户买 put 对冲）
call_oi = rng.integers(5000, 25000, len(strikes)).astype(float)
put_oi = rng.integers(8000, 30000, len(strikes)).astype(float)
# put 偏度：低行权价 put OI 更大
put_oi *= np.linspace(1.8, 0.6, len(strikes))
call_oi *= np.linspace(0.6, 1.6, len(strikes))

S_grid = np.linspace(3600, 4400, 161)
gex_profile = np.zeros_like(S_grid)
for i, S in enumerate(S_grid):
    g = bs_gamma(S, strikes, T, sigma)
    # dealer: +call gamma, -put gamma（惯例符号），单位：每 1% 涨跌的 delta 变化（亿元名义）
    gex_profile[i] = np.sum(g * call_oi * 100 * S**2 * 0.01) - np.sum(g * put_oi * 100 * S**2 * 0.01)
gex_profile /= 1e8

# gamma flip 点
flip_idx = np.argmin(np.abs(gex_profile))
S_flip = S_grid[flip_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(S_grid, gex_profile, lw=2, color="#1f77b4")
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(S_flip, color="crimson", ls="--", lw=1.2, label=f"Gamma 翻转点 ≈ {S_flip:.0f}")
ax.fill_between(S_grid, gex_profile, 0, where=gex_profile > 0, alpha=0.15, color="green", label="正 GEX：对冲流压制波动")
ax.fill_between(S_grid, gex_profile, 0, where=gex_profile < 0, alpha=0.15, color="red", label="负 GEX：对冲流放大波动")
ax.set_xlabel("标的价格")
ax.set_ylabel("做市商 GEX（亿元 / 1% 变动）")
ax.set_title("做市商伽马敞口剖面与 Gamma 翻转点")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/gex-profile.png", dpi=130)
plt.close(fig)
print(f"flip point: {S_flip:.0f}, GEX at S0: {gex_profile[np.argmin(np.abs(S_grid-S0))]:.1f}")

# ---------- 2. 带对冲反馈的价格模拟 ----------
def simulate(gex_sign, kappa=0.35, n_days=250, n_intraday=26, sigma_d=0.011, seed=1):
    """gex_sign>0: 对冲流抵抗价格变动（负反馈）; <0: 追涨杀跌（正反馈）"""
    r = np.random.default_rng(seed)
    dt_vol = sigma_d / np.sqrt(n_intraday)
    paths = np.zeros((n_days, n_intraday + 1))
    for d in range(n_days):
        p = 0.0
        prev = 0.0
        for t in range(1, n_intraday + 1):
            shock = r.normal(0, dt_vol)
            # 对冲流：与最近一段收益成比例。正 GEX -> 反向对冲（-kappa*ret），负 GEX -> 同向（+kappa*ret）
            hedge = -np.sign(gex_sign) * kappa * (p - prev) if t > 1 else 0.0
            prev = p
            p = p + shock + hedge
            paths[d, t] = p
    daily_ret = paths[:, -1]
    rv = np.sqrt(np.sum(np.diff(paths, axis=1) ** 2, axis=1))
    return paths, daily_ret, rv

paths_pos, ret_pos, rv_pos = simulate(+1, seed=7)
paths_neg, ret_neg, rv_neg = simulate(-1, seed=7)

ann = np.sqrt(252)
print(f"正GEX: 日收益std={ret_pos.std()*100:.2f}%, 年化RV={rv_pos.mean()*ann*100:.1f}%")
print(f"负GEX: 日收益std={ret_neg.std()*100:.2f}%, 年化RV={rv_neg.mean()*ann*100:.1f}%")

# 自相关（30分钟收益一阶自相关）
def ac1(paths):
    d = np.diff(paths, axis=1)
    a = []
    for row in d:
        a.append(np.corrcoef(row[:-1], row[1:])[0, 1])
    return np.mean(a)

ac_pos, ac_neg = ac1(paths_pos), ac1(paths_neg)
print(f"日内收益一阶自相关: 正GEX={ac_pos:.3f}, 负GEX={ac_neg:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for i in range(12):
    axes[0].plot(paths_pos[i] * 100, alpha=0.6, lw=1)
    axes[1].plot(paths_neg[i] * 100, alpha=0.6, lw=1)
axes[0].set_title(f"正 GEX 环境（日内自相关 {ac_pos:+.2f}）")
axes[1].set_title(f"负 GEX 环境（日内自相关 {ac_neg:+.2f}）")
for ax in axes:
    ax.set_xlabel("日内时间（15 分钟 bar）")
axes[0].set_ylabel("累计收益（%）")
fig.suptitle("对冲反馈下的日内价格路径：压制 vs 放大", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/intraday-paths.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- 3. GEX 分组的已实现波动与后续收益 ----------
# 模拟 1000 天：GEX 水平随市场涨跌演化（下跌 -> put 需求 -> GEX 变负）
n = 1000
gex_level = np.zeros(n)
mkt = np.zeros(n)
rv_series = np.zeros(n)
g = 5.0
r2 = np.random.default_rng(11)
for t in range(n):
    kappa_t = 0.30 * np.tanh(abs(g) / 5)
    sign = 1 if g > 0 else -1
    _, dr, rv = simulate(sign, kappa=kappa_t, n_days=1, n_intraday=26, seed=1000 + t)
    mkt[t] = dr[0]
    rv_series[t] = rv[0]
    gex_level[t] = g
    # GEX 动态：市场下跌时客户买 put -> dealer short gamma 增加
    g = 0.97 * g + 60 * dr[0] + r2.normal(0, 0.8)

q = np.quantile(gex_level, [0.2, 0.4, 0.6, 0.8])
groups = np.digitize(gex_level, q)
labels = ["G1\n(最负)", "G2", "G3", "G4", "G5\n(最正)"]
cc_by_g = [mkt[groups == i].std() * ann * 100 for i in range(5)]
absret_by_g = [np.abs(mkt)[groups == i].mean() * 100 for i in range(5)]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(5)
b1 = ax.bar(x - 0.2, cc_by_g, 0.4, label="年化收盘-收盘波动率（%）", color="#1f77b4")
ax2 = ax.twinx()
b2 = ax2.bar(x + 0.2, absret_by_g, 0.4, label="日均绝对收益（%）", color="#ff7f0e")
ax.set_xticks(x, labels)
ax.set_xlabel("GEX 五分组")
ax.set_ylabel("年化收盘-收盘波动率（%）")
ax2.set_ylabel("日均绝对收益（%）")
ax.set_title("GEX 分组下的波动结构：负 GEX 组波动显著更高")
lines = [b1, b2]
ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/gex-vol-groups.png", dpi=130)
plt.close(fig)
print("CC vol by group:", [f"{v:.1f}" for v in cc_by_g])
print("abs ret by group:", [f"{v:.2f}" for v in absret_by_g])

# ---------- 4. 基于 GEX 的波动择时策略 ----------
# 策略：负 GEX -> 买跨式（做多波动近似：|ret|-成本），正 GEX -> 卖跨式
# 跨式成本用日均绝对收益定价（而非日内RV），保证对照组公平
straddle_cost = np.abs(mkt).mean()
pnl_long = np.abs(mkt) - straddle_cost * 1.05  # 5% 溢价成本
pnl_strat = np.where(gex_level < np.quantile(gex_level, 0.3), pnl_long,
             np.where(gex_level > np.quantile(gex_level, 0.7), -pnl_long, 0.0))
cum_strat = np.cumsum(pnl_strat)
cum_alwayslong = np.cumsum(pnl_long)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cum_strat * 100, lw=1.6, label="GEX 择时（负买波动/正卖波动）", color="#2ca02c")
ax.plot(cum_alwayslong * 100, lw=1.2, label="始终做多波动（对照）", color="gray", ls="--")
ax.set_xlabel("交易日")
ax.set_ylabel("累计损益（%）")
ax.set_title("GEX 波动择时 vs 无脑做多波动")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/gex-timing-pnl.png", dpi=130)
plt.close(fig)

sr_strat = pnl_strat.mean() / pnl_strat.std() * ann
sr_long = pnl_long.mean() / pnl_long.std() * ann
print(f"择时策略 Sharpe={sr_strat:.2f}, 总收益={cum_strat[-1]*100:.1f}%")
print(f"永远做多波动 Sharpe={sr_long:.2f}, 总收益={cum_alwayslong[-1]*100:.1f}%")
