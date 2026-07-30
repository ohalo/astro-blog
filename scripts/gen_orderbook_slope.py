#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""订单簿斜率信号：模拟 LOB + 斜率不平衡预测短期方向"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/order-imbalance-slope-signal"
os.makedirs(OUT, exist_ok=True)

# ---------- 模拟盘口快照序列 ----------
# 潜在压力变量 x_t（AR(1)），决定买卖盘挂单密度的不对称
T = 60000                 # 快照数（约一个月的 5 秒快照）
phi = 0.995
x = np.zeros(T)
eps = rng.standard_normal(T)
for t in range(1, T):
    x[t] = phi * x[t-1] + np.sqrt(1 - phi**2) * eps[t]

TICK = 0.01
L = 10                    # 档位数
mid = np.zeros(T); mid[0] = 20.0

# 挂单密度模型：Q_bid(k) = A_b * exp(-kappa_b * k)，斜率由 kappa 决定
# 买压大 (x>0) -> 买盘密度衰减慢（kappa_b 小、深度厚）
beta_k = 0.18             # x 对 kappa 的影响
kappa0 = 0.35
A0 = 800.0

kap_b = kappa0 * np.exp(-beta_k * x)
kap_a = kappa0 * np.exp(+beta_k * x)
noise_b = rng.lognormal(0, 0.35, (T, L))
noise_a = rng.lognormal(0, 0.35, (T, L))
kgrid = np.arange(1, L + 1)
Qb = A0 * np.exp(-kap_b[:, None] * kgrid[None, :]) * noise_b   # 各档买量
Qa = A0 * np.exp(-kap_a[:, None] * kgrid[None, :]) * noise_a   # 各档卖量

# 价格生成：短期漂移与 x 相关 + 微观噪声（每快照）
drift_bps = 0.030         # x 每单位对下一快照 mid 的漂移（bp）
sig_bps = 1.8
ret = drift_bps * 1e-4 * x + sig_bps * 1e-4 * rng.standard_normal(T)
mid = 20.0 * np.exp(np.cumsum(ret))

# ---------- 信号构造 ----------
def slope_est(Q):
    """对 log 深度做 OLS：log Q_k = a - kappa * k，返回 kappa"""
    lq = np.log(Q)
    k = kgrid - kgrid.mean()
    return -(lq @ k) / (k @ k)     # 逐行向量化：Q shape (T,L)

kb_hat = slope_est(Qb)
ka_hat = slope_est(Qa)
slope_sig = ka_hat - kb_hat        # >0: 卖盘衰减快=卖压弱 -> 看涨

# 对照信号：L1 不平衡（只用一档）
imb1 = (Qb[:, 0] - Qa[:, 0]) / (Qb[:, 0] + Qa[:, 0])
# 对照信号：前5档总量不平衡
d5b, d5a = Qb[:, :5].sum(1), Qa[:, :5].sum(1)
imb5 = (d5b - d5a) / (d5b + d5a)

def zs(v): return (v - v.mean()) / v.std()
sigs = {"斜率差 (10档)": zs(slope_sig), "一档不平衡": zs(imb1), "五档量不平衡": zs(imb5)}

# ---------- 1) 示意图：两种压力状态下的盘口形状 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=140, sharey=True)
for ax, xx, ttl in [(axes[0], +1.5, "买压状态 (x=+1.5)"), (axes[1], -1.5, "卖压状态 (x=-1.5)")]:
    qb = A0 * np.exp(-kappa0 * np.exp(-beta_k * xx) * kgrid)
    qa = A0 * np.exp(-kappa0 * np.exp(+beta_k * xx) * kgrid)
    ax.barh(-kgrid, qb, color="#2ca02c", alpha=0.8, label="买盘深度")
    ax.barh(kgrid, qa, color="#d62728", alpha=0.8, label="卖盘深度")
    ax.set_title(ttl); ax.set_xlabel("挂单量")
    ax.set_yticks(list(-kgrid) + list(kgrid))
    ax.set_yticklabels([f"B{k}" for k in kgrid] + [f"A{k}" for k in kgrid], fontsize=6)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")
axes[0].set_ylabel("档位")
fig.suptitle("挂单密度的指数衰减形状：压力方向改变斜率", y=1.0)
fig.tight_layout(); fig.savefig(f"{OUT}/lob-shape-regimes.png"); plt.close(fig)

# ---------- 2) 预测能力：IC vs 预测步长 ----------
horizons = [1, 2, 5, 10, 20, 40, 80, 160]
logmid = np.log(mid)
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
ic_table = {}
for name, s in sigs.items():
    ics = []
    for h in horizons:
        fwd = logmid[h:] - logmid[:-h]
        ics.append(np.corrcoef(s[:-h], fwd)[0, 1] * 100)
    ic_table[name] = ics
    ax.plot(horizons, ics, "o-", ms=4, label=name)
ax.set_xscale("log"); ax.set_xticks(horizons); ax.set_xticklabels(horizons)
ax.set_xlabel("预测步长（快照数，5 秒/快照，对数轴）"); ax.set_ylabel("IC（%）")
ax.set_title("三种盘口信号对未来 mid 收益的预测 IC")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/ic-vs-horizon.png"); plt.close(fig)

# ---------- 3) 分位数组合：信号十分位 vs 未来20快照收益 ----------
h = 20
fwd = (logmid[h:] - logmid[:-h]) * 1e4    # bp
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
width = 0.26
for i, (name, s) in enumerate(sigs.items()):
    sv = s[:-h]
    qs = np.quantile(sv, np.linspace(0, 1, 11))
    means = [fwd[(sv >= qs[j]) & (sv < qs[j+1] if j < 9 else sv <= qs[10])].mean() for j in range(10)]
    ax.bar(np.arange(10) + (i - 1) * width, means, width=width, label=name, alpha=0.85)
ax.set_xticks(range(10)); ax.set_xticklabels([f"D{d+1}" for d in range(10)])
ax.set_xlabel("信号十分位"); ax.set_ylabel("未来 20 快照平均收益（bp）")
ax.set_title("信号分位 vs 未来收益：单调性检查（100 秒视界）")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/decile-monotonic.png"); plt.close(fig)

# ---------- 4) 成本现实检验：净收益 vs 半价差 ----------
# 简单策略：|z|>1 触发方向仓位，持有 h 快照，每次进出付一次半价差 + 手续费
spread_bp = (0.01 / 20.0) * 1e4 / 2      # 半价差 ~2.5bp
def strat_pnl(s, h, cost_bp):
    pos = np.where(s > 1, 1, np.where(s < -1, -1, 0))[:-h]
    entry = np.diff(np.concatenate([[0], pos])) != 0
    pnl = pos * fwd - np.abs(np.diff(np.concatenate([[0], pos]))) * cost_bp
    # 近似：每次仓位变化收一次成本
    return pnl

fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
costs = np.linspace(0, 6, 25)
res_cost = {}
for name, s in sigs.items():
    nets = []
    for c in costs:
        p = strat_pnl(s, h, c)
        ann = p.mean()   # bp / 20快照
        nets.append(ann)
    res_cost[name] = nets
    ax.plot(costs, nets, "-", label=name)
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(spread_bp, color="black", ls="--", lw=1, label=f"半价差 ≈ {spread_bp:.1f}bp")
ax.set_xlabel("单边交易成本（bp）"); ax.set_ylabel("每笔平均净收益（bp / 100 秒）")
ax.set_title("成本现实检验：几 bp 的成本就能吃掉全部信号")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/net-vs-cost.png"); plt.close(fig)

# 关键数字
def cross_zero(nets):
    for c, v in zip(costs, nets):
        if v <= 0: return c
    return costs[-1]

# t-stat of slope IC at h=20 (block bootstrap粗略: 用有效样本 N/h)
s0 = sigs["斜率差 (10档)"][:-h]
icc = np.corrcoef(s0, fwd)[0, 1]
neff = len(fwd) / h
tstat = icc * np.sqrt(neff)

summary = {
    "T": T, "L": L,
    "ic20_slope": round(float(icc) * 100, 2),
    "tstat20_slope_neff": round(float(tstat), 2),
    "ic_table_h20": {k: round(v[horizons.index(20)], 2) for k, v in ic_table.items()},
    "ic_table_h1": {k: round(v[0], 2) for k, v in ic_table.items()},
    "ic_table_h160": {k: round(v[-1], 2) for k, v in ic_table.items()},
    "breakeven_cost_bp": {k: round(float(cross_zero(v)), 2) for k, v in res_cost.items()},
    "half_spread_bp": round(float(spread_bp), 2),
    "gross_per_trade_bp": {k: round(float(v[0]), 3) for k, v in res_cost.items()},
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
with open("/Users/halo/workspace/astro-blog/scripts/_obslope_results.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
