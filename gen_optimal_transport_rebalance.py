#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最优传输再平衡 配图生成 (4 张真实图表)

机制(自洽合成, 仅用于演示方法):
  * 8 资产多资产组合, 给定战略目标权重 w_tgt, 真实持仓已漂移成 w_cur
  * 把「再平衡」表述为最优传输: 源质量=当前权重, 汇需求=目标权重,
    成本矩阵 C: C_ii=0 (原地不动零成本), C_ij (i≠j) = 基础比例成本 κ,
    部分特定资产对给「折价」(可实物交换/篮子交易, 近乎零成本)
  * 用 scipy 线性规划求解最小成本传输方案 T (N×N), T_ij = 从资产 i 移到 j 的质量
  * 三种再平衡成本对比:
      (a) 朴素现金中枢: 每个资产卖出→现金→买入, 成本 = κ·Σ|Δw| = 2κ·换手率
      (b) OT 均匀: 直接换仓(现货对现货), 成本 = κ·½Σ|Δw| = κ·换手率 (省一半)
      (c) OT 含配对折价: 利用廉价配对, 成本 < κ·换手率
  * 图1: 当前 vs 目标权重(漂移问题)
  * 图2: OT 传输方案热力图(实际交易结构: 谁流向谁)
  * 图3: 三种再平衡成本对比(省了多少)
  * 图4: 回测——用 W1(½Σ|Δw|) 作再平衡触发阈值 vs 固定周期, 成本/漂移权衡
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import linprog

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "optimal-transport-rebalance"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"ot": "#4C72B0", "naive": "#C44E52", "disc": "#55A868", "grid": "#DDDDDD",
     "cur": "#DD8452", "tgt": "#8172B3", "band": "#CCB974"}

names = ["沪深300", "国债", "黄金", "商品", "美股", "可转债", "REITs", "现金"]
N = len(names)
w_tgt = np.array([0.30, 0.20, 0.12, 0.10, 0.13, 0.05, 0.05, 0.05])
w_cur = np.array([0.38, 0.16, 0.13, 0.11, 0.09, 0.04, 0.04, 0.05])
assert abs(w_tgt.sum() - 1) < 1e-9 and abs(w_cur.sum() - 1) < 1e-9

kappa = 0.001  # 基础比例交易成本 10bps

def build_cost(discount_pairs=None):
    """C_ii=0, 其余=kappa; discount_pairs 中 (i,j) 给极小成本(实物/篮子交换)"""
    Cm = np.full((N, N), kappa)
    np.fill_diagonal(Cm, 0.0)
    if discount_pairs:
        for (i, j), v in discount_pairs.items():
            Cm[i, j] = v
            Cm[j, i] = v
    return Cm

def solve_ot(w0, w1, Cm):
    """最小成本传输: min Σ Cm_ij T_ij, s.t. 行和=w0, 列和=w1, T≥0"""
    c = Cm.flatten()
    A_eq = []
    b_eq = []
    # 行约束
    for i in range(N):
        row = np.zeros(N * N)
        row[i * N:(i + 1) * N] = 1.0
        A_eq.append(row); b_eq.append(w0[i])
    # 列约束 (去掉最后一列, 由行和=列和=1 推出)
    for j in range(N - 1):
        col = np.zeros(N * N)
        col[j::N] = 1.0
        A_eq.append(col); b_eq.append(w1[j])
    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * (N * N), method="highs")
    T = res.x.reshape(N, N)
    return T, float(res.fun)

# 成本对比
Cm_uniform = build_cost()
Cm_disc = build_cost({(5, 1): 0.0001, (1, 5): 0.0001, (4, 0): 0.0002, (0, 4): 0.0002})

T_uniform, cost_uniform = solve_ot(w_cur, w_tgt, Cm_uniform)
T_disc, cost_disc = solve_ot(w_cur, w_tgt, Cm_disc)

turnover = 0.5 * np.sum(np.abs(w_tgt - w_cur))      # ½Σ|Δw| = W1(0/1 成本)
cost_naive = kappa * np.sum(np.abs(w_tgt - w_cur))  # 现金中枢 = 2κ·换手率
cost_ot_uniform = cost_uniform                       # = κ·换手率
cost_ot_disc = cost_disc

# ---------------------------------------------------------------------------
# 图 1: 当前 vs 目标权重
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(N); w = 0.38
ax.bar(x - w / 2, w_cur * 100, w, color=C["cur"], label="当前持仓(已漂移)")
ax.bar(x + w / 2, w_tgt * 100, w, color=C["tgt"], label="战略目标权重")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("权重 (%)")
ax.set_title("权重漂移: 沪深300 超配至 38%、国债低配至 16%, 需再平衡")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "weights_drift.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 图 2: OT 传输方案热力图 (含折价配对)
# ---------------------------------------------------------------------------
T_show = T_disc.copy()
np.fill_diagonal(T_show, np.nan)  # 不显示对角(原地持有)
fig, ax = plt.subplots(figsize=(8.2, 6.8))
im = ax.imshow(T_show * 100, cmap="viridis")
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("→ 流向(目标资产)")
ax.set_ylabel("← 来源(当前资产)")
ax.set_title("OT 最小成本传输方案: 非对角格=实际换仓流量\n(沪深300→国债、美股→可转债等直接互换, 避免现金来回)")
for i in range(N):
    for j in range(N):
        if i != j and T_show[i, j] > 1e-4:
            ax.text(j, i, f"{T_show[i,j]*100:.1f}", ha="center", va="center",
                    color="white" if T_show[i, j] > np.nanmax(T_show) * 0.5 else "black",
                    fontsize=7)
cbar = fig.colorbar(im, ax=ax); cbar.set_label("传输质量 (×100)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ot_plan.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 图 3: 三种再平衡成本对比
# ---------------------------------------------------------------------------
labels = ["朴素现金中枢\n(卖→现金→买)", "OT 直接换仓\n(均匀成本)", "OT 含配对折价\n(实物/篮子)"]
costs = [cost_naive, cost_ot_uniform, cost_ot_disc]
cols = [C["naive"], C["ot"], C["disc"]]
fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.bar(labels, [c * 1e4 for c in costs], color=cols)
ax.set_ylabel("再平衡总成本 (基点 = ×10⁻⁴)")
ax.set_title("三种再平衡成本: OT 直接换仓省一半, 折价配对再降一截")
for b, c in zip(bars, costs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
            f"{c*1e4:.1f}", ha="center", fontsize=9)
ax.grid(True, color=C["grid"], axis="y")
# 标注节省
ax.text(1, cost_ot_uniform * 1e4 + 2, f"-{100*(1-cost_ot_uniform/cost_naive):.0f}%",
        ha="center", color=C["ot"], fontsize=9, fontweight="bold")
ax.text(2, cost_ot_disc * 1e4 + 2, f"-{100*(1-cost_ot_disc/cost_naive):.0f}%",
        ha="center", color=C["disc"], fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cost_compare.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 图 4: 回测——W1 触发阈值 vs 固定周期 再平衡
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260719)
T_days = 504  # 约 2 年日度, 让漂移充分累积、阈值有触发机会
ann_vol = np.array([0.30, 0.06, 0.18, 0.22, 0.26, 0.12, 0.30, 0.006])
corr = np.eye(N)
# 构造一个有结构的低维相关
pairs = [(0, 4), (0, 6), (1, 5), (2, 3), (4, 6)]
for (a, b) in pairs:
    corr[a, b] = corr[b, a] = 0.45
Dm = np.diag(ann_vol)
cov = Dm @ corr @ Dm / 252.0
L = np.linalg.cholesky(cov)
rets = rng.standard_normal((T_days, N)) @ L.T

w = w_tgt.copy()
eps_thr = 0.04  # W1 触发阈值 4%
cost_periodic = []; cost_thr = []; cost_never = [0.0]
drift_periodic = []; drift_thr = []; drift_never = []
w_p = w_tgt.copy(); w_t = w_tgt.copy(); w_n = w_tgt.copy()
cp = 0.0; ct = 0.0
for d in range(T_days):
    r = rets[d]
    # 权重随收益漂移
    for ww in (w_p, w_t, w_n):
        ww[:] = ww * (1 + r) / (1 + ww @ r)
    # 周期(每月=21天)再平衡
    if (d + 1) % 21 == 0:
        cp += 0.5 * np.sum(np.abs(w_p - w_tgt))
        w_p[:] = w_tgt
    # 阈值再平衡
    drift_t = 0.5 * np.sum(np.abs(w_t - w_tgt))
    if drift_t > eps_thr:
        ct += drift_t
        w_t[:] = w_tgt
    cost_periodic.append(cp)
    cost_thr.append(ct)
    cost_never.append(0.0)
    drift_periodic.append(0.5 * np.sum(np.abs(w_p - w_tgt)))
    drift_thr.append(0.5 * np.sum(np.abs(w_t - w_tgt)))
    drift_never.append(0.5 * np.sum(np.abs(w_n - w_tgt)))

cost_periodic = np.array(cost_periodic) * kappa * 1e4  # 转基点
cost_thr = np.array(cost_thr) * kappa * 1e4
drift_periodic = np.array(drift_periodic) * 100
drift_thr = np.array(drift_thr) * 100
drift_never = np.array(drift_never) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
ax1.plot(cost_periodic, color=C["naive"], lw=1.8, label=f"固定周期(每月) 累计成本")
ax1.plot(cost_thr, color=C["ot"], lw=1.8, label=f"W1 阈值(>{eps_thr:.0%}) 累计成本")
ax1.set_ylabel("累计交易成本 (基点)")
ax1.set_title("W1 距离作再平衡触发器: 省交易成本且不失控漂移")
ax1.legend(fontsize=8); ax1.grid(True, color=C["grid"], lw=0.6)
ax2.plot(drift_periodic, color=C["naive"], lw=1.6, label="固定周期 偏离目标")
ax2.plot(drift_thr, color=C["ot"], lw=1.6, label="W1 阈值 偏离目标")
ax2.plot(drift_never, color=C["grid"], lw=1.6, ls="--", label="永不调仓(对照)")
ax2.set_xlabel("交易日")
ax2.set_ylabel("当前偏离目标 (½Σ|Δw| ×100)")
ax2.legend(fontsize=8); ax2.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rebalance_backtest.png")); plt.close(fig)

print("=" * 66)
print("最优传输再平衡 关键数字")
print("=" * 66)
print("资产数 N =", N, " 基础成本 κ =", kappa)
print("换手率(T=½Σ|Δw|) = %.4f" % turnover)
print("朴素现金中枢成本 = %.5f (%.1f bps)" % (cost_naive, cost_naive * 1e4))
print("OT 均匀直接换仓  = %.5f (%.1f bps), 省 %.1f%%" %
      (cost_ot_uniform, cost_ot_uniform * 1e4, 100 * (1 - cost_ot_uniform / cost_naive)))
print("OT 含配对折价    = %.5f (%.1f bps), 省 %.1f%%" %
      (cost_ot_disc, cost_ot_disc * 1e4, 100 * (1 - cost_ot_disc / cost_naive)))
print("对角留存(原地不动)质量 = %.4f / 1.0" % np.trace(T_disc))
n_trades = int((np.triu(T_disc, 1) > 1e-4).sum() + (np.tril(T_disc, -1) > 1e-4).sum())
print("OT 折价方案非零换仓对数 =", n_trades)
print("回测: 固定周期累计成本 = %.1f bps" % cost_periodic[-1])
print("回测: W1阈值累计成本   = %.1f bps (省 %.0f%%)" %
      (cost_thr[-1], 100 * (1 - cost_thr[-1] / max(cost_periodic[-1], 1e-9))))
print("回测末日偏离: 周期=%.2f%% 阈值=%.2f%% 永不=%.2f%%" %
      (drift_periodic[-1], drift_thr[-1], drift_never[-1]))
print("DONE ->", OUT, os.listdir(OUT))
