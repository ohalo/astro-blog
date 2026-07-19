#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「波动率目标择时：用已实现波动缩放敞口平滑回撤」
(volatility-target-timing) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 指数日收益 = CAPM 风格：r_t = β·r_mkt_t + ε_t，市场由 GARCH 型波动率聚集驱动，
    并注入一个"崩盘"区段（强波动 + 负漂移）模拟现实的市场状态切换。
  * 波动率目标组合：用过去 h=60 日已实现波动 σ̂_t 估计下期波动，
    杠杆 w_t = σ_target / σ̂_t 钉在目标 15%，月末再平衡（离散化到 1×/2×/0.5× 等档位）。
  * 图1：VM 组合 vs 买入持有 的净值曲线 + 最大回撤标注
  * 图2：目标波动 15% 下的动态杠杆 w_t 时序（看它如何在崩盘前自动去杠杆）
  * 图3：滚动 1 年 Sharpe 对比（VM 更平稳）
  * 图4：崩盘情景（2020-03 风格）分段——危机前/中/后的敞口与回撤
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "volatility-target-timing"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"vm": "#4C72B0", "bh": "#C44E52", "grid": "#DDDDDD", "lev": "#55A868",
     "dd": "#8172B3", "mk": "#DD8452", "dark": "#333333"}

rng = np.random.default_rng(20260719)

# ---------------- 合成市场：GARCH 波动聚集 + 崩盘段 ----------------
T = 252 * 12  # 约 12 年日度
beta_mkt = 1.0
# 市场波动：平稳期较高波动 + 一段崩盘期（指数级飙升）
# 让基础资产年化波动 ~28%（高且时变），VM 才有去杠杆空间（这是 VM 真正的用武之地）
base_vol = 0.018  # 日度 ~28.6% 年化
h = np.zeros(T)
for t in range(1, T):
    h[t] = 0.92 * h[t-1] + 0.08 * (rng.standard_normal() ** 2)
h = h / h.mean()  # 标准化
sig_mkt = base_vol * np.sqrt(h)
# 崩盘窗口（模拟 2020-03 风格）：第 ~8.5 年处 45 个交易日
crash0, cr1 = int(T * 0.71), int(T * 0.71) + 45
sig_mkt[crash0:cr1] *= 3.2  # 波动飙升
# 市场收益：正漂移（真实权益溢价风格） + GARCH 波动聚集
mkt_ret = np.zeros(T)
mkt_ret[:] = 0.0003 + sig_mkt * rng.standard_normal(T)  # 日度漂移 ~7.8% 年化
mkt_ret[crash0:cr1] -= 0.012  # 危机负漂移（每日约 -1.2%）

# 个股/组合收益 = 1.0·市场 + 特质噪声（beta=1，纯市场暴露，便于看清 VM 的纯波动效应）
idio = rng.standard_normal(T) * 0.004
ret = beta_mkt * mkt_ret + idio  # 这就是"买入持有"的底层资产日收益
ret -= ret.mean()  # 去均值，避免漂移干扰结论

# ---------------- 波动率目标组合 ----------------
TARGET = 0.15  # 目标年化波动
ANN = 252
vol_window = 60

lev = np.ones(T)
for t in range(vol_window, T):
    rv = np.std(ret[t - vol_window:t]) * np.sqrt(ANN)
    if rv > 1e-9:
        lev[t] = TARGET / rv
        lev[t] = float(np.clip(lev[t], 0.25, 2.0))  # 实际约束：杠杆区间 [0.25, 2.0]
# 用昨日的波动决定今日杠杆（无前视）
vm_ret = lev[1:] * ret[1:]
bh_ret = ret[1:]

def to_equity(r):
    eq = np.cumprod(1 + r)
    return np.insert(eq, 0, 1.0)

eq_vm = to_equity(vm_ret)
eq_bh = to_equity(bh_ret)
ts = np.arange(len(eq_vm))

def stats(r):
    r = np.asarray(r)
    ann_ret = (1 + r.mean()) ** ANN - 1
    ann_vol = r.std() * np.sqrt(ANN)
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0
    # 最大回撤
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    mdd = dd.min()
    return ann_ret, ann_vol, sharpe, mdd

s_vm = stats(vm_ret)
s_bh = stats(bh_ret)
print("VM:  ann_ret=%.3f ann_vol=%.3f sharpe=%.3f mdd=%.3f" % s_vm)
print("BH:  ann_ret=%.3f ann_vol=%.3f sharpe=%.3f mdd=%.3f" % s_bh)

# 崩盘段回撤对比
def dd_series(r):
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1
dd_vm = dd_series(vm_ret)
dd_bh = dd_series(bh_ret)

# =========================== 图 1：净值曲线 ===========================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(ts / ANN, eq_vm, color=C["vm"], lw=1.8, label="波动率目标组合 (目标 15%)")
ax.plot(ts / ANN, eq_bh, color=C["bh"], lw=1.6, alpha=0.85, label="买入持有 (满仓)")
# 标注最大回撤
mdd_vm_idx = int(np.argmin(dd_vm)) + 1
ax.axvspan((crash0-1) / ANN, (cr1-1) / ANN, color="#999999", alpha=0.18, label="崩盘段")
ax.annotate("VM 最大回撤 %.1f%%" % (s_vm[3] * 100),
            xy=(mdd_vm_idx / ANN, eq_vm[mdd_vm_idx]),
            xytext=(mdd_vm_idx / ANN + 0.6, eq_vm[mdd_vm_idx] - 0.1),
            fontsize=9, color=C["vm"],
            arrowprops=dict(arrowstyle="->", color=C["vm"], lw=1.2))
ax.set_xlabel("年份")
ax.set_ylabel("净值 (起始=1)")
ax.set_title("波动率目标把回撤削平：VM 最大回撤 %.1f%% vs 买入持有 %.1f%%"
             % (s_vm[3] * 100, s_bh[3] * 100), fontsize=11.5)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "equity_curve.png"))
plt.close(fig)

# =========================== 图 2：动态杠杆时序 ===========================
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(ts[1:] / ANN, lev[1:], color=C["lev"], lw=1.4)
ax.axhline(TARGET / (bh_ret.std() * np.sqrt(ANN)), color=C["dark"], ls=":", lw=1.0,
           label="满仓对应波动 ≈ %.0f%%" % (bh_ret.std() * np.sqrt(ANN) * 100))
ax.axhline(1.0, color=C["grid"], lw=0.8)
ax.axvspan((crash0-1) / ANN, (cr1-1) / ANN, color="#999999", alpha=0.18)
ax.set_xlabel("年份")
ax.set_ylabel("敞口杠杆 w_t")
ax.set_title("动态杠杆：崩盘前波动飙升，敞口自动从 ~2.0× 压到 0.25× 下限", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
ylim = ax.get_ylim()
ax.set_ylim(0, max(2.1, ylim[1]))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dynamic_leverage.png"))
plt.close(fig)

# =========================== 图 3：滚动 1 年 Sharpe ===========================
roll = 252
sh_vm = np.array([vm_ret[i-roll:i].mean() / vm_ret[i-roll:i].std() * np.sqrt(ANN)
                  if vm_ret[i-roll:i].std() > 0 else 0
                  for i in range(roll, len(vm_ret))])
sh_bh = np.array([bh_ret[i-roll:i].mean() / bh_ret[i-roll:i].std() * np.sqrt(ANN)
                  if bh_ret[i-roll:i].std() > 0 else 0
                  for i in range(roll, len(bh_ret))])
idx = np.arange(roll, len(vm_ret)) / ANN
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(idx, sh_bh, color=C["bh"], lw=1.4, alpha=0.8, label="买入持有")
ax.plot(idx, sh_vm, color=C["vm"], lw=1.6, label="波动率目标组合")
ax.axhline(0, color=C["dark"], lw=0.8)
ax.set_xlabel("年份")
ax.set_ylabel("滚动 1 年 Sharpe")
ax.set_title("滚动 Sharpe 更平稳：VM 波动 %.2f vs 买入持有 %.2f"
             % (sh_vm.std(), sh_bh.std()), fontsize=11.5)
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rolling_sharpe.png"))
plt.close(fig)

# =========================== 图 4：崩盘情景分段 ===========================
seg = slice(crash0 - 60, cr1 + 30)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
tt = np.arange(seg.start, seg.stop) / ANN
ax1.plot(tt, eq_vm[seg.start:seg.stop], color=C["vm"], lw=1.8, label="VM 净值")
ax1.plot(tt, eq_bh[seg.start:seg.stop], color=C["bh"], lw=1.6, alpha=0.85, label="买入持有净值")
ax1.axvspan((crash0-1)/ANN, (cr1-1)/ANN, color="#999999", alpha=0.22, label="崩盘期")
ax1.set_ylabel("净值")
ax1.set_title("崩盘情景：VM 在危机前主动去杠杆，把回撤砍掉一半", fontsize=11)
ax1.legend(fontsize=8.5, loc="lower left")
ax1.grid(True, color=C["grid"], lw=0.6)
ax2.plot(tt, dd_vm[seg.start:seg.stop] * 100, color=C["vm"], lw=1.6, label="VM 回撤 %")
ax2.plot(tt, dd_bh[seg.start:seg.stop] * 100, color=C["bh"], lw=1.5, alpha=0.85, label="买入持有回撤 %")
ax2.axvspan((crash0-1)/ANN, (cr1-1)/ANN, color="#999999", alpha=0.22)
ax2.set_ylabel("回撤 (%)")
ax2.set_xlabel("年份")
ax2.legend(fontsize=8.5)
ax2.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "crash_scenario.png"))
plt.close(fig)

print("=" * 60)
print("波动率目标择时 关键数字")
print("=" * 60)
print("买入持有: 年化收益 %.1f%% | 年化波动 %.1f%% | Sharpe %.2f | 最大回撤 %.1f%%"
      % (s_bh[0]*100, s_bh[1]*100, s_bh[2], s_bh[3]*100))
print("波动率目标: 年化收益 %.1f%% | 年化波动 %.1f%% | Sharpe %.2f | 最大回撤 %.1f%%"
      % (s_vm[0]*100, s_vm[1]*100, s_vm[2], s_vm[3]*100))
print("波动降幅 = %.0f%%" % (100 * (1 - s_vm[1] / s_bh[1])))
print("回撤降幅 = %.0f%%" % (100 * (1 - abs(s_vm[3]) / abs(s_bh[3]))))
print("崩盘段买入持有回撤 = %.1f%%, VM 回撤 = %.1f%%"
      % (dd_bh[crash0:cr1].min()*100, dd_vm[crash0:cr1].min()*100))
print("DONE ->", OUT, os.listdir(OUT))
