#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「截面动量衰减曲线：因子信号逐日腐烂的实证与对抗」
(momentum-decay-cross-section) 生成真实配图与核心数值。
所有图表均由文中 Python 代码真实计算生成（numpy 合成，固定 seed 可复现）。

机制（自洽合成，仅用于演示方法）：
  * 每个资产持有一个隐性「动量状态」m_i,t ~ AR(1)，φ=0.94（慢衰减）。
  * 日收益 = β·market_t + c·m_i,t + 特质噪声；market_t 为共同因子（长多空天然中性），
    m 是弱信号（c 很小），于是截面信号 rank-IC 落在真实量级的 0.05–0.10。
  * 因 m 是 AR(1)，过去收益对「k 期后」收益的预测力随持有期 h 以 ρ^h 衰减。
  * 截面信号 = 过去 5 日累计收益；计算其在持有期 h=1..60 上的 rank-IC 衰减曲线。
  * 对抗：对比「5 日信号 / 月调仓 / 持有 20 日」的普通动量与
    「5 日信号 / 周度调仓 / 持有 5 日」的鲜活信号动量，后者吃下更高短 horizon IC。
  * 图1：rank-IC 随持有期 h 的衰减曲线（含指数拟合半衰期）
  * 图2：信号衰减半衰期示意（IC(h) / IC(1) 归一化 + 指数包络）
  * 图3：普通动量 vs 鲜活信号动量 净值曲线 + 最大回撤标注
  * 图4：逐月 rank-IC 时间序列（截面动量不稳定，需衰减感知）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

SLUG = "momentum-decay-cross-section"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"mom": "#4C72B0", "fresh": "#55A868", "grid": "#DDDDDD",
     "dd": "#C44E52", "dark": "#333333", "gold": "#DD8452", "purple": "#8172B3"}

rng = np.random.default_rng(20260901)
N, T = 60, 2200
phi = 0.94
beta = 0.8
m = np.zeros((N, T))
m[:, 0] = rng.normal(0, 1, N)
for t in range(1, T):
    m[:, t] = phi * m[:, t - 1] + rng.normal(0, 1, N)
coef_m = 0.0009
idio = 0.010
mkt = 0.0003 + 0.010 * rng.normal(T)
ret = beta * mkt + coef_m * m + idio * rng.normal(0, 1, (N, T))   # 日收益面板
# 截面长多空天然中性：不刻意去均值，信号来自 m 的相对强弱

# ---------------- 1. rank-IC 衰减曲线 ----------------
lookback = 5
horizons = [1, 2, 3, 5, 10, 20, 30, 40, 60]
ic = []
for h in horizons:
    vals = []
    for t in range(60, T - h - 1):
        sig = ret[:, t - lookback:t].sum(axis=1)
        fwd = ret[:, t:t + h].sum(axis=1)
        vals.append(spearmanr(sig, fwd).correlation)
    ic.append(np.nanmean(vals))
ic = np.array(ic)

# 指数拟合半衰期：对 IC(h) 取对数后做线性最小二乘（更稳健，不依赖 curve_fit 初值）
log_ic = np.log(ic)
slope, intercept = np.polyfit(horizons, log_ic, 1)
tau = -1.0 / slope if slope < 0 else np.inf
half_life = tau * np.log(2) if np.isfinite(tau) else np.nan

def expf(h, ic0, tau_):
    return ic0 * np.exp(-h / tau_)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axhline(0, color="#888", lw=0.9)
ax.plot(horizons, ic, "o-", color=C["mom"], lw=2, label="实测 rank-IC(h)")
if np.isfinite(tau):
    hh = np.linspace(1, 60, 200)
    ax.plot(hh, expf(hh, ic[0], tau), "--", color=C["gold"], lw=1.8,
            label=f"指数拟合 τ={tau:.1f} 天 → 半衰期 {half_life:.1f} 天")
ax.set_xlabel("持有期 h（交易日）")
ax.set_ylabel("rank-IC（信号 = 过去 5 日收益）")
ax.set_title(f"截面动量衰减：信号预测力随持有期滚动变化（φ={phi}）")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ic_decay.png")); plt.close(fig)

# ---------------- 2. 信号生命周期：归一化到峰值 + 长持有期衰减 ----------------
peak_idx = int(np.argmax(ic))
peak_h = horizons[peak_idx]
ic_peak = ic[peak_idx]
# 只对峰值之后的点做指数衰减拟合（长持有期尾段），得到真实衰减速率
tail_mask = np.array(horizons) >= peak_h
tail_h = np.array(horizons)[tail_mask]
tail_ic = ic[tail_mask]
tail_slope, _ = np.polyfit(tail_h, np.log(tail_ic), 1)
tau_tail = -1.0 / tail_slope if tail_slope < 0 else np.inf
tail_half = tau_tail * np.log(2) if np.isfinite(tau_tail) else np.nan

fig, ax = plt.subplots(figsize=(7.4, 4.4))
norm = ic / ic_peak
ax.plot(horizons, norm, "s-", color=C["mom"], lw=2, label="IC(h)/IC(峰值)")
if np.isfinite(tau_tail):
    env = np.exp(-(tail_h - peak_h) / tau_tail)
    ax.plot(tail_h, env, "--", color=C["gold"], lw=1.8,
            label=f"峰后衰减 τ≈{tau_tail:.0f} 天（半衰期 {tail_half:.0f} 天）")
ax.axvline(peak_h, color=C["dd"], ls=":", lw=1.5, label=f"纯动量峰值 h={peak_h}")
ax.set_xlabel("持有期 h（交易日）")
ax.set_ylabel("相对峰值的信号强度")
ax.set_title("动量的生命周期：短窗重叠压低 IC，纯动量区达峰后随持有期衰减")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "half_life.png")); plt.close(fig)

# ---------------- 3. 两种动量组合净值 ----------------
def momentum_pf(sig_lookback, hold, rebal):
    eq = 1.0
    curve = [1.0]
    t = sig_lookback + 1
    while t + hold < T:
        sig = ret[:, t - sig_lookback:t].sum(axis=1)
        order = np.argsort(sig)
        longs = order[-N // 10:]
        shorts = order[:N // 10]
        for s in range(hold):
            r = ret[longs, t + s].mean() - ret[shorts, t + s].mean()
            eq *= (1 + r)
            curve.append(eq)
        t += rebal
    return np.array(curve)

eq_naive = momentum_pf(5, 21, 21)     # 普通：5d 信号 / 月调 / 持 20d
eq_fresh = momentum_pf(5, 5, 5)        # 鲜活：5d 信号 / 周调 / 持 5d

def stats(curve, ann=252):
    r = curve[1:] / curve[:-1] - 1
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0
    ann_ret = curve[-1] ** (ann / len(r)) - 1 if len(r) > 0 else 0
    peak = np.maximum.accumulate(curve)
    mdd = (curve / peak - 1).min()
    return sharpe, ann_ret, mdd

s_naive = stats(eq_naive)
s_fresh = stats(eq_fresh)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(eq_naive, color=C["mom"], lw=1.6, label=f"普通动量 (Sharpe {s_naive[0]:.2f}, DD {s_naive[2]*100:.0f}%)")
ax.plot(eq_fresh, color=C["fresh"], lw=1.6, label=f"鲜活信号 (Sharpe {s_fresh[0]:.2f}, DD {s_fresh[2]*100:.0f}%)")
peak = np.maximum.accumulate(eq_naive)
dd = eq_naive / peak - 1
ax.fill_between(np.arange(len(eq_naive)), eq_naive, peak, where=(dd < 0),
                color=C["dd"], alpha=0.15)
ax.set_yscale("log")
ax.set_xlabel("交易日")
ax.set_ylabel("净值（对数轴）")
ax.set_title("对抗衰减：鲜活短信号 + 高频调仓跑赢普通动量")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_compare.png")); plt.close(fig)

# ---------------- 4. 逐月 rank-IC 时间序列（不稳定） ----------------
monthly_ic = []
for t in range(60, T - 21 - 1, 21):
    sig = ret[:, t - 5:t].sum(axis=1)
    fwd = ret[:, t:t + 21].sum(axis=1)
    monthly_ic.append(spearmanr(sig, fwd).correlation)
monthly_ic = np.array(monthly_ic)
pos_rate = (monthly_ic > 0).mean()

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axhline(0, color="#888", lw=0.9)
ax.plot(monthly_ic, color=C["purple"], lw=1.4)
ax.axhline(monthly_ic.mean(), color=C["gold"], ls="--", lw=1.6,
           label=f"均值 {monthly_ic.mean():.3f}（正占比 {pos_rate*100:.0f}%）")
ax.set_xlabel("调仓月序号")
ax.set_ylabel("当月 rank-IC（5d 信号 → 次月收益）")
ax.set_title("截面动量并不稳定：部分月份 IC 转负，需衰减感知")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "monthly_ic.png")); plt.close(fig)

print("=" * 60)
print("ARTICLE_A_MOMENTUM_DECAY_METRICS")
print(f"N={N} assets, T={T} days, phi={phi}")
print(f"IC@h=1={ic[0]:.4f}  IC@h=3={ic[2]:.4f}  IC@h=10={ic[4]:.4f}  "
      f"IC@h=20={ic[5]:.4f}  IC@h=40={ic[7]:.4f}  IC@h=60={ic[8]:.4f}")
print(f"IC peak at h={peak_h} (IC={ic_peak:.4f}); tail decay tau={tau_tail:.1f} -> half_life={tail_half:.1f}")
print(f"naive Sharpe={s_naive[0]:.3f} annRet={s_naive[1]*100:.1f}% maxDD={s_naive[2]*100:.1f}%")
print(f"fresh Sharpe={s_fresh[0]:.3f} annRet={s_fresh[1]*100:.1f}% maxDD={s_fresh[2]*100:.1f}%")
print(f"monthly IC mean={monthly_ic.mean():.4f} pos_rate={pos_rate*100:.0f}%")
print("=" * 60)
