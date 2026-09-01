#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「截面动量衰减曲线：因子信号逐日腐烂的实证与对抗」
(momentum-decay-cross-section) 生成真实配图与核心数值。
所有图表均由文中 Python 代码真实计算生成（numpy 合成，固定 seed 可复现）。

机制（自洽合成，仅用于演示方法）：
  * 每个资产持有一个隐性「动量状态」m_i,t ~ AR(1)，φ=0.96（慢衰减）。
  * 日收益 = c·m_i,t + 特质噪声；于是过去收益是 m 的带噪代理，
    而 m 的 AR(1) 结构意味着信号对远期收益的预测力随持有期 h 以 ρ^h 衰减。
  * 截面信号 = 过去 20 日累计收益；计算其在持有期 h=1..60 上的 rank-IC 衰减曲线。
  * 对抗：对比「20 日信号 / 月度调仓 / 持有 20 日」的朴素动量与
    「5 日信号 / 周度调仓 / 持有 5 日」的鲜活信号动量，后者吃下更高短 horizon IC。
  * 图1：rank-IC 随持有期 h 的衰减曲线（含指数拟合半衰期）
  * 图2：信号衰减半衰期示意（IC(h) / IC(1) 归一化 + 指数包络）
  * 图3：朴素动量 vs 鲜活信号动量 净值曲线 + 最大回撤标注
  * 图4：逐月 rank-IC 时间序列（截面动量不稳定，需衰减感知）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
phi = 0.96
m = np.zeros((N, T))
m[:, 0] = rng.normal(0, 1, N)
for t in range(1, T):
    m[:, t] = phi * m[:, t - 1] + rng.normal(0, 1, N)
coef_m = 0.006
idio = 0.004
ret = 0.0003 + coef_m * m + idio * rng.normal(0, 1, (N, T))   # 日收益面板
# 去均值，避免漂移污染截面排序
ret = ret - ret.mean(axis=0, keepdims=True)

# ---------------- 1. rank-IC 衰减曲线 ----------------
def rank_ic(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y).correlation

lookback = 20
horizons = [1, 2, 3, 5, 10, 20, 30, 40, 60]
ic = []
for h in horizons:
    vals = []
    for t in range(60, T - h - 1):
        sig = ret[:, t - lookback:t].sum(axis=1)
        fwd = ret[:, t:t + h].sum(axis=1)
        vals.append(rank_ic(sig, fwd))
    ic.append(np.nanmean(vals))
ic = np.array(ic)

# 指数拟合半衰期：IC(h) ≈ IC0 * exp(-h/tau)
from scipy.optimize import curve_fit
def expf(h, ic0, tau):
    return ic0 * np.exp(-h / tau)
popt, _ = curve_fit(expf, horizons, ic, p0=[ic[0], 10])
tau = popt[1]
half_life = tau * np.log(2)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axhline(0, color="#888", lw=0.9)
ax.plot(horizons, ic, "o-", color=C["mom"], lw=2, label="实测 rank-IC(h)")
hh = np.linspace(1, 60, 200)
ax.plot(hh, expf(hh, *popt), "--", color=C["gold"], lw=1.8,
        label=f"指数拟合 τ={tau:.1f} 天 → 半衰期 {half_life:.1f} 天")
ax.set_xlabel("持有期 h（交易日）")
ax.set_ylabel("rank-IC（信号 = 过去 20 日收益）")
ax.set_title(f"截面动量衰减：信号预测力随持有期逐日腐烂（φ={phi}）")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ic_decay.png")); plt.close(fig)

# ---------------- 2. 归一化衰减 / 半衰期示意 ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
norm = ic / ic[0]
ax.plot(horizons, norm, "s-", color=C["mom"], lw=2, label="IC(h)/IC(1)")
env = np.exp(-np.array(horizons) / tau)
ax.plot(horizons, env, "--", color=C["gold"], lw=1.8, label=f"e^(-h/τ), τ={tau:.1f}")
ax.axhline(0.5, color=C["dd"], ls=":", lw=1.5, label=f"半衰期 h={half_life:.1f}")
ax.axvline(half_life, color=C["dd"], ls=":", lw=1.5)
ax.set_xlabel("持有期 h（交易日）")
ax.set_ylabel("归一化信号强度")
ax.set_title("信号腐烂的半衰期：过半预测力在约 17 个交易日内蒸发")
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

eq_naive = momentum_pf(20, 21, 21)     # 朴素：20d 信号 / 月调 / 持 20d
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
ax.plot(eq_naive, color=C["mom"], lw=1.6, label=f"朴素动量 (Sharpe {s_naive[0]:.2f}, DD {s_naive[2]*100:.0f}%)")
ax.plot(eq_fresh, color=C["fresh"], lw=1.6, label=f"鲜活信号 (Sharpe {s_fresh[0]:.2f}, DD {s_fresh[2]*100:.0f}%)")
peak = np.maximum.accumulate(eq_naive)
dd = eq_naive / peak - 1
ax.fill_between(np.arange(len(eq_naive)), eq_naive, peak, where=(dd < 0),
                color=C["dd"], alpha=0.15)
ax.set_yscale("log")
ax.set_xlabel("交易日")
ax.set_ylabel("净值（对数轴）")
ax.set_title("对抗衰减：鲜活短信号 + 高频调仓跑赢朴素动量")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_compare.png")); plt.close(fig)

# ---------------- 4. 逐月 rank-IC 时间序列（不稳定） ----------------
monthly_ic = []
for t in range(60, T - 21 - 1, 21):
    sig = ret[:, t - 20:t].sum(axis=1)
    fwd = ret[:, t:t + 21].sum(axis=1)
    monthly_ic.append(rank_ic(sig, fwd))
monthly_ic = np.array(monthly_ic)
pos_rate = (monthly_ic > 0).mean()
neg_streak = int(np.max(np.diff(np.where(np.diff((monthly_ic < 0).astype(int)) == 1)[0]) - 1)) if (monthly_ic < 0).any() else 0

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axhline(0, color="#888", lw=0.9)
ax.plot(monthly_ic, color=C["purple"], lw=1.4)
ax.axhline(monthly_ic.mean(), color=C["gold"], ls="--", lw=1.6,
           label=f"均值 {monthly_ic.mean():.3f}（正占比 {pos_rate*100:.0f}%）")
ax.set_xlabel("调仓月序号")
ax.set_ylabel("当月 rank-IC（20d 信号 → 次月收益）")
ax.set_title("截面动量并不稳定：约 1/4 月份 IC 转负，需衰减感知")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "monthly_ic.png")); plt.close(fig)

print("=" * 60)
print("ARTICLE_A_MOMENTUM_DECAY_METRICS")
print(f"N={N} assets, T={T} days, phi={phi}")
print(f"IC@h=1={ic[0]:.4f}  IC@h=3={ic[2]:.4f}  IC@h=10={ic[4]:.4f}  "
      f"IC@h=20={ic[5]:.4f}  IC@h=40={ic[7]:.4f}  IC@h=60={ic[8]:.4f}")
print(f"tau={tau:.2f} days, half_life={half_life:.2f} days")
print(f"naive Sharpe={s_naive[0]:.3f} annRet={s_naive[1]*100:.1f}% maxDD={s_naive[2]*100:.1f}%")
print(f"fresh Sharpe={s_fresh[0]:.3f} annRet={s_fresh[1]*100:.1f}% maxDD={s_fresh[2]*100:.1f}%")
print(f"monthly IC mean={monthly_ic.mean():.4f} pos_rate={pos_rate*100:.0f}%")
print("=" * 60)
