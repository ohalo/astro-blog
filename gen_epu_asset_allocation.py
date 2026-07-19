#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""经济政策不确定性 EPU 与股债配置切换 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成月度 EPU 指数(仿 Baker-Bloom-Davis): 基线 ~100 + 多起政策事件尖峰(大选/债务上限/加息周期)衰减
  * 两类资产:
        股票 r_e: 正 drift + 市场 beta, 对 EPU 月变化 dEPU 负向加载(政策不确定→股票跌)
        长债 r_b: 低 drift + 负 beta, 对 dEPU 正向加载(避险→债券涨, flight-to-quality)
        现金 r_c: 无风险月息
  * EPU 配置策略: 用月末已知 EPU[t-1] 决定第 t 月股票/债券暴露(高 EPU 切债券), 无前瞻
  * 对比净值: 买入持有股票 vs 60/40 股债 vs EPU 切换
  * 横截面/回归: 股票/债券月收益 ~ dEPU, 斜率为 EPU 风险暴露
  * 机制分解: 高 EPU 月 vs 低 EPU 月的平均收益差(债券相对占优)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "epu-asset-allocation"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "gold": "#C9A227", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
T = 360                               # 360 月(30 年)

# ---------- EPU 指数 ----------
epu = np.zeros(T)
events = {24: 180, 70: 140, 130: 210, 200: 120, 260: 170, 320: 150}   # 政策事件尖峰幅度
event_shock = np.zeros(T)
for m, s in events.items():
    for k in range(8):
        if m + k < T:
            event_shock[m + k] += s * np.exp(-k / 3.0)
epu[0] = 100
for t in range(1, T):
    epu[t] = epu[t-1] + 0.30 * (100 - epu[t-1]) + rng.normal(0, 6) + event_shock[t]
epu = np.maximum(epu, 20)
dep = np.diff(epu)                    # 月度 EPU 变化 (T-1,)

# ---------- 资产收益 ----------
MKT = rng.normal(0.006, 0.035, T-1)
r_e = 0.008 + MKT + (-0.0009) * dep + rng.normal(0, 0.040, T-1)   # 股票: EPU↑ 跌
r_b = 0.003 - 0.10 * MKT + (0.00075) * dep + rng.normal(0, 0.018, T-1)  # 债券: EPU↑ 涨
r_c = 0.0020 * np.ones(T-1)           # 现金无风险

# ---------- EPU 配置策略 (无前瞻) ----------
th = np.percentile(epu, 75)           # 高 EPU 阈值 = 75 分位
epu_lag = epu[:-1]                    # 用 t-1 月末 EPU 决定 t 月暴露
w_eq = np.where(epu_lag <= th, 0.6, 0.0)    # 高 EPU 时股票清零
w_bd = np.where(epu_lag <= th, 0.4, 1.0)    # 切到债券(含原 40%)
strat = w_eq * r_e + w_bd * r_b
# 60/40 基准
bench = 0.6 * r_e + 0.4 * r_b

# ---------- 净值 ----------
nav_e = np.cumprod(1 + r_e)
nav_b = np.cumprod(1 + r_b)
nav_strat = np.cumprod(1 + strat)
nav_bench = np.cumprod(1 + bench)

def ann(r): return (np.prod(1 + r))**(12/len(r)) - 1
def sharpe(r): return r.mean()/r.std()*np.sqrt(12)
def mdd(r):
    peak = np.maximum.accumulate(r); return (r/peak - 1).min()

# ---------- EPU 风险暴露回归 ----------
X = np.column_stack([np.ones(T-1), dep, MKT])
coef_e, *_ = np.linalg.lstsq(X, r_e, rcond=None)
coef_b, *_ = np.linalg.lstsq(X, r_b, rcond=None)

# ---------- 高/低 EPU 月收益差 ----------
hi = dep > np.percentile(dep, 75)
lo = dep < np.percentile(dep, 25)
eq_hi = r_e[hi].mean(); eq_lo = r_e[lo].mean()
bd_hi = r_b[hi].mean(); bd_lo = r_b[lo].mean()

summary = {
    "epu_load_eq": float(coef_e[1]), "epu_load_bd": float(coef_b[1]),
    "eq_ann": ann(r_e), "bd_ann": ann(r_b),
    "strat_ann": ann(strat), "bench_ann": ann(bench),
    "strat_sharpe": sharpe(strat), "bench_sharpe": sharpe(bench),
    "eq_sharpe": sharpe(r_e), "bd_sharpe": sharpe(r_b),
    "strat_mdd": mdd(nav_strat), "bench_mdd": mdd(nav_bench), "eq_mdd": mdd(nav_e),
    "eq_hi": eq_hi, "eq_lo": eq_lo, "bd_hi": bd_hi, "bd_lo": bd_lo,
}

# ===================== 图 1: EPU 路径 =====================
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(epu, color=C["purple"], lw=1.8, label="合成 EPU 指数")
for m, s in events.items():
    if m < T:
        ax.axvline(m, color=C["red"], ls=":", lw=0.8, alpha=0.5)
ax.axhline(th, color=C["line"], ls="--", lw=1.0, label=f"高 EPU 阈值(75 分位)={th:.0f}")
ax.set_title("合成经济政策不确定性 EPU 指数(政策事件尖峰 + 均值回复)", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("EPU")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/epu_path.png"); plt.close(fig)

# ===================== 图 2: 股票 vs 债券净值 =====================
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(nav_e, color=C["red"], lw=1.8, label=f"股票(年化 {ann(r_e)*100:.1f}%)")
ax.plot(nav_b, color=C["green"], lw=1.8, label=f"长债(年化 {ann(r_b)*100:.1f}%)")
ax.set_title("股票 vs 长债净值: EPU 尖峰期债券充当避险垫", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/assets_nav.png"); plt.close(fig)

# ===================== 图 3: EPU 配置策略净值 =====================
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(nav_e, color=C["red"], lw=1.3, alpha=0.7, label=f"买入持有股票(Sharpe {sharpe(r_e):.2f})")
ax.plot(nav_bench, color=C["orange"], lw=1.6, label=f"60/40 股债(Sharpe {sharpe(bench):.2f})")
ax.plot(nav_strat, color=C["purple"], lw=1.8, label=f"EPU 切换(Sharpe {sharpe(strat):.2f})")
ax.set_title("EPU 股债配置切换: 高 EPU 切债券能否跑赢 60/40?", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/epu_timing_nav.png"); plt.close(fig)

# ===================== 图 4: EPU 风险暴露回归 =====================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
xr = np.linspace(dep.min(), dep.max(), 50)
axes[0].scatter(dep, r_e, s=8, color=C["red"], alpha=0.4)
axes[0].plot(xr, coef_e[0] + coef_e[1]*xr + coef_e[2]*dep.mean(), color=C["line"], lw=1.5)
axes[0].set_title(f"股票月收益 ~ Δ EPU\n斜率(暴露)={coef_e[1]:.5f} <0", fontsize=11)
axes[0].set_xlabel("Δ EPU"); axes[0].set_ylabel("股票月收益"); axes[0].grid(alpha=0.3)
axes[1].scatter(dep, r_b, s=8, color=C["green"], alpha=0.4)
axes[1].plot(xr, coef_b[0] + coef_b[1]*xr + coef_b[2]*dep.mean(), color=C["line"], lw=1.5)
axes[1].set_title(f"债券月收益 ~ Δ EPU\n斜率(暴露)={coef_b[1]:.5f} >0", fontsize=11)
axes[1].set_xlabel("Δ EPU"); axes[1].set_ylabel("债券月收益"); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/epu_regression.png"); plt.close(fig)

# ===================== 图 5: 高/低 EPU 月收益差 =====================
fig, ax = plt.subplots(figsize=(8, 4.2))
labels = ["股票(高-低 EPU)", "债券(高-低 EPU)"]
vals = [eq_hi - eq_lo, bd_hi - bd_lo]
cols = [C["red"], C["green"]]
bars = ax.bar(labels, [v*100 for v in vals], color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v*100, f"{v*100:+.2f}%", ha="center", va="bottom", fontsize=10)
ax.axhline(0, color=C["line"], lw=1.0)
ax.set_title("高 EPU 月 vs 低 EPU 月: 平均月收益差", fontsize=12)
ax.set_ylabel("月收益差(%)"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/regime_returns.png"); plt.close(fig)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("EPU 配图已生成:", os.listdir(OUT))
print(json.dumps(summary, indent=2, ensure_ascii=False))
