#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""谷歌趋势作为另类情绪信号: 用搜索量预测短期波动 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成日度搜索量指数(仿 Google Trends): 基线 50 + 多起恐慌/狂热尖峰(指数衰减) + 噪声
  * 资产: 日收益 r, 真实波动 sigma_t 由 GARCH(1,1) 驱动; 搜索量水平尖峰(恐慌)→未来数日波动抬升
    (恐慌搜索领先于波动, 与 Preis et al. 2013 / Vlastakis & Markellos 2012 的"水平信号"一致)
  * 用滞后搜索量水平(标准化)预测未来 5 日已实现波动, 计算样本外 R^2(温和为正, 非奇迹)
  * 对比: 高搜索量水平日 vs 低搜索量水平日后的实际波动差
  * 用搜索量构造"恐慌择时": 高搜索量期降低杠杆, 检验回撤改善
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

SLUG = "google-trends-sentiment"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "gold": "#C9A227", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
N = 1500                              # 1500 交易日(~6 年)

# ---------- 搜索量指数(自洽) ----------
trend = np.zeros(N)
events = {120: 45, 380: 60, 650: 50, 900: 70, 1150: 55, 1380: 48}  # 恐慌/狂热尖峰
shock = np.zeros(N)
for d, s in events.items():
    for k in range(10):
        if d + k < N:
            shock[d + k] += s * np.exp(-k / 3.5)
trend[0] = 50
for t in range(1, N):
    trend[t] = trend[t-1] + 0.20 * (50 - trend[t-1]) + rng.normal(0, 2.5) + shock[t]
trend = np.clip(trend, 5, 100)

# ---------- 资产: GARCH 波动 + 搜索量水平尖峰→波动 抬升 ----------
mu = 0.0004
omega, alpha, beta = 0.0000030, 0.10, 0.85     # persistence 0.95, 常态日波动 ~0.8%
sigma = np.zeros(N); sigma[0] = 0.012
ret = np.zeros(N)
for t in range(1, N):
    # 搜索量水平尖峰(恐慌)提升未来波动: 水平越高于基线, 波动乘数越大(封顶 +50%)
    excess = max(trend[t] - 50, 0) / 50.0          # 0 ~ 1
    vol_boost = 1.0 + 1.5 * excess
    sig2 = omega + alpha * ret[t-1]**2 + beta * sigma[t-1]**2
    sigma[t] = min(np.sqrt(max(sig2, 1e-8)) * vol_boost, 0.06)   # 封顶 6% 日波动
    ret[t] = mu + sigma[t] * rng.standard_normal()

rv5 = np.array([np.sqrt(np.sum(ret[t:t+5]**2)) for t in range(N-5)])   # 未来 5 日已实现波动

# ---------- 用滞后搜索量水平(标准化)预测未来波动 ----------
trend_std = (trend - trend.mean()) / trend.std()
X = np.column_stack([np.ones(N-6), trend_std[:N-6]])     # 用 t 日水平预测 t+1..t+5 波动
coef, *_ = np.linalg.lstsq(X, rv5[:N-6], rcond=None)
pred = X @ coef
ss_res = np.sum((rv5[:N-6] - pred)**2)
ss_tot = np.sum((rv5[:N-6] - rv5[:N-6].mean())**2)
oos_r2 = 1 - ss_res/ss_tot

# ---------- 高/低搜索量水平日后的实际波动 ----------
hi = trend_std[:N-6] > np.percentile(trend_std[:N-6], 75)
lo = trend_std[:N-6] < np.percentile(trend_std[:N-6], 25)
rv_hi = rv5[:N-6][hi].mean(); rv_lo = rv5[:N-6][lo].mean()

# ---------- 恐慌择时: 高搜索量水平期降杠杆 ----------
lev = np.where(trend[5:N] > np.percentile(trend[5:N], 75), 0.3, 1.0)
nav_full = np.cumprod(1 + ret[5:N])
nav_timing = np.cumprod(1 + lev * ret[5:N])

def ann(r): return (np.prod(1 + r))**(252/len(r)) - 1
def sharpe(r): return r.mean()/r.std()*np.sqrt(252)
def mdd(r):
    peak = np.maximum.accumulate(r); return (r/peak - 1).min()

summary = {
    "oos_r2": float(oos_r2), "trend_load": float(coef[1]),
    "rv_hi": float(rv_hi), "rv_lo": float(rv_lo),
    "full_ann": ann(ret[5:N]) if np.isfinite(ann(ret[5:N])) else 0.0,
    "timing_ann": ann(lev*ret[5:N]) if np.isfinite(ann(lev*ret[5:N])) else 0.0,
    "full_sharpe": sharpe(ret[5:N]), "timing_sharpe": sharpe(lev*ret[5:N]),
    "full_mdd": mdd(nav_full), "timing_mdd": mdd(nav_timing),
}

# ===================== 图 1: 搜索量路径 =====================
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(trend, color=C["orange"], lw=1.5, label="合成 Google Trends 搜索量指数")
for d, s in events.items():
    if d < N:
        ax.axvline(d, color=C["red"], ls=":", lw=0.8, alpha=0.5)
ax.set_title("合成搜索量指数: 恐慌/狂热尖峰(指数衰减)", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("搜索量(0-100)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/trend_path.png"); plt.close(fig)

# ===================== 图 2: 搜索量 vs 未来波动 =====================
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(rv5, color=C["net"], lw=1.0, alpha=0.6, label="未来 5 日已实现波动")
ax2 = ax.twinx()
ax2.plot(trend[5:N], color=C["orange"], lw=1.2, alpha=0.7, label="搜索量(右轴)")
ax.set_title("搜索量水平尖峰领先于未来波动上行(另类情绪前瞻)", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("已实现波动", color=C["net"])
ax2.set_ylabel("搜索量", color=C["orange"]); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/trend_vol.png"); plt.close(fig)

# ===================== 图 3: 预测能力回归 =====================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.scatter(trend_std[:N-6], rv5[:N-6], s=6, color=C["purple"], alpha=0.35)
xr = np.linspace(trend_std[:N-6].min(), trend_std[:N-6].max(), 50)
ax.plot(xr, coef[0] + coef[1]*xr, color=C["line"], lw=1.8)
ax.set_title(f"搜索量水平 → 未来 5 日波动: 样本外 R² = {oos_r2:.3f}", fontsize=12)
ax.set_xlabel("搜索量水平(标准化)"); ax.set_ylabel("未来 5 日已实现波动"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/trend_predict.png"); plt.close(fig)

# ===================== 图 4: 高/低搜索量水平日波动差 =====================
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(["高搜索量日", "低搜索量日"], [rv_hi*100, rv_lo*100], color=[C["red"], C["green"]])
for b, v in zip(bars, [rv_hi, rv_lo]):
    ax.text(b.get_x()+b.get_width()/2, v*100, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=10)
ax.set_title("高 vs 低搜索量日: 后续 5 日平均已实现波动", fontsize=12)
ax.set_ylabel("已实现波动(%)"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/regime_vol.png"); plt.close(fig)

# ===================== 图 5: 恐慌择时净值 =====================
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(nav_full, color=C["red"], lw=1.5, alpha=0.8, label=f"满仓(Sharpe {sharpe(ret[5:N]):.2f}, MDD {mdd(nav_full)*100:.0f}%)")
ax.plot(nav_timing, color=C["green"], lw=1.8, label=f"高搜索量降杠杆(Sharpe {sharpe(lev*ret[5:N]):.2f}, MDD {mdd(nav_timing)*100:.0f}%)")
ax.set_title("恐慌择时: 高搜索量期降杠杆, 回撤改善", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/vol_timing_nav.png"); plt.close(fig)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Google Trends 配图已生成:", os.listdir(OUT))
print(json.dumps(summary, indent=2, ensure_ascii=False))
