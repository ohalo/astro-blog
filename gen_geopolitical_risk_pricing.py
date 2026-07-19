#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地缘政治风险指数 GPR 与资产定价 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成月度 GPR 指数(仿 Caldara & Iacoviello 2022): 基线 ~100 + 多起地缘事件尖峰(战争/恐袭/贸易战)衰减
  * 两类资产:
        股票 r_e: 正 drift + 市场 beta, 对 GPR 月变化 dgpr 负向加载(风险-off 下跌)
        黄金 r_g: 低 drift + 低 beta, 对 dgpr 正向加载(避险上行)
        现金 r_c: 无风险月息
  * GPR 择时策略: 用月末已知 GPR[t-1] 决定第 t 月股票暴露(高 GPR 切现金), 无前瞻
  * 对比净值: 买入持有股票 vs GPR 择时(股/现金) vs 黄金
  * 横截面/回归: 股票月收益 ~ dgpr, 斜率为 GPR 风险暴露(应为负)
  * 机制分解: 高 GPR 月 vs 低 GPR 月的平均收益差(避险资产相对占优)
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

SLUG = "geopolitical-risk-pricing"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "gold": "#C9A227", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
T = 240                               # 240 月(20 年)

# ---------- GPR 指数 ----------
gpr = np.zeros(T)
events = {18: 220, 55: 150, 110: 250, 150: 110, 200: 190}   # 地缘事件尖峰幅度
event_shock = np.zeros(T)
for m, s in events.items():
    for k in range(6):
        if m + k < T:
            event_shock[m + k] += s * np.exp(-k / 2.2)
gpr[0] = 100
for t in range(1, T):
    gpr[t] = gpr[t-1] + 0.25 * (100 - gpr[t-1]) + rng.normal(0, 4) + event_shock[t]
gpr = np.maximum(gpr, 20)
dgpr = np.diff(gpr)                   # 月度 GPR 变化 (T-1,)

# ---------- 资产收益 ----------
MKT = rng.normal(0.006, 0.03, T-1)
r_e = 0.009 + MKT + (-0.0011) * dgpr + rng.normal(0, 0.035, T-1)   # 股票: GPR↑ 跌
r_g = 0.003 + 0.15 * MKT + (0.00085) * dgpr + rng.normal(0, 0.022, T-1)  # 黄金: GPR↑ 涨
r_c = 0.0015 * np.ones(T-1)           # 现金无风险

# ---------- GPR 择时策略 (无前瞻) ----------
th = np.percentile(gpr, 80)           # 高 GPR 阈值 = 80 分位
exposure = np.where(gpr[:-1] <= th, 1.0, 0.0)   # 用 t-1 月末 GPR 决定 t 月股票暴露
strat = exposure * r_e + (1 - exposure) * r_c

# ---------- 黄金套保叠加 (无前瞻, 保留股票 beta) ----------
wg = np.where(gpr[:-1] <= th, 0.0, 0.40)   # 高 GPR 时把 40% 仓位从股票切到黄金
strat2 = (1 - wg) * r_e + wg * r_g
nav_strat2 = np.cumprod(1 + strat2)

# ---------- 净值 ----------
nav_e = np.cumprod(1 + r_e)
nav_g = np.cumprod(1 + r_g)
nav_strat = np.cumprod(1 + strat)

def ann(r): return (np.prod(1 + r))**(12/len(r)) - 1
def sharpe(r): return r.mean()/r.std()*np.sqrt(12)
def mdd(r):
    peak = np.maximum.accumulate(r); return (r/peak - 1).min()

# ---------- GPR 风险暴露回归 ----------
X = np.column_stack([np.ones(T-1), dgpr, MKT])
coef, *_ = np.linalg.lstsq(X, r_e, rcond=None)
gpr_load = coef[1]

# ---------- 高/低 GPR 月收益差 ----------
hi = dgpr > np.percentile(dgpr, 80)
lo = dgpr < np.percentile(dgpr, 20)
eq_hi = r_e[hi].mean(); eq_lo = r_e[lo].mean()
gold_hi = r_g[hi].mean(); gold_lo = r_g[lo].mean()

summary = {
    "gpr_load": float(gpr_load), "mkt_load": float(coef[2]),
    "eq_ann": ann(r_e), "gold_ann": ann(r_g), "strat_ann": ann(strat),
    "strat2_ann": ann(strat2), "eq_sharpe": sharpe(r_e), "strat_sharpe": sharpe(strat),
    "strat2_sharpe": sharpe(strat2),
    "eq_mdd": mdd(nav_e), "strat_mdd": mdd(nav_strat), "strat2_mdd": mdd(nav_strat2),
    "eq_hi": float(eq_hi), "eq_lo": float(eq_lo),
    "gold_hi": float(gold_hi), "gold_lo": float(gold_lo),
    "threshold": float(th), "high_gpr_months": int(hi.sum()),
}
print(json.dumps(summary, indent=2))

# ================= 图 1: GPR 指数路径 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(gpr, color=C["red"], lw=1.8)
ax.axhline(th, color=C["line"], ls="--", lw=1.2, label=f"高 GPR 阈值(80 分位)={th:.0f}")
lbls = {18: "地区冲突", 55: "贸易摩擦", 110: "全面战争", 150: "恐袭", 200: "地缘升级"}
for m, s in events.items():
    ax.annotate(lbls[m], (m, gpr[m]), fontsize=8, color=C["red"],
                xytext=(0, 8), textcoords="offset points", ha="center")
ax.set_title("合成地缘政治风险指数 GPR(事件尖峰 + 均值回复)", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("GPR 指数")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/gpr_path.png"); plt.close(fig)

# ================= 图 2: 股票 vs 黄金 净值 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(nav_e, color=C["net"], lw=2, label=f"股票 {ann(r_e)*100:.1f}%/yr")
ax.plot(nav_g, color=C["gold"], lw=2, label=f"黄金 {ann(r_g)*100:.1f}%/yr")
ax.axhline(1.0, color=C["grid"], lw=1)
ax.set_title("股票 vs 黄金净值: GPR 尖峰期黄金充当避险垫", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/assets_nav.png"); plt.close(fig)

# ================= 图 3: GPR 择时策略 vs 买入持有 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(nav_strat, color=C["green"], lw=2, label=f"GPR 择时(股/现金) {ann(strat)*100:.1f}%/yr")
ax.plot(nav_strat2, color=C["purple"], lw=2, label=f"黄金套保叠加(股·黄金) {ann(strat2)*100:.1f}%/yr")
ax.plot(nav_e, color=C["net"], lw=1.6, ls="--", label=f"买入持有股票 {ann(r_e)*100:.1f}%/yr")
ax.axhline(1.0, color=C["grid"], lw=1)
ax.set_title("GPR 择时策略净值: 高 GPR 切现金, 砍掉尾部回撤", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/gpr_timing_nav.png"); plt.close(fig)

# ================= 图 4: GPR 变化 vs 股票收益 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.scatter(dgpr, r_e*100, s=18, alpha=0.5, color=C["net"])
xs = np.linspace(dgpr.min(), dgpr.max(), 50)
ax.plot(xs, (coef[0] + coef[1]*xs + coef[2]*MKT.mean())*100, color=C["red"], lw=2,
        label=f"GPR 风险暴露={gpr_load:.4f}")
ax.set_title("股票月收益 ~ GPR 变化: 斜率为负(风险-off)", fontsize=12)
ax.set_xlabel("GPR 月度变化 ΔGPR"); ax.set_ylabel("股票月收益 (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/gpr_eq_scatter.png"); plt.close(fig)

# ================= 图 5: 高/低 GPR 月平均收益 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
xpos = np.arange(2); w = 0.35
ax.bar(xpos - w/2, [eq_lo*100, eq_hi*100], w, color=C["net"], label="股票")
ax.bar(xpos + w/2, [gold_lo*100, gold_hi*100], w, color=C["gold"], label="黄金")
ax.set_xticks(xpos); ax.set_xticklabels(["低 GPR 月", "高 GPR 月"])
ax.axhline(0, color=C["line"], lw=1)
ax.set_title("高 vs 低 GPR 月: 股票回撤、黄金占优", fontsize=12)
ax.set_ylabel("月均收益 (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/regime_returns.png"); plt.close(fig)

print("IMAGES_WRITTEN:", sorted(os.listdir(OUT)))
