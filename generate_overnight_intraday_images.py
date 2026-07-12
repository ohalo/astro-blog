#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「隔夜效应与盘中效应：把收益拆成『收盘到开盘』和『开盘到收盘』」
(overnight-intraday-anomaly) 生成真实配图与真实统计数字。

所有图表与数字均由文中 Python 逻辑真实计算生成：
  1) overnight_cum.png      —— 三类净值：隔夜-only / 盘中-only / 买入持有(全时段)
  2) overnight_hist.png     —— 隔夜 vs 盘中 日收益分布(均值明显错位)
  3) overnight_strategy.png —— 隔夜持有策略净值 vs 买入持有，标注收益/波动/回撤
  4) overnight_weekday.png  —— 按星期几拆开：隔夜与盘中的平均收益(含周末效应)

机制(用对数收益可加性拆解一天)：
  r_total_t = (log open_t - log close_{t-1})   # 隔夜：前一日收盘 → 当日开盘
            + (log close_t - log open_t)         # 盘中：当日开盘 → 当日收盘
金融里著名的「隔夜效应」：权益溢价几乎全部来自隔夜段，盘中段接近 0 甚至为负。
本脚本用固定种子合成 30 年×252 日 ≈ 7560 个交易日，埋入该 Stylized Fact：
  - 隔夜段：正漂移 μ_o≈+0.00030/日；桥接周末的隔夜(周五收盘→周一开盘)额外 +0.00150
  - 盘中段：微负漂移 μ_i≈-0.00003/日
  - 隔夜波动 0.5%/日、盘中波动 0.6%/日
仅为演示机制，非真实行情。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "overnight-intraday-anomaly")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E2E2E2",
     "overnight": "#4C72B0", "intraday": "#DD8452", "total": "#2F4B7C"}

# =====================================================================
# 1) 合成日度收益，拆分为隔夜段与盘中段
# =====================================================================
def simulate(n_days=7560, seed=20260712):
    rng = np.random.default_rng(seed)
    # 连续工作日：第 k 天星期 = k%5 (0=周一 ... 4=周五)
    weekday = np.arange(n_days) % 5
    # 隔夜漂移：周一的隔夜桥接周末(周五收盘→周一开盘)额外 boost
    mu_o = np.full(n_days, 0.00030)
    mu_o[weekday == 0] += 0.00150          # 周末效应：周一隔夜显著更高
    mu_i = np.full(n_days, -0.00003)        # 盘中微负漂移
    sig_o, sig_i = 0.005, 0.006
    r_o = rng.normal(mu_o, sig_o, n_days)    # 隔夜收益
    r_i = rng.normal(mu_i, sig_i, n_days)    # 盘中收益
    r_total = r_o + r_i
    return r_o, r_i, r_total, weekday

r_o, r_i, r_total, weekday = simulate()

# 三类策略净值
nv_overnight = np.cumprod(1 + r_o)          # 只持隔夜：收盘买、开盘卖
nv_intraday = np.cumprod(1 + r_i)           # 只持盘中：开盘买、收盘卖
nv_total = np.cumprod(1 + r_total)          # 买入持有(全时段)

def ann_ret(nv, ppy=252):
    yrs = (len(nv) - 1) / ppy
    return (nv[-1] / nv[0]) ** (1 / yrs) - 1

def vol(nv, ppy=252):
    rets = np.diff(nv) / nv[:-1]
    return rets.std() * np.sqrt(ppy)

def sharpe(nv, ppy=252):
    rets = np.diff(nv) / nv[:-1]
    return rets.mean() / rets.std() * np.sqrt(ppy)

def max_dd(nv):
    peak = np.maximum.accumulate(nv)
    return (nv / peak - 1).min()

ar_o, ar_i, ar_t = ann_ret(nv_overnight), ann_ret(nv_intraday), ann_ret(nv_total)
vol_o, vol_i, vol_t = vol(nv_overnight), vol(nv_intraday), vol(nv_total)
sh_o, sh_i, sh_t = sharpe(nv_overnight), sharpe(nv_intraday), sharpe(nv_total)
mdd_o, mdd_i, mdd_t = max_dd(nv_overnight), max_dd(nv_intraday), max_dd(nv_total)

# 隔夜贡献占比
contrib = r_o.sum() / r_total.sum()

print(f"[合成] {len(r_o)} 交易日 | 隔夜年化={ar_o*100:.2f}% 盘中年化={ar_i*100:.2f}% 全时段年化={ar_t*100:.2f}%")
print(f"[波动] 隔夜 σ={vol_o*100:.2f}% 盘中 σ={vol_i*100:.2f}% 全时段 σ={vol_t*100:.2f}%")
print(f"[Sharpe] 隔夜={sh_o:.2f} 盘中={sh_i:.2f} 全时段={sh_t:.2f}")
print(f"[最大回撤] 隔夜={mdd_o*100:.2f}% 盘中={mdd_i*100:.2f}% 全时段={mdd_t*100:.2f}%")
print(f"[隔夜贡献] 总收益中 {contrib*100:.1f}% 来自隔夜段")

# =====================================================================
# 2) 图1：三类净值
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(nv_overnight, label=f"隔夜-only 年化{ar_o*100:.1f}%", color=C["overnight"], lw=2)
ax.plot(nv_intraday, label=f"盘中-only 年化{ar_i*100:.1f}%", color=C["intraday"], lw=2)
ax.plot(nv_total, label=f"买入持有 年化{ar_t*100:.1f}%", color=C["total"], lw=2, ls="--")
ax.set_yscale("log")
ax.set_xlabel("交易日"); ax.set_ylabel("净值(对数轴, 初始=1)")
ax.set_title("隔夜-only 几乎吃下全部权益溢价，盘中段是拖累")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "overnight_cum.png"), dpi=130); plt.close()

# =====================================================================
# 3) 图2：隔夜 vs 盘中 日收益分布
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
bins = np.linspace(-0.04, 0.04, 60)
ax.hist(r_o, bins=bins, alpha=0.6, color=C["overnight"], label=f"隔夜 均值{r_o.mean()*100:.3f}%")
ax.hist(r_i, bins=bins, alpha=0.6, color=C["intraday"], label=f"盘中 均值{r_i.mean()*100:.3f}%")
ax.axvline(r_o.mean(), color=C["overnight"], ls="--", lw=1.5)
ax.axvline(r_i.mean(), color=C["intraday"], ls="--", lw=1.5)
ax.set_xlabel("日度收益"); ax.set_ylabel("频数")
ax.set_title("两段均值明显错位：隔夜在 0 之上，盘中外 0 之下")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "overnight_hist.png"), dpi=130); plt.close()

# =====================================================================
# 4) 图3：隔夜持有策略 vs 买入持有(线性轴，标注指标)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(nv_overnight, label=f"隔夜持有 Sharpe={sh_o:.2f}", color=C["overnight"], lw=2)
ax.plot(nv_total, label=f"买入持有 Sharpe={sh_t:.2f}", color=C["total"], lw=2, ls="--")
ax.set_xlabel("交易日"); ax.set_ylabel("净值(初始=1)")
ax.set_title("同样收益、更低波动：隔夜持有 Sharpe 远高于买入持有")
txt = (f"隔夜-only: 年化 {ar_o*100:.1f}% / σ {vol_o*100:.1f}% / 回撤 {mdd_o*100:.1f}%\n"
        f"买入持有: 年化 {ar_t*100:.1f}% / σ {vol_t*100:.1f}% / 回撤 {mdd_t*100:.1f}%")
ax.annotate(txt, xy=(0.02, 0.04), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "overnight_strategy.png"), dpi=130); plt.close()

# =====================================================================
# 5) 图4：按星期几拆开
# =====================================================================
labels = ["周一", "周二", "周三", "周四", "周五"]
mo = [r_o[weekday == d].mean() for d in range(5)]
mi = [r_i[weekday == d].mean() for d in range(5)]
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(5); w = 0.38
ax.bar(x - w/2, np.array(mo)*100, w, label="隔夜均值", color=C["overnight"])
ax.bar(x + w/2, np.array(mi)*100, w, label="盘中均值", color=C["intraday"])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("平均日收益 (%)")
ax.set_title("按星期几：周一隔夜含周末效应显著偏高，盘中全周微负")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "overnight_weekday.png"), dpi=130); plt.close()

print("\nDONE ->", D)
for f in sorted(os.listdir(D)):
    print("  ", f, os.path.getsize(os.path.join(D, f)), "bytes")
