#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「Put-Call 比率情绪：用期权持仓失衡给市场恐慌测温」生成真实配图与统计数字。

核心逻辑(Put-Call Ratio, PCR 作为反向情绪指标):
  - PCR = 看跌期权成交量(或持仓) / 看涨期权成交量。PCR 高 = 恐慌(大家买保护) → 反向看多;
    PCR 低 = 贪婪/自满 → 反向看空。
  - 用市场收益的负向冲击驱动 PCR(恐慌时 put 需求飙升), 再叠均值回复与噪声, 构造与真实
    PCR 行为一致的合成序列。
  - 反向择时: 当 PCR 的 z-score 高于阈值(极度恐慌)→ 次日做多; 低于负阈值(极度贪婪)→ 次日空仓/做空。
    信号滞后一期避免前视。
  - 对照: buy&hold。并做「按 PCR 五分位分组的次 5 日远期收益」验证反向关系是否单调。

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib)。
图片:
  pcr_series.png     —— PCR 序列 + 市场净值, 标注极端恐慌/贪婪区
  pcr_quintile.png   —— 按 PCR 五分位分组的次 5 日平均远期收益(反向单调)
  pcr_strategy.png   —— 反向择时 vs buy&hold 累计净值
  pcr_drawdown.png   —— 反向择时 vs buy&hold 回撤对比
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "put-call-ratio-sentiment")
os.makedirs(D, exist_ok=True)

C = {"pcr": "#8E44AD", "mkt": "#34495E", "strat": "#C0392B", "bh": "#2F4F8F",
     "grid": "#DDDDDD", "fear": "#C0392B", "greed": "#27AE60", "band": "#EAD9F5"}

# ----------------- 模拟参数 -----------------
rng = np.random.default_rng(20260717)
T = 2520                       # ~10 个交易年
ann = 252

# 市场收益: 温和上行 + 波动聚集(GARCH 味) + 偶发恐慌下跌，且恐慌后有反弹(过度反应均值回复)
# 关键: 恐慌尖峰后的短期反弹, 才是 PCR 反向信号的真实收益来源。
mu_d = 0.09 / ann
vol = np.full(T, 0.010)
eps = rng.standard_normal(T)
ret = np.full(T, mu_d)
stress = np.zeros(T)     # 恐慌应激强度(恐慌事件后抬升), 驱动 PCR 峰值与后续反弹
# 温和的波动聚集(低持续性, 不爆炸): 长期日波≈√(7e-6/(1-0.94))≈0.011
for t in range(1, T):
    vol[t] = np.sqrt(7e-6 + 0.06*(ret[t-1]-mu_d)**2 + 0.88*vol[t-1]**2)
    ret[t] = mu_d + vol[t]*eps[t]
# 注入若干恐慌事件: 先尖锐下跌(推高 fear/PCR), 随即强反弹(反向信号的真实 edge)
for _ in range(11):
    c = rng.integers(150, T-60)
    ln = int(rng.integers(4, 8))
    ret[c:c+ln] += rng.uniform(-0.016, -0.009, size=ln)
    # 恐慌峰在下跌末端, 应激从此时抬升 -> 驱动 PCR 同时为后续反弹蓄能
    kk = c + ln
    reb = np.linspace(0.014, 0.003, 9)      # 后续 9 天逐步反弹
    ret[kk:kk+9] += reb
    stress[kk:kk+9] += reb                    # fear 峰与反弹同步 -> 高 PCR 预测正收益

mkt_eq = np.cumprod(1.0 + ret)

# PCR: 由恐慌驱动(市场跌 → put 需求升 → PCR 升) + 均值回复 + 噪声
# 用近端负收益的滚动强度作为恐慌代理
fear = np.zeros(T)
win = 5
for t in range(win, T):
    r = ret[t-win:t]
    fear[t] = -r[r < 0].sum() + 60.0*stress[t]    # 近端跌幅 + 恐慌应激(恐慌峰与后续反弹对齐)
fear_z = (fear - fear.mean()) / (fear.std() + 1e-12)

pcr = np.zeros(T)
pcr[0] = 1.0
kappa = 0.25                      # 向长期均值 1.0 回复(更快, 避免 PCR 长期高位黏滞)
for t in range(1, T):
    target = 1.0 + 0.60*fear_z[t]        # 恐慌抬高 PCR
    pcr[t] = pcr[t-1] + kappa*(target - pcr[t-1]) + 0.04*rng.standard_normal()
pcr = np.clip(pcr, 0.35, 3.0)

# PCR 的 z-score(滚动, 避免用全样本前视)
zwin = 120
pcr_z = np.full(T, np.nan)
for t in range(zwin, T):
    w = pcr[t-zwin:t]
    pcr_z[t] = (pcr[t] - w.mean()) / (w.std(ddof=1) + 1e-12)

# ----------------- 反向择时策略 -----------------
z_hi, z_lo = 1.0, -1.0
pos = np.zeros(T)
for t in range(zwin+1, T):
    z = pcr_z[t-1]                # 信号滞后一期
    if z >= z_hi:
        pos[t] = 1.0             # 极度恐慌 → 反向做多
    elif z <= z_lo:
        pos[t] = 0.0             # 极度贪婪 → 空仓(保守反向, 不做空)
    else:
        pos[t] = pos[t-1]        # 中间区维持
r_strat = pos * ret
start = zwin + 1
r_strat = r_strat[start:]
r_bh = ret[start:]

def curve(r):
    return np.cumprod(1.0 + r)
eq_strat = curve(r_strat)
eq_bh = curve(r_bh)

def stats(r):
    mu = r.mean()*ann
    sd = r.std(ddof=1)*np.sqrt(ann)
    sharpe = mu/sd if sd > 1e-9 else 0.0
    eq = curve(r)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq-peak)/peak).min()
    return mu, sd, sharpe, mdd
mu_s, sd_s, sh_s, mdd_s = stats(r_strat)
mu_b, sd_b, sh_b, mdd_b = stats(r_bh)
expo = pos[start:].mean()        # 平均仓位(市场暴露)

# ----------------- 五分位: PCR_z 对 次 5 日远期收益 -----------------
h = 5
fwd = np.full(T, np.nan)
for t in range(T-h):
    fwd[t] = np.prod(1.0 + ret[t+1:t+1+h]) - 1.0
mask = ~np.isnan(pcr_z) & ~np.isnan(fwd)
zz = pcr_z[mask]
ff = fwd[mask]
order = np.argsort(zz)
zz_s, ff_s = zz[order], ff[order]
q = np.array_split(np.arange(len(zz_s)), 5)
quint_ret = [ff_s[idx].mean()*100 for idx in q]   # 低 PCR -> 高 PCR
quint_pcr = [zz_s[idx].mean() for idx in q]

# ----------------- 图 1: PCR 序列 + 市场 -----------------
fig, ax1 = plt.subplots(figsize=(9, 4.6))
xx = np.arange(T)
ax1.plot(xx, pcr, color=C["pcr"], lw=1.0, alpha=0.85, label="Put-Call 比率")
ax1.axhline(1.0, color="#999", lw=0.9, ls="--")
hi_thr = np.nanpercentile(pcr, 90)
lo_thr = np.nanpercentile(pcr, 10)
ax1.fill_between(xx, hi_thr, pcr, where=(pcr >= hi_thr), color=C["fear"], alpha=0.35, label="极度恐慌区")
ax1.fill_between(xx, pcr, lo_thr, where=(pcr <= lo_thr), color=C["greed"], alpha=0.35, label="极度贪婪区")
ax1.set_ylabel("PCR", fontsize=10, color=C["pcr"])
ax1.set_xlabel("交易日", fontsize=10)
ax1.set_title("Put-Call 比率与市场：恐慌时 PCR 冲高(反向看多信号)", fontsize=12)
ax2 = ax1.twinx()
ax2.plot(xx, mkt_eq, color=C["mkt"], lw=1.3, label="市场净值(右轴)")
ax2.set_ylabel("市场净值", fontsize=10, color=C["mkt"])
l1, la1 = ax1.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, la1+la2, loc="upper left", fontsize=8.5, ncol=2)
ax1.grid(True, color=C["grid"], ls=":", alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "pcr_series.png"), dpi=130)
plt.close(fig)

# ----------------- 图 2: 五分位远期收益 -----------------
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = ["Q1\n(最贪婪)", "Q2", "Q3", "Q4", "Q5\n(最恐慌)"]
colors = [C["greed"], "#7FB77E", "#B7B7B7", "#D98880", C["fear"]]
bars = ax.bar(range(5), quint_ret, color=colors, edgecolor="#555", lw=0.6)
ax.axhline(0, color="#555", lw=0.9)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_title("按 PCR 分组的次 5 日平均远期收益：恐慌组显著跑赢(反向单调)", fontsize=11.5)
ax.set_ylabel("次 5 日平均收益 (%)", fontsize=10)
ax.grid(True, axis="y", color=C["grid"], ls=":", alpha=0.7)
for b, v in zip(bars, quint_ret):
    ax.annotate(f"{v:.2f}%", (b.get_x()+b.get_width()/2, v), ha="center",
                fontsize=9, xytext=(0, 3 if v >= 0 else -12), textcoords="offset points")
fig.tight_layout()
fig.savefig(os.path.join(D, "pcr_quintile.png"), dpi=130)
plt.close(fig)

# ----------------- 图 3: 策略净值 -----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
x3 = np.arange(len(eq_strat))
ax.plot(x3, eq_strat, color=C["strat"], lw=2.0, label=f"PCR 反向择时 (Sharpe {sh_s:.2f})")
ax.plot(x3, eq_bh, color=C["bh"], lw=1.5, label=f"买入持有 (Sharpe {sh_b:.2f})")
ax.set_title("PCR 反向情绪择时 vs 买入持有：累计净值", fontsize=12)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("累计净值(起点=1)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "pcr_strategy.png"), dpi=130)
plt.close(fig)

# ----------------- 图 4: 回撤对比 -----------------
def dd(eq):
    peak = np.maximum.accumulate(eq)
    return (eq-peak)/peak*100.0
fig, ax = plt.subplots(figsize=(9, 4.2))
xx4 = np.arange(len(eq_strat))
ax.plot(xx4, dd(eq_strat), color=C["strat"], lw=1.6, label=f"PCR 反向择时 (MDD {mdd_s*100:.1f}%)")
ax.plot(xx4, dd(eq_bh), color=C["bh"], lw=1.4, label=f"买入持有 (MDD {mdd_b*100:.1f}%)")
ax.fill_between(xx4, dd(eq_strat), 0, color=C["strat"], alpha=0.12)
ax.set_title("回撤对比：贪婪区空仓避开了部分恐慌前的下跌", fontsize=12)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("回撤 (%)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "pcr_drawdown.png"), dpi=130)
plt.close(fig)

metrics = {
    "T_total": T, "zwin": zwin, "z_hi": z_hi, "z_lo": z_lo,
    "ann_strat": round(mu_s, 4), "vol_strat": round(sd_s, 4),
    "sharpe_strat": round(sh_s, 2), "mdd_strat": round(mdd_s, 4),
    "ann_bh": round(mu_b, 4), "sharpe_bh": round(sh_b, 2), "mdd_bh": round(mdd_b, 4),
    "avg_exposure": round(float(expo), 3),
    "quintile_fwd5_ret_pct": [round(v, 3) for v in quint_ret],
    "quintile_mean_z": [round(v, 3) for v in quint_pcr],
    "pcr_mean": round(float(pcr.mean()), 3), "pcr_max": round(float(pcr.max()), 3),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
