#!/usr/bin/env python3
"""
为文章「杠杆厌恶因子(Frazzini-Pedersen)：被杠杆约束压低的低 Beta 溢价」(leverage-aversion-factor)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 生成 N 只股票、T 个月的月度超额收益：r_i = α_i + β_i·r_m + ε_i。
  * 杠杆约束的核心设定：α_i = C·(1 − β_i) —— 低 beta 股票被赋予正 alpha、
    高 beta 股票被赋予负 alpha，即「证券市场线(SML)太平坦」。这是杠杆厌恶
    异象的来源：受杠杆约束的投资者无法满足「想要高市场敞口」的需求，于是
    蜂拥买入高 beta 股票、把其价格推高（预期收益被压低），同时抛售低 beta
    股票、把其价格压低（预期收益被抬高）。
  * 杠杆厌恶因子（BAB）：做多低 beta 十分位、做空高 beta 十分位，两条腿各自
    按 1/β 加杠杆，使组合 beta≈0（市场中立），截获低 beta 正 alpha 与高 beta
    负 alpha 的落差 → 正的超额收益。

注意：本模拟嵌入杠杆厌恶异常以演示机制（与全库高阶文一致，如 SVJ 嵌入 VRP），
真实数据中的异象来自杠杆约束 + 投资者偏好，文末已说明。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "leverage-aversion-factor")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "low": "#55A868",
     "high": "#C44E52", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "q": ["#C44E52", "#D68A5C", "#CCB974", "#8FB980", "#55A868"]}

rng = np.random.default_rng(20260714)
N = 300
T = 240                                       # 20 年月度
Rf = 0.002                                    # 月度无风险
mu_m = 0.005                                  # 市场月度超额收益均值
sig_m = 0.040                                 # 市场月度波动
C_ANOM = 0.0060                               # 杠杆厌恶异常强度：alpha = C·(1-beta)
ANOM_ANN = C_ANOM * 12

beta = rng.uniform(0.3, 2.2, N)               # 个股 beta，跨 0.3~2.2
idio = rng.uniform(0.015, 0.045, N)           # 特质月度波动
mkt_ex = rng.normal(mu_m, sig_m, T)           # 市场月度超额收益
eps = rng.standard_normal((T, N)) * idio
alpha = C_ANOM * (1.0 - beta)                 # 低 beta -> 正, 高 beta -> 负
ret = alpha[None, :] + beta[None, :] * mkt_ex[:, None] + eps
excess = ret - Rf

# 年化统计
ann_ex = excess.mean(0) * 12
ann_vol = excess.std(0, ddof=1) * np.sqrt(12)

# CAPM SML 检验：ann_ex ~ beta
mkt_prem_ann = mkt_ex.mean() * 12               # 理论 SML 斜率 = 已实现市场溢价
A = np.vstack([np.ones(N), beta]).T
coef = np.linalg.lstsq(A, ann_ex, rcond=None)[0]
capm_int = coef[0]
capm_slope = coef[1]
beta_grid = np.linspace(beta.min(), beta.max(), 50)
theory_line = mkt_prem_ann * beta_grid          # 理论 SML 过原点(截距用已实现溢价)
actual_line = capm_int + capm_slope * beta_grid

# 十分位
order = np.argsort(beta)
deciles = np.array_split(order, 10)
D1 = deciles[0]                               # 低 beta 十分位
D10 = deciles[9]                              # 高 beta 十分位
beta_D1 = beta[D1].mean()
beta_D10 = beta[D10].mean()
ex_D1 = excess[:, D1].mean(1)                 # 低 beta 腿月度超额(等权)
ex_D10 = excess[:, D10].mean(1)               # 高 beta 腿月度超额(等权)

# 杠杆厌恶因子(BAB)：两条腿各按 1/beta 加杠杆 -> 组合 beta≈0（市场中立）
lev_long = 1.0 / beta_D1
lev_short = 1.0 / beta_D10
bab_ex = lev_long * ex_D1 - lev_short * ex_D10
bab_beta = lev_long * beta_D1 - lev_short * beta_D10      # ≈ 0（两条腿都放大到市场 beta=1，多空抵消）
bab_ann = bab_ex.mean() * 12
bab_vol = bab_ex.std(ddof=1) * np.sqrt(12)
bab_shp = bab_ann / bab_vol

# 各十分位年化收益（用于柱状图）
dec_ann = np.array([excess[:, d].mean(1).mean() * 12 for d in deciles])
dec_beta = np.array([beta[d].mean() for d in deciles])

# 市场组合（等权全市场）
mkt_ann = excess.mean(1).mean() * 12
mkt_vol = excess.mean(1).std(ddof=1) * np.sqrt(12)
mkt_shp = mkt_ann / mkt_vol

# 高 beta 组合（做多 D10 不加杠杆，作为对照：它拿到高 beta 却没拿到足额补偿）
hb_ann = ex_D10.mean() * 12
hb_vol = ex_D10.std(ddof=1) * np.sqrt(12)
hb_shp = hb_ann / hb_vol

print(f"[LA] 理论 SML 斜率(市场溢价年化) = {mkt_prem_ann:.4f}")
print(f"[LA] 实际 SML 斜率 = {capm_slope:.4f}  (应明显 < 理论)")
print(f"[LA] 实际 SML 截距 = {capm_int:.4f}  (应为正, 低 beta 被抬高)")
print(f"[LA] D1 beta={beta_D1:.3f} D10 beta={beta_D10:.3f} 杠杆 long={lev_long:.2f} short={lev_short:.2f}")
print(f"[LA] BAB 年化={bab_ann:.4f} 波动={bab_vol:.4f} Sharpe={bab_shp:.3f} beta={bab_beta:.3f} (市场中立)")
print(f"[LA] 全市场 年化={mkt_ann:.4f} 波动={mkt_vol:.4f} Sharpe={mkt_shp:.3f}")
print(f"[LA] 高beta腿 年化={hb_ann:.4f} 波动={hb_vol:.4f} Sharpe={hb_shp:.3f}")
print(f"[LA] 十分位年化收益(低->高beta): " + " ".join(f"{x:.3f}" for x in dec_ann))
print(f"[LA] 十分位CAPM预测收益(低->高beta): " + " ".join(f"{x:.3f}" for x in dec_beta * mkt_prem_ann))
print(f"[LA] 十分位alpha(实际-预测, 低->高beta): " + " ".join(f"{x:.3f}" for x in (dec_ann - dec_beta * mkt_prem_ann)))

# ---------------- 图 1: 证券市场线 SML 扁平化 ----------------
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(beta, ann_ex, s=22, c=C["mkt"], alpha=0.55, edgecolors="none", label="个股 (年化超额收益 vs β)")
ax.plot(beta_grid, theory_line, color=C["high"], lw=2.4, label=f"理论 SML (斜率={mkt_prem_ann:.3f})")
ax.plot(beta_grid, actual_line, color=C["low"], lw=2.4, ls="--", label=f"实际 SML (斜率={capm_slope:.3f})")
ax.set_xlabel("Beta (β)", fontsize=13)
ax.set_ylabel("年化超额收益", fontsize=13)
ax.set_title("杠杆约束下的证券市场线(SML)被压平\n受约束投资者推高高 β 价格 → 高 β 拿不到足额补偿", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax.grid(True, color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "sml_flattening.png"), dpi=130)
plt.close(fig)

# ---------------- 图 2: 各 beta 十分位「实际收益 vs CAPM 预测收益」(alpha 可视化) ----------------
dec_pred = dec_beta * mkt_prem_ann               # CAPM 预测：每块 beta * 市场溢价
x = np.arange(10)
fig, ax = plt.subplots(figsize=(10, 6))
w = 0.4
b1 = ax.bar(x - w/2, dec_ann, w, color=C["low"], label="实际年化收益")
b2 = ax.bar(x + w/2, dec_pred, w, color=C["high"], alpha=0.8, label="CAPM 预测收益 (β·市场溢价)")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Beta 十分位 (1=最低 β, 10=最高 β)", fontsize=13)
ax.set_ylabel("年化收益", fontsize=13)
ax.set_title("各 Beta 十分位：高 β 实际收益远低于 CAPM 预测\n柱间缺口=被压低的 alpha，低 β 反被抬高", fontsize=13.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(range(1, 11))
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "beta_decile_alpha.png"), dpi=130)
plt.close(fig)

# ---------------- 图 3: 累积净值 BAB vs 市场 vs 高β腿 ----------------
bab_cum = np.cumprod(1 + bab_ex)
mkt_cum = np.cumprod(1 + excess.mean(1))
hb_cum = np.cumprod(1 + ex_D10)
t = np.arange(T)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, bab_cum, color=C["low"], lw=2.4, label=f"杠杆厌恶因子 BAB (Sharpe={bab_shp:.2f})")
ax.plot(t, mkt_cum, color=C["mkt"], lw=1.8, ls="-", label=f"全市场等权 (Sharpe={mkt_shp:.2f})")
ax.plot(t, hb_cum, color=C["high"], lw=1.8, ls="--", label=f"高 β 腿(做多D10, Sharpe={hb_shp:.2f})")
ax.set_xlabel("月份", fontsize=13)
ax.set_ylabel("累积净值 (初始=1)", fontsize=13)
ax.set_title("杠杆厌恶因子长期跑赢市场与高 β 腿\n做多低 β、做空高 β，各按 1/β 加杠杆", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax.grid(True, color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "bab_cumulative.png"), dpi=130)
plt.close(fig)

print("[LA] images written to", D)
