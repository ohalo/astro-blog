#!/usr/bin/env python3
"""
为文章「做空 Beta(BaB)：押注高 Beta 股票的脆弱性」(bet-against-beta)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 生成 N 只股票、T 个月的月度超额收益：r_i = α_i + β_i·r_m + ε_i。
  * 杠杆约束下的核心设定：α_i = C·(1 − β_i) —— 低 beta 股票被赋予正 alpha、
    高 beta 股票被赋予负 alpha，即「证券市场线(SML)太平坦」。这是 BaB 异象的来源：
    高 beta 股票没有按 CAPM 拿到足额补偿。
  * BaB 组合构造（Frazzini & Pedersen 2014）：做多低 beta 十分位、做空高 beta
    十分位，两条腿各自按 1/β 加杠杆，使组合 beta≈0（市场中立），但截获
    低 beta 正 alpha 与高 beta 负 alpha 的落差 → 正的超额收益。

注意：本模拟嵌入 BaB 异常以演示机制（与全库高阶文一致，如 SVJ 嵌入 VRP），
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
D = os.path.join(BASE, "bet-against-beta")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "low": "#55A868",
     "high": "#C44E52", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "q": ["#C44E52", "#D68A5C", "#CCB974", "#8FB980", "#55A868"]}

rng = np.random.default_rng(20260713)
N = 300
T = 240                                       # 20 年月度
Rf = 0.002                                    # 月度无风险
mu_m = 0.005                                  # 市场月度超额收益均值
sig_m = 0.040                                 # 市场月度波动
C_ANOM = 0.0030                               # BaB 异常强度：alpha = C·(1-beta)
ANOM_ANN = C_ANOM * 12

beta = rng.uniform(0.3, 2.2, N)               # 个股 beta，跨 0.3~2.2
idio = rng.uniform(0.015, 0.045, N)           # 特质月度波动(调低以让 alpha 信号主导)
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
theory_line = Rf * 12 + mkt_prem_ann * beta_grid
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

# BaB 组合：两条腿各按 1/beta 加杠杆 -> 组合 beta≈0
lev_long = 1.0 / beta_D1
lev_short = 1.0 / beta_D10
bab_ex = lev_long * ex_D1 - lev_short * ex_D10
bab_beta = lev_long * beta_D1 - lev_short * beta_D10      # ≈ 0
bab_ann = bab_ex.mean() * 12
bab_vol = bab_ex.std(ddof=1) * np.sqrt(12)
bab_shp = bab_ann / bab_vol

# BaB 的两条腿各自 alpha 贡献（月度）
long_mean_alpha = alpha[D1].mean()
short_mean_alpha = alpha[D10].mean()
leg_long_contrib = lev_long * long_mean_alpha
leg_short_contrib = -lev_short * short_mean_alpha

# 对照：等权全市场、纯高 beta 十分位
mkt_eq_ex = excess.mean(1)
mkt_eq_ann = mkt_eq_ex.mean() * 12
mkt_eq_vol = mkt_eq_ex.std(ddof=1) * np.sqrt(12)
mkt_eq_shp = mkt_eq_ann / mkt_eq_vol
hbeta_ann = ex_D10.mean() * 12
hbeta_vol = ex_D10.std(ddof=1) * np.sqrt(12)
hbeta_shp = hbeta_ann / hbeta_vol

# BaB 的 CAPM 回归（组合层面）
A2 = np.vstack([np.ones(T), mkt_ex]).T
bab_coef = np.linalg.lstsq(A2, bab_ex, rcond=None)[0]
bab_alpha_capm = bab_coef[0] * 12
bab_beta_capm = bab_coef[1]

def netvalue(r):
    return np.cumprod(1.0 + r)

eq_bab = netvalue(bab_ex)
eq_mkt = netvalue(mkt_eq_ex)
eq_hb = netvalue(ex_D10)

def maxdd(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / peak))

mdd_bab = maxdd(eq_bab)
mdd_mkt = maxdd(eq_mkt)
mdd_hb = maxdd(eq_hb)

print("===== BaB KEY NUMBERS =====")
print(f"N={N} T={T} mkt_prem_ann={mkt_prem_ann*100:.2f}% capm_int(actual)={capm_int*100:.2f}% capm_slope(actual)={capm_slope*100:.2f}%")
print(f"beta_D1={beta_D1:.3f} beta_D10={beta_D10:.3f} lev_long={lev_long:.3f} lev_short={lev_short:.3f}")
print(f"bab_beta(construct)={bab_beta:.4f} bab_ann={bab_ann*100:.2f}% bab_vol={bab_vol*100:.2f}% bab_shp={bab_shp:.2f} mdd_bab={mdd_bab*100:.1f}%")
print(f"leg_long_contrib(monthly)={leg_long_contrib*100:.4f}% leg_short_contrib(monthly)={leg_short_contrib*100:.4f}%")
print(f"mkt_eq_ann={mkt_eq_ann*100:.2f}% mkt_eq_shp={mkt_eq_shp:.2f} hbeta_ann={hbeta_ann*100:.2f}% hbeta_shp={hbeta_shp:.2f} mdd_hb={mdd_hb*100:.1f}%")
print(f"bab_alpha_capm={bab_alpha_capm*100:.2f}% bab_beta_capm={bab_beta_capm:.3f}")

# ============================================================
# 图 1：beta vs 年化超额收益 + 理论 SML(正斜率) vs 实际 SML(过平)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
sc = ax.scatter(beta, ann_ex * 100, c=ann_vol * 100, cmap="RdYlGn_r", s=18, alpha=0.8)
cb = fig.colorbar(sc, ax=ax); cb.set_label("年化波动 (%)")
ax.plot(beta_grid, theory_line * 100, color=C["thr"], lw=2.2, ls="--",
        label="CAPM 理论 SML：斜率=市场溢价 %.1f%% (正)" % (mkt_prem_ann * 100))
ax.plot(beta_grid, actual_line * 100, color=C["high"], lw=2.4,
        label="实际拟合 SML：斜率 %.1f%% (过平 → 高 beta 拿不到足额补偿)" % (capm_slope * 100))
ax.set_xlabel("个股 Beta"); ax.set_ylabel("年化超额收益 (%)")
ax.set_title("BaB 异象核心：理论说『高 beta 该高收益』(虚线)，数据里 SML 几乎平(红线)")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "bab_sml.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：BaB 两条腿的杠杆中性构造 + alpha 贡献
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
groups = ["低 beta 腿 (做多)", "高 beta 腿 (做空)"]
raw_beta = [beta_D1, -beta_D10]
lev_beta = [lev_long * beta_D1, -lev_short * beta_D10]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w/2, raw_beta, w, color=C["mkt"], label="原始平均 beta")
b2 = ax.bar(x + w/2, lev_beta, w, color=C["accent"], label="加杠杆后 |beta|=1")
for i, (rb, lb) in enumerate(zip(raw_beta, lev_beta)):
    ax.text(i - w/2, rb + (0.03 if rb >= 0 else -0.06), f"{rb:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, lb + (0.03 if lb >= 0 else -0.06), f"{lb:.2f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("组合 beta 贡献")
ax.set_title("BaB 杠杆中性构造：两条腿各按 1/β 加杠杆，组合 beta≈0 (净敞口=0)")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
ax.text(0.5, -0.42, f"杠杆因子 做多腿 1/β={lev_long:.2f}  ·  做空腿 1/β={lev_short:.2f}",
        transform=ax.transAxes, ha="center", fontsize=9, color=C["accent"])
plt.tight_layout()
plt.savefig(os.path.join(D, "bab_leverage.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：累计净值 BaB vs 等权市场 vs 高 beta 十分位
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.plot(eq_bab, color=C["eq"], lw=2.0, label=f"BaB 组合 (年化 {bab_ann*100:.1f}%, Sharpe {bab_shp:.2f})")
ax.plot(eq_mkt, color=C["mkt"], lw=1.6, ls="--", label=f"等权市场 (年化 {mkt_eq_ann*100:.1f}%, Sharpe {mkt_eq_shp:.2f})")
ax.plot(eq_hb, color=C["high"], lw=1.6, ls=":", label=f"高 beta 十分位 (年化 {hbeta_ann*100:.1f}%, Sharpe {hbeta_shp:.2f})")
ax.set_xlabel("月份"); ax.set_ylabel("净值 (起始=1)")
ax.set_title("BaB 用近零市场 beta 跑赢：高 beta 腿风险调整后明显最弱")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "bab_cumulative.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：BaB 组合 CAPM 回归 —— beta≈0 但正 alpha
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.scatter(mkt_ex * 100, bab_ex * 100, s=12, alpha=0.5, color=C["eq"])
mg = np.linspace(mkt_ex.min(), mkt_ex.max(), 50)
ax.plot(mg * 100, (bab_coef[0] + bab_coef[1] * mg) * 100, color=C["high"], lw=2.2,
        label=f"拟合：β={bab_beta_capm:.2f}, α={bab_alpha_capm*100:.2f}%/年")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("市场月度超额收益 (%)"); ax.set_ylabel("BaB 月度超额收益 (%)")
ax.set_title("BaB 组合的 CAPM 归因：beta≈0 (市场中立)，截获显著正 alpha")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "bab_capm.png"), dpi=130)
plt.close()

print("IMAGES WRITTEN:", os.listdir(D))
