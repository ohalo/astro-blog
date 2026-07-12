#!/usr/bin/env python3
"""
为文章「低波动异象：低风险股票为何长期跑赢高风险」(low-volatility-anomaly)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 生成 N 只股票、T 个月的月度收益：r_i = α_i + β_i·r_m + ε_i。
  * 关键设定：α_i = a0 − b·(σ_i^total − mean σ) —— 低总波动股票被赋予更高截距
    （这正是实证里的「低波动异常 / 特质波动异象」）。市场按 β 给正补偿，但低波动
    股票的特质 alpha 足以让其风险调整后收益反超高波动股票。
  * 截面排序：按实现波动分五组(Quintile)，展示 Q1(低波动) 收益与 Sharpe 都更高。
  * CAPM 检验：把个股均收益对 β 回归，理论斜率=市场预期溢价(正)，实际拟合斜率≈平/负
    —— 这就是异象的核心证据：风险与收益的关系破了。

注意：本模拟嵌入了低波动异常以演示机制（与全库高阶文一致，如 SVJ 嵌入 VRP），
真实数据中的异象来自杠杆约束 + 注意力/彩票偏好等行为因素，文末已说明。
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
D = os.path.join(BASE, "low-volatility-anomaly")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "low": "#55A868",
     "high": "#C44E52", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "q": ["#55A868", "#8FB980", "#CCB974", "#D68A5C", "#C44E52"]}

rng = np.random.default_rng(20260713)
N = 200
T = 240                                       # 20 年月度
Rf = 0.002                                    # 月度无风险
mu_m = 0.005                                  # 市场月度超额收益均值
sig_m = 0.040                                 # 市场月度波动
ANOM_B = 0.25                                 # 低波动异常强度
beta = rng.uniform(0.5, 1.5, N)              # 个股 beta
idio = rng.uniform(0.020, 0.075, N)          # 特质月度波动
true_vol = np.sqrt((beta * sig_m) ** 2 + idio ** 2)      # 真实总波动
mvol = true_vol.mean()
alpha = 0.0030 - ANOM_B * (true_vol - mvol)  # 月度；低波动 -> 高 alpha（异常来源）
mkt = rng.normal(mu_m, sig_m, T)
eps = rng.standard_normal((T, N)) * idio
ret = alpha[None, :] + beta[None, :] * mkt[:, None] + eps

# 实现统计量（年化）
ann_ret = ret.mean(0) * 12
ann_vol = ret.std(0, ddof=1) * np.sqrt(12)
sharpe = ann_ret / ann_vol

# 按实现波动分五组
order = np.argsort(ann_vol)
qs = np.array_split(order, 5)
q_ret = np.array([ann_ret[q].mean() for q in qs])
q_vol = np.array([ann_vol[q].mean() for q in qs])
q_shp = np.array([sharpe[q].mean() for q in qs])
# 每只股票所属五分位
edges = [ann_vol[order[int(N * k / 5)]] for k in range(1, 5)]
grp = np.clip(np.searchsorted(edges, ann_vol), 0, 4)

# CAPM 检验：均收益 ~ beta 回归
mkt_prem_ann = (mkt.mean() - Rf) * 12
A = np.vstack([np.ones(N), beta]).T
coef = np.linalg.lstsq(A, ann_ret, rcond=None)[0]
capm_slope = coef[1]
capm_int = coef[0]
beta_grid = np.linspace(beta.min(), beta.max(), 50)
capm_line = Rf * 12 + mkt_prem_ann * beta_grid
actual_line = capm_int + capm_slope * beta_grid

# 组合：低波动(Q1) / 高波动(Q5) / 等权全市场
low_idx = qs[0]; high_idx = qs[4]
low_ret = ret[:, low_idx].mean(1)
high_ret = ret[:, high_idx].mean(1)
mkt_ret = ret.mean(1)


def netvalue(r):
    return np.cumprod(1.0 + r)


def maxdd(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / peak))


eq_low = netvalue(low_ret); eq_high = netvalue(high_ret); eq_mkt = netvalue(mkt_ret)
mdd_low = maxdd(eq_low); mdd_high = maxdd(eq_high); mdd_mkt = maxdd(eq_mkt)
corr_low_mkt = np.corrcoef(low_ret, mkt_ret)[0, 1]


# ============================================================
# 图 1：beta vs 年化收益 散点 + CAPM 理论线(正) vs 实际拟合线(平/负)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
sc = ax.scatter(beta, ann_ret * 100, c=ann_vol * 100, cmap="RdYlGn_r", s=22, alpha=0.8)
cb = fig.colorbar(sc, ax=ax); cb.set_label("年化波动 (%)")
ax.plot(beta_grid, capm_line * 100, color=C["thr"], lw=2.2, ls="--",
        label="CAPM 理论线：斜率=市场预期溢价 %.1f%% (正)" % (mkt_prem_ann * 100))
ax.plot(beta_grid, actual_line * 100, color=C["high"], lw=2.2,
        label="实际拟合线：斜率 %.1f%% (平/负 → 异象)" % (capm_slope * 100))
ax.set_xlabel("个股 Beta"); ax.set_ylabel("年化收益 (%)")
ax.set_title("低波动异象核心：理论说『高风险高收益』(虚线正斜率)，数据却近乎平坦(红线)")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "lv_beta_return.png"), dpi=130)
plt.close()


# ============================================================
# 图 2：波动率五分位 收益 & Sharpe 柱状
# ============================================================
labels = ["Q1 低波动", "Q2", "Q3", "Q4", "Q5 高波动"]
x = np.arange(5); width = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
ax1.bar(x, q_ret * 100, width, color=C["q"], alpha=0.9)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("年化收益 (%)")
ax1.set_title("按波动分组的平均年化收益：低波动组反而最高")
for i, v in enumerate(q_ret * 100):
    ax1.text(i, v + (0.15 if v >= 0 else -0.4), "%.1f" % v, ha="center", fontsize=8)
ax1.grid(True, color=C["grid"], lw=0.6, axis="y")
ax2.bar(x, q_shp, width, color=C["q"], alpha=0.9)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("年化 Sharpe")
ax2.set_title("Sharpe 单调随波动下降：低风险高夏普，高风险低夏普")
for i, v in enumerate(q_shp):
    ax2.text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=8)
ax2.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "lv_quintile_bars.png"), dpi=130)
plt.close()


# ============================================================
# 图 3：低波动 vs 高波动 vs 市场 净值 + 最大回撤
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.0))
xr = np.arange(T)
ax.plot(xr, eq_low, color=C["low"], lw=1.4, label="低波动组合(Q1) 回撤 %.0f%%" % (mdd_low * 100))
ax.plot(xr, eq_high, color=C["high"], lw=1.2, label="高波动组合(Q5) 回撤 %.0f%%" % (mdd_high * 100))
ax.plot(xr, eq_mkt, color=C["mkt"], lw=1.0, ls="--", label="等权全市场 回撤 %.0f%%" % (mdd_mkt * 100))
ax.axhline(1, color="black", lw=0.5)
ax.set_xlabel("月份"); ax.set_ylabel("净值 (期初=1)")
ax.set_title("低波动组合：终点更高、路径更平（回撤更浅）")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "lv_netvalue.png"), dpi=130)
plt.close()


# ============================================================
# 图 4：实现波动 vs Sharpe 散点（标志性低波动云）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.0))
sc = ax.scatter(ann_vol * 100, sharpe, c=[C["q"][g] for g in grp], s=22, alpha=0.85)
ax.set_xlabel("年化波动 (%)"); ax.set_ylabel("年化 Sharpe")
ax.set_title("标志性低波动云：波动越往右，Sharpe 越往下塌——高风险并未换来高收益")
ax.plot(q_vol * 100, q_shp, color="black", lw=1.2, marker="o", zorder=5, label="五分位均值")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "lv_vol_sharpe_scatter.png"), dpi=130)
plt.close()


# ============================================================
# 关键数字
# ============================================================
print("=== 低波动异象 Low-Volatility Anomaly 关键数字 ===")
print("样本 %d 只股票 × %d 月(20年); 市场月均超额 %.2f%%, 波动 %.1f%%"
      % (N, T, mu_m * 100, sig_m * 100))
print("CAPM 理论斜率(市场预期溢价) = %.1f%%/年 ; 实际拟合斜率 = %.1f%%/年 (平/负=异象)"
      % (mkt_prem_ann * 100, capm_slope * 100))
print("按实现波动五分位（Q1低 → Q5高）：")
for i in range(5):
    print("  %-9s 年化收益 %5.1f%%  年化波动 %5.1f%%  Sharpe %.2f"
          % (labels[i], q_ret[i] * 100, q_vol[i] * 100, q_shp[i]))
print("低波动组合(Q1): 年化 %.1f%% 波动 %.1f%% Sharpe %.2f 最大回撤 %.0f%%"
      % (ann_ret[low_idx].mean() * 100, ann_vol[low_idx].mean() * 100,
         sharpe[low_idx].mean(), mdd_low * 100))
print("高波动组合(Q5): 年化 %.1f%% 波动 %.1f%% Sharpe %.2f 最大回撤 %.0f%%"
      % (ann_ret[high_idx].mean() * 100, ann_vol[high_idx].mean() * 100,
         sharpe[high_idx].mean(), mdd_high * 100))
print("等权全市场:     年化 %.1f%% 波动 %.1f%% Sharpe %.2f 最大回撤 %.0f%%"
      % (mkt_ret.mean() * 12 * 100, mkt_ret.std(ddof=1) * np.sqrt(12) * 100,
         (mkt_ret.mean() * 12) / (mkt_ret.std(ddof=1) * np.sqrt(12)), mdd_mkt * 100))
print("低波动组合与全市场相关 = %.2f" % corr_low_mkt)
print("图片已保存到:", D)
