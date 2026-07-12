#!/usr/bin/env python3
"""
为文章「质量因子(QMJ)：好公司如何变成可交易 Alpha」(quality-minus-junk)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 为每只股票造 4 个质量子维度（z 分数）：盈利能力、成长性、安全性、派息。
    质量得分 q = 四维度均值。质量得分与未来 alpha 正相关：α_i = C·q_i。
  * 这就是 Asness-Frazzini-Pedersen(2019) 的 Quality-Minus-Junk 思想：
    好公司(高 q)被系统性低估、垃圾公司(低 q)被高估，做多高质量、做空垃圾
    得到市场中立的正 alpha。
  * QMJ 因子：做多质量最高十分位、做空质量最低十分位(等权，组合 beta≈0)。

注意：本模拟嵌入「质量 → alpha」关系以演示机制（与全库高阶文一致），
真实世界质量溢价来自行为偏差与套利约束，文末已说明。
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
D = os.path.join(BASE, "quality-minus-junk")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "low": "#C44E52",
     "high": "#55A868", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "q": ["#C44E52", "#D68A5C", "#CCB974", "#8FB980", "#55A868"]}

rng = np.random.default_rng(20260713)
N = 300
T = 240                                       # 20 年月度
Rf = 0.002
mu_m = 0.005
sig_m = 0.040
C_Q = 0.0020                                 # 质量 → alpha 强度(月度)

# 四个质量子维度(z 分数)，彼此弱相关
profit = rng.normal(0, 1, N)
growth = 0.3 * profit + rng.normal(0, np.sqrt(1 - 0.09), N)
safety = 0.3 * profit + rng.normal(0, np.sqrt(1 - 0.09), N)
payout = rng.normal(0, 1, N)
q = (profit + growth + safety + payout) / 4.0  # 质量得分

# beta 与质量弱负相关(好公司更"安全"=低 beta)：仅注入微弱相关以贴近现实
beta = rng.uniform(0.4, 1.8, N) - 0.15 * q
beta = np.clip(beta, 0.4, 1.9)
idio = rng.uniform(0.015, 0.050, N)

mkt_ex = rng.normal(mu_m, sig_m, T)
eps = rng.standard_normal((T, N)) * idio
alpha = C_Q * q                              # 高质量 -> 正 alpha
ret = alpha[None, :] + beta[None, :] * mkt_ex[:, None] + eps
excess = ret - Rf

# 质量五分位(按 q)
order = np.argsort(q)
qs = np.array_split(order, 5)
q_ret = np.array([excess[:, idx].mean(0).mean() * 12 for idx in qs])
q_vol = np.array([excess[:, idx].mean(0).std(ddof=1) * np.sqrt(12) for idx in qs])
q_shp = q_ret / q_vol

# 质量子维度在 Q1(垃圾) vs Q5(高质量) 的平均 z
prof_q = np.array([profit[idx].mean() for idx in qs])
grow_q = np.array([growth[idx].mean() for idx in qs])
safe_q = np.array([safety[idx].mean() for idx in qs])
pay_q = np.array([payout[idx].mean() for idx in qs])

# QMJ 因子：做多 Q5(高质量) 做空 Q1(垃圾)，等权
Q5 = qs[4]; Q1 = qs[0]
ex_Q5 = excess[:, Q5].mean(1)
ex_Q1 = excess[:, Q1].mean(1)
qmj = ex_Q5 - ex_Q1
qmj_ann = qmj.mean() * 12
qmj_vol = qmj.std(ddof=1) * np.sqrt(12)
qmj_shp = qmj_ann / qmj_vol

# QMJ 的 CAPM 回归
A2 = np.vstack([np.ones(T), mkt_ex]).T
qmj_coef = np.linalg.lstsq(A2, qmj, rcond=None)[0]
qmj_alpha_capm = qmj_coef[0] * 12
qmj_beta_capm = qmj_coef[1]

# 对照：等权市场
mkt_eq_ex = excess.mean(1)
mkt_eq_ann = mkt_eq_ex.mean() * 12
mkt_eq_vol = mkt_eq_ex.std(ddof=1) * np.sqrt(12)
mkt_eq_shp = mkt_eq_ann / mkt_eq_vol

def netvalue(r):
    return np.cumprod(1.0 + r)

def maxdd(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / peak))

eq_qmj = netvalue(qmj)
eq_mkt = netvalue(mkt_eq_ex)
mdd_qmj = maxdd(eq_qmj)
mdd_mkt = maxdd(eq_mkt)

# 质量得分 vs alpha 散点(验证嵌入关系)
ann_ex = excess.mean(0) * 12
Aq = np.vstack([np.ones(N), q]).T
qcoef = np.linalg.lstsq(Aq, ann_ex, rcond=None)[0]

print("===== QMJ KEY NUMBERS =====")
print(f"N={N} T={T} q_ret(Q1->Q5)={[round(x*100,2) for x in q_ret]}%")
print(f"q_shp(Q1->Q5)={[round(x,2) for x in q_shp]}")
print(f"qmj_ann={qmj_ann*100:.2f}% qmj_vol={qmj_vol*100:.2f}% qmj_shp={qmj_shp:.2f} mdd_qmj={mdd_qmj*100:.1f}%")
print(f"qmj_alpha_capm={qmj_alpha_capm*100:.2f}% qmj_beta_capm={qmj_beta_capm:.3f}")
print(f"mkt_eq_ann={mkt_eq_ann*100:.2f}% mkt_eq_shp={mkt_eq_shp:.2f} mdd_mkt={mdd_mkt*100:.1f}%")
print(f"q->alpha slope={qcoef[1]*100:.3f}%/年 per z  intercept={qcoef[0]*100:.2f}%")
print(f"Q1 vs Q5 sub-scores profit={prof_q[0]:.2f}/{prof_q[4]:.2f} growth={grow_q[0]:.2f}/{grow_q[4]:.2f} safety={safe_q[0]:.2f}/{safe_q[4]:.2f} payout={pay_q[0]:.2f}/{pay_q[4]:.2f}")

# ============================================================
# 图 1：质量五分位 年化收益 & Sharpe
# ============================================================
labels = ["Q1 垃圾", "Q2", "Q3", "Q4", "Q5 高质量"]
x = np.arange(5); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5.4))
b1 = ax.bar(x - w/2, q_ret * 100, w, color=C["q"], label="年化超额收益 (%)")
ax2 = ax.twinx()
b2 = ax2.bar(x + w/2, q_shp, w, color=C["accent"], alpha=0.85, label="Sharpe")
for i in range(5):
    ax.text(i - w/2, q_ret[i] * 100 + 0.2, f"{q_ret[i]*100:.1f}", ha="center", fontsize=8)
    ax2.text(i + w/2, q_shp[i] + 0.02, f"{q_shp[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("年化超额收益 (%)"); ax2.set_ylabel("Sharpe")
ax.set_title("质量五分位：从垃圾(Q1)到高质量(Q5)，收益与 Sharpe 单调上升")
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(D, "qmj_quintile.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：质量子维度 Q1 vs Q5
# ============================================================
subs = ["盈利能力", "成长性", "安全性", "派息"]
q1v = [prof_q[0], grow_q[0], safe_q[0], pay_q[0]]
q5v = [prof_q[4], grow_q[4], safe_q[4], pay_q[4]]
x = np.arange(4); w = 0.36
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.bar(x - w/2, q1v, w, color=C["low"], label="Q1 垃圾 (低质量)")
ax.bar(x + w/2, q5v, w, color=C["high"], label="Q5 高质量")
for i in range(4):
    ax.text(i - w/2, q1v[i] + (0.05 if q1v[i] >= 0 else -0.12), f"{q1v[i]:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, q5v[i] + (0.05 if q5v[i] >= 0 else -0.12), f"{q5v[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(subs)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("平均 z 分数")
ax.set_title("高质量公司在四个维度上全面占优：这就是『质量』的构成")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "qmj_components.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：累计净值 QMJ vs 等权市场
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.plot(eq_qmj, color=C["eq"], lw=2.0, label=f"QMJ 因子 (年化 {qmj_ann*100:.1f}%, Sharpe {qmj_shp:.2f})")
ax.plot(eq_mkt, color=C["mkt"], lw=1.6, ls="--", label=f"等权市场 (年化 {mkt_eq_ann*100:.1f}%, Sharpe {mkt_eq_shp:.2f})")
ax.set_xlabel("月份"); ax.set_ylabel("净值 (起始=1)")
ax.set_title("QMJ 市场中立却稳定向上：好公司溢价不靠承担市场风险")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "qmj_cumulative.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：质量得分 vs 年化 alpha 散点 + QMJ 组合 CAPM 归因
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
# 左：质量得分 vs alpha
axes[0].scatter(q, ann_ex * 100, s=14, alpha=0.6, color=C["eq"])
gg = np.linspace(q.min(), q.max(), 50)
axes[0].plot(gg, (qcoef[0] + qcoef[1] * gg) * 100, color=C["high"], lw=2.2,
             label=f"斜率 {qcoef[1]*100:.2f}%/年 per z")
axes[0].set_xlabel("质量得分 q (z)"); axes[0].set_ylabel("年化超额收益 (%)")
axes[0].set_title("质量 → alpha：正相关嵌入")
axes[0].legend(fontsize=8); axes[0].grid(True, color=C["grid"], lw=0.6)
# 右：QMJ 组合 CAPM
axes[1].scatter(mkt_ex * 100, qmj * 100, s=12, alpha=0.5, color=C["accent"])
mg = np.linspace(mkt_ex.min(), mkt_ex.max(), 50)
axes[1].plot(mg * 100, (qmj_coef[0] + qmj_coef[1] * mg) * 100, color=C["high"], lw=2.2,
             label=f"β={qmj_beta_capm:.2f}, α={qmj_alpha_capm*100:.2f}%/年")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("市场月度超额收益 (%)"); axes[1].set_ylabel("QMJ 月度超额收益 (%)")
axes[1].set_title("QMJ 的 CAPM 归因：低 beta + 正 alpha")
axes[1].legend(fontsize=8); axes[1].grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "qmj_capm.png"), dpi=130)
plt.close()

print("IMAGES WRITTEN:", os.listdir(D))
