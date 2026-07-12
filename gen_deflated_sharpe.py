#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「夏普比率的统计量误差与 Deflated Sharpe Ratio：别被好运气骗了」生成真实配图与真实统计数字。

核心概念(López de Prado, "Deflated Sharpe Ratio" / "Probabilistic Sharpe Ratio"):
  - 夏普比率估计量有抽样误差: SE(SR_hat) ≈ sqrt( (1 + 0.5*SR_hat^2) / T ), T=样本数(越短越不准)
  - 概率夏普比率 PSR(SR*)=Φ( ((SR_hat-SR*)√T) / sqrt(1 - γ3·SR_hat + (γ4-1)/4·SR_hat^2) )
      γ3=偏度, γ4=峰度(非超额); 分子标准化统计量, 分母修正偏度/厚尾
  - 多选偏差: 你从 N 个策略里挑了最好的, 它的高 SR 可能是被「挑出来」的
  - 去膨胀夏普 DSR = Φ( (PSR^{-1}(SR*) - E[V]) / sqrt(Var[V]) )
      V = max_i Z_i, Z_i 为各策略标准化统计量; 在 H0(真 SR=SR*) 下 Z_i~N(0,1) iid
      E[V], Var[V] 取 N 个独立标准正态最大值的矩(MC 估计)

所有图表与数字均由文中 Python 逻辑真实计算生成(自包含, 仅依赖 numpy/scipy)。

图片:
  dsr_sampling.png   —— 真 SR=0 时 SR_hat 的抽样分布(±SE 带), 直观看噪声能有多大
  dsr_psr_curve.png  —— 给定估计 SR_hat, PSR 随基准 SR* 下降; 偏度/厚尾修正把它压得更低
  dsr_bestofn.png    —— 「最优之一」偏差: 真 SR=0 时, N 个里挑出的最大 Z 随 N 膨胀
  dsr_psr_vs_dsr.png —— 被选中的策略: 朴素 PSR 逼近 1, 但 DSR 被钉在 ~50%(只是噪声里的最佳)
"""
import os
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "deflated-sharpe-ratio")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#DDDDDD",
     "acc": "#DD8452", "mem": "#4C72B0", "stat": "#8172B3", "gold": "#CCB974",
     "teal": "#17BECF"}

RES = {}

# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def skew_kurt(r):
    r = np.asarray(r, float)
    m = r - r.mean()
    m2 = (m * m).mean()
    m3 = (m ** 3).mean()
    m4 = (m ** 4).mean()
    return m3 / m2 ** 1.5, m4 / m2 ** 2  # 偏度, 峰度(非超额)

def psr(sr_hat, T, sr_star, skew, kurt):
    denom = np.sqrt(max(1.0 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr_star) * np.sqrt(T) / denom
    return stats.norm.cdf(z), z

def sr_of(r):
    r = np.asarray(r, float)
    return r.mean() / r.std(ddof=1)

# =====================================================================
# 图 1: 真 SR=0 时 SR_hat 的抽样分布
# =====================================================================
T = 60                       # 60 个月 ≈ 5 年月度收益
rng = np.random.default_rng(20260712)
M = 40000
rets = rng.standard_normal((M, T))        # 均值0 方差1 -> 真 SR=0
sr_hat = rets.mean(axis=1) / rets.std(axis=1, ddof=1)
mc_mean = sr_hat.mean()
mc_std = sr_hat.std(ddof=1)
se_closed = 1.0 / np.sqrt(T)             # 真 SR≈0 时 SE≈1/√T
RES["sr_sampling"] = dict(T=T, mc_mean=round(float(mc_mean),4),
                          mc_std=round(float(mc_std),4),
                          se_closed=round(float(se_closed),4),
                          p_sr_gt_0p4=round(float((sr_hat > 0.4).mean()),4),
                          p_sr_gt_0p5=round(float((sr_hat > 0.5).mean()),4))

fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.hist(sr_hat, bins=80, density=True, color=C["mem"], alpha=0.55, edgecolor="white", lw=0.3)
ax.axvline(0, color="#222", lw=1.4, label="真 SR = 0")
ax.axvspan(-2*se_closed, 2*se_closed, color=C["gold"], alpha=0.18,
           label="±2 SE (≈±%.2f)" % (2*se_closed))
ax.axvline(0.5, color=C["dn"], lw=2.0, ls="--", label="看起来不错的 SR = 0.5")
ax.set_xlabel("估计的夏普比率 SR_hat (T=%d 个月)" % T)
ax.set_ylabel("抽样密度")
ax.set_title("真 SR=0 的策略, 60 个月样本里 SR_hat 能晃到多大?\n噪声本身就能给你 ±0.4 的『好策略』")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "dsr_sampling.png"), dpi=130); plt.close()

# =====================================================================
# 图 2: PSR 曲线 + 偏度/厚尾修正
# =====================================================================
# 构造一个有偏度、厚尾、且 SR_hat≈0.5 的「策略收益样本」(典型股票策略: 偶发崩盘=负偏+肥尾)
# 90% 常态 + 10% 崩盘成分: 崩盘成分在左尾拉出长尾 -> 负偏、峰度高
rng2 = np.random.default_rng(70712)
n2 = 60
comp = rng2.random(n2) < 0.12                 # 12% 概率进入崩盘成分(左尾)
base = np.where(comp, rng2.normal(-4.5, 2.5, n2), rng2.normal(0.25, 1.0, n2))
x = (base - base.mean()) / base.std(ddof=1)   # 居中、单位方差(保留偏度/峰度形状)
r2 = x + 0.5                                  # 平移使 SR_hat = mean/std ≈ 0.5
sr_hat2 = sr_of(r2)
sk2, ku2 = skew_kurt(r2)
RES["skew_sample"] = dict(sr_hat=round(float(sr_hat2),4), skew=round(float(sk2),3),
                          kurt=round(float(ku2),3), n=n2)
sr_star_grid = np.linspace(-0.1, 0.8, 60)
psr_naive = stats.norm.cdf(sr_hat2 * np.sqrt(n2) / np.sqrt(1 + 0.5 * sr_hat2**2)) * np.ones_like(sr_star_grid)
psr_corr = np.array([psr(sr_hat2, n2, s, sk2, ku2)[0] for s in sr_star_grid])
psr_ignore = stats.norm.cdf(sr_hat2 * np.sqrt(n2)) * np.ones_like(sr_star_grid)  # 完全忽略高阶矩
RES["psr_compare"] = dict(sr_hat=round(float(sr_hat2),3),
                          skew=round(float(sk2),3), kurt=round(float(ku2),3),
                          psr0_naive=round(float(psr_naive[np.argmin(np.abs(sr_star_grid-0.0))]),4),
                          psr0_corr=round(float(psr_corr[np.argmin(np.abs(sr_star_grid-0.0))]),4),
                          psr0_ignore=round(float(psr_ignore[0]),4),
                          psr0p3_naive=round(float(psr_naive[np.argmin(np.abs(sr_star_grid-0.3))]),4),
                          psr0p3_corr=round(float(psr_corr[np.argmin(np.abs(sr_star_grid-0.3))]),4))
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.plot(sr_star_grid, psr_ignore, color=C["up"], lw=2.0, label="忽略偏度/峰度 (分母=1)")
ax.plot(sr_star_grid, psr_naive, color=C["mem"], lw=2.0, label="正态近似 (分母=√(1+0.5SR²))")
ax.plot(sr_star_grid, psr_corr, color=C["dn"], lw=2.2, label="偏度/峰度修正 PSR")
ax.axvline(0.0, color="#444", ls=":", lw=1.2)
ax.set_xlabel("基准夏普比率 SR* (要证明真 SR 超过它)")
ax.set_ylabel("PSR(SR*) = P(真 SR > SR*)")
ax.set_title("同样的 SR_hat=%.2f: 收益越『歪/肥尾』, 真实 PSR 越低" % sr_hat2)
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
ax.set_ylim(-0.02, 1.05)
plt.tight_layout(); plt.savefig(os.path.join(D, "dsr_psr_curve.png"), dpi=130); plt.close()

# =====================================================================
# 图 3: 最优之一偏差 —— 真 SR=0, N 个里挑最大 Z
# =====================================================================
Ns = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=int)
reps = 2500
mean_maxZ = []
p95_maxZ = []
for N in Ns:
    r = rng.standard_normal((reps, N, T))
    sr = r.mean(axis=2) / r.std(axis=2, ddof=1)
    Z = sr * np.sqrt(T)
    mx = Z.max(axis=1)
    mean_maxZ.append(mx.mean())
    p95_maxZ.append(np.percentile(mx, 95))
mean_maxZ = np.array(mean_maxZ); p95_maxZ = np.array(p95_maxZ)
RES["bestofn"] = dict(
    Ns=Ns.tolist(),
    mean_maxZ=[round(float(x),3) for x in mean_maxZ],
    p95_maxZ=[round(float(x),3) for x in p95_maxZ],
    z_at_100=round(float(mean_maxZ[Ns==100][0]),3),
)
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.fill_between(Ns, mean_maxZ - (p95_maxZ-mean_maxZ), p95_maxZ,
                color=C["acc"], alpha=0.18, label="90% 区间 (P5–P95)")
ax.plot(Ns, mean_maxZ, color=C["acc"], lw=2.2, marker="o", ms=4, label="平均最大 Z")
ax.plot(Ns, np.sqrt(2*np.log(Ns)), color=C["eq"], lw=1.6, ls="--",
        label="参考: √(2 ln N)")
ax.axhline(stats.norm.ppf(0.95), color=C["dn"], ls=":", lw=1.4, label="单策略 95% 显著阈值 Z=1.645")
ax.set_xscale("log")
ax.set_xlabel("测试的策略数量 N (全部真 SR=0)")
ax.set_ylabel("选出策略的标准化统计量 Z = SR_hat·√T")
ax.set_title("全噪声策略里挑最好的, Z 也会随 N 膨胀到 2~3 —— 看似『显著』实则是选法偏差")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "dsr_bestofn.png"), dpi=130); plt.close()

# =====================================================================
# 图 4: PSR vs DSR —— 同一个候选策略(SR_hat 固定), 你声称测过的策略数 N 越多, DSR 越塌
#   叙事: 你找到一个 5 年月度 SR_hat=0.4 的策略。若只测了这 1 个(N=1), 朴素 PSR≈99.9%;
#         若你其实从 1000 个里挑出来的, DSR 会被『选法偏差』打到接近 0。
#   公式: DSR(N) = Φ( (Z_sel - E[V_N]) / sqrt(Var[V_N]) ),
#         V_N = max of N iid N(0,1) (H0 下每个策略的标准化统计量), E/V 取 MC 矩
#         Z_sel = 候选策略自身标准化统计量 (固定不随 N 变)
# =====================================================================
rng3 = np.random.default_rng(20260712)
SR_HAT = 0.4                                  # 候选策略估计夏普
T4 = 60                                       # 60 个月
DEN = np.sqrt(1 + 0.5 * SR_HAT ** 2)          # 正态近似分母(偏度/峰度=0)
Z_SEL = SR_HAT * np.sqrt(T4) / DEN           # 候选策略标准化统计量(固定)
PSR_FLAT = stats.norm.cdf(Z_SEL)              # 朴素 PSR(与 N 无关)

def max_moments(N, reps=20000):
    z = rng3.standard_normal((reps, N)).max(axis=1)
    return z.mean(), z.var()

psr_sel, dsr_sel = [], []
for N in Ns:
    E_V, Var_V = max_moments(N, reps=20000 if N <= 200 else 8000)
    PSR = PSR_FLAT                                  # 朴素: 完全忽略选法
    DSR = stats.norm.cdf((Z_SEL - E_V) / np.sqrt(Var_V))
    psr_sel.append(PSR); dsr_sel.append(DSR)
psr_sel = np.array(psr_sel); dsr_sel = np.array(dsr_sel)
RES["psr_vs_dsr"] = dict(
    SR_hat=SR_HAT, T=T4, Z_sel=round(float(Z_SEL),3),
    PSR_single=round(float(PSR_FLAT),4),
    Ns=Ns.tolist(),
    psr=[round(float(x),4) for x in psr_sel],
    dsr=[round(float(x),4) for x in dsr_sel],
    psr_at_100=round(float(psr_sel[Ns==100][0]),4),
    dsr_at_100=round(float(dsr_sel[Ns==100][0]),4),
)
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.axhline(PSR_FLAT, color=C["up"], lw=2.4, ls="--", label="朴素 PSR(与 N 无关)≈%.3f" % PSR_FLAT)
ax.plot(Ns, dsr_sel, color=C["dn"], lw=2.4, marker="s", ms=4, label="去膨胀 DSR(扣除选法偏差)")
ax.axhline(0.5, color=C["gold"], ls=":", lw=1.6, label="50% 基准")
ax.set_xscale("log")
ax.set_xlabel("你声称测过的策略数量 N")
ax.set_ylabel("概率")
ax.set_title("同一个 SR_hat=%.1f (T=%d) 的候选策略:\n朴素 PSR 死守 %.3f, DSR 随『搜过多少』暴跌到 ~0" % (SR_HAT, T4, PSR_FLAT))
ax.set_ylim(-0.02, 1.08)
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "dsr_psr_vs_dsr.png"), dpi=130); plt.close()

# =====================================================================
print(json.dumps(RES, ensure_ascii=False, indent=2))
print("\n图片已保存到:", D)
