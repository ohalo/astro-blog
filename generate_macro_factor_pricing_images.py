#!/usr/bin/env python3
"""
为文章「宏观因子定价模型：把收益率曲线拆成水平/斜率/曲率风险」(macro-factor-pricing)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由 Nelson-Siegel 三因子 + 观测噪声自洽合成，仅用于演示方法；真实落地见文末路径）：
  Nelson-Siegel 把期限 τ 上的收益率拆成 3 个宏观因子：
    y(τ) = β0 + β1 * (1-e^{-τ/λ})/(τ/λ) + β2 * [(1-e^{-τ/λ})/(τ/λ) - e^{-τ/λ}]
  - β0 = 水平(level)，长端水平，所有期限同载；
  - β1 = 斜率(slope)，短端高长端低；
  - β2 = 曲率(curvature)，中端凸、两端负。
  模拟 β0/β1/β2 为慢变过程 → 用最小二乘把收益率曲线「反拟合」出三因子，
  再对全样本收益率矩阵做 PCA，展示前 3 个主成分恰好对应 level/slope/curvature，
  解释 ~99% 的跨期限波动（即「宏观因子定价」）。
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
D = os.path.join(BASE, "macro-factor-pricing")
os.makedirs(D, exist_ok=True)

C = {"lvl": "#2F4B7C", "slp": "#C44E52", "cur": "#55A868", "grid": "#DDDDDD",
     "fit": "#8172B3", "obs": "#999999", "pc": "#DD8452"}

taus = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])   # 期限（年）
lam = 2.5                                              # 衰减参数（曲率峰值在中期）

def ns_loadings(tau, lam=lam):
    x = tau / lam
    L0 = np.ones_like(tau)                             # level
    L1 = (1 - np.exp(-x)) / x                          # slope
    L2 = L1 - np.exp(-x)                               # curvature
    return np.vstack([L0, L1, L2]).T                   # T × 3

# ============================================================
# 1) 模拟三因子时间序列（慢变、带趋势/均值回复）
# ============================================================
T = 252 * 12                                           # 约 12 年日度
rng = np.random.default_rng(7)
b0 = np.zeros(T); b1 = np.zeros(T); b2 = np.zeros(T)
b0[0], b1[0], b2[0] = 0.035, 0.012, 0.0
for t in range(1, T):
    b0[t] = 0.992 * b0[t - 1] + 0.008 * 0.035 + rng.normal(0, 0.0006)
    b1[t] = 0.95 * b1[t - 1] + 0.05 * 0.012 + rng.normal(0, 0.0009)
    b2[t] = 0.90 * b2[t - 1] + rng.normal(0, 0.0011)
L = ns_loadings(taus)
Y = L @ np.vstack([b0, b1, b2]) + rng.normal(0, 0.0004, size=(len(taus), T))  # T × N
Y = Y.T                                               # T × N

# ============================================================
# 2) 用最小二乘把每天的曲线反拟合成三因子
# ============================================================
bhat = np.zeros((T, 3))
for t in range(T):
    bhat[t] = np.linalg.lstsq(L, Y[t], rcond=None)[0]
Yfit = bhat @ L.T
ss_res = ((Y - Yfit) ** 2).sum(axis=1)
ss_tot = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
r2 = 1 - ss_res / ss_tot

# ============================================================
# 3) PCA：前 3 主成分解释多少跨期限波动
# ============================================================
Yc = Y - Y.mean(axis=0)
cov = np.cov(Yc, rowvar=False)
eigval, eigvec = np.linalg.eigh(cov)
order = np.argsort(eigval)[::-1]
eigval = eigval[order]; eigvec = eigvec[:, order]
var_exp = eigval / eigval.sum()
cum_var = np.cumsum(var_exp)

# ============================================================
# 主绘图
# ============================================================
t = np.arange(T)

# ---------- 图 1：NS 拟合一条代表曲线（拟合 vs 观测）----------
idx = T - 1
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.scatter(taus, Y[idx] * 100, color=C["obs"], s=40, zorder=3, label="观测收益率")
ax.plot(taus, Yfit[idx] * 100, color=C["fit"], lw=2.2, label="Nelson-Siegel 拟合")
ax.set_xlabel("期限 (年)"); ax.set_ylabel("收益率 (%)")
ax.set_title("Nelson-Siegel 三因子拟合收益率曲线 (R²=%.4f)" % r2[idx])
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "ns_fit.png"), dpi=130); plt.close()

# ---------- 图 2：三因子时间序列 ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, bhat[:, 0] * 100, color=C["lvl"], lw=1.4, label="β0 水平(level)")
ax.plot(t, bhat[:, 1] * 100, color=C["slp"], lw=1.4, label="β1 斜率(slope)")
ax.plot(t, bhat[:, 2] * 100, color=C["cur"], lw=1.4, label="β2 曲率(curvature)")
ax.set_xlabel("交易日"); ax.set_ylabel("因子值 (%)")
ax.set_title("宏观三因子随时间演化：水平/斜率/曲率各自缓慢移动")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "ns_factors.png"), dpi=130); plt.close()

# ---------- 图 3：各期限对三因子的载荷（loading 曲线）----------
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(taus, L[:, 0], color=C["lvl"], lw=2.0, marker="o", ms=3, label="水平载荷")
ax.plot(taus, L[:, 1], color=C["slp"], lw=2.0, marker="o", ms=3, label="斜率载荷")
ax.plot(taus, L[:, 2], color=C["cur"], lw=2.0, marker="o", ms=3, label="曲率载荷")
ax.axhline(0, color="#888", lw=0.6, ls="--")
ax.set_xlabel("期限 (年)"); ax.set_ylabel("因子载荷")
ax.set_title("三因子的期限结构载荷：斜率递减、曲率中端凸")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "ns_loadings.png"), dpi=130); plt.close()

# ---------- 图 4：PCA 方差解释（前 3 主成分 ≈ 宏观三因子）----------
fig, ax = plt.subplots(figsize=(11, 4.4))
bars = ax.bar(range(1, len(var_exp) + 1), var_exp[:10] * 100,
              color=[C["pc"] if i < 3 else C["obs"] for i in range(10)])
ax.plot(range(1, len(cum_var) + 1), cum_var[:10] * 100, color=C["lvl"],
        lw=1.8, marker="o", ms=4, label="累计解释度")
ax.axhline(cum_var[2] * 100, color=C["slp"], ls="--", lw=1.0,
           label="前3主成分 = %.1f%%" % (cum_var[2] * 100))
ax.set_xlabel("主成分序号"); ax.set_ylabel("解释方差 (%)")
ax.set_title("收益率曲线是三维的：前 3 主成分≈水平/斜率/曲率，解释 ~99%")
ax.legend(loc="lower right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "ns_pricing.png"), dpi=130); plt.close()

print("=== 宏观因子定价 (Nelson-Siegel) 关键数字 ===")
print("样本: 日度 %d 天 (约 %.1f 年)，%d 个期限" % (T, T / 252, len(taus)))
print("NS 拟合中位数 R² = %.4f (min=%.4f)" % (np.median(r2), r2.min()))
print("前3主成分累计解释方差 = %.2f%%" % (cum_var[2] * 100))
print("PC1=%.2f%% PC2=%.2f%% PC3=%.2f%%" %
      (var_exp[0] * 100, var_exp[1] * 100, var_exp[2] * 100))
print("β0(水平) 区间 [%.2f%%, %.2f%%]" % (bhat[:, 0].min() * 100, bhat[:, 0].max() * 100))
print("β1(斜率) 区间 [%.2f%%, %.2f%%]" % (bhat[:, 1].min() * 100, bhat[:, 1].max() * 100))
print("\n图片已保存到:", D)
