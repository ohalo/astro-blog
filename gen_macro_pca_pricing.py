#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宏观PCA定价 配图生成 (4 张真实图表)

机制(自洽合成, 仅用于演示方法):
  * 12 个资产(行业/宽基指数)的日收益, 由 3 个潜在宏观因子线性驱动 + 特质噪声:
      r_k = β_k1·F1(市场) + β_k2·F2(利率/价值成长) + β_k3·F3(商品/通胀) + ε_k
    - F1 全正载(普涨普跌), F2 价值-成长劈叉, F3 商品链偏正、利率链偏负
  * 在训练期估计样本协方差, 做 PCA:
      - 图1 碎石图: 前 3 主成分解释 ~95% 截面波动 = 「宏观因子定价」的实证
      - 图2 前 3 主成分(特征组合)在各资产上的载荷 = 因子结构
  * 关键应用: 低维协方差 = 对样本协方差做 PCA 收缩
      - 用 top-k 主成分重构协方差 Σ̂_k = Σ_{j≤k} λ_j v_j v_jᵀ + δ·I
      - 用它求最小方差组合, 在测试期看真实波动率: 全样本协方差过拟合(噪声),
        PCA-3 收缩后测试波动更低更稳
  * 图3: 三种协方差估计下最小方差组合权重
  * 图4: 测试期累计波动率对比(全样本 vs PCA-1 vs PCA-3 vs 等权)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "macro-pca-pricing"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"pc1": "#4C72B0", "pc2": "#C44E52", "pc3": "#55A868", "grid": "#DDDDDD",
     "full": "#8172B3", "pca3": "#DD8452", "pca1": "#999999", "eq": "#CCB974"}

names = ["沪深300", "中证500", "中证1000", "上证50", "金融", "消费", "医药",
         "科技", "新能源", "军工", "煤炭", "黄金"]
K = len(names)
rng = np.random.default_rng(20260719)

# ---- 3 个潜在宏观因子的载荷结构 ----
B = np.zeros((K, 3))
B[:, 0] = 0.6 + 0.2 * rng.standard_normal(K)          # 市场: 全正
value_growth = np.array([1, 0.6, -0.4, 1.1, 1.2, 0.3, -0.3, -0.9, -0.7, -0.2, 0.5, -0.5])
B[:, 1] = 0.5 * value_growth / value_growth.std()     # 利率/价值成长劈叉
commodity_inflation = np.array([-0.2, -0.3, -0.4, -0.1, -0.5, -0.1, 0.0, 0.2, 0.6, 0.4, 1.0, 0.9])
B[:, 2] = 0.5 * commodity_inflation / commodity_inflation.std()  # 商品/通胀

# ---- 因子收益 + 特质噪声 ----
T_total = 1008  # 约 4 年
fvol = np.array([0.016, 0.011, 0.013])
F = rng.standard_normal((T_total, 3)) * fvol          # 因子日收益
idio = rng.standard_normal((T_total, K)) * 0.004       # 特质噪声(远小于因子, 让3因子主导截面)
R = F @ B.T + idio                                     # T × K 资产收益

# ---- 训练/测试切分 ----
split = T_total // 2
Rtr, Rte = R[:split], R[split:]

# ---- 训练期 PCA ----
def pca_cov(Rtr):
    Xc = Rtr - Rtr.mean(axis=0)
    cov = Xc.T @ Xc / (len(Xc) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return eigval[order], eigvec[:, order], cov

ev_tr, evec_tr, cov_tr = pca_cov(Rtr)
var_exp = ev_tr / ev_tr.sum()
cum_var = np.cumsum(var_exp)

delta = np.median(np.diag(cov_tr)) * 0.1  # 收缩对角正则项

def shrink_cov(ev, evec, k):
    S = delta * np.eye(K)
    for j in range(k):
        S += ev[j] * np.outer(evec[:, j], evec[:, j])
    return S

def min_var_weights(cov):
    inv = np.linalg.inv(cov + 1e-10 * np.eye(K))
    ones = np.ones(K)
    w = inv @ ones
    return w / w.sum()

cov_full = cov_tr                                        # 朴素全样本协方差(无收缩, 过拟合噪声)
cov_pca1 = shrink_cov(ev_tr, evec_tr, 1)
cov_pca3 = shrink_cov(ev_tr, evec_tr, 3)
w_full = min_var_weights(cov_full)
w_pca1 = min_var_weights(cov_pca1)
w_pca3 = min_var_weights(cov_pca3)
w_eq = np.ones(K) / K

# 测试期真实组合波动率 (基于训练期全样本估计, 仅作基准对照)
def port_vol(w, Rte):
    r = Rte @ w
    return r.std() * np.sqrt(252) * 100
vol_full = port_vol(w_full, Rte)
vol_pca1 = port_vol(w_pca1, Rte)
vol_pca3 = port_vol(w_pca3, Rte)
vol_eq = port_vol(w_eq, Rte)

# ============================================================================
# 短估计窗口过拟合实验: 真实场景下 T 往往远小于 K 的“有效维数”
# 用 120 天滚动窗口每月重估协方差, 持有到下月; 比较 raw 与 PCA-3 最小方差组合的
# 真实实现波动率 —— raw 在短窗口下过拟合(权重剧烈摆动), PCA-3 收缩更稳定
# ============================================================================
win = 90
step = 21  # 月度再平衡
raw_held = []
pca_held = []
raw_w_series = []
pca_w_series = []
start = split  # 从训练/测试分界开始
for t0 in range(start, T_total - win - step, step):
    Rwin = R[t0:t0 + win]
    Xc = Rwin - Rwin.mean(axis=0)
    cov_w = Xc.T @ Xc / (len(Xc) - 1)
    ev_w, evec_w, _ = pca_cov(Rwin)
    c_full_w = cov_w
    c_pca3_w = shrink_cov(ev_w, evec_w, 3)
    w_r = min_var_weights(c_full_w)
    w_p = min_var_weights(c_pca3_w)
    raw_w_series.append(w_r)
    pca_w_series.append(w_p)
    # 持有到下一窗口(t0+win 起 step 天)
    Rh = R[t0 + win:t0 + win + step]
    raw_held.append((Rh @ w_r).std() * np.sqrt(252) * 100)
    pca_held.append((Rh @ w_p).std() * np.sqrt(252) * 100)
raw_held = np.array(raw_held); pca_held = np.array(pca_held)
raw_w_series = np.array(raw_w_series); pca_w_series = np.array(pca_w_series)
vol_raw_short = raw_held.mean()
vol_pca_short = pca_held.mean()
gross_raw = np.abs(raw_w_series).sum(axis=1).mean()
gross_pca = np.abs(pca_w_series).sum(axis=1).mean()
# 月度换手率(相邻窗口权重变化)—— PCA-3 收缩后更稳 = 更低交易成本
turn_raw = np.mean([np.sum(np.abs(raw_w_series[i+1]-raw_w_series[i])) for i in range(len(raw_w_series)-1)])
turn_pca = np.mean([np.sum(np.abs(pca_w_series[i+1]-pca_w_series[i])) for i in range(len(pca_w_series)-1)])

# PC1 与市场(等权)收益相关性
mkt = Rtr @ np.ones(K) / K
pc1_score = (Rtr - Rtr.mean(0)) @ evec_tr[:, 0]
corr_pc1_mkt = np.corrcoef(pc1_score, mkt)[0, 1]

# ---------------------------------------------------------------------------
# 图 1: 碎石图 (方差解释)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.0))
bars = ax.bar(range(1, K + 1), var_exp * 100,
              color=[C["pc1"] if i == 0 else (C["pc2"] if i == 1 else (C["pc3"] if i == 2 else C["grid"])) for i in range(K)])
ax.plot(range(1, K + 1), cum_var * 100, color="#2F4B7C", lw=1.8, marker="o", ms=4, label="累计解释度")
ax.axhline(cum_var[2] * 100, color=C["pc3"], ls="--", lw=1.0,
           label="前3主成分 = %.1f%%" % (cum_var[2] * 100))
ax.set_xlabel("主成分序号"); ax.set_ylabel("解释方差 (%)")
ax.set_title("截面波动是低维的: 前 3 主成分解释 %.1f%% 的 12 资产协方差" % (cum_var[2] * 100))
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pca_scree.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 图 2: 前 3 主成分载荷 (因子结构)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(K); w = 0.26
ax.bar(x - w, evec_tr[:, 0], w, color=C["pc1"], label="PC1 (市场)")
ax.bar(x, evec_tr[:, 1], w, color=C["pc2"], label="PC2 (价值-成长)")
ax.bar(x + w, evec_tr[:, 2], w, color=C["pc3"], label="PC3 (商品/通胀)")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("主成分载荷")
ax.set_title("前 3 主成分载荷: PC1 普遍为正(市场), PC2/PC3 劈叉(风格分化)")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pca_loadings.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 图 3: 三种协方差估计下最小方差组合权重
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(K); w = 0.26
ax.bar(x - w, w_full * 100, w, color=C["full"], label="全样本协方差")
ax.bar(x, w_pca1 * 100, w, color=C["pca1"], label="PCA-1 收缩")
ax.bar(x + w, w_pca3 * 100, w, color=C["pca3"], label="PCA-3 收缩")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("权重 (%)")
ax.set_title("最小方差组合权重: 全样本协方差过拟合噪声, PCA 收缩更平滑")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pca_weights.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 图 4: 短窗口滚动重估 —— raw vs PCA-3 最小方差组合真实实现波动率
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
ax1.plot(raw_w_series[:, 7] * 100, color=C["full"], lw=1.6, label="raw 全样本协方差")
ax1.plot(pca_w_series[:, 7] * 100, color=C["pca3"], lw=1.6, label="PCA-3 收缩")
ax1.set_ylabel("科技股权重 (%)")
ax1.set_title("短窗口(120天)重估: raw 权重剧烈摆动(过拟合), PCA-3 收缩更稳健")
ax1.legend(fontsize=8); ax1.grid(True, color=C["grid"], lw=0.6)
idx = np.arange(len(raw_held))
ax2.plot(idx, raw_held, color=C["full"], lw=1.6, label=f"raw 实现波动 ({vol_raw_short:.1f}%)")
ax2.plot(idx, pca_held, color=C["pca3"], lw=1.8, label=f"PCA-3 实现波动 ({vol_pca_short:.1f}%)")
ax2.set_xlabel("月度再平衡回合"); ax2.set_ylabel("月度实现年化波动 (%)")
ax2.set_title("短窗口下 PCA-3 最小方差组合真实波动更低 (= 抗过拟合)")
ax2.legend(fontsize=8); ax2.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pca_backtest.png")); plt.close(fig)

print("=" * 66)
print("宏观PCA定价 关键数字")
print("=" * 66)
print("资产数 K =", K, " 训练期 T =", split, " 测试期 =", len(Rte))
print("前3主成分累计解释方差 = %.1f%%" % (cum_var[2] * 100))
print("PC1=%.1f%% PC2=%.1f%% PC3=%.1f%%" % (var_exp[0]*100, var_exp[1]*100, var_exp[2]*100))
print("PC1 与等权市场收益相关 = %.3f (=市场因子)" % corr_pc1_mkt)
print("特质噪声占收益方差 ≈ %.1f%%" % (var_exp[3:].sum() * 100))
print("测试期最小方差组合真实年化波动率(全样本估计, 仅基准):")
print("  全样本协方差 = %.2f%%" % vol_full)
print("  PCA-1 收缩   = %.2f%%" % vol_pca1)
print("  PCA-3 收缩   = %.2f%%" % vol_pca3)
print("  等权         = %.2f%%" % vol_eq)
print("短窗口(90天)滚动重估 实现年化波动:")
print("  raw 全样本协方差 = %.2f%% (平均杠杆|w| = %.2f, 月度换手=%.2f)" % (vol_raw_short, gross_raw, turn_raw))
print("  PCA-3 收缩       = %.2f%% (平均杠杆|w| = %.2f, 月度换手=%.2f)" % (vol_pca_short, gross_pca, turn_pca))
print("  PCA-3 相对 raw 波动降幅 = %.0f%%" % (100 * (1 - vol_pca_short / vol_raw_short)))
print("  PCA-3 相对 raw 换手降幅 = %.0f%%" % (100 * (1 - turn_pca / turn_raw)))
print("DONE ->", OUT, os.listdir(OUT))
