#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「等风险贡献组合(ERC)：让每个资产对总风险的贡献完全相等」生成真实配图与统计数字。

核心逻辑(基于 Spinu/Maillard/Roncalli 等 ERC / Risk Budgeting 文献):
  - 马科维茨最大夏普对协方差极度敏感(估计噪声-> 极端权重/空头)
  - 风险平价/等风险贡献放弃预测收益, 只要求「每个资产对组合总风险(波动率)的边际贡献相等」
  - ERC 权重满足:  w_i * (Σw)_i == 常数的解; 当相关性一致时退化为逆波动加权
  - 用「 cyclical coordinate descent (CCD)」迭代求解: 固定其他资产, 对每个资产解析更新
  - 对照: ERC vs 等权 vs 最小方差 vs 逆波动, 看风险贡献是否均匀

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  erc_risk_contribution.png   —— 各资产风险贡献对比(ERC 均匀 / 等权不均)
  erc_weights.png             —— 各类权重分配对比
  erc_convergence.png         —— CCD 迭代使风险贡献方差收敛到 0
  erc_corr_robust.png         —— 不同相关性结构下 ERC 权重与风险贡献
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
D = os.path.join(BASE, "equal-risk-contribution")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260717)

# ================= 资产设定 =================
n = 6
# 年化波动率(故意拉开差距: 股票高、债券低)
vols = np.array([0.22, 0.18, 0.15, 0.10, 0.08, 0.05])
# 相关性矩阵: 股票间高相关、与债券低/负相关
C = np.array([
    [1.00, 0.70, 0.55, 0.20, 0.10, -0.10],
    [0.70, 1.00, 0.60, 0.25, 0.12, -0.05],
    [0.55, 0.60, 1.00, 0.30, 0.15, 0.00],
    [0.20, 0.25, 0.30, 1.00, 0.50, 0.20],
    [0.10, 0.12, 0.15, 0.50, 1.00, 0.35],
    [-0.10, -0.05, 0.00, 0.20, 0.35, 1.00],
])
Cov = (vols[:, None] * vols[None, :]) * C

def risk_contrib(w, Cov):
    port_var = w @ Cov @ w
    mrc = Cov @ w              # 边际风险贡献
    rc = w * mrc               # 成分风险贡献
    return rc, np.sqrt(port_var)

def rc_variance(w, Cov):
    rc, _ = risk_contrib(w, Cov)
    return rc.var()

# ---- 各类权重方案 ----
w_equal = np.repeat(1.0 / n, n)

# 逆波动加权(单资产风险平价: 假设不相关时的最优)
w_ivol = (1.0 / vols); w_ivol /= w_ivol.sum()

# 最小方差(解析: 对协方差求逆归一)—— 注意会偏债券、可能负权重
try:
    w_mv = np.linalg.solve(Cov, np.ones(n)); w_mv /= w_mv.sum()
except Exception:
    w_mv = w_equal.copy()

# ---- ERC 求解: 风险预算固定点迭代 (Spinu 2013) ----
# ERC 条件: 每个资产的风险贡献 RC_i = w_i·(Σw)_i 相等.
# 固定点: w_i ∝ 1/(Σw)_i  (归一化后 RC_i 完全相等).
# 证明: 若 w_i = c/(Σw)_i, 则 RC_i = w_i·(Σw)_i = c 为常数.
def erc_fixedpoint(Cov, max_iter=2000, tol=1e-13):
    n = Cov.shape[0]
    w = np.repeat(1.0 / n, n)          # 初始化等权
    for it in range(max_iter):
        sw = Cov @ w                    # (Σw) 向量
        w_new = 1.0 / sw                # ∝ 1/(Σw)_i
        w_new = w_new / w_new.sum()     # 归一化, 保持 Σw=1
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            return w, it + 1
        w = w_new
    return w, max_iter

w_erc, iters = erc_fixedpoint(Cov)
log(f"ERC 迭代收敛步数={iters}")
log(f"ERC 权重={np.round(w_erc, 3)}")

# ---- 风险贡献对比 ----
rc_equal, _ = risk_contrib(w_equal, Cov)
rc_erc, _ = risk_contrib(w_erc, Cov)
rc_ivol, _ = risk_contrib(w_ivol, Cov)
rc_mv, _ = risk_contrib(w_mv, Cov)
log(f"风险贡献方差 -> 等权:{rc_equal.var():.2e} 逆波动:{rc_ivol.var():.2e} 最小方差:{rc_mv.var():.2e} ERC:{rc_erc.var():.2e}")
log(f"ERC 风险贡献(每资产%)= {np.round(rc_erc/rc_erc.sum()*100, 2)}")

# ================= 绘图 =================
plt.rcParams["figure.dpi"] = 130
names = [f"A{i+1}" for i in range(n)]
C_ERC, C_OTHER, C_BAR = "#2563eb", "#dc2626", "#16a34a"

# 图1: 风险贡献对比(等权 vs ERC)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
axes[0].bar(names, rc_equal / rc_equal.sum() * 100, color=C_OTHER)
axes[0].set_title("等权: 风险贡献不均", fontsize=11)
axes[0].set_ylabel("风险贡献占比 (%)"); axes[0].set_ylim(0, 45)
axes[0].axhline(100/n, color="gray", ls=":", lw=1)
for i, v in enumerate(rc_equal / rc_equal.sum() * 100):
    axes[0].text(i, v + 0.8, f"{v:.0f}", ha="center", fontsize=9)
axes[1].bar(names, rc_erc / rc_erc.sum() * 100, color=C_ERC)
axes[1].set_title("ERC: 风险贡献均匀", fontsize=11)
axes[1].set_ylim(0, 45)
axes[1].axhline(100/n, color="gray", ls=":", lw=1)
for i, v in enumerate(rc_erc / rc_erc.sum() * 100):
    axes[1].text(i, v + 0.8, f"{v:.0f}", ha="center", fontsize=9)
fig.suptitle("等风险贡献 vs 等权: 谁在替组合扛风险?", fontsize=12)
for ax in axes: ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(D, "erc_risk_contribution.png")); plt.close(fig)

# 图2: 权重对比
fig, ax = plt.subplots(figsize=(7.5, 4.4))
wmat = np.vstack([w_equal, w_ivol, w_mv, w_erc]) * 100
labels = ["等权", "逆波动", "最小方差", "ERC"]
colors = ["#9ca3af", "#f59e0b", C_OTHER, C_ERC]
x = np.arange(n); bw = 0.2
for k, (row, lab, c) in enumerate(zip(wmat, labels, colors)):
    ax.bar(x + (k - 1.5) * bw, row, bw, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("权重 (%)"); ax.set_title("各类方案权重分配对比", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(D, "erc_weights.png")); plt.close(fig)

# 图3: 固定点迭代收敛轨迹(风险贡献方差)
def erc_fp_trace(Cov, max_iter=200):
    n = Cov.shape[0]; w = np.repeat(1.0 / n, n)
    trace = []
    for it in range(max_iter):
        sw = Cov @ w
        w = 1.0 / sw
        w = w / w.sum()
        trace.append(rc_variance(w, Cov))
    return np.array(trace)
trace = erc_fp_trace(Cov)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(trace, color=C_ERC, lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("迭代步"); ax.set_ylabel("风险贡献方差 (log)")
ax.set_title("CCD 迭代: 风险贡献方差收敛到 0", fontsize=12)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "erc_convergence.png")); plt.close(fig)

# 图4: 不同相关性结构下 ERC 的鲁棒性
# 构造高相关 vs 低相关两种市场, 看 ERC 风险贡献是否仍均匀
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, rho, title in [(axes[0], 0.7, "高相关市场 (ρ≈0.7)"), (axes[1], 0.15, "低相关市场 (ρ≈0.15)")]:
    Ch = np.eye(n) * (1 - rho) + rho
    Covh = (vols[:, None] * vols[None, :]) * Ch
    wh, _ = erc_fixedpoint(Covh)
    rch, _ = risk_contrib(wh, Covh)
    ax.bar(names, rch / rch.sum() * 100, color=C_ERC)
    ax.axhline(100/n, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 45); ax.set_title(title, fontsize=11); ax.grid(alpha=0.3, axis="y")
fig.suptitle("ERC 在不同相关性结构下风险贡献仍≈均匀", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(D, "erc_corr_robust.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(os.listdir(D)))
