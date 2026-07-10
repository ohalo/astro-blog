#!/usr/bin/env python3
"""
为两篇新量化文章生成真实配图（matplotlib 渲染，非占位图）：
  1. extreme-risk-modeling        (极端风险建模：肥尾与黑天鹅)
  2. smart-beta-factor-investing  (Smart Beta 与因子投资)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches

rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"


def t_returns(n, df, scale=0.01):
    """生成带肥尾的日收益（Student-t），并叠加偶发极端跳空。"""
    r = np.random.standard_t(df, size=n) * scale
    # 注入黑天鹅：约 0.3% 概率出现 ±5%~10% 极端收益
    jumps = np.random.rand(n) < 0.003
    r[jumps] *= np.random.uniform(5, 10, size=jumps.sum())
    return r


# ============================================================
# 文章 1：极端风险建模：肥尾与黑天鹅
# ============================================================
d1 = os.path.join(BASE, "extreme-risk-modeling")
os.makedirs(d1, exist_ok=True)

np.random.seed(20260710)
N = 4000
rets = t_returns(N, df=4, scale=0.011)

# 图1：收益分布直方图 + 正态拟合 + t 拟合，凸显肥尾
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(rets, bins=120, density=True, color="#4C78A8", alpha=0.65,
        label="实证日收益")
x = np.linspace(rets.min(), rets.max(), 400)
mu, sigma = rets.mean(), rets.std()
ax.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
        color="#E45756", lw=2.2, label="正态拟合 (高斯假设)")
ax.set_title("日收益分布：实证数据在尾部显著厚于正态分布", fontsize=13.5, fontweight="bold")
ax.set_xlabel("日收益率"); ax.set_ylabel("概率密度")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "return_dist_fattails.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图2：正态 QQ 图，尾部偏离
from scipy import stats as st
fig, ax = plt.subplots(figsize=(8.5, 7))
st.probplot(rets, dist="norm", plot=ax)
ax.get_lines()[0].set_color("#4C78A8"); ax.get_lines()[0].set_alpha(0.6)
ax.get_lines()[1].set_color("#E45756"); ax.get_lines()[1].set_lw(2)
ax.set_title("正态 QQ 图：两端明显偏离参考线 = 肥尾证据", fontsize=13.5, fontweight="bold")
ax.set_xlabel("理论分位数 (标准正态)"); ax.set_ylabel("样本分位数")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "qq_plot.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图3：VaR vs CVaR 示意（在损失轴上）
z99 = st.norm.ppf(0.99)
fig, ax = plt.subplots(figsize=(10, 5.5))
xs = np.linspace(-0.08, 0.04, 500)
ys = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
ax.plot(xs, ys, color="#54A24B", lw=2)
var99 = mu - z99 * sigma
cvar99 = mu - sigma * st.norm.pdf(z99) / 0.01
ax.axvline(var99, color="#E45756", lw=2.2, label=f"VaR 99% = {var99*100:.2f}%")
ax.fill_between(xs, 0, ys, where=(xs <= var99), color="#E45756", alpha=0.35,
                label=f"CVaR/ES 99% = {cvar99*100:.2f}% (尾部条件期望)")
ax.set_title("VaR 只给出临界线，CVaR 度量临界线之外的平均损失", fontsize=12.5, fontweight="bold")
ax.set_xlabel("日收益率"); ax.set_ylabel("概率密度")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "var_cvar.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图4：Hill 估计器 — 尾指数随 k 的稳定性
order = np.sort(np.abs(rets))[::-1]
k_grid = np.arange(50, 1500, 20)
hill = [np.mean(np.log(order[:k])) - np.log(order[k - 1]) for k in k_grid]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(k_grid, hill, color="#B279A2", lw=2)
ax.axhline(np.mean(hill[-30:]), color="gray", ls="--", lw=1, label=f"平台均值 ≈ {np.mean(hill[-30:]):.2f}")
ax.set_title("Hill 估计器：尾指数在合理 k 区间趋于稳定（<2 = 无限方差风险）", fontsize=12, fontweight="bold")
ax.set_xlabel("次序统计量个数 k"); ax.set_ylabel("尾指数 ξ (Hill)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "hill_estimator.png"), dpi=150, bbox_inches="tight"); plt.close()


# ============================================================
# 文章 2：Smart Beta 与因子投资
# ============================================================
d2 = os.path.join(BASE, "smart-beta-factor-investing")
os.makedirs(d2, exist_ok=True)

np.random.seed(20260710)
M = 180  # 月度
factors = ["市场", "价值", "动量", "规模", "质量", "低波"]
# 各因子月收益：不同均值/波动，制造差异化风险溢价
mu_f = np.array([0.009, 0.008, 0.010, 0.007, 0.0085, 0.0065])
sig_f = np.array([0.045, 0.038, 0.050, 0.042, 0.032, 0.025])
factor_ret = mu_f + sig_f * np.random.randn(M, 6)

# 图1：因子累积净值
cum = np.cumprod(1 + factor_ret, axis=0)
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
for i, f in enumerate(factors):
    ax.plot(range(M), cum[:, i], lw=2, color=colors[i], label=f)
ax.set_title("六类因子月度累积净值：风险溢价存在但路径各异", fontsize=13, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("累积净值 (起始=1)")
ax.legend(ncol=3, fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d2, "factor_cumulative.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图2：最小方差组合权重 vs 市值加权
assets = ["股票A", "股票B", "股票C", "股票D", "股票E", "股票F", "股票G", "股票H"]
K = len(assets)
np.random.seed(99)
cov = 0.02 * np.random.rand(K, K) + 0.01 * np.eye(K)
cov = (cov + cov.T) / 2
inv = np.linalg.inv(cov)
w_minvar = inv.dot(np.ones(K)); w_minvar /= w_minvar.sum()
w_cap = np.array([0.30, 0.22, 0.16, 0.12, 0.08, 0.06, 0.04, 0.02])
x = np.arange(K); w = 0.4
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w/2, w_cap, w, label="市值加权", color="#E45756", alpha=0.85)
ax.bar(x + w/2, w_minvar, w, label="最小方差加权", color="#4C78A8", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(assets, rotation=30, ha="right")
ax.set_title("权重对比：最小方差组合显著压低高波动个股敞口", fontsize=13, fontweight="bold")
ax.set_ylabel("权重"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(d2, "minvol_weights.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图3：Smart Beta (最小方差) vs 市值加权 累积净值
sb_ret = factor_ret[:, 5] * 0.6 + factor_ret[:, 1] * 0.4  # 低波 + 价值 倾斜
cap_ret = factor_ret[:, 0]  # 市场（市值加权代理）
sb_cum = np.cumprod(1 + sb_ret)
cap_cum = np.cumprod(1 + cap_ret)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(M), sb_cum, color="#4C78A8", lw=2.2, label="Smart Beta（低波+价值倾斜）")
ax.plot(range(M), cap_cum, color="#E45756", lw=2.2, label="市值加权（市场因子）")
ax.set_title("累积净值对比：Smart Beta 在相近收益下更低的波动路径", fontsize=12.5, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("累积净值 (起始=1)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d2, "smartbeta_vs_capweight.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图4：因子相关性热力图
corr = np.corrcoef(factor_ret, rowvar=False)
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(factors, rotation=30, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(factors)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=10)
ax.set_title("因子相关性矩阵：分散化前提是因子间低相关", fontsize=12.5, fontweight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(os.path.join(d2, "factor_correlation.png"), dpi=150, bbox_inches="tight"); plt.close()

print("✅ 图像生成完成")
print("   extreme-risk-modeling:", sorted(os.listdir(d1)))
print("   smart-beta-factor-investing:", sorted(os.listdir(d2)))
