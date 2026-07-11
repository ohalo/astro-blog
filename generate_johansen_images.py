#!/usr/bin/env python3
"""
为文章「Johansen 多变量协整：不止两两配对，三组一起套利的统计基础」(johansen-multivariate-cointegration)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. johansen_series.png      三只共享共同趋势的价格序列（三变量协整）
  2. johansen_residual.png    协整残差 z-score：平稳（均值回复），不同于原始价格
  3. johansen_eig.png         特征值条形图：1 个远大于 0 → 协整秩 r=1
  4. johansen_spread_trade.png 三腿组合价差与交易带：三组一起套利的可视化

数值校验：Johansen 估计的协整向量应与真实 beta=(1,-0.5,-0.5) 一致；
          迹检验应判定协整秩 r=1。
"""
import os
import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "johansen-multivariate-cointegration")
os.makedirs(D, exist_ok=True)

C = {"p1": "#4C72B0", "p2": "#C44E52", "p3": "#55A868", "grid": "#DDDDDD", "band": "#DDDDDD"}
np.set_printoptions(suppress=True, precision=4)

# ============================================================
# Osterwald-Lenum (1992) 渐近临界值（含截距项、CE 中无趋势）
# 键 (n, r)：n=变量数, r=原假设下协整关系个数
# ============================================================
TRACE_CV = {
    (2, 0): (17.853, 19.960, 24.601), (2, 1): (7.503, 9.240, 12.970),
    (3, 0): (32.004, 34.910, 41.070), (3, 1): (17.853, 19.960, 24.601),
    (3, 2): (7.503, 9.240, 12.970),
    (4, 0): (49.652, 53.120, 60.160), (4, 1): (32.004, 34.910, 41.070),
    (4, 2): (17.853, 19.960, 24.601), (4, 3): (7.503, 9.240, 12.970),
}
MAXEIG_CV = {
    (2, 0): (14.965, 16.540, 20.434), (2, 1): (7.141, 8.790, 12.252),
    (3, 0): (28.981, 31.606, 37.145), (3, 1): (14.965, 16.540, 20.434),
    (3, 2): (7.141, 8.790, 12.252),
    (4, 0): (45.286, 48.460, 55.314), (4, 1): (28.981, 31.606, 37.145),
    (4, 2): (14.965, 16.540, 20.434), (4, 3): (7.141, 8.790, 12.252),
}

# ============================================================
# 模拟三只协整的价格序列：从误差修正模型(VECM)生成
#   dX_t = alpha * (beta' X_{t-1}) + eps_t ，beta=(1,-0.5,-0.5)
# 这样 X 是 I(1) 且存在 1 个协整关系；Johansen 应能还原 beta 与秩 r=1。
# ============================================================
def simulate(T=1000, seed=42):
    rng = np.random.default_rng(seed)
    beta = np.array([1.0, -0.5, -0.5])
    alpha = np.array([-0.06, 0.03, 0.03])     # 调整速度（负值=>均值回复）
    Pi = np.outer(alpha, beta)                 # 误差修正矩阵（秩 1）
    Sigma = np.diag([0.01, 0.01, 0.01])
    X = np.zeros((T, 3))
    X[0] = rng.normal(0, 0.1, 3)
    for t in range(1, T):
        eps = rng.multivariate_normal(np.zeros(3), Sigma)
        X[t] = X[t - 1] + Pi @ X[t - 1] + eps
    return X

X = simulate(T=2500, seed=42)
n = 3
T = X.shape[0]

# ============================================================
# Johansen 检验（VECM: dX_t = Pi X_{t-1} + eps_t，含截距；VAR 阶数 p=1）
#   R0 = dX 对常数项回归的残差； R1 = X_{t-1} 对常数项回归的残差
#   （X_{t-1} 本身不是短期回归元，否则会把协整信息从 R0 中抹掉）
# ============================================================
def johansen(X):
    dX = np.diff(X, axis=0)              # (T-1, n)
    Xlag = X[:-1]                         # (T-1, n)
    TT = dX.shape[0]
    R0 = dX - dX.mean(0)                  # 残差：dX 净短期（常数的）动态
    R1 = Xlag - Xlag.mean(0)             # 残差：水平净常数动态
    S00 = R0.T @ R0 / TT
    S11 = R1.T @ R1 / TT
    S01 = R0.T @ R1 / TT
    S10 = S01.T
    # 广义特征值问题：A x = lambda S11 x，A = S10 S00^{-1} S01
    A = S10 @ np.linalg.inv(S00) @ S01
    lam, beta = sla.eig(A, S11)
    lam = np.real(lam)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    beta = beta[:, order]
    # 萃取协整向量（按第 0 个分量归一化）
    beta_norm = beta / beta[0, :]
    # 调整速度 alpha（r=1）
    b1 = beta[:, :1]
    alpha = S01 @ b1 @ np.linalg.inv(b1.T @ S11 @ b1)   # (n,1)
    # 迹统计量与最大特征值统计量
    # 约定：lambda 降序排列，lambda[0] 最大；Q_r = -T sum_{i=r}^{n-1} ln(1-lambda_i)
    trace_stat = [-TT * np.sum(np.log(1 - lam[r:])) for r in range(n)]
    max_stat = [-TT * np.log(1 - lam[r]) if r < n else 0.0 for r in range(n)]
    return lam, beta_norm, alpha, np.array(trace_stat), np.array(max_stat), TT

lam, beta_norm, alpha, trace_stat, max_stat, TT = johansen(X)

# ============================================================
# 校验与关键数字
# ============================================================
print("=== Johansen 估计结果（n=3, T=%d）===" % TT)
print("特征值 lambda =", np.round(lam, 4))
print("估计协整向量 beta(归一化) =", np.round(beta_norm[:, 0], 4), " （真实 (1,-0.5,-0.5)）")
print("调整速度 alpha =", np.round(alpha[:, 0], 4))
print("\n迹统计量 Trace: r<=0=%.2f  r<=1=%.2f  r<=2=%.2f" % (trace_stat[0], trace_stat[1], trace_stat[2]))
print("max-eig:    r=0=%.2f  r=1=%.2f  r=2=%.2f" % (max_stat[0], max_stat[1], max_stat[2]))
print("\n临界值(95%%): Trace", [TRACE_CV[(3, r)][1] for r in range(3)],
      " MaxEig", [MAXEIG_CV[(3, r)][1] for r in range(3)])

# 用估计 beta 构造价差与半衰期
b = beta_norm[:, 0]
spread = X @ b
kappa = float(b @ alpha[:, 0])                 # 误差修正速度
half_life = -np.log(2) / np.log(1 + kappa) if (1 + kappa) > 0 else np.nan
print("\n协整价差 kappa = beta'alpha = %.4f   半衰期 ~= %.1f 期" % (kappa, half_life))
print("价差一阶自相关 = %.4f （<1 即均值回复）" % np.corrcoef(spread[:-1], spread[1:])[0, 1])

# ============================================================
# 图 1：三只价格序列
# ============================================================
t = np.arange(T)
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, X[:, 0], color=C["p1"], lw=0.9); ax[0].set_ylabel("资产 1\np1")
ax[1].plot(t, X[:, 1], color=C["p2"], lw=0.9); ax[1].set_ylabel("资产 2\np2")
ax[2].plot(t, X[:, 2], color=C["p3"], lw=0.9); ax[2].set_ylabel("资产 3\np3")
for a in ax:
    a.grid(True, color=C["grid"], lw=0.6); a.set_xlim(0, T)
ax[2].set_xlabel("交易日")
fig.suptitle("三只价格序列：走势同向（共享共同趋势），但水平与斜率各异", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(D, "johansen_series.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：协整残差 z-score（平稳）对比单只价格的 z-score
# ============================================================
z_spread = (spread - spread.mean()) / spread.std()
z_p1 = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, z_p1, color=C["p1"], lw=0.8, alpha=0.6, label="资产1 价格 z-score（非平稳/I(1)）")
ax.plot(t, z_spread, color="black", lw=1.2, label="协整残差（平稳/均值回复）")
ax.axhline(2, color=C["p2"], ls="--", lw=1)
ax.axhline(-2, color=C["p2"], ls="--", lw=1)
ax.set_xlabel("交易日"); ax.set_ylabel("z-score")
ax.set_title("协整残差围绕 0 波动（平稳），而单只价格持续漂移（非平稳）")
ax.legend(loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "johansen_residual.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：特征值条形图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(["特征根1", "特征根2", "特征根3"], lam, color=[C["p1"], C["p2"], C["p3"]])
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("特征值")
ax.set_title("Johansen 特征值：第1个显著>0，第2/3个接近0 -> 协整秩 r=1")
for b, v in zip(bars, lam):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, "%.3f" % v, ha="center", fontsize=10)
ax.set_ylim(0, max(lam) * 1.25)
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "johansen_eig.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：三腿组合价差与交易带（三组一起套利）
# ============================================================
entry = 1.5
pos = np.zeros(T)
for i in range(1, T):
    if z_spread[i - 1] > entry:
        pos[i] = -1      # 价差偏高 -> 做空组合
    elif z_spread[i - 1] < -entry:
        pos[i] = 1       # 价差偏低 -> 做多组合
    else:
        pos[i] = pos[i - 1]
ret = np.zeros(T)
ret[1:] = pos[:-1] * np.diff(z_spread)
eq = np.cumsum(ret)

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(t, z_spread, color="black", lw=1.1, label="协整价差 z-score")
ax[0].axhline(entry, color=C["p2"], ls="--", lw=1, label="±%.1f 入场带" % entry)
ax[0].axhline(-entry, color=C["p2"], ls="--", lw=1)
ax[0].fill_between(t, -entry, entry, color=C["band"], alpha=0.4)
ax[0].set_ylabel("价差 z-score"); ax[0].legend(loc="upper right")
ax[0].set_title("三腿协整组合：价差越界即反向开仓（三组一起套利）")
ax[1].plot(t, eq, color=C["p1"], lw=1.0, label="策略净值（价差 delta 近似）")
ax[1].axhline(0, color="black", lw=0.6)
ax[1].set_ylabel("累计收益"); ax[1].set_xlabel("交易日")
ax[1].legend(loc="upper left"); ax[1].grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "johansen_spread_trade.png"), dpi=130)
plt.close()

print("\n图片已保存到:", D)
