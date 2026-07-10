#!/usr/bin/env python3
"""
为文章「最大分散化组合：用相关性加权超越风险平价」(maximum-diversification-portfolio) 生成真实配图。
数据：模拟 6 资产日度收益，估计协方差后对比 等权 / 风险平价(ERC) / 最小方差 / 最大分散化 4 种权重。
图表：
  1. mdp_weights.png              四种权重对比（长仓）
  2. mdp_diversification_ratio.png 四种组合的分散化比率 DR 对比
  3. mdp_corr_heatmap.png         资产相关性矩阵（结构说明为何低相关资产被推到前台）
  4. mdp_equity.png               样本外 4 年净值对比（log 轴）+ 最大回撤
全部为真实数值计算，非占位图。
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
D = os.path.join(BASE, "maximum-diversification-portfolio")
os.makedirs(D, exist_ok=True)
np.random.seed(12)

NAMES = ["美股", "欧股", "新兴市场", "国债", "黄金", "商品"]

# ============================================================
# 1) 模拟 6 资产日度收益（双因子结构 + 特异波动），保证低相关资产存在
# ============================================================
T = 252 * 10
ann_mu = np.array([0.09, 0.08, 0.10, 0.03, 0.05, 0.07])
# 因子：风险偏好(高波动) + 避险(低波动)
vm = 0.00018
f1 = np.zeros(T)
for t in range(T):
    vm = max(0.00006, vm + 0.04 * (0.00018 - vm) + np.random.randn() * 0.00005)
    if np.random.rand() < 0.007:
        vm *= 3.2
    f1[t] = np.random.randn() * np.sqrt(vm)
f2 = np.random.randn(T) * np.sqrt(0.00003) + 0.00002   # 避险因子
L = np.array([
    [1.00, -0.25],   # 美股
    [0.95, -0.20],   # 欧股
    [0.85, -0.10],   # 新兴市场
    [-0.10, 0.85],   # 国债
    [0.10, 0.55],    # 黄金
    [0.45, -0.05],   # 商品
])
idio_var = np.array([0.00006, 0.00007, 0.00010, 0.00002, 0.00008, 0.00013])
idio_mu = ann_mu / 252.0
R = np.zeros((T, 6))
for i in range(6):
    R[:, i] = idio_mu[i] + L[i, 0] * f1 + L[i, 1] * f2 + np.random.randn(T) * np.sqrt(idio_var[i])

# 训练/测试切分（避免样本内前视）：前 6 年估计协方差，后 4 年测净值
split = int(T * 0.6)
R_train, R_test = R[:split], R[split:]
Sigma = np.cov(R_train.T) * 252.0
sigma = np.sqrt(np.diag(Sigma))
corr = Sigma / np.outer(sigma, sigma)
print("年化波动:", np.round(sigma, 3))
print("相关矩阵:\n", np.round(corr, 2))

# ============================================================
# 2) 四种权重
# ============================================================
n = 6
w_ew = np.repeat(1.0 / n, n)

def erc_weights(S, n_iter=3000):
    from scipy.optimize import minimize
    def obj(w):
        pv = w @ S @ w
        rc = w * (S @ w) / pv
        return np.sum((rc - 1.0 / n) ** 2)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, None)] * n
    res = minimize(obj, w_ew, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": n_iter, "ftol": 1e-14})
    return res.x / res.x.sum()

def min_var(S):
    from scipy.optimize import minimize
    def obj(w):
        return w @ S @ w
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, None)] * n
    res = minimize(obj, w_ew, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 3000, "ftol": 1e-14})
    return res.x / res.x.sum()

def max_div(S, sig):
    from scipy.optimize import minimize
    def obj(w):
        dr = (w @ sig) / np.sqrt(w @ S @ w)
        return -dr
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, None)] * n
    res = minimize(obj, w_ew, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 3000, "ftol": 1e-14})
    return res.x / res.x.sum()

w_rp = erc_weights(Sigma)
w_mv = min_var(Sigma)
w_mdp = max_div(Sigma, sigma)

def dr(w, S, sig):
    return (w @ sig) / np.sqrt(w @ S @ w)

dr_ew, dr_rp, dr_mv, dr_mdp = dr(w_ew, Sigma, sigma), dr(w_rp, Sigma, sigma), \
    dr(w_mv, Sigma, sigma), dr(w_mdp, Sigma, sigma)
print("\n权重:")
for nm, w in [("等权", w_ew), ("风险平价", w_rp), ("最小方差", w_mv), ("最大分散化", w_mdp)]:
    print(f"  {nm}: {np.round(w*100,1)}  DR={dr(w,Sigma,sigma):.3f}")

# ============================================================
# 3) 图1：四种权重对比
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(n); wbar = 0.2
labels = ["等权", "风险平价(ERC)", "最小方差", "最大分散化"]
cols = ["#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c"]
ws = [w_ew, w_rp, w_mv, w_mdp]
for j, (lab, w, c) in enumerate(zip(labels, ws, cols)):
    ax.bar(x + (j - 1.5) * wbar, w * 100, wbar, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=11)
ax.set_ylabel("权重 (%)", fontsize=11)
ax.set_title("最大分散化组合：把低波动、低相关资产（国债/黄金）推到前台", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5, ncol=4, loc="upper center"); ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdp_weights.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：分散化比率 DR
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.2))
vals = [dr_ew, dr_rp, dr_mv, dr_mdp]
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("分散化比率 DR = (w·σ)/√(w'Σw)", fontsize=11)
ax.set_title("分散化比率：数值越大 = 组合波动相对平均波动压得越低", fontsize=12.5, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.18); ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdp_diversification_ratio.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：相关性矩阵热力图
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(NAMES, fontsize=10)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                fontsize=9, color="black" if abs(corr[i, j]) < 0.7 else "white")
ax.set_title("资产相关性矩阵：国债/黄金与权益低相关，是 MDP 的「分散化弹药」", fontsize=11.5, fontweight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdp_corr_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：样本外 4 年净值对比
# ============================================================
def nav_from_weights(weights, Rmat):
    r = Rmat @ weights
    return np.cumprod(1 + r)

def perf(nav):
    rets = np.diff(nav) / nav[:-1]
    cagr = nav[-1] ** (252.0 / len(nav)) - 1
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)
    vol = rets.std() * np.sqrt(252)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return cagr, sharpe, vol, mdd

nav_ew = nav_from_weights(w_ew, R_test)
nav_rp = nav_from_weights(w_rp, R_test)
nav_mv = nav_from_weights(w_mv, R_test)
nav_mdp = nav_from_weights(w_mdp, R_test)
pe = {f"等权 DR={dr_ew:.2f}": perf(nav_ew), "风险平价 DR={:.2f}".format(dr_rp): perf(nav_rp),
      "最小方差 DR={:.2f}".format(dr_mv): perf(nav_mv), "最大分散化 DR={:.2f}".format(dr_mdp): perf(nav_mdp)}
for k, v in pe.items():
    print(f"  {k}: CAGR={v[0]:.1%} Sharpe={v[1]:.2f} Vol={v[2]:.1%} MDD={v[3]:.1%}")

fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(nav_ew, color="#ff7f0e", lw=1.4, label=f"等权 (CAGR={pe['等权 DR={:.2f}'.format(dr_ew)][0]:.1%}, Sharpe={pe['等权 DR={:.2f}'.format(dr_ew)][1]:.2f})")
ax.plot(nav_rp, color="#1f77b4", lw=1.4, label=f"风险平价 (Sharpe={pe['风险平价 DR={:.2f}'.format(dr_rp)][1]:.2f}, MDD={pe['风险平价 DR={:.2f}'.format(dr_rp)][3]:.0%})")
ax.plot(nav_mv, color="#9467bd", lw=1.4, label=f"最小方差 (Sharpe={pe['最小方差 DR={:.2f}'.format(dr_mv)][1]:.2f})")
ax.plot(nav_mdp, color="#2ca02c", lw=1.8, label=f"最大分散化 (CAGR={pe['最大分散化 DR={:.2f}'.format(dr_mdp)][0]:.1%}, Sharpe={pe['最大分散化 DR={:.2f}'.format(dr_mdp)][1]:.2f}, MDD={pe['最大分散化 DR={:.2f}'.format(dr_mdp)][3]:.0%})")
ax.set_xlabel("交易日（样本外 4 年）", fontsize=11)
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_title("样本外净值：最大分散化在波动更低的前提下收益不落下风", fontsize=12.5, fontweight="bold")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=8.8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "mdp_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ 最大分散化组合配图生成完成：", sorted(os.listdir(D)))
