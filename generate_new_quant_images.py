#!/usr/bin/env python3
"""
为本次两篇「非重复」量化文章生成真实配图（matplotlib 渲染，非占位图）：
  1. shap-interpretable-ml-quant   (SHAP 与可解释机器学习：让量化因子不再黑箱)
  2. copula-tail-dependence-risk   (Copula 模型在组合风险管理中的应用：捕捉尾部相依)

说明：环境无 shap 库，故用「蒙特卡洛 Shapley 值估计(Shapley Sampling)」从原理实现，
     结果与 shap.TreeExplainer 在期望上一致，且零外部依赖。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm, t as student_t

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
np.random.seed(20260711)


# ============================================================
# 文章 1：SHAP 与可解释机器学习
# ============================================================
d1 = os.path.join(BASE, "shap-interpretable-ml-quant")
os.makedirs(d1, exist_ok=True)

# ---------- 合成因子数据 + 训练树模型 ----------
N = 2000
pe = np.random.uniform(5, 60, N)            # 市盈率
pb = np.random.uniform(0.5, 12, N)         # 市净率
roe = np.random.uniform(-5, 30, N)         # ROE
mom20 = np.random.normal(0, 0.15, N)       # 20日动量
vol20 = np.random.uniform(0.01, 0.08, N)   # 20日波动率
turn = np.random.uniform(0.5, 8, N)        # 换手率
size = np.random.uniform(10, 5000, N)      # 市值(亿)
lev = np.random.uniform(0.1, 0.85, N)      # 杠杆率
X = np.column_stack([pe, pb, roe, mom20, vol20, turn, size, lev])
feat_names = ["PE", "PB", "ROE", "动量20", "波动率20", "换手率", "市值", "杠杆率"]

# 目标：下月收益（含信号 + 非线性 + 噪声）
signal = (
    0.10 * roe
    + 0.35 * mom20
    - 0.40 * (vol20 - 0.04)
    - 0.015 * (pe - 30)
    - 0.02 * (pb - 4)
    + 0.05 * np.tanh(turn / 4)          # 非线性：换手率适中较好
    - 0.08 * (lev - 0.4) ** 2
)
y = signal + np.random.normal(0, 0.20, N)

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08,
                                  subsample=0.9, random_state=1)
model.fit(X, y)

# ---------- 蒙特卡洛 Shapley 值估计 ----------
background = X.mean(axis=0)            # 背景(期望值)作为缺失特征的填充
M = X.shape[1]
N_SHAP = 260
K = 120                               # 每样本排列数
idx = np.random.choice(N, N_SHAP, replace=False)
X_shap = X[idx]

all_with, all_without, idx_i, idx_j = [], [], [], []
for i in idx:
    inst = X[i].copy()
    for _ in range(K):
        perm = np.random.permutation(M)
        x_with = background.copy()
        x_without = background.copy()
        for j in perm:
            x_with[j] = inst[j]
            all_with.append(x_with.copy())
            all_without.append(x_without.copy())
            idx_i.append(i)
            idx_j.append(j)
            x_without[j] = inst[j]

all_with = np.array(all_with)
all_without = np.array(all_without)
pred_with = model.predict(all_with)
pred_without = model.predict(all_without)
contrib = pred_with - pred_without

row_of = {glob: r for r, glob in enumerate(idx)}
phi = np.zeros((N_SHAP, M))
for k, (glob, j) in enumerate(zip(idx_i, idx_j)):
    phi[row_of[glob], j] += contrib[k]
phi /= K

shap_abs = np.abs(phi).mean(axis=0)        # 全局重要性 = mean|SHAP|
order = np.argsort(shap_abs)[::-1]
print("SHAP 全局重要性(均值|SHAP|):", dict(zip([feat_names[o] for o in order], shap_abs[order].round(4))))


# ---- 图1：全局重要性（SHAP 均值|SHAP|）条形图 ----
cols = [feat_names[o] for o in order]
vals = shap_abs[order]
fig, ax = plt.subplots(figsize=(11, 5.6))
bars = ax.barh(cols[::-1], vals[::-1], color="#1f77b4")
for b, v in zip(bars, vals[::-1]):
    ax.text(v + vals.max() * 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}",
            va="center", fontsize=9.5, fontweight="bold")
ax.set_xlabel("平均 |SHAP| 值（对收益预测的贡献强度）", fontsize=11)
ax.set_title("SHAP 全局特征重要性：哪些因子真正驱动了收益预测", fontsize=13.5, fontweight="bold")
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d1, "shap_importance.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---- 图2：SHAP 摘要蜂群图(beeswarm) ----
S = 130                                   # 抽样展示实例数
sidx = np.random.choice(N_SHAP, S, replace=False)
fig, ax = plt.subplots(figsize=(11, 6.2))
for r in sidx:
    inst = X_shap[r]
    for j in range(M):
        v = phi[r, j]
        ax.scatter(inst[j], v, s=14, alpha=0.45,
                   color=plt.cm.coolwarm((inst[j] - X[:, j].min()) /
                                         (X[:, j].max() - X[:, j].min() + 1e-9)))
ax.set_xticks(range(M))
ax.set_xticklabels(feat_names, fontsize=10)
ax.set_ylabel("SHAP 值（对该样本收益预测的贡献）", fontsize=11)
ax.axhline(0, color="gray", lw=1)
ax.set_title("SHAP 摘要图：每个因子的取值(颜色)如何推动预测向上/向下", fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(d1, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---- 图3：SHAP 依赖图（ROE，全局最重要的因子）----
j = feat_names.index("ROE")
fig, ax = plt.subplots(figsize=(11, 5.6))
xs = X[idx, j]
ys = phi[:, j]
cm = plt.cm.viridis((xs - xs.min()) / (xs.max() - xs.min() + 1e-9))
ax.scatter(xs, ys, c=cm, s=22, alpha=0.55)
# 叠加趋势线
z = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min(), xs.max(), 50)
ax.plot(xx, np.polyval(z, xx), color="#d62728", lw=2.2, label=f"趋势(斜率≈{z[0]:.3f})")
ax.set_xlabel("ROE（%）", fontsize=11)
ax.set_ylabel("SHAP 值（ROE 对收益预测的贡献）", fontsize=11)
ax.set_title("SHAP 依赖图：ROE 越高，对收益预测的正向贡献越大", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d1, "shap_dependence.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---- 图4：单样本瀑布图(waterfall) ----
r = 0
inst = X[idx[r]]
base = model.predict(background.reshape(1, -1))[0]
contrib_sorted = sorted(zip(feat_names, phi[r], inst), key=lambda t: abs(t[1]), reverse=True)[:8]
fig, ax = plt.subplots(figsize=(11, 5.8))
cum = base
ax.bar(0, base, color="#999999", width=0.6)
ax.text(0, base + 0.02, f"基准 {base:.2f}", ha="center", fontsize=9)
labels = ["基准"]
prev = base
for k, (fn, sv, raw) in enumerate(contrib_sorted, start=1):
    color = "#2ca02c" if sv >= 0 else "#d62728"
    ax.bar(k, abs(sv), bottom=(prev if sv >= 0 else prev + sv), color=color, width=0.6)
    cum += sv
    ax.text(k, cum + (0.02 if sv >= 0 else -0.05), f"{sv:+.2f}", ha="center", fontsize=8.5)
    labels.append(fn)
ax.bar(len(contrib_sorted) + 1, cum, color="#1f77b4", width=0.6)
ax.text(len(contrib_sorted) + 1, cum + 0.02, f"预测 {cum:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(contrib_sorted) + 2))
ax.set_xticklabels(["基准(E[f])"] + [c[0] for c in contrib_sorted] + ["模型输出"], fontsize=8.5, rotation=30)
ax.set_ylabel("累积贡献 → 收益预测", fontsize=11)
ax.set_title("SHAP 瀑布图：单个样本的预测如何由各因子逐项叠加而成", fontsize=12.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(d1, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 文章 2：Copula 模型与尾部相依
# ============================================================
d2 = os.path.join(BASE, "copula-tail-dependence-risk")
os.makedirs(d2, exist_ok=True)

# 资产相关性矩阵（6 个资产）
assets = ["股票", "商品", "REITs", "高收益债", "新兴市场", "黄金"]
rho = np.array([
    [1.00, 0.55, 0.60, 0.50, 0.58, 0.10],
    [0.55, 1.00, 0.45, 0.35, 0.50, 0.25],
    [0.60, 0.45, 1.00, 0.40, 0.45, 0.12],
    [0.50, 0.35, 0.40, 1.00, 0.42, 0.08],
    [0.58, 0.50, 0.45, 0.42, 1.00, 0.15],
    [0.10, 0.25, 0.12, 0.08, 0.15, 1.00],
])
L = np.linalg.cholesky(rho)
nu = 4   # t-copula 自由度（重尾）

# ---------- 图1：高斯 copula vs t-copula 散点（取资产0-1，rho=0.55）----------
nsim = 4000
z = np.dot(L[:2, :2], np.random.randn(2, nsim))
u_gauss = norm.cdf(z)                      # 高斯 copula
w = np.random.chisquare(nu, nsim) / nu
z_t = np.dot(L[:2, :2], np.random.randn(2, nsim)) / np.sqrt(w)
u_t = student_t.cdf(z_t, nu)               # t-copula
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
axes[0].scatter(u_gauss[0], u_gauss[1], s=8, alpha=0.35, color="#1f77b4")
axes[0].set_title("高斯 Copula：四角无异常聚集", fontsize=12, fontweight="bold")
axes[0].set_xlabel("U₁（股票分位）"); axes[0].set_ylabel("U₂（商品分位）")
axes[1].scatter(u_t[0], u_t[1], s=8, alpha=0.35, color="#d62728")
axes[1].set_title(f"t-Copula(ν={nu})：危机时四角严重聚集", fontsize=12, fontweight="bold")
axes[1].set_xlabel("U₁（股票分位）"); axes[1].set_ylabel("U₂（商品分位）")
fig.suptitle("尾部相依可视化：极端行情里，资产会「一起跳水」", fontsize=13.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(d2, "copula_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------- 图2：尾部相依系数热力图（高斯 vs t）----------
def lower_tail_dep(u1, u2, q=0.05):
    return np.mean((u2 < q) & (u1 < q)) / max(np.mean(u1 < q), 1e-9)

# 用更大样本估计两两尾部相依
big = 200000
# 高斯 copula 大样本
z = np.dot(L, np.random.randn(6, big))
U_g = norm.cdf(z)
# t-copula 大样本
w = np.random.chisquare(nu, big) / nu
z_t = np.dot(L, np.random.randn(6, big)) / np.sqrt(w)
U_t = student_t.cdf(z_t, nu)

n = len(assets)
Tg = np.zeros((n, n)); Tt = np.zeros((n, n))
for a in range(n):
    for b in range(n):
        Tg[a, b] = lower_tail_dep(U_g[a], U_g[b]) if a != b else 1.0
        Tt[a, b] = lower_tail_dep(U_t[a], U_t[b]) if a != b else 1.0

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
im0 = axes[0].imshow(Tg, cmap="YlOrRd", vmin=0, vmax=0.6)
axes[0].set_title("高斯 Copula 下尾相依 ≈ 0", fontsize=12, fontweight="bold")
axes[0].set_xticks(range(n)); axes[0].set_yticks(range(n))
axes[0].set_xticklabels(assets, rotation=40, fontsize=8); axes[0].set_yticklabels(assets, fontsize=8)
for a in range(n):
    for b in range(n):
        axes[0].text(b, a, f"{Tg[a,b]:.2f}", ha="center", va="center", fontsize=7.5)
im1 = axes[1].imshow(Tt, cmap="YlOrRd", vmin=0, vmax=0.6)
axes[1].set_title(f"t-Copula(ν={nu}) 下尾相依显著>0", fontsize=12, fontweight="bold")
axes[1].set_xticks(range(n)); axes[1].set_yticks(range(n))
axes[1].set_xticklabels(assets, rotation=40, fontsize=8); axes[1].set_yticklabels(assets, fontsize=8)
for a in range(n):
    for b in range(n):
        axes[1].text(b, a, f"{Tt[a,b]:.2f}", ha="center", va="center", fontsize=7.5)
fig.suptitle("尾部相依系数热力图：危机相关 ≠ 平时相关", fontsize=13.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(d2, "tail_dependence_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------- 图3：组合 VaR 对比（高斯 copula vs t-copula）----------
def simulate_portfolio(copula="gauss", nsim=100000):
    if copula == "gauss":
        z = np.dot(L, np.random.randn(6, nsim))
        U = norm.cdf(z)
    else:
        w = np.random.chisquare(nu, nsim) / nu
        z = np.dot(L, np.random.randn(6, nsim)) / np.sqrt(w)
        U = student_t.cdf(z, nu)
    # 边际：各资产收益用正态边际（轻尾），以隔离「copula=相依结构」的纯效应
    R = norm.ppf(U) * 0.012   # 缩放使日波动约 1.2%
    port = R.mean(axis=0)                     # 等权组合
    return port

port_g = simulate_portfolio("gauss")
port_t = simulate_portfolio("t")

var_g_95 = np.percentile(port_g, 5)
var_t_95 = np.percentile(port_t, 5)
var_g_99 = np.percentile(port_g, 1)
var_t_99 = np.percentile(port_t, 1)
print(f"VaR95 高斯={var_g_95:.4f} t={var_t_95:.4f} | VaR99 高斯={var_g_99:.4f} t={var_t_99:.4f}")

fig, ax = plt.subplots(figsize=(11, 5.6))
bins = np.linspace(-0.12, 0.06, 80)
ax.hist(port_g, bins=bins, alpha=0.5, density=True, color="#1f77b4", label="高斯 Copula 组合")
ax.hist(port_t, bins=bins, alpha=0.5, density=True, color="#d62728", label="t-Copula 组合")
ax.axvline(var_g_95, color="#1f77b4", ls="--", lw=1.6)
ax.axvline(var_t_95, color="#d62728", ls="--", lw=1.6)
ax.axvline(var_t_99, color="#d62728", ls=":", lw=1.6)
ax.annotate(f"t-Copula VaR95={var_t_95*100:.1f}%", xy=(var_t_95, 8),
            xytext=(var_t_95-0.02, 22), fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.set_xlabel("等权组合日收益", fontsize=11)
ax.set_ylabel("密度", fontsize=11)
ax.set_title("组合 VaR 对比：t-Copula 在极端尾部给出更诚实（更深）的风险", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(d2, "copula_var.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------- 图4：经验 copula 等高线（t-copula 示例）----------
z = np.dot(L[:2, :2], np.random.randn(2, 6000))
U = student_t.cdf(z, nu)
fig, ax = plt.subplots(figsize=(11, 5.6))
# 用 2D 直方图近似经验 copula 密度
h, xe, ye = np.histogram2d(U[0], U[1], bins=24, range=[[0, 1], [0, 1]])
extent = [xe[0], xe[-1], ye[0], ye[-1]]
im = ax.imshow(h.T, origin="lower", extent=extent, cmap="Blues")
ax.contour(xe[:-1], ye[:-1], h.T, levels=6, colors="red", linewidths=1.0, alpha=0.8)
ax.set_xlabel("U₁（资产1 分位）", fontsize=11)
ax.set_ylabel("U₂（资产2 分位）", fontsize=11)
ax.set_title("经验 Copula 密度：四角堆积 = 尾部相依", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="联合密度")
plt.tight_layout()
plt.savefig(os.path.join(d2, "empirical_copula.png"), dpi=150, bbox_inches="tight")
plt.close()


print("✅ 图像生成完成")
print("   shap-interpretable-ml-quant:", sorted(os.listdir(d1)))
print("   copula-tail-dependence-risk:", sorted(os.listdir(d2)))
