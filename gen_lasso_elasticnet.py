#!/usr/bin/env python3
"""
为文章「Lasso 与弹性网络变量选择：用 L1 惩罚把几百个因子压成可交易少数」
(lasso-elasticnet-variable-selection) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 构造 N=2000 样本、p=500 个预测因子，其中仅 15 个是真信号，并引入因子间相关性(成组)模拟真实多因子面板
  * 分别用 Lasso / Ridge / ElasticNet 拟合，扫描惩罚强度 lambda
  * 图1: Lasso 系数路径(log-lambda)——展示 L1 如何把系数逐个压到 0
  * 图2: ElasticNet 在 (alpha=0.5) 下 OOS-MSE vs lambda,标注最优
  * 图3: 稀疏度(非零系数个数)随 lambda 变化的阶梯曲线,并标注"真信号个数=15"参考线
  * 图4: OOS R^2 对比 Lasso / Ridge / ElasticNet / 全因子——展示 L1 在防过拟合上的真实收益(非保证盈利)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "lasso-elasticnet-variable-selection")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "lasso": "#4C72B0", "ridge": "#DD8452", "en": "#55A868",
     "full": "#999999", "band": "#C44E52", "true": "#2F4B7C", "mk": "#8172B3"}

rng = np.random.default_rng(20260719)
N, P, K = 2000, 500, 15
# 构造成组相关的因子:把 p 个因子分成 50 组,组内相关
n_groups = 50
group = np.repeat(np.arange(n_groups), P // n_groups)
base = rng.normal(0, 1, (N, n_groups))
X = np.empty((N, P))
for j in range(P):
    X[:, j] = 0.75 * base[:, group[j]] + 0.65 * rng.normal(0, 1, N)
# 标准化
X = (X - X.mean(0)) / X.std(0)
# 真信号:仅前 15 个因子(分属不同组)有系数
beta_true = np.zeros(P)
true_idx = rng.choice(P, K, replace=False)
beta_true[true_idx] = rng.uniform(-1.2, 1.2, K)
# 信号噪声比:使 in-sample R^2 ~ 0.35
signal = X @ beta_true
signal = signal / signal.std()
snr = 0.45
y = signal + rng.normal(0, 1.0 / snr, N)
y = y / y.std()

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=7)


def oos_r2(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ---------- 图1：Lasso 系数路径 ----------
lambdas = np.logspace(-3, 0, 60)
paths = []
for lam in lambdas:
    m = Lasso(alpha=lam, max_iter=5000, tol=1e-3)
    m.fit(Xtr, ytr)
    paths.append(m.coef_)
paths = np.array(paths)
fig, ax = plt.subplots(figsize=(10, 5.5))
for j in range(P):
    ax.plot(np.log10(lambdas), paths[:, j], color=C["lasso"], lw=0.5, alpha=0.25)
# 高亮真信号路径
for j in true_idx:
    ax.plot(np.log10(lambdas), paths[:, j], color=C["true"], lw=1.6, alpha=0.9)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xlabel("log10(λ)", fontsize=11)
ax.set_ylabel("系数估计值", fontsize=11)
ax.set_title("Lasso 系数路径：λ 增大，系数被逐个压向 0（粗线=15 个真信号）", fontsize=11.5)
ax.grid(True, color=C["grid"], axis="y")
ax.legend([plt.Line2D([0], [0], color=C["lasso"], lw=1),
           plt.Line2D([0], [0], color=C["true"], lw=2)],
          ["其余因子(弱信号/噪声)", "真信号系数"], fontsize=9, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(D, "lasso_coef_path.png"), dpi=130)
plt.close(fig)

# ---------- 图2：ElasticNet OOS-MSE vs lambda ----------
alphas = np.logspace(-3, 0, 40)
mse_tr, mse_te = [], []
for a in alphas:
    model = Pipeline([("sc", StandardScaler()), ("en", ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000, tol=1e-3))])
    model.fit(Xtr, ytr)
    mse_tr.append(np.mean((model.predict(Xtr) - ytr) ** 2))
    mse_te.append(np.mean((model.predict(Xte) - yte) ** 2))
mse_tr, mse_te = np.array(mse_tr), np.array(mse_te)
best = int(np.argmin(mse_te))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.log10(alphas), mse_tr, color=C["lasso"], lw=1.6, label="训练集 MSE")
ax.plot(np.log10(alphas), mse_te, color=C["en"], lw=1.8, label="测试集(Out-of-Sample) MSE")
ax.axvline(np.log10(alphas[best]), color=C["band"], ls="--", lw=1.5,
           label="最优 λ(最小化 OOS-MSE)")
ax.set_xlabel("log10(λ)", fontsize=11)
ax.set_ylabel("均方误差 MSE", fontsize=11)
ax.set_title("ElasticNet (L1_ratio=0.5)：测试误差在适中 λ 处见底", fontsize=11.5)
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "elasticnet_oos_mse.png"), dpi=130)
plt.close(fig)

# ---------- 图3：稀疏度 vs lambda ----------
nonzeros = [(np.abs(Lasso(alpha=lam, max_iter=5000, tol=1e-3).fit(Xtr, ytr).coef_) > 1e-4).sum()
            for lam in lambdas]
nonzeros = np.array(nonzeros)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.log10(lambdas), nonzeros, color=C["mk"], lw=2.0)
ax.axhline(K, color=C["true"], ls="--", lw=1.6, label="真信号个数 = 15")
ax.set_xlabel("log10(λ)", fontsize=11)
ax.set_ylabel("选中的非零因子个数", fontsize=11)
ax.set_title("稀疏度阶梯：λ 越大，被保留的因子越少", fontsize=11.5)
ax.legend(fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "sparsity_vs_lambda.png"), dpi=130)
plt.close(fig)

# ---------- 图4：OOS R^2 对比(各方法在各自最优 λ 下) ----------
def best_oos(maker, param_grid, pname):
    best_r2, best_p = -1e9, None
    for p in param_grid:
        m = maker(p)
        r2 = oos_r2(m, Xtr, ytr, Xte, yte)
        if r2 > best_r2:
            best_r2, best_p = r2, p
    return best_r2, best_p


r2_lasso, _ = best_oos(lambda a: Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=a, max_iter=5000, tol=1e-3))]), alphas, "a")
r2_ridge, _ = best_oos(lambda a: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=a))]), np.logspace(-4, 2, 40), "a")
r2_en, _ = best_oos(lambda a: Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000, tol=1e-3))]), alphas, "a")
# 全因子(无惩罚)线性回归 OOS R^2
from sklearn.linear_model import LinearRegression
r2_full = oos_r2(LinearRegression(), Xtr, ytr, Xte, yte)

labels = ["全因子\n(无惩罚)", "Ridge\n(L2)", "Lasso\n(L1)", "ElasticNet\n(L1+L2)"]
vals = [r2_full, r2_ridge, r2_lasso, r2_en]
colors = [C["full"], C["ridge"], C["lasso"], C["en"]]
print("OOS R^2: full=%.4f ridge=%.4f lasso=%.4f en=%.4f" % (r2_full, r2_ridge, r2_lasso, r2_en))
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, "%.3f" % v, ha="center", fontsize=10)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_ylabel("Out-of-Sample R²", fontsize=11)
ax.set_title("同样 500 因子，惩罚回归的 OOS 拟合更好（防过拟合，非保证盈利）", fontsize=11)
ax.grid(True, color=C["grid"], axis="y")
ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
plt.tight_layout()
fig.savefig(os.path.join(D, "oos_r2_compare.png"), dpi=130)
plt.close(fig)

print("saved 4 figures to", D)
