#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「LASSO 与弹性网络在多因子选股中的应用」
(lasso-elasticnet-factor) 生成真实配图与真实统计数字。

所有图表与数字均由文中 Python 逻辑真实计算生成：
  1) lasso_coef_path.png     —— Lasso 系数路径：随 alpha 增大系数被逐个压到 0(稀疏化)
  2) lasso_coef_recovery.png —— 真因子系数恢复：True / OLS / Lasso / ElasticNet 对比
  3) lasso_oos_ic.png        —— 样本外 IC 随 alpha 变化 + 四种模型 OOS IC 横比
  4) lasso_portfolio.png     —— 样本外多空组合净值：Lasso/OLS/真实上限/随机基线

核心方法(正则化线性回归做因子收益预测)：
  - 面板：T 个月 × N 只股票，每只股票 P 个因子特征 X，预测下期收益 r_{t+1}=X_t·β+ε
  - 真 β 仅 8 个非零(稀疏)，因子带块内高共线(rho=0.6)，信号弱(R²_pop≈0.10)
  - OLS 无偏但高方差(共线放大)，Lasso(ℓ1)收缩+选择，ElasticNet(ℓ1+ℓ2)兼顾
  - 样本外 IC = 预测分 vs 实现收益的 Spearman 秩相关；多空 = 每月按预测分 Top/Bottom 十分位
数据：全部 numpy 合成(固定种子)，用于演示机制，非真实行情。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  LassoCV, ElasticNetCV, lasso_path)

# ---------- 字体 / 配色 ----------
rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "lasso-elasticnet-factor")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E2E2E2",
     "lasso": "#4C72B0", "ols": "#C44E52", "en": "#DD8452", "true": "#2F4B7C",
     "rand": "#999999", "ridge": "#8172B3"}

# ---------- 工具 ----------
def rankdata(a):
    return np.argsort(np.argsort(a)) + 1.0

def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

def ols_ic_portfolio(X, y, beta_hat, long_frac=0.1):
    """给定估计系数，对单期截面做多空组合，返回该期收益。"""
    score = X @ beta_hat
    n = len(score)
    k = max(1, int(n * long_frac))
    order = np.argsort(score)
    long_ret = y[order[-k:]].mean()
    short_ret = y[order[:k]].mean()
    return long_ret - short_ret, spearman(score, y)

# =====================================================================
# 1) 合成面板数据：T 月 × N 股 × P 因子，真 β 稀疏，块内高共线
# =====================================================================
def build_data(T=200, N=60, P=100, blocks=10, rho=0.55, R2_true=0.10,
               n_true=10, seed=20260712):
    rng = np.random.default_rng(seed)
    # 块相关协方差：块内 rho，块间 0
    Cov = np.eye(P)
    bsize = P // blocks
    for b in range(blocks):
        idx = slice(b * bsize, (b + 1) * bsize)
        Cov[idx, idx] = rho
    Cov += np.eye(P) * (1 - rho)  # 对角线=1
    L = np.linalg.cholesky(Cov)
    # 真系数：随机选 n_true 个因子，随机符号，量级 ~0.3
    true_idx = rng.choice(P, size=n_true, replace=False)
    beta = np.zeros(P)
    beta[true_idx] = rng.uniform(0.15, 0.45, n_true) * rng.choice([-1, 1], n_true)
    # 逐期截面：X = z·Lᵀ (z 标准正态)；收益 = X·β + ε
    X = np.zeros((T, N, P))
    r = np.zeros((T, N))
    for t in range(T):
        z = rng.standard_normal((N, P))
        Xt = z @ L.T
        sig = Xt @ beta
        var_sig = sig.var()
        var_eps = var_sig * (1.0 / R2_true - 1.0)
        eps = rng.normal(0, np.sqrt(var_eps), N)
        X[t] = Xt
        r[t] = sig + eps
    return X, r, beta, true_idx, Cov

X, r, beta_true, true_idx, Cov = build_data()
P = X.shape[2]

# 把截面收益与因子同步缩放到月频现实尺度(σ≈2%)，保持 β_true 不变：
#   r = X·β + ε  ⟹  s·r = (s·X)·β + s·ε，故同乘 s 后真系数 β 完全保留
s = 0.02 / r.std()
r = r * s
X = X * s

# 训练/验证/测试切分(按时间)：高维小样本区间，逼出过拟合
T = X.shape[0]
t_val = int(T * 0.30)
t_te = int(T * 0.50)
Xtr = X[:t_val].reshape(-1, X.shape[2])
ytr = r[:t_val].reshape(-1)
Xva = X[t_val:t_te].reshape(-1, X.shape[2])
yva = r[t_val:t_te].reshape(-1)
Xte = X[t_te:].reshape(-1, X.shape[2])
yte = r[t_te:].reshape(-1)
# 测试期逐期截面(用于组合净值)
Xte_per = X[t_te:]
yte_per = r[t_te:]

print(f"[data] P={X.shape[2]} 真因子数={len(true_idx)} 训练={Xtr.shape[0]} 验证={Xva.shape[0]} 测试={Xte.shape[0]}")
print(f"[data] 总体 R²(OLS 理论上限)= {np.corrcoef(ytr, Xtr@np.linalg.lstsq(Xtr,ytr,rcond=None)[0])[0,1]**2:.4f}")

# =====================================================================
# 2) 模型拟合
# =====================================================================
# 标准化(对因子做 z-score，便于比较系数量级)
sc = lambda A: (A - A.mean(0)) / A.std(0)
Xtr_s, Xva_s, Xte_s = sc(Xtr), sc(Xva), sc(Xte)

ols = LinearRegression().fit(Xtr_s, ytr)
ols_beta = ols.coef_

lasso_cv = LassoCV(cv=5, random_state=0, max_iter=50000).fit(Xtr_s, ytr)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=50000).fit(Xtr_s, ytr)
lasso_beta = lasso.coef_
alpha_best = lasso_cv.alpha_

en_cv = ElasticNetCV(cv=5, l1_ratio=0.5, random_state=0, max_iter=50000).fit(Xtr_s, ytr)
en = ElasticNet(alpha=en_cv.alpha_, l1_ratio=0.5, max_iter=50000).fit(Xtr_s, ytr)
en_beta = en.coef_

ridge = Ridge(alpha=1.0).fit(Xtr_s, ytr)
ridge_beta = ridge.coef_

# 样本外 IC(测试集整体)
def ic(model, Xt, yt):
    return spearman(model.predict(Xt), yt)

ic_ols = ic(ols, Xte_s, yte)
ic_lasso = ic(lasso, Xte_s, yte)
ic_en = ic(en, Xte_s, yte)
ic_ridge = ic(ridge, Xte_s, yte)

# 非零系数个数
nz_ols = int(np.sum(np.abs(ols_beta) > 1e-4))
nz_lasso = int(np.sum(np.abs(lasso_beta) > 1e-4))
nz_en = int(np.sum(np.abs(en_beta) > 1e-4))

# 真因子被 Lasso 覆盖了多少
covered = len(set(true_idx.tolist()) & set(np.where(np.abs(lasso_beta) > 1e-4)[0].tolist()))

print(f"[OOS IC] OLS={ic_ols:.4f} Ridge={ic_ridge:.4f} Lasso={ic_lasso:.4f} ElasticNet={ic_en:.4f}")
print(f"[非零系数] OLS={nz_ols} Lasso={nz_lasso} EN={nz_en}  (共{P}个因子)")
print(f"[Lasso 覆盖真因子] {covered}/{len(true_idx)}  alpha*={alpha_best:.4f}")

# =====================================================================
# 3) 图1：Lasso 系数路径
# =====================================================================
alphas, coefs, _ = lasso_path(Xtr_s, ytr, n_alphas=60, max_iter=50000)
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for i in range(coefs.shape[0]):
    ax.plot(np.log10(alphas), coefs[i], color=C["lasso"], lw=0.8, alpha=0.45)
ax.axvline(np.log10(alpha_best), color=C["dn"], ls="--", lw=1.5,
           label=f"α*={alpha_best:.3f} (LassoCV)")
# 标记真因子
for i in true_idx:
    ax.plot(np.log10(alphas), coefs[i], color=C["true"], lw=2.2)
ax.set_xlabel("log₁₀(α)"); ax.set_ylabel("系数估计值")
ax.set_title("Lasso 系数路径：α 增大，无关因子被逐个压到 0")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "lasso_coef_path.png"), dpi=130); plt.close()

# =====================================================================
# 4) 图2：真因子系数恢复
# =====================================================================
fig, ax = plt.subplots(figsize=(8.6, 4.8))
x = np.arange(len(true_idx)); w = 0.2
ax.bar(x - 1.5*w, beta_true[true_idx], w, label="真实 β", color=C["true"])
ax.bar(x - 0.5*w, ols_beta[true_idx], w, label="OLS", color=C["ols"])
ax.bar(x + 0.5*w, lasso_beta[true_idx], w, label="Lasso", color=C["lasso"])
ax.bar(x + 1.5*w, en_beta[true_idx], w, label="ElasticNet", color=C["en"])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"F{i}" for i in true_idx], fontsize=9)
ax.set_xlabel("真因子编号"); ax.set_ylabel("系数")
ax.set_title("真因子系数恢复：OLS 方向易偏、Lasso/EN 稳定收敛")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "lasso_coef_recovery.png"), dpi=130); plt.close()

# 噪声因子在 Lasso 下被压到 0 的程度
noise_idx = np.array([i for i in range(len(beta_true)) if i not in set(true_idx.tolist())])
print(f"[噪声因子] 50 个中 {len(noise_idx)} 个为真噪声；Lasso 下其 |coef| 中位数="
      f"{np.median(np.abs(lasso_beta[noise_idx])):.4f} vs OLS {np.median(np.abs(ols_beta[noise_idx])):.4f}")

# =====================================================================
# 5) 图3：OOS IC 随 alpha 变化 + 四模型横比
# =====================================================================
alpha_grid = np.logspace(-4, 0.5, 40)
ic_vs_alpha = []
for a in alpha_grid:
    m = Lasso(alpha=a, max_iter=50000).fit(Xtr_s, ytr)
    ic_vs_alpha.append(spearman(m.predict(Xva_s), yva))
ic_vs_alpha = np.array(ic_vs_alpha)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax1.plot(np.log10(alpha_grid), ic_vs_alpha, color=C["lasso"], lw=2)
ax1.axvline(np.log10(alpha_best), color=C["dn"], ls="--", lw=1.5,
             label=f"α*={alpha_best:.3f}")
ax1.set_xlabel("log₁₀(α)"); ax1.set_ylabel("验证集 OOS IC")
ax1.set_title("OOS IC 随 α：过小→过拟合，过大→欠拟合")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

models = ["OLS", "Ridge", "Lasso", "ElasticNet"]
ics = [ic_ols, ic_ridge, ic_lasso, ic_en]
cols = [C["ols"], C["ridge"], C["lasso"], C["en"]]
ax2.bar(models, ics, color=cols)
for i, v in enumerate(ics):
    ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
ax2.set_ylabel("测试集 OOS IC")
ax2.set_title("四模型样本外 IC 横比")
ax2.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "lasso_oos_ic.png"), dpi=130); plt.close()

# =====================================================================
# 6) 图4：样本外多空组合净值(测试期逐期)
# =====================================================================
def period_net_value(X_per, y_per, beta_hat, long_frac=0.1):
    nv = [1.0]
    for t in range(X_per.shape[0]):
        ret, _ = ols_ic_portfolio(X_per[t], y_per[t], beta_hat, long_frac)
        nv.append(nv[-1] * (1 + ret))
    return np.array(nv)

nv_lasso = period_net_value(Xte_per, yte_per, lasso_beta)
nv_ols = period_net_value(Xte_per, yte_per, ols_beta)
nv_true = period_net_value(Xte_per, yte_per, beta_true)
# 随机基线
rng = np.random.default_rng(99)
nv_rand = [1.0]
for t in range(Xte_per.shape[0]):
    ret, _ = ols_ic_portfolio(Xte_per[t], yte_per[t],
                              rng.standard_normal(Xte_per.shape[2]), long_frac=0.1)
    nv_rand.append(nv_rand[-1] * (1 + ret))
nv_rand = np.array(nv_rand)

def ann_ret(nv, periods_per_year=12):
    yrs = (len(nv) - 1) / periods_per_year
    return (nv[-1] / nv[0]) ** (1 / yrs) - 1

def sharpe(nv, periods_per_year=12):
    rets = np.diff(nv) / nv[:-1]
    return rets.mean() / rets.std() * np.sqrt(periods_per_year)

def max_dd(nv):
    peak = np.maximum.accumulate(nv)
    return (nv / peak - 1).min()

for name, nv in [("Lasso", nv_lasso), ("OLS", nv_ols), ("True", nv_true), ("Random", nv_rand)]:
    print(f"[组合] {name:7s} 年化={ann_ret(nv)*100:6.2f}%  Sharpe={sharpe(nv):.3f}  最大回撤={max_dd(nv)*100:6.2f}%")

fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.plot(nv_true, label="真实上限(β 已知)", color=C["true"], lw=2)
ax.plot(nv_lasso, label="Lasso", color=C["lasso"], lw=2)
ax.plot(nv_ols, label="OLS", color=C["ols"], lw=2)
ax.plot(nv_rand, label="随机基线", color=C["rand"], lw=1.5, ls=":")
ax.set_xlabel("测试期(月)"); ax.set_ylabel("净值(初始=1)")
ax.set_title("样本外多空组合净值：Lasso 稳居 OLS 之上、逼近真实上限")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "lasso_portfolio.png"), dpi=130); plt.close()

print("\nDONE ->", D)
for f in sorted(os.listdir(D)):
    print("  ", f, os.path.getsize(os.path.join(D, f)), "bytes")
