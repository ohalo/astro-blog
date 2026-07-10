#!/usr/bin/env python3
"""
为文章「贝叶斯资产配置：用共轭先验把观点变成权重」(bayesian-asset-allocation)
生成真实配图。所有图表均由文中 Python 代码真实计算生成
（正态-逆威沙特 NIW 共轭先验 → 后验 → 后验均值-方差权重 + 可信带）。

图表：
  1. bayes_shrinkage.png        样本均值 vs 后验均值（共轭先验的收缩效应）
  2. bayes_predictive.png       某资产收益的后验预测分布（先验→数据→后验更新）
  3. bayes_frontier.png         经典样本前沿 vs 贝叶斯后验前沿（含可信带）
  4. bayes_weights.png          样本权重 vs 贝叶斯权重（随先验强度 τ 变化）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import wishart
from scipy.special import gammaln

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "bayesian-asset-allocation")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============================================================
# 0) 模拟 5 个资产的多变量收益（真实 μ, Σ 已知）
# ============================================================
p = 5
true_mu = np.array([0.10, 0.07, 0.12, 0.05, 0.09])          # 年化期望
true_cov = np.array([
    [0.040, 0.018, 0.010, 0.006, 0.014],
    [0.018, 0.030, 0.008, 0.004, 0.010],
    [0.010, 0.008, 0.055, 0.012, 0.016],
    [0.006, 0.004, 0.012, 0.020, 0.006],
    [0.014, 0.010, 0.016, 0.006, 0.035],
])
labels = ["股票A", "股票B", "股票C", "债券D", "商品E"]
T = 120                                                      # 历史观测数（样本偏小，凸显贝叶斯价值）
R = rng.multivariate_normal(true_mu / 12.0, true_cov / 12.0, size=T)  # 月度收益
# 样本统计量
xbar = R.mean(axis=0)
S = np.cov(R, rowvar=False)
print("样本均值(年化):", np.round(xbar * 12, 3))
print("真实均值(年化):", np.round(true_mu, 3))

# ============================================================
# 1) 正态-逆威沙特 NIW 共轭先验 → 后验
#    先验: μ|Σ ~ N(μ0, Σ/κ0),  Σ ~ IW(Λ0, ν0)
#    后验: μ|Σ ~ N(μ_n, Σ/κ_n), Σ ~ IW(Λ_n, ν_n)
#      κ_n = κ0 + T
#      μ_n = (κ0 μ0 + T xbar) / (κ0 + T)          ← 样本均值向先验收缩
#      ν_n = ν0 + T
#      Λ_n = Λ0 + S_term + (κ0 T)/(κ0+T) (xbar-μ0)(xbar-μ0)'
# ============================================================
mu0 = np.full(p, 0.07 / 12.0)    # 先验观点：所有资产年化期望 7%（换算成月度）
kappa0 = 12.0                   # 先验等效样本量（=1年的月度观测）
nu0 = p + 2.0                   # 先验自由度
Lambda0 = np.diag([0.04, 0.03, 0.05, 0.02, 0.035]) * (nu0)  # 先验尺度

def niw_posterior(xbar, S, T, mu0, kappa0, nu0, Lambda0):
    kappa_n = kappa0 + T
    mu_n = (kappa0 * mu0 + T * xbar) / kappa_n
    nu_n = nu0 + T
    diff = (xbar - mu0).reshape(-1, 1)
    Lambda_n = Lambda0 + (T - 1) * S + (kappa0 * T / kappa_n) * (diff @ diff.T)
    return mu_n, kappa_n, nu_n, Lambda_n

mu_n, kappa_n, nu_n, Lambda_n = niw_posterior(xbar, S, T, mu0, kappa0, nu0, Lambda0)
print("后验均值(年化):", np.round(mu_n * 12, 3))

# ============================================================
# 图 1：收缩效应（样本均值 → 后验均值）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(p)
w = 0.38
ax.bar(x - w / 2, xbar * 12 * 100, w, color="#9ecae1", label="样本均值（数据 alone）")
ax.bar(x + w / 2, mu_n * 12 * 100, w, color="#d62728", label="后验均值（先验+数据）")
ax.plot(x, true_mu * 100, "kD", ms=7, label="真实均值（生成参数）")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("年化期望收益 (%)", fontsize=11)
ax.set_title("共轭先验的收缩：后验均值被「拉」向先验观点 7%", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.2, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "bayes_shrinkage.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 2：某资产（股票A）收益的后验预测分布
#    后验预测: y_new | data ~ t(μ_n, (1+1/κ_n) Λ_n/(ν_n-p+1), ν_n-p+1)
# ============================================================
def posterior_predictive_density(mu_n, kappa_n, nu_n, Lambda_n, ygrid):
    # 多元 t 的边缘：用一维 t 近似（对第 1 个资产）
    i = 0
    scale = np.sqrt((1 + 1 / kappa_n) * Lambda_n[i, i] / (nu_n - p + 1))
    df = nu_n - p + 1
    z = (ygrid - mu_n[i]) / scale
    from scipy.special import gammaln
    logpdf = (gammaln((df + 1) / 2) - gammaln(df / 2)
              - 0.5 * np.log(df * np.pi * scale ** 2)
              - (df + 1) / 2 * np.log1p(z ** 2 / df))
    return np.exp(logpdf)

yg = np.linspace(-0.10, 0.14, 400)
# 先验预测密度（用先验参数）
prior_scale = np.sqrt((1 + 1 / kappa0) * Lambda0[0, 0] / (nu0 - p + 1))
df0 = nu0 - p + 1
zp = (yg - mu0[0]) / prior_scale
from scipy.special import gammaln
logpdf_prior = (gammaln((df0 + 1) / 2) - gammaln(df0 / 2)
                - 0.5 * np.log(df0 * np.pi * prior_scale ** 2)
                - (df0 + 1) / 2 * np.log1p(zp ** 2 / df0))
post_dens = posterior_predictive_density(mu_n, kappa_n, nu_n, Lambda_n, yg)
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(yg * 100, np.exp(logpdf_prior), color="#9ecae1", lw=2.2, label="先验预测分布")
ax.plot(yg * 100, post_dens, color="#d62728", lw=2.2, label="后验预测分布（观测 120 期后）")
# 叠加真实资产A月收益直方图
ax.hist(R[:, 0] * 100, bins=20, density=True, color="#c7c7c7", alpha=0.45, label="真实样本月收益")
ax.axvline(true_mu[0] / 12 * 100, color="black", lw=1.4, ls="--", label="真实均值")
ax.set_xlabel("月度收益 (%)", fontsize=11)
ax.set_ylabel("密度", fontsize=11)
ax.set_title("贝叶斯更新：后验预测分布比先验更紧、更贴近数据", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.0)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "bayes_predictive.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 3：经典样本前沿 vs 贝叶斯后验前沿（含可信带）
#    经典: 用 (xbar, S) 直接算均值-方差前沿
#    贝叶斯: 用后验均值 μ_n 与后验期望协方差 E[Σ|data]
#      E[Σ|data] = Λ_n / (ν_n - p - 1)
# ============================================================
def eff_frontier(mu, cov, npoints=60):
    inv = np.linalg.inv(cov)
    ones = np.ones(p)
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    D = A * C - B ** 2
    rets = np.linspace(mu.min() * 1.2, mu.max() * 1.2, npoints)
    vols = np.sqrt((A * rets ** 2 - 2 * B * rets + C) / D)
    return rets, vols

# 后验期望协方差
E_Sigma_post = Lambda_n / (nu_n - p - 1)
rets_s, vols_s = eff_frontier(xbar * 12, S * 12)        # 经典（年化）
rets_b, vols_b = eff_frontier(mu_n * 12, E_Sigma_post * 12)  # 贝叶斯

# 后验协方差的可信带：用 NIW 抽样若干 Σ，画前沿包络
Sigma_samples = []
for _ in range(400):
    # 从 IW(Λ_n, ν_n) 抽样 Σ：先抽 W~Wishart(Λ_n^{-1}, ν_n)，再取逆
    W = wishart(df=int(round(nu_n)), scale=np.linalg.inv(Lambda_n)).rvs(random_state=rng)
    Sigma_samples.append(np.linalg.inv(W))
Sigma_samples = np.array(Sigma_samples)
upper = []; lower = []
for rt in rets_b:
    vs = []
    for Sig in Sigma_samples:
        try:
            _, vv = eff_frontier(mu_n * 12, Sig * 12)
            # 找到该收益对应的最小波动
            vs.append(np.interp(rt, rets_b, vv))
        except Exception:
            pass
    upper.append(np.percentile(vs, 90)); lower.append(np.percentile(vs, 10))

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(vols_s * 100, rets_s * 100, color="#1f77b4", lw=2.2, label="经典样本前沿（用 xbar, S）")
ax.plot(vols_b * 100, rets_b * 100, color="#d62728", lw=2.4, label="贝叶斯后验前沿（用 μ_n, E[Σ|data]）")
ax.fill_between(np.array(upper) * 100, np.array(lower) * 100, rets_b * 100,
                color="#d62728", alpha=0.10, label="后验前沿 90% 可信带")
ax.set_xlabel("年化波动 (%)", fontsize=11)
ax.set_ylabel("年化收益 (%)", fontsize=11)
ax.set_title("贝叶斯前沿：收缩后更靠内，可信带量化了估计不确定性", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "bayes_frontier.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 4：样本权重 vs 贝叶斯权重（随先验强度 κ0 变化）
#    用全局最小方差组合（对预期收益不敏感，便于展示协方差收缩）
# ============================================================
def min_var_weights(mu, cov):
    inv = np.linalg.inv(cov)
    ones = np.ones(p)
    w = inv @ ones
    return w / w.sum()

w_sample = min_var_weights(xbar * 12, S * 12)
ks = [1, 3, 6, 12, 30, 60, 120]
def weights_for_kappa(k0):
    mn, kn, nn, Ln = niw_posterior(xbar, S, T, mu0, k0, nu0, Lambda0)
    return min_var_weights(mn * 12, Ln / (nn - p - 1) * 12)

Wk = np.array([weights_for_kappa(k0) for k0 in ks])
fig, ax = plt.subplots(figsize=(11, 5.6))
for j in range(p):
    ax.plot(ks, Wk[:, j] * 100, "o-", lw=1.8, label=labels[j])
ax.axhline(0, color="gray", lw=0.8)
ax.plot(ks, np.tile(w_sample * 100, (len(ks), 1)), color="black", lw=1.0, ls=":", alpha=0.5,
        label="样本最小方差权重（κ0→∞ 基准）")
ax.set_xscale("log")
ax.set_xlabel("先验强度 κ0（等效样本量，越大越信任数据）", fontsize=10.5)
ax.set_ylabel("权重 (%)", fontsize=11)
ax.set_title("先验强度如何把权重从「数据独断」平滑拉向稳健区", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=8.5, ncol=2)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(D, "bayes_weights.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ 贝叶斯资产配置配图生成完成：{sorted(os.listdir(D))}")
