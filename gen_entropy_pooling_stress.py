#!/usr/bin/env python3
"""
为文章「熵池化压力测试：把先验观点与历史分布融合成压力情景」
(entropy-pooling-stress) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * 先验：用学生-t 生成 K 个 3 资产（股/债/金）月度收益情景 x_k，先验 p_k = 1/K。
  * 压力观点：把历史压力写成概率约束 E^q[1{X∈A}] = q_target，
        即 Σ_k q_k·1{x_k∈A} = q_target（A = 某灾难区域）。
  * 熵池化：min_q Σ q_k ln(q_k/p_k) s.t. Σ q_k = 1, Σ q_k·f_i(x_k) = μ_i
        对偶 g(λ)=λ·μ − ln Σ p_k exp(−λ·f(x_k))；牛顿+回溯线搜索求 λ；
        后验 q_k ∝ p_k·exp(−λ·f(x_k))。全程 log 空间防溢出。
  * 应用：用后验情景重算组合（60/30/10 股债金）收益分布，读出 95% VaR / CVaR，
        与先验对比，量化一个观点把尾部风险抬高了多少。

注意：本模拟用于演示机制，数字量级参考真实多资产，但非真实数据。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "entropy-pooling-stress")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "post": "#C44E52",
      "prior": "#999999", "grid": "#DDDDDD", "mkt": "#CCB974", "gold": "#8172B3"}

rng = np.random.default_rng(20260718)
K = 4000
# 股/债/金 月度收益先验：学生-t（肥尾）多元，相关结构近似真实
mu0 = np.array([0.0070, 0.0030, 0.0045])
sd = np.array([0.0400, 0.0150, 0.0350])
corr = np.array([[1.0, -0.20, 0.10],
                 [-0.20, 1.0, 0.05],
                 [0.10, 0.05, 1.0]])
L = np.linalg.cholesky(corr)
df = 5.0
z = rng.standard_t(df, size=(K, 3))            # 标准化 t
x = mu0 + (z @ L.T) * sd                     # 3 资产月度收益情景 (K,3)
p = np.full(K, 1.0 / K)                       # 等权先验

W = np.array([0.60, 0.30, 0.10])            # 股 60 / 债 30 / 金 10
rp = x @ W                                     # 组合月度收益 (K,)


from scipy.optimize import brentq


def entropy_pooling(f_mat, targets, p, max_iter=200):
    """min KL(q||p) s.t. E_q[f] = targets。
    f_mat: (K, m) 约束函数；targets: (m,) 目标期望。
    单约束用 brentq（稳健）；多约束用阻尼牛顿。返回 q, lam。
    """
    def q_of(lam):
        lq = np.log(p) - f_mat @ lam
        m_ = lq.max()
        w = np.exp(lq - m_)
        return w / w.sum()

    def grad(lam):
        q = q_of(lam)
        return targets - f_mat.T @ q

    if f_mat.shape[1] == 1:
        g0 = grad(np.array([0.0]))[0]
        lo, hi = -80.0, 80.0
        glo, ghi = grad(np.array([lo]))[0], grad(np.array([hi]))[0]
        if g0 * glo <= 0:
            lam = np.array([brentq(lambda v: grad(np.array([v]))[0], lo, 0.0)])
        elif g0 * ghi <= 0:
            lam = np.array([brentq(lambda v: grad(np.array([v]))[0], 0.0, hi)])
        else:
            lam = np.array([0.0])
        return q_of(lam), lam

    lam = np.zeros(f_mat.shape[1])
    for _ in range(max_iter):
        q = q_of(lam)
        g = targets - f_mat.T @ q
        if np.max(np.abs(g)) < 1e-9:
            break
        Ef = f_mat.T @ q
        Ef2 = (f_mat ** 2).T @ q
        H = -(Ef2 - Ef ** 2)
        H = np.where(np.abs(H) < 1e-12, -1e-12, H)
        step = g / H
        lq0 = np.log(p) - f_mat @ lam
        m0 = lq0.max()
        obj = lam @ targets - (np.log(np.exp(lq0 - m0).sum()) + m0)
        t = 1.0
        for _ in range(40):
            lt = lam + t * step
            lq2 = np.log(p) - f_mat @ lt
            m2 = lq2.max()
            Z2 = np.exp(lq2 - m2).sum()
            obj2 = lt @ targets - (np.log(Z2) + m2)
            if obj2 <= obj - 1e-4 * t * (g @ step):
                break
            t *= 0.5
        lam = lam + t * step
    return q_of(lam), lam


def w_quantile(vals, weights, alpha):
    """加权分位数：按 weights 对 vals 排序后插值 CDF。"""
    order = np.argsort(vals)
    vs = vals[order]
    ws = weights[order]
    cdf = np.cumsum(ws) / ws.sum()
    return float(np.interp(alpha, cdf, vs))


def w_cvar(vals, weights, alpha):
    """加权 CVaR：低于 alpha 分位的加权平均。"""
    q = w_quantile(vals, weights, alpha)
    m = vals <= q
    return float(np.sum(weights[m] * vals[m]) / np.sum(weights[m]))


# ---------- 三种历史压力情景 ----------
scenarios = {
    "2008 GFC（股灾）":     (x[:, 0] < -0.12, 0.22),
    "2020 新冠（急跌）":     (x[:, 0] < -0.13, 0.10),
    "2022 利率冲击（债跌）": (x[:, 1] < -0.04, 0.18),
}

results = {}
for name, (mask, target) in scenarios.items():
    f = mask.astype(float).reshape(-1, 1)
    q, lam = entropy_pooling(f, np.array([target]), p)
    q_prior, cvar_prior = w_quantile(rp, p, 0.05), w_cvar(rp, p, 0.05)
    q_post, cvar_post = w_quantile(x @ W, q, 0.05), w_cvar(x @ W, q, 0.05)
    results[name] = dict(q=q, lam=lam,
                         var_prior=q_prior, cvar_prior=cvar_prior,
                         var_post=q_post, cvar_post=cvar_post,
                         p_crash_prior=mask.mean(), p_crash_post=mask @ q)

print("先验 VaR95=%.4f CVaR=%.4f" % (w_quantile(rp, p, 0.05), w_cvar(rp, p, 0.05)))
for n, r in results.items():
    print("%s: λ=%.3f P(灾难) %.3f->%.3f | VaR95 %.4f->%.4f | CVaR %.4f->%.4f" % (
        n, r["lam"][0], r["p_crash_prior"], r["p_crash_post"],
        r["var_prior"], r["var_post"], r["cvar_prior"], r["cvar_post"]))

# ============ 图 1：先验 vs 压力后验（股灾情景）密度 ============
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist(x[:, 0], bins=60, weights=p, density=True, color=C["prior"],
        alpha=0.55, label="先验（历史分布）", edgecolor="white", linewidth=0.3)
q1 = results["2008 GFC（股灾）"]["q"]
ax.hist(x[:, 0], bins=60, weights=q1, density=True, color=C["post"],
        alpha=0.55, label="压力后验（GFC 观点）", edgecolor="white", linewidth=0.3)
ax.axvline(-0.15, color=C["post"], ls="--", lw=1.2, label="灾难阈值 −15%")
ax.set_xlabel("股票月度收益率", fontsize=11)
ax.set_ylabel("概率密度", fontsize=11)
ax.set_title("熵池化：一个「股灾概率 25%」的观点把左尾显著抬高", fontsize=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "ep_prior_vs_stressed_density.png"), dpi=130)
plt.close(fig)

# ============ 图 2：组合 P&L CDF，先验 vs 三种压力 ============
fig, ax = plt.subplots(figsize=(10, 4.6))
def cdf_vals(r):
    s = np.sort(r)
    return s, np.arange(1, len(s) + 1) / len(s)
sv, sc = cdf_vals(rp)
ax.plot(sv, sc, color=C["prior"], lw=1.8, label="先验 CDF")
for name, col in [("2008 GFC（股灾）", C["post"]),
                   ("2020 新冠（急跌）", C["eq"]),
                   ("2022 利率冲击（债跌）", C["gd"])]:
    rq = x[results[name]["q"] > 1e-6] @ W
    v, c = cdf_vals(rq)
    ax.plot(v, c, lw=1.4, color=col, label=name)
ax.axhline(0.05, color="#888888", ls=":", lw=1.0)
ax.set_xlabel("组合月度收益率", fontsize=11)
ax.set_ylabel("累积概率", fontsize=11)
ax.set_title("组合收益 CDF：压力后验把 5% 分位（VaR）明显左移", fontsize=12)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "ep_portfolio_cdf_stress.png"), dpi=130)
plt.close(fig)

# ============ 图 3：95% VaR（损失幅度）先验 vs 三种压力 ============
fig, ax = plt.subplots(figsize=(10, 4.6))
names = list(results.keys())
prior_var = -results["2008 GFC（股灾）"]["var_prior"]
post_var = [-results[n]["var_post"] for n in names]
xpos = np.arange(len(names))
ax.bar(xpos - 0.2, [prior_var] * len(names), width=0.4, color=C["prior"],
        label="先验 VaR95（损失）")
ax.bar(xpos + 0.2, post_var, width=0.4, color=C["post"],
        label="压力后验 VaR95（损失）")
for i, (pv, qv) in enumerate(zip([prior_var] * len(names), post_var)):
    ax.text(i - 0.2, pv + 0.002, "%.1f%%" % (pv * 100), ha="center", fontsize=8)
    ax.text(i + 0.2, qv + 0.002, "%.1f%%" % (qv * 100), ha="center", fontsize=8, color=C["post"])
ax.set_xticks(xpos)
ax.set_xticklabels([n.split("（")[0] for n in names], fontsize=10)
ax.set_ylabel("95% VaR（月度损失幅度）", fontsize=11)
ax.set_title("同一组合，三种历史压力情景下的 95% VaR 抬升", fontsize=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, color=C["grid"], axis="y")
plt.tight_layout()
fig.savefig(os.path.join(D, "ep_var_by_scenario.png"), dpi=130)
plt.close(fig)

print("saved 3 figures to", D)
