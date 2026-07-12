#!/usr/bin/env python3
"""
为文章「熵池化(Entropy Pooling)：用最小相对熵把主观情景变成概率分布」(entropy-pooling)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 先验：用历史模拟法生成 K 个多资产收益情景 x_k，先验概率 p_k = 1/K（等权）。
  * 观点：把主观观点写成线性矩约束 E^q[f_i(X)] = μ_i，即 Σ_k q_k f_i(x_k) = μ_i。
  * 熵池化：在「满足约束」下找与先验 p 相对熵最小的后验 q：
        min_q  Σ_k q_k ln(q_k / p_k)   s.t.  Σ_k q_k = 1,  Σ_k q_k f_i(x_k) = μ_i
    对偶 g(λ) = λ·μ − ln Σ_k p_k exp(−λ·f(x_k))；牛顿一步 λ += Cov_q(f)⁻¹ (μ − E_q[f])。
    后验闭式 q_k ∝ p_k · exp(−λ·f(x_k))。全程在 log 空间计算，避免指数溢出。
  * 应用：用后验情景分布重算均值-方差组合权重，看一个观点如何把配置从先验推到后验。

注意：本模拟用于演示机制，数字量级参考真实多资产（月度收益/波动），但非真实数据。
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
D = os.path.join(BASE, "entropy-pooling")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "post": "#C44E52",
     "prior": "#BBBBBB", "grid": "#DDDDDD", "thr": "#888888", "mkt": "#CCB974"}

rng = np.random.default_rng(20260713)
K = 4000
# 3 资产（月度收益）先验多元正态：股票 / 债券 / 黄金
mu0 = np.array([0.0070, 0.0030, 0.0045])
sd = np.array([0.0400, 0.0150, 0.0350])
corr = np.array([[1.0, -0.20, 0.10],
                 [-0.20, 1.0, 0.05],
                 [0.10, 0.05, 1.0]])
cov = np.diag(sd) @ corr @ np.diag(sd)
L = np.linalg.cholesky(cov)
X = rng.standard_normal((K, 3)) @ L.T + mu0          # 情景矩阵 (K×3)
p = np.ones(K) / K

NAMES = ["股票 Equity", "债券 Bonds", "黄金 Gold"]


# ============================================================
# 熵池化求解器（向量约束牛顿法，对偶闭式，log 空间稳定）
# ============================================================
def g_lam(lam, f, p, tgt):
    """熵池化对偶函数 G(λ) = λ·μ − ln Σ_k p_k exp(λ·f_k)（Meucci 约定 q∝p·exp(+λf)），log 空间稳定。"""
    a = np.log(p) + (lam @ f.T)
    m = a.max()
    logZ = m + np.log(np.sum(np.exp(a - m)))
    return float(lam @ tgt - logZ)


def ep_solve(f, target, p, maxit=100, reg=1e-9, tol=1e-12):
    f = np.atleast_2d(f.T).T                        # (K, m)
    tgt = np.atleast_1d(np.asarray(target, dtype=float))
    lam = np.zeros(f.shape[1])
    gv = g_lam(lam, f, p, tgt)
    for _ in range(maxit):
        ef = lam @ f.T
        a0 = ef.max()
        w = np.exp(ef - a0)                          # 稳定：q ∝ p·exp(+λ·f)
        Z = (w * p).sum()
        q = (w * p) / Z
        fbar = (q[:, None] * f).sum(0)
        grad = tgt - fbar                            # = ∇G(λ)
        if np.max(np.abs(grad)) < tol:
            break
        d = f - fbar
        Cov = (d * q[:, None]).T @ d + reg * np.eye(f.shape[1])
        step = np.linalg.solve(Cov, grad)            # 牛顿方向（对偶凹→上升方向）
        # 回溯线搜索：保证对偶目标 G 单调上升，避免步长过大越过根
        alpha = 1.0
        gnew = g_lam(lam + alpha * step, f, p, tgt)
        while (not np.isfinite(gnew) or gnew < gv - 1e-10) and alpha > 1e-12:
            alpha *= 0.5
            gnew = g_lam(lam + alpha * step, f, p, tgt)
        lam = lam + alpha * step
        gv = gnew
    # 末次稳定重算
    ef = lam @ f.T
    a0 = ef.max()
    w = np.exp(ef - a0)
    q = (w * p) / (w * p).sum()
    return q, lam


def proj_simplex(v):
    v = np.clip(v, 0, None)
    s = v.sum()
    return v / s if s > 1e-300 else np.ones_like(v) / len(v)


def mv_weights(q, X, gamma=30.0, iters=6000):
    mu = X.T @ q
    Sig = X.T @ (X * q[:, None]) - np.outer(mu, mu) + 1e-8 * np.eye(X.shape[1])
    Lmax = max(float(np.max(np.linalg.eigvalsh(gamma * Sig))), 1e-8)
    step = 1.0 / Lmax
    w = np.ones(3) / 3.0
    for _ in range(iters):
        g = mu - gamma * (Sig @ w)
        w = proj_simplex(w + step * g)
    return w


def kl_div(q, p):
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))


def entropy(q):
    m = q > 0
    return float(-np.sum(q[m] * np.log(q[m])))


def wmean(mask, q):
    return float(np.average(mask, weights=q))


# ============================================================
# 先验 + 两类观点
# ============================================================
prior_crash = wmean((X[:, 0] < -0.06), p)          # 股票单月暴跌>6% 的崩盘情景
prior_mu = X.T @ p

# 观点1（软倾斜）：我们认为股票远期月收益应为 +1.2%（高于先验 ~0.7%）
q1, lam1 = ep_solve(X[:, 0], 0.012, p)
post1_eq = float(X[:, 0] @ q1)
post1_crash = wmean((X[:, 0] < -0.06), q1)

# 观点2（压力情景）：把「股票单月暴跌>6%」崩盘情景的主观概率抬到 15%
crash_target = 0.15
crash_ind = (X[:, 0] < -0.06).astype(float)
q2, lam2 = ep_solve(crash_ind, crash_target, p)
post2_crash = wmean(crash_ind, q2)
post2_eq = float(X[:, 0] @ q2)

w0 = mv_weights(p, X)
w1 = mv_weights(q1, X)
w2 = mv_weights(q2, X)

kl1 = kl_div(q1, p)
kl2 = kl_div(q2, p)
H0 = entropy(p)
H1 = entropy(q1)
H2 = entropy(q2)


# ============================================================
# 图 1：情景云 —— 先验（等权）vs 后验（看多股票观点）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
for ax, Q, title, cmap, ispost in [
    (axes[0], None, "先验情景：等权 p_k = 1/K（所有情景同等可能）", "Greys", False),
    (axes[1], q1, "后验情景：看多股票观点下，概率质量向高股票收益区移动", "Reds", True),
]:
    if ispost:
        sc = ax.scatter(X[:, 0], X[:, 1], c=Q * K, cmap=cmap, s=7, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("后验概率质量 (×K)")
    else:
        ax.scatter(X[:, 0], X[:, 1], c=C["prior"], s=7, alpha=0.5)
    ax.axvline(0, color=C["thr"], lw=0.6)
    ax.axhline(0, color=C["thr"], lw=0.6)
    ax.set_xlabel("股票月度收益")
    ax.set_ylabel("债券月度收益")
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ep_scenarios_prior_posterior.png"), dpi=130)
plt.close()


# ============================================================
# 图 2：股票收益边际分布 先验 vs 后验（看多观点）
# ============================================================
bins = np.linspace(-0.10, 0.10, 50)
prior_w, _ = np.histogram(X[:, 0], bins=bins, weights=p, density=True)
post_w, _ = np.histogram(X[:, 0], bins=bins, weights=q1, density=True)
centers = 0.5 * (bins[:-1] + bins[1:])
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.plot(centers, prior_w, color=C["prior"], lw=2, label="先验 p（看多前）")
ax.plot(centers, post_w, color=C["post"], lw=2, label="后验 q（看多股票观点）")
ax.axvline(prior_mu[0], color=C["eq"], ls="--", lw=1, label="先验均值 %.2f%%" % (prior_mu[0] * 100))
ax.axvline(post1_eq, color=C["bd"], ls="--", lw=1, label="后验均值 %.2f%%" % (post1_eq * 100))
ax.set_xlabel("股票月度收益"); ax.set_ylabel("概率密度")
ax.set_title("熵池化把主观观点「编码」进分布：后验均值被推到目标 1.20%，且分布被拉宽")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "ep_marginal_equity.png"), dpi=130)
plt.close()


# ============================================================
# 图 3：均值-方差组合权重 先验 vs 观点1 vs 观点2
# ============================================================
labels = ["先验 (无观点)", "观点1：看多股票", "观点2：崩盘压力"]
W = np.array([w0, w1, w2])
x = np.arange(3); width = 0.25
fig, ax = plt.subplots(figsize=(10, 5.0))
for i in range(3):
    ax.bar(x + (i - 1) * width, W[:, i], width, label=NAMES[i],
           color=[C["eq"], C["bd"], C["gd"]][i], alpha=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("组合权重")
ax.set_title("一个观点改写整张配置：看多股票加仓股票；崩盘压力砍股票、加债券黄金")
ax.legend(fontsize=8, ncol=3); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "ep_portfolio_weights.png"), dpi=130)
plt.close()


# ============================================================
# 图 4：压力阶梯 —— 崩盘概率目标越高，股票权重越低、对先验扭曲(KL)越大
# ============================================================
targets = np.array([0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40])
eq_w = []; kls = []
for tg in targets:
    qq, _ = ep_solve(crash_ind, tg, p)
    eq_w.append(float(mv_weights(qq, X)[0]))
    kls.append(kl_div(qq, p))
fig, ax1 = plt.subplots(figsize=(10, 5.0))
ax1.plot(targets * 100, eq_w, color=C["eq"], lw=2, marker="o", label="股票权重")
ax1.set_xlabel("施加的崩盘概率目标 (%%) ，先验 ≈ %.1f%%" % (prior_crash * 100))
ax1.set_ylabel("股票权重", color=C["eq"])
ax1.tick_params(axis="y", labelcolor=C["eq"])
ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(targets * 100, kls, color=C["post"], lw=2, marker="s", label="KL(q‖p)")
ax2.set_ylabel("相对熵 KL(q‖p)", color=C["post"])
ax2.tick_params(axis="y", labelcolor=C["post"])
ax1.set_title("压力阶梯：越极端的观点越偏离先验（KL↑），配置越去风险（股票权重↓）")
plt.tight_layout()
plt.savefig(os.path.join(D, "ep_stress_ladder.png"), dpi=130)
plt.close()


# ============================================================
# 关键数字
# ============================================================
print("=== 熵池化 Entropy Pooling 关键数字 ===")
print("情景数 K=%d, 资产=股票/债券/黄金, 先验等权 p_k=1/K" % K)
print("先验均值(月): 股票 %.2f%% 债券 %.2f%% 黄金 %.2f%%"
      % (prior_mu[0] * 100, prior_mu[1] * 100, prior_mu[2] * 100))
print("先验崩盘(股票月跌>6%%)概率 = %.2f%%" % (prior_crash * 100))
print("--- 观点1：E^q[r_股票]=1.20%%/月 ---")
print("后验达成 E^q[r_股票]=%.3f%% ; 崩盘概率由 %.2f%% -> %.2f%%"
      % (post1_eq * 100, prior_crash * 100, post1_crash * 100))
print("KL(q1‖p)=%.4f ; 后验熵 H(q1)=%.3f (先验熵 H(p)=%.3f)"
      % (kl1, H1, H0))
print("--- 观点2：崩盘概率=%.0f%% ---" % (crash_target * 100))
print("后验达成崩盘概率=%.2f%% ; E^q[r_股票]由 %.2f%% -> %.2f%%"
      % (post2_crash * 100, prior_mu[0] * 100, post2_eq * 100))
print("KL(q2‖p)=%.4f ; 后验熵 H(q2)=%.3f" % (kl2, H2))
print("--- 均值-方差权重(γ=30) ---")
print("先验:     股票 %.1f%% 债券 %.1f%% 黄金 %.1f%%"
      % (w0[0] * 100, w0[1] * 100, w0[2] * 100))
print("观点1(看多): 股票 %.1f%% 债券 %.1f%% 黄金 %.1f%%"
      % (w1[0] * 100, w1[1] * 100, w1[2] * 100))
print("观点2(崩盘): 股票 %.1f%% 债券 %.1f%% 黄金 %.1f%%"
      % (w2[0] * 100, w2[1] * 100, w2[2] * 100))
print("图片已保存到:", D)
