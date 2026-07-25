"""熵池化观点融合 配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/entropy-pooling-views"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---------- 1. 构造先验场景 ----------
J = 20000  # 场景数
N = 4      # 资产数: 股票A, 股票B, 债券, 黄金
mu_true = np.array([0.08, 0.10, 0.03, 0.05]) / 252
vol = np.array([0.22, 0.28, 0.05, 0.15]) / np.sqrt(252)
corr = np.array([
    [1.0, 0.6, -0.1, 0.1],
    [0.6, 1.0, -0.15, 0.05],
    [-0.1, -0.15, 1.0, 0.2],
    [0.1, 0.05, 0.2, 1.0],
])
cov = np.outer(vol, vol) * corr
# 用 t 分布做肥尾场景（月度）
L = np.linalg.cholesky(cov)
z = rng.standard_t(df=5, size=(J, N))
z = z / np.sqrt(5 / 3)  # 标准化方差
X = mu_true * 21 + (z @ L.T) * np.sqrt(21)  # 月度收益场景

p = np.full(J, 1.0 / J)  # 先验概率

names = ["股票A", "股票B", "债券", "黄金"]

# ---------- 2. 熵池化求解 ----------
def entropy_pooling(X, p, A, b):
    """min KL(q||p) s.t. A q = b, sum q = 1. 对偶求解。"""
    Aall = np.vstack([np.ones(len(p)), A])
    ball = np.concatenate([[1.0], b])

    def dual(lam):
        logq = np.log(p) - 1.0 - Aall.T @ lam
        q = np.exp(logq)
        return q.sum() + lam @ ball

    def grad(lam):
        logq = np.log(p) - 1.0 - Aall.T @ lam
        q = np.exp(logq)
        return ball - Aall @ q

    res = minimize(dual, np.zeros(Aall.shape[0]), jac=grad, method="L-BFGS-B")
    lam = res.x
    q = np.exp(np.log(p) - 1.0 - Aall.T @ lam)
    return q / q.sum()

# 观点1: 股票B 月度期望收益 = -1% (先验约 +0.83%)
# 观点2: 黄金月度期望收益 = +1.5% (先验约 +0.42%)
view_B = -0.01
view_G = 0.015
A = np.vstack([X[:, 1], X[:, 3]])
b = np.array([view_B, view_G])
q = entropy_pooling(X, p, A, b)

ens_prior = 1.0
ens_post = np.exp(-(q * np.log(q / p)).sum())  # 相对有效场景比例 exp(-KL)
print("KL divergence:", (q * np.log(q / p)).sum())
print("effective scenario ratio:", ens_post)
print("prior mean B, G (monthly):", X[:, 1].mean(), X[:, 3].mean())
print("posterior mean B, G:", q @ X[:, 1], q @ X[:, 3])
print("prior mean A (monthly):", X[:, 0].mean(), "posterior mean A:", q @ X[:, 0])
print("prior vol B:", X[:,1].std(), "posterior vol B:", np.sqrt(q @ (X[:,1] - q@X[:,1])**2))

# ---------- 图1: 先验 vs 后验分布 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
bins = np.linspace(-0.25, 0.25, 90)
for ax, idx, view, t in [(axes[0], 1, view_B, "股票B：观点 E[r]=-1%"),
                          (axes[1], 3, view_G, "黄金：观点 E[r]=+1.5%")]:
    ax.hist(X[:, idx], bins=bins, weights=p, alpha=0.55, label=f"先验 (均值 {X[:,idx].mean()*100:.2f}%)", color="#4878b0", density=False)
    ax.hist(X[:, idx], bins=bins, weights=q, alpha=0.55, label=f"后验 (均值 {(q@X[:,idx])*100:.2f}%)", color="#d1605e", density=False)
    ax.axvline(view, color="k", ls="--", lw=1.2, label=f"观点值 {view*100:.1f}%")
    ax.set_title(t)
    ax.set_xlabel("月度收益")
    ax.set_ylabel("概率质量")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.22, 0.22)
fig.suptitle("熵池化：同一批场景，只改概率不改数据", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/prior_posterior_dist.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- 图2: 场景权重倾斜 + 有效场景数 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sub = rng.choice(J, 4000, replace=False)
sc = axes[0].scatter(X[sub, 1], X[sub, 3], c=q[sub] / p[sub], s=5, cmap="coolwarm",
                     vmin=0, vmax=3)
axes[0].set_xlabel("股票B 月度收益")
axes[0].set_ylabel("黄金 月度收益")
axes[0].set_title("后验/先验概率比：股票B跌、黄金涨的场景被加权")
plt.colorbar(sc, ax=axes[0], label="q / p")

ratio = np.sort(q / p)[::-1]
axes[1].plot(np.arange(J) / J * 100, ratio, color="#4878b0")
axes[1].axhline(1.0, color="k", ls="--", lw=1)
axes[1].set_xlabel("场景排名分位 (%)")
axes[1].set_ylabel("q / p")
axes[1].set_title(f"概率倾斜曲线（有效场景比例 exp(-KL) = {ens_post*100:.1f}%）")
fig.tight_layout()
fig.savefig(f"{OUT}/scenario_tilt.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- 图3: 置信度混合下的组合权重路径 ----------
def opt_weights(mu, cov, gamma=8.0):
    def neg_u(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0, 1)] * len(mu)
    res = minimize(neg_u, np.full(len(mu), 1 / len(mu)), bounds=bnds, constraints=cons)
    return res.x

cs = np.linspace(0, 1, 21)
W = []
for c in cs:
    qc = (1 - c) * p + c * q
    mu_c = qc @ X
    Xc = X - mu_c
    cov_c = (Xc * qc[:, None]).T @ Xc
    W.append(opt_weights(mu_c, cov_c))
W = np.array(W)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4878b0", "#d1605e", "#6aa56e", "#c8a227"]
ax.stackplot(cs, W.T, labels=names, colors=colors, alpha=0.85)
ax.set_xlabel("观点置信度 c（0 = 纯先验，1 = 完全采纳观点）")
ax.set_ylabel("组合权重")
ax.set_title("置信度加权后验 q(c) = (1-c)·p + c·q 下的最优权重路径")
ax.legend(loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"{OUT}/confidence_weights.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 打印权重端点供正文引用
print("weights c=0:", dict(zip(names, np.round(W[0], 3))))
print("weights c=0.5:", dict(zip(names, np.round(W[10], 3))))
print("weights c=1:", dict(zip(names, np.round(W[-1], 3))))
print("done")
