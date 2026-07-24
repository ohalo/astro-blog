import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(3)
OUT = "public/images/oneclass-svm-anomaly"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# "Normal" market is BIMODAL: two regimes (calm & stressed).
#   feature space: (市场因子, 波动因子)
# Anomalies live in the VALLEY between the two regimes -- days that
# match NEITHER regime. Gaussian/Mahalanobis rates the valley as the
# center (blind); OC-SVM learns a non-convex support and catches them.
# ---------------------------------------------------------------
n_anom = 30
k = 500

def clipped_gaussian(mean, cov, m, chi2=6.0):
    """Sample a Gaussian but drop the far tails so each regime is a compact blob."""
    mean = np.asarray(mean, float)
    s = rng.multivariate_normal(mean, cov, int(m * 1.8))
    d = np.einsum("ij,jk,ik->i", s - mean, np.linalg.inv(cov), s - mean)
    return s[d < chi2][:m]

calm = clipped_gaussian([-2.5, -2.5], [[0.18, 0.07], [0.07, 0.18]], k)
stress = clipped_gaussian([2.5, 2.5], [[0.22, -0.09], [-0.09, 0.22]], k)
Xn = np.vstack([calm, stress])
N = len(Xn)
regime = np.r_[np.zeros(len(calm)), np.ones(len(stress))].astype(int)

# anomalies in the valley near the global mean (Mahalanobis-blind)
an = rng.multivariate_normal([0.0, 0.0], [[0.10, 0.0], [0.0, 0.10]], n_anom)
X = np.vstack([Xn, an])
labels = np.r_[np.zeros(N), np.ones(n_anom)].astype(int)

mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd
n = len(Xs)

# ---------------------------------------------------------------
# One-Class SVM (RBF) dual, solved with projected gradient.
#   min 0.5 a^T K a   s.t. 0<=a<=C, sum a = 1,   C = 1/(nu*n)
#   decision f(x) = sum a_i K(x_i,x); boundary at rho (margin SVs)
# ---------------------------------------------------------------
def rbf(A, B, gamma):
    a2 = (A ** 2).sum(1)[:, None]
    b2 = (B ** 2).sum(1)[None, :]
    return np.exp(-gamma * np.maximum(a2 + b2 - 2 * A @ B.T, 0))

def project_capped_simplex(v, C, total=1.0, iters=80):
    """Project v onto {0<=a<=C, sum a = total} via bisection on tau."""
    lo, hi = v.min() - C, v.max()
    for _ in range(iters):
        tau = 0.5 * (lo + hi)
        if np.clip(v - tau, 0, C).sum() > total:
            lo = tau
        else:
            hi = tau
    return np.clip(v - 0.5 * (lo + hi), 0, C)

def fit_ocsvm(K, nu, iters=900):
    C = 1.0 / (nu * n)
    a = project_capped_simplex(np.full(n, C), C)
    lr = 1.0 / (np.linalg.norm(K, 2) + 1e-9)
    for _ in range(iters):
        a = project_capped_simplex(a - lr * (K @ a), C)
    f = K @ a
    sv = (a > 1e-8) & (a < C - 1e-8)
    rho = np.median(f[sv]) if sv.sum() > 0 else np.quantile(f, nu)
    return a, f, rho

def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1 = pos.sum(); n0 = (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def topk_prec(y, s, kk):
    return y[np.argsort(-s)[:kk]].mean()

nu = 0.05
gamma = 1.0
K = rbf(Xs, Xs, gamma)
a, f, rho = fit_ocsvm(K, nu)
score = rho - f                      # higher = more anomalous
ocsvm_auc = auc(labels, score)

# Mahalanobis baseline (single Gaussian on "normal")
Cov = np.cov(Xs.T)
Ci = np.linalg.pinv(Cov)
maha = np.einsum("ij,jk,ik->i", Xs, Ci, Xs)
maha_auc = auc(labels, maha)

print(f"OC-SVM AUC={ocsvm_auc:.3f}  Mahalanobis AUC={maha_auc:.3f}")
print(f"top-{n_anom} precision: OC-SVM {topk_prec(labels,score,n_anom):.2f} "
      f"Maha {topk_prec(labels,maha,n_anom):.2f}")

# ---------- FIG 1 (cover): decision boundary vs the two regimes ----------
xx, yy = np.meshgrid(
    np.linspace(Xs[:, 0].min() - 0.4, Xs[:, 0].max() + 0.4, 260),
    np.linspace(Xs[:, 1].min() - 0.4, Xs[:, 1].max() + 0.4, 260),
)
grid = np.c_[xx.ravel(), yy.ravel()]
fg = (rho - rbf(grid, Xs, gamma) @ a).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(9, 6.2))
cs = ax.contourf(xx, yy, fg, levels=24, cmap="RdBu_r", alpha=0.85)
ax.contour(xx, yy, fg, levels=[0], colors="k", linewidths=2.2)
ax.scatter(Xs[:N][regime == 0, 0], Xs[:N][regime == 0, 1], s=12,
           c="#1f4e79", alpha=0.45, label="正常·平静区制")
ax.scatter(Xs[:N][regime == 1, 0], Xs[:N][regime == 1, 1], s=12,
           c="#2e7d32", alpha=0.45, label="正常·承压区制")
ax.scatter(Xs[N:, 0], Xs[N:, 1], s=60, c="yellow", edgecolors="k",
           label="异常日（落在两区制之间）", zorder=5)
ax.set_title("One-Class SVM：用高维边界圈出『不正常』的交易日")
ax.set_xlabel("市场因子（标准化）"); ax.set_ylabel("波动因子（标准化）")
ax.legend(loc="upper left", framealpha=0.9)
fig.colorbar(cs, ax=ax, label="决策函数值（<0 判为异常）")
fig.tight_layout(); fig.savefig(f"{OUT}/cover.png", dpi=120); plt.close(fig)

# ---------- FIG 2: OC-SVM vs Mahalanobis score, why Gaussian fails ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc = axes[0].scatter(Xs[:, 0], Xs[:, 1], c=maha, cmap="viridis", s=14)
axes[0].scatter(Xs[N:, 0], Xs[N:, 1], s=55, facecolors="none",
                edgecolors="red", linewidths=1.6, label="真实异常")
axes[0].set_title(f"马氏距离：把『山谷』当中心 (AUC {maha_auc:.3f})")
axes[0].set_xlabel("市场因子"); axes[0].set_ylabel("波动因子")
axes[0].legend(loc="upper left"); fig.colorbar(sc, ax=axes[0], label="马氏距离")

sc2 = axes[1].scatter(Xs[:, 0], Xs[:, 1], c=score, cmap="viridis", s=14)
axes[1].scatter(Xs[N:, 0], Xs[N:, 1], s=55, facecolors="none",
                edgecolors="red", linewidths=1.6, label="真实异常")
axes[1].set_title(f"OC-SVM 分数：非凸支撑抓住山谷 (AUC {ocsvm_auc:.3f})")
axes[1].set_xlabel("市场因子"); axes[1].set_ylabel("波动因子")
axes[1].legend(loc="upper left"); fig.colorbar(sc2, ax=axes[1], label="异常分数")
fig.tight_layout(); fig.savefig(f"{OUT}/score_map.png", dpi=120); plt.close(fig)

# ---------- FIG 3: nu sensitivity ----------
nus = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
precs, recs = [], []
for nv in nus:
    _, fv, rhov = fit_ocsvm(K, nv, iters=500)
    sc = rhov - fv
    flagged = sc > 0
    tp = (flagged & (labels == 1)).sum()
    precs.append(tp / max(flagged.sum(), 1))
    recs.append(tp / n_anom)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nus, precs, "o-", color="#4C72B0", label="精确率 Precision")
ax.plot(nus, recs, "s-", color="#C44E52", label="召回率 Recall")
ax.set_title("ν 参数敏感性：污染率旋钮直接决定精确率/召回率取舍")
ax.set_xlabel("ν（异常比例上界 / 支持向量比例下界）"); ax.set_ylabel("比例")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/nu_sensitivity.png", dpi=120); plt.close(fig)

# ---------- FIG 4: gamma sensitivity ----------
gammas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
aucs = []
for g in gammas:
    Kg = rbf(Xs, Xs, g)
    _, fg2, rg = fit_ocsvm(Kg, nu, iters=600)
    aucs.append(auc(labels, rg - fg2))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gammas, aucs, "o-", color="#55A868", label="OC-SVM AUC")
ax.axhline(maha_auc, color="#C44E52", ls="--", label=f"马氏距离基线 {maha_auc:.3f}")
ax.set_xscale("log")
ax.set_title("RBF γ 敏感性：太小欠拟合成一团、太大把每个点都圈成孤岛")
ax.set_xlabel("γ（RBF 核宽度，对数轴）"); ax.set_ylabel("AUC")
ax.grid(alpha=0.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/gamma_sensitivity.png", dpi=120); plt.close(fig)

print("saved figures to", OUT)
