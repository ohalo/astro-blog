"""Wasserstein 鲁棒组合优化 配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/robust-optimization-wasserstein"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(11)

# ---------- 市场设定：5 个资产 ----------
N = 5
names = ["资产1", "资产2", "资产3", "资产4", "资产5"]
mu_true = np.array([0.05, 0.06, 0.07, 0.08, 0.09]) / 252
vol_true = np.array([0.12, 0.15, 0.18, 0.22, 0.28]) / np.sqrt(252)
corr = 0.35 + 0.65 * np.eye(N)
cov_true = np.outer(vol_true, vol_true) * corr
L = np.linalg.cholesky(cov_true)

def sample_returns(T, seed):
    r = np.random.default_rng(seed)
    z = r.standard_t(df=5, size=(T, N)) / np.sqrt(5 / 3)
    return mu_true + z @ L.T

# ---------- 优化器 ----------
def mv_weights(mu, cov, gamma=6.0):
    def f(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(f, np.full(N, 1 / N), bounds=[(0, 1)] * N, constraints=cons)
    return res.x

def wass_robust_weights(R, eps, gamma=6.0):
    """Wasserstein-DRO 均值方差近似:
    sup_{W(Q,P_hat)<=eps} 下的最坏期望 ≈ 样本目标 + eps * ||w||_2 (Lipschitz 正则)。
    """
    mu_hat = R.mean(axis=0)
    Rc = R - mu_hat
    cov_hat = Rc.T @ Rc / len(R)

    def f(w):
        base = w @ mu_hat - 0.5 * gamma * w @ cov_hat @ w
        return -(base - eps * np.linalg.norm(w))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(f, np.full(N, 1 / N), bounds=[(0, 1)] * N, constraints=cons)
    return res.x

def perf(w, R_oos, gamma=6.0):
    port = R_oos @ w
    mu = port.mean() * 252
    sd = port.std() * np.sqrt(252)
    return mu, sd, mu / sd

# ---------- 实验1: 样本内外差距 ----------
T_in = 252  # 一年日度样本估计
n_trials = 300
eps_grid = [0.0, 2e-4, 5e-4, 1e-3, 2e-3]

R_oos = sample_returns(252 * 40, seed=999)  # 长期真实分布近似

records = {e: [] for e in eps_grid}
w_store = {e: [] for e in eps_grid}
for t in range(n_trials):
    R_in = sample_returns(T_in, seed=1000 + t)
    for e in eps_grid:
        w = wass_robust_weights(R_in, e) if e > 0 else mv_weights(R_in.mean(0), np.cov(R_in.T))
        mu_i = (R_in @ w).mean() * 252
        mu_o, sd_o, sr_o = perf(w, R_oos)
        records[e].append((mu_i, mu_o, sr_o))
        w_store[e].append(w)

for e in eps_grid:
    arr = np.array(records[e])
    print(f"eps={e}: in-sample mu={arr[:,0].mean():.3f}, oos mu={arr[:,1].mean():.3f}, "
          f"oos SR={arr[:,2].mean():.3f}, gap={(arr[:,0]-arr[:,1]).mean():.3f}")

# 图1: 乐观差距（样本内承诺 vs 样本外兑现）
fig, ax = plt.subplots(figsize=(10, 5))
labels = ["ε=0\n(经典MV)"] + [f"ε={e:g}" for e in eps_grid[1:]]
gaps = [np.array(records[e])[:, 0] - np.array(records[e])[:, 1] for e in eps_grid]
bp = ax.boxplot(gaps, tick_labels=labels, showmeans=True, patch_artist=True)
for patch, c in zip(bp["boxes"], plt.cm.coolwarm(np.linspace(0.85, 0.15, len(eps_grid)))):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_ylabel("样本内年化收益 − 样本外年化收益")
ax.set_title(f"乐观差距：经典 MV 平均高估 {gaps[0].mean()*100:.1f}%，ε 增大逐步压缩（{n_trials} 次重抽样）")
fig.tight_layout()
fig.savefig(f"{OUT}/optimism_gap.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 图2: 权重稳定性
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, e, title in [(axes[0], 0.0, "经典 MV（ε=0）"), (axes[1], 1e-3, "Wasserstein 鲁棒（ε=1e-3）")]:
    Wm = np.array(w_store[e])
    bp = ax.boxplot([Wm[:, i] for i in range(N)], tick_labels=names, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4878b0" if e == 0 else "#6aa56e"); patch.set_alpha(0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("权重")
    turnover = np.abs(np.diff(Wm, axis=0)).sum(axis=1).mean()
    ax.set_title(f"{title}：跨样本平均权重变动 {turnover:.2f}")
fig.suptitle("同一市场、300 份一年期样本：鲁棒化让权重对估计噪声脱敏", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/weight_stability.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("turnover mv:", np.abs(np.diff(np.array(w_store[0.0]), axis=0)).sum(axis=1).mean())
print("turnover dro:", np.abs(np.diff(np.array(w_store[1e-3]), axis=0)).sum(axis=1).mean())

# 图3: eps 扫描——样本外 Sharpe 的倒U形
eps_scan = np.array([0, 1e-4, 2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 8e-3])
sr_mean, sr_lo, sr_hi = [], [], []
for e in eps_scan:
    srs = []
    for t in range(150):
        R_in = sample_returns(T_in, seed=5000 + t)
        w = wass_robust_weights(R_in, e) if e > 0 else mv_weights(R_in.mean(0), np.cov(R_in.T))
        srs.append(perf(w, R_oos)[2])
    srs = np.array(srs)
    sr_mean.append(srs.mean()); sr_lo.append(np.percentile(srs, 10)); sr_hi.append(np.percentile(srs, 90))

sr_mean = np.array(sr_mean)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(eps_scan * 1e4, sr_mean, "o-", color="#d1605e", label="样本外 Sharpe 均值")
ax.fill_between(eps_scan * 1e4, sr_lo, sr_hi, alpha=0.2, color="#d1605e", label="10%-90% 分位带")
best = eps_scan[np.argmax(sr_mean)]
ax.axvline(best * 1e4, color="k", ls="--", lw=1, label=f"最优 ε ≈ {best:g}")
# 等权基准
w_eq = np.full(N, 1 / N)
ax.axhline(perf(w_eq, R_oos)[2], color="#4878b0", ls=":", lw=1.5, label="等权基准")
ax.set_xlabel("ε (×1e-4)")
ax.set_ylabel("样本外年化 Sharpe")
ax.set_title("半径扫描：ε 太小回到过拟合，太大退化成等权/保守化过度")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eps_scan.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("eps scan SR:", dict(zip(eps_scan.tolist(), np.round(sr_mean, 3).tolist())))
print("eq SR:", perf(w_eq, R_oos)[2])
print("done")
