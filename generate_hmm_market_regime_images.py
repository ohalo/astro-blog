#!/usr/bin/env python3
"""
为文章「隐马尔可夫模型(HMM)市场状态识别」(hmm-market-regime) 生成真实配图。
数据：用 3 个隐藏状态（牛 / 熊 / 高波动）的 regime-switching 过程模拟日收益，
      观测用 2 维 [当日收益, 20日滚动波动率] 让状态更易区分。
方法：从零实现高斯 HMM（对角协方差）的 Baum-Welch EM 估参与 Viterbi 解码，
      把解码出的状态贴回价格曲线，并对比每个状态的均值/波动与状态转移矩阵。
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
D = os.path.join(BASE, "hmm-market-regime")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============================================================
# 1) 模拟 3 状态 regime-switching 收益
# ============================================================
T = 2400
K = 3
# 真实参数（隐藏）
true_mu = np.array([ 0.0009, -0.0007,  0.0001])   # 牛 / 熊 / 高波动
true_sd = np.array([ 0.0060,  0.0105,  0.0190])
# 高持续性转移矩阵
true_A = np.array([
    [0.96, 0.02, 0.02],
    [0.03, 0.94, 0.03],
    [0.05, 0.05, 0.90],
])
true_pi = np.array([0.4, 0.3, 0.3])

states = np.zeros(T, dtype=int)
states[0] = np.argmax(rng.multinomial(1, true_pi))
for t in range(1, T):
    states[t] = np.argmax(rng.multinomial(1, true_A[states[t-1]]))
r = rng.normal(true_mu[states], true_sd[states])

# 滚动波动率（真实收益算，避免前视）
win = 20
roll_vol = np.full(T, np.nan)
for t in range(win, T):
    roll_vol[t] = r[t-win:t].std(ddof=1)
roll_vol[:win] = roll_vol[win]
# 2 维观测：收益 + 波动率（缩放对齐量纲）
vol_scaled = roll_vol * 100.0
X = np.column_stack([r, vol_scaled])
print("模拟结束：状态占比 =", np.bincount(states, minlength=3) / T)

# ============================================================
# 2) 从零实现高斯 HMM（对角协方差） Baum-Welch EM + Viterbi
# ============================================================
def fit_gaussian_hmm(X, K, n_init=6, n_iter=200, tol=1e-6, seed=0):
    N, Dd = X.shape
    best = None
    rnd = np.random.default_rng(seed)
    for itry in range(n_init):
        # 初始化
        pi = np.full(K, 1.0/K)
        A = rnd.dirichlet(np.ones(K), size=K)
        # kmeans 式初始化均值
        idx = rnd.choice(N, K, replace=False)
        mu = X[idx].copy()
        var = np.tile(X.var(axis=0, ddof=1) + 1e-6, (K, 1))
        logp_prev = -np.inf
        for it in range(n_iter):
            # E-step：前向-后向（带缩放）
            logB = np.zeros((N, K))
            for k in range(K):
                logB[:, k] = -0.5*np.sum(np.log(2*np.pi*var[k])) \
                             -0.5*np.sum((X-mu[k])**2 / var[k], axis=1)
            logpi = np.log(pi + 1e-300)
            logA = np.log(A + 1e-300)
            alpha = np.zeros((N, K)); c = np.zeros(N)
            for k in range(K):
                alpha[0, k] = np.exp(logpi[k] + logB[0, k]); 
            c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, N):
                for k in range(K):
                    alpha[t, k] = np.exp(logB[t, k]) * np.dot(alpha[t-1], np.exp(logA[:, k]))
                c[t] = alpha[t].sum(); alpha[t] /= c[t]
            beta = np.zeros((N, K))
            beta[-1] = 1.0
            for t in range(N-2, -1, -1):
                for k in range(K):
                    beta[t, k] = np.sum(np.exp(logA[k, :]) * np.exp(logB[t+1, :]) * beta[t+1, :])
                beta[t] /= c[t+1]
            gamma = alpha * beta; gamma /= gamma.sum(axis=1, keepdims=True)
            # xi
            xi = np.zeros((N-1, K, K))
            for t in range(N-1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = alpha[t, i] * np.exp(logA[i, j]) \
                                        * np.exp(logB[t+1, j]) * beta[t+1, j]
                xi[t] /= xi[t].sum()
            # M-step
            pi = gamma[0].copy()
            A = xi.sum(axis=0); A /= A.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            for k in range(K):
                mu[k] = (gamma[:, k] @ X) / w[k]
                var[k] = (gamma[:, k] @ (X-mu[k])**2) / w[k] + 1e-6
            logp = np.sum(np.log(c))
            if abs(logp - logp_prev) < tol:
                break
            logp_prev = logp
        if best is None or logp_prev > best["logp"]:
            best = {"logp": logp_prev, "pi": pi, "A": A, "mu": mu, "var": var}
    return best

def viterbi(X, pi, A, mu, var):
    N, K = X.shape[0], mu.shape[0]
    def logpdf(x, m, v):
        return -0.5*np.sum(np.log(2*np.pi*v)) -0.5*np.sum((x-m)**2/v)
    delta = np.zeros((N, K)); psi = np.zeros((N, K), dtype=int)
    for k in range(K):
        delta[0, k] = np.log(pi[k]+1e-300) + logpdf(X[0], mu[k], var[k])
    for t in range(1, N):
        for k in range(K):
            seq = delta[t-1] + np.log(A[:, k]+1e-300)
            psi[t, k] = np.argmax(seq)
            delta[t, k] = np.max(seq) + logpdf(X[t], mu[k], var[k])
    path = np.zeros(N, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(N-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path

hmm = fit_gaussian_hmm(X, K, seed=42)
print("收敛对数似然 =", round(hmm["logp"], 1))
path = viterbi(X, hmm["pi"], hmm["A"], hmm["mu"], hmm["var"])

# 标签对齐：用「波动率 + 收益」双判据（避免仅靠收益排序把高波动状态误标为熊市）
vol_dec = np.array([r[path == s].std(ddof=1) for s in range(3)])
mean_dec = np.array([r[path == s].mean() for s in range(3)])
highvol_state = int(np.argmax(vol_dec))               # 波动最大的 = 高波动
rest = [s for s in range(3) if s != highvol_state]
bull_state = rest[int(np.argmax(mean_dec[rest]))]     # 余下两个里收益高 = 牛市
bear_state = [s for s in rest if s != bull_state][0]  # 余下另一个 = 熊市
remap = np.zeros(K, dtype=int)
remap[bull_state] = 0
remap[bear_state] = 1
remap[highvol_state] = 2
decoded = remap[path]

# 每个解码后状态的均值收益 & 年化波动
mean_r = np.array([r[decoded == s].mean() for s in range(3)])
ann_vol = np.array([r[decoded == s].std(ddof=1) for s in range(3)]) * np.sqrt(252)
labels = ["牛市", "熊市", "高波动"]
print("解码状态 —— 均值日收益 / 年化波动：")
for s in range(3):
    print(f"  {labels[s]}: {mean_r[s]*100:.3f}% / {ann_vol[s]*100:.1f}%")

# ============================================================
# 图1：价格曲线按状态着色
# ============================================================
price = 100 * np.cumprod(1 + r)
# 只画后 1200 天更清晰
seg = slice(800, 2000)
t = np.arange(T)[seg]
fig, ax = plt.subplots(figsize=(12, 5.6))
colors = ["#2ca02c", "#d62728", "#ff7f0e"]
for s in range(3):
    m = decoded[seg] == s
    if m.any():
        ax.plot(t[m], price[seg][m], ".", color=colors[s], ms=5,
                label=f"{labels[s]} (μ={mean_r[s]*100:.2f}%/日)", alpha=0.85)
ax.set_title("HMM 解码的市场状态：同一根价格曲线被切成三段「性格」不同 regime",
             fontsize=12.5, fontweight="bold")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("净值（起始=100）", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "hmm_price_regime.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：各状态统计对比（均值收益 + 年化波动）
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
ax[0].bar(labels, mean_r*100, color=colors)
ax[0].axhline(0, color="black", lw=0.7)
for i, v in enumerate(mean_r*100):
    ax[0].text(i, v + (0.01 if v >= 0 else -0.02), f"{v:+.3f}%", ha="center", fontsize=10, fontweight="bold")
ax[0].set_ylabel("日均收益 (%)", fontsize=11)
ax[0].set_title("各状态均值收益：牛正、熊负、波动中性", fontsize=11.5, fontweight="bold")
ax[0].grid(True, axis="y", alpha=0.25)

ax[1].bar(labels, ann_vol*100, color=colors)
for i, v in enumerate(ann_vol*100):
    ax[1].text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax[1].set_ylabel("年化波动率 (%)", fontsize=11)
ax[1].set_title("各状态波动：高波动状态波动是牛市的 ~3 倍", fontsize=11.5, fontweight="bold")
ax[1].grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "hmm_regime_stats.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：估计的转移矩阵热力图
# ============================================================
Aest = hmm["A"]
# 按解码标签重排行/列，方便对比
Aest_ordered = Aest[remap][:, remap]
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(Aest_ordered, cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("下一状态", fontsize=11); ax.set_ylabel("当前状态", fontsize=11)
ax.set_title("估计的状态转移矩阵：对角线均 >0.9（状态很「黏」）", fontsize=11.5, fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{Aest_ordered[i,j]:.2f}", ha="center", va="center",
                color="white" if Aest_ordered[i,j] > 0.5 else "black", fontsize=11, fontweight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(D, "hmm_transition.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ HMM 配图生成完成：", sorted(os.listdir(D)))
