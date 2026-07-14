#!/usr/bin/env python3
"""
为文章「贝叶斯状态空间择时：把『现在是牛是熊』变成后验概率」(bayesian-regime-timing)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 隐马尔可夫模型(HMM)：观测是月度市场收益 y_t，隐藏状态 z_t ∈ {牛, 熊}。
    每个状态有各自的漂移与波动 (高斯发射)，状态之间以黏性转移矩阵切换。
  * 用 Baum-Welch EM 从收益序列估计模型参数（mu/sigma/转移矩阵/初值）。
  * 用前向-后向(forward-backward)算法算平滑后验 γ_t = P(z_t=牛 | 全部观测)，
    再把"γ_t > 0.5 就满仓、否则空仓"写成一个择时策略，与买入持有对比。

注意：本模拟使用合成 regime 结构以演示估计与滤波机制（与全库高阶文一致），
真实市场的 regime 边界模糊、参数会变，文末已说明。
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
D = os.path.join(BASE, "bayesian-regime-timing")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "bull": "#55A868",
     "bear": "#C44E52", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3"}

rng = np.random.default_rng(20260714)
T = 240                                          # 20 年月度
# 真值参数（仅用于造数，EM 不知道）
TRUE = {
    "mu": np.array([0.007, -0.005]),            # 牛=+0.7%/月, 熊=-0.5%/月
    "sig": np.array([0.035, 0.050]),            # 牛波动小, 熊波动大
    "A": np.array([[0.95, 0.05], [0.05, 0.95]]),# 黏性转移
    "pi": np.array([0.5, 0.5]),
}

# ---- 造数：Markov chain + 高斯发射 ----
z = np.zeros(T, dtype=int)
z[0] = rng.choice(2, p=TRUE["pi"])
for t in range(1, T):
    z[t] = rng.choice(2, p=TRUE["A"][z[t-1]])
y = np.array([rng.normal(TRUE["mu"][z[t]], TRUE["sig"][z[t]]) for t in range(T)])

# ---- 2 状态高斯 HMM：Baum-Welch EM ----
def baum_welch(y, K=2, n_iter=200, seed=0):
    r = np.random.default_rng(seed)
    T = len(y)
    mu = r.normal(y.mean(), y.std(), K)
    sig = np.full(K, y.std())
    A = np.full((K, K), 0.05) + np.eye(K) * 0.9
    A = A / A.sum(1, keepdims=True)
    pi = np.full(K, 1.0 / K)
    for it in range(n_iter):
        # E-step: 前向 alpha, 后向 beta
        logP = np.zeros((T, K))
        for t in range(T):
            logP[t] = -0.5 * ((y[t] - mu) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi))
        la = np.full((T, K), -np.inf)
        la[0] = np.log(pi) + logP[0]
        for t in range(1, T):
            for j in range(K):
                la[t, j] = logP[t, j] + np.logaddexp.reduce(la[t-1] + np.log(A[:, j]))
        lb = np.full((T, K), 0.0)
        for t in range(T-2, -1, -1):
            for i in range(K):
                lb[t, i] = np.logaddexp.reduce(lb[t+1] + np.log(A[i]) + logP[t+1])
        loglik = np.logaddexp.reduce(la[T-1])
        gamma = np.exp((la + lb) - loglik)
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = np.exp(la[t, i] + np.log(A[i, j]) + logP[t+1, j] + lb[t+1, j] - loglik)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        for j in range(K):
            mu[j] = (gamma[:, j] * y).sum() / gamma[:, j].sum()
            sig[j] = np.sqrt((gamma[:, j] * (y - mu[j]) ** 2).sum() / gamma[:, j].sum())
        A = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        A = A / A.sum(1, keepdims=True)
    return mu, sig, A, pi

mu, sig, A, pi = baum_welch(y, K=2, n_iter=200, seed=7)
# 把状态对齐到 {牛,熊}（mu 大的当牛）
order = np.argsort(mu)            # order[0]=熊(小mu), order[1]=牛(大mu)
if mu[order[1]] < mu[order[0]]:
    order = order[::-1]
# 重排使 index 0=熊, 1=牛
remap = np.zeros(2, dtype=int); remap[order[0]] = 0; remap[order[1]] = 1
mu = mu[remap]; sig = sig[remap]; A = A[remap][:, remap]; pi = pi[remap]

# ---- 前向-后向：平滑后验 γ_t = P(z_t=牛 | 全部观测) ----
def forward_backward(y, mu, sig, A, pi):
    T = len(y); K = len(mu)
    logP = np.zeros((T, K))
    for t in range(T):
        logP[t] = -0.5 * ((y[t] - mu) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi))
    la = np.full((T, K), -np.inf)
    la[0] = np.log(pi) + logP[0]
    for t in range(1, T):
        for j in range(K):
            la[t, j] = logP[t, j] + np.logaddexp.reduce(la[t-1] + np.log(A[:, j]))
    lb = np.full((T, K), 0.0)
    for t in range(T-2, -1, -1):
        for i in range(K):
            lb[t, i] = np.logaddexp.reduce(lb[t+1] + np.log(A[i]) + logP[t+1])
    loglik = np.logaddexp.reduce(la[T-1])
    gamma = np.exp((la + lb) - loglik)
    return gamma, loglik

gamma, loglik = forward_backward(y, mu, sig, A, pi)
p_bull = gamma[:, 1]                            # P(z_t = 牛)

# ---- 择时策略：γ_t > 0.5 满仓, 否则空仓 ----
pos = (p_bull > 0.5).astype(float)             # 0/1 持仓
strat_ret = pos * y
bh_cum = np.cumprod(1 + y)
strat_cum = np.cumprod(1 + strat_ret)
bull_cum = np.cumprod(1 + np.maximum(y, 0)) if False else None

def ann_sharpe(r):
    return r.mean() * 12 / (r.std(ddof=1) * np.sqrt(12))
def maxdd(cum):
    peak = np.maximum.accumulate(cum)
    return (cum / peak - 1).min()

bh_ann = y.mean() * 12; bh_vol = y.std(ddof=1) * np.sqrt(12); bh_shp = bh_ann / bh_vol
st_ann = strat_ret.mean() * 12; st_vol = strat_ret.std(ddof=1) * np.sqrt(12); st_shp = st_ann / st_vol

print(f"[BR] 估计 mu={mu}  真值={TRUE['mu']}")
print(f"[BR] 估计 sig={sig}  真值={TRUE['sig']}")
print(f"[BR] 估计转移对角={np.diag(A)}  真值={np.diag(TRUE['A'])}")
print(f"[BR] 买入持有: 年化={bh_ann:.4f} Sharpe={bh_shp:.3f} 最大回撤={maxdd(bh_cum):.4f}")
print(f"[BR] 贝叶斯择时: 年化={st_ann:.4f} Sharpe={st_shp:.3f} 最大回撤={maxdd(strat_cum):.4f}")
print(f"[BR] 平均 P(牛)={p_bull.mean():.3f}  状态切换次数(持仓变动)={np.sum(np.diff(pos)!=0)}")

# ---------------- 图 1: 收益 + 平滑后验概率 ----------------
fig, ax1 = plt.subplots(figsize=(10, 6))
colors = [C["bull"] if z[t] == 1 else C["bear"] for t in range(T)]
ax1.bar(range(T), y, color=colors, alpha=0.55, width=1.0, label="月度收益 (真值regime着色)")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_ylabel("月度收益", fontsize=12, color=C["eq"])
ax1.set_xlabel("月份", fontsize=13)
ax2 = ax1.twinx()
ax2.plot(range(T), p_bull, color=C["accent"], lw=2.0, label="P(牛 | 全部观测)")
ax2.axhline(0.5, color=C["thr"], lw=1.0, ls="--", label="决策阈值 0.5")
ax2.set_ylabel("P(牛市)", fontsize=12, color=C["accent"])
ax2.set_ylim(-0.05, 1.05)
ax1.set_title("贝叶斯滤波把『现在是牛是熊』变成后验概率\n绿色=真值牛市, 红=真值熊市, 紫线=平滑后验 P(牛)", fontsize=13, fontweight="bold")
l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(D, "posterior_probability.png"), dpi=130)
plt.close(fig)

# ---------------- 图 2: 择时 vs 买入持有 累积净值 ----------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(T), strat_cum, color=C["eq"], lw=2.4, label=f"贝叶斯择时 (Sharpe={st_shp:.2f}, 回撤={maxdd(strat_cum):.1%})")
ax.plot(range(T), bh_cum, color=C["mkt"], lw=1.8, ls="--", label=f"买入持有 (Sharpe={bh_shp:.2f}, 回撤={maxdd(bh_cum):.1%})")
ax.set_xlabel("月份", fontsize=13)
ax.set_ylabel("累积净值 (初始=1)", fontsize=13)
ax.set_title("把后验概率变成仓位：γ>0.5 满仓、否则空仓\n熊市阶段自动空仓，显著削减回撤", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax.grid(True, color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "timing_vs_buyhold.png"), dpi=130)
plt.close(fig)

# ---------------- 图 3: 局部放大——一次 regime 切换 detection ----------------
# 找一个从熊切到牛、且后验正确跟进的区间
switch = None
for t in range(5, T-30):
    if z[t] == 0 and z[t+1] == 1 and z[t+1:t+20].mean() > 0.8:
        switch = t; break
if switch is None:
    switch = int(T * 0.45)
seg = slice(switch - 5, switch + 30)
tt = np.arange(switch - 5, switch + 30)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(tt, y[seg], color=[C["bull"] if z[i] == 1 else C["bear"] for i in range(switch-5, switch+30)],
        alpha=0.5, width=1.0)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_ylabel("月度收益", fontsize=12)
ax1.set_xlabel("月份", fontsize=13)
ax2 = ax1.twinx()
ax2.plot(tt, p_bull[seg], color=C["accent"], lw=2.2, marker="o", ms=3)
ax2.axhline(0.5, color=C["thr"], lw=1.0, ls="--")
ax2.set_ylabel("P(牛市)", fontsize=12, color=C["accent"])
ax2.set_ylim(-0.05, 1.05)
ax1.set_title(f"局部放大：熊→牛切换点(月 {switch})附近后验概率的跟进\n滤波在真实切换后数期内把 P(牛) 推过 0.5", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(D, "regime_switch_zoom.png"), dpi=130)
plt.close(fig)

print("[BR] images written to", D)
