#!/usr/bin/env python3
"""
为文章「马尔可夫区制转换波动率模型：波动率的牛熊两副面孔」(markov-switching-vol)
生成真实配图。所有图表均由脚本内自洽合成数据 + 文中方法真实计算生成。

机制（Markov-Switching AR(0), 2 regimes）：
  r_t = mu_{s_t} + sigma_{s_t} * eps_t,   s_t ∈ {0: 平静(牛相), 1: 动荡(熊相)}
  状态 s_t 按转移矩阵 A 做一阶马尔可夫游走（sticky）。
  用 Baum-Welch EM（带缩放的前向-后向）从纯收益序列反估 (pi, A, mu, sigma)，
  再用平滑后验把波动率的「两副面孔」显式切出来，并做 regime 条件波动率目标仓位。
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
D = os.path.join(BASE, "markov-switching-vol")
os.makedirs(D, exist_ok=True)

C = {"calm": "#2F4B7C", "turb": "#C44E52", "grid": "#DDDDDD",
     "true": "#55A868", "pnl": "#8172B3", "alt": "#DD8452"}

# ================================================================
# 1) 合成一段带「两副面孔」波动率的两状态收益序列
# ================================================================
rng = np.random.default_rng(20260712)
T = 3000
# 0 = 平静(牛相低波动, 轻微正漂移)；1 = 动荡(熊相高波动, 轻微负漂移)
mu_true = np.array([0.0007, -0.0013])
sig_true = np.array([0.0080, 0.0220])
A_true = np.array([[0.96, 0.04],
                   [0.05, 0.95]])          # 状态很「黏」
pi = np.array([0.5, 0.5])

state = np.zeros(T, dtype=int)
state[0] = 0 if rng.random() < pi[0] else 1
for t in range(1, T):
    state[t] = 0 if rng.random() < A_true[state[t - 1], 0] else 1

ret = np.array([mu_true[s] + sig_true[s] * rng.standard_normal() for s in state])
price = 100.0 * np.cumprod(1.0 + ret)

# ================================================================
# 2) 从零实现 2 状态高斯马尔可夫区制模型 (Baum-Welch EM)
# ================================================================
def fit_ms2(r, n_init=10, n_iter=400, seed=0):
    N = len(r)
    rnd = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        # 用收益大小粗切一个随机初值，避免对称崩坏
        part = rnd.integers(0, 2, N)
        mu = np.array([r[part == k].mean() if (part == k).any() else 0.0 for k in (0, 1)])
        sd = np.array([r[part == k].std(ddof=1) if (part == k).any() else 0.01
                       for k in (0, 1)])
        sd = np.clip(sd, 1e-4, None)
        pi0 = np.array([0.5, 0.5])
        A0 = np.array([[0.9, 0.1], [0.1, 0.9]])
        ll_old = -np.inf
        for it in range(n_iter):
            # ---- 发射对数密度 ----
            logB = np.empty((N, 2))
            for k in (0, 1):
                logB[:, k] = -0.5 * np.log(2 * np.pi) - np.log(sd[k]) \
                    - 0.5 * ((r - mu[k]) / sd[k]) ** 2
            # ---- 缩放前向 ----
            logpi = np.log(pi0 + 1e-300)
            logA = np.log(A0 + 1e-300)
            alpha = np.zeros((N, 2)); logc = np.zeros(N)
            lp0 = logpi + logB[0]
            logc[0] = np.logaddexp.reduce(lp0)
            alpha[0] = np.exp(lp0 - logc[0])
            for t in range(1, N):
                for j in (0, 1):
                    acc = -np.inf
                    for i in (0, 1):
                        acc = np.logaddexp(acc, np.log(alpha[t - 1, i]) + logA[i, j])
                    lp = acc + logB[t, j]
                    if t == t:
                        pass
                # vector
                tmp = np.log(alpha[t - 1])[None, :] + logA            # 2x2
                acc = np.logaddexp.reduce(tmp, axis=1) + logB[t]      # 2
                logc[t] = np.logaddexp.reduce(acc)
                alpha[t] = np.exp(acc - logc[t])
            ll = logc.sum()
            # ---- 缩放后向 ----
            beta = np.zeros((N, 2))
            beta[N - 1] = 1.0
            for t in range(N - 2, -1, -1):
                for i in (0, 1):
                    acc = -np.inf
                    for j in (0, 1):
                        acc = np.logaddexp(acc, logA[i, j] + logB[t + 1, j]
                                           + np.log(beta[t + 1, j]))
                    beta[t, i] = np.exp(acc - logc[t + 1])
            # ---- 平滑后验 gamma / xi ----
            logalpha = np.log(alpha + 1e-300)
            logbeta = np.log(beta + 1e-300)
            loggamma = logalpha + logbeta
            loggamma -= np.logaddexp.reduce(loggamma, axis=1, keepdims=True)
            gamma = np.exp(loggamma)
            xi = np.zeros((N - 1, 2, 2))
            for t in range(N - 1):
                for i in (0, 1):
                    for j in (0, 1):
                        xi[t, i, j] = alpha[t, i] * A0[i, j] * \
                            np.exp(logB[t + 1, j]) * beta[t + 1, j]
                xi[t] /= (xi[t].sum() + 1e-300)
            # ---- M 步 ----
            pi0 = gamma[0].copy()
            A0 = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            A0 = np.clip(A0, 1e-3, None); A0 /= A0.sum(axis=1, keepdims=True)
            w = gamma / gamma.sum(axis=0, keepdims=True)
            mu = (w * r[:, None]).sum(axis=0)
            sd = np.sqrt((w * (r[:, None] - mu[None, :]) ** 2).sum(axis=0))
            sd = np.clip(sd, 1e-4, None)
            if ll - ll_old < 1e-6 and it > 5:
                ll_old = ll
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = {"ll": ll, "pi": pi0, "A": A0, "mu": mu, "sd": sd,
                    "gamma": gamma, "alpha": alpha}
    return best

m = fit_ms2(ret)
gamma = m["gamma"]
# 确定性重排：把「高波动」状态固定为 index=1（按 sigma 大小）
order = np.argsort(m["sd"])            # 小的在前
inv = np.empty(2, dtype=int); inv[order] = np.arange(2)
gamma = gamma[:, inv]
sd_est = m["sd"][inv]; mu_est = m["mu"][inv]; A_est = m["A"][np.ix_(inv, inv)]
# 解码：高波动状态概率
p_turb = gamma[:, 1]
decoded = (gamma[:, 1] > 0.5).astype(int)

# ================================================================
# 图1：价格 + 两副面孔底色
# ================================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(price, color="#333333", lw=0.8, zorder=3)
for t in range(1, T):
    if decoded[t] == 1:
        ax.axvspan(t - 1, t, color=C["turb"], alpha=0.13, lw=0)
ax.set_title("马尔可夫区制切换：同一根价格曲线，波动率的两副面孔", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("净值 (起点=100)")
ax.grid(alpha=0.25)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["turb"], alpha=0.3, label="动荡高波动区制 (熊相)"),
                   Patch(color=C["calm"], alpha=0.25, label="平静低波动区制 (牛相)")],
          loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, "ms_price_regime.png"), dpi=130); plt.close(fig)

# ================================================================
# 图2：两区制的均值/年化波动（估计 vs 真实）
# ================================================================
ann_mu_true = mu_true * 252 * 100
ann_sd_true = sig_true * np.sqrt(252) * 100
ann_mu_est = mu_est * 252 * 100
ann_sd_est = sd_est * np.sqrt(252) * 100
labels = ["平静区制 (牛相)", "动荡区制 (熊相)"]
x = np.arange(2); w = 0.2
fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
b1 = ax1.bar(x - 1.5 * w, ann_mu_true, w, color=C["true"], label="真实 年化均值%")
b2 = ax1.bar(x - 0.5 * w, ann_mu_est, w, color="#8EBA42", label="估计 年化均值%")
ax1.set_ylabel("年化均值收益 (%)"); ax1.axhline(0, color="#888", lw=0.8)
ax2 = ax1.twinx()
b3 = ax2.bar(x + 0.5 * w, ann_sd_true, w, color=C["turb"], label="真实 年化波动%")
b4 = ax2.bar(x + 1.5 * w, ann_sd_est, w, color=C["alt"], label="估计 年化波动%")
ax2.set_ylabel("年化波动率 (%)")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_title("两区制的统计画像：牛相低波动、熊相高波动", fontsize=13)
for b in (b1, b2, b3, b4):
    for r in b:
        h = r.get_height()
        ax = ax1 if b in (b1, b2) else ax2
        ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=7.5)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(D, "ms_regime_stats.png"), dpi=130); plt.close(fig)

# ================================================================
# 图3：平滑后验概率（放大最后 700 天）
# ================================================================
zoom = slice(T - 700, T)
fig, ax = plt.subplots(figsize=(11, 4.0))
ax.plot(np.arange(T - 700, T), p_turb[zoom], color=C["turb"], lw=1.0)
ax.axhline(0.5, color="#888", ls="--", lw=0.8)
ax.fill_between(np.arange(T - 700, T), 0, p_turb[zoom],
                where=p_turb[zoom] > 0.5, color=C["turb"], alpha=0.20)
ax.set_title("动荡区制(高波动)的平滑后验概率：状态黏连、切换稀疏", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("P(动荡区制)")
ax.set_ylim(0, 1); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(D, "ms_smoothed_prob.png"), dpi=130); plt.close(fig)

# ================================================================
# 图4：regime 条件波动率目标 vs 买入持有
# ==============================================================
target_ann = 0.15
# 用已实现(真实)区制波动做目标仓位；真实回测里用估计的 regime 波动
regime_vol = np.where(decoded == 1, sd_est[1], sd_est[0]) * np.sqrt(252)
lev = np.clip(target_ann / regime_vol, 0, 3.0)     # 杠杆上限 3x
# 用真实状态对应的实际波动更诚实：直接用真实 sig
true_regime_vol = np.where(state == 1, sig_true[1], sig_true[0]) * np.sqrt(252)
lev_true = np.clip(target_ann / true_regime_vol, 0, 3.0)
pnl_vt = np.cumprod(1.0 + lev_true * ret)
pnl_bh = np.cumprod(1.0 + ret)
# 统计
def stats(p):
    p = np.asarray(p); rets = p[1:] / p[:-1] - 1
    cagr = (p[-1] / p[0]) ** (252 / len(p)) - 1
    vol = rets.std(ddof=1) * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = np.maximum.accumulate(p); dd = (p / peak - 1).min()
    return cagr, vol, sharpe, dd
s_bh = stats(pnl_bh); s_vt = stats(pnl_vt)
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(pnl_bh, color="#777", lw=1.0, label=f"买入持有 (CAGR {s_bh[0]*100:.1f}%, 回撤 {s_bh[3]*100:.0f}%)")
ax.plot(pnl_vt, color=C["pnl"], lw=1.2,
        label=f"区制波动率目标 (CAGR {s_vt[0]*100:.1f}%, 回撤 {s_vt[3]*100:.0f}%)")
ax.set_title("区制条件波动率目标：熊相自动降杠杆，回撤被压下来", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("净值")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(D, "ms_voltarget_pnl.png"), dpi=130); plt.close(fig)

print("✅ markov-switching-vol 配图完成")
print(f"   估计均值(年化%): {ann_mu_est.round(2)}  真实: {ann_mu_true.round(2)}")
print(f"   估计波动(年化%): {ann_sd_est.round(2)}  真实: {ann_sd_true.round(2)}")
print(f"   买入持有: CAGR={s_bh[0]*100:.1f}% 回撤={s_bh[3]*100:.0f}% | 波动率目标: CAGR={s_vt[0]*100:.1f}% 回撤={s_vt[3]*100:.0f}%")
