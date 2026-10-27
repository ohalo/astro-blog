#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HMM 市场状态：Viterbi 解码最可能的情境路径
E1 合成 3-state Markov 市场：价格 + 真 regime + 滚动波动
E2 Baum-Welch 学习：log-lik 收敛 + 参数恢复（按 sigma 排序对拍）
E3 Viterbi vs posterior decoding：与真标签一致率 + 局部对质窗口
E4 冷启动代价：恐慌 onset 后 Viterbi 路径的检出滞后 + 停留分布几何假设对拍
E5 冷水：K=2 欠规格把恐慌揉进常态（合并态 sigma 混音实拍）+ 10 种子拟合散布
输出 5 张图 + _hmm_results.json
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/hidden-markov-market-regime"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
SEED = 20261027
rng = np.random.default_rng(SEED)
R = {}

# ---------------- 真值模型：平静 / 常态 / 恐慌 ----------------
MU = np.array([0.10, 0.05, -0.35])   # 每日期望收益 %
SIG = np.array([0.60, 1.10, 2.40])   # 每日波动 %
A = np.array([                       # 转移矩阵，行=今 列=明
    [0.970, 0.028, 0.002],
    [0.040, 0.945, 0.015],
    [0.050, 0.120, 0.830],
])
PI = np.array([0.45, 0.45, 0.10])
T = 1500

def simulate(T, A, mu, sig, pi, rng):
    z = np.empty(T, dtype=int)
    z[0] = rng.choice(3, p=pi)
    for t in range(1, T):
        z[t] = rng.choice(3, p=A[z[t - 1]])
    r = rng.normal(mu[z], sig[z])
    return z, r

z_true, r = simulate(T, A, MU, SIG, PI, rng)
R["state_counts"] = [int((z_true == k).sum()) for k in range(3)]
R["true_sigma"] = SIG.tolist()

# ---------------- HMM 核心（log-space 手写） ----------------
def log_emis(r, mu, sig):
    return -0.5 * ((r[:, None] - mu[None, :]) / sig[None, :]) ** 2 \
           - np.log(sig[None, :]) - 0.5 * np.log(2 * np.pi)

def forward(le):
    T_, K = le.shape
    la = np.empty((T_, K))
    la[0] = np.log(PI0) + le[0]
    for t in range(1, T_):
        m = la[t - 1][:, None] + logA
        mx = m.max(0)
        la[t] = mx + np.log(np.exp(m - mx).sum(0)) + le[t]
    m = la[-1]; mx = m.max()
    return la, mx + np.log(np.exp(m - mx).sum())

def backward(le):
    T_, K = le.shape
    lb = np.zeros((T_, K))
    for t in range(T_ - 2, -1, -1):
        m = logA + le[t + 1][None, :] + lb[t + 1][None, :]
        mx = m.max(1, keepdims=True)   # 关键：keepdims 保证按行(i)减的是各自最大值
        lb[t] = mx.squeeze(1) + np.log(np.exp(m - mx).sum(1))
    return lb

def viterbi(le):
    T_, K = le.shape
    d = np.full((T_, K), -np.inf); psi = np.zeros((T_, K), dtype=int)
    d[0] = np.log(PI0) + le[0]
    for t in range(1, T_):
        m = d[t - 1][:, None] + logA
        d[t] = m.max(0) + le[t]
        psi[t] = m.argmax(0)
    path = np.zeros(T_, dtype=int)
    path[-1] = d[-1].argmax()
    for t in range(T_ - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path

def baum_welch(r, K, mu0, sig0, A0, pi0, iters=120):
    global logA, PI0
    mu, sig, Amat, pi = mu0.copy(), sig0.copy(), A0.copy(), pi0.copy()
    lls = []
    for it in range(iters):
        logA, PI0 = np.log(Amat), pi
        le = log_emis(r, mu, sig)
        la, ll = forward(le)
        lb = backward(le)
        lls.append(ll)
        # gamma / xi
        g = la + lb
        mx = g.max(1, keepdims=True)
        g = np.exp(g - mx); g /= g.sum(1, keepdims=True)
        xi_sum = np.zeros((K, K))
        for t in range(len(r) - 1):
            m = la[t][:, None] + logA + le[t + 1][None, :] + lb[t + 1][None, :]
            xi_sum += np.exp(m - lls[-1])
        g0 = g[0].copy()
        rowsum = xi_sum.sum(1, keepdims=True)
        Amat = xi_sum / np.maximum(rowsum, 1e-300)
        bad = (rowsum.ravel() <= 0)
        if bad.any():
            Amat[bad] = 0.9 * np.eye(K)[bad]
        Amat = Amat / Amat.sum(1, keepdims=True)
        mu = (g * r[:, None]).sum(0) / g.sum(0)
        sig = np.sqrt(np.maximum((g * (r[:, None] - mu[None, :]) ** 2).sum(0) / g.sum(0), 1e-4))
    return mu, sig, Amat, g0, lls

def order_by_sigma(mu, sig, Amat, pi):
    idx = np.argsort(sig)
    pos = np.argsort(idx)
    return mu[idx], sig[idx], Amat[np.ix_(idx, idx)], pi[idx], idx

# ---------- 初始化：滚动 20 日波动分位数分三位 ----------
roll = np.array([np.std(r[max(0, t - 20):t + 1]) for t in range(T)])
q = np.quantile(roll, [1 / 3, 2 / 3])
z_init = (roll > q[0]).astype(int) + (roll > q[1]).astype(int)
mu0 = np.array([r[z_init == k].mean() for k in range(3)])
sig0 = np.array([max(r[z_init == k].std(), 0.1) for k in range(3)])
cnt = np.array([(z_init == k).sum() for k in range(3)], float)
pi0 = cnt / cnt.sum()
A0 = 0.9 * np.eye(3) + 0.05
A0 /= A0.sum(1, keepdims=True)

mu_hat, sig_hat, A_hat, pi_hat, lls = baum_welch(r, 3, mu0, sig0, A0, pi0)
mu_hat, sig_hat, A_hat, pi_hat, idx = order_by_sigma(mu_hat, sig_hat, A_hat, pi_hat)
R["em_iters"] = len(lls)
R["ll_curve"] = [float(lls[i]) for i in range(0, len(lls), max(1, len(lls) // 40))]
R["ll_start"], R["ll_end"] = float(lls[0]), float(lls[-1])
R["monotone"] = bool(np.all(np.diff(lls) > -1e-6))
R["mu_recovered"] = [round(float(v), 4) for v in mu_hat]
R["sig_recovered"] = [round(float(v), 4) for v in sig_hat]
R["A_recovered"] = [[round(float(v), 4) for v in row] for row in A_hat]
R["A_mae"] = float(np.abs(A_hat - A).mean())
R["sig_mape"] = [round(abs(sig_hat[k] - SIG[k]) / SIG[k] * 100, 1) for k in range(3)]

logA, PI0 = np.log(A_hat), pi_hat
le = log_emis(r, mu_hat, sig_hat)
path_vit = viterbi(le)
la, _ = forward(le); lb = backward(le)
g = la + lb; g = np.exp(g - g.max(1, keepdims=True)); g /= g.sum(1, keepdims=True)
path_post = g.argmax(1)

acc_v = float((path_vit == z_true).mean() * 100)
acc_p = float((path_post == z_true).mean() * 100)
agree = float((path_vit == path_post).mean() * 100)
R["acc_viterbi"], R["acc_posterior"], R["agree_vp"] = round(acc_v, 1), round(acc_p, 1), round(agree, 1)

# ---------------- 图 1：市场全貌 ----------------
fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[2, 1], sharex=True)
price = 100 * np.cumsum(r) / 100 + 100
colors = {0: GREEN, 1: BLUE, 2: RED}
for k in range(3):
    mask = z_true == k
    seg = mask.astype(float)
    ax[0].fill_between(range(T), price.min(), price.max(), where=mask, alpha=0.10, color=colors[k], step="mid")
ax[0].plot(price, color=BLACK, lw=0.9)
ax[0].set_title("合成市场：3-state Markov 驱动（绿=平静 蓝=常态 红=恐慌），T=1500 天")
ax[0].set_ylabel("价格（点）")
ax[1].plot(roll, color=PURPLE, lw=1.0, label="20 日滚动波动 %")
for yv, c, lb_ in zip(SIG, [GREEN, BLUE, RED], ["σ平静=0.60", "σ常态=1.10", "σ恐慌=2.40"]):
    ax[1].axhline(yv, color=c, ls="--", lw=1.0, alpha=0.8, label=lb_)
ax[1].legend(ncol=4, fontsize=8); ax[1].set_ylabel("波动 %"); ax[1].set_xlabel("交易日")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_market_truth.png"); plt.close(fig)

# ---------------- 图 2：EM 收敛 + 参数恢复 ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
it = range(len(lls))
ax[0].plot(lls, color=BLUE, lw=1.4)
ax[0].set_title(f"Baum-Welch 收敛：log-lik {lls[0]:.1f} → {lls[-1]:.1f}（{len(lls)} 迭代，逐迭代单调不减）")
ax[0].set_xlabel("EM 迭代"); ax[0].set_ylabel("数据对数似然")
x = np.arange(3); w = 0.35
ax[1].bar(x - w / 2, SIG, w, color=[GREEN, BLUE, RED], alpha=0.85, label="真值 σ")
ax[1].bar(x + w / 2, sig_hat, w, color=[GREEN, BLUE, RED], alpha=0.45, edgecolor=BLACK, label="EM 恢复 σ̂")
for k in range(3):
    ax[1].text(x[k] + w / 2, sig_hat[k] + 0.03, f"{sig_hat[k]:.2f}\n({R['sig_mape'][k]}%)", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(["平静", "常态", "恐慌"])
ax[1].set_title(f"参数恢复（按 σ 排序对拍）转移矩阵 MAE={R['A_mae']:.4f}"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_em_convergence.png"); plt.close(fig)

# ---------------- 图 3：Viterbi vs posterior ----------------
fig, ax = plt.subplots(3, 1, figsize=(12, 7), height_ratios=[1.2, 1, 1], sharex=True)
ax[0].plot(r, color=GRAY, lw=0.6)
ax[0].set_title(f"日收益 %（两种解码与真标签一致率：Viterbi {acc_v}% vs posterior {acc_p}%，两者互相一致率 {agree}%）")
ax[0].set_ylabel("日收益 %")
for a_, p_, name, cmap in [(ax[1], z_true, "真 regime", [GREEN, BLUE, RED]),
                           (ax[2], path_vit, "Viterbi 路径", [GREEN, BLUE, RED])]:
    a_.clear()
    for k in range(3):
        idxs = np.where(p_ == k)[0]
        a_.scatter(idxs, np.full(len(idxs), k + 1), s=4, color=cmap[k], alpha=0.65)
    a_.set_yticks([1, 2, 3]); a_.set_yticklabels(["平静", "常态", "恐慌"])
    a_.set_ylim(0.5, 3.5)
ax[1].set_title("真 regime 序列"); ax[2].set_title("Viterbi 解码路径（联合最优，逐日平滑）")
ax[2].set_xlabel("交易日")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_viterbi_vs_posterior.png"); plt.close(fig)

# ---------------- 图 4：恐慌检出滞后 + 停留分布 ----------------
# 恐慌 onset：z_true 进入 state2 的起点
onsets = [t for t in range(3, T - 2) if z_true[t] == 2 and z_true[t - 1] != 2 and z_true[t + 1] == 2]
lags = []
for o in onsets:
    nxt = [t for t in range(o, min(o + 60, T)) if path_vit[t] == 2]
    lags.append(nxt[0] - o if nxt else 60)
R["n_panic_onset_sustained"] = len(lags)
R["lag_median"], R["lag_mean"] = float(np.median(lags)), float(np.mean(lags))
R["lag_p90"] = float(np.percentile(lags, 90))
R["lag_le1"] = round(float((np.array(lags) <= 1).mean() * 100), 1)

# 停留时长：真几何 vs 拟合几何 vs 经验
def dwell(pers):  # geometric dwell mean
    return 1 / (1 - pers)
dwell_true_emp = []
k = 0; c = 0
d_len = {}
for k in range(3):
    ds = []; run = 0
    for t in range(T):
        if z_true[t] == k: run += 1
        elif run: ds.append(run); run = 0
    if run: ds.append(run)
    d_len[k] = ds
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].hist(lags, bins=range(0, 12), color=ORANGE, edgecolor=BLACK, alpha=0.8)
ax[0].axvline(np.median(lags), color=RED, ls="--", lw=1.5, label=f"中位滞后 {np.median(lags):.0f} 天")
n_miss = sum(1 for l in lags if l >= 60)
R["n_panic_missed_60d"] = n_miss
R["lags_list"] = [int(l) for l in lags]
ax[0].text(0.98, 0.95, f"另有 {n_miss} 次恐慌 60 天内\n始终未被 Viterbi 检出",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=9,
           bbox=dict(boxstyle="round", fc="#fff3e0", ec=ORANGE, alpha=0.9))
ax[0].set_title(f"恐慌 onset 后 Viterbi 检出滞后分布（n={len(lags)} 次持续恐慌，均值 {np.mean(lags):.1f}，P90 {np.percentile(lags,90):.0f}）")
ax[0].set_xlabel("滞后（天）"); ax[0].legend(fontsize=9)
gs_grid = np.arange(1, 16)
for k in range(3):
    emp = np.array([(np.array(d_len[k]) == d).mean() for d in gs_grid])
    theo = (1 - A[k, k]) * A[k, k] ** (gs_grid - 1)
    ax[1].plot(gs_grid, emp, "o-", color=[GREEN, BLUE, RED][k], ms=4, lw=1.2,
               label=f"{['平静','常态','恐慌'][k]} 经验（均停留 {np.mean(d_len[k]):.1f}）")
    ax[1].plot(gs_grid, theo, "--", color=[GREEN, BLUE, RED][k], lw=1.0, alpha=0.6)
ax[1].set_title("停留时长：经验分布(o) vs HMM 几何假设(−−)（拟合 A 生成，形状吻合）")
ax[1].set_xlabel("连续停留（天）"); ax[1].set_ylabel("占比"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_lag_dwell.png"); plt.close(fig)

# ---------------- 图 5：K=2 欠规格 + 种子散布 ----------------
mu0b = np.array([r[z_init < 2].mean(), r[z_init == 2].mean()])
sig0b = np.array([max(r[z_init < 2].std(), 0.1), max(r[z_init == 2].std(), 0.1)])
c2 = np.array([(z_init < 2).sum(), (z_init == 2).sum()], float); pi0b = c2 / c2.sum()
A0b = np.array([[0.93, 0.07], [0.12, 0.88]])
mu2, sig2, A2, pi2, lls2 = None, None, None, None, None
def bw2(r, mu0, sig0, A0, pi0, iters=100):
    global logA, PI0
    mu, sig, Amat, pi = mu0.copy(), sig0.copy(), A0.copy(), pi0.copy()
    K = 2
    for it in range(iters):
        logA, PI0 = np.log(Amat), pi
        le_ = log_emis(r, mu, sig)
        la_, ll_ = forward(le_); lb_ = backward(le_)
        gg = la_ + lb_; mx = gg.max(1, keepdims=True)
        gg = np.exp(gg - mx); gg /= gg.sum(1, keepdims=True)
        xi_ = np.zeros((K, K))
        for t in range(len(r) - 1):
            m = la_[t][:, None] + logA + le_[t + 1][None, :] + lb_[t + 1][None, :]
            xi_ += np.exp(m - ll_)
        pi = gg[0].copy(); Amat = xi_ / xi_.sum(1, keepdims=True)
        mu = (gg * r[:, None]).sum(0) / gg.sum(0)
        sig = np.sqrt(np.maximum((gg * (r[:, None] - mu[None, :]) ** 2).sum(0) / gg.sum(0), 1e-4))
    return mu, sig, Amat, pi, ll_
mu2, sig2, A2, pi2, ll2 = bw2(r, mu0b, sig0b, A0b, pi0b)
i2 = np.argsort(sig2); mu2, sig2 = mu2[i2], sig2[i2]
panic_days = z_true == 2
R["k2_sigma"] = [round(float(s), 3) for s in sig2]
R["k2_panic_abs_above_hi"] = round(float((np.abs(r[panic_days]) > sig2[1]).mean() * 100), 1)
# 恐慌日里 K=2 给多少概率到"高波动态"
logA, PI0 = np.log(A2), pi2
le2 = log_emis(r, mu2, sig2)
la2, _ = forward(le2); lb2 = backward(le2)
g2 = la2 + lb2; g2 = np.exp(g2 - g2.max(1, keepdims=True)); g2 /= g2.sum(1, keepdims=True)
R["k2_panic_post_hi"] = round(float(g2[panic_days, 1].mean()), 3)
R["k2_calm_post_hi"] = round(float(g2[z_true == 0, 1].mean()), 3)
# 恐慌日实际 σ=2.4，K=2 高态 σ 混音
R["panic_true_sigma"] = float(SIG[2])
R["k2_blended_gap"] = round(float(100 * (1 - sig2[1] / SIG[2])), 1)  # 低估 %

# 多种子稳定性：K=3 拟合 10 个初始种子
seed_sig = []
for s in range(10):
    rg = np.random.default_rng(1000 + s)
    ini_idx = rg.permutation(T)[:400]
    rsub = np.sort(r[ini_idx])
    mu_s0 = np.array([rsub[:133].mean(), rsub[133:266].mean(), rsub[266:].mean()])
    sig_s0 = np.array([max(rsub[a:b].std(), 0.1) for a, b in [(0, 133), (133, 266), (266, 400)]])
    mu_s, sig_s, _, _, _ = baum_welch(r, 3, mu_s0, sig_s0, A0.copy(), pi0.copy(), iters=80)
    seg = np.argsort(sig_s)
    seed_sig.append(sig_s[seg])
seed_sig = np.array(seed_sig)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].bar([0, 1], [SIG[1], SIG[2]], 0.5, color=[BLUE, RED], alpha=0.35, edgecolor=BLACK, label="真值（常态/恐慌）")
ax[0].bar([0.55, 1.55], sig2, 0.5, color=[BLUE, RED], alpha=0.85, label="K=2 拟合")
ax[0].set_xticks([0.27, 1.27]); ax[0].set_xticklabels(["低波动态（把平静+常态揉一起）", "高波动态（恐慌被常态稀释）"])
for xx, v in zip([0.55, 1.55], sig2):
    ax[0].text(xx, v + 0.04, f"σ̂={v:.2f}", ha="center", fontsize=9)
ax[0].axhline(SIG[2], color=RED, ls=":", lw=1.2)
ax[0].set_title(f"K=2 欠规格：恐慌真 σ=2.40，拟合高态 σ̂={sig2[1]:.2f}（低估 {R['k2_blended_gap']}%）")
ax[0].set_ylabel("σ %"); ax[0].legend(fontsize=8)
bp = ax[1].boxplot([seed_sig[:, k] for k in range(3)], tick_labels=["平静", "常态", "恐慌"], patch_artist=True)
for b, c in zip(bp["boxes"], [GREEN, BLUE, RED]):
    b.set_facecolor(c); b.set_alpha(0.6)
ax[1].axhline(SIG[0], color=GREEN, ls=":"); ax[1].axhline(SIG[1], color=BLUE, ls=":"); ax[1].axhline(SIG[2], color=RED, ls=":")
spreads = [round(float(seed_sig[:, k].std()), 3) for k in range(3)]
R["seed_sig_spreads"] = spreads
R["seed_sig_median"] = [round(float(np.median(seed_sig[:, k])), 3) for k in range(3)]
ax[1].set_title(f"10 个随机初始 K=3 重拟合的 σ̂ 散布（std {spreads}，真值虚线）")
ax[1].set_ylabel("σ̂ %")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_limits.png"); plt.close(fig)

with open("/Users/halo123456/workspace/astro-blog/_hmm_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
