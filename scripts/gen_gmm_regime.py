#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GMM regime 分类：EM 算法把市场分成多态
E1 3-regime 合成市场 (r, v) 二维特征：EM 参数恢复 vs k-means 硬聚类准确率
E2 EM 收敛：对数似然逐迭代单调上升（保证）+ 参数恢复误差
E3 模型选择：BIC 扫描 K=1..6，真 K=3 是否被选出
E4 时序 regime：Markov 转移 + 后验概率曲线 + 分 regime 的前瞻波动
E5 诚实局限：分量塌缩（σ→0，LL 爆炸）与方差下限正则化
"""
import os, json, itertools
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

OUT = "/Users/halo123456/workspace/astro-blog/public/images/gaussian-mixture-regime-classification"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE, TEAL = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7", "#00cec9"
rng = np.random.default_rng(20261027)
R = {}

# ---- DGP：3 个隐藏 regime（平静/常态/恐慌），特征 = (日收益 r, 波动代理 v)
TRUE_MU = np.array([[0.05, 0.6], [0.02, 1.2], [-0.15, 3.0]])
TRUE_SD = np.array([[0.40, 0.12], [0.80, 0.25], [2.80, 0.60]])
TRUE_W = np.array([0.55, 0.32, 0.13])

def sample(n):
    s = rng.choice(3, size=n, p=TRUE_W)
    mu = TRUE_MU[s] + rng.normal(0, 1, (n, 2)) * TRUE_SD[s]
    return mu, s

def logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return (np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis))).ravel()

def gmm_ll(X, w, mu, sd):
    d = X.shape[1]
    z = -0.5 * ((X[:, None, :] - mu[None]) / sd[None]) ** 2 - np.log(sd[None]) - 0.5 * d * np.log(2 * np.pi)
    return logsumexp(z.sum(2) + np.log(w), axis=1).sum()

def em(X, K, seed=None, sd_floor=1e-6, iters=300):
    r = np.random.default_rng(seed)
    idx = r.choice(len(X), K, replace=False)
    mu = X[idx].copy(); sd = np.full((K, X.shape[1]), X.std(0) * 0.8); w = np.full(K, 1 / K)
    lls = []
    for it in range(iters):
        sd = np.maximum(sd, sd_floor)
        d = X.shape[1]
        z = -0.5 * ((X[:, None, :] - mu[None]) / sd[None]) ** 2 - np.log(sd[None]) - 0.5 * d * np.log(2 * np.pi)
        lz = z.sum(2) + np.log(w)[None]
        lse = logsumexp(lz, axis=1)
        ll = lse.sum()
        lls.append(ll)
        g = np.exp(lz - lse[:, None])
        Nk = g.sum(0) + 1e-12
        w = Nk / len(X)
        mu = (g[:, :, None] * X[:, None, :]).sum(0) / Nk[:, None]
        sd = np.sqrt((g[:, :, None] * (X[:, None, :] - mu[None]) ** 2).sum(0) / Nk[:, None])
    return w, mu, sd, g, np.array(lls)

def kmeans(X, K, seed=0, iters=80):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(K):
            if (lab == k).any():
                C[k] = X[lab == k].mean(0)
    return lab, C

def best_acc(lab, s, K=3):
    return max(float(np.mean(np.array(p)[lab] == s)) for p in itertools.permutations(range(K)))

# ---------- E1 参数恢复 & k-means 对照
x1, s1 = sample(3000)
w1, mu1, sd1, g1, _ = em(x1, 3, seed=7)
lab1 = g1.argmax(1)
acc_gmm = best_acc(lab1, s1)
lab_km, _ = kmeans(x1, 3, seed=7)
acc_km = best_acc(lab_km, s1)
order = np.argsort(-mu1[:, 1])  # 按 v 中心排序对齐真 regime 的反序
w_err = float(np.abs(np.sort(w1) - np.sort(TRUE_W)).max())
R["E1"] = {"acc_gmm": round(acc_gmm, 4), "acc_kmeans": round(acc_km, 4), "w_err_max": round(w_err, 4)}
fig, ax = plt.subplots(figsize=(8.6, 6.2))
cols = [GREEN, BLUE, RED]
for k in range(3):
    m = lab1 == k
    ax.scatter(x1[m, 0] * 100, x1[m, 1], s=6, alpha=0.5, color=cols[k], label=f"GMM 分量 {k}（{m.sum()} 点）")
ang = np.linspace(0, 2 * np.pi, 100)
for k in range(3):
    ax.plot(mu1[k, 0] * 100 + 2 * sd1[k, 0] * np.cos(ang), mu1[k, 1] + 2 * sd1[k, 1] * np.sin(ang), color=BLACK, lw=1.2)
    ax.scatter(mu1[k, 0] * 100, mu1[k, 1], marker="X", s=120, color=BLACK, zorder=5)
ax.axvline(0, color=GRAY, lw=0.8); ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xlabel("日收益 (%)"); ax.set_ylabel("波动代理 v (%)")
ax.set_title(f"E1 EM 拟合的三态：软标签恢复率 {acc_gmm*100:.1f}%，k-means 硬聚类只有 {acc_km*100:.1f}%\n椭圆 = 估计的 ±2σ，黑叉 = 分量中心")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_fit_scatter.png", bbox_inches="tight"); plt.close(fig)
# ---------- E2 EM 收敛单调性 + 参数恢复
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
for sd0, c, lb in [(0.05, BLUE, "sd 初始 0.05（近塌缩）"), (0.5, TEAL, "sd 初始 0.5"), (3.0, ORANGE, "sd 初始 3.0（过胖）")]:
    r0 = np.random.default_rng(3)
    idx = r0.choice(len(x1), 3, replace=False)
    mu = x1[idx].copy(); sd = np.full((3, 2), sd0); w = np.full(3, 1 / 3)
    lls = []; prev = -np.inf; mono = True
    for it in range(120):
        sd = np.maximum(sd, 1e-6)
        z = -0.5 * ((x1[:, None] - mu[None]) / sd[None]) ** 2 - np.log(sd[None]) - np.log(2 * np.pi)
        lz = z.sum(2) + np.log(w)[None]
        lse = logsumexp(lz, axis=1)
        ll = lse.sum()
        if ll < prev - 1e-8:
            mono = False
        prev = ll; lls.append(ll)
        g = np.exp(lz - lse[:, None]); Nk = g.sum(0) + 1e-12
        w = Nk / len(x1); mu = (g[:, :, None] * x1[:, None, :]).sum(0) / Nk[:, None]
        sd = np.sqrt((g[:, :, None] * (x1[:, None, :] - mu[None]) ** 2).sum(0) / Nk[:, None])
    if len(lls) > 1:
        mono = bool(np.all(np.diff(lls) >= -1e-8))
    ax.plot(lls, color=c, lw=1.8, label=f"{lb}，单调 {mono}")
ax.set_xlabel("EM 迭代次数"); ax.set_ylabel("对数似然")
ax.set_xscale("log")
ax.set_title("E2a EM 的招牌保证：LL 逐迭代只增不减（3 组病态初始全部验证）")
ax.legend(fontsize=8)
ax = axes[1]
mu_true_sorted = TRUE_MU[np.argsort(-TRUE_MU[:, 1])]
mu_est_sorted = mu1[np.argsort(-mu1[:, 1])]
labels = ["w平静", "w常态", "w恐慌", "μr恐慌", "σr恐慌", "μv恐慌", "σv恐慌"]
true_v = [TRUE_W[0], TRUE_W[1], TRUE_W[2], TRUE_MU[2, 0], TRUE_SD[2, 0], TRUE_MU[2, 1], TRUE_SD[2, 1]]
est_v = [np.sort(w1)[2], np.sort(w1)[1], np.sort(w1)[0], mu_est_sorted[2, 0], mu_est_sorted[2, 0] * 0 + mu_est_sorted[0, 0] * 0 + mu_est_sorted[0, 0], mu_est_sorted[2, 1], mu_est_sorted[0, 1]]
# 用恐慌分量（v 最大）对比
kpanic = int(np.argmax(mu1[:, 1]))
est_v = [np.sort(w1)[::-1][0], np.sort(w1)[::-1][1], np.sort(w1)[::-1][2], mu1[kpanic, 0], sd1[kpanic, 0], mu1[kpanic, 1], sd1[kpanic, 1]]
ax.bar(np.arange(7) - 0.18, true_v, 0.36, color=GRAY, label="真值")
ax.bar(np.arange(7) + 0.18, est_v, 0.36, color=PURPLE, label="EM 估计")
ax.set_xticks(np.arange(7)); ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0, color=BLACK, lw=0.8)
ax.set_title("E2b 恐慌态参数恢复：权重/均值/方差逐一对上手")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_em_convergence.png", bbox_inches="tight"); plt.close(fig)
R["E2"] = {"panic_mu": [round(float(mu1[kpanic, 0]), 3), round(float(TRUE_MU[2, 0]), 3)], "panic_sd_v": [round(float(sd1[kpanic, 1]), 3), round(float(TRUE_SD[2, 1]), 3)], "panic_w": [round(float(w1[kpanic]), 3), round(float(TRUE_W[2]), 3)]}
print("fig2", R["E2"])

# ---------- E3 BIC 扫描
def fit_best_ll(X, K, seeds=(1, 2, 3)):
    best = -np.inf; bw = bmu = bsd = None
    for sd0 in [0.1, 0.5, 2.0]:
        for s in seeds:
            w, mu, sd, g, lls = em(X, K, seed=s * 10 + int(sd0 * 7), iters=250)
            # 固定初始（em 内部随机选初始点，这里近似即可）
            if lls[-1] > best:
                best = lls[-1]; bw, bmu, bsd = w, mu, sd
    return best
Ks = np.arange(1, 7)
bic_l = []; ll_list = []
x3, s3 = sample(1500)
for K in Ks:
    ll = fit_best_ll(x3, K)
    p = 3 * K - 1 + 2 * K + 2 * K  # (w:K-1, mu:2K, sd:2K) 但 3K-1+2K+2K = 7K-1 式不对，逐项：
    p = (K - 1) + 2 * K + 2 * K
    ll_list.append(ll); bic_l.append(2 * abs(np.nan_to_num(ll, nan=-1e9)) + p * np.log(len(x3)) if not np.isfinite(ll) else -2 * ll + p * np.log(len(x3)))
bic_arr = np.array(bic_l)
k_bic = int(Ks[np.argmin(np.where(np.isfinite(bic_arr), bic_arr, np.inf))])
R["E3"] = {"bic": [round(float(b), 1) for b in bic_arr], "k_selected": k_bic, "k_true": 3}
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.plot(Ks, np.where(np.isfinite(bic_arr), bic_arr, np.nan), "o-", color=BLUE, lw=2, ms=8)
ax.axvline(3, color=GRAY, ls=":", lw=1.2)
ax.annotate("真 K=3", (3, np.nanmin(bic_arr) * 0.6), color=GRAY, fontsize=10)
bb = Ks[np.argmin(np.where(np.isfinite(bic_arr), bic_arr, np.inf))]
ax.annotate(f"BIC 选出 K={bb}", (bb, np.nanmin(bic_arr)), textcoords="offset points", xytext=(20, 30), color=RED, fontsize=11, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("分量数 K"); ax.set_ylabel("BIC = −2·LL + p·ln n（越低越好）")
ax.set_xticks(Ks)
ax.set_title("E3 该分几个态？BIC 惩罚复杂度替你做决定")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_bic_selection.png", bbox_inches="tight"); plt.close(fig)
# ---------- E4 时序 regime：Markov 链生成 + 软后验 + 前瞻波动
T = 1200
P = np.array([[0.975, 0.022, 0.003], [0.05, 0.92, 0.03], [0.08, 0.15, 0.77]])  # 转移矩阵
st = np.zeros(T, dtype=int); st[0] = 0
for t in range(1, T):
    st[t] = rng.choice(3, p=P[st[t - 1]])
r_t = TRUE_MU[st, 0] / 100 + rng.normal(0, TRUE_SD[st, 0] / 100, T)
v_t = TRUE_MU[st, 1] + rng.normal(0, TRUE_SD[st, 1], T)
X4 = np.column_stack([r_t * 100, v_t])
w4, mu4, sd4, g4, _ = em(X4, 3, seed=11)
# 对齐标签到真 regime：用混淆矩阵最优排列（3!=6）
lab4 = g4.argmax(1)
# 直接按 v 中心排序映射：分量顺序 -> regime 身份 = argsort(mu4[:,1]) 反序
mapk = np.argsort(np.argsort(mu4[:, 1]))  # k -> true regime id（v 中心最小=平静 0，最大=恐慌 2）
lab4_aligned = mapk[lab4]
agree = float(np.mean(lab4_aligned == st))
fut_vol = v_t[1:]
grp = {}
for k in range(3):
    m = (lab4_aligned[:-1] == k)
    grp[k] = (float(fut_vol[m].mean()), float(np.percentile(fut_vol[m], 90)) if m.sum() > 10 else np.nan, int(m.sum()))
# 简单 regime 择时：恐慌态后验 >0.5 空仓，否则持有
panic_k = int(np.argmax(mu4[:, 1]))
pos = (g4[:, panic_k] < 0.5).astype(float)
ret_strat = r_t[1:] * pos[:-1]
sharpe_bh = float(r_t[1:].mean() / r_t[1:].std() * np.sqrt(252))
sharpe_st = float(ret_strat.mean() / (ret_strat.std() + 1e-12) * np.sqrt(252))
maxdd_bh = float(np.min(np.cumprod(1 + r_t[1:]) / np.maximum.accumulate(np.cumprod(1 + r_t[1])) - 1))
maxdd_st = float(np.min(np.cumprod(1 + ret_strat) / np.maximum.accumulate(np.cumprod(1 + ret_strat)) - 1))
R["E4"] = {"agree": round(agree, 4), "fut_vol_by_regime": {str(k): [round(g[0], 2), round(g[1], 2) if np.isfinite(g[1]) else None, g[2]] for k, g in grp.items()}, "sharpe_bh": round(sharpe_bh, 2), "sharpe_regime": round(sharpe_st, 2), "maxdd_bh": round(maxdd_bh * 100, 1), "maxdd_regime": round(maxdd_st * 100, 1)}
fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), height_ratios=[2.2, 1], sharex=False)
ax = axes[0]
ax.plot(np.cumprod(1 + r_t), color=BLACK, lw=1.1)
for k in range(3):
    ax.fill_between(np.arange(T), 0, np.cumprod(1 + r_t), where=(lab4_aligned == k), alpha=0.28, color=cols[k], step="mid")
ax.set_yscale("log")
ax.set_ylabel("净值（log）")
ax.set_title(f"E4a EM 事后分态（底色=软标签）：与真 regime 一致率 {agree*100:.1f}%，恐慌态次日 v 均值 {grp[2][0]:.2f}% vs 平静态 {grp[0][0]:.2f}%；恐慌态空仓 Sharpe {sharpe_bh:.2f}→{sharpe_st:.2f}")
ax = axes[1]
for kk in range(3):
    ax.plot(g4[:, kk], lw=1.0, color=[GREEN, BLUE, RED][mapk[kk]], label=f"成分 {kk} 后验")
ax.axhline(0.5, color=GRAY, ls="--", lw=1)
ax.set_xlabel("交易日"); ax.set_ylabel("责任度 γ（行归一）")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_regime_timeline.png", bbox_inches="tight"); plt.close(fig)
fig, ax = plt.subplots(figsize=(8.0, 4.6))
ks = sorted(grp, key=lambda k: grp[k][0])
names = {ks[0]: "平静态", ks[1]: "常态", ks[2]: "恐慌态"}
ax.bar(range(3), [grp[k][0] for k in ks], 0.5, color=[GREEN, BLUE, RED], alpha=0.85)
ax.errorbar(range(3), [grp[k][0] for k in ks], yerr=[[0, 0, 0], [grp[k][1] - grp[k][0] if grp[k][1] else 0 for k in ks]], fmt="none", ecolor=BLACK, capsize=5)
for i, k in enumerate(ks):
    ax.text(i, grp[k][1] + 0.1 if grp[k][1] else grp[k][0] + 0.1, f"P90={grp[k][1]:.2f}\nn={grp[k][2]}", ha="center", fontsize=8)
    ax.text(i, -0.25, f"{grp[k][0]:.2f}", ha="center", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels([names[k] for k in ks])
ax.set_ylabel("次日波动代理 v 均值 (%)")
ax.set_yticks([])
ax.set_title(f"E4b 分态的前瞻波动：恐慌态次日 v 均值 {grp[ks[2]][0]:.2f}% vs 平静态 {grp[ks[0]][0]:.2f}%\n恐慌态空仓规则：Sharpe {sharpe_bh:.2f}→{sharpe_st:.2f}，最大回撤 {maxdd_bh*100:.1f}%→{maxdd_st*100:.1f}%")
ax_ylim_fix = ax.set_ylim(-0.5, max(g[1] or 1 for g in grp.values()) + 0.8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_forward_vol.png", bbox_inches="tight"); plt.close(fig)
print("fig4/6", R["E4"])

# ---------- E5 分量塌缩与正则化
# 初始故意让一个分量贴着某个最密点、σ 极小
def em_track(X, K, mu0, sd0, sd_floor, iters=120):
    mu = mu0.copy(); sd = sd0.copy(); w = np.full(K, 1 / K); lls = []; sds = []
    for it in range(iters):
        sd = np.maximum(sd, sd_floor)
        z = -0.5 * ((X[:, None] - mu[None]) / sd[None]) ** 2 - np.log(sd[None]) - np.log(2 * np.pi)
        lz = z.sum(2) + np.log(w)[None]
        lse = logsumexp(lz, axis=1); ll = lse.sum(); lls.append(ll)
        sds.append(sd.min())
        g = np.exp(lz - lse[:, None]); Nk = g.sum(0) + 1e-12
        w = Nk / len(X); mu = (g[:, :, None] * X[:, None, :]).sum(0) / Nk[:, None]
        sd = np.sqrt((g[:, :, None] * (X[:, None, :] - mu[None]) ** 2).sum(0) / Nk[:, None])
    return np.array(lls), np.array(sds)
dense_pt = x1[np.lexsort((x1[:, 0], x1[:, 1]))[len(x1) // 2]]
mu0 = np.vstack([TRUE_MU[:2], dense_pt]); sd0 = np.vstack([TRUE_SD[:2], [[1e-3, 1e-3]]])
lls_raw, sds_raw = em_track(x1, 3, mu0, sd0, 1e-9)
lls_reg, sds_reg = em_track(x1, 3, mu0, sd0, 0.05)
R["E5"] = {"ll_raw_final": round(float(lls_raw[-1]), 1), "ll_reg_final": round(float(lls_reg[-1]), 1), "sd_min_raw": float(f"{sds_raw[-1]:.2e}"), "growth_10x": bool(lls_raw[-1] > lls_raw[5] + 50)}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.plot(lls_raw, color=RED, lw=1.8, label=f"无下限: LL={lls_raw[-1]:.0f}（每迭代 +{np.diff(lls_raw).mean():.1f}，仍在涨）")
ax.plot(lls_reg, color=GREEN, lw=1.8, label=f"σ 下限 0.05: LL={lls_reg[-1]:.0f}")
ax.set_xlabel("EM 迭代"); ax.set_ylabel("对数似然"); ax.set_xscale("log")
ax.set_title("E5a 一个分量贴着最密点出生：无正则化 LL 走向 +∞")
ax.legend(fontsize=8)
ax = axes[1]
ax.semilogy(sds_raw, color=RED, lw=1.8, label="最小 σ：无下限 → 1e-9 地板")
ax.semilogy(sds_reg, color=GREEN, lw=1.8, label="最小 σ：下限 0.05 兜住")
ax.set_xlabel("EM 迭代"); ax.set_ylabel("全体分量最小 σ（log）"); ax.set_xscale("log")
ax.set_title("E5b 塌缩的体检指标：最小 σ 是否一路归零")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_degenerate.png", bbox_inches="tight"); plt.close(fig)
print("fig5", R["E5"])

with open("/Users/halo123456/workspace/astro-blog/_gmm_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("DONE")
