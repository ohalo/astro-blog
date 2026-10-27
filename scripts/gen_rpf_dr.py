#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26e-A: 随机投影与压缩感知因子降维 配图 + 核心统计
DGP: N=400 只股票, M=6 个潜在因子 F, K=50 个可观测因子 = F 的线性组合 + 噪声。
图1: JL 引理实证——成对距离畸变 vs 投影维度 d，对比理论充分维度 d*。
图2: 压缩感知相变——信号仅依赖 k=5 个因子时 Lasso(手写坐标下降) 重建误差 vs d。
图3: Lasso 稀疏暴露恢复——真实关键因子 vs 系数 Top-k 命中。
图4: 邻域结构保持——投影前后成对距离散点 + Top-10 近邻保持率。
图5: 组合回测——全50因子 / JL d=15 / PCA6 / 前视真6因子 Ridge 选股。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti"]:
    plt.rcParams["font.family"] = [f]
    break
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/random-projection-factor-reduction"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE, TEAL = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7", "#00cec9"

STATS = {}
N, K, M = 400, 50, 6
rng0 = np.random.default_rng(42)

F = rng0.standard_normal((N, M))
W_true = np.zeros((M, K))
for g in range(K):
    idx = rng0.choice(M, size=3, replace=False)
    W_true[idx, g] = rng0.normal(0, 1, 3)
NOISE_SD = 0.45
X_obs = F @ W_true + rng0.normal(0, NOISE_SD, (N, K))
X_obs /= X_obs.std(axis=0)
Xs = X_obs / X_obs.std(axis=0)

# ================= 图1: JL 畸变 vs 投影维度 =================
eps_t = 0.20
d_th = int(np.ceil(8 * np.log(N) / (eps_t * eps_t - eps_t ** 3 / 3)))
d_list = [8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120]
maxdist, mindist, frac_band = [], [], []
med_r, p05, p95 = [], [], []
Xn_g = X_obs / np.linalg.norm(X_obs, axis=1, keepdims=True)
DX_g = 1 - Xn_g @ Xn_g.T
iu_g = np.triu_indices(N, 1)
dx_g = DX_g[iu_g]
for d in d_list:
    rng = np.random.default_rng(7)
    Rp = rng.standard_normal((K, d)) / np.sqrt(d)
    Y = X_obs @ Rp
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    DY = 1 - Yn @ Yn.T
    dy = DY[iu_g]
    ratio = np.clip((dy + 1e-12) / (dx_g + 1e-12), 0, 3)
    maxdist.append(float(ratio.max())); mindist.append(float(ratio.min()))
    med_r.append(float(np.median(ratio)))
    p05.append(float(np.percentile(ratio, 5))); p95.append(float(np.percentile(ratio, 95)))
    frac_band.append(float(np.mean((ratio >= 1 - eps_t) & (ratio <= 1 + eps_t))))
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(d_list, p95, "o-", color=BLUE, label="距离比 P95")
ax.plot(d_list, p05, "s-", color=TEAL, label="距离比 P5")
ax.plot(d_list, med_r, "d-", color=GREEN, ms=4, label="距离比中位数")
ax.axhline(1 + eps_t, color=RED, ls="--", lw=1.0, alpha=.7)
ax.axhline(1 - eps_t, color=RED, ls="--", lw=1.0, alpha=.7, label="±ε = ±20% 带")
ax.plot(d_list, maxdist, ":", color=BLUE, alpha=.4, label="全配对 max/min")
ax.plot(d_list, mindist, ":", color=TEAL, alpha=.4)
ax.set_xlabel("投影后维度 d"); ax.set_ylabel("成对距离畸变比 d′/d")
ax.set_ylim(-0.1, 2.6)
d_ok = next((d for d, hi, lo in zip(d_list, p95, p05) if hi <= 1 + eps_t and lo >= 1 - eps_t), None)
if d_ok:
    ax.axvline(d_ok, color=PURPLE, ls=":", lw=1.4)
    ax.annotate(f"P5–P95 全入带\nd={d_ok}", xy=(d_ok * 0.6, 1.75), fontsize=8, color=PURPLE)
ax.axvline(d_th, color=ORANGE, ls="--", lw=1.5)
ax.annotate(f"JL 最坏情形上界 d*={d_th}\n远大于原始维度 50 (极端保守)", xy=(70, 0.35), fontsize=8, color=ORANGE, ha="center")
ax.set_title(f"JL 引理实证：{N} 只股票 {K} 维因子随机投影的成对距离畸变")
ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout(); fig.savefig(f"{OUT}/jl-distortion.png"); plt.close(fig)
STATS["jl_d_star"] = d_th
STATS["frac_band_d16"] = frac_band[d_list.index(16)]
STATS["frac_band_d25"] = frac_band[d_list.index(25)]
STATS["p95_d16"] = p95[d_list.index(16)]
STATS["p05_d16"] = p05[d_list.index(16)]
STATS["d_pband_ok"] = d_ok
STATS["med_d25"] = med_r[d_list.index(25)]

# ================= 图2: 压缩感知相变 (样本复杂度) =================
k_sparse = 5
sig_cols = rng0.choice(K, size=k_sparse, replace=False)
w_sig = rng0.normal(0, 2, k_sparse)
w50 = np.zeros(K); w50[sig_cols] = w_sig
NOISE_CS = 0.3
# 相变实验用独立良好性态设计矩阵 (iid 高斯列, 近似满足 RIP);
# 真实因子阵 Xs 高度相干, 稀疏恢复另见图3
rng_cs = np.random.default_rng(777)
Xcs = rng_cs.standard_normal((N, K))
Xcs = (Xcs - Xcs.mean(0)) / Xcs.std(0)
y_cs = Xcs @ w50 + rng_cs.normal(0, NOISE_CS, N)
y_rms = float(np.sqrt(np.mean(y_cs ** 2)))

def lasso_cd(Xm, yv, lam, n_iter=300):
    p = Xm.shape[1]
    beta = np.zeros(p); r = yv - Xm @ beta
    norms = (Xm ** 2).sum(axis=0)
    for _ in range(n_iter):
        for j in range(p):
            rho = Xm[:, j] @ r + norms[j] * beta[j]
            beta[j] = np.sign(rho) * max(abs(rho) - lam, 0) / norms[j]
            r = yv - Xm @ beta
    return beta

m_scan = [8, 12, 16, 20, 25, 30, 35, 40, 50, 60, 80, 120, 200, 400]
# 外部测试集: 与训练同分布的 100 只新股 (m=400 时留出法会退化)
Xte = rng_cs.standard_normal((100, K)); Xte = (Xte - Xte.mean(0)) / Xte.std(0)
yte = Xte @ w50 + rng_cs.normal(0, NOISE_CS, 100)
re_lasso_m, re_ols_m, supp_m = [], [], []
for m in m_scan:
    errs_l, errs_o, hits = [], [], []
    for s in range(10):
        rng = np.random.default_rng(500 + s)
        perm = rng.permutation(N)
        itr = perm[:m]
        Xt, yt = Xcs[itr], y_cs[itr]
        lam = 0.55 * NOISE_CS * np.sqrt(2 * np.log(K)) * np.sqrt(m)
        b = lasso_cd(Xt, yt, lam, n_iter=400)
        if not np.all(np.isfinite(b)):
            b = np.zeros(K)
        errs_l.append(np.linalg.norm(Xte @ b - yte) / np.linalg.norm(yte))
        bo = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        errs_o.append(np.linalg.norm(Xte @ bo - yte) / np.linalg.norm(yte))
        top = np.argsort(-np.abs(b))[:k_sparse]
        hits.append(len(set(top) & set(sig_cols)) / k_sparse)
    re_lasso_m.append(float(np.mean(errs_l))); re_ols_m.append(float(np.mean(errs_o)))
    supp_m.append(float(np.mean(hits)))
m_theory = 2 * k_sparse * np.log(K / k_sparse)
noise_floor = float(np.linalg.norm(yte - Xte @ w50) / np.linalg.norm(yte))
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(m_scan, re_lasso_m, "o-", color=PURPLE, label="Lasso 留出股票相对误差")
ax.plot(m_scan, re_ols_m, "s-", color=GRAY, label="OLS 留出股票相对误差")
ax.axvline(m_theory, color=ORANGE, ls=":", lw=1.5, label=f"CS 样本量下界 2k·ln(K/k) = {m_theory:.1f}")
ax.axvline(K, color=BLACK, ls="--", lw=1, alpha=.5, label="可观测因子数 K=50")
ax.axhline(noise_floor, color=PURPLE, ls=":", lw=1, alpha=.4, label=f"噪声地板 {noise_floor:.3f}")
ax.set_xlabel("用于拟合的股票数 m"); ax.set_ylabel("留出股票预测相对误差")
ax.set_title("压缩感知相变：信号仅依赖 5/50 个因子时 Lasso 的样本复杂度骤变")
ax.legend(fontsize=7.5); ax.set_yscale("log")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(m_scan, supp_m, "^--", color=BLUE, alpha=.7, label="支撑度命中率")
ax2.set_ylabel("Lasso Top-5 命中真实因子比例", color=BLUE); ax2.set_ylim(0, 1.15)
l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/cs-phase-transition.png"); plt.close(fig)
STATS["m_theory"] = float(m_theory)
STATS["noise_floor"] = float(noise_floor)
for m in (16, 25, 35, 50, 80, 200, 400):
    STATS[f"lasso_m{m}"] = re_lasso_m[m_scan.index(m)]
for m in (16, 25, 50, 80, 120):
    STATS[f"ols_m{m}"] = re_ols_m[m_scan.index(m)]
STATS["supp_m25"] = supp_m[m_scan.index(25)]
STATS["supp_m35"] = supp_m[m_scan.index(35)]
STATS["supp_m50"] = supp_m[m_scan.index(50)]
m_sfull = next((m for m, sh in zip(m_scan, supp_m) if sh >= 0.95), None)
STATS["m_support_full"] = m_sfull

# ================= 图3: 稀疏暴露恢复 (相干真实因子阵上) =================
rng_b3 = np.random.default_rng(313)
y_b3 = Xs @ w50 + rng_b3.normal(0, NOISE_CS, N)
b_full = lasso_cd(Xs, y_b3, 0.6)
topk = np.argsort(-np.abs(b_full))[:k_sparse]
support_hit = len(set(topk) & set(sig_cols))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(np.arange(K) - 0.2, np.where(np.isin(np.arange(K), sig_cols), 1, 0), width=0.4,
       color=RED, alpha=.55, label="真实暴露(5 个关键因子)")
ax.bar(np.arange(K) + 0.2, b_full / np.abs(b_full).max(), width=0.4,
       color=BLUE, alpha=.8, label="Lasso 归一化系数")
ax.set_xlabel("因子编号"); ax.set_ylabel("归一化幅度")
ax.set_title(f"Lasso 稀疏暴露恢复：Top-{k_sparse} 命中 {support_hit}/{k_sparse} 个真实因子 (K=50 全维)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/sparse-recovery.png"); plt.close(fig)
STATS["support_hit"] = int(support_hit); STATS["hit_list"] = sorted(int(x) for x in sig_cols)
STATS["full_coef_nonzero"] = int((np.abs(b_full) > 1e-8).sum())
STATS["true_weights"] = [round(float(w), 3) for w in w_sig]

# ================= 图4: 邻域结构保持 =================
rngn = np.random.default_rng(555)
d4 = 15
Rp4 = rngn.standard_normal((K, d4)) / np.sqrt(d4)
Y4 = Xs @ Rp4
Xn4 = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
Yn4 = Y4 / np.linalg.norm(Y4, axis=1, keepdims=True)
Sx = 1 - Xn4 @ Xn4.T; Sy = 1 - Yn4 @ Yn4.T
iu = np.triu_indices(N, 1)
sub = rngn.choice(len(iu[0]), 6000, replace=False)
cd = float(np.corrcoef(Sx[iu][sub], Sy[iu][sub])[0, 1])

def knn_set(Sim, k=10):
    out = set()
    for i in range(N):
        s = Sim[i].copy(); s[i] = -1e9
        for j in np.argsort(-s)[:k]:
            out.add((i, int(j)))
    return out

orig = knn_set(Xn4 @ Xn4.T); proj = knn_set(Yn4 @ Yn4.T)
keep = len(orig & proj) / len(orig)
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.scatter(Sx[iu][sub], Sy[iu][sub], s=2, alpha=.15, color=BLUE)
ax.plot([0, 2], [0, 2], color=RED, lw=1, ls="--", alpha=.6)
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xlabel("原始 50 维成对距离"); ax.set_ylabel(f"投影 {d4} 维后成对距离")
ax.set_title(f"邻域结构保持 (d=15)：距离比相关 r={cd:.3f}，Top-10 近邻保持率 {keep:.1%}")
fig.tight_layout(); fig.savefig(f"{OUT}/neighborhood.png"); plt.close(fig)
STATS["pair_corr"] = cd; STATS["knn_keep"] = float(keep)

# ================= 图5: 组合回测 =================
# DGP: 全局稀疏系数 theta (50 维仅 5 个非零) -> 截面漂移 mu_i = x_i·theta
# 日收益 = mu_i(漂移) + 市场共性冲击 + 个股噪声; 每 10 日调仓, 用过去 120 日
# 实现收益重新拟合打分, 收益曲线拼接(无前视)。
T = 500
rngb = np.random.default_rng(99)
theta = np.zeros(K)
tcols = rngb.choice(K, size=5, replace=False)
theta[tcols] = rngb.normal(0, 1, 5)
mu = Xs @ theta
mu = mu / mu.std() * 0.0011          # 截面漂移 std 0.11%/日
market = rngb.normal(0, 0.010, T)
idio = rngb.normal(0, 0.018, (T, N))
ret_mat = mu[None, :] + market[:, None] + idio

HOLD, LOOK = 10, 120
rjl = np.random.default_rng(4242).standard_normal((K, 15)) / np.sqrt(15)
_, _, Vt = np.linalg.svd(X_obs - X_obs.mean(0), full_matrices=False)
V6 = Vt[:6].T
Zjl = Xs @ rjl

def ridge_pred(Xtr_fea, ytr, lam):
    b = np.linalg.solve(Xtr_fea.T @ Xtr_fea + lam * np.eye(Xtr_fea.shape[1]), Xtr_fea.T @ ytr)
    return Xtr_fea @ b

reb_days = list(range(LOOK, T - HOLD + 1, HOLD))
scores = {"full50": [], "jl15": [], "lasso50": [], "oracle": []}
for t in reb_days:
    ytr = ret_mat[t - LOOK:t].mean(axis=0)      # 过去120日平均收益(去均值)
    ytr = ytr - ytr.mean()
    scores["full50"].append(ridge_pred(Xs, ytr, 40.0))
    scores["jl15"].append(ridge_pred(Zjl, ytr, 40.0 / np.sqrt(15) * np.sqrt(K) * 0.6))
    bL = lasso_cd(Xs, ytr, 0.35 * np.abs(Xs.T @ ytr).max())
    scores["lasso50"].append(Xs @ bL)
    scores["oracle"].append(mu)

def pooled(sc_key):
    segs = []
    for i, t in enumerate(reb_days):
        pick = np.argsort(-scores[sc_key][i])[:40]
        segs.append(ret_mat[t:t + HOLD][:, pick].mean(axis=1))
    c = np.concatenate(segs)
    return float(c.mean() / c.std() * np.sqrt(252)), c

def ic_pooled(sc_key):
    ics = []
    for i, t in enumerate(reb_days):
        fut = ret_mat[t:t + HOLD].mean(axis=0)
        s = scores[sc_key][i]
        r1 = np.argsort(np.argsort(s)); r2 = np.argsort(np.argsort(fut))
        ics.append(np.corrcoef(r1, r2)[0, 1])
    return float(np.mean(ics)), np.array(ics)

keys = ["full50", "jl15", "lasso50", "oracle"]
names = {"full50": "全 50 因子 Ridge", "jl15": "JL 投影 d=15 Ridge",
         "lasso50": "全维 Lasso(稀疏)", "oracle": "前视真 mu(oracle)"}
cols = {"full50": GRAY, "jl15": PURPLE, "lasso50": BLUE, "oracle": GREEN}
shs, ics_all, curvs = {}, {}, {}
for kname in keys:
    shs[kname], curvs[kname] = pooled(kname)
    ic_all = ic_pooled(kname)
    ics_all[kname] = ic_all[0]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for kname in keys:
    axes[0].plot(np.cumprod(1 + curvs[kname]), color=cols[kname], lw=1.3,
                 label=f"{names[kname]} SR={shs[kname]:.2f}")
axes[0].legend(fontsize=7.5); axes[0].set_title(f"Top-40 组合拼接净值 ({len(reb_days)} 个调仓期)")
axes[0].set_xlabel("交易日")
xv = np.arange(len(keys))
axes[1].bar(xv, [ics_all[k] for k in keys], color=[cols[k] for k in keys], alpha=.85)
for i, kname in enumerate(keys):
    axes[1].text(i, ics_all[kname] + 0.003, f"{ics_all[kname]:.3f}", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks(xv); axes[1].set_xticklabels([names[k].replace(" ", "\n", 1) for k in keys], fontsize=7.5)
axes[1].set_ylabel("打分与未来 10 日收益的截面秩相关")
axes[1].set_title("选股 10 日 IC")
fig.suptitle("JL 投影 d=15 与全 50 因子的选股表现对比 (theta 真值仅 5/50 非零)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/portfolio.png"); plt.close(fig)
STATS["sharpe"] = {k: shs[k] for k in keys}
STATS["ic"] = {k: ics_all[k] for k in keys}
STATS["mu_top40_capture"] = float(np.mean(
    [len(set(np.argsort(-scores["jl15"][i])[:40]) & set(np.argsort(-mu)[:40])) / 40
     for i in range(0, len(reb_days), 3)]))
STATS["theta_cols"] = sorted(int(x) for x in tcols)
STATS["n_reb"] = len(reb_days)
STATS["hold_days"] = sum(HOLD for _ in reb_days)

with open("/Users/halo123456/workspace/astro-blog/_rpf_stats.json", "w") as fh:
    json.dump(STATS, fh, indent=1, ensure_ascii=False)
print(json.dumps(STATS, indent=1, ensure_ascii=False))
print("images:", sorted(os.listdir(OUT)))
