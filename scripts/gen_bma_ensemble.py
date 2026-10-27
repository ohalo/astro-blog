#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""贝叶斯模型平均预测集成（BMA）
E1 模型空间全景：256 个候选子模型后验，真模型 vs 冠军
E2 累计后验质量 + 有效模型数 + 模型族碎片化
E3 PIP 60 种子 vs 全模型 OLS t：孪生因子对拍
E4 OOS 对决 60 种子：样本内冠军 / 5折CV选模 / BMA / 等权Top-64 / oracle
E5 g 先验敏感性 + 尺寸先验（40 种子均值）
g-prior(Zellner)+Jeffreys(σ²) 闭式：log p(D|M) = -(k/2)ln(1+g) -(n/2)ln(SST-(g/(1+g))·ESS)
输出 5 张图 + _bma_results.json，seed=20261027
"""
import os, json, math
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

OUT = "/Users/halo123456/workspace/astro-blog/public/images/bayesian-model-averaging-ensemble"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE, TEAL = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7", "#00cec9"
R = {}

FNAMES = ["动量", "动量孪生", "价值", "波动", "波动孪生", "噪声1", "噪声2", "噪声3"]
K = 8
TRUE_SET = (0, 2, 4)
BETA_TRUE = np.array([0.35, 0.22, 0.28])
SIG_E = 1.15
NTOT = 300
NTR = 240
GS = np.array([0.1, 1, 5, 30, 150, 300, 1200, 12000, 2 ** 26], dtype=float)

SUBS = [()] + [tuple(j for j in range(K) if (m >> j) & 1) for m in range(1, 1 << K)]
SUB_ARR = [np.array(list(S)) if S else np.array([], dtype=int) for S in SUBS]
KS = np.array([len(S) for S in SUBS])
SIZ_PRIOR_LOG = np.array([-math.log(math.comb(K, k)) for k in KS])
sub_lbl = ["＋".join(FNAMES[j] for j in S) if S else "空模型" for S in SUBS]
true_idx = SUBS.index(TRUE_SET)

def make_data(n, rng):
    f = rng.standard_normal((n, K))
    f[:, 1] = 0.90 * f[:, 0] + np.sqrt(1 - 0.81) * rng.standard_normal(n)
    f[:, 3] = 0.80 * f[:, 4] + np.sqrt(1 - 0.64) * rng.standard_normal(n)
    f[:, 5:] = rng.standard_normal((n, 3))
    y = f[:, list(TRUE_SET)] @ BETA_TRUE + SIG_E * rng.standard_normal(n)
    return f, y

def fit_all(f, y):
    """每个子模型一次 OLS：返回 (RSS[M], Beta[M](长度K，未用位置置0), intercept[M])"""
    n = len(y)
    M = len(SUBS)
    rss = np.zeros(M); betas = np.zeros((M, K)); ints = np.zeros(M)
    for i, arr in enumerate(SUB_ARR):
        Xd = np.column_stack([np.ones(n), f[:, arr]]) if len(arr) else np.ones((n, 1))
        b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ b
        rss[i] = resid @ resid
        ints[i] = b[0]
        betas[i, arr] = b[1:]
    return rss, betas, ints

def weights_from_rss(rss, y, g, size_prior=False):
    SST = float(((y - y.mean()) ** 2).sum())
    ess = SST - rss
    lml = -0.5 * KS * np.log1p(g) - 0.5 * len(y) * np.log(np.maximum(SST - (g / (1 + g)) * ess, 1e-12))
    if size_prior:
        lml = lml + SIZ_PRIOR_LOG
    w = np.exp(lml - lml.max()); w /= w.sum()
    return w

def pred_matrix(ftr, ytr, fte, betas, ints):
    return ints[:, None] + betas @ fte.T

def effn(w):
    p = w[w > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))

def pip_from_w(w, nseed_list=None):
    pip = np.zeros(K)
    nz = np.nonzero(w > 1e-12)[0]
    for i in nz:
        for j in SUBS[i]:
            pip[j] += w[i]
    return pip

def t_stats_full(f, y):
    n = len(y)
    Xd = np.column_stack([np.ones(n), f])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ b
    rss = resid @ resid
    se = np.sqrt(np.diag(np.linalg.inv(Xd.T @ Xd)) * rss / (n - K - 1))
    return b[1:] / se[1:]

# ---------------- fig1 + fig2：固定演示种子 ----------------
rng = np.random.default_rng(20261027)
fd, yd = make_data(NTOT, rng)
rss_d, betas_d, ints_d = fit_all(fd[:NTR], yd[:NTR])
w_d = weights_from_rss(rss_d, yd[:NTR], NTOT)
t1 = int(np.argmax(w_d))
R["demo"] = {"top1_model": sub_lbl[t1], "top1_post": float(w_d[t1]),
             "true_model": sub_lbl[true_idx], "true_post": float(w_d[true_idx]),
             "true_is_top1": bool(t1 == true_idx), "effN": effn(w_d)}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(yd[:NTR], lw=1.0, color=BLACK, alpha=.85, label="y（日收益%）")
ax[0].plot(fd[:NTR, 0], lw=.8, color=BLUE, alpha=.6, label="动量 f1")
ax[0].plot(fd[:NTR, 4], lw=.8, color=ORANGE, alpha=.6, label="波动 f5")
ax[0].axvline(NTR, color=RED, ls="--", lw=1, label="训练/测试分割")
ax[0].legend(fontsize=8)
ax[0].set_title("(a) 合成市场：y=0.35动量+0.22价值+0.28波动+噪声\n候选里混着孪生 f2≈0.9f1、f4≈0.8f5 和三个纯噪声")
order = np.argsort(-w_d)[:12]
ax[1].bar(range(12), w_d[order], color=[RED if i == true_idx else GRAY for i in order])
ax[1].set_xticks(range(12))
ax[1].set_xticklabels([sub_lbl[i].replace("＋", "+") for i in order], rotation=38, ha="right", fontsize=7)
for i in order:
    if i == true_idx:
        ax[1].annotate(f"真模型\n{w_d[i]:.3f}", xy=(list(order).index(i), w_d[i] + .004), fontsize=8, color=RED, ha="center")
ax[1].set_title(f"(b) 256 子模型后验 Top-12：冠军 {sub_lbl[t1]}（{w_d[t1]:.3f}）\n真模型 {sub_lbl[true_idx]} 只拿到 {w_d[true_idx]:.3f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_model_landscape.png"); plt.close(fig)

sw = np.sort(w_d)[::-1]; cum = np.cumsum(sw)
npos = int(np.searchsorted(cum, 0.95) + 1)
fam = {}
for S, wi in zip(SUBS, w_d):
    momfam = (0 in S) + (1 in S); volfam = (3 in S) + (4 in S)
    if momfam >= 2 or volfam >= 2:
        c = "孪生父子同入(冗余)"
    elif momfam == 1 and volfam == 1 and (2 in S) and len(S) == 3:
        c = {TRUE_SET: "★真模型", (1, 2, 4): "动量孪生版", (0, 2, 3): "波动孪生版", (1, 2, 3): "双孪生版"}.get(tuple(S), "同构族")
    elif len(S) == 3 and (2 not in S):
        c = "漏掉价值因子"
    elif len(S) <= 2:
        c = "欠规格(≤2因子)"
    else:
        c = "其它"
    fam[c] = fam.get(c, 0.0) + float(wi)
R["cum"] = {"models_for_95pct": npos, "effN": effn(w_d),
            "family_mass": {k: round(v, 4) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(range(1, len(SUBS) + 1), cum, lw=1.6, color=BLUE)
ax[0].axhline(.95, color=RED, ls="--", lw=1); ax[0].axvline(npos, color=RED, ls=":", lw=1)
ax[0].annotate(f"前 {npos}/256 个模型\n占 95% 后验质量", xy=(npos + 15, .5), fontsize=10, color=RED)
ax[0].set_xlabel("按后验降序的模型序号"); ax[0].set_ylabel("累计后验质量"); ax[0].set_xlim(1, 256)
ax[0].set_title(f"(a) 模型不确定性实拍：有效模型数 exp(H)={effn(w_d):.1f}\n『冠军』扛着的其实是一整个模型部落")
fk = sorted(fam.items(), key=lambda kv: -kv[1])[:7]
ax[1].bar(range(len(fk)), [v for _, v in fk], color=[RED, ORANGE, PURPLE, BLUE, TEAL, GREEN, GRAY][:len(fk)])
ax[1].set_xticks(range(len(fk)))
ax[1].set_xticklabels([k for k, _ in fk], rotation=22, ha="right", fontsize=8)
for i, (k, v) in enumerate(fk):
    ax[1].text(i, v + .004, f"{v:.3f}", ha="center", fontsize=8)
ax[1].set_title("(b) 模型族质量汇总：孪生把后验切成碎片\n同构族各拿各的，逐模型比较看不见全貌")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_cumulative_family.png"); plt.close(fig)
print("fig1/fig2 done", flush=True)

# ---------------- fig3 PIP 60 seeds ----------------
NSEED3 = 60
pip_mat = np.zeros((NSEED3, K)); t_full = np.zeros((NSEED3, 4))
for s in range(NSEED3):
    r = np.random.default_rng(1000 + s)
    f, y = make_data(NTOT, r)
    rss, _, _ = fit_all(f, y)
    w = weights_from_rss(rss, y, NTOT)
    nz = np.nonzero(w > 1e-12)[0]
    for i in nz:
        for j in SUBS[i]:
            pip_mat[s, j] += w[i]
    t = t_stats_full(f, y)
    t_full[s] = [t[0], t[1], t[3], t[4]]
pip_m = pip_mat.mean(0); pip_se = pip_mat.std(0, ddof=1) / np.sqrt(NSEED3)
sigA = np.abs(t_full[:, 0]) > 1.65; sigB = np.abs(t_full[:, 1]) > 1.65
sigC = np.abs(t_full[:, 2]) > 1.65; sigD = np.abs(t_full[:, 3]) > 1.65
both_mom = float(np.mean(sigA & sigB)); both_vol = float(np.mean(sigC & sigD))
one_mom = float(np.mean(sigA ^ sigB)); one_vol = float(np.mean(sigC ^ sigD))
R["pip"] = {"per_factor": {FNAMES[j]: {"mean": float(pip_m[j]), "se": float(pip_se[j])} for j in range(K)},
            "family_pip": {"动量族(动量+孪生)": float(pip_m[0] + pip_m[1]), "波动族(波动+孪生)": float(pip_m[3] + pip_m[4])},
            "ols_both_sig": {"mom_pair": both_mom, "vol_pair": both_vol},
            "ols_exactly_one_sig": {"mom_pair": one_mom, "vol_pair": one_vol}}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
cols = [RED if j in TRUE_SET else (ORANGE if j in (1, 3) else GRAY) for j in range(K)]
ax[0].bar(range(K), pip_m, color=cols, alpha=.85)
ax[0].errorbar(range(K), pip_m, yerr=1.96 * pip_se, fmt="none", ecolor=BLACK, capsize=3)
ax[0].axhline(.5, color=BLUE, ls="--", lw=1); ax[0].axhline(.9, color=RED, ls=":", lw=1)
ax[0].set_xticks(range(K)); ax[0].set_xticklabels(FNAMES, rotation=30, ha="right", fontsize=8)
for j in [0, 1, 3, 4]:
    ax[0].text(j, pip_m[j] + .05, f"{pip_m[j]:.2f}", ha="center", fontsize=8)
ax[0].set_title("(a) PIP（60 种子均值±95%%CI）：真因子>0.9、噪声<0.1；\n孪生单票≈0.5，但族质量 动量%.2f / 波动%.2f≈1" % (pip_m[0] + pip_m[1], pip_m[3] + pip_m[4]))
ax[1].boxplot([t_full[:, 0], t_full[:, 1], t_full[:, 2], t_full[:, 3]], tick_labels=["t(动量)", "t(动量孪生)", "t(波动)", "t(波动孪生)"])
ax[1].axhline(1.65, color=RED, ls="--", lw=1); ax[1].axhline(-1.65, color=RED, ls="--", lw=1); ax[1].axhline(0, color=GRAY, lw=.8)
ax[1].tick_params(labelsize=8)
ax[1].set_title(f"(b) 全模型 OLS |t|（10% 线虚线）：孪生同显著率\n动量对 {both_mom:.0%}/波动对 {both_vol:.0%}，恰好一进一出 {one_mom:.0%}/{one_vol:.0%}\n——显著性裁判在复制品之间掷硬币")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_pip_vs_ols.png"); plt.close(fig)
print("fig3 done", flush=True)

# ---------------- fig4 OOS 60 seeds ----------------
NSEED4 = 60
names4 = ["单模型冠军", "5折CV选模", "BMA(g=n)", "等权Top-64", "oracle真模型"]
mse4 = np.zeros((NSEED4, 5)); wrong_in = 0; wrong_cv = 0
for s in range(NSEED4):
    r = np.random.default_rng(5000 + s)
    f, y = make_data(NTOT, r)
    ftr, ytr, fte, yte = f[:NTR], y[:NTR], f[NTR:], y[NTR:]
    rss, betas, ints = fit_all(ftr, ytr)
    P = pred_matrix(ftr, ytr, fte, betas, ints)  # (257, 60)
    SSTr = float(((ytr - ytr.mean()) ** 2).sum())
    r2adj = np.empty(len(SUBS))
    for i in range(len(SUBS)):
        r2adj[i] = 1 - (rss[i] / (NTR - KS[i] - 1)) / SSTr
    # 5折时序 CV
    folds = np.array_split(np.arange(NTR), 5)
    cvm = np.zeros(len(SUBS))
    for kx in range(5):
        te_i = folds[kx]; tr_i = np.concatenate([folds[q] for q in range(5) if q != kx])
        rss_f, bet_f, int_f = fit_all(ftr[tr_i], ytr[tr_i])
        pred = int_f[:, None] + bet_f @ ftr[te_i].T
        cvm += ((pred - ytr[te_i][None, :]) ** 2).sum(1)
    cvm /= NTR
    b_in = int(np.argmax(r2adj)); b_cv = int(np.argmin(cvm))
    if SUBS[b_in] != TRUE_SET: wrong_in += 1
    if SUBS[b_cv] != TRUE_SET: wrong_cv += 1
    w = weights_from_rss(rss, ytr, NTOT)
    top64 = np.argsort(-r2adj)[1:65]
    mse4[s, 0] = float(((P[b_in] - yte) ** 2).mean())
    mse4[s, 1] = float(((P[b_cv] - yte) ** 2).mean())
    mse4[s, 2] = float(((w @ P - yte) ** 2).mean())
    mse4[s, 3] = float(((P[top64].mean(0) - yte) ** 2).mean())
    io = int(np.where(np.asarray([S == TRUE_SET for S in SUBS]))[0][0])
    mse4[s, 4] = float(((P[io] - yte) ** 2).mean())
R["oos"] = {"per_est": {names4[c]: {"median": float(np.median(mse4[:, c])), "p90": float(np.percentile(mse4[:, c], 90)),
                                    "mean": float(mse4[:, c].mean())} for c in range(5)},
            "single_best_wrong_pct": wrong_in / NSEED4, "cv_wrong_pct": wrong_cv / NSEED4,
            "single_over_bma": float(np.median(mse4[:, 0]) / np.median(mse4[:, 2]))}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].boxplot([mse4[:, c] for c in range(5)], tick_labels=names4)
for i in range(5):
    ax[0].scatter([i + 1] * NSEED4, mse4[:, i], s=9, alpha=.35, color=[RED, ORANGE, BLUE, PURPLE, GREEN][i])
ax[0].tick_params(labelsize=8); ax[0].set_ylabel("60 天样本外 MSE")
ax[0].set_title(f"(a) 60 种子 OOS：样本内冠军 {R['oos']['single_best_wrong_pct']:.0%} 的时间选错模型\nCV 也有 {R['oos']['cv_wrong_pct']:.0%} 失手；点全画出，重点看右尾")
ordr = list(np.argsort(np.median(mse4, 0)))
ax[1].bar(range(5), np.median(mse4, 0)[ordr], color=[{0: RED, 1: ORANGE, 2: BLUE, 3: PURPLE, 4: GREEN}[c] for c in ordr])
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([names4[i] for i in ordr], fontsize=9)
for i, c in enumerate(ordr):
    ax[1].text(i, np.median(mse4[:, c]) + .015, f"{np.median(mse4[:, c]):.3f}", ha="center", fontsize=10)
ax[1].set_title(f"(b) 中位数对决：冠军的账单是 BMA 的 {R['oos']['single_over_bma']:.2f}×\noracle（真模型 OLS）是天花板，BMA 贴住它")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_oos_showdown.png"); plt.close(fig)
print("fig4 done", flush=True)

# ---------------- fig5 g 敏感性 + 尺寸先验 ----------------
NSEED5 = 40
eff_g = [[] for _ in GS]; mse_g = [[] for _ in GS]; mt_g = [[] for _ in GS]; mse_sz = []
for s in range(NSEED5):
    r = np.random.default_rng(9000 + s)
    f, y = make_data(NTOT, r)
    ftr, ytr, fte, yte = f[:NTR], y[:NTR], f[NTR:], y[NTR:]
    rss, betas, ints = fit_all(ftr, ytr)
    P = pred_matrix(ftr, ytr, fte, betas, ints)
    for gi, g in enumerate(GS):
        w = weights_from_rss(rss, ytr, g)
        eff_g[gi].append(effn(w)); mse_g[gi].append(float(((w @ P - yte) ** 2).mean())); mt_g[gi].append(float(w[true_idx]))
    w_sz = weights_from_rss(rss, ytr, NTOT, size_prior=True)
    mse_sz.append(float(((w_sz @ P - yte) ** 2).mean()))
eff_g = [float(np.mean(x)) for x in eff_g]; mse_g = [float(np.mean(x)) for x in mse_g]; mt_g = [float(np.mean(x)) for x in mt_g]
mse_sz_m = float(np.mean(mse_sz))
_, w_dummy = None, None
w_uni = weights_from_rss(rss_d, yd[:NTR], NTOT); w_siz = weights_from_rss(rss_d, yd[:NTR], NTOT, size_prior=True)
R["g_sens"] = {"gs": [float(x) for x in GS], "effective_models": eff_g, "mean_oos": mse_g, "mass_true": mt_g,
               "eff_uniform_demo": effn(w_uni), "eff_sizeprior_demo": effn(w_siz),
               "mean_oos_sizeprior": mse_sz_m, "best_g": float(GS[int(np.argmin(mse_g))]),
               "g_prior_vs_uniform_gain": float(np.mean(mse_g) - mse_sz_m)}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(GS, eff_g, "o-", color=BLUE, label="有效模型数 exp(H)")
ax[0].plot(GS, mt_g, "s--", color=RED, label="真模型平均后验（n=300 全样本）")
ax[0].set_xscale("log"); ax[0].legend(fontsize=8); ax[0].set_xlabel("信息先验 g"); ax[0].axvline(NTOT, color=GRAY, ls=":", lw=1)
lo_g, hi_g = GS[0], GS[-1]
ax[0].annotate(f"g={lo_g}: 有效{eff_g[0]:.1f}个\n(接近全模型)", xy=(lo_g, eff_g[0]), xytext=(lo_g * 2, eff_g[0] * 0.75), fontsize=8, color=BLUE)
ax[0].annotate(f"g→∞: 有效{eff_g[-1]:.1f}个\n(接近空模型)", xy=(hi_g, eff_g[-1]), xytext=(hi_g / 40, eff_g[-1] + max(eff_g) * 0.15), fontsize=8, color=RED)
ax[0].set_title("(a) g 把 256 维模型空间从『全都要』拧到『全不要』\n真模型后验在中段 g 才抬起来")
ax[1].plot(GS, mse_g, "o-", color=PURPLE, label="均匀先验 BMA（40种子均值）")
ax[1].axhline(mse_sz_m, color=TEAL, ls="--", lw=1.4, label=f"尺寸先验 BMA={mse_sz_m:.3f}")
ax[1].set_xscale("log"); ax[1].axvline(NTOT, color=GRAY, ls=":", lw=1)
bg = GS[int(np.argmin(mse_g))]
ax[1].annotate(f"最优 g={bg:g}\nOOS={min(mse_g):.3f}", xy=(bg, min(mse_g)), xytext=(bg * 4, min(mse_g) * 1.05), fontsize=9, color=PURPLE)
ax[1].legend(fontsize=8); ax[1].set_ylabel("60 天样本外 MSE"); ax[1].set_xlabel("g")
ax[1].set_title(f"(b) OOS 随 g 波动 {min(mse_g):.3f}→{max(mse_g):.3f}（{100 * (max(mse_g) / min(mse_g) - 1):.0f}%）\n『默认 g=n 就行』经不经得起逐点对照？这里就是对照表")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_g_prior_sensitivity.png"); plt.close(fig)

with open(os.path.join(OUT, "_bma_results.json"), "w") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("ALL DONE")
print(json.dumps(R, ensure_ascii=False, indent=1))
