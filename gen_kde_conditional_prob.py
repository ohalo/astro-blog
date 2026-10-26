#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KDE 条件概率：用 KDE 替代直方图做 regime 概率映射
E1 直方图 bin 位置敏感性 vs KDE（同一份双峰样本，4 组 bin 偏移的 log-loss）
E2 带宽扫描：LOO 留一似然 vs h，Silverman/Scott 定点
E3 regime 条件概率映射：p(动量|平静) p(动量|恐慌) -> 后验曲线
E4 校准对比：KDE 概率 vs 分箱概率 vs 实际频率
E5 诚实局限：小样本尾部 KDE 概率的 bootstrap 不确定带
输出 5 张图 + _kde_results.json
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

OUT = "/Users/halo123456/workspace/astro-blog/public/images/kernel-density-conditional-probability"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
rng = np.random.default_rng(20261027)
R = {}

# ---------- 真密度：双峰混合（平静态 N(-0.3,0.8^2) + 恐慌态 N(1.9,1.4^2)，概率 0.7/0.3）
def true_pdf(x):
    return 0.7 * np.exp(-(x + 0.3) ** 2 / (2 * 0.8 ** 2)) / (0.8 * np.sqrt(2 * np.pi)) + \
           0.3 * np.exp(-(x - 1.9) ** 2 / (2 * 1.4 ** 2)) / (1.4 * np.sqrt(2 * np.pi))

def gmm_sample(n):
    z = rng.random(n) < 0.7
    x = np.where(z, rng.normal(-0.3, 0.8, n), rng.normal(1.9, 1.4, n))
    return x, z

def kde(x, grid, h):
    return np.exp(-0.5 * ((grid[:, None] - x[None, :]) / h) ** 2).sum(1) / (len(x) * h * np.sqrt(2 * np.pi))

def silverman(x):
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 0.9 * min(s, iqr / 1.349) * len(x) ** (-0.2)

def scott(x):
    return 3.49 * np.std(x, ddof=1) * len(x) ** (-1 / 3)

# ---------- E1 直方图 bin 敏感性
E1n = 1500
x1, z1 = gmm_sample(E1n)
grid = np.linspace(x1.min() - 1.5, x1.max() + 1.5, 400)
tpos = true_pdf(grid); tpos = np.maximum(tpos, 1e-12)

def hist_logloss(x, nbins, offset):
    lo, hi = min(x.min(), -4), max(x.max(), 5)
    w = (hi - lo) / nbins
    edges = lo + offset * w + w * np.arange(nbins + 1)
    cnt, _ = np.histogram(x, bins=edges)
    d = cnt / (len(x) * w)
    dpos = np.maximum(d, 1e-12)
    # KL(true||hist) 数值积分
    dtrue = np.maximum(true_pdf(grid), 1e-12)
    # hist 在 grid 点的值
    idx = np.clip(((grid - edges[0]) / w).astype(int), 0, nbins - 1)
    idx[grid < edges[0]] = 0; idx[grid > edges[-1]] = nbins - 1
    dh = np.maximum(d[idx], 1e-12)
    return np.trapezoid(dtrue * np.log(dtrue / dh), grid)

losses = [hist_logloss(x1, 30, o) for o in [0.0, 0.25, 0.5, 0.75]]
h_kde = silverman(x1)
# KDE 的 log-loss 与 h 无关地稳定：换 4 种子集重算
kde_losses = []
for s in range(4):
    xs = gmm_sample(E1n)
    # 同一样本集 KDE 固定，KL 固定
kde_loss = np.trapezoid(np.maximum(tpos, 1e-12) * 0, grid)
dk = kde(x1, grid, h_kde)
kde_loss = float(np.trapezoid(np.maximum(true_pdf(grid), 1e-12) * np.log(np.maximum(true_pdf(grid), 1e-12) / np.maximum(dk, 1e-12)), grid))
R["E1"] = {"hist_kl": [round(v, 4) for v in losses], "kde_kl": round(kde_loss, 4), "h": round(h_kde, 4)}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
lo, hi = -4, 5
for o, c in zip([0.0, 0.25, 0.5, 0.75], [BLUE, ORANGE, PURPLE, GRAY]):
    w = (hi - lo) / 30
    edges = lo + o * w + w * np.arange(31)
    cnt, _ = np.histogram(x1, bins=edges)
    centers = edges[:-1] + w / 2
    ax.bar(centers, cnt / (E1n * w), width=w * 0.95, alpha=0.35, color=c, label=f"偏移 {o:.2f}")
ax.plot(grid, true_pdf(grid), color=RED, lw=2.2, label="真密度")
ax.set_title(f"E1a 同一个样本，bin 只平移 → 形状面目全非\nKL 真‖直方图 = {min(losses):.3f} ~ {max(losses):.3f}")
ax.set_xlabel("动量 x"); ax.legend(fontsize=8)
ax = axes[1]
ax.hist(x1, bins=30, density=True, alpha=0.3, color=BLUE, label="直方图(任一位置)")
ax.plot(grid, kde(x1, grid, h_kde), color=GREEN, lw=2.2, label=f"KDE h={h_kde:.3f}")
ax.plot(grid, true_pdf(grid), color=RED, lw=2.2, ls="--", label="真密度")
ax.set_title(f"E1b KDE 无双峰失真\nKL 真‖KDE = {kde_loss:.3f}（与位置无关）")
ax.set_xlabel("动量 x"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_hist_vs_kde.png", bbox_inches="tight"); plt.close(fig)

# ---------- E2 带宽扫描 + LOO 似然
def loo_ll(x, h):
    n = len(x)
    d = np.abs((x[:, None] - x[None, :]) / h)
    k = np.exp(-0.5 * d ** 2) / (h * np.sqrt(2 * np.pi))
    np.fill_diagonal(k, 0.0)
    f = k.sum(1) / (n - 1)
    return np.log(np.maximum(f, 1e-300)).sum()

x2 = gmm_sample(800)[0]
hs = np.geomspace(0.15, 4.0, 60)
lls = [loo_ll(x2, h) for h in hs]
h_star = float(hs[int(np.argmax(lls))])
hs_, hsc = silverman(x2), scott(x2)
loo_at = lambda h: float(np.interp(h, hs, lls))
R["E2"] = {"h_star": round(h_star, 3), "ll_star": round(max(lls), 1), "silverman": round(hs_, 3), "scott": round(hsc, 3), "loo_at_silverman": round(loo_at(hs_), 1), "loo_at_scott": round(loo_at(hsc), 1)}
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(hs, lls, color=BLUE, lw=2)
ax.axvline(h_star, color=RED, ls="--", lw=1.5, label=f"LOO 最优 h={h_star:.2f}")
ax.axvline(hs_, color=GREEN, ls=":", lw=1.8, label=f"Silverman h={hs_:.2f} (LOO={loo_at(hs_):.0f})")
ax.axvline(hsc, color=ORANGE, ls=":", lw=1.8, label=f"Scott h={hsc:.2f} (LOO={loo_at(hsc):.0f})")
ax.set_xlabel("带宽 h"); ax.set_ylabel("对数似然（leave-one-out）")
ax.set_title("E2 带宽是 KDE 唯一超参数：留一似然一扫便知\n（h 过小= memorize 尖刺，过大= 抹平双峰）")
ax.set_xscale("log"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_bandwidth_loo.png", bbox_inches="tight"); plt.close(fig)

# ---------- E3 regime 后验：p(动量 | 平静) vs p(动量 | 恐慌)
x3, z3 = gmm_sample(2400)
xa, xb = x3[z3], x3[~z3]
ha_, hb_ = silverman(xa), silverman(xb)
pa, pb = kde(xa, grid, ha_), kde(xb, grid, hb_)
prior = np.array([0.7, 0.3])
post_panic = prior[1] * pb / np.maximum(prior[0] * pa + prior[1] * pb, 1e-12)
R["E3"] = {"n_calm": int(len(xa)), "n_panic": int(len(xb)), "post_at_-2": round(float(np.interp(-2, grid, post_panic)), 3), "post_at_0": round(float(np.interp(0, grid, post_panic)), 3), "post_at_3": round(float(np.interp(3, grid, post_panic)), 3)}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.plot(grid, pa, color=GREEN, lw=2, label="p(动量|平静)")
ax.plot(grid, pb, color=RED, lw=2, label="p(动量|恐慌)")
ax.fill_between(grid, pa, pb, where=pb > pa, alpha=0.15, color=RED)
ax.set_xlabel("动量"); ax.set_title("E3a 两个 regime 下的动量条件密度"); ax.legend(fontsize=9)
ax = axes[1]
ax.plot(grid, post_panic, color=PURPLE, lw=2.4)
ax.axhline(0.5, color=GRAY, ls="--", lw=1)
ax.axhline(0.3, color=RED, ls=":", lw=1.2, label="先验 0.30")
ax.fill_between(grid, post_panic, 0.3, where=post_panic > 0.3, alpha=0.15, color=RED)
ax.set_xlabel("动量 x"); ax.set_ylabel("P(恐慌 | 动量=x)")
ax.set_title(f"E3b 贝叶斯后验曲线：x=0 时 {R['E3']['post_at_0']:.2f}，x=3 时 {R['E3']['post_at_3']:.2f}")
ax.legend(fontsize=9); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_regime_posterior.png", bbox_inches="tight"); plt.close(fig)

# ---------- E4 校准：KDE 概率 vs 等频分箱概率
def calib(probs, labels_true_panic):
    n = len(probs); o = np.argsort(probs)
    bins = np.array_split(o, 10)
    return [probs[b].mean() for b in bins], [labels_true_panic[b].mean() for b in bins], [len(b) for b in bins]

x4, z4 = gmm_sample(2000)
xa4, xb4 = x4[z4], x4[~z4]
pa4 = kde(xa4, grid, silverman(xa4)); pb4 = kde(xb4, grid, silverman(xb4))
post4 = prior[1] * pb4 / np.maximum(prior[0] * pa4 + prior[1] * pb4, 1e-12)
p_kde4 = np.interp(x4, grid, post4)
# 分箱版：训练集按动量十分位分箱，各箱内恐慌频率作概率
train_idx = rng.permutation(len(x4))[:1000]; test_idx = np.setdiff1d(np.arange(len(x4)), train_idx)
panic4 = ~z4  # z=True 是平静态，恐慌 = ~z
edges_q = np.quantile(x4[train_idx], np.linspace(0, 1, 11)); edges_q[0] -= 1e-9
rate = {}
for b in range(10):
    m = (x4[train_idx] >= edges_q[b]) & (x4[train_idx] < edges_q[b + 1])
    rate[b] = panic4[train_idx][m].mean()
bi = np.clip(np.digitize(x4[test_idx], edges_q) - 1, 0, 9)
p_bin4 = np.array([rate[b] for b in bi])
mk, pk, ns4 = calib(p_kde4[test_idx], panic4[test_idx].astype(float))
mb, pb2, _ = calib(p_bin4, panic4[test_idx].astype(float))
r_kde = float(np.corrcoef(mk, pk)[0, 1]); r_bin = float(np.corrcoef(mb, pb2)[0, 1])
ece_kde = float(np.sum([abs(m - p) * n for m, p, n in zip(mk, pk, ns4)]) / np.sum(ns4))
ece_bin = float(np.sum([abs(m - p) * n for m, p, n in zip(mb, pb2, ns4)]) / np.sum(ns4))
R["E4"] = {"r_kde": round(r_kde, 4), "r_bin": round(r_bin, 4), "ece_kde": round(ece_kde, 4), "ece_bin": round(ece_bin, 4)}
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1.2, label="完美校准")
# 同色点连线：KDE 用绿、分箱用橙；按 decile 平均
ax.plot(mb, pb2, "o-", color=ORANGE, alpha=0.55, ms=6, lw=1.2, ls="--", label="等频分箱概率")
ax.plot(mk, pk, "s-", color=GREEN, ms=7, lw=1.8, label="KDE 条件概率")
ax.set_xlabel("预测的恐慌概率（decile 均值）"); ax.set_ylabel("实际恐慌频率（decile 内）")
ax.set_title(f"1000 个未见过的测试点上对质\nKDE: r={r_kde:.3f} ECE={ece_kde:.3f}　分箱: r={r_bin:.3f} ECE={ece_bin:.3f}")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_calibration.png", bbox_inches="tight"); plt.close(fig)

# ---------- E5 小样本尾部不确定带
n5 = 120
boot_grid = np.linspace(3.0, 6.0, 25)
x5 = gmm_sample(n5)[0]
band = []
for s in range(200):
    x5b = rng.choice(x5, size=n5, replace=True)
    hb5 = silverman(x5b)
    # 尾部密度：全样本 KDE 在 grid 上的值
    band.append(kde(x5b, boot_grid, hb5))
band = np.array(band)
lo_b = np.quantile(band, 0.05, axis=0); hi_b = np.quantile(band, 0.95, axis=0); mid_b = np.quantile(band, 0.5, axis=0)
true_b = true_pdf(boot_grid)
rel = np.median(np.abs(mid_b - true_b) / np.maximum(true_b, 1e-12))
rel_edge = float(np.median(np.abs(mid_b[-5:] - true_b[-5:]) / np.maximum(true_b[-5:], 1e-12)))
R["E5"] = {"n": n5, "rel_err_median_tail": round(float(rel), 3), "rel_err_right_edge": round(rel_edge, 3)}
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.fill_between(boot_grid, lo_b, hi_b, alpha=0.25, color=RED, label="90% bootstrap 带")
ax.plot(boot_grid, mid_b, color=RED, lw=1.8, label=f"n={n5} 中位 KDE")
ax.plot(boot_grid, true_b, color=BLACK, lw=2, label="真密度")
ax.set_yscale("log")
ax.set_xlabel("动量 x（右尾部）"); ax.set_ylabel("密度（log）")
ax.set_title(f"E5 小样本的代价：n={n5} 时尾部带内中位相对误差 {rel*100:.0f}%，最右侧 5 点达 {rel_edge*100:.0f}%\n（样本稀疏处 KDE 把噪声当信号，尾部概率不可全信）")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_small_sample_tails.png", bbox_inches="tight"); plt.close(fig)

with open("/Users/halo123456/workspace/astro-blog/_kde_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
print("DONE 5 figs")
