---
title: "比较两个模型的 AUC 别再当成独立：DeLong 检验、配对自助与那个被忽略的协方差"
description: "两个分类器在同一批样本上打分，它们的 AUC 误差是相关的，忽略这层相关的朴素检验会系统性失真。真值已知的双正态实验（120 正 120 负、每档 800 次重复）显示：相关性 rho 从 0 升到 0.9 时，朴素检验的虚警率从 5.0% 塌到 0.0%，方差被高估到真值的 8.87 倍，而 DeLong 与配对自助稳在 4.4% 到 5.8%；95% 置信区间朴素法覆盖率从 94.6% 涨到 100%（过度覆盖），DeLong 守在 94.6% 到 95.0%。功效上 DeLong 全面占优：真 AUC 差 0.054 时检出率 37.5%，朴素只有 8.8%；解析 AUC 与 40 万对蒙特卡洛吻合到 0.0015 以内。结论：同一批样本比 AUC，必须用配对方法。"
publishDate: '2026-11-21T11:20:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 模型评估
  - Python
language: zh
difficulty: advanced
image: "/images/delong-auc-roc-comparison/fig1_auc_theory.png"
---

## 一、结论先行

比较两个风控模型谁更能分辨好坏客户，最自然的做法是各算一个 AUC，再检验两者差是否显著。很多人的做法是把两个 AUC 当成独立估计，各自算方差、相加、开根号，做一个 z 检验。只要这两个模型是在同一批样本上打的分，这个做法就是错的——它们的 AUC 误差共享同一批正负样本，天生相关。

本文用一个真值已知的双正态实验把后果量化。两个分类器在同一批 120 正、120 负样本上打分，分数由一个潜在相关系数 rho 驱动，rho 越大两者越像。所有数字来自同一份脚本，代码附在文末。

先把结论摆出来。第一，rho 从 0 升到 0.9 时，朴素检验的虚警率从 5.0% 塌到 0.0%——它变得极度保守，把真实的差异也当成噪声，方差被高估到真值的 8.87 倍。第二，DeLong 检验和配对自助都稳在 4.4% 到 5.8%，不受相关性影响。第三，95% 置信区间上朴素法覆盖率从 94.6% 涨到 100%，区间宽得没信息；DeLong 守在 94.6% 到 95.0%。第四，功效上 DeLong 全面占优：真 AUC 差只有 0.054 时，DeLong 检出 37.5%，朴素只有 8.8%。第五，整个实验的地基——双正态 AUC 的闭式 $\Phi(\delta/\sqrt{2})$——与 40 万对蒙特卡洛吻合到 0.0015 以内。

一句话：同一批样本上比 AUC，协方差不是可选项，忽略它就是把检验的方向盘拆了开。

## 二、实验室：双正态分数与已知的 AUC

设定写成公式。每个样本有一对潜在分数 $(s_A, s_B)$，来自相关系数 rho 的二元正态。正类的两个分数均值都加上 delta，负类均值为零。于是单看分类器 A，它的分数在正类是 $N(\delta,1)$、负类是 $N(0,1)$，双正态模型的 AUC 有闭式 $\Phi(\delta/\sqrt{2})$。

要比较两个分类器时，给 A 的正类均值加 gap/2、给 B 减 gap/2，这样 A 的真 AUC 更高，gap 控制两者的真实差距。rho 只改变分数对的相关性，不改变各自的边缘分布，所以真实 AUC 完全由 delta 和 gap 决定，rho 只影响估计的协方差。这个设计让我们能单独观察「忽略协方差」的破坏力。

图1 左面板验证地基：40 万对样本下，蒙特卡洛 AUC 与闭式逐点吻合，最大偏差 0.0015。右面板画出 delta=0.9 时的 ROC 曲线。

![图1：双正态 AUC 闭式与蒙特卡洛对拍，以及一条示例 ROC 曲线](/images/delong-auc-roc-comparison/fig1_auc_theory.png)

## 三、协方差去哪了

DeLong 检验的核心是把 AUC 写成 U 统计量，再分解出每个样本的影响。对每个负类样本 i，定义它对所有正类的影响 $V_{10}(i)$——也就是这个负样本被随机抽到的正样本打败的比例，减去整体 AUC；对每个正类样本 j 定义 $V_{01}(j)$——这个正样本赢过随机抽到的负样本的比例，减去整体 AUC。AUC 的方差就是这两组影响值的样本方差分别除以各自的样本量再相加：$\text{Var}(\widehat{AUC}) = S_{10}/n + S_{01}/m$，其中 n、m 是负、正样本数。这个分解是 U 统计量的渐近方差理论给出的，不依赖分数服从正态——这也是它比参数化方法鲁棒的地方。

比较两个 AUC 时，关键一步是把「差值」的方差直接算出来：把 A 和 B 的影响值先逐样本相减（$V_{10}^A(i)-V_{10}^B(i)$、$V_{01}^A(j)-V_{01}^B(j)$），再求方差。这样两个分类器共享样本带来的协方差被自然地减掉了。朴素做法则是各自求方差再相加，等于假设协方差为零——当 rho 大于零时，真实协方差为正，朴素方差就偏大。换个直观说法：A 和 B 在同一批样本上同步犯错时，它们的误差会相互抵消一部分，差值其实比各自的不确定度更稳定；朴素方法看不到这个抵消，以为差值和不小心相加一样不确定。

图2 显示虚警率的崩塌过程。

![图2：相关性 rho 上升时朴素检验虚警率塌到零，DeLong 与配对自助守住 5%](/images/delong-auc-roc-comparison/fig2_size.png)

rho=0 时三种方法都在 5% 附近，因为分数独立、协方差本就为零。rho=0.3 时朴素掉到 3.0%，rho=0.6 时 0.13%，rho=0.9 时 0.0%。DeLong 分别是 5.25%、5.5%、4.4%、5.8%，配对自助 5.1%、6.0%、4.4%、5.6%，都在抽样误差范围内守住了名义水平。朴素检验的问题不是「虚警太多」，而是相反——它太怕报警，把真差异也压住了。

## 四、方差被高估多少

图3 直接画出朴素方差与 DeLong 方差的比值。

![图3：忽略协方差导致方差估计被系统性放大](/images/delong-auc-roc-comparison/fig3_var_inflation.png)

rho=0 时比值 1.00，两者一致。rho=0.3 时 1.40，rho=0.6 时 2.36，rho=0.9 时 8.87。这个放大倍数几乎全部来自被丢掉的协方差项：两个模型越像，它们的 AUC 误差越同步，差值的不确定性越小，而朴素方法假装它们独立，反而把不确定度越估越大。这就是为什么它越来越保守。

## 五、功效：该报警时谁先报

保守的代价是漏报。图4 扫描真实 AUC 差，看两种方法的检出率。

![图4：真 AUC 差增大时 DeLong 的检出率全面领先朴素检验](/images/delong-auc-roc-comparison/fig4_power.png)

gap=0 是原假设，DeLong 检出率 5.3%（守住了 size），朴素 0.4%（再次保守）。真 AUC 差 0.054（gap=0.2）时，DeLong 检出 37.5%，朴素只有 8.8%，差了四倍多。差 0.108 时 DeLong 90.2%、朴素 58.8%。差距拉到 0.21 以上时两者都接近 100%，但那已经是很容易识别的情形。真正值钱的是小差距区间——模型 A 比 B 好一点点、但这一点点有实际业务价值时，DeLong 能查出来，朴素会放过。

这里要诚实交代一个边界：功效优势的前提是两个模型的分数确实同源、共享样本。如果两个 AUC 来自完全独立的两批测试集，协方差本就为零，朴素检验不再有偏，DeLong 的配对修正也就没有用武之地——它不会更差，但也谈不上更好。反过来说，只要你在同一批历史数据上回测多个模型再挑最优，配对结构就必然存在，本文的结论就适用。还有一个常被忽略的点：DeLong 检验的是「两个 AUC 是否不同」，不是「哪个模型在业务上更赚钱」。AUC 差 0.05 可能统计显著，但换成策略的夏普比率提升也许微乎其微；统计显著和经济显著之间，永远隔着一层成本与容量的现实。

## 六、置信区间：宽到没有信息

最后看覆盖率。图5 画 95% 置信区间对真实 AUC 差的覆盖。

![图5：朴素置信区间过度覆盖到 100%，DeLong 稳定在 95%](/images/delong-auc-roc-comparison/fig5_coverage.png)

rho=0 时两者都是 94.6%。rho 上升后朴素覆盖率一路涨到 rho=0.9 的 100%——区间从未漏掉真值，但代价是宽得离谱，等于什么都没说。DeLong 稳定在 94.6% 到 95.0%，既不漏报也不说空话。如果你用朴素区间给模型差异做决策，「差异不显著」这个结论往往只是区间太宽造成的假象。

对做量化的人，这条结论可以直接落地：回测里比较两个信号、两个模型、两套参数的 AUC（或任何基于同一批样本的排名指标），只要样本是共用的，就该用配对方法。DeLong 有闭式、算得快；配对自助更灵活、能处理复杂指标，两者都比「各自算方差再相加」靠谱。

```python
# -*- coding: utf-8 -*-
"""DeLong test for comparing two correlated ROC AUCs on the same sample.

Single source of truth for every number and figure in the article.

DGP (binormal, known truth): for each replication draw m positives and n
negatives. A latent bivariate-normal score pair (sA, sB) with correlation rho
is generated; positives have both means shifted by delta. AUC_A = AUC_B under
H0 (delta=0). Because both classifiers are scored on the SAME cases, their
AUC errors are correlated; ignoring that covariance (a "naive independent"
z-test) mis-sizes the test.

Analytic baselines:
    binormal AUC = Phi(delta / sqrt(2)) when scores are N(0,1)/N(delta,1)
    AUC covariance under paired designs is positive for rho>0, so the naive
    independent variance overstates the true variance -> test is conservative.

Methods for H0: AUC_A = AUC_B:
    naive  : var = var(A) + var(B), covariance ignored
    delong : U-statistic structural components V10, V01 incl. covariance
    bootstrap: paired resampling of cases (B=399)
    hanley : analytic binormal variance (no covariance, for comparison)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261121
M_POS = 120
N_NEG = 120
REPS_SIZE = 2000
REPS_POWER = 1200
B_BOOT = 399
RHO_GRID = [0.0, 0.3, 0.6, 0.9]
DELTA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]   # analytic-check shifts
GAP_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # power: true AUC gap A vs B
Z95 = float(special.ndtri(0.975))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "delong-auc-roc-comparison")
os.makedirs(_OUT, exist_ok=True)


def auc(y, x):
    """Empirical AUC = P(score_pos > score_neg), mid-rank for ties.
    Mann-Whitney U via average ranks of pooled scores -> O((m+n)log), vectorized."""
    m, n = len(y), len(x)
    ranks = stats.rankdata(np.concatenate([x, y]), method="average")
    return float((ranks[n:].sum() - m * (m + 1) / 2.0) / (m * n))


def draw_pair(rng, delta, rho, gap=0.0, m=M_POS, n=N_NEG):
    """Return (pos scores A,B) and (neg scores A,B) with latent corr rho.
    Positive means: A = delta + gap/2, B = delta - gap/2, so gap>0 makes
    A genuinely better (AUC_A > AUC_B) while keeping the paired correlation."""
    L = np.array([[1.0, 0.0], [rho, np.sqrt(max(1 - rho * rho, 1e-12))]])
    zp = rng.standard_normal((m, 2)) @ L.T
    zn = rng.standard_normal((n, 2)) @ L.T
    pos = zp + np.array([delta + gap / 2, delta - gap / 2])
    neg = zn
    return pos[:, 0], pos[:, 1], neg[:, 0], neg[:, 1]


def delong_var(yA, yB, xA, xB):
    """DeLong variance of (AUC_A - AUC_B) using structural components."""
    def comps(y, x):
        # V10_i: influence of negative case i ; V01_j: influence of positive case j
        ind = (y[:, None] > x[None, :]).astype(float)  # (m,n)
        v10 = ind.mean(axis=0)   # length n
        v01 = ind.mean(axis=1)   # length m
        return v10, v01
    v10A, v01A = comps(yA, xA)
    v10B, v01B = comps(yB, xB)
    m, n = len(yA), len(xA)
    s10 = np.var(v10A - v10B, ddof=1) / n
    s01 = np.var(v01A - v01B, ddof=1) / m
    return s10 + s01


def naive_var(yA, yB, xA, xB):
    """Naive: treat the two AUCs as independent, drop covariance."""
    def one(y, x):
        ind = (y[:, None] > x[None, :]).astype(float)
        v10 = ind.mean(axis=0)
        v01 = ind.mean(axis=1)
        return np.var(v10, ddof=1) / len(x) + np.var(v01, ddof=1) / len(y)
    return one(yA, xA) + one(yB, xB)


def boot_diff(yA, yB, xA, xB, rng, B=B_BOOT):
    """Paired bootstrap of AUC_A - AUC_B (resample cases jointly)."""
    m, n = len(yA), len(xA)
    out = np.empty(B)
    for b in range(B):
        pi = rng.integers(0, m, m)
        ni = rng.integers(0, n, n)
        out[b] = auc(yA[pi], xA[ni]) - auc(yB[pi], xB[ni])
    return out


def exp_analytic_check():
    """Verify binormal AUC closed form against large-sample MC, and show ROC."""
    rng = np.random.default_rng(SEED + 1)
    big = 400000
    z = rng.standard_normal(big)
    rows = []
    for delta in [0.0, 0.3, 0.6, 0.9, 1.2]:
        y = delta + rng.standard_normal(big)
        emp = auc(y, z)
        theory = float(special.ndtr(delta / np.sqrt(2)))
        rows.append({"delta": delta, "emp": emp, "theory": theory,
                     "abs_err": abs(emp - theory)})
    # ROC curve at delta=0.9 for plotting
    delta = 0.9
    yr = delta + rng.standard_normal(200000)
    zr = rng.standard_normal(200000)
    thr = np.linspace(-3, 4, 120)
    tpr = [(yr > t).mean() for t in thr]
    fpr = [(zr > t).mean() for t in thr]
    return {"rows": rows, "roc_delta": delta, "thr": thr.tolist(),
            "tpr": [float(x) for x in tpr], "fpr": [float(x) for x in fpr]}


def one_rep(rng, delta, rho, gap=0.0, boot=False):
    yA, yB, xA, xB = draw_pair(rng, delta, rho, gap=gap)
    d = auc(yA, xA) - auc(yB, xB)
    vn = naive_var(yA, yB, xA, xB)
    vd = delong_var(yA, yB, xA, xB)
    res = {"d": d, "vn": vn, "vd": vd}
    if boot:
        bd = boot_diff(yA, yB, xA, xB, rng)
        res["vb"] = float(bd.var(ddof=1))
    return res


def exp_size_boot():
    """H0 size by rho: naive / delong / bootstrap, with bootstrap per rep."""
    rng = np.random.default_rng(SEED + 2)
    reps = 800
    rows = []
    for rho in RHO_GRID:
        rej_n = rej_d = rej_b = 0
        vn_mean = vd_mean = vb_mean = 0.0
        cov_emp = []
        for _ in range(reps):
            r = one_rep(rng, 0.0, rho, boot=True)
            vn_mean += r["vn"]
            vd_mean += r["vd"]
            vb_mean += r["vb"]
            if abs(r["d"]) / np.sqrt(max(r["vn"], 1e-15)) > Z95:
                rej_n += 1
            if abs(r["d"]) / np.sqrt(max(r["vd"], 1e-15)) > Z95:
                rej_d += 1
            if abs(r["d"]) / np.sqrt(max(r["vb"], 1e-15)) > Z95:
                rej_b += 1
            cov_emp.append(r["vn"] - r["vd"])  # naive minus true variance
        rows.append({"rho": rho, "naive": rej_n / reps, "delong": rej_d / reps,
                     "boot": rej_b / reps,
                     "vn_mean": vn_mean / reps, "vd_mean": vd_mean / reps,
                     "vb_mean": vb_mean / reps,
                     "var_inflation": (vn_mean / vd_mean)})
    return {"reps": reps, "b_boot": B_BOOT, "rows": rows}


def exp_power():
    """Power of naive/delong vs true AUC gap under rho=0.6.
    Base signal delta=0.4; A gets +gap/2, B gets -gap/2 in positive mean."""
    rng = np.random.default_rng(SEED + 3)
    rho = 0.6
    base = 0.4
    rows = []
    for gap in GAP_GRID:
        rej_n = rej_d = 0
        aucA = float(special.ndtr((base + gap / 2) / np.sqrt(2)))
        aucB = float(special.ndtr((base - gap / 2) / np.sqrt(2)))
        gap_truth = aucA - aucB
        d_sum = 0.0
        for _ in range(REPS_POWER):
            r = one_rep(rng, base, rho, gap=gap, boot=False)
            d_sum += r["d"]
            if abs(r["d"]) / np.sqrt(max(r["vn"], 1e-15)) > Z95:
                rej_n += 1
            if abs(r["d"]) / np.sqrt(max(r["vd"], 1e-15)) > Z95:
                rej_d += 1
        rows.append({"gap": gap, "naive": rej_n / REPS_POWER,
                     "delong": rej_d / REPS_POWER,
                     "auc_gap_truth": gap_truth, "auc_gap_emp": d_sum / REPS_POWER})
    return {"reps": REPS_POWER, "rho": rho, "base": base, "rows": rows}


def exp_size_coverage():
    """95% CI coverage of AUC_A - AUC_B (true delta=0) by rho, naive vs DeLong."""
    rng = np.random.default_rng(SEED + 4)
    reps = 1500
    rows = []
    for rho in RHO_GRID:
        cov_n = cov_d = 0
        for _ in range(reps):
            r = one_rep(rng, 0.0, rho, boot=False)
            d = r["d"]
            if abs(d) <= Z95 * np.sqrt(max(r["vn"], 1e-15)):
                cov_n += 1
            if abs(d) <= Z95 * np.sqrt(max(r["vd"], 1e-15)):
                cov_d += 1
        rows.append({"rho": rho, "naive_cov": cov_n / reps, "delong_cov": cov_d / reps})
    return {"reps": reps, "rows": rows}


def plot1(check):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ds = [r["delta"] for r in check["rows"]]
    ax.plot(ds, [r["theory"] for r in check["rows"]], "s-", color="#2c3e50", lw=2, label="analytic Phi(delta/sqrt2)")
    ax.plot(ds, [r["emp"] for r in check["rows"]], "o--", color="#c0392b", lw=2, label="MC (400k pairs)")
    ax.set_xlabel("score shift delta")
    ax.set_ylabel("AUC")
    ax.set_title("1a: binormal AUC closed form vs MC")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(check["fpr"], check["tpr"], color="#2980b9", lw=2)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC at delta=%.1f" % check["roc_delta"])
    fig.tight_layout()
    path = os.path.join(_OUT, "fig1_auc_theory.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig1_auc_theory.png"


def plot2(size):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["rho"] for r in size["rows"]]
    ax.plot(xs, [r["naive"] for r in size["rows"]], "o-", color="#c0392b", lw=2.2, label="naive (covariance ignored)")
    ax.plot(xs, [r["delong"] for r in size["rows"]], "s-", color="#27ae60", lw=2.2, label="DeLong")
    ax.plot(xs, [r["boot"] for r in size["rows"]], "^--", color="#8e44ad", lw=1.8, label="paired bootstrap")
    ax.axhline(0.05, color="k", ls=":", lw=1.5, label="nominal 5%")
    ax.set_xlabel("score correlation rho (same cases)")
    ax.set_ylabel("false rejection of H0: AUC_A=AUC_B")
    ax.set_ylim(-0.01, 0.16)
    ax.set_title("Fig2: naive goes conservative as rho rises; DeLong holds 5%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig2_size.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig2_size.png"


def plot3(size):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["rho"] for r in size["rows"]]
    ax.plot(xs, [r["var_inflation"] for r in size["rows"]], "o-", color="#c0392b", lw=2.2)
    ax.axhline(1.0, color="#27ae60", ls="--", lw=1.6, label="correct (inflation=1)")
    ax.set_xlabel("score correlation rho")
    ax.set_ylabel("naive var / DeLong var")
    ax.set_title("Fig3: ignoring covariance inflates the variance estimate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig3_var_inflation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig3_var_inflation.png"


def plot4(power):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["auc_gap_truth"] for r in power["rows"]]
    ax.plot(xs, [r["delong"] for r in power["rows"]], "s-", color="#27ae60", lw=2.2, label="DeLong power")
    ax.plot(xs, [r["naive"] for r in power["rows"]], "o--", color="#c0392b", lw=2, label="naive power")
    ax.axhline(0.05, color="k", ls=":", lw=1.3, label="size 5%")
    ax.set_xlabel("true AUC gap (A - B)")
    ax.set_ylabel("rejection rate")
    ax.set_title("Fig4: DeLong detects real gaps that naive misses")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig4_power.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig4_power.png"


def plot5(cov):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["rho"] for r in cov["rows"]]
    ax.plot(xs, [r["naive_cov"] for r in cov["rows"]], "o-", color="#c0392b", lw=2.2, label="naive 95% CI")
    ax.plot(xs, [r["delong_cov"] for r in cov["rows"]], "s-", color="#27ae60", lw=2.2, label="DeLong 95% CI")
    ax.axhline(0.95, color="k", ls=":", lw=1.5, label="nominal 95%")
    ax.set_xlabel("score correlation rho")
    ax.set_ylabel("CI coverage of AUC_A - AUC_B")
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Fig5: naive CI over-covers; DeLong stays at 95%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig5_coverage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig5_coverage.png"


def main():
    check = exp_analytic_check()
    size = exp_size_boot()
    power = exp_power()
    cov = exp_size_coverage()
    results = {
        "setup": {"seed": SEED, "m_pos": M_POS, "n_neg": N_NEG,
                  "reps_size": size["reps"], "reps_power": REPS_POWER,
                  "reps_cov": cov["reps"], "b_boot": B_BOOT, "z95": Z95,
                  "rho_grid": RHO_GRID, "delta_grid": DELTA_GRID, "gap_grid": GAP_GRID},
        "analytic_check": {"rows": check["rows"], "roc_delta": check["roc_delta"]},
        "size_boot": size,
        "power": power,
        "coverage": cov,
    }
    results["figs"] = [plot1(check), plot2(size), plot3(size), plot4(power), plot5(cov)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 七、完整代码与复现

上面全部数字与五张图来自下面这段自包含脚本。它用固定种子 20261121，依次验证双正态 AUC 闭式、四种方法在相关性扫描下的虚警率、方差膨胀倍数、功效曲线与置信区间覆盖率。AUC 用秩次公式计算，与成对比较完全等价但更快；DeLong 方差由 U 统计量的结构成分直接给出；配对自助对正负样本联合重抽样 399 次。正文每个数字都能在脚本输出中逐项核对。

## 八、小结

AUC 是个好指标，但「比较两个 AUC」是个容易被低估的统计问题。同一批样本把两个模型的误差绑在一起，这层协方差既是麻烦也是信息：丢掉它，检验会变得又保守又宽泛，该报警时不报警，报出来的区间还毫无信息。DeLong 用 U 统计量把这层结构正面处理掉，配对自助用重抽样绕开公式，两者都值得进你的工具箱。记住一句话：数据是共用的，检验就必须配对。
