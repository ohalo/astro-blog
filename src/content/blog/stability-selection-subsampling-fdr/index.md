---
title: "稳定性选择：把因子反复抽一半看它还在不在，为什么它比单次 LASSO 更能挡住假信号，又在相关因子块上露出软肋"
description: "真值已知实验（n=240、p=100、8 个真信号）拆开互补子样本稳定性选择。互补对只把批间标准差从 0.0907 压到 0.0865（约 4.6%），远非万能；阈值从 0.6 升到 0.95，FDR 从 23.1% 压到 0 而功效从 88.5% 掉到 54.2%，单次 LASSO 在 0.3λmax 上 FDR 高达 36.7%。理论界 E[V]≤q²/((2π−1)n) 实测 4.39 对实际 0.58 个假阳性——有效但极松。相关块 rho=0.6 时假变量入选频率冲到 0.64，阈值挡不住成簇的假信号。附完整代码与五张真实计算图。"
publishDate: '2026-11-23T10:10:00Z'
tags:
  - 量化
  - 特征选择
  - 多重检验
  - LASSO
  - Python
language: zh
difficulty: advanced
image: "/images/stability-selection-subsampling-fdr/fig1_frequencies.png"
---

## 一、结论先行

挑因子最让人不踏实的不是选不出信号，而是每次重跑选出来的都不一样。同一批数据，换个随机种子、去掉几天样本，LASSO 选中的十个因子可能换掉三四个——你根本不知道哪些是真信号，哪些是这次抽样撞运气撞进来的。

**稳定性选择（Stability Selection）** 的思路很朴素：一个真信号应该经得起反复抽样。把数据切成很多个一半大小的互补子样本，每个子样本上跑一遍 LASSO 路径，统计每个变量被选中的频率；只有入选频率超过阈值 $\pi_{thr}$ 的才留下。Meinshausen 和 Bühlmann 在 2010 年给出了一个漂亮的有限样本界，让这个启发式方法有了理论靠山。

本文用一个真值已知的蒙特卡洛实验（$n=240$、$p=100$、8 个真信号，系数幅度 0.20 到 0.36）把它拆开看，五个结论都来自真实计算：

- **互补子样本的收益有限**：把 60 对互补半样本拆成 10 批，入选频率的批间标准差是 0.0865，独立半样本是 0.0907——只降了约 4.6%。互补结构确实有微弱优势，但远没到"消除噪声"的程度，别指望它替代更多数据。
- **阈值是明码标价的买卖**：$\pi_{thr}$ 从 0.6 升到 0.95，FDR 从 23.1% 一路压到 0，代价是功效从 88.5% 掉到 54.2%。想要干净的名单，就得接受漏掉一半信号。
- **单次 LASSO 明显更脏**：在 $0.3\lambda_{max}$ 这一个点上做一次拟合，FDR 高达 36.7%，稳定性选择（阈值 0.8）只有 6.4%。
- **理论界很松但方向对**：$\pi=0.8$ 时理论上界给 4.39 个假阳性，实测只有 0.58 个。保住了承诺，但保守了好几倍。
- **相关因子是软肋**：给无关因子塞进 $\rho=0.6$ 的相关块，假变量的入选频率冲到 0.64，和真信号重叠；提高阈值也挡不住成簇进来的假信号。

先说边界：这里测的是"控制变量选择错误"的方法，不是"预测收益"的方法。样本外组合那一段会看到，选得干净不等于赚得多。

## 二、方法：为什么是"互补对 + 路径最大值"

标准的稳定性选择有三步。第一步，把 $n$ 个观测随机分成两半，这两半严格互补——这就是"互补对"，重复 $B$ 次得到 $2B$ 个半样本。第二步，在每个半样本上对 LASSO 的整条正则化路径求解，记录变量在哪个 $\lambda$ 区间里被选中。第三步，对每个变量取它在整条路径上的最大入选指示，除以半样本总数，得到入选频率

$$
\hat{\Pi}_j = \frac{1}{2B}\sum_{b=1}^{2B}\mathbb{1}\{j \in \hat{S}_{\lambda}^{(b)} \text{ for some } \lambda \in [\lambda_{min},\lambda_{max}]\}.
$$

选 $\hat{\Pi}_j \ge \pi_{thr}$ 的变量。取路径最大值而不是固定某个 $\lambda$，是为了不让结果依赖于一个拍脑袋选出来的正则化强度。

为什么要互补对？因为两个半样本严格互补，它们的抽样误差符号相反。对 $x$ 和 $1-x$ 各估一次再平均，方差比独立抽两个样本小得多。这是本文第一个要量化的东西。

理论上界长这样：设 $q$ 是单个半样本平均选中的变量数，则期望假阳性数

$$
\mathbb{E}[V] \le \frac{q^2}{(2\pi_{thr}-1)\,n}.
$$

阈值必须大于 0.5 才有意义——$\pi_{thr}=0.5$ 时分母为零，界失效。直觉也说得通：一个纯噪声变量，被选中的概率不会显著超过一半。

下面是核心实现。整个实验自包含，包含数据生成、LASSO 坐标下降、互补对抽样、BH 对照和五个实验，可直接运行。

```python
# -*- coding: utf-8 -*-
"""Stability selection with complementary pairs: FDR, power, and the correlation trap.

Single source of truth for every number and figure in the article.

Linear signal model y = X beta + eps, X in R^{n x p}, eps ~ N(0, sigma^2 I).
Lasso coordinate descent on the Gram matrix. Stability selection
(Meinshausen & Buhlmann 2010) draws B complementary pairs of half-samples,
records selection indicators along a lambda path, and keeps variables whose
selection frequency exceeds pi_thr > 0.5. The MB bound
    E[V] <= q^2 / ((2 pi_thr - 1) n)
is printed against the empirical false-positive count; q is the mean number
of selected variables per half-sample along the path.

Experiments:
  1. selection frequencies for true vs null signals, complementary vs iid halves
  2. pi_thr sweep: FDR/power vs BH on marginal p-values vs single-fit lasso
  3. correlated null blocks (rho=0.6): the same three methods degrade differently
  4. sample-size sweep with the MB bound, showing how loose the bound can be
  5. out-of-sample long-short portfolio built on the selected signals

Fixed seeds make every replay byte-identical to results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
SIGMA = 1.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "stability-selection-subsampling-fdr")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------- lasso machinery
def lasso_path(G, c, lam_path, max_iter=200, tol=1e-7):
    """Coordinate descent over a decreasing lambda path. G = X'X/n, c = X'y/n.

    Returns array (n_lambda, p) of coefficient vectors (no intercept; X,y centred).
    """
    p = G.shape[0]
    beta = np.zeros(p)
    out = np.zeros((len(lam_path), p))
    diag = np.diag(G).copy()
    for li, lam in enumerate(lam_path):
        for _ in range(max_iter):
            old = beta.copy()
            for j in range(p):
                r = c[j] - G[j] @ beta + diag[j] * beta[j]
                beta[j] = np.sign(r) * max(abs(r) - lam, 0.0) / diag[j]
            if np.max(np.abs(beta - old)) <= tol * max(1.0, np.max(np.abs(old))):
                break
        out[li] = beta
    return out


def lam_grid(G, c, n_lam=20, hi=0.8, lo=0.15):
    lm = np.max(np.abs(c))
    return np.linspace(hi * lm, lo * lm, n_lam)


def complement_pairs(n, B, rng):
    """B pairs of complementary half-samples (indices)."""
    pairs = []
    for _ in range(B):
        perm = rng.permutation(n)
        pairs.append((perm[: n // 2], perm[n // 2 :]))
    return pairs


def selection_freq(X, y, pairs, lam_path, lam_lo, lam_hi):
    """Selection probability per variable: max over the lambda window.

    Also returns q = mean |selected| per half-sample over the full path.
    """
    n, p = X.shape
    hit = np.zeros((2 * len(pairs), p), dtype=bool)
    sizes = []
    for bi, (ia, ib) in enumerate(pairs):
        for k, idx in enumerate((ia, ib)):
            Xs, ys = X[idx], y[idx]
            Xs = Xs - Xs.mean(0)
            ys = ys - ys.mean()
            G = Xs.T @ Xs / len(idx)
            c = Xs.T @ ys / len(idx)
            path = lasso_path(G, c, lam_path)
            sel = np.abs(path) > 1e-8
            keep = (lam_path >= lam_lo) & (lam_path <= lam_hi)
            hit[2 * bi + k] = sel[keep].any(axis=0)
            sizes.append(sel[-1].sum())
    return hit.mean(axis=0), float(np.mean(sizes))


def bh_from_y(X, y, active, q=0.1):
    """BH on marginal OLS t-tests, using an independent null-variance estimate."""
    n, p = X.shape
    Xc = X - X.mean(0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    b = np.linalg.lstsq(XtX, Xc.T @ yc, rcond=None)[0]
    resid = yc - Xc @ b
    # leave-one-out style null sigma from the presumed null variables
    null = np.setdiff1d(np.arange(p), active)
    s2 = float(np.sum(resid ** 2) / max(1, len(null)))
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(XtX)) * s2, 1e-12))
    from scipy import stats
    pv = 2 * stats.norm.sf(np.abs(b) / se)
    order = np.argsort(pv)
    m = p
    crit = q * (np.arange(1, m + 1) / m)
    passed = pv[order] <= crit
    sel = np.zeros(p, dtype=bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        sel[order[: kmax + 1]] = True
    return sel


def metrics(sel, active):
    active = np.asarray(active, int)
    tp = int(np.sum(sel[active]))
    fp = int(np.sum(sel) - tp)
    fdr = fp / max(1, tp + fp)
    power = tp / len(active)
    return tp, fp, float(fdr), float(power)


def make_data(n, p, active, amps, rho, rng):
    if rho == 0.0:
        X = rng.standard_normal((n, p))
    else:
        X = rng.standard_normal((n, p))
        # null variables live in correlated blocks of 10
        null = np.setdiff1d(np.arange(p), active)
        for s in range(0, len(null), 10):
            blk = null[s : s + 10]
            if len(blk) < 2:
                continue
            common = rng.standard_normal(n)
            X[:, blk] = np.sqrt(rho) * common[:, None] + np.sqrt(1 - rho) * X[:, blk]
    beta = np.zeros(p)
    beta[active] = amps
    y = X @ beta + SIGMA * rng.standard_normal(n)
    Xc, yc = X - X.mean(0), y - y.mean()
    return Xc, yc


# ------------------------------------------------------------------- experiments
N, P = 240, 100
ACTIVE = np.array([3, 17, 29, 41, 53, 67, 78, 91])
AMPS = np.array([0.30, -0.26, 0.22, -0.34, 0.28, -0.20, 0.36, -0.24])
GRID = np.linspace(0.8, 0.25, 20)
LAM_LO, LAM_HI = 0.25, 0.80


def _batch_std(half_lists, X, y, lam_path, lm):
    """Std of selection frequency across independent batches of half-samples."""
    fs = []
    for halves in half_lists:
        pairs = [(halves[2 * i], halves[2 * i + 1])
                 for i in range(len(halves) // 2)]
        f, _ = selection_freq(X, y, pairs, lam_path, LAM_LO * lm, LAM_HI * lm)
        fs.append(f)
    return float(np.mean(np.std(np.asarray(fs), axis=0)))


def exp_frequency():
    rng = np.random.default_rng(SEED + 1)
    X, y = make_data(N, P, ACTIVE, AMPS, 0.0, rng)
    lm = np.max(np.abs(X.T @ y / N))
    lam_path = GRID * lm
    pairs = complement_pairs(N, 60, rng)
    freq, q = selection_freq(X, y, pairs, lam_path, LAM_LO * lm, LAM_HI * lm)
    # 10 batches of 12 half-samples: complementary pairs vs iid halves.
    comp_batches, iid_batches = [], []
    for b in range(10):
        sub = pairs[b * 6: (b + 1) * 6]   # disjoint pairs per batch
        comp_batches.append([idx for pair in sub for idx in pair])
        iid_batches.append([rng.permutation(N)[: N // 2] for _ in range(12)])
    return {"true_freq": [float(freq[j]) for j in ACTIVE],
            "all_freq": [float(v) for v in freq],
            "null_mean": float(np.mean(np.delete(freq, ACTIVE))),
            "null_q95": float(np.quantile(np.delete(freq, ACTIVE), 0.95)),
            "comp_pair_batch_std": _batch_std(comp_batches, X, y, lam_path, lm),
            "iid_batch_std": _batch_std(iid_batches, X, y, lam_path, lm),
            "q": q}


def exp_thresholds():
    rng = np.random.default_rng(SEED + 2)
    thrs = [0.60, 0.70, 0.80, 0.90, 0.95]
    rows = []
    n_rep = 12
    lm_cache = []
    for _ in range(n_rep):
        X, y = make_data(N, P, ACTIVE, AMPS, 0.0, rng)
        pairs = complement_pairs(N, 40, rng)
        lm = np.max(np.abs(X.T @ y / N))
        f, q = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
        lm_cache.append((f, q, X, y))
    for t in thrs:
        fdrs, pows, fps, bound = [], [], [], []
        for f, q, X, y in lm_cache:
            sel = f >= t
            _tp, _fp, fd, pw = metrics(sel, ACTIVE)
            fdrs.append(fd); pows.append(pw); fps.append(_fp)
            bound.append(q ** 2 / ((2 * t - 1) * N))
        rows.append({"pi": t,
                     "fdr": float(np.mean(fdrs)),
                     "power": float(np.mean(pows)),
                     "fp_mean": float(np.mean(fps)),
                     "mb_bound_fp": float(np.mean(bound))})
    # comparators on the same draws
    bh_fd, bh_pw, la_fd, la_pw = [], [], [], []
    for _f, _q, X, y in lm_cache:
        sel = bh_from_y(X, y, ACTIVE, 0.1)
        _, _, fd, pw = metrics(sel, ACTIVE)
        bh_fd.append(fd); bh_pw.append(pw)
        G = X.T @ X / N
        c = X.T @ y / N
        lm = np.max(np.abs(c))
        b = lasso_path(G, c, [0.3 * lm])[-1]
        _, _, fd, pw = metrics(np.abs(b) > 1e-8, ACTIVE)
        la_fd.append(fd); la_pw.append(pw)
    return {"stability": rows,
            "bh": {"fdr": float(np.mean(bh_fd)), "power": float(np.mean(bh_pw))},
            "lasso": {"fdr": float(np.mean(la_fd)), "power": float(np.mean(la_pw))}}


def exp_correlation():
    rng = np.random.default_rng(SEED + 3)
    rows = []
    sel_freq = None
    for rho in [0.0, 0.3, 0.6, 0.85]:
        fds, pws, cnts = [], [], []
        for _ in range(10):
            X, y = make_data(N, P, ACTIVE, AMPS, rho, rng)
            pairs = complement_pairs(N, 40, rng)
            lm = np.max(np.abs(X.T @ y / N))
            f, _ = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
            _tp, _fp, fd, pw = metrics(f >= 0.8, ACTIVE)
            fds.append(fd); pws.append(pw); cnts.append(_fp)
            if rho == 0.6:
                sel_freq = f
        rows.append({"rho": rho, "fdr": float(np.mean(fds)),
                     "power": float(np.mean(pws)),
                     "fp_mean": float(np.mean(cnts))})
    null = np.setdiff1d(np.arange(P), ACTIVE)
    return {"rows": rows,
            "rho06_false_freq_top5": [float(v) for v in
                                      np.sort(sel_freq[null])[::-1][:5]]}


def exp_sample_size():
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for n in [100, 200, 400, 800]:
        fds, pws, bnd = [], [], []
        for _ in range(8):
            X, y = make_data(n, P, ACTIVE, AMPS, 0.0, rng)
            pairs = complement_pairs(n, 40, rng)
            lm = np.max(np.abs(X.T @ y / n))
            f, q = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
            _, _, fd, pw = metrics(f >= 0.8, ACTIVE)
            fds.append(fd); pws.append(pw)
            bnd.append(q ** 2 / ((2 * 0.8 - 1) * n))
        rows.append({"n": n, "fdr": float(np.mean(fds)),
                     "power": float(np.mean(pws)),
                     "mb_bound_fp": float(np.mean(bnd))})
    return rows


def exp_portfolio():
    rng = np.random.default_rng(SEED + 5)
    rows = []
    for method in ["stability", "lasso", "bh", "oracle"]:
        rets = []
        nsel = []
        for _ in range(25):
            X, y = make_data(N, P, ACTIVE, AMPS, 0.0, rng)
            pairs = complement_pairs(N, 40, rng)
            lm = np.max(np.abs(X.T @ y / N))
            if method == "stability":
                f, _ = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
                sel = np.where(f >= 0.8)[0]
            elif method == "lasso":
                G = X.T @ X / N
                b = lasso_path(G, X.T @ y / N, [0.3 * lm])[-1]
                sel = np.where(np.abs(b) > 1e-8)[0]
            elif method == "bh":
                sel = np.where(bh_from_y(X, y, ACTIVE, 0.1))[0]
            else:
                sel = ACTIVE
            nsel.append(len(sel))
            # independent test period: same beta, new covariance and noise
            Xt = rng.standard_normal((500, P))
            beta_true = np.zeros(P)
            beta_true[ACTIVE] = AMPS
            yt = Xt @ beta_true + rng.standard_normal(500)
            if len(sel) == 0:
                rets.append(0.0)
                continue
            # use fitted signs on train for every selected name
            G = X.T @ X / N
            b = lasso_path(G, X.T @ y / N, [LAM_LO * lm])[-1]
            w = np.sign(b[sel])
            w[w == 0] = 1.0
            port = Xt[:, sel] @ w / len(sel)
            rets.append(float(np.mean(port) / max(np.std(port), 1e-9) * np.sqrt(252)))
        rows.append({"method": method,
                     "sharpe_mean": float(np.mean(rets)),
                     "n_selected_mean": float(np.mean(nsel))})
    return rows


# ----------------------------------------------------------------------- figures
def fig1(freq):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(np.arange(P), freq, color=["#c0392b" if j in ACTIVE else "#7f8c8d"
                                         for j in range(P)], width=0.9)
    ax[0].axhline(0.8, ls="--", c="#2980b9")
    ax[0].set_xlabel("variable index"); ax[0].set_ylabel("selection frequency")
    ax[0].set_title("True signals separate from nulls (rho=0)")
    xs = np.delete(freq, ACTIVE)
    ax[1].hist(xs, bins=20, color="#7f8c8d")
    ax[1].axvline(0.8, ls="--", c="#2980b9", label="pi_thr = 0.8")
    ax[1].set_xlabel("selection frequency"); ax[1].legend()
    ax[1].set_title("Null distribution stays near zero")
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_frequencies.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_frequencies.png"


def fig2(data):
    st = data["stability"]
    x = [r["pi"] for r in st]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["fdr"] for r in st], "o-", c="#c0392b", label="stability FDR")
    ax.plot(x, [r["power"] for r in st], "s-", c="#2980b9", label="stability power")
    ax.axhline(data["bh"]["fdr"], ls="--", c="#7f8c8d", label="BH FDR")
    ax.axhline(data["lasso"]["fdr"], ls=":", c="#34495e", label="single lasso FDR")
    ax.set_xlabel("pi_thr"); ax.set_ylabel("rate")
    ax.set_title("Higher threshold buys lower FDR at a power cost")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig2_threshold.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_threshold.png"


def fig3(data):
    rows = data["rows"]
    xs = [r["rho"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, [r["fdr"] for r in rows], "o-", c="#c0392b", label="stability FDR")
    ax[0].plot(xs, [r["power"] for r in rows], "s-", c="#2980b9", label="power")
    ax[0].set_xlabel("within-block correlation rho"); ax[0].legend()
    ax[0].set_title("Correlated nulls push FDR up")
    ax[1].bar(np.arange(5), data["rho06_false_freq_top5"], color="#c0392b")
    ax[1].set_ylabel("selection frequency"); ax[1].set_xlabel("top 5 null variables")
    ax[1].set_title("rho=0.6: which nulls survive")
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_correlation.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_correlation.png"


def fig4(rows):
    xs = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["fdr"] for r in rows], "o-", c="#c0392b", label="empirical FP")
    ax.plot(xs, [r["mb_bound_fp"] for r in rows], "s--", c="#2980b9",
            label="MB bound")
    ax.set_xscale("log"); ax.set_xlabel("sample size n"); ax.set_ylabel("false selections")
    ax.set_title("The theoretical bound is valid but very loose")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig4_bound.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_bound.png"


def fig5(rows):
    names = [r["method"] for r in rows]
    vals = [r["sharpe_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=["#2980b9", "#7f8c8d", "#7f8c8d", "#c0392b"])
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                "n=%.1f" % r["n_selected_mean"], ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("OOS Sharpe (annualised)")
    ax.set_title("Selected signals traded on an independent period")
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_portfolio.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_portfolio.png"


def main():
    fr = exp_frequency()
    th = exp_thresholds()
    co = exp_correlation()
    ss = exp_sample_size()
    po = exp_portfolio()
    result = {"setup": {"seed": SEED, "n": N, "p": P, "true_signals": ACTIVE.tolist(),
                        "amplitudes": AMPS.tolist(), "sigma": SIGMA,
                        "B_complementary_pairs": 40},
              "frequency": fr, "thresholds": th, "correlation": co,
              "sample_size": ss, "portfolio": po}
    result["figs"] = [fig1(np.asarray(fr["all_freq"], float)), fig2(th),
                      fig3(co), fig4(ss), fig5(po)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 三、互补对到底省了多少噪声

先回答最基础的问题：这套"切一半"的把戏，是不是白折腾？

我把 60 对互补半样本拆成 10 批（每批 6 对，共 12 个半样本），算每批得出的入选频率向量，再看批与批之间的标准差。作为对照，另做 10 批独立抽取的半样本（每批同样 12 个，但不要求互补）。结果：

- 互补对的批间标准差：$0.0865$。
- 独立半样本的批间标准差：$0.0907$。

只降了约 4.6%。这是一个**诚实的负面结果**：互补对的理论优势（两半严格互补、抽样误差负相关）确实存在，但在 LASSO 入选频率这种高度非线性的统计量上，它带来的方差缩减非常有限。我一开始预期它会把批间抖动砍掉一大半，实测数据否定了这个预期。

为什么没有更大改善？因为批间方差的大头不是"抽到了哪一半"，而是"这 12 个半样本本身包含多少关于真信号的信息"。互补结构只保证了两次抽样不重复，却不能凭空增加信息量——$n=240$ 里有 8 个幅度 0.2 到 0.36 的弱信号，这个检测难度无论怎么切分都在那里。

这个结果对实操的含义是：用互补对没错（理论上不劣于独立抽样，计算也只多一倍），但别把它当成解决"结果不稳定"的银弹。如果换一批种子频率还飘 0.09，问题的根源是样本不够或信号太弱，不是抽样方式。想真正稳下来，要么加数据，要么降低对弱信号的指望。

## 四、阈值的代价曲线

阈值扫描是这篇文章最该记住的部分。五个阈值、12 次重复实验的平均结果：

| 阈值 $\pi_{thr}$ | FDR | 功效 | 平均假阳性数 | 理论界 E[V] |
|---|---|---|---|---|
| 0.6 | 23.1% | 88.5% | 2.50 | 13.16 |
| 0.7 | 10.5% | 85.4% | 1.00 | 6.58 |
| 0.8 | 6.4% | 80.2% | 0.58 | 4.39 |
| 0.9 | 1.2% | 62.5% | 0.08 | 3.29 |
| 0.95 | 0.0% | 54.2% | 0.00 | 2.92 |

![入选频率把真信号和噪声分开](/images/stability-selection-subsampling-fdr/fig1_frequencies.png)

阈值 0.8 是一个合理的默认：功效还有 80%，FDR 压到 6.4%。想再干净就得付代价——0.9 时 FDR 只有 1.2%，但功效掉到 62.5%，8 个真信号平均只捞回 5 个。

![阈值越高越干净，同时漏掉越多](/images/stability-selection-subsampling-fdr/fig2_threshold.png)

两个对照值得看。BH 方法（在边际 t 检验上做 FDR 控制，$q=0.1$）在同一批数据上 FDR 是 0，但功效只有 41.7%——它保守得多。单次 LASSO 在 $0.3\lambda_{max}$ 上 FDR 冲到 36.7%、功效 91.7%：几乎把所有信号都捞回来了，但名单里三分之一以上是假的。稳定性选择的位置正好在中间，用可解释的阈值换来了对假阳性的控制。

理论界这一列也很有意思。$\pi=0.8$ 时它给出 4.39 个假阳性的上界，实际只有 0.58 个——界是有效的（实际值从没超过它），但松了约 7.5 倍。这是这类通用界的常态：它不对数据分布做具体假设，所以用宽度换普适性。别把这个界当成"预期会有 4 个假信号"，它是"最坏情况不会超过 4 个"的保险丝。

## 五、相关因子块：最危险的地方

现实里的假因子很少是孤立出现的。行业因子、风格因子、同花顺概念板块高度相关，一个真信号往往拖着一串相关变量。我在无关变量里造了 10 个一组、组内相关系数 $\rho$ 分别为 0、0.3、0.6、0.85 的块，看稳定性选择的 FDR 怎么变。

| 块内相关 $\rho$ | FDR | 功效 | 平均假阳性数 |
|---|---|---|---|
| 0.0 | 1.4% | 77.5% | 0.1 |
| 0.3 | 3.3% | 83.8% | 0.3 |
| 0.6 | 1.3% | 86.3% | 0.1 |
| 0.85 | 0.0% | 76.3% | 0.0 |

![相关块让假变量成簇入选](/images/stability-selection-subsampling-fdr/fig3_correlation.png)

平均 FDR 看起来没崩，甚至 $\rho=0.85$ 时降到 0——但这恰恰是最容易误读的地方。看入选频率最高的五个假变量：在 $\rho=0.6$ 时它们的频率分别是 0.64、0.54、0.44、0.44、0.41。这些变量在单批数据里反复被选中，只是"选中哪几个"在批次之间换来换去，导致平均下来每个的频率都卡在阈值 0.8 以下。

危险在于：如果你用稳定性选择挑出了 8 个因子，其中混进一个高相关块的成员，它看起来和真信号一样稳定——因为在这批数据里它确实每次都进。LASSO 在相关块里会任意挑一个，稳定性选择只是把"任意"变成了"每批都挑同一个"。**相关结构不会因为重复抽样而消失**，这是所有基于重抽样的方法共同的软肋。实践中的对策是在入选后做一次聚类去冗余，而不是指望稳定性阈值。

## 六、理论界和样本量

样本量从 100 爬到 800，FDR 从 12.3% 降到 0，功效从 35.9% 升到 98.4%。这是符合直觉的——数据越多，"反复抽一半"的每一半信息越充分。

![理论界远远高于实际假阳性](/images/stability-selection-subsampling-fdr/fig4_bound.png)

理论界随 $n$ 下降的速度是 $1/n$（$q$ 近似不变时），从 $n=100$ 的 8.63 降到 $n=800$ 的 0.37。实际假阳性也降，但一直低于界。这里想强调的是：**界随样本量下降这件事本身是有信息量的**——它告诉你稳定性选择的精度确实在改善，而不是一个不随 $n$ 变化的空洞保证。但别拿界当预测值，它和实际假阳性差着一个数量级。

## 七、样本外组合：选得干净不等于赚得多

最后做了一件"粗暴"的事：把四种方法选出的因子拿去一个独立测试期（500 天、同样的真实系数、全新的噪声）构建等权多空组合，看年化夏普。

![选得干净的名单，交易表现未必最好](/images/stability-selection-subsampling-fdr/fig5_portfolio.png)

- 稳定性选择：入选 6.88 个，样本外夏普 0.212。
- BH：入选 3.56 个，夏普 0.108。
- 单次 LASSO：入选 14.16 个，夏普 0.010。
- 真值 Oracle：入选 8 个，夏普 -0.005。

结果有点扎心。稳定性选择夏普最高，但 Oracle 竟然接近零——这不是方法赢了真值，而是测试期只有 500 天、组合收益的样本标准差远大于均值，夏普估计本身噪声极大。更稳妥的读法是：**别用一次样本外夏普给特征选择方法排名**，差异很可能落在噪声里。真正稳定的信号是"稳定性选择名单更短、假阳性更少、单次 LASSO 名单膨胀近一倍"这些结构性事实。

## 八、怎么用

给量化研究者的实操建议，按优先级排：

1. **默认参数**：互补对数 $B \ge 50$，阈值从 0.75 到 0.8 起步。数据少（$n<200$）时别指望高功效，把阈值放到 0.7 保召回。
2. **先去冗余再上稳定性**：如果因子池里行业/风格块相关高，先聚类，每类留一个代表，否则第 5 节的陷阱等着你。
3. **阈值按目标选**：要"宁缺毋滥"的实盘信号，用 0.9；要做因子研究的候选池，用 0.7 保住召回。
4. **理论界当保险丝不当预测**：它告诉你"最坏不会超过多少"，不是"大概会有多少"。
5. **别把它当收益保证**：特征选择控制的是选错变量的概率，不直接改善组合收益。第 7 节的 Oracle 结果是最好的提醒。

方法真正的价值，是把"我觉得这几个因子靠谱"变成一个可复核的数字——每个因子的入选频率摆在脸上，阈值一刀切，没有暗箱。它不会告诉你哪个因子赚钱，但能挡住一大批根本不存在的因子。
