---
title: "AUC 一模一样，钱却差 3.9 倍：概率校准才是方向预测能不能下注的开关"
description: "三份 AUC 完全相同的方向预测，凯利对数增长差 3.9 倍。在真值已知的逻辑概率路径上现场对拍：12000 天测试集里 sigmoid(2.2·logit p) 与 sigmoid(0.55·logit p) 的 AUC 都是 0.757426、方向命中率都是 0.69375，但 Brier 是 0.214131 对 0.206283（真值 0.198856），公平赔率凯利日均对数增长 0.028349 对 0.092052。校准不是免费的：n=50 时 Platt 与保序回归的样本外 Brier 0.224935 / 0.218080 都输给原始分 0.216207；基准率从 -0.20 跳到 +0.50 后，Platt 与真概率的差距从 0.000012 拉到 0.014846。ECE 自己也有噪声：把真概率分 200 箱仍量出 0.112944，log-log 斜率 0.497565。校准残差一阶自相关 0.117423，朴素 t 检验把 5% 误报做成 0.21，HAC 40 阶与区块自助法 L=40 才降到 0.060 / 0.0733。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-12T09:30:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 概率预测
  - Python
language: zh
difficulty: advanced
---

## 一、三条完全重合的排序，三种完全不同的下注结果

方向预测报告里最常见的一行是「AUC 0.76」。这个数只回答一件事：随机抽一个上涨日和一个下跌日，模型给上涨日打分更高的概率有多大。它对任何严格单调变换都不敏感——把分数整体放大、压缩、取对数，AUC 一个字都不变。可一旦要把分数变成仓位，用到的就不是排序而是概率的数值本身。

我造了一个真值已知的过程来把这件事量化。潜变量 $x_t$ 是 AR(1)、$\phi=0.85$，$z_t$ 独立标准正态，真实概率 $p_t=\sigma(-0.20+0.9x_t+0.6z_t)$，$y_t\sim\mathrm{Bernoulli}(p_t)$。测试集 12000 天。三份预测分别是真概率 $p_t$、过度自信的 $\sigma(2.2\,\mathrm{logit}\,p_t)$、以及保守的 $\sigma(0.55\,\mathrm{logit}\,p_t)$。三者严格单调等价。

实测 AUC 全部是 0.757426，极差精确为 0，0.5 阈值下的方向命中率也全部是 0.69375。但 Brier 分数分别是 0.198856、0.214131、0.206283，十等分分位箱的 ECE 是 0.011186、0.114214、0.077350，对数损失 0.582050、0.664798、0.601095。换成公平赔率下的凯利式下注 $f=2q-1$，日均对数增长是 0.111097、0.028349、0.092052：真概率版本是过度自信版本的 3.9 倍；日对数增长低于 −1 的天数占比 2.475%、11.625%、0.1%。同一个排序，一个稳定赚钱，一个每隔十天就挨一记重锤。

![可靠性图与三份预测的概率分布](/images/probability-calibration-brier-isotonic/fig1_reliability.png)

Brier 是严格恰当评分规则，唯一最小值在真概率处；AUC 只关心次序。两者衡量的根本不是一件事。更麻烦的是它们可以给出相反的排名：只用 $x$ 的部分信息模型 AUC 0.717184、Brier 0.213313，而用全部信息但放大 $a$ 倍 logit 的模型 AUC 高到 0.757426，Brier 却随 $a$ 从 0.198856（$a=1$）涨到 0.239516（$a=4$），在 $a^\*=2.151542$ 处越过部分信息模型。换句话说，只要过度自信系数超过 2.15，AUC 更高的那个模型在概率意义上更差。

![AUC 相同而 Brier 与凯利增长分化，以及排名反转的临界点](/images/probability-calibration-brier-isotonic/fig2_auc_brier_kelly.png)

## 二、把分数映射回概率：Platt 与保序回归

校准的做法是拿一段带标签的校准集，学一个从原始分数到概率的单调映射。Platt 缩放用带极小 L2 惩罚的牛顿法拟合 $\sigma(a m+b)$，只有两个参数；保序回归用 PAV（pool adjacent violators）合并违序块，是非参数的阶梯函数，灵活得多。原始分数取 $m=2.2\times\mathrm{logit}$，即真线性预测子的 2.2 倍，训练集与测试集都来自同一基准率。

校准集大小从 50 扫到 4000，每个规模独立重复 200 次，测试集每次 3000 个样本。原始分的样本外 Brier 稳定在 0.2162 附近（它没有学到任何东西）。**n=50 时两种校准反而都变差**：Platt 0.224935、保序 0.218080，都输给原始分 0.216207，而真概率是 0.200078；保序回归只有在 47.5% 的重复里赢过原始分。到 n=100 两者才一起越过原始分（0.210182 与 0.211209）。

这里有个容易写错的结论：按**均值**看，n=50 时保序回归 0.218080 优于 Platt 0.224935；但按逐次重复的对战胜率看，保序只赢 18%，在七个规模上分别是 18%、6.5%、2%、1%、1%、5%、6.5%，从没超过一半。原因是 Platt 在小样本上有少数次拟合发散，均值的标准误 0.003006 是保序 0.000772 的近四倍，几个极端值把均值拖高了。

把中位数摆出来就更清楚：n=50 时 Platt 的 Brier 中位数只有 0.206151，明显好于保序的 0.216326 与原始分的 0.216318，可它的均值是全场最差。也就是说，「保序均值更优」和「Platt 在 82% 的重复里更优」并不矛盾，风险集中在尾部而不是典型表现。用均值总结重尾损失分布会得出相反结论，必须同时看中位数与胜率；反过来，如果目标是控制最坏情况，Platt 那条更肥的尾巴又会让选择翻转回去。

还要说清一个有利于 Platt 的前提：本例的失校准恰好是「线性预测子乘 2.2」，两参数模型是正确设定。到 n=4000 时 Platt 0.200596 几乎贴住真概率 0.200517，保序 0.201321 反而多出约 0.0008，那是非参数灵活性付的代价。若真实失校准是非单调的，结论会反过来。

![校准样本量与 Platt / 保序回归的样本外 Brier，以及保序赢过原始分的比例](/images/probability-calibration-brier-isotonic/fig3_calibration_sample_size.png)

## 三、基准率一变，校准映射就过期了

把 6000 天切成两段：前 3000 天基准 log-odds 为 −0.20，后 3000 天跳到 +0.50。真概率均值从 0.452859 变成 0.612583，而模型仍沿用训练期的截距与 2.2 倍斜率，第二段的原始分均值只有 0.459834——它是系统性地把概率报低了。用前 1500 天拟合 Platt 得到 $a=0.456151$、$b=0.103142$。

第一段的测试集上，Platt 的 Brier 0.193670 几乎等于真概率的 0.193659（差 0.000012），原始分是 0.205885，校准是干净的胜利。到第二段，Platt 0.209644、保序 0.211398、原始分 0.237247，校准仍然有用，但与真概率 0.194798 的差距扩大到 0.014846，ECE 也从原始分 0.178322 只降到 0.118106。用第二段标签事后重拟合 Platt（这当然不可实盘实现）能回到 0.194816，说明坏的不是映射形式，而是映射的可迁移性。

用同一批随机数把跳变幅度从 −0.20 扫到 +1.1：Platt 的 Brier 从 0.203798 单调升到 0.224810，原始分从 0.221596 升到 0.270609，而真概率的 Brier 反而从 0.203407 降到 0.168167。真概率变好是因为概率更极端、事件更好预测，校准模型变差是因为它在跟一个自己不知道的平移对抗。这个反向剪刀差是「环境变了」最直观的信号。

![区制跳变前后的滚动 Brier，以及跳变幅度扫描](/images/probability-calibration-brier-isotonic/fig4_regime_shift.png)

## 四、ECE 是统计量，不是真理

ECE 依赖分箱。对上面那个过度自信模型，等宽十箱量出 0.127031，等频十箱量出 0.129498，差别不大——因为它的分数本来就铺得很开。换一个更阴险的构造：其余全部等于真概率，只把 $p$ 最高的 2% 强行改成 0.99。等宽十箱只给最右侧的箱子 2% 的权重，ECE 只有 0.012749；等频十箱强制每箱 10% 样本，尾部误差被完整计入，ECE 变成 0.020256，相差 1.59 倍。同一个模型，换个分箱就能从「基本校准」变成「明显失校准」。

更根本的是，即便预测**就是真概率**，ECE 也不是零。6000 个样本、等频十箱量出 0.019139。200 次重复、每次 2000 个样本、把真概率当预测，分箱数从 5 加到 200，ECE 依次是 0.018095、0.025158、0.035048、0.055221、0.079691、0.112944；对箱数做 log-log 回归斜率 0.497565，正好对应 $\sqrt{B/n}$ 的有限样本噪声。所以报 ECE 必须同时写清箱数与样本量，否则调一下 B 就能把结论翻过来。

![ECE 的分箱偏差与校准残差的序列相关检验](/images/probability-calibration-brier-isotonic/fig5_ece_bins_and_hac.png)

## 五、整体校准通过，残差里仍藏着可预测的东西

最后一盆冷水留给检验本身。构造一个「边际校准」但漏掉持续变量 $x$ 的模型 $q=\sigma(c^\*+0.6z)$：在 40 万样本的先导样本上用二分法解出 $c^\*=-0.171470$，使 $\mathbb E[q]=\mathbb E[p]=0.460381$。300 次重复、每次 2000 天，平均残差 −0.000996，原假设确实成立。但 $x$ 的自相关 0.85 会漏进残差，实测一阶自相关 0.117423。

于是朴素 iid t 检验把名义 5% 的误报做成 0.21；Bartlett 核的 Newey-West HAC 取 5 阶仍误报 0.14，取 20 阶 0.0733，40 阶 0.060；循环移动区块自助法块长 10 误报 0.1267，块长 40 降到 0.0733。HAC 标准误相对朴素标准误从 1.217 倍（5 阶）涨到 1.504 倍（40 阶）。两个诚实的边界：一是 HAC 与区块自助法各自也有带宽，带宽不足时「已经修正」是假象；二是即便带宽给够，0.060 与 0.0733 也没精确回到 0.05，有限样本下这些修正并不精确。另外这里只验证了边际校准，给定 $x$ 的条件校准按构造是失败的，残差自相关正是它的指纹。

## 六、结论与适用边界

报方向模型时，AUC、Brier（或对数损失）、以及注明箱数与样本量的 ECE 必须一起给，缺一个就可能把「排序好」误读成「概率准」。校准映射是在特定环境里估出来的统计量：样本量不够会过拟合，基准率一漂移就失效，残差里还有序列依赖，做区间或检验要用 HAC / 区块自助法并交代带宽。

本文所有 $p_t$ 来自已知生成过程，所以「真概率」不是估出来的；真实市场里概率不可观测，只能靠大样本频率近似，校准评估本身还要再吃一层估计误差。Platt 假设 sigmoid 型失校准，保序回归假设单调性；凯利用的是无成本公平赔率，没有手续费、冲击与赔率偏差；ECE 的可比性依赖固定分箱规则。这些都是把结论搬到实盘前必须重新算的账。

下面这段代码就是本文全部数值与五张图的来源，自包含可直接运行，输出与站点同目录的 results.json 逐项一致。

```python
"""Probability calibration: identical AUC, very different Brier scores.

This file IS the runnable code block embedded in the article
`probability-calibration-brier-isotonic`. Run it directly:

    python3 article_core_calibration_20261112pm.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default cwd).
Every number quoted in the article comes from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 2026111201
B0 = -0.20          # base log-odds in regime 1
BX, BZ = 0.90, 0.60  # true coefficients on the two signals
PHI = 0.85          # persistence of the latent signal x
OVER = 2.20         # over-confidence slope of the raw model


def sigmoid(v):
    return expit(v)


def logit(p):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


def auc(q, y):
    """Mann-Whitney AUC; monotone transforms of q give exactly the same value."""
    order = np.argsort(q, kind="mergesort")
    rank = np.empty(len(q), dtype=float)
    rank[order] = np.arange(1, len(q) + 1, dtype=float)
    n1 = float(y.sum())
    n0 = len(y) - n1
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(q, y):
    return float(np.mean((q - y) ** 2))


def logloss(q, y):
    p = np.clip(q, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(q, y, bins=10, mode="quantile"):
    if mode == "quantile":
        edges = np.quantile(q, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges[1:-1], q, side="right"), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(float(q[m].mean() - y[m].mean()))
    return float(out)


def kelly_log_growth(q, y):
    """Fair-odds binary bet f = 2q-1; wealth multiplier 2q on a win."""
    p = np.clip(q, 1e-9, 1 - 1e-9)
    return np.where(y == 1, np.log(2 * p), np.log(2 * (1 - p)))


# ---------------- DGP ----------------
def draw_path(rng, T, b, ar=True):
    z = rng.standard_normal(T)
    if ar:
        x = np.empty(T)
        x[0] = rng.standard_normal()
        e = rng.standard_normal(T)
        for t in range(1, T):
            x[t] = PHI * x[t - 1] + np.sqrt(1 - PHI ** 2) * e[t]
    else:
        x = rng.standard_normal(T)
    lin = b + BX * x + BZ * z
    p = sigmoid(lin)
    y = (rng.random(T) < p).astype(float)
    return x, z, lin, p, y


# ---------------- calibration maps ----------------
def platt_fit(m, y, l2=1e-6, iters=80):
    """Newton fit of sigmoid(a*m+b) by log loss, with a tiny L2 ridge."""
    a, b = 1.0, 0.0
    sm = float(m.mean())
    m = m - sm                      # centre for numerical stability
    for _ in range(iters):
        pr = sigmoid(a * m + b)
        g = pr - y
        w = pr * (1 - pr) + 1e-12
        grad = np.array([np.dot(g, m), g.sum()]) + l2 * np.array([a, b])
        H = np.array([[np.dot(w, m * m) + l2, np.dot(w, m)],
                      [np.dot(w, m), w.sum() + l2]])
        step = np.linalg.solve(H, grad)
        a -= step[0]
        b -= step[1]
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(a), float(b - a * sm)


def platt_pred(m, ab):
    a, b = ab
    return sigmoid(a * m + b)


def isotonic_fit(s, y):
    """PAV (pool adjacent violators), returning an interpolatable step map."""
    order = np.argsort(s, kind="mergesort")
    xs, ys = s[order], y[order].astype(float)
    X, Y, W = [], [], []
    for xi, yi in zip(xs, ys):
        X.append([xi]); Y.append(yi); W.append(1.0)
        while len(Y) >= 2 and Y[-2] > Y[-1] + 1e-15:
            w = W[-2] + W[-1]
            v = (Y[-2] * W[-2] + Y[-1] * W[-1]) / w
            X[-2] = X[-2] + X[-1]
            Y[-2] = v
            W[-2] = w
            X.pop(); Y.pop(); W.pop()
    xp, yp = [], []
    for blk, v in zip(X, Y):
        xp.extend([blk[0], blk[-1]]); yp.extend([v, v])
    xp, yp = np.asarray(xp), np.asarray(yp)
    keep = np.concatenate([[True], np.diff(xp) > 0])
    return xp[keep], yp[keep]


def isotonic_pred(xp, yp, s):
    return np.interp(s, xp, yp)


# ---------------- experiments ----------------
def run_all():
    S, F = {}, {}

    # --- Exp 1: identical AUC, different Brier / calibration / Kelly growth ---
    rng = np.random.default_rng(SEED + 1)
    x, z, lin, p, y = draw_path(rng, 12000, B0, ar=True)
    q_or = p.copy()
    q_ov = sigmoid(OVER * logit(p))
    q_un = sigmoid(0.55 * logit(p))
    models = {"oracle": q_or, "overconfident": q_ov, "underconfident": q_un}
    S["exp1"] = {}
    for name, q in models.items():
        g = kelly_log_growth(q, y)
        S["exp1"][name] = {
            "auc": auc(q, y), "brier": brier(q, y), "logloss": logloss(q, y),
            "ece_quantile10": ece(q, y, 10, "quantile"),
            "ece_equalwidth10": ece(q, y, 10, "width"),
            "directional_accuracy": float(np.mean((q >= 0.5) == (y == 1))),
            "kelly_mean_log_growth": float(g.mean()),
            "kelly_q05_log_growth": float(np.quantile(g, 0.05)),
            "kelly_frac_below_minus1": float(np.mean(g < -1.0)),
        }
    S["exp1"]["auc_max_spread"] = max(v["auc"] for v in S["exp1"].values() if isinstance(v, dict)) - \
                                  min(v["auc"] for v in S["exp1"].values() if isinstance(v, dict))

    # AUC-vs-Brier ranking reversal: full-information overconfident model vs
    # partial-information (x only) underconfident model, scan the slope a.
    q_partial = sigmoid(0.80 * (B0 + BX * x))
    b_part = brier(q_partial, y)
    a_grid = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0])
    br_full = np.array([brier(sigmoid(a * lin), y) for a in a_grid])
    # first a where the higher-AUC full model loses on Brier
    cross = np.where(br_full > b_part)[0]
    a_star = None
    if len(cross):
        j = int(cross[0])
        if j > 0:
            a_star = float(a_grid[j - 1] + (b_part - br_full[j - 1]) /
                           (br_full[j] - br_full[j - 1]) * (a_grid[j] - a_grid[j - 1]))
        else:
            a_star = float(a_grid[0])
    S["exp1"]["ranking_reversal"] = {
        "auc_full": auc(sigmoid(2.5 * lin), y), "auc_partial": auc(q_partial, y),
        "brier_partial": b_part,
        "a_grid": [float(v) for v in a_grid],
        "brier_full_grid": [float(v) for v in br_full],
        "a_star": a_star,
        "brier_full_at_2_5": float(br_full[np.argmin(abs(a_grid - 2.5))]),
    }
    F["exp1"] = dict(q_or=q_or, q_ov=q_ov, q_un=q_un, y=y,
                     a_grid=a_grid, br_full=br_full, b_part=b_part,
                     q_partial=q_partial, g_or=kelly_log_growth(q_or, y),
                     g_ov=kelly_log_growth(q_ov, y))

    # --- Exp 2: calibration sample size; isotonic overfits at small n ---
    n_grid = [50, 100, 200, 500, 1000, 2000, 4000]
    reps = 200
    n_test = 3000
    acc = {k: np.zeros((len(n_grid), reps)) for k in ["raw", "platt", "isotonic", "oracle"]}
    frac_iso_better = np.zeros(len(n_grid))
    frac_iso_beats_platt = np.zeros(len(n_grid))
    for i, n in enumerate(n_grid):
        rr = np.random.default_rng(SEED + 100 + i)
        for r in range(reps):
            _, _, lcal, pcal, ycal = draw_path(rr, n, B0, ar=False)
            _, _, lte, pte, yte = draw_path(rr, n_test, B0, ar=False)
            mcal, mte = OVER * lcal, OVER * lte
            qraw = sigmoid(mte)
            qpl = platt_pred(mte, platt_fit(mcal, ycal))
            xp, yp = isotonic_fit(mcal, ycal)
            qiso = isotonic_pred(xp, yp, mte)
            acc["raw"][i, r] = brier(qraw, yte)
            acc["platt"][i, r] = brier(qpl, yte)
            acc["isotonic"][i, r] = brier(qiso, yte)
            acc["oracle"][i, r] = brier(pte, yte)
        frac_iso_better[i] = np.mean(acc["isotonic"][i] < acc["raw"][i])
        frac_iso_beats_platt[i] = np.mean(acc["isotonic"][i] < acc["platt"][i])
    S["exp2"] = {
        "n_grid": n_grid, "reps": reps, "n_test": n_test,
        "brier_mean": {k: [float(v) for v in acc[k].mean(1)] for k in acc},
        "brier_median": {k: [float(v) for v in np.median(acc[k], axis=1)] for k in acc},
        "brier_se": {k: [float(v) for v in acc[k].std(1, ddof=1) / np.sqrt(reps)] for k in acc},
        "frac_iso_beats_raw": [float(v) for v in frac_iso_better],
        "frac_iso_beats_platt": [float(v) for v in frac_iso_beats_platt],
    }
    # A mean can be dragged up by rare divergences: report BOTH the mean and
    # the win rate, and define "beats" by the win rate, not by the mean.
    fw = np.array(S["exp2"]["frac_iso_beats_platt"])
    idx = np.where(fw > 0.5)[0]
    S["exp2"]["iso_beats_platt_from_n_by_winrate"] = int(n_grid[int(idx[0])]) if len(idx) else None
    for k in ["platt", "isotonic"]:
        f = np.array(S["exp2"]["brier_mean"][k]) < np.array(S["exp2"]["brier_mean"]["raw"])
        S["exp2"][f"{k}_beats_raw_from_n_by_mean"] = int(n_grid[int(np.argmax(f))]) if f.any() else None
    F["exp2"] = dict(n_grid=n_grid, acc=acc, frac=frac_iso_better)

    # --- Exp 3: prior shift breaks the learned calibration map ---
    T = 6000
    rng = np.random.default_rng(SEED + 300)
    x, z, _, _, y = draw_path(rng, T, B0, ar=True)   # y generated with regime-1 base rate
    # rebuild labels with a piecewise base rate
    bpath = np.concatenate([np.full(T // 2, B0), np.full(T - T // 2, 0.50)])
    p_true = sigmoid(bpath + BX * x + BZ * z)
    u = rng.random(T)                       # common random numbers for every scan
    y = (u < p_true).astype(float)
    m = OVER * (B0 + BX * x + BZ * z)               # model keeps the regime-1 intercept
    cal_sl = slice(0, 1500)
    ab = platt_fit(m[cal_sl], y[cal_sl])
    xp, yp = isotonic_fit(m[cal_sl], y[cal_sl])
    q_raw = sigmoid(m)
    q_pl = platt_pred(m, ab)
    q_iso = isotonic_pred(xp, yp, m)
    s1 = slice(1500, 3000)
    s2 = slice(3000, T)
    ab_post = platt_fit(m[s2], y[s2])                # post-hoc, NOT implementable
    q_post = platt_pred(m[s2], ab_post)
    S["exp3"] = {
        "shift_day": T // 2, "platt_a": ab[0], "platt_b": ab[1],
        "mean_p_regime1": float(p_true[s1].mean()),
        "mean_p_regime2": float(p_true[s2].mean()),
        "mean_qraw_regime2": float(q_raw[s2].mean()),
        "brier_regime1": {k: brier(q[s1], y[s1]) for k, q in
                          [("raw", q_raw), ("platt", q_pl), ("isotonic", q_iso),
                           ("oracle", p_true)]},
        "brier_regime2": {k: brier(q[s2], y[s2]) for k, q in
                          [("raw", q_raw), ("platt", q_pl), ("isotonic", q_iso),
                           ("oracle", p_true)]},
        "brier_regime2_post_hoc_platt": brier(q_post, y[s2]),
        "ece_regime2": {"raw": ece(q_raw[s2], y[s2], 10, "quantile"),
                        "platt": ece(q_pl[s2], y[s2], 10, "quantile")},
    }
    # shift-magnitude scan (reuse the same latent x,z and the same y-noise draws)
    shifts = [-0.2, 0.0, 0.2, 0.5, 0.8, 1.1]
    scan = []
    for b2 in shifts:
        pp = sigmoid(np.concatenate([np.full(T // 2, B0), np.full(T - T // 2, b2)]) + BX * x + BZ * z)
        yy = (u < pp).astype(float)
        scan.append({
            "b2": b2,
            "raw": brier(q_raw[s2], yy[s2]),
            "platt": brier(q_pl[s2], yy[s2]),
            "isotonic": brier(q_iso[s2], yy[s2]),
            "oracle": brier(pp[s2], yy[s2]),
        })
    S["exp3"]["shift_scan"] = scan
    # rolling Brier (window 250) for the figure
    win = 250
    def roll_brier(q):
        d = (q - y) ** 2
        c = np.concatenate([[0.0], np.cumsum(d)])
        return (c[win:] - c[:-win]) / win
    F["exp3"] = dict(roll_raw=roll_brier(q_raw), roll_pl=roll_brier(q_pl),
                     roll_iso=roll_brier(q_iso), roll_or=roll_brier(p_true),
                     shift_day=T // 2, win=win, scans=scan)

    # --- Exp 4: ECE is a statistic, not a truth ---
    rng = np.random.default_rng(SEED + 400)
    _, _, lin4, p4, y4 = draw_path(rng, 6000, B0, ar=False)
    q4 = sigmoid(OVER * lin4)
    # A forecast that is calibrated almost everywhere but destroys the top 2%
    # tail: equal-width bins give that tail almost no weight, quantile bins do.
    q_tail = p4.copy()
    cut = np.quantile(p4, 0.98)
    q_tail[p4 >= cut] = 0.99
    S["exp4"] = {
        "ece_over_equalwidth10": ece(q4, y4, 10, "width"),
        "ece_over_quantile10": ece(q4, y4, 10, "quantile"),
        "ece_oracle_quantile10": ece(p4, y4, 10, "quantile"),
        "tail_corrupted_frac": 0.02,
        "ece_tail_equalwidth10": ece(q_tail, y4, 10, "width"),
        "ece_tail_quantile10": ece(q_tail, y4, 10, "quantile"),
        "tail_bin_mass_equalwidth": float(np.mean(q_tail >= 0.9)),
    }
    bin_grid = [5, 10, 20, 50, 100, 200]
    rb = 200
    bias = np.zeros((len(bin_grid), rb))
    rr = np.random.default_rng(SEED + 401)
    for i, B in enumerate(bin_grid):
        for r in range(rb):
            _, _, _, pp, yy = draw_path(rr, 2000, B0, ar=False)
            bias[i, r] = ece(pp, yy, B, "quantile")   # pp IS the true probability
    S["exp4"]["bin_grid"] = bin_grid
    S["exp4"]["ece_bias_mean"] = [float(v) for v in bias.mean(1)]
    S["exp4"]["ece_bias_se"] = [float(v) for v in bias.std(1, ddof=1) / np.sqrt(rb)]
    S["exp4"]["sqrt_B_scaling_slope"] = float(
        np.polyfit(np.log(bin_grid), np.log(np.maximum(bias.mean(1), 1e-12)), 1)[0])
    F["exp4"] = dict(q4=q4, y4=y4, bin_grid=bin_grid, bias=bias)

    # --- Exp 5: calibration residuals inherit serial dependence ---
    # Fix c so the marginal model sigmoid(c + BZ*z) has the same mean as p.
    rp = np.random.default_rng(SEED + 500)
    zp = rp.standard_normal(400000)
    xp_ = rp.standard_normal(400000)
    target = float(sigmoid(B0 + BX * xp_ + BZ * zp).mean())
    lo, hi = -3.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(mid + BZ * zp).mean() < target:
            lo = mid
        else:
            hi = mid
    c_star = (lo + hi) / 2
    T5, R5 = 2000, 300
    maxlag, nb = 40, 200
    hac_lags, block_lens = [5, 20, 40], [10, 40]
    keys = ["naive"] + [f"hac{L}" for L in hac_lags] + [f"block{L}" for L in block_lens]
    rej = {k: 0 for k in keys}
    ac1_list, se_ratio5, se_ratio40, mu_list = [], [], [], []
    r5 = np.random.default_rng(SEED + 501)
    for rep in range(R5):
        x5 = np.empty(T5)
        x5[0] = r5.standard_normal()
        e5 = r5.standard_normal(T5)
        for t in range(1, T5):
            x5[t] = PHI * x5[t - 1] + np.sqrt(1 - PHI ** 2) * e5[t]
        z5 = r5.standard_normal(T5)
        p5 = sigmoid(B0 + BX * x5 + BZ * z5)
        y5 = (r5.random(T5) < p5).astype(float)
        q5 = sigmoid(c_star + BZ * z5)            # omits the persistent x
        res = y5 - q5                            # E[res] = 0 (marginally calibrated)
        ac1_list.append(float(np.corrcoef(res[:-1], res[1:])[0, 1]))
        mu = float(res.mean()); mu_list.append(mu)
        rc = res - mu
        naive_se = float(res.std(ddof=1) / np.sqrt(T5))
        gam = np.empty(maxlag + 1)
        gam[0] = float(np.dot(rc, rc) / T5)
        for k in range(1, maxlag + 1):
            gam[k] = float(np.dot(rc[k:], rc[:-k]) / T5)
        lr = {}
        for L in hac_lags:
            v = gam[0] + 2 * sum((1 - k / (L + 1)) * gam[k] for k in range(1, L + 1))
            lr[L] = float(np.sqrt(max(v, 1e-18) / T5))
        se_ratio5.append(lr[5] / naive_se)
        se_ratio40.append(lr[40] / naive_se)
        rej["naive"] += abs(mu / naive_se) > 1.96
        for L in hac_lags:
            rej[f"hac{L}"] += abs(mu / lr[L]) > 1.96
        for L in block_lens:                      # circular moving-block bootstrap
            nblk = int(np.ceil(T5 / L))
            starts = r5.integers(0, T5, size=(nb, nblk))
            idxb = ((starts[:, :, None] + np.arange(L)[None, None, :]) % T5).reshape(nb, -1)[:, :T5]
            bm = res[idxb].mean(1)
            lo_b, hi_b = np.quantile(bm, [0.025, 0.975])
            rej[f"block{L}"] += (lo_b > 0) or (hi_b < 0)
    S["exp5"] = {
        "T": T5, "reps": R5, "bootstrap_reps": nb,
        "hac_lags": hac_lags, "block_lengths": block_lens,
        "pilot_c_star": float(c_star), "pilot_mean_p": target,
        "mean_residual_across_reps": float(np.mean(mu_list)),
        "lag1_autocorr_mean": float(np.mean(ac1_list)),
        "hac5_over_naive_se_ratio": float(np.mean(se_ratio5)),
        "hac40_over_naive_se_ratio": float(np.mean(se_ratio40)),
        "reject": {k: v / R5 for k, v in rej.items()},
    }
    F["exp5"] = dict(ac1=ac1_list, rejs=S["exp5"]["reject"])
    return S, F


# ---------------- figures ----------------
def make_figures(F):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    # fig 1: reliability diagram + score distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for q, lab, col in [(F["exp1"]["q_or"], "oracle p", "#16a34a"),
                        (F["exp1"]["q_ov"], "overconfident x2.2", "#d62728"),
                        (F["exp1"]["q_un"], "underconfident x0.55", "#2563eb")]:
        e = np.quantile(q, np.linspace(0, 1, 11)); e[0], e[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(e[1:-1], q, side="right"), 0, 9)
        cx = [q[idx == b].mean() for b in range(10)]
        cy = [F["exp1"]["y"][idx == b].mean() for b in range(10)]
        ax[0].plot(cx, cy, "o-", color=col, label=lab, ms=4, lw=1.4)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax[0].set_xlabel("mean predicted probability (quantile bins)")
    ax[0].set_ylabel("observed frequency"); ax[0].set_title("Reliability diagram (10 quantile bins)")
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[1].hist(F["exp1"]["q_or"], bins=40, range=(0, 1), histtype="step",
               density=True, color="#16a34a", label="oracle p")
    ax[1].hist(F["exp1"]["q_ov"], bins=40, range=(0, 1), histtype="step",
               density=True, color="#d62728", label="overconfident")
    ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("density")
    ax[1].set_title("Same ranking, different spread"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_reliability.png", dpi=130); plt.close(fig)

    # fig 2: AUC identical / Brier differs / Kelly / ranking reversal scan
    q_or, q_ov, q_un, y = F["exp1"]["q_or"], F["exp1"]["q_ov"], F["exp1"]["q_un"], F["exp1"]["y"]
    names = ["oracle", "overconfident", "underconfident"]
    qs = [q_or, q_ov, q_un]
    aucs = [auc(q, y) for q in qs]
    brs = [brier(q, y) for q in qs]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    xpos = np.arange(3)
    ax[0].bar(xpos - 0.2, aucs, 0.4, color="#2563eb", label="AUC")
    ax[0].set_ylim(0.5, 0.9); ax[0].set_xticks(xpos); ax[0].set_xticklabels(names, fontsize=7.5)
    for i, v in enumerate(aucs): ax[0].text(i - 0.2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax[0].set_title("AUC: identical (monotone transforms)")
    ax2 = ax[0].twinx(); ax2.bar(xpos + 0.2, brs, 0.4, color="#d62728", label="Brier")
    ax2.set_ylim(0, max(brs) * 1.45); ax2.grid(False)
    for i, v in enumerate(brs): ax2.text(i + 0.2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("Brier (lower is better)", color="#d62728")
    g_or, g_ov = F["exp1"]["g_or"], F["exp1"]["g_ov"]
    ax[1].plot(np.cumsum(g_or), color="#16a34a", label="oracle", lw=1.3)
    ax[1].plot(np.cumsum(g_ov), color="#d62728", label="overconfident", lw=1.3)
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("test day"); ax[1].set_ylabel("cumulative log wealth")
    ax[1].set_title("Fair-odds Kelly: identical AUC, different wealth")
    ax[1].legend(frameon=False, fontsize=8)
    ag, bg, bp = F["exp1"]["a_grid"], F["exp1"]["br_full"], F["exp1"]["b_part"]
    ax[2].plot(ag, bg, "o-", color="#7c3aed", lw=1.4, ms=4, label="full-info model (higher AUC)")
    ax[2].axhline(bp, color="#111827", ls="--", lw=1.2, label="partial-info model (lower AUC)")
    ax[2].set_xlabel("over-confidence slope a"); ax[2].set_ylabel("Brier")
    ax[2].set_title("Where Brier reverses the AUC ranking")
    ax[2].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig2_auc_brier_kelly.png", dpi=130); plt.close(fig)

    # fig 3: calibration sample size
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ng, acc, frac = F["exp2"]["n_grid"], F["exp2"]["acc"], F["exp2"]["frac"]
    cols = {"raw": "#d62728", "platt": "#2563eb", "isotonic": "#f59e0b", "oracle": "#16a34a"}
    for k, v in acc.items():
        m, s = v.mean(1), v.std(1, ddof=1) / np.sqrt(v.shape[1])
        ax[0].errorbar(ng, m, yerr=s, label=k, color=cols[k], lw=1.5, marker="o", ms=3.5,
                       capsize=2)
    ax[0].set_xscale("log"); ax[0].set_xlabel("calibration sample size n")
    ax[0].set_ylabel("out-of-sample Brier"); ax[0].set_title("Isotonic needs data; Platt needs 2 params")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(ng, frac, "o-", color="#7c3aed", lw=1.5, ms=4)
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("calibration sample size n"); ax[1].set_ylabel("P(isotonic beats raw)")
    ax[1].set_title("Fraction of replications where isotonic helps")
    fig.tight_layout(); fig.savefig(OUT / "fig3_calibration_sample_size.png", dpi=130); plt.close(fig)

    # fig 4: regime shift
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for arr, lab, col in [(F["exp3"]["roll_raw"], "raw", "#d62728"),
                          (F["exp3"]["roll_pl"], "Platt", "#2563eb"),
                          (F["exp3"]["roll_iso"], "isotonic", "#f59e0b"),
                          (F["exp3"]["roll_or"], "oracle", "#16a34a")]:
        ax[0].plot(arr, lw=1.0, color=col, label=lab, alpha=0.9)
    ax[0].axvline(F["exp3"]["shift_day"] - F["exp3"]["win"], color="k", ls="--", lw=1.2)
    ax[0].text(F["exp3"]["shift_day"] - F["exp3"]["win"] + 40, ax[0].get_ylim()[1] * 0.95,
               "prior shift", fontsize=8)
    ax[0].set_xlabel("day"); ax[0].set_ylabel("rolling Brier (window 250)")
    ax[0].set_title("A calibration map is conditional on the regime")
    ax[0].legend(frameon=False, fontsize=8, ncol=2)
    sc = F["exp3"]["scans"]
    b2 = [s["b2"] for s in sc]
    for key, col in [("raw", "#d62728"), ("platt", "#2563eb"),
                     ("isotonic", "#f59e0b"), ("oracle", "#16a34a")]:
        ax[1].plot(b2, [s[key] for s in sc], "o-", color=col, lw=1.4, ms=4, label=key)
    ax[1].set_xlabel("post-shift base log-odds b2"); ax[1].set_ylabel("Brier on regime 2")
    ax[1].set_title("The larger the shift, the less the map transfers")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_regime_shift.png", dpi=130); plt.close(fig)

    # fig 5: ECE binning + serial dependence in calibration tests
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    bg, bias = F["exp4"]["bin_grid"], F["exp4"]["bias"]
    ax[0].errorbar(bg, bias.mean(1), yerr=bias.std(1, ddof=1) / np.sqrt(bias.shape[1]),
                   color="#7c3aed", marker="o", ms=4, capsize=2, lw=1.4,
                   label="ECE of a TRUE probability")
    ax[0].set_xscale("log"); ax[0].set_xlabel("number of bins B")
    ax[0].set_ylabel("measured ECE"); ax[0].set_title("ECE is upward-biased even when calibrated")
    ax[0].axhline(0, color="k", lw=0.8, ls="--"); ax[0].legend(frameon=False, fontsize=8)
    rj = F["exp5"]["rejs"]
    bars = ["naive iid", "HAC 5", "HAC 20", "HAC 40", "block 10", "block 40"]
    vals = [rj["naive"], rj["hac5"], rj["hac20"], rj["hac40"], rj["block10"], rj["block40"]]
    ax[1].bar(bars, vals, color=["#d62728", "#93c5fd", "#60a5fa", "#2563eb", "#86efac", "#16a34a"])
    ax[1].axhline(0.05, color="k", ls="--", lw=1.1, label="nominal 5%")
    for i, v in enumerate(vals): ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax[1].set_ylim(0, max(vals) * 1.35 + 0.02); ax[1].set_ylabel("rejection rate of a TRUE null")
    ax[1].set_title("Bandwidth matters for calibration tests")
    ax[1].tick_params(axis="x", labelsize=7.5)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig5_ece_bins_and_hac.png", dpi=130); plt.close(fig)


def _ser(o):
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _ser(o.tolist())
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, F = run_all()
    make_figures(F)
    (OUT / "results.json").write_text(
        json.dumps(_ser(S), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("AUC spread across monotone transforms =", S["exp1"]["auc_max_spread"])
    print("Brier oracle / over / under =", S["exp1"]["oracle"]["brier"],
          S["exp1"]["overconfident"]["brier"], S["exp1"]["underconfident"]["brier"])
    print("calibration beats raw from n =", S["exp2"]["platt_beats_raw_from_n_by_mean"],
          S["exp2"]["isotonic_beats_raw_from_n_by_mean"])
    print("reject rates =", S["exp5"]["reject"])
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
```
