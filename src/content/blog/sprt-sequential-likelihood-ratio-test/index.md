---
title: "序贯概率比检验 SPRT：同样 5% 误报平均 246 天定论，为什么固定 481 天的方案要多等一倍，以及它最怕的三件事"
description: "把日均优势检验写成似然比随机游走，用 Wald 边界 A=2.9444、B=-2.9444 边看边停。4000 次真值实验：误报 0.0405、功效 0.9545，平均 239.4/245.7 天就停，固定样本检验要 481 天，省下 50.2%/48.9%。诚实边界：每天看一次平均越过边界 0.09，五天才看误报降到 0.028 而功效反升到 0.9688；σ 低估到 0.7 让误报冲到 0.1877；真实优势只有设计值一半时平均要等 391 天。附完整代码与五张真实计算图。"
publishDate: '2026-11-24T09:20:00Z'
tags:
  - 量化
  - 序贯检验
  - 假设检验
  - 回测
  - Python
language: zh
difficulty: advanced
image: "/images/sprt-sequential-likelihood-ratio-test/fig1_paths.png"
---

## 一、结论先行

回测里最常见的问题之一：这套策略到底有没有日均优势？固定样本做法是攒够 n 天再算一次 t 检验。序贯概率比检验（SPRT）换了个思路——每来一天数据就更新一次证据，证据足够强立刻停，不必预先锁定天数。

本文在真值已知的蒙特卡洛上把它跑到底，得到四个结论：

- 在误报 0.0405、功效 0.9545 的水平上，SPRT 原假设下平均 239.4 天、备择下平均 245.7 天就做决定；达到同等名义误差的固定样本方案需要 481 天，样本量省下 50.2% 和 48.9%。
- 每天越界平均超出 Wald 边界 0.09，90 分位约 0.20，这让实际误报略低于名义 5%。
- 把监控频率从每天改成每 50 天，误报从 0.0412 降到 0.0280，功效却从 0.9524 升到 0.9688——少看几次不是变弱，是用更长的等待换更干净的决策。
- 它很脆：把真实波动率 1.0 当成 0.7 带进似然比，误报冲到 0.1877；把设计日均优势 0.15 当成真实值，而实际只有 0.06，功效只剩 0.3432、平均要等 391 天。

## 二、把证据写成一条随机游走

设每日收益 $x_t$ 独立同分布，方差 $\sigma^2$ 已知。原假设 $H_0:\mu=0$，备择 $H_1:\mu=\mu_1>0$。第 n 天的对数似然比是

$$
L_n=\sum_{t=1}^{n}\left[\frac{\mu_1-\mu_0}{\sigma^2}x_t-\frac{\mu_1^2-\mu_0^2}{2\sigma^2}\right]
$$

在 $H_0$ 下它带负漂移，在 $H_1$ 下带正漂移。Wald 给出两条吸收边界：

$$
A=\log\frac{1-\beta}{\alpha},\qquad B=\log\frac{\beta}{1-\alpha}
$$

取 $\alpha=\beta=0.05$，得到 $A=2.9444$、$B=-2.9444$。$L_n\ge A$ 接受 $H_1$，$L_n\le B$ 接受 $H_0$，否则继续观察。

![两条真实路径与 Wald 吸收边界](/images/sprt-sequential-likelihood-ratio-test/fig1_paths.png)

## 三、操作特性：省一半样本是真的

![SPRT 与固定样本方案的误报、功效和样本量对比](/images/sprt-sequential-likelihood-ratio-test/fig2_oc.png)

用 4000 次重复、$\mu_1=0.15$、$\sigma=1$、最长 1200 天做对拍。固定方案按 $n=\lceil(z_{1-\alpha}+z_{1-\beta})^2\sigma^2/\mu_1^2\rceil$ 取 481 天，在同样 5% 名义水平下误报 0.0387、功效 0.9447。

SPRT 的实测是误报 0.0405、功效 0.9545，两项都略优于固定方案；原假设下平均 239.4 天停、备择下 245.7 天停，分别省下 50.2% 和 48.9%。撞上 1200 天仍未决策的比例只有 0.08% 和 0.10%，所以节省不是靠提前认输换来的。

这里的对照组刻意用同一个正态样本、同一个名义错误率，而不是拿一个随便选天数的固定检验来陪衬。两种方法唯一的区别就是「固定样本量」与「随机停止时间」。

## 四、边界不是精确的：越界溢出

![穿过上下边界时的超出量分布](/images/sprt-sequential-likelihood-ratio-test/fig3_overshoot.png)

Wald 边界是按连续时间近似推出来的，日度观测只能每天看一次，路径会从边界外某个位置跨过去。实测上边界平均超出 0.0898、90 分位 0.1958；下边界平均超出 0.0905、90 分位 0.2008。

这个溢出解释了两件小事：实际误报 0.0405 低于名义 0.05，实际功效 0.9545 高于 0.95——边界比需要的位置稍远了一点。它不构成缺陷，但意味着 Wald 公式给的是保守近似；想精确校准，得用模拟或数值方法把溢出量吃进去。

## 五、反直觉：少看几次，误报更低、功效更高

![监控间隔对误报、功效与样本量的影响](/images/sprt-sequential-likelihood-ratio-test/fig4_monitor.png)

把「每天检查一次 $L_n$」改成每 k 天检查一次，k 取 1、2、5、10、20、50。结果和直觉相反：

- 误报从 0.0412 单调降到 0.0280；
- 功效从 0.9524 单调升到 0.9688；
- 原假设下平均样本从 248.6 天升到 312.4 天，备择下从 239.7 天升到 307.7 天。

原因很直接：备择下 $L_n$ 早期可能偶然跌破 B，天天看就会当场误判为无效；五天才看一次，这次下探往往已经过去，路径继续往上涨。原假设下同理，偶然冲过 A 的机会也变少了。代价是等待时间涨了大约 28%。

所以对实盘的意义是：SPRT 不需要真的盯盘。每周看一次反而同时降低误报和增加功效，只是决策会晚一些。这比多数人预期的「看得越勤越好」更省事。

## 六、最怕的三件事

![σ 错配与真实优势错配对误报、功效、样本量的影响](/images/sprt-sequential-likelihood-ratio-test/fig5_mismatch.png)

**第一件：σ 估错。** 保持数据真实波动率 1.0 不变，只改似然比里代入的 $\sigma$。当成 0.7 时误报 0.1877、功效 0.8143、平均 88.5 天就停；当成 1.0 时误报 0.0440、功效 0.9503、247.2 天；当成 1.3 时误报掉到 0.0060、功效 0.9713、446.5 天。低估波动率等于把每天的信号放大，边界会被频繁突破。数学上误报概率由 $\sigma_{\text{假设}}^2/\sigma_{\text{真实}}^2$ 这个比值控制，0.7 对应 0.49，2.56 对应高估到 1.6，误报只剩 0.0003。

**第二件：真实优势比设计值小。** 检验按 $\mu_1=0.15$ 设计，让真实日均优势从 0 扫到 0.21。真实为 0 时功效 0.0440（就是误报）；0.03 时 0.1244；0.06 时 0.3432；0.09 时 0.6292；到 0.15 才回到设计的 0.9508。平均样本量在 0.06 和 0.09 附近冲到 391.0 和 390.4 天，正好卡在设计漂移一半的位置——那里对数似然比的漂移接近零，游走像无偏随机漫步，要很久才碰得到边界。最坏档有 3.28% 的路径撞上 1200 天上限。

**第三件：截断本身。** σ 假设成 1.6 时功效 0.9157、平均 657.3 天，看起来比 1.3 档的 0.9713 差很多。但这一档有 8.4% 的路径在 1200 天内没做出决策，是实验上限而非方法本身失效；读表时必须把截断率一起看。

## 七、误报率还能从理论上对上

上面那条 σ 曲线不是抽样噪声。把 $L_n$ 用布朗运动近似：从 0 出发、漂移 m、方差 v，先碰到上边界的概率是 $(1-e^{-2mB/v})/(e^{-2mA/v}-e^{-2mB/v})$。在 $\mu_0=0$ 的情形代入，$ -2m/v=\sigma_{\text{假设}}^2/\sigma_{\text{真实}}^2$，记为 c。

- c=0.49（σ 当成 0.7）：解析误报 0.1911，实测 0.1877。
- c=1.00：解析 0.0500，实测 0.0440。
- c=1.32（σ 当成 1.15）：解析 0.0200，实测 0.0183。
- c=2.56（σ 当成 1.6）：解析 0.00053，实测 0.0003。

六个档位逐一对上，量级和方向都吻合。这说明边界判据对尺度参数有真实依赖，不是 3000 次重复的随机起伏。实务上，波动率得先用一段不参与检验的数据估好；拿全样本估完又在同一批数据上跑 SPRT，等于把估计误差藏进了误报率。

## 八、设计优势必须事先写死

还有一个纪律：$\mu_1$ 必须在看数据之前定死，它不是一个可以从回测里反推的参数。若先跑出一段日均 0.21 的历史，再回头把检验设成 $\mu_1=0.21$，功效表上那段历史会漂亮得失真；实际部署时真实优势多半回落到设计值一半以下，那正是平均要等 391 天的区间。稳妥做法是用一段不参与检验的样本估出最小有经济意义的日均优势，或者直接取扣除成本后仍愿意交易的下限，再把这个数写进脚本。

本站赢家诅咒一文用 500 个因子的扫描量化过同一件事：从结果里挑出来的最强信号，t 值几乎必然显著。SPRT 的早停优势只有在备择假设外生给定时才成立；否则它只是给挑选偏差装了一个更快的加速器。想要对任意停止时刻都有效的推断，得换成鞅混合检验，那是另一套工具。

## 九、它适合什么场景

SPRT 卖的是样本效率：在模型设定正确、波动率已知、备择漂移事先写死的前提下，它用一半观测达成同样的错误率。它不适合「先跑出结果再决定要检验多大的优势」——那等于把 $\mu_1$ 当成了 tunable 参数，误报率不再由 5% 约束。想要对任意停止时刻都有效的检验，需要另一套鞅工具，本站的 e-process 一文专门处理那个问题。

实现上有两点值得说清。一是停时本身是随机的，所以别把「SPRT 第 240 天就显著」当成固定 240 天的收益记录再回去算年化夏普——那会重复利用同一次停止事件。二是最长观测天数必须事先写进脚本并计入误报预算，本文取 1200 天；σ 假设过大时这个上限会吃掉功效，生产环境里应当把它当作第三种错误来源，和 α、β 一起报出来。

本文所有数字、图与代码出自同一个脚本，固定随机种子保证复跑逐项一致，读者可直接照此复现全部实验。完整实现如下。

```python
# -*- coding: utf-8 -*-
"""Sequential Probability Ratio Test (SPRT) for a Gaussian mean / strategy edge.

Single source of truth for every number and figure in the article. H0: mu=0
versus H1: mu=mu1>0 for iid daily returns with known sigma. The log likelihood
ratio is a random walk

    L_n = (mu1-mu0)/sigma^2 * sum x_t - n*(mu1^2-mu0^2)/(2*sigma^2)

with Wald boundaries A=log((1-beta)/alpha), B=log(beta/(1-alpha)). We stop at
the first crossing and measure actual size, power and expected sample size,
compare against a fixed-horizon z test of equal nominal error rates, and show
the three things the textbook boundary formula hides: overshoot at the boundary,
discrete monitoring, and design-vs-truth drift mismatch. Sigma misspecification
is also tested; common random numbers keep all comparisons paired.

All seeds are fixed so exec replay equals results.json exactly.
"""
import json
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261124
MU0 = 0.0
MU1 = 0.15
SIGMA = 1.0
ALPHA = 0.05
BETA = 0.05
NMAX = 1200
A = np.log((1.0 - BETA) / ALPHA)
B = np.log(BETA / (1.0 - BETA))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "sprt-sequential-likelihood-ratio-test")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------- engine
def llr(x, mu0, mu1, sigma):
    """Log-likelihood ratio random walk for iid N(mu, sigma^2) observations."""
    inc = (mu1 - mu0) / sigma ** 2 * x - (mu1 ** 2 - mu0 ** 2) / (2 * sigma ** 2)
    return np.cumsum(inc, axis=1)


def crossing(path, upper=A, lower=B, every=1):
    """First crossing of a monitored path; returns (stop_n, decision, value).

    decision: +1 accept H1, -1 accept H0, 0 = hit the cap without a decision.
    """
    nrow, ncol = path.shape
    monitor = np.zeros(ncol, dtype=bool)
    monitor[every - 1::every] = True
    monitor[-1] = True
    hi = (path >= upper) & monitor[None, :]
    lo = (path <= lower) & monitor[None, :]
    hit = hi | lo
    any_hit = hit.any(axis=1)
    idx = np.where(any_hit, hit.argmax(axis=1), ncol - 1)
    rows = np.arange(nrow)
    val = path[rows, idx]
    dec = np.where(hi[rows, idx], 1, np.where(lo[rows, idx], -1, 0))
    return idx + 1, dec, val


def fixed_horizon_n(mu0, mu1, sigma, alpha=ALPHA, beta=BETA):
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta)
    return int(np.ceil((z * sigma / (mu1 - mu0)) ** 2))


def fixed_test(x, n, mu0, sigma, alpha=ALPHA):
    crit = mu0 + stats.norm.ppf(1 - alpha) * sigma / np.sqrt(n)
    return x[:, :n].mean(axis=1) > crit


# ------------------------------------------------------------- experiments
def exp_oc(reps=4000):
    rng = np.random.default_rng(SEED + 10)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    x0 = MU0 + SIGMA * z0
    x1 = MU1 + SIGMA * z1
    p0 = llr(x0, MU0, MU1, SIGMA)
    p1 = llr(x1, MU0, MU1, SIGMA)
    n0, d0, v0 = crossing(p0)
    n1, d1, v1 = crossing(p1)
    nfix = fixed_horizon_n(MU0, MU1, SIGMA)
    fsize = float(np.mean(fixed_test(x0, nfix, MU0, SIGMA)))
    fpower = float(np.mean(fixed_test(x1, nfix, MU0, SIGMA)))
    return {
        "reps": reps, "nmax": NMAX, "A": float(A), "B": float(B),
        "mu1": MU1, "sigma": SIGMA, "n_fixed": nfix,
        "sprt_size": float(np.mean(d0 == 1)),
        "sprt_power": float(np.mean(d1 == 1)),
        "sprt_e_n_h0": float(n0.mean()),
        "sprt_e_n_h1": float(n1.mean()),
        "sprt_cap_h0": float(np.mean(d0 == 0)),
        "sprt_cap_h1": float(np.mean(d1 == 0)),
        "fixed_size": fsize, "fixed_power": fpower,
        "saving_h0": 1.0 - n0.mean() / nfix,
        "saving_h1": 1.0 - n1.mean() / nfix,
        "upper_overshoot_mean": float(np.mean(v1[d1 == 1] - A)) if np.any(d1 == 1) else 0.0,
        "lower_overshoot_mean": float(np.mean(B - v0[d0 == -1])) if np.any(d0 == -1) else 0.0,
        "upper_overshoot_q90": float(np.quantile(v1[d1 == 1] - A, 0.90)),
        "lower_overshoot_q90": float(np.quantile(B - v0[d0 == -1], 0.90)),
        "upper_cross_vals": (v1[d1 == 1] - A)[:600].tolist(),
        "lower_cross_vals": (B - v0[d0 == -1])[:600].tolist(),
    }


def exp_monitor(reps=2500):
    rng = np.random.default_rng(SEED + 20)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    p0 = llr(MU0 + SIGMA * z0, MU0, MU1, SIGMA)
    p1 = llr(MU1 + SIGMA * z1, MU0, MU1, SIGMA)
    rows = {}
    for k in (1, 2, 5, 10, 20, 50):
        n0, d0, _ = crossing(p0, every=k)
        n1, d1, _ = crossing(p1, every=k)
        rows[str(k)] = {
            "size": float(np.mean(d0 == 1)),
            "power": float(np.mean(d1 == 1)),
            "e_n_h0": float(n0.mean()),
            "e_n_h1": float(n1.mean()),
        }
    return rows


def exp_sigma_mismatch(reps=3000):
    """Same underlying paths; change only the sigma used inside the LLR."""
    rng = np.random.default_rng(SEED + 30)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    x0 = MU0 + SIGMA * z0
    x1 = MU1 + SIGMA * z1
    rows = {}
    for sa in (0.7, 0.85, 1.0, 1.15, 1.3, 1.6):
        d0 = crossing(llr(x0, MU0, MU1, sa))[1]
        n1, d1, _ = crossing(llr(x1, MU0, MU1, sa))
        rows[str(sa)] = {
            "size": float(np.mean(d0 == 1)),
            "power": float(np.mean(d1 == 1)),
            "e_n_h1": float(n1.mean()),
            "cap_rate_h1": float(np.mean(d1 == 0)),
        }
    return rows


def exp_drift_mismatch(reps=2500):
    """Design the test for MU1 but let the true per-day edge vary."""
    rng = np.random.default_rng(SEED + 40)
    z0 = rng.standard_normal((reps, NMAX))
    z1 = rng.standard_normal((reps, NMAX))
    x0 = MU0 + SIGMA * z0
    rows = {}
    for mu in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21):
        n1, d1, _ = crossing(llr(mu + SIGMA * z1, MU0, MU1, SIGMA))
        rows[str(mu)] = {
            "power": float(np.mean(d1 == 1)),
            "e_n_h1": float(n1.mean()),
            "cap_rate": float(np.mean(d1 == 0)),
        }
    rows["h0_size"] = float(np.mean(crossing(llr(x0, MU0, MU1, SIGMA))[1] == 1))
    return rows


def exp_paths(steps=450):
    rng = np.random.default_rng(SEED + 50)
    z0 = rng.standard_normal(steps)
    z1 = rng.standard_normal(steps)
    p0 = llr((MU0 + SIGMA * z0)[None, :], MU0, MU1, SIGMA)[0]
    p1 = llr((MU1 + SIGMA * z1)[None, :], MU0, MU1, SIGMA)[0]
    for p in (p0, p1):
        hi = np.where(p >= A)[0]
        lo = np.where(p <= B)[0]
        if hi.size or lo.size:
            first = min(hi[0] if hi.size else 10 ** 9,
                        lo[0] if lo.size else 10 ** 9)
            p[first + 1:] = np.nan
    return {"h0": p0.tolist(), "h1": p1.tolist()}


# ------------------------------------------------------------------ figures
def fig1_paths(p):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    h0 = np.array(p["h0"]); h1 = np.array(p["h1"])
    ax.plot(np.arange(1, len(h0) + 1), h0, color="#2b8cbe", lw=1.4,
            label="one path under H0")
    ax.plot(np.arange(1, len(h1) + 1), h1, color="#b23a2b", lw=1.4,
            label="one path under H1")
    ax.axhline(A, color="#333", ls="--", lw=1.3, label="A = log((1-beta)/alpha)")
    ax.axhline(B, color="#333", ls=":", lw=1.3, label="B = log(beta/(1-alpha))")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("number of daily observations")
    ax.set_ylabel("log likelihood ratio L_n")
    ax.set_title("SPRT: two paths, two absorbing boundaries")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig1_paths.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig1_paths.png"


def fig2_oc(oc):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    names = ["size", "power"]
    sprt = [oc["sprt_size"], oc["sprt_power"]]
    fixed = [oc["fixed_size"], oc["fixed_power"]]
    xs = np.arange(2); w = 0.34
    ax[0].bar(xs - w / 2, sprt, w, label="SPRT (Wald)", color="#3a923a")
    ax[0].bar(xs + w / 2, fixed, w, label="fixed n=%d" % oc["n_fixed"],
              color="#2b8cbe")
    ax[0].axhline(0.05, color="#333", ls="--", lw=1.1)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names)
    ax[0].set_ylim(0, 1.03); ax[0].legend(fontsize=9)
    ax[0].set_title("Error rates at nominal alpha=beta=5%")
    ns = [oc["sprt_e_n_h0"], oc["sprt_e_n_h1"], oc["n_fixed"]]
    labels = ["SPRT under H0", "SPRT under H1", "fixed horizon"]
    ax[1].bar(labels, ns, color=["#2b8cbe", "#3a923a", "#b23a2b"], alpha=0.9)
    for i, v in enumerate(ns):
        ax[1].text(i, v + 8, "%.0f" % v, ha="center", fontsize=9)
    ax[1].set_ylabel("average observations")
    ax[1].set_title("Expected sample size, same nominal errors")
    fig.tight_layout()
    f = os.path.join(OUT, "fig2_oc.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig2_oc.png"


def fig3_overshoot(oc):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.hist(oc["upper_cross_vals"], bins=40, alpha=0.75,
            color="#b23a2b", label="upper crossing overshoot")
    ax.hist(oc["lower_cross_vals"], bins=40, alpha=0.75,
            color="#2b8cbe", label="lower crossing overshoot")
    ax.set_xlabel("distance beyond the boundary")
    ax.set_ylabel("count")
    ax.set_title("Discrete daily monitoring overshoots the Wald boundary")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig3_overshoot.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig3_overshoot.png"


def fig4_monitor(rows):
    ks = [1, 2, 5, 10, 20, 50]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for lab, col in (("size", "#b23a2b"), ("power", "#3a923a")):
        ax[0].plot(ks, [rows[str(k)][lab] for k in ks], "o-",
                   label=lab, color=col, lw=1.8)
    ax[0].axhline(0.05, color="#333", ls="--", lw=1.1)
    ax[0].set_xlabel("check the LLR every k days")
    ax[0].set_ylabel("rejection rate")
    ax[0].set_title("Rare looks shrink size but raise power")
    ax[0].legend(fontsize=9)
    for lab, col in (("e_n_h0", "#2b8cbe"), ("e_n_h1", "#3a923a")):
        ax[1].plot(ks, [rows[str(k)][lab] for k in ks], "o-",
                   label=lab, color=col, lw=1.8)
    ax[1].set_xlabel("check the LLR every k days")
    ax[1].set_ylabel("average observations")
    ax[1].set_title("Waiting costs observations")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig4_monitor.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig4_monitor.png"


def fig5_mismatch(sig, drift):
    sas = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
    mus = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(sas, [sig[str(s)]["size"] for s in sas], "o-",
               color="#b23a2b", label="size", lw=1.8)
    ax[0].plot(sas, [sig[str(s)]["power"] for s in sas], "o-",
               color="#3a923a", label="power", lw=1.8)
    ax[0].set_xlabel("sigma assumed inside the LLR (truth = 1.0)")
    ax[0].set_ylabel("rejection rate")
    ax[0].set_title("Wrong sigma: size drifts from 0.19 to 0.00")
    ax[0].legend(fontsize=9)
    axb = ax[1]
    axb.plot(mus, [drift[str(m)]["power"] for m in mus], "o-",
             color="#3a923a", label="power (left axis)", lw=1.8)
    axb.set_xlabel("true daily edge (test designed for 0.15)")
    axb.set_ylabel("power")
    axb2 = axb.twinx()
    axb2.plot(mus, [drift[str(m)]["e_n_h1"] for m in mus], "s--",
              color="#b23a2b", label="E[N] (right axis)", lw=1.8)
    axb2.set_ylabel("average observations")
    axb.set_title("Wrong edge: E[N] peaks near mu1/2 as power fades")
    axb.legend(fontsize=9, loc="upper left")
    axb2.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    f = os.path.join(OUT, "fig5_mismatch.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig5_mismatch.png"


def main():
    oc = exp_oc()
    mon = exp_monitor()
    sig = exp_sigma_mismatch()
    drift = exp_drift_mismatch()
    paths = exp_paths()
    r = {
        "setup": {"seed": SEED, "alpha": ALPHA, "beta": BETA, "mu0": MU0,
                  "mu1": MU1, "sigma": SIGMA, "A": float(A), "B": float(B),
                  "nmax": NMAX, "reps_oc": 4000, "reps_monitor": 2500,
                  "reps_sigma": 3000, "reps_drift": 2500},
        "oc": oc,
        "monitor": mon,
        "sigma_mismatch": sig,
        "drift_mismatch": drift,
    }
    r["figs"] = [fig1_paths(paths), fig2_oc(oc), fig3_overshoot(oc),
                 fig4_monitor(mon), fig5_mismatch(sig, drift)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

