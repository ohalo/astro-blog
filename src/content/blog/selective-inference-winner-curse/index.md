---
title: "赢家诅咒与后选择推断：扫描 500 个因子选出最强的那个，为什么它的 t 值几乎必然显著"
description: "一个真值已知的蒙特卡洛实验（每档 6000 次重复）拆开量化研究最常见的陷阱：扫描 p 个信号、报告最显著的那个。纯噪声下最大 |t| 的期望从 p=5 的 1.574 爬到 p=500 的 3.239，与解析积分逐档吻合；把它当成预先设定的单次检验，5% 虚警率从 p=5 的 23.3% 一路冲到 p=200 的 100%，而 Bonferroni、选择校正与样本分割三法都稳在 4% 到 5%。被选中信号的 95% 置信区间覆盖率从 p=10 的 78.8% 崩到 p=200 的 2.9%，样本分割却守在 94.7% 到 96.0%。最扎心的是功效表：真效应为零时，朴素方法宣称显著的比例高达 92.8%，而真正选对信号只有 2.3%。结论：从一堆信号里挑冠军，冠军的成绩单必须按挑选过程重新校准。"
publishDate: '2026-11-21T11:10:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 因子研究
  - Python
language: zh
difficulty: advanced
image: "/images/selective-inference-winner-curse/fig1_selection_bias.png"
---

## 一、结论先行

量化研究员每天在做同一件事：扫一遍候选信号，挑出表现最好的那个，然后认真报告它的 t 值和置信区间。问题在于，「挑出最好的」这个动作本身就已经改变了统计量的分布。你拿它当成一个事先约定好的单次检验来读，得到的显著性是虚的。

本文用一个真值已知的实验室把这件事量化。设定极简：p 个候选信号，每个的估计值是真效应加一个标准正态噪声（方差已知为 1）。研究员扫描全部 p 个，选出估计值最大的那个，然后报告它。所有数字来自同一份脚本，代码附在文末，可逐项复现。

先把结论摆出来。第一，纯噪声下最大估计值的期望随 p 单调爬升：p=5 时是 1.574，p=50 时 2.503，p=500 时 3.239，与解析积分逐档吻合到 0.012 以内。第二，把这个冠军当成单次检验，名义 5% 的虚警率在 p=5 就有 23.3%，p=100 达 99.4%，p=200 后是 100%——扫得越多，越「必然」发现显著因子。第三，Bonferroni、基于最大值分布的选择校正、样本分割三种修正都能把虚警率按在 4% 到 5%。第四，被选中信号的 95% 置信区间覆盖率随 p 崩塌，从 p=10 的 78.8% 掉到 p=200 的 2.9%，而样本分割稳定在 94.7% 到 96.0%。第五，也是最值得贴屏幕上的一条：真效应全为零时，朴素方法宣称「发现显著因子」的比例是 92.8%，而真正选对那个唯一真信号只有 2.3%。

一句话：从一群信号里挑冠军，冠军的成绩单必须按挑选过程重新算，否则你只是在给噪声颁奖。

## 二、实验室：扫描、挑选、然后报告

设定写成公式。第 j 个信号的估计是 $\hat\theta_j = \theta_j + z_j$，其中 z 是独立标准正态。研究员用 $\hat w = \arg\max_j \hat\theta_j$ 挑出索引，报告 $t = \hat\theta_{\hat w}$ 并检验它是否为零。

这个设定看似 toy，实则抓住了选股因子挖掘的本质：你有几百个想法，数据只给你一次机会，你自然会挑那个数字最好看的。真值已知的好处是所有理论值都能算。p 个独立标准正态最大值的分布函数是 $\Phi(t)^p$，这是后面所有解析基准的源头。

图1 显示纯噪声下冠军的成绩单有多离谱。

![图1：扫描信号数 p 与最大估计值的关系，蒙特卡洛与解析积分逐档吻合](/images/selective-inference-winner-curse/fig1_selection_bias.png)

p=5 时冠军平均 1.574，还在常见的 1.96 阈值之下；p=20 时升到 2.175，已经越过；p=500 时高达 3.239，相当于一个三标准差以上的「重大发现」。蒙特卡洛与解析曲线几乎重叠（p=500 时 3.239 对 3.242），说明实验没有搭错。注意这里连一个真信号都没有，全部是噪声。扫描本身就在凭空制造效应量。

## 三、虚警率：从 23% 到 100%

看推断。若把冠军的 t 当成事先约定的单次检验，用单尾 5% 阈值 1.645 判断，虚警率是多少？解析答案是 $1-\Phi(1.645)^p$。p=5 时 0.226，p=20 时 0.642，p=100 时 0.994——模拟值 0.233、0.647、0.994 与之吻合。

![图2：四种检验方法在全零下的虚警率，朴素法随 p 冲到 1，三种修正守住 5%](/images/selective-inference-winner-curse/fig2_size.png)

图2 把四种方法放一起。朴素 z 检验（红）从 0.233 一路冲到 1.000。Bonferroni（蓝）用 $\alpha/p$ 的阈值，选择校正（绿）直接用 $1-\Phi(t)^p$ 作为 p 值——后者正是把挑选过程纳入分布后的精确修正——样本分割（紫）用一半数据挑选、另一半独立检验。三者都稳在 4% 到 5%：p=500 时分别是 4.2%、4.3%、4.9%。

这里有个值得咀嚼的点：Bonferroni 和选择校正在本设定下表现几乎一样。原因是「挑最大值」恰好是 Bonferroni 最擅长的场景——它守的就是族系错误率上界。一旦选择规则变复杂（比如 Lasso 逐步筛选），Bonferroni 会明显保守，而选择校正仍能精确。但那是另一篇文章的故事，本文先把最干净的最大值情形钉死。

## 四、置信区间：名义 95%，实际 2.9%

虚警率说的是「有没有假发现」，覆盖率说的是「估计准不准」。做因子研究时后者更要命：你不仅想知道因子有没有用，还想知道它的效应有多大。

设定稀疏真值：p 个信号里只有第 0 个真效应为 0.8，其余全为零。在第 A 样本上挑冠军，然后比较两种 95% 区间对「被选中索引真效应」的覆盖：朴素区间用同一份数据算 $\hat\theta_A[w]\pm 1.96$，样本分割区间用独立的第 B 样本算 $\hat\theta_B[w]\pm 1.96$。

![图3：被选中信号的置信区间覆盖率，朴素法崩塌，样本分割稳定在 95% 附近](/images/selective-inference-winner-curse/fig3_coverage.png)

图3 的结果触目惊心。朴素覆盖率从 p=10 的 0.788 掉到 p=200 的 0.029——区间宽度明明是标准的 3.92，却只覆盖到真值不到百分之三的时间。原因就是选择偏差把中心 $\hat\theta_A[w]$ 系统性抬高了，真值被挤到区间左边。样本分割覆盖率稳定在 0.947 到 0.960，因为它用没参与挑选的数据估计，中心不偏。两个区间一样宽，但一个可信一个不可信——宽度相同不代表质量相同。

## 五、冠军的估计分布：为什么均值是 2.31

p=50、唯一真效应 1.0 时，看被报告估计值的分布。

![图4：朴素报告值与样本分割估计的分布对比，朴素明显右偏](/images/selective-inference-winner-curse/fig4_distribution.png)

图4 里朴素报告值（红）的均值是 2.313，而真效应只有 1.0——高估了一倍多。样本分割估计（紫）均值 0.124，更接近真值：因为 50 个信号里只有 1 个是真的，被选中的那个绝大多数时候其实是噪声（选对真信号的比例只有 13.3%），于是分割估计的均值被大量零效应拉低。这不是分割法「错」，而是诚实承认「你挑中的多半不是真货」。朴素法把噪声的极值当成效应，才有了那个虚高的 2.31。

## 六、功效的幻觉：92.8% 的「发现」全是假的

最后看一个最能说明问题的表。固定 p=50，扫描唯一真效应的强度。

![图5：朴素「显著」比例、选择校正可靠发现比例、真正选对信号比例随真效应变化](/images/selective-inference-winner-curse/fig5_power.png)

真效应为零时，朴素方法宣称显著的比例是 0.928，选择校正是 0.050（正好是设定的虚警率），而真正选对那个信号只有 0.023。换句话说，朴素方法报出来的「重大发现」里，几乎全部是噪声冠军。真效应升到 3.0 时，选对率爬到 0.760，选择校正的检出率 0.489 仍然低于朴素的 0.990——因为选择校正宁可漏报也不虚报。这正是严谨的代价：它把「扫了很多次」这件事诚实计入，所以门槛更高。

对量化实操的启示很直接：如果你的策略是从几百个因子里挑出来的，回测里那个漂亮的夏普比率大概率含大量选择偏差。别用样本内数字给它定价，留一段没参与筛选的数据单独验，或者按挑选过程做选择校正。

```python
# -*- coding: utf-8 -*-
"""Winner's curse and post-selection inference: scanning p signals, reporting
the top one, and the collapse of naive inference.

Single source of truth for every number and figure in the article.

Setup (known variance sigma=1, p candidate signals):
    theta_hat_j = theta_j + z_j,   z_j ~ N(0,1) iid
Researcher scans all p, picks w = argmax theta_hat_j, and reports that signal.

Analytic baselines (iid standard normal):
    P(max_j Z_j <= t) = Phi(t)^p
    E[max |Z_j|]      = integral_0^inf t d[(2 Phi(t)-1)^p]
Four inference rules for the selected winner (one-sided, alpha=0.05):
    naive    : reject if t > z_{0.95}                       (size -> 1 as p grows)
    bonferroni: reject if p*(1-Phi(t)) < 0.05               (size = 0.05, conservative)
    selective: reject if 1-Phi(t)^p < 0.05                  (size = 0.05, exact)
    split    : select on one half, test on the other half   (size = 0.05, costs data)
Known-variance selective CI inverts the truncated-normal p-value conditional
on the selection event {theta_hat_w > m}, m = runner-up value.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, special

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261121
REPS_SIZE = 6000
REPS_COVER = 1500
REPS_DIST = 8000
REPS_POWER = 4000
P_GRID = [5, 10, 20, 50, 100, 200, 500]
Z95 = float(special.ndtri(0.95))
ALPHA = 0.05

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "selective-inference-winner-curse")
os.makedirs(_OUT, exist_ok=True)


def _cdf_max(t, p):
    return special.ndtr(t) ** p


def analytic_expected_absmax(p):
    """E[max |Z|] for p iid N(0,1): integrate t * d[(2Phi(t)-1)^p].
    Density of max|Z| = p*(2Phi(t)-1)^{p-1} * 2 phi(t)."""
    pdf = lambda t: p * (2 * special.ndtr(t) - 1) ** (p - 1) * 2 * np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
    val, _ = integrate.quad(lambda t: t * pdf(t), 0, 12, limit=300)
    return float(val)


def exp_curse():
    """Selection bias: expected top-|theta_hat| under pure noise, MC vs analytic."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for p in P_GRID:
        draws = np.abs(rng.standard_normal((REPS_SIZE, p))).max(axis=1)
        mc = float(draws.mean())
        an = analytic_expected_absmax(p)
        rows.append({"p": p, "mc": mc, "analytic": an,
                     "naive_z": Z95, "excess_vs_1_96": mc - 1.96})
    return {"reps": REPS_SIZE, "rows": rows}


def exp_size():
    """False-discovery size of four rules under the global null theta=0."""
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for p in P_GRID:
        m = rng.standard_normal((REPS_SIZE, p)).max(axis=1)
        naive = float((m > Z95).mean())
        bonf = float(((1 - special.ndtr(m)) < ALPHA / p).mean())
        sel = float(((1 - special.ndtr(m) ** p) < ALPHA).mean())
        # split: select winner on first half signals, test it on an independent replicate
        half = p
        m1 = rng.standard_normal((REPS_SIZE, half)).max(axis=1)
        test = rng.standard_normal(REPS_SIZE)  # independent measurement of selected index's theta
        split = float((test > Z95).mean())
        rows.append({"p": p, "naive": naive, "bonferroni": bonf,
                     "selective": sel, "split": split,
                     "naive_analytic": 1 - special.ndtr(Z95) ** p})
    return {"reps": REPS_SIZE, "alpha": ALPHA, "rows": rows}


def exp_coverage(theta_star=0.8):
    """Sparse truth: index 0 has theta*, rest 0. Select top-1 on sample A, then
    compare two 95% intervals for the selected index's true effect:
      naive : theta_hat_A[w] +/- 1.96   (same data used to select -> biased)
      split : theta_hat_B[w] +/- 1.96   (fresh independent sample -> honest)
    Coverage of the TRUE effect of the selected index (0 if a noise index won)."""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for p in [10, 20, 50, 100, 200]:
        naive_cov = split_cov = 0
        naive_width_sum = split_width_sum = 0.0
        for _ in range(REPS_COVER):
            th = np.zeros(p)
            th[0] = theta_star
            xa = th + rng.standard_normal(p)
            xb = th + rng.standard_normal(p)
            w = int(np.argmax(xa))
            true_w = theta_star if w == 0 else 0.0
            ta, tb = xa[w], xb[w]
            if ta - 1.96 <= true_w <= ta + 1.96:
                naive_cov += 1
            if tb - 1.96 <= true_w <= tb + 1.96:
                split_cov += 1
            naive_width_sum += 2 * 1.96
            split_width_sum += 2 * 1.96
        rows.append({"p": p,
                     "naive_cov": naive_cov / REPS_COVER,
                     "split_cov": split_cov / REPS_COVER,
                     "naive_width": naive_width_sum / REPS_COVER,
                     "split_width": split_width_sum / REPS_COVER})
    return {"reps": REPS_COVER, "theta_star": theta_star, "rows": rows}


def exp_distribution(p=50, theta_star=1.0):
    """Distribution of the reported estimate: naive full-sample vs split-sample."""
    rng = np.random.default_rng(SEED + 4)
    naive_vals, split_vals, chosen_true = [], [], []
    half = max(2, p // 2)
    for _ in range(REPS_DIST):
        th = np.zeros(p)
        th[0] = theta_star
        x = th + rng.standard_normal(p)
        w = int(np.argmax(x))
        naive_vals.append(float(x[w]))
        chosen_true.append(theta_star if w == 0 else 0.0)
        # independent split on a fresh sample of same design
        xs = th + rng.standard_normal(p)
        split_vals.append(float(xs[w]))
    return {"reps": REPS_DIST, "p": p, "theta_star": theta_star,
            "naive": naive_vals, "split": split_vals, "chosen_true": chosen_true,
            "naive_mean": float(np.mean(naive_vals)),
            "split_mean": float(np.mean(split_vals)),
            "sel_true_signal_frac": float(np.mean(np.array(chosen_true) == theta_star))}


def exp_power(p=50):
    """Reliable discovery rate vs effect size: naive claim vs selective claim."""
    rng = np.random.default_rng(SEED + 5)
    theta_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for ts in theta_grid:
        naive_sig = sel_sig = true_pos = 0
        for _ in range(REPS_POWER):
            th = np.zeros(p)
            th[0] = ts
            x = th + rng.standard_normal(p)
            w = int(np.argmax(x))
            t = x[w]
            if t > Z95:
                naive_sig += 1
            if (1 - special.ndtr(t) ** p) < ALPHA:
                sel_sig += 1
            if w == 0:
                true_pos += 1
        rows.append({"theta": ts,
                     "naive_claim": naive_sig / REPS_POWER,
                     "selective_claim": sel_sig / REPS_POWER,
                     "true_selected": true_pos / REPS_POWER})
    return {"reps": REPS_POWER, "p": p, "rows": rows}


def plot1(curse):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ps = [r["p"] for r in curse["rows"]]
    mc = [r["mc"] for r in curse["rows"]]
    an = [r["analytic"] for r in curse["rows"]]
    ax.plot(ps, mc, "o-", color="#c0392b", lw=2.2, label="MC: E[max |theta_hat|]")
    ax.plot(ps, an, "s--", color="#2c3e50", lw=1.8, label="analytic integral")
    ax.axhline(1.96, color="#27ae60", ls=":", lw=1.8, label="naive threshold 1.96")
    ax.axhline(0.0, color="gray", lw=1, label="true effect = 0")
    ax.set_xscale("log")
    ax.set_xlabel("number of scanned signals p")
    ax.set_ylabel("reported top-|theta_hat|")
    ax.set_title("Fig1: scanning more noise manufactures larger 'effects'")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig1_selection_bias.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig1_selection_bias.png"


def plot2(size):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ps = [r["p"] for r in size["rows"]]
    ax.plot(ps, [r["naive"] for r in size["rows"]], "o-", color="#c0392b", lw=2.2, label="naive z-test")
    ax.plot(ps, [r["bonferroni"] for r in size["rows"]], "s-", color="#2980b9", lw=1.8, label="Bonferroni")
    ax.plot(ps, [r["selective"] for r in size["rows"]], "^--", color="#27ae60", lw=1.8, label="selective (exact)")
    ax.plot(ps, [r["split"] for r in size["rows"]], "d:", color="#8e44ad", lw=1.8, label="sample splitting")
    ax.axhline(0.05, color="k", ls=":", lw=1.5, label="nominal 5%")
    ax.set_xscale("log")
    ax.set_xlabel("number of scanned signals p")
    ax.set_ylabel("false rejection under global null")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Fig2: naive size -> 1; the three corrections hold 5%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig2_size.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig2_size.png"


def plot3(cover):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ps = [r["p"] for r in cover["rows"]]
    ax.plot(ps, [r["naive_cov"] for r in cover["rows"]], "o-", color="#c0392b", lw=2.2, label="naive 95% CI (same data)")
    ax.plot(ps, [r["split_cov"] for r in cover["rows"]], "s-", color="#27ae60", lw=2.2, label="split-sample 95% CI")
    ax.axhline(0.95, color="k", ls=":", lw=1.5, label="nominal 95%")
    ax.set_xscale("log")
    ax.set_xlabel("number of scanned signals p")
    ax.set_ylabel("coverage of selected signal")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Fig3: naive coverage collapses; split-sample CI stays near 95%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig3_coverage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig3_coverage.png"


def plot4(dist):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bins = np.linspace(-1.5, 4.5, 60)
    ax.hist(dist["split"], bins=bins, density=True, alpha=0.6, color="#8e44ad", label="split-sample estimate")
    ax.hist(dist["naive"], bins=bins, density=True, alpha=0.6, color="#c0392b", label="naive reported top")
    ax.axvline(dist["theta_star"], color="k", ls="--", lw=1.6,
               label="theta* = %.1f" % dist["theta_star"])
    ax.axvline(dist["naive_mean"], color="#c0392b", ls=":", lw=1.6,
               label="naive mean %.2f" % dist["naive_mean"])
    ax.set_xlabel("reported effect size")
    ax.set_ylabel("density")
    ax.set_title("Fig4: naive estimate is right-skewed by the selection gate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig4_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig4_distribution.png"


def plot5(power):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = [r["theta"] for r in power["rows"]]
    ax.plot(xs, [r["naive_claim"] for r in power["rows"]], "o-", color="#c0392b", lw=2.2, label="naive 'significant' claim")
    ax.plot(xs, [r["selective_claim"] for r in power["rows"]], "s-", color="#27ae60", lw=2.2, label="selective reliable claim")
    ax.plot(xs, [r["true_selected"] for r in power["rows"]], "^--", color="#2980b9", lw=1.8, label="true signal actually picked")
    ax.axhline(0.05, color="k", ls=":", lw=1.3, label="nominal 5%")
    ax.set_xlabel("true effect size theta* of the one real signal")
    ax.set_ylabel("rate over replications")
    ax.set_title("Fig5: naive 'power' at theta=0 is pure false discovery")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(_OUT, "fig5_power.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "fig5_power.png"


def main():
    curse = exp_curse()
    size = exp_size()
    cover = exp_coverage()
    dist = exp_distribution()
    power = exp_power()
    results = {
        "setup": {"seed": SEED, "z95": Z95, "alpha": ALPHA,
                  "reps_size": REPS_SIZE, "reps_cover": REPS_COVER,
                  "reps_dist": REPS_DIST, "reps_power": REPS_POWER,
                  "p_grid": P_GRID},
        "curse": curse,
        "size": size,
        "coverage": cover,
        "distribution": {k: v for k, v in dist.items() if k not in ("naive", "split", "chosen_true")},
        "power": power,
    }
    results["figs"] = [plot1(curse), plot2(size), plot3(cover), plot4(dist), plot5(power)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 七、完整代码与复现

上面全部数字与五张图来自下面这段自包含脚本。它用固定的种子 20261121，依次计算选择偏差、四种检验的虚警率、两种区间的覆盖率、估计分布与功效表，最后把结果汇总到 results.json。纯噪声下最大值的解析期望用数值积分单独算出来，与蒙特卡洛逐档对拍。正文中每个数字都能在脚本输出中逐项核对。

## 八、小结

「从一堆里挑最好的」是人类做研究的本能，也是统计推断最容易翻车的地方。冠军的成绩单天生带偏：期望更高、区间更窄、显著性更强，但这些漂亮数字里很大一部分是挑选过程本身的产物。本文给出的三条修正——Bonferroni 守上界、选择校正算精确分布、样本分割用干净数据——共同的道理只有一个：读一个数字之前，先问它是怎么被选出来的。
