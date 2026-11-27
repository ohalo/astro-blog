---
title: "方差齐性检验在厚尾收益上为什么会误报成灾：Bartlett 虚警率冲到 66%，Brown-Forsythe 却守在 4%"
description: "真值已知的 1600 次蒙特卡洛拆开四种方差齐性检验。正态下四家虚警率都在 4% 至 5%，t(3) 厚尾下 Bartlett 冲到 55.75%、对数正态下 65.81%，Brown-Forsythe 与 Fligner 守在 3.6% 上下。样本量从 30 增到 480，Bartlett 虚警反而从 47.31% 爬到 73.94%。构造方差相同、尾部更厚的两个策略后，Fligner 竟有 25.00% 的概率喊出「B 更波动」。附完整代码与五张真实计算图。"
publishDate: '2026-11-27T03:10:00Z'
tags:
  - 量化
  - 统计推断
  - 稳健统计
  - 假设检验
  - Python
language: zh
difficulty: advanced
image: "/images/variance-equality-test-robust-heavy-tails/fig01_size.png"
---

## 一、结论先行：你在比的到底是方差，还是尾巴

量化研究里到处都在问「两段的波动是不是一样」：策略上线前后、牛熊两段、A 股和美股、两个再平衡频率。最顺手的工具是 Bartlett 检验， scipy 里一行 `stats.bartlett` 就能出 p 值。问题是 Bartlett 有一个经常被忽略的前提——数据得来自正态总体。收益不是正态的，它有肥尾。

本文用真值已知的合成数据把这件事量化。四组样本量各 60，1600 次重复，显著性水平 5%。结果是：正态世界裡四家检验都守规矩，虚警率 4% 到 5%；一旦换成 t(3) 厚尾，Bartlett 的虚警率飙到 **55.75%**，对数正态更是 **65.81%**；而 Brown-Forsythe 与 Fligner 分别稳在 3.63% 和 3.63%。换句话说，在厚尾收益上跑 Bartlett，你有一半以上的「显著波动差异」是凭空报出来的。

更反直觉的是样本量的作用。直觉上数据越多检验越准，但在 t(3) 下把每组样本从 30 加 to 480，Bartlett 的虚警率从 47.31% **升到** 73.94%。它把厚尾当作越来越确凿的证据。这不是 Bartlett 的 bug，而是它的逻辑 inherently 依赖四阶矩：尾部越厚、样本越多，它对「正态」的违背看得越清楚，于是越坚决地拒绝原假设——只是拒绝的理由和你想问的问题不是一回事。

## 二、四种检验在检验什么

先把工具摆清楚。给定 $k$ 组样本，原假设是各组方差相等。

Bartlett 检验构造 $T=\frac{(N-k)\log s_p^2-\sum_i (n_i-1)\log s_i^2}{1+\frac{1}{3(k-1)}\left(\sum_i\frac{1}{n_i-1}-\frac{1}{N-k}\right)}$，在原假设下近似服从 $\chi^2(k-1)$。它对正态性极其敏感，因为推导里用到了样本方差的对数正态近似。

Levene 检验换了个思路：不直接比方差，而是对每个样本取它到本组均值的绝对偏差 $z_{ij}=|x_{ij}-\bar x_i|$，再对 $z$ 做单因素方差分析。Brown-Forsythe 是 Levene 的改进版，把中心从均值换成中位数 $z_{ij}=|x_{ij}-\tilde x_i|$，对偏斜更稳。Fligner-Killeen 再进一步，先把绝对偏差转成秩再用正态得分，完全不依赖矩。

一句话概括：Bartlett 比的是二阶矩，另外三家比的是「离中心的典型距离」。当收益有厚尾时，二阶矩会被少数极端值主导，而中位数距离和秩相对迟钝——这就是稳健性来源。

![四种检验的虚警率](/images/variance-equality-test-robust-heavy-tails/fig01_size.png)

图一给出四种分布下的虚警率。正态那一组四家都贴在 5% 虚线附近；t(5) 轻度厚尾下 Bartlett 已经爬到 28.25%；t(3) 到 55.75%；对数正态这种偏斜加厚尾直接冲到 65.81%。同期 Levene 均值中心在对数正态下也有 19.25% 的虚警——它用的是均值，会被偏斜尾巴拽走。Brown-Forsythe 和 Fligner 全程守在 4% 到 9% 之间，其中 Brown-Forsythe 在偏斜下最稳（4.00%）。

## 三、样本量不平衡帮倒忙吗

一个常见猜测是：Bartlett 的问题来自样本量不平衡，配平就好了。第二个实验用三组样本量 20、60、300 的高度不平衡设计，对照 60、60、60 的平衡设计，分布取 t(3)。

![样本量不平衡的影响](/images/variance-equality-test-robust-heavy-tails/fig02_unbalanced.png)

结果是否定的：平衡时 Bartlett 虚警 55.75%，不平衡时 54.75%，几乎没变。Brown-Forsythe 从 3.63% 微升到 5.56%，Fligner 从 3.63% 到 4.81%，仍在合理区间。也就是说症结不在设计是否平衡，而在分布尾部本身。调平样本量是很多人下意识做的「修正」，但它对这个病无效。

## 四、功效的真相：高拒绝率不等于高检出率

有人会替 Bartlett 辩护：它拒绝得多，是因为它敏感，能更早发现真实的方差差异。第三个实验直接检验这个说法。把后两组 scale 设为第一组的 $r$ 倍，$r$ 从 1.0 扫到 2.0，记录拒绝率。

![真实方差差异下的功效](/images/variance-equality-test-robust-heavy-tails/fig03_power.png)

正态世界里这个辩护成立：$r=1.4$ 时 Bartlett 功效 86.44%，Brown-Forsythe 77.25%，Fligner 76.63%，Bartlett 确实最强。但厚尾世界里要看 $r=1.0$ 那一列——此时根本没有真实差异，Bartlett 拒绝率 57.69%，而稳健检验都在 4% 到 6%。Bartlett 在 $r=1.6$ 时 91.25% 的「检出率」里，一大半是从这个虚警底盘上叠出来的。

诚实的读法是：Bartlett 的优势只在正态假设成立时兑现。一旦尾部变厚，它多出来的拒绝全是无意义的。这里没有「敏感但爱误报」的善意解读——它的检验统计量分布假设错了，p 值本身就不可信。

## 五、一个更贴近实盘的反例：波动相同，尾巴不同

第四个实验把场景做成一个直接的决策问题。策略 A 的日收益是标准正态，策略 B 是自由度 3 的 t 分布除以 $\sqrt{3}$，两者方差都精确等于 1，样本各 200 天。问：检验会不会说「B 的波动显著高于 A」？

![等波动不同尾的误判](/images/variance-equality-test-robust-heavy-tails/fig04_tail_misjudge.png)

Bartlett 有 **11.13%** 的概率这么喊，Brown-Forsythe 是 6.56%。但这里出现了本实验最值得记的反转：Fligner-Killeen 的误判率高达 **25.00%**，是四家里最差的。原因不难理解：Fligner 用绝对偏差的秩，t(3) 的极端值让很多观测离中位数很远、秩次靠前，它被这个「典型距离更大」的信号带偏，把厚尾读成了尺度差异。Levene 均值中心也有 6.25%，相对温和。

这条结论要写进工作流：稳健检验不是「放之四海皆准」。Brown-Forsythe 在本文四个实验里始终没有离谱表现；Fligner 在前三个实验里很稳，一旦遇到「方差相同、尾部不同」就翻车。如果你要比较的是波动率本身的差异，Brown-Forsythe 是更保守的选择；如果你还要防备偏斜，再看 Levene 的结果做交叉验证。

![样本量与虚警](/images/variance-equality-test-robust-heavy-tails/fig05_n.png)

第五张图把样本量扫描补齐：t(3) 下每组样本 30、60、120、240、480，Bartlett 虚警从 47.31% 单调爬到 73.94%，Brown-Forsythe 从 3.50% 到 5.19%，Fligner 从 4.19% 到 5.50%。「多留点数据就稳了」在这里是错的——对 Bartlett，更多数据只是让它更自信地违背前提。

## 六、落到工作流上的三条建议

第一，把 Bartlett 从默认工具箱里降级。除非你有充分理由相信数据近似正态（比如某些理论残差的诊断），否则不要用它的 p 值做波动差异的结论。第二，默认跑 Brown-Forsythe，它用中位数做中心，同时挡住了偏斜和厚尾，本文所有场景下虚警都贴近 5%。第三，当你的两组数据尾部明显不同时，别只跑一个检验——Fligner 在这种情形下会给出相反的信号，把它和 Brown-Forsythe 一起看，结论不一致就把「尾部差异」和「方差差异」分开报告。

再看一个把问题放大的场景：你不是在比两段，而是在几十个股票、几十个时间窗上批量做方差比较。如果每段都用 Bartlett 且收益厚尾，单段虚警率就过半，整批结果里会出现大量「波动结构变化」的假发现。这种批量筛选恰恰是量化研究里最常见的用法——扫一遍全市场找「波动率显著抬升」的标的。此时稳健检验的价值不是多检出几个真信号，而是把假信号的总量压下来，让后面的人工复核不至于被噪声淹没。Brown-Forsythe 在本文的单段实验里虚警不到 5%，放到批量场景仍会按名义水平累积，至少你知道自己付的是多少误报预算。

还有一个容易被忽略的点：这四种检验比的是「离散程度是否相等」，不是「风险是否相等」。如果你的策略对上行和下行波动的敏感度不同，或者你关心的是左尾损失而非双侧波动，方差齐性检验回答的就不是你的问题。那种情形下该看的是单边尾部的比较，比如分位数回归或对下偏矩做检验，而不是把两组收益塞进 Bartlett 然后读 p 值。工具和问题必须对得上，否则再稳健的检验也是在回答错误的问题。

这些实验的数字来自合成数据，目的是把每种检验的失效边界量化和可复现，不是声称某个真实市场的波动率如何。真实收益还有序列相关、波动聚集和结构性变点，那些会让所有检验的独立性假设进一步失效。但有一点不会变：一个建立在正态假设上的 p 值，用在肥尾数据上时，它的字面含义就不再是你以为的那个。

```python
# -*- coding: utf-8 -*-
"""Variance-equality tests under heavy tails: Bartlett vs robust alternatives.
Self-contained: all Monte Carlo data, five real figures and results.json from one script.
"""
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(20261127)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "variance-equality-test-robust-heavy-tails")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
NREP = 1600


def draw(rng, dist, n, scale=1.0):
    """Draw n samples from a distribution with unit scale, then apply scale."""
    if dist == "normal":
        x = rng.normal(0.0, 1.0, n)
    elif dist == "t5":
        x = rng.standard_t(5, n) / np.sqrt(5.0 / 3.0)
    elif dist == "t3":
        x = rng.standard_t(3, n) / np.sqrt(3.0)
    elif dist == "lognormal":
        x = rng.lognormal(0.0, 0.8, n) / np.exp(0.32)
    else:
        raise ValueError(dist)
    return x * scale


def run_tests(groups):
    """Return p-values of four variance-equality tests on a list of samples."""
    return (
        stats.bartlett(*groups).pvalue,
        stats.levene(*groups, center="mean").pvalue,
        stats.levene(*groups, center="median").pvalue,   # Brown-Forsythe
        stats.fligner(*groups).pvalue,
    )


TESTS = ["bartlett", "levene_mean", "brown_forsythe", "fligner"]
DISTS = ["normal", "t5", "t3", "lognormal"]


def size_experiment(rng, dist, sizes):
    """False-rejection rate under H0 (all groups share the same scale)."""
    rej = np.zeros(4, dtype=int)
    for _ in range(NREP):
        groups = [draw(rng, dist, n) for n in sizes]
        ps = run_tests(groups)
        for j, p in enumerate(ps):
            if p < ALPHA:
                rej[j] += 1
    return (rej / NREP).astype(float)


def main():
    res = {}
    rng = RNG

    # exp1: size under equal sample sizes, four distributions
    sizes_bal = [60, 60, 60]
    size_bal = {}
    for d in DISTS:
        size_bal[d] = size_experiment(rng, d, sizes_bal)
    res["exp1_tests"] = TESTS
    res["exp1_dists"] = DISTS
    res["exp1_size"] = {d: [float(v) for v in size_bal[d]] for d in DISTS}
    res["exp1_bartlett_normal"] = float(size_bal["normal"][0])
    res["exp1_bartlett_t3"] = float(size_bal["t3"][0])
    res["exp1_bartlett_lognormal"] = float(size_bal["lognormal"][0])
    res["exp1_bf_t3"] = float(size_bal["t3"][2])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(DISTS))
    w = 0.2
    colors = ["#c0392b", "#e67e22", "#2c7fb8", "#27ae60"]
    for j, (t, c) in enumerate(zip(TESTS, colors)):
        vals = [size_bal[d][j] for d in DISTS]
        ax.bar(x + (j - 1.5) * w, vals, w, label=t, color=c)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0, label="nominal 5%")
    ax.set_xticks(x); ax.set_xticklabels(["normal", "t(5)", "t(3)", "lognormal"])
    ax.set_ylabel("false rejection rate"); ax.set_ylim(0, 0.75)
    ax.set_title("Bartlett size explodes on heavy tails; robust tests hold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_size.png"), dpi=130); plt.close()

    # exp2: unbalanced sample sizes amplify the distortion
    sizes_unb = [20, 60, 300]
    size_unb = {}
    for d in ["normal", "t3"]:
        size_unb[d] = size_experiment(rng, d, sizes_unb)
    res["exp2_sizes_balanced"] = sizes_bal
    res["exp2_sizes_unbalanced"] = sizes_unb
    res["exp2_size"] = {d: [float(v) for v in size_unb[d]] for d in ["normal", "t3"]}
    res["exp2_bartlett_t3_unbalanced"] = float(size_unb["t3"][0])
    res["exp2_bartlett_t3_balanced"] = float(size_bal["t3"][0])

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(4)
    w = 0.35
    for j, (t, c) in enumerate(zip(TESTS, colors)):
        vals = [size_bal["t3"][j], size_unb["t3"][j]]
        ax.bar(x[:2] + (j - 1.5) * w, vals, w, label=t, color=c)
    for j, (t, c) in enumerate(zip(TESTS, colors)):
        vals = [size_bal["normal"][j], size_unb["normal"][j]]
        ax.bar(x[2:] + (j - 1.5) * w, vals, w, color=c, alpha=0.45)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0)
    ax.set_xticks([0.5, 2.5]); ax.set_xticklabels(["t(3)", "normal (faded)"])
    ax.set_ylabel("false rejection rate"); ax.set_ylim(0, 0.8)
    ax.set_title("Unbalanced groups: does distortion grow or shrink?")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_unbalanced.png"), dpi=130); plt.close()

    # exp3: power under a真实 variance ratio, normal vs t(3)
    ratios = [1.0, 1.2, 1.4, 1.6, 2.0]
    power = {"normal": {t: [] for t in TESTS}, "t3": {t: [] for t in TESTS}}
    for d in ["normal", "t3"]:
        for r in ratios:
            cnt = np.zeros(4, dtype=int)
            for _ in range(NREP):
                groups = [draw(rng, d, 80, 1.0), draw(rng, d, 80, r), draw(rng, d, 80, r)]
                ps = run_tests(groups)
                for j, p in enumerate(ps):
                    if p < ALPHA:
                        cnt[j] += 1
            for j, t in enumerate(TESTS):
                power[d][t].append(float(cnt[j] / NREP))
    res["exp3_ratios"] = ratios
    res["exp3_power"] = power
    res["exp3_bartlett_normal_r10"] = power["normal"]["bartlett"][0]
    res["exp3_bartlett_t3_r10"] = power["t3"]["bartlett"][0]
    res["exp3_fligner_t3_r16"] = power["t3"]["fligner"][3]
    res["exp3_bartlett_t3_r16"] = power["t3"]["bartlett"][3]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, d in zip(axes, ["normal", "t3"]):
        for t, c in zip(TESTS, colors):
            ax.plot(ratios, power[d][t], "o-", label=t, color=c)
        ax.axhline(ALPHA, color="#333", ls="--", lw=0.8)
        ax.set_xlabel("true scale ratio (group 2&3 vs 1)")
        ax.set_title(f"Power, {d}")
    axes[0].set_ylabel("rejection rate (power)")
    axes[0].legend(fontsize=7)
    fig.suptitle("At ratio=1 Bartlett already rejects on t(3): not power, but size")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_power.png"), dpi=130); plt.close()

    # exp4: two strategies with equal volatility but different tails
    mis = np.zeros(4, dtype=int)
    for _ in range(NREP):
        a = rng.normal(0.0, 1.0, 200)
        b = rng.standard_t(3, 200) / np.sqrt(3.0)
        ps = run_tests([a, b])
        for j, p in enumerate(ps):
            if p < ALPHA and np.var(b) > np.var(a):
                mis[j] += 1
    mis_rate = (mis / NREP).astype(float)
    res["exp4_tests"] = TESTS
    res["exp4_misjudge"] = [float(v) for v in mis_rate]
    res["exp4_bartlett_misjudge"] = float(mis_rate[0])
    res["exp4_bf_misjudge"] = float(mis_rate[2])

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(TESTS, mis_rate, color=colors)
    for bar, v in zip(bars, mis_rate):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0, label="true 5%")
    ax.set_ylabel('rate of claiming "B more volatile"')
    ax.set_ylim(0, max(0.3, float(mis_rate.max()) * 1.2))
    ax.set_title("Equal volatility, heavier tails: how often each test cries wolf")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_tail_misjudge.png"), dpi=130); plt.close()

    # exp5: does the size distortion vanish as n grows?
    ns = [30, 60, 120, 240, 480]
    nsize = {t: [] for t in TESTS}
    for n in ns:
        s = size_experiment(rng, "t3", [n, n, n])
        for j, t in enumerate(TESTS):
            nsize[t].append(float(s[j]))
    res["exp5_ns"] = ns
    res["exp5_size"] = nsize
    res["exp5_bartlett_n30"] = nsize["bartlett"][0]
    res["exp5_bartlett_n480"] = nsize["bartlett"][-1]
    res["exp5_fligner_n480"] = nsize["fligner"][-1]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for t, c in zip(TESTS, colors):
        ax.plot(ns, nsize[t], "o-", label=t, color=c)
    ax.axhline(ALPHA, color="#333", ls="--", lw=1.0, label="nominal 5%")
    ax.set_xscale("log"); ax.set_xlabel("per-group sample size n (t(3))")
    ax.set_ylabel("false rejection rate")
    ax.set_title("More data does not cure Bartlett under heavy tails")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_n.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```


## 七、参考阅读

方差估计本身的稳健性可以接着看 [Tyler M 估计量](/blog/tyler-m-estimator-covariance/)与[最小协方差行列式](/blog/robust-covariance-mcd/)：它们处理的是「协方差矩阵被异常值污染后怎么估」，本文处理的是「两组波动是否相等怎么检验」，一个估参数、一个做判断。厚尾收益的建模见 [alpha 稳定分布](/blog/alpha-stable-return-distribution/)。
