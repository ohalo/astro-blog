---
title: "Wilcoxon 符号秩检验：一个 -9% 的交易日如何让配对 t 检验哑火，厚尾下它又凭什么多抓 18 个百分点功效"
description: "配对策略收益差的三种检验：配对 t、符号检验、Wilcoxon 符号秩。n=100 真值实验里 t(3) 厚尾下 Wilcoxon 功效 0.9267 对 t 的 0.7477，拉普拉斯 0.8340 对 0.7100；偏斜原假设下 t 误报 0.9117、Wilcoxon 0.3850，只有符号检验守住 0.0353。往 200 天样本塞一个 -20σ 的日子，t 检出率从 0.7975 掉到 0，Wilcoxon 稳在 0.7275。附小样本精确临界、120 天实盘对照与五张真实计算图。"
publishDate: '2026-11-24T09:10:00Z'
tags:
  - 量化
  - 统计检验
  - 稳健统计
  - 策略评估
  - Python
language: zh
difficulty: advanced
image: "/images/wilcoxon-signed-rank-robust-test/fig1_distributions.png"
---

## 一、结论先行

比较两个策略谁更赚钱，最自然的做法是算每天的收益差 $d_t=r_{A,t}-r_{B,t}$，再检验它是否以零为中心。标准答案是配对 t 检验，但它在实盘里有个致命弱点：分母的样本标准差会被某一天极端亏损单独抬高。

本文用真值已知的蒙特卡洛把三种检验放在同一批数据上对比：配对 t、只看正负号的符号检验、给 $|d_t|$ 排秩再加符号的 Wilcoxon 符号秩检验。四个硬结论：

- 正态下 t 略胜：功效 0.6877 对 Wilcoxon 的 0.6720，差距只有 1.6 个百分点，符号检验只有 0.4537。
- 厚尾下 Wilcoxon 大胜：t(3) 上 0.9267 对 0.7477，拉普拉斯上 0.8340 对 0.7100。
- 偏斜原假设下 t 彻底失控：中位数为零、均值不为零的偏斜分布里，t 的误报率 0.9117，Wilcoxon 也有 0.3850，只有符号检验守在 0.0353。
- 一个交易日就能让 t 闭嘴：200 天样本里塞进一个 -20 倍标准差的坏日子，t 检出率从 0.7975 掉到 0，Wilcoxon 稳在 0.7275。

## 二、三种检验到底在检验什么

配对 t 检验统计的是均值：$t=\bar d/(s_d/\sqrt n)$。它用的是差值的数值大小，所以信息利用充分，代价是 $s_d$ 无界——单日极端值能把它抬到任意大。

符号检验只数正的 $d_t$ 有多少个，在原假设 $P(d_t>0)=0.5$ 下服从二项分布。它不管差值有多极端，因此完全稳健；但也因此丢掉了幅度信息，正态下功效天然吃亏。

Wilcoxon 符号秩走中间路线：对 $|d_t|$ 排序得到秩 $R_t$，把正负号贴回秩上，取正秩之和 $W^+$。无并列时 $E[W^+]=n(n+1)/4$，方差 $n(n+1)(2n+1)/24$；有并列时减去 $\sum(t_j^3-t_j)/48$ 的修正项，再用正态近似算双侧 p 值。它用的不是原始幅度而是幅度的次序，所以极端值再大也只占一个名次。

一个关键区别：t 检验针对均值，符号检验针对中位数，符号秩在原假设对称时针对中位数，分布偏斜时它实际上检验的是成对平均值分布的中心，不再等于中位数。把它当成万能的中位数检验，是后面那个 0.3850 误报率的根源。

## 三、五种分布的正面对决

![五种噪声分布下三种检验的误报率与功效](/images/wilcoxon-signed-rank-robust-test/fig1_distributions.png)

设定 n=100、每档 3000 次重复，噪声分别取正态、缩放后的 t(3)、拉普拉斯、1% 污染正态（其中 1% 的样本标准差放大 5 倍）、中心化指数分布。前四种都关于零对称，最后一种把中位数精确移到零但均值是 $1-\log 2\approx0.307$。功效档给所有差值加 0.25 的平移。

- **正态**：t 的误报 0.0540、功效 0.6877；Wilcoxon 0.0570、0.6720；符号 0.0390、0.4537。t 的主场，但优势小得可怜。
- **t(3)**：t 功效 0.7477，符号 0.8423，Wilcoxon 0.9267。Wilcoxon 比 t 多抓 17.9 个百分点。
- **拉普拉斯**：Wilcoxon 0.8340、符号 0.8263、t 0.7100。尖峰厚尾下前两者几乎一样好。
- **1% 污染正态**：Wilcoxon 0.7673、t 0.7167、符号 0.5456。污染只在 1% 的样本上，t 就被拖掉 5 个百分点。
- **偏斜**：t 误报 0.9117，Wilcoxon 0.3850，符号 0.0353。这三种里只有符号检验是对的，因为它唯一不依赖对称性。

偏斜那一栏最值得盯：t 的高误报不是方差估错，而是它检验的均值本来就不等于零；Wilcoxon 虽然不碰幅度，却因为正负秩的分布不对称而系统性偏向一侧。一个策略的日收益分布几乎不可能严格对称，这意味着「差值均值显著为正」和「策略有正中位数优势」是两件不同的事。

## 四、样本量改变不了方法的排序

![样本量扫描：正态与 t(3) 下三种检验的功效曲线](/images/wilcoxon-signed-rank-robust-test/fig2_sample_size.png)

固定平移 0.35，n 从 20 扫到 320。正态下 n=20 时 t 0.3180、Wilcoxon 0.3010、符号 0.1995；n=80 时 0.8650、0.8585、0.6370；n=160 后 t 与 Wilcoxon 都超过 0.99。t 与 Wilcoxon 的差距在正态下始终只有一到两个百分点。

t(3) 下排序一开始就不同：n=20 时 Wilcoxon 0.5175 已高于 t 和符号的 0.4395；n=40 时 0.8215 对 t 的 0.6700；n=80 时 0.9860 对 0.8950。样本量增加不会让 t 追上来，因为厚尾造成的效率损失是渐近的，不是小样本偏差。符号检验在 n=160 时已经到 1.0000，靠的是只数符号反而不受极端值干扰。

## 五、一个坏日子就能让 t 检验失效

![单日极端值注入时三种检验的检出率与平均 p 值](/images/wilcoxon-signed-rank-robust-test/fig3_outlier.png)

这组实验把 200 天的真效应设为 0.20，噪声标准正态，然后把第一天的差值减去 M。M=0 时 t 的检出率 0.7975、Wilcoxon 0.7775；M=2 时分别 0.7550、0.7350；M=5 时 t 掉到 0.6475 而 Wilcoxon 还有 0.7275；M=10 时 t 只剩 0.3725；M=20 时 t 是 0.0150，M≥50 直接归零。Wilcoxon 全程稳定在 0.7275，符号检验稳定在 0.5300 到 0.5550。

数学上说清楚了这件事：往样本里塞一个幅度 M，均值被拉走约 M/n，样本标准差却被拉到约 $M/\sqrt n$，标准误约 $M/n$，于是 t 统计量趋近于 1 而不是发散——无论 M 多大，t 都拿不到证据。符号检验只贡献一个负号，Wilcoxon 只贡献一个最大秩，两者对 M 的敏感度是常数级的。

这条结论可以直接翻译成交易语言：如果你的策略回测里有一天因为熔断、跌停或数据错误产生极端值，配对 t 检验会把整段回测判成无优势，而秩检验不会。它不是在帮你掩盖那一天，而是在告诉你「剩下的 199 天仍然一致地支持这个方向」。

## 六、小样本要用精确零分布

![小样本下精确临界值与正态近似的实测误报率](/images/wilcoxon-signed-rank-robust-test/fig4_exact.png)

正态近似在 n 很小时偏激进。用动态规划枚举全部 $2^n$ 种符号组合，可以算出 $W^+$ 的精确零分布。n=8 时精确临界偏移 15.0、近似临界 14.0，实测误报 0.0390 对 0.0533；n=12 时 26.0 对 24.98，误报 0.0470 对 0.0553；n=16 时 39.0 对 37.90，误报 0.0405 对 0.0462。到 n=20 两者都给 0.0482，n=30 都是 0.0528，近似已经够用。

样本少于 20 个交易日时别用 1.96 那套，直接枚举符号组合更稳妥。代码里这套 DP 只算一次，成本可以忽略。另外，本文手写的 p 值与 scipy.stats.wilcoxon 在同一个 60 天样本上逐位相同，双侧 p 都是 0.000671249488118936，实现本身有对拍基准。

## 七、120 天回测里的三种判决

![120 天策略收益差与三种检验的 p 值对比](/images/wilcoxon-signed-rank-robust-test/fig5_live.png)

最后造一段具体的配对回测：B 是基准，A 每天有 0.0035 的日均超额、0.011 的独立噪声，第 48 天 A 多亏 0.087。120 天里 A 有 78 天跑赢 B，平均日差 0.002333、中位数 0.003761、累计 0.280006。结果是配对 t 给 p=0.0653，5% 水平下不显著；符号检验 p=0.0013，Wilcoxon p=0.0020，两个秩方法都显著。

三种检验看的是同一段数据，分歧完全来自那一天：t 被 -0.087 拉高了标准差，秩方法只看得出大多数日子 A 都在前面。这不是说秩检验一定正确，而是说当收益分布有跳跃时，「均值是否显著」和「多数时候是否占优」本来就该分开回答。

## 八、结值、零值与配对前提

实盘收益常被最小报价单位切成离散格子，差值为零的日子不少。三种检验对零值的处理不同：符号检验直接剔除零值再数正负，Wilcoxon 把零值剔除后对剩下的绝对值排秩，并列值取平均秩并进入方差修正项，t 检验则把零当作普通观测保留。代码里的 wilcoxon_p 用 stats.rankdata 处理并列、用 $\sum(t_j^3-t_j)/48$ 修正方差，sign_p 只对非零项做二项检验。若 120 天里有 30 天完全持平，符号检验的有效样本只剩 90，功效会明显下降；这时不能把「没有差异」混同为「支持原假设」。

另一个前提是配对。把 A 的 120 天收益和 B 的另外 120 天收益硬凑成差值，日历错位会引入噪声，检验失去意义。日收益还常带序列相关，本文实验用的是独立新息；若差值序列的一阶自相关不为零，三种检验的标准误都会偏小。处理办法是先用区块自助或 HAC 校准（本站依赖数据自助法一文有完整对照），而不是换一个秩统计量。秩方法修的是幅度分布，不修时间依赖。

## 九、怎么选

正态且干净的数据用 t，它是效率上限；厚尾、污染、或者回测里有可疑极端日，用 Wilcoxon；分布明显偏斜、或者你只敢假设中位数为零而不敢假设对称，用符号检验——代价是正态下功效从 0.69 掉到 0.45。三者都不告诉你可以放宽止损，它们只告诉你证据有多强。本文的完整实现如下，脚本本身即所有数字与图片的来源。

```python
# -*- coding: utf-8 -*-
"""Wilcoxon signed-rank vs paired t-test vs sign test, on paired strategy returns.

Single source of truth for every number and figure in the article. The paired
difference d_t = r_A,t - r_B,t is tested for a zero centre. Three tests are
compared on the same data:

  * paired t: uses the mean, divides by a sample SD that a single extreme day
    can inflate without bound;
  * sign test: uses only the count of positive differences, distribution-free
    for the median but throws away all magnitudes;
  * Wilcoxon signed-rank: rank |d_t|, attach the sign, sum positive ranks.
    Its asymptotic variance carries a tie correction.

Five truth-known experiments: (1) size and power under five null distributions
including heavy tails, contamination and skewness; (2) power versus sample
size; (3) one outlier injected at growing magnitude; (4) exact small-sample
null versus the normal approximation; (5) a concrete 120-day backtest where the
three tests disagree. A cross-check against scipy.stats.wilcoxon is included.
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "wilcoxon-signed-rank-robust-test")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------- three tests
def wilcoxon_p(d):
    """Two-sided signed-rank p-value, normal approximation with tie correction."""
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    n = nz.size
    if n < 2:
        return 1.0
    a = np.abs(nz)
    r = stats.rankdata(a)
    w_plus = r[nz > 0].sum()
    mu = n * (n + 1) / 4.0
    _, cnt = np.unique(a, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - np.sum(cnt ** 3 - cnt) / 48.0
    z = (w_plus - mu) / np.sqrt(max(var, 1e-12))
    return float(2.0 * stats.norm.sf(abs(z)))


def wilcoxon_stat(d):
    """Return (W+, n, mu, variance) used by the exact-size experiment."""
    nz = np.asarray(d, dtype=float)
    nz = nz[nz != 0]
    n = nz.size
    a = np.abs(nz)
    r = stats.rankdata(a)
    w = float(r[nz > 0].sum())
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    return w, n, mu, var


def sign_p(d):
    """Two-sided exact binomial sign test on non-zero differences."""
    nz = np.asarray(d, dtype=float)
    nz = nz[nz != 0]
    n = nz.size
    if n == 0:
        return 1.0
    k = int((nz > 0).sum())
    lower = stats.binom.cdf(k, n, 0.5)
    upper = stats.binom.sf(k - 1, n, 0.5)
    return float(min(1.0, 2.0 * min(lower, upper)))


def t_p(d):
    """Two-sided paired t-test p-value."""
    d = np.asarray(d, dtype=float)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    if se <= 0:
        return 1.0
    t = d.mean() / se
    return float(2.0 * stats.t.sf(abs(t), n - 1))


def exact_wilcoxon_crit(n, alpha=0.05):
    """Exact lower critical sum for W+ over all 2^n sign patterns (no ties)."""
    S = n * (n + 1) // 2
    dp = np.zeros(S + 1, dtype=np.float64)
    dp[0] = 1.0
    for r in range(1, n + 1):
        new = dp.copy()
        new[r:] += dp[:S + 1 - r]
        dp = new
    cdf = np.cumsum(dp) / (2.0 ** n)
    below = np.where(cdf <= alpha / 2.0 + 1e-12)[0]
    lower = int(below[-1]) if below.size else 0
    mu = S / 2.0
    return lower, float(mu - lower)


# ------------------------------------------------------------- DGP helpers
def base_noise(kind, rng, n):
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "t3":
        return stats.t.rvs(3, size=n, random_state=rng) / np.sqrt(3.0)
    if kind == "laplace":
        return stats.laplace.rvs(size=n, random_state=rng) / np.sqrt(2.0)
    if kind == "contaminated":
        z = rng.standard_normal(n)
        big = rng.standard_normal(n) * 5.0
        m = rng.random(n) < 0.01
        z[m] = big[m]
        return z / np.sqrt(0.99 + 0.01 * 25.0)
    if kind == "skewed":
        # centred exponential: median exactly zero, mean 1-log(2), right skew
        return rng.exponential(scale=1.0, size=n) - np.log(2.0)
    raise ValueError(kind)


# ------------------------------------------------------------- experiments
def exp_distributions(reps=3000, n=100):
    kinds = ["normal", "t3", "laplace", "contaminated", "skewed"]
    out = {}
    for j, kind in enumerate(kinds):
        rng = np.random.default_rng(SEED + 100 + j)
        for delta in (0.0, 0.25):
            e = base_noise(kind, rng, n * reps).reshape(reps, n)
            x = e + delta
            key = kind + ("_size" if delta == 0.0 else "_power")
            out[key] = {
                "delta": delta,
                "t": float(np.mean([t_p(row) < 0.05 for row in x])),
                "sign": float(np.mean([sign_p(row) < 0.05 for row in x])),
                "wilcoxon": float(np.mean([wilcoxon_p(row) < 0.05 for row in x])),
            }
    # independent cross-check of our p-value against scipy on one fixed sample
    rng = np.random.default_rng(SEED + 777)
    d = base_noise("normal", rng, 60) + 0.3
    sp = float(stats.wilcoxon(d).pvalue)
    out["scipy_check"] = {"ours": wilcoxon_p(d), "scipy": sp,
                          "abs_diff": abs(wilcoxon_p(d) - sp)}
    return out


def exp_sample_size(reps=2000, deltas=(0.35,)):
    out = {}
    for j, kind in enumerate(("normal", "t3")):
        rng = np.random.default_rng(SEED + 300 + j)
        grid = {}
        for n in (20, 40, 80, 160, 320):
            e = base_noise(kind, rng, n * reps).reshape(reps, n)
            x = e + deltas[0]
            grid[str(n)] = {
                "t": float(np.mean([t_p(row) < 0.05 for row in x])),
                "sign": float(np.mean([sign_p(row) < 0.05 for row in x])),
                "wilcoxon": float(np.mean([wilcoxon_p(row) < 0.05 for row in x])),
            }
        out[kind] = {"delta": deltas[0], "grid": grid}
    return out


def exp_outlier(reps=400, n=200):
    rng = np.random.default_rng(SEED + 400)
    clean = 0.20 + rng.standard_normal((reps, n))
    rows = {}
    for M in (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        x = clean.copy()
        x[:, 0] -= M
        tp = np.array([t_p(r) for r in x])
        wp = np.array([wilcoxon_p(r) for r in x])
        sp = np.array([sign_p(r) for r in x])
        rows[str(M)] = {
            "t_p_mean": float(tp.mean()),
            "t_detect": float(np.mean(tp < 0.05)),
            "wilcoxon_p_mean": float(wp.mean()),
            "wilcoxon_detect": float(np.mean(wp < 0.05)),
            "sign_detect": float(np.mean(sp < 0.05)),
        }
    return rows


def exp_exact_small_n(reps=6000):
    rows = {}
    for n in (8, 12, 16, 20, 30):
        lower, c_exact = exact_wilcoxon_crit(n)
        c_asym = 1.959963984540054 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        rng = np.random.default_rng(SEED + 500 + n)
        x = rng.standard_normal((reps, n))
        sz_exact = []
        sz_asym = []
        for row in x:
            w, nn, mu, var = wilcoxon_stat(row)
            sz_exact.append(abs(w - mu) >= c_exact - 1e-9)
            sz_asym.append(abs(w - mu) >= c_asym)
        rows[str(n)] = {
            "lower_critical_sum": lower,
            "c_exact": c_exact,
            "c_asymptotic": float(c_asym),
            "size_exact": float(np.mean(sz_exact)),
            "size_asymptotic": float(np.mean(sz_asym)),
        }
    return rows


def exp_live_backtest(n=120):
    rng = np.random.default_rng(SEED + 900)
    rb = 0.0003 + 0.012 * rng.standard_normal(n)
    ra = rb + 0.0035 + 0.011 * rng.standard_normal(n)
    d = ra - rb
    d[47] -= 0.09  # one bad day for strategy A
    cum = np.cumsum(d)
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "sd_diff": float(d.std(ddof=1)),
        "t_p": t_p(d),
        "sign_p": sign_p(d),
        "wilcoxon_p": wilcoxon_p(d),
        "positive_days": int((d > 0).sum()),
        "worst_day_index": int(np.argmin(d)),
        "worst_day": float(d.min()),
        "final_cum": float(cum[-1]),
        "cum": cum.tolist(),
        "diffs": d.tolist(),
    }


# ------------------------------------------------------------------ figures
def fig1_distributions(res):
    kinds = ["normal", "t3", "laplace", "contaminated", "skewed"]
    labels = ["normal", "t(3)", "Laplace", "contam 1%", "skewed"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    w = 0.26
    xs = np.arange(len(kinds))
    methods = [("t", "#b23a2b"), ("sign", "#2b8cbe"), ("wilcoxon", "#3a923a")]
    for panel, suffix, target in ((0, "_size", 0.05), (1, "_power", 0.05)):
        for i, (m, col) in enumerate(methods):
            vals = [res[k + suffix][m] for k in kinds]
            ax[panel].bar(xs + (i - 1) * w, vals, w, label=m, color=col, alpha=0.88)
        ax[panel].axhline(target, color="#333", ls="--", lw=1.2)
        ax[panel].set_xticks(xs)
        ax[panel].set_xticklabels(labels, fontsize=9)
        ax[panel].set_ylim(0, 1.02)
        ax[panel].set_title("size (delta=0): dashed = 5%" if panel == 0
                            else "power (delta=0.25)")
        ax[panel].legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_distributions.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_distributions.png"


def fig2_sample_size(res):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for j, (kind, title) in enumerate((("normal", "normal noise"),
                                       ("t3", "t(3) noise"))):
        grid = res[kind]["grid"]
        ns = sorted(int(k) for k in grid)
        for m, col in (("t", "#b23a2b"), ("sign", "#2b8cbe"),
                       ("wilcoxon", "#3a923a")):
            ax[j].plot(ns, [grid[str(n)][m] for n in ns], "o-",
                       label=m, color=col, lw=1.8)
        ax[j].set_title(title + "  delta=0.35")
        ax[j].set_xlabel("sample size n")
        ax[j].set_ylim(0, 1.02)
        ax[j].legend(fontsize=9)
    ax[0].set_ylabel("rejection rate at 5%")
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_sample_size.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_sample_size.png"


def fig3_outlier(rows):
    ms = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for key, lab, col in (("t_p_mean", "paired t (mean p)", "#b23a2b"),
                          ("wilcoxon_p_mean", "Wilcoxon (mean p)", "#3a923a"),
                          ("sign_p_mean", "sign (mean p)", "#2b8cbe")):
        if key in rows[str(ms[0])]:
            ax.plot(ms, [rows[str(m)][key] for m in ms], "o-",
                    label=lab, color=col, lw=1.8)
    ax.plot(ms, [rows[str(m)]["t_detect"] for m in ms], "s--",
            color="#b23a2b", alpha=0.5, label="t detection rate")
    ax.plot(ms, [rows[str(m)]["wilcoxon_detect"] for m in ms], "s--",
            color="#3a923a", alpha=0.5, label="Wilcoxon detection rate")
    ax.axhline(0.05, color="#333", ls=":", lw=1.2)
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlabel("injected outlier magnitude M (one day out of 200)")
    ax.set_ylabel("mean p-value / detection rate")
    ax.set_title("One bad day: t loses the signal, signed-rank does not")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_outlier.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_outlier.png"


def fig4_exact(rows):
    ns = sorted(int(k) for k in rows)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(ns, [rows[str(n)]["size_exact"] for n in ns], "o-",
            color="#3a923a", lw=1.8, label="exact critical value")
    ax.plot(ns, [rows[str(n)]["size_asymptotic"] for n in ns], "o-",
            color="#b23a2b", lw=1.8, label="normal approximation 1.96 sigma")
    ax.axhline(0.05, color="#333", ls="--", lw=1.2, label="nominal 5%")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("empirical size")
    ax.set_title("Small samples: exact null vs normal approximation")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_exact.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_exact.png"


def fig5_live(live):
    d = np.array(live["diffs"])
    cum = np.array(live["cum"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(np.arange(1, len(cum) + 1), cum, color="#2b8cbe", lw=1.5)
    ax[0].axvline(live["worst_day_index"] + 1, color="#b23a2b", ls="--", lw=1.3)
    ax[0].annotate("one -9% day", xy=(live["worst_day_index"] + 1,
                                      cum[live["worst_day_index"]]),
                   xytext=(18, 8), textcoords="offset points", fontsize=9,
                   color="#b23a2b")
    ax[0].set_xlabel("trading day")
    ax[0].set_ylabel("cumulative daily difference")
    ax[0].set_title("Strategy A minus B: cumulative edge")
    names = ["paired t", "sign", "Wilcoxon"]
    vals = [live["t_p"], live["sign_p"], live["wilcoxon_p"]]
    cols = ["#b23a2b", "#2b8cbe", "#3a923a"]
    ax[1].bar(names, vals, color=cols, alpha=0.88)
    ax[1].axhline(0.05, color="#333", ls="--", lw=1.2)
    ax[1].set_ylabel("two-sided p-value")
    ax[1].set_title("Same 120 days, three verdicts")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.012, "%.4f" % v, ha="center", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_live.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_live.png"


def main():
    dist = exp_distributions()
    size = exp_sample_size()
    out = exp_outlier()
    exact = exp_exact_small_n()
    live = exp_live_backtest()
    r = {
        "setup": {"seed": SEED, "alpha": 0.05, "reps_distributions": 3000,
                  "n_distributions": 100, "reps_size": 2000,
                  "reps_outlier": 400, "n_outlier": 200,
                  "reps_exact": 6000},
        "distributions": dist,
        "sample_size": size,
        "outlier": out,
        "exact": exact,
        "live": live,
    }
    r["figs"] = [fig1_distributions(dist), fig2_sample_size(size),
                 fig3_outlier(out), fig4_exact(exact), fig5_live(live)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

