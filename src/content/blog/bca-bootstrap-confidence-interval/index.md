---
title: "自助置信区间里的偏差校正：BCa 凭什么比 percentile 准，以及它在小样本和边界上照样翻车"
description: "percentile 自助区间默认 bootstrap 分布是样本均值的无偏镜像，一旦统计量有偏、抽样分布会随参数变斜，它就左右失衡。本文在同一批 bootstrap 重抽上并排算 normal、percentile、basic 和 BCa 四种区间，真值已知实验显示：指数均值上 percentile 左漏 8.53%、右漏 1.40%，BCa 把两头拉回 6.00% 对 2.13%；正态方差上 percentile 覆盖 83.8%，BCa 拉到 86.07%；相关系数上 BCa 覆盖 93.78% 最高。但 BCa 不是万能：正态均值这种最干净的场子它反而只有 92.07%，输给 normal 的 93.00%；二项比例真值 0.02 时 z0 饱和 63.62%，BCa 和 percentile 一起崩到 36.38%。附完整代码与五张真实计算图。"
publishDate: '2026-11-23T09:10:00Z'
tags:
  - 量化
  - 统计推断
  - 自助法
  - 置信区间
  - Python
language: zh
difficulty: advanced
image: "/images/bca-bootstrap-confidence-interval/fig1_coverage.png"
---

## 一、结论先行

给一个统计量做置信区间，最省事的 bootstrap 做法是：重抽 B 次，把 $\hat\theta^*_b$ 排好队，取 2.5% 和 97.5% 分位数当区间端点。这就是 percentile 区间，它隐含一个很强的假设——bootstrap 分布是抽样分布的「无偏镜像」，而且形状不随真实参数变。

这两个假设一旦破掉，区间就会向一侧倾斜。BCa（bias-corrected and accelerated，Efron 1987）针对的正是这两处：用 $z_0$ 修中位数偏差，用加速因子 $a$ 修「抽样分布随参数变斜变宽」。

本文在**同一批 bootstrap 重抽**上并排算 normal、percentile、basic、BCa 四种区间，配对比较，避免把重抽样噪声误当成方法差异。真值已知的结果分两面：

- BCa 在偏斜统计量上确实赢。指数均值场景 percentile 覆盖 90.07%，BCa 91.87%；正态方差 percentile 83.80%、BCa 86.07%；相关系数 BCa 93.78% 四种里最高。
- BCa 不是万能。正态均值这种教科书最干净的场子，它只有 92.07%，反而输给 normal 近似和 basic 的 93.00%；二项比例真值 0.02 时 $z_0$ 饱和 63.62%，BCa 和 percentile 一起崩到 36.38%。

## 二、四种区间吃的是同一批重抽

设 $\hat\theta$ 是样本统计量，$\hat\theta^*_1,\dots,\hat\theta^*_B$ 是 B 个 bootstrap 重抽统计量。四种区间都从这批数出发：

- **normal**：$\hat\theta \pm 1.96\,\widehat{se}_{boot}$，假设抽样分布对称且近似正态。
- **percentile**：直接取 $\hat\theta^*$ 的 2.5% 与 97.5% 分位数。
- **basic**：$[2\hat\theta-\hat\theta^*_{0.975},\ 2\hat\theta-\hat\theta^*_{0.025}]$，把 percentile 关于 $\hat\theta$ 翻过去。
- **BCa**：先把目标分位数改到校正后的 $\alpha_1,\alpha_2$，再取 $\hat\theta^*$ 分位数。

配对是关键。四个方法共享同一组 $\hat\theta^*$，于是「谁覆盖高」反映的是端点算法本身，而不是这一轮恰好抽到什么样本。

## 三、$z_0$ 和加速因子各修什么

$z_0$ 度量中位数偏差。它数 bootstrap 分布里有多少比例落在点估计下方：

$$
z_0=\Phi^{-1}\!\left(\frac{\#\{\hat\theta^*_b<\hat\theta\}}{B}\right)
$$

如果 $\hat\theta$ 正好在 bootstrap 分布中位数上，比例约 0.5，$z_0\approx 0$。指数样本的均值有右偏，点估计常落在 bootstrap 分布偏左的位置，于是 $z_0>0$——本文单个指数样本实测 $\hat\theta=0.8661$、$z_0=0.0841$。

加速因子 $a$ 度量「抽样分布的标准差会不会随真参数变」。用刀切法（jackknife）逐次剔一个观测得到 $\hat\theta_{(i)}$，令 $u_i=\bar{\hat\theta}_{(\cdot)}-\hat\theta_{(i)}$：

$$
a=\frac{\sum_i u_i^3}{6\big(\sum_i u_i^2\big)^{3/2}}
$$

它就是刀切估计的偏度除以 6。直觉是：如果参数越大分布越斜，固定分位数端点就该跟着「加速」移动，而不是死守 2.5% 和 97.5%。指数均值场景 $a$ 平均 0.0498、标准差 0.0237，稳定为正；正态均值的 $a$ 均值只有 0.000。

有了两者，端点分位数改成：

$$
\alpha_1=\Phi\!\left(z_0+\frac{z_0+z_{0.025}}{1-a(z_0+z_{0.025})}\right),\quad \alpha_2=\Phi\!\left(z_0+\frac{z_0+z_{0.975}}{1-a(z_0+z_{0.975})}\right)
$$

$\alpha_1,\alpha_2$ 不再是固定的 0.025/0.975，而是随样本偏斜自动移位。那个样本算出来 $\alpha_1=0.0573$、$\alpha_2=0.9931$——左端点被推到 5.73% 分位，右端点几乎顶到最大端。

## 四、五个真值场景的覆盖

![配对 bootstrap 区间覆盖率：BCa 只在偏差与偏度确实存在的场景占优](/images/bca-bootstrap-confidence-interval/fig1_coverage.png)

正态均值（n=20，真值 0）：normal 0.9300、percentile 0.9287、basic 0.9300、BCa 0.9207。四种都在 95% 名义线下（小样本用 1.96 本身就偏窄），BCa 还略低——没有偏差可修时，多估的两个参数纯属引入噪声。

指数均值（真值 1）：normal 0.8920、percentile 0.9007、basic 0.8773、BCa 0.9187。正态近似输得最惨，因为它假设对称，而指数抽样分布明摆着右偏。

对数正态均值（真值 $e^{0.125}$）：normal 0.8987、percentile 0.9007、basic 0.8900、BCa 0.9027。四种差距都小，BCa 只是略胜。

正态方差（用无偏样本方差，真值 1）：normal 0.8413、percentile 0.8380、basic 0.8380、BCa 0.8607。卡方分布本身就偏，BCa 把覆盖从 83.8% 拉到 86.07%，但离 95% 仍远。这是小样本下所有区间方法的共同短板。

相关系数（二元正态 $\rho=0.6$，n=30，B=1200）：normal 0.9144、percentile 0.9322、basic 0.8756、BCa 0.9378。相关系数有界、非线性，basic 的翻折端点最容易越界，BCa 的移位分位数在这里最稳。

## 五、失衡的尾部被拉回来了

![percentile 区间左右漏失不对称，BCa 重新平衡两端](/images/bca-bootstrap-confidence-interval/fig2_tail_balance.png)

覆盖率高不代表区间好，还得看漏失落在哪一边。指数均值上，percentile 左漏（真值落在区间右侧，即区间整体偏低）8.53%、右漏只有 1.40%，严重偏向左——它没意识到点估计系统性低估。BCa 把两端拉到 6.00% 对 2.13%。

正态方差更夸张：percentile 左漏 15.73%、右漏 0.47%，几乎是单边区间；BCa 改成 10.33% 对 3.60%。相关性场景则相反，percentile 右漏（2.44% 对 4.33%）更多，说明相关系数分布向左偏，BCa 同样把两端拉近到 2.67% 对 3.56%。

这种「失衡—再平衡」正是 BCa 的价值：它不改变区间长度的主导因素，只是把端点挪到该在的位置。

## 六、拆开一个样本看

![单个指数样本：bootstrap 分布与两种端点位置](/images/bca-bootstrap-confidence-interval/fig3_mechanics.png)

取一个具体的指数样本，n=20，样本均值 0.8661，真值 1.0。bootstrap 分布偏度 0.4455。四种区间：

- normal：[0.5149, 1.2173]
- percentile：[0.5554, 1.2558]
- basic：[0.4764, 1.1767]
- BCa：[0.6027, 1.3599]

BCa 的左端点比 percentile 抬高 0.0473、右端点抬高 0.1041，整体上移——因为 $z_0=0.0841>0$ 说明点估计落在 bootstrap 分布偏左处，端点该往右补。它成功把真值 1.0 纳入更合理的居中位置，而 basic 区间右端 1.1767 也覆盖到了，但整体偏左更多。

## 七、加速因子单独贡献多少

![刀切加速因子在偏斜统计量上稳定为正](/images/bca-bootstrap-confidence-interval/fig4_acceleration.png)

为了拆开 $z_0$ 和 $a$ 的贡献，做一次消融：保留 $z_0$ 校正但把 $a$ 强行设成 0。指数均值上，「只做 $z_0$」覆盖 91.87%，完整 BCa 覆盖 92.50%——加速因子单独贡献 0.63 个百分点。这个数字不大，但方向与理论一致：偏斜统计量上，加速项确实再推一把。

值得警惕的是，加速因子本身也是估计量。n=20 时它的标准差 0.0237，5% 到 95% 分位在 0.0175 到 0.0951，样本量越小抖动越大。这是 BCa 在小样本不稳定的根源之一。

## 八、边界和小样本照样翻车

![z0 在比例边界饱和；小样本下没有方法达到名义覆盖](/images/bca-bootstrap-confidence-interval/fig5_boundary.png)

二项比例的实验中，真比例 $p$ 取 0.02、0.05、0.10、0.30、0.50，n=20。当 $p$ 很小时，大量样本一个成功事件都没抽到，bootstrap 分布几乎全堆在 0，$z_0$ 顶到 $\Phi^{-1}(1-1/B)$ 的饱和上限：

- $p=0.02$：$z_0$ 饱和比例 63.62%，percentile 和 BCa 覆盖都只有 36.38%。
- $p=0.05$：饱和 36.75%，覆盖 63.00% 对 63.25%。
- $p=0.10$：饱和 12.00%，覆盖 88.00% 对 88.00%。
- $p=0.30$：饱和 0.13%，percentile 95.63%、BCa 96.37%。
- $p=0.50$：饱和 0%，percentile 96.00%、BCa 95.63%。

结论很清楚：极端比例下所有区间方法都失效，BCa 并不比 percentile 强。问题出在数据本身——样本里没有足够信息，任何端点算法都救不回来。

小样本扫描也不乐观。指数均值 n=5 时四种方法覆盖全部在 77.11% 到 79.22%；n=10 时 BCa 89.56% 略好；n=80 时 BCa 94.33% 才勉强接近名义。样本量不够时，bootstrap 分布本身就是粗糙近似，$z_0$ 和 $a$ 都估不准。

## 九、为什么偏偏是这两个修正

理解 BCa 的适用边界，得先弄清 percentile 区间到底错在哪。它把 bootstrap 分布 $\hat\theta^*-\hat\theta$ 的分位数直接当成 $\hat\theta-\theta$ 的抽样误差分布，这依赖一个「平移等价」假设：从样本到总体的误差分布，和从重抽样本到原样本的误差分布形状一致。

当统计量是样本均值时，这个假设近似成立，因为均值是线性的、渐近正态。可换成方差、相关系数、比率这类非线性统计量，抽样分布的偏度和尺度会随真参数值变化。举个直观的例子：相关性越接近 1，相关系数的抽样分布越被边界挤向左边，形状根本不是平移不变的。这时 percentile 拿一个固定形状去套所有参数值，端点必然偏。

$z_0$ 修的是「点估计在 bootstrap 分布里偏不居中」。它默认中位数偏差和总体偏差同方向，用 $\Phi^{-1}$ 把比例转成标准正态刻度。加速因子 $a$ 修的是「尺度随参数变」，理论上来自 $\sigma_\theta/(d\theta/d\mu)$ 这类导数比，刀切法是它的通用数值近似。两个修正各有明确靶子，所以 BCa 是「有条件地更好」，不是「无条件更高级」。

还有一条常被忽略的对照路线：bootstrap-t（studentized bootstrap）。它不挪分位数，而是给每次重抽算一个 t 统计量再取分位数，理论上能达到二阶精度，在厚尾场景下往往比 BCa 更稳。代价是每次重抽都要估标准误，n 小时极不稳定，计算量也大。本文没有把它纳入对拍，是因为它引入了「标准误怎么估」这个新的混杂变量，会模糊「端点算法本身」的比较。真要追求精度，bootstrap-t 值得单独验证。

## 十、怎么选

别把 BCa 当默认升级。真正值得上 BCa 的场景是：统计量明显偏斜（方差、相关系数、比率）、样本量至少三四十、真值不在参数空间边界上。这时它能把 percentile 的单边漏失拉回来几个百分点，代价只是多算一遍刀切。

正态均值、对称统计量、小样本、极端比例这四类场景，BCa 给不了承诺——甚至会像正态均值那样小输一场。诚实的做法是先画出 bootstrap 分布的偏度和 $z_0$，再决定要不要为这层校正付钱。

代码在下方，从重抽、刀切到五张图全部现场计算，固定种子 20261123，改参数即可复现。

```python
# -*- coding: utf-8 -*-
"""BCa bootstrap confidence intervals: when the correction earns its keep.

Single source of truth for every number and figure in the article.

Four intervals are built from the SAME B bootstrap replicates so the comparison
is paired: normal approximation, percentile, basic (reverse-percentile) and
Efron's BCa (bias-corrected and accelerated, 1987). BCa fixes two things the
plain percentile interval gets wrong: median bias (through z0, the fraction of
bootstrap replicates below the point estimate) and the fact that the sampling
distribution can skew/scale with the parameter (through the jackknife
acceleration a).

    z0 = Phi^{-1}( #{theta*_b < theta_hat} / B )
    a  = sum_i (theta_bar - theta_(i))^3 / (6 [sum_i (theta_bar - theta_(i))^2]^{3/2})
    alpha1 = Phi( z0 + (z0+z_alpha) / (1 - a (z0+z_alpha)) )
    alpha2 = Phi( z0 + (z0+z_{1-alpha}) / (1 - a (z0+z_{1-alpha})) )
    CI_BCa = [ theta*_(alpha1), theta*_(alpha2) ]

We test on truth-known scenes: normal mean, exponential mean, lognormal mean,
normal variance (skewed statistic), and bivariate correlation (bounded,
nonlinear). Then we look under the hood at one sample, at the acceleration
distribution, at the small-n / boundary failure of z0, and at sample-size
sweeps. All scenes share fixed seeds so exec-replay matches results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
B = 2000                 # bootstrap replicates per sample
ZLO, ZHI = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
Z95 = 1.959963984540054

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "bca-bootstrap-confidence-interval")
os.makedirs(_OUT, exist_ok=True)


# ---------------------------------------------------------------- estimators
def jackknife(x, theta_fn):
    n = len(x)
    return np.array([theta_fn(np.delete(x, i, axis=0)) for i in range(n)])


def bca_endpoints(tb, theta_hat, x, theta_fn):
    """Return (lo, hi, z0, a, alpha1, alpha2) from a bootstrap vector tb."""
    b = len(tb)
    tj = jackknife(x, theta_fn)
    u = tj.mean() - tj
    s2 = float(np.sum(u ** 2))
    a = float(np.sum(u ** 3) / (6.0 * s2 ** 1.5)) if s2 > 0 else 0.0
    prop = float(np.mean(tb < theta_hat))
    z0 = float(stats.norm.ppf(min(max(prop, 1.0 / b), 1.0 - 1.0 / b)))

    def adj(z):
        den = 1.0 - a * (z0 + z)
        if abs(den) < 1e-12:
            return np.nan
        return float(stats.norm.cdf(z0 + (z0 + z) / den))

    a1, a2 = adj(ZLO), adj(ZHI)
    lo = float(np.quantile(tb, a1)) if np.isfinite(a1) else np.nan
    hi = float(np.quantile(tb, a2)) if np.isfinite(a2) else np.nan
    return lo, hi, z0, a, a1, a2


def four_intervals(tb, theta_hat):
    """Paired normal / percentile / basic intervals plus percentile cutoffs."""
    se = float(np.std(tb, ddof=1))
    p_lo, p_hi = np.quantile(tb, [0.025, 0.975])
    return {
        "normal": (theta_hat - Z95 * se, theta_hat + Z95 * se),
        "percentile": (float(p_lo), float(p_hi)),
        "basic": (2 * theta_hat - float(p_hi), 2 * theta_hat - float(p_lo)),
    }


def corr_rows(xb):
    """Batch Pearson correlation for xb (B, n, 2)."""
    a = xb[:, :, 0]
    b = xb[:, :, 1]
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(ac ** 2, axis=1) * np.sum(bc ** 2, axis=1))
    return np.where(den > 0, np.sum(ac * bc, axis=1) / np.where(den > 0, den, 1.0), 0.0)


METHODS = ["normal", "percentile", "basic", "BCa"]


def coverage_scene(kind, n, true, B, R, seed):
    """Monte Carlo coverage of the four intervals on a truth-known scene."""
    rng = np.random.default_rng(seed)
    two_d = kind == "corr"

    def sample():
        if kind == "normal_mean":
            return rng.standard_normal(n)
        if kind == "exp_mean":
            return rng.exponential(1.0, n)
        if kind == "lognormal_mean":
            return rng.lognormal(0.0, 0.5, n)
        if kind == "normal_var":
            return rng.standard_normal(n)
        if kind == "corr":
            rho = 0.6
            cov = np.array([[1.0, rho], [rho, 1.0]])
            return rng.multivariate_normal([0.0, 0.0], cov, n)
        raise ValueError(kind)

    if kind == "normal_var":
        def theta_fn(x):
            return float(np.var(x, ddof=1))
        def theta_rows(xb):
            return np.var(xb, axis=1, ddof=1)
    elif two_d:
        def theta_fn(x):
            a, b = x[:, 0], x[:, 1]
            return float(np.corrcoef(a, b)[0, 1])
        def theta_rows(xb):
            return corr_rows(xb)
    else:
        def theta_fn(x):
            return float(np.mean(x))
        def theta_rows(xb):
            return xb.mean(axis=1)

    hit = {m: 0 for m in METHODS}
    left = {m: 0 for m in METHODS}
    right = {m: 0 for m in METHODS}
    width = {m: [] for m in METHODS}
    z0s, as_ = [], []
    for _ in range(R):
        x = sample()
        th = theta_fn(x)
        idx = rng.integers(0, n, (B, n))
        xb = x[idx]
        tb = theta_rows(xb)
        ints = four_intervals(tb, th)
        lo_c, hi_c, z0, a, _, _ = bca_endpoints(tb, th, x, theta_fn)
        ints["BCa"] = (lo_c, hi_c)
        z0s.append(z0)
        as_.append(a)
        for m in METHODS:
            lo, hi = ints[m]
            if np.isfinite(lo) and np.isfinite(hi):
                hit[m] += int(lo <= true <= hi)
                left[m] += int(hi < true)
                right[m] += int(lo > true)
                width[m].append(hi - lo)
    return {
        "kind": kind, "n": n, "B": B, "R": R, "true": float(true), "seed": seed,
        "cover": {m: hit[m] / R for m in METHODS},
        "left_miss": {m: left[m] / R for m in METHODS},
        "right_miss": {m: right[m] / R for m in METHODS},
        "width": {m: float(np.mean(width[m])) for m in METHODS},
        "z0_mean": float(np.mean(z0s)), "a_mean": float(np.mean(as_)),
        "a_sd": float(np.std(as_)),
    }


def exp_coverage():
    scenes = [
        coverage_scene("normal_mean", 20, 0.0, B, 1500, 1001),
        coverage_scene("exp_mean", 20, 1.0, B, 1500, 1002),
        coverage_scene("lognormal_mean", 20, float(np.exp(0.125)), B, 1500, 1003),
        coverage_scene("normal_var", 20, 1.0, B, 1500, 1004),
        coverage_scene("corr", 30, 0.6, 1200, 900, 1005),
    ]
    return scenes


def exp_mechanics():
    """One exponential sample: show the bootstrap distribution and both cuts."""
    rng = np.random.default_rng(2001)
    n = 20
    x = rng.exponential(1.0, n)
    theta_hat = float(np.mean(x))
    tb = x[rng.integers(0, n, (B, n))].mean(axis=1)
    ints = four_intervals(tb, theta_hat)
    lo_c, hi_c, z0, a, a1, a2 = bca_endpoints(tb, theta_hat, x,
                                              lambda z: float(np.mean(z)))
    return {
        "n": n, "theta_hat": theta_hat, "true": 1.0,
        "boot_skew": float(stats.skew(tb)),
        "z0": z0, "a": a, "alpha1": a1, "alpha2": a2,
        "normal": list(ints["normal"]),
        "percentile": list(ints["percentile"]),
        "basic": list(ints["basic"]),
        "bca": [lo_c, hi_c],
        "tb": tb.tolist(),
        "x": x.tolist(),
    }


def exp_acceleration():
    """Distribution of the jackknife acceleration a on a skewed statistic."""
    rng = np.random.default_rng(3001)
    n = 20
    a_all = []
    a_zero_cover, a_full_cover = 0, 0
    R = 800
    for _ in range(R):
        x = rng.exponential(1.0, n)
        th = float(np.mean(x))
        tb = x[rng.integers(0, n, (B, n))].mean(axis=1)
        lo_c, hi_c, z0, a, _, _ = bca_endpoints(tb, th, x,
                                                lambda z: float(np.mean(z)))
        a_all.append(a)
        # "z0 only" = set acceleration to zero but keep the same z0.
        def adj_zero(z):
            return float(stats.norm.cdf(z0 + (z0 + z)))
        lo_z = float(np.quantile(tb, adj_zero(ZLO)))
        hi_z = float(np.quantile(tb, adj_zero(ZHI)))
        a_zero_cover += int(lo_z <= 1.0 <= hi_z)
        a_full_cover += int(lo_c <= 1.0 <= hi_c)
    return {
        "n": n, "R": R, "a_mean": float(np.mean(a_all)),
        "a_sd": float(np.std(a_all)),
        "a_q05": float(np.quantile(a_all, 0.05)),
        "a_q95": float(np.quantile(a_all, 0.95)),
        "cover_z0_only": a_zero_cover / R,
        "cover_bca_full": a_full_cover / R,
        "a_all": a_all,
    }


def exp_boundary():
    """z0 saturation when the bootstrap replicate mass piles on one side."""
    out = {}
    for p in [0.02, 0.05, 0.10, 0.30, 0.50]:
        rng = np.random.default_rng(4000 + int(p * 100))
        n = 20
        R = 800
        hit_p, hit_c = 0, 0
        sat = 0
        widths = []
        for _ in range(R):
            x = (rng.random(n) < p).astype(float)
            th = float(np.mean(x))
            if th == 0.0:
                # no successes in the sample: every interval degenerates at 0
                hit_p += int(0.0 <= p <= 0.0)
                hit_c += int(0.0 <= p <= 0.0)
                sat += 1
                widths.append(0.0)
                continue
            tb = x[rng.integers(0, n, (B, n))].mean(axis=1)
            p_lo, p_hi = np.quantile(tb, [0.025, 0.975])
            hit_p += int(p_lo <= p <= p_hi)
            lo_c, hi_c, z0, a, _, _ = bca_endpoints(tb, th, x,
                                                    lambda z: float(np.mean(z)))
            hit_c += int(lo_c <= p <= hi_c)
            sat += int(abs(z0) >= stats.norm.ppf(1 - 1.0 / B) - 1e-9)
            widths.append(hi_c - lo_c)
        out[str(p)] = {
            "p": p, "n": n, "R": R,
            "cover_percentile": hit_p / R,
            "cover_bca": hit_c / R,
            "z0_sat_frac": sat / R,
            "bca_width": float(np.mean(widths)),
        }
    return out


def exp_sample_size():
    """Coverage vs n on the skewed exponential-mean scene."""
    out = {}
    for n in [5, 10, 20, 40, 80]:
        sc = coverage_scene("exp_mean", n, 1.0, 1200, 900, 5000 + n)
        out[str(n)] = {"n": n, "cover": sc["cover"], "width": sc["width"]}
    return out


# ------------------------------------------------------------------- figures
def fig1_coverage(scenes):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    names = [s["kind"] for s in scenes]
    x = np.arange(len(names))
    w = 0.19
    colors = {"normal": "#9aa6b2", "percentile": "#5b8ff9",
              "basic": "#61ddaa", "BCa": "#b23a2b"}
    for j, m in enumerate(METHODS):
        vals = [s["cover"][m] for s in scenes]
        ax.bar(x + (j - 1.5) * w, vals, w, label=m, color=colors[m])
    ax.axhline(0.95, color="#333", ls="--", lw=1.4, label="nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("coverage of true value")
    ax.set_title("Paired bootstrap intervals, B=2000: BCa wins only where bias+skew bite")
    ax.legend(ncol=5, fontsize=9, loc="lower right")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_coverage.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_coverage.png"


def fig2_tail_balance(scenes):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = [s["kind"] for s in scenes]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for m, color in [("left_miss", "#5b8ff9"), ("right_miss", "#b23a2b")]:
        vals = [s[m]["percentile"] for s in scenes]
        ax.bar(x, vals, 0.55, bottom=bottom, color=color, alpha=0.85,
               label=m.replace("_", " ") + " (percentile)")
        bottom += np.array(vals)
    for m, key, mk in [("BCa left", "left_miss", "o"), ("BCa right", "right_miss", "s")]:
        vals = [s[key]["BCa"] for s in scenes]
        ax.plot(x, vals, mk + "k", ms=5, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("fraction of misses")
    ax.set_title("Percentile interval misses asymmetrically; BCa rebalances the tails")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_tail_balance.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_tail_balance.png"


def fig3_mechanics(m):
    tb = np.array(m["tb"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(tb, bins=60, color="#c9d8ee", edgecolor="none", density=True)
    ax.axvline(m["true"], color="#2e7d55", lw=2, label="truth = 1.0")
    ax.axvline(m["theta_hat"], color="#333", lw=1.6, ls="--", label="theta_hat")
    p_lo, p_hi = m["percentile"]
    b_lo, b_hi = m["bca"]
    ax.axvspan(p_lo, p_hi, color="#5b8ff9", alpha=0.18, label="percentile CI")
    ax.axvspan(b_lo, b_hi, color="#b23a2b", alpha=0.16, label="BCa CI")
    ax.set_xlabel("bootstrap mean")
    ax.set_ylabel("density")
    ax.set_title("One sample: z0=%.3f, a=%.3f; percentile [%.3f,%.3f] vs BCa [%.3f,%.3f]"
                 % (m["z0"], m["a"], p_lo, p_hi, b_lo, b_hi))
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_mechanics.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_mechanics.png"


def fig4_acceleration(acc):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.hist(acc["a_all"], bins=45, color="#f2c4b4", edgecolor="none", density=True)
    ax.axvline(0, color="#333", ls="--", lw=1.3)
    ax.axvline(acc["a_mean"], color="#b23a2b", lw=2,
               label="mean a=%.3f" % acc["a_mean"])
    ax.set_xlabel("jackknife acceleration a")
    ax.set_ylabel("density")
    ax.set_title("Acceleration is positive on a right-skewed statistic: "
                 "z0-only cover %.3f -> BCa cover %.3f"
                 % (acc["cover_z0_only"], acc["cover_bca_full"]))
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_acceleration.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_acceleration.png"


def fig5_boundary(bd, ss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ps = sorted(float(k) for k in bd)
    a0 = ax[0]
    a0.plot(ps, [bd[str(p)]["cover_percentile"] for p in ps], "s-",
            color="#5b8ff9", label="percentile")
    a0.plot(ps, [bd[str(p)]["cover_bca"] for p in ps], "o-",
            color="#b23a2b", label="BCa")
    a0.plot(ps, [bd[str(p)]["z0_sat_frac"] for p in ps], "^--",
            color="#888", label="z0 saturated")
    a0.axhline(0.95, color="#333", ls="--", lw=1.2)
    a0.set_xlabel("true binomial p (n=20)")
    a0.set_ylabel("fraction")
    a0.set_title("z0 saturates at the boundary: p near 0 or 1")
    a0.legend(fontsize=8)
    ns = sorted(int(k) for k in ss)
    a1 = ax[1]
    for m, color in [("normal", "#9aa6b2"), ("percentile", "#5b8ff9"),
                     ("basic", "#61ddaa"), ("BCa", "#b23a2b")]:
        a1.plot(ns, [ss[str(n)]["cover"][m] for n in ns], "o-", color=color, label=m)
    a1.axhline(0.95, color="#333", ls="--", lw=1.2)
    a1.set_xlabel("sample size n (exponential mean)")
    a1.set_ylabel("coverage")
    a1.set_title("Small n: neither percentile nor BCa reaches nominal")
    a1.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_boundary.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_boundary.png"


def main():
    scenes = exp_coverage()
    mech = exp_mechanics()
    acc = exp_acceleration()
    bd = exp_boundary()
    ss = exp_sample_size()
    r = {
        "setup": {"seed": SEED, "B": B, "methods": METHODS,
                  "alpha": 0.05, "corr_rho": 0.6},
        "coverage": scenes,
        "mechanics": {k: v for k, v in mech.items() if k not in ("tb", "x")},
        "acceleration": {k: v for k, v in acc.items() if k != "a_all"},
        "boundary": bd,
        "sample_size": ss,
    }
    r["figs"] = [fig1_coverage(scenes), fig2_tail_balance(scenes),
                 fig3_mechanics(mech), fig4_acceleration(acc),
                 fig5_boundary(bd, ss)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

```
