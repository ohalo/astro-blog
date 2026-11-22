---
title: "经验似然置信区间：不估方差也能做推断，但 Wilks 定理在小样本和偏斜下会收多少税"
description: "经验似然用数据自己长出的权重替代方差估计，Wilks 定理说 -2log 似然比服从卡方(1)，95% 区间只要卡方阈值 3.841。本文在真值已知的正态、移位指数与 t(5) 分布上做 320 次重复实验：正态 n=30 经验似然覆盖 0.9219、Wald 0.9313、Bootstrap 0.9250，宽度 0.6766 对 0.7042；偏斜 n=30 经验似然宽度反而涨到 0.8448（Wald 0.6852），覆盖率却只有 0.9125；Wilks 统计量的 95% 分位在偏斜 n=30 高达 5.2317，名义误报率 0.0967 而非 0.05。但它也不是全输：偏斜样本右臂/左臂长度比 1.3485，Wald 恒为 1；真均值右移 0.30 时经验似然功效 0.9594，Wald 只有 0.8875，左移 0.30 时则回落到 0.6969 输给 Wald 的 0.7406。附完整可运行代码与五张真实计算图。"
publishDate: '2026-11-22T11:10:00Z'
tags:
  - 量化
  - 统计推断
  - 经验似然
  - 置信区间
  - Python
language: zh
difficulty: advanced
image: "/images/empirical-likelihood-wilks-inference/fig1_el_shape.png"
---

## 一、结论先行

给一个均值做置信区间，教科书路径是先估标准误，再套正态或 t 分位数。这条路要回答一个经常被跳过的问题：标准误本身怎么估？分布偏斜、厚尾、样本只有三四十个点时， plug-in 的标准误会先知先觉地把区间摆到一个并不对称的位置。

经验似然（Empirical Likelihood, EL）走另一条路。它不给每个观测预设正态形状，而是让数据自己决定权重：在满足候选均值 $\mu$ 的约束下，最大化 $\prod_i p_i$。得到的似然比 $-\,2\log R(\mu)$ 由 Wilks 定理渐近服从 $\chi^2(1)$，于是 95% 区域只要一个固定阈值 3.841——不需要方差估计，也不需要预先指定分布族。

这套优雅的交换在有限样本里要付税。真值已知的蒙特卡洛（320 次重复、固定种子 20261122）给出四句话。第一，正态样本上 EL 区间更窄（n=30 时宽 0.6766，Wald 0.7042），但覆盖只有 0.9219，比 Wald 的 0.9313 更低：省下的宽度是用覆盖度换的。第二，偏斜样本上 EL 不再省宽度，n=30 时宽 0.8448，比 Wald 的 0.6852 多 23%，覆盖仍只有 0.9125。第三，Wilks 校准在偏斜小样本明显失效，$\ell(\mu_0)$ 的 95% 分位冲到 5.2317，名义 5% 检验实际误报 0.0967。第四，EL 换来的好处是形状自适应：偏斜 n=60 时右臂长度是左臂的 1.3485 倍（Wald 恒为 1），真均值右移 0.30 时功效 0.9594 对 Wald 的 0.8875；但左移 0.30 时它只有 0.6969，反而输给 Wald 的 0.7406。所有数字由文末代码现场计算，可逐项复现。

## 二、把似然比写成约束优化

设样本 $x_1,\dots,x_n$，候选均值 $\mu$。经验似然在单纯形上最大化乘积概率：

$$
\max_{p_1,\dots,p_n}\ \prod_{i=1}^n p_i,\quad \text{s.t.}\ \sum_i p_i=1,\ \ \sum_i p_i x_i=\mu
$$

拉格朗日一阶条件给出 $p_i = 1/\bigl(n[1+\lambda(x_i-\mu)]\bigr)$，其中乘子 $\lambda$ 由方程 $\sum_i (x_i-\mu)/[1+\lambda(x_i-\mu)]=0$ 唯一确定。把它代回，并与无约束最优 $p_i=1/n$ 相比：

$$
\ell(\mu) = -2\log \frac{R(\mu)}{R(\hat\mu)} = 2\sum_{i=1}^n \log\bigl(1+\lambda(x_i-\mu)\bigr)
$$

$\ell(\hat\mu)=0$，离样本均值越远越大。Wilks 定理保证在真值处 $\ell(\mu_0)\Rightarrow\chi^2(1)$，因此 $\ell(\mu)\le 3.841$ 就是渐近 95% 置信域。实现上把 $\lambda$ 的根锁在区间 $(-1/\max(x_i-\mu),\ -1/\min(x_i-\mu))$ 内，用带括号保护的牛顿迭代求解，再在样本均值附近的自适应网格上找 $\ell$ 穿过 3.841 的两个端点。

![图1：三种分布下经验似然比曲线的形状与卡方阈值](/images/empirical-likelihood-wilks-inference/fig1_el_shape.png)

图1 用 n=60 的单个样本展示形状。正态样本上 EL 端点 -0.0410 到 0.4185，Wald 是 -0.0426 到 0.4139，两者接近。移位指数样本上 EL 端点 -0.2431 到 0.2439，右臂 0.2764 对左臂 0.2106，明显向右张开；Wald 则机械地给出 -0.2744 到 0.2093 的对称区间。双峰样本上 EL 曲线同样是一个碗，说明均值这一矩条件本身不保证产生分离的置信域——那种更有趣的现象需要非参数矩条件。

## 三、Wilks 校准的账面数字和实际数字

理论说 $\ell(\mu_0)$ 的平均是 1，95% 分位是 3.841。实际呢？每种设定抽 600 个样本，在真值处计算 $\ell$。

![图2：经验似然统计量的经验分位数对卡方分位数](/images/empirical-likelihood-wilks-inference/fig2_wilks_qq.png)

正态 n=120 的平均 $\ell$ 是 1.0184，95% 分位 3.9890，误报 0.0533，几乎完美。正态 n=30 平均升到 1.1451，95% 分位 4.0525，误报 0.0633——已经比名义值高出三分之一。偏斜是最贵的：移位指数 n=30 平均 1.2779，95% 分位 5.2317，误报 0.0967；n=120 降到 1.0462、3.8990、0.0567。t(5) 厚尾反而温和，n=30 平均 1.1305、误报 0.0633，n=120 误报 0.0517。

这组数字给出一个实用的校正直觉：如果愿意把阈值按 $E[\ell]/1$ 放大，偏斜 n=30 要用 4.9 而不是 3.84。但这里的 $E[\ell]$ 是实验里算出来的，不是从数据里免费得到的；真要用就得做 bootstrap 校准，代价是每重抽一次都要求一次乘子方程。这正是经验似然 theory 漂亮、落地要算力的地方。

## 四、覆盖率：更窄不等于更准

把 EL、Wald 和百分位 bootstrap 放在同一批样本上比较，样本量取 30、60、120、240。

![图3：三种置信区间在正态、偏斜与厚尾分布上的实际覆盖率](/images/empirical-likelihood-wilks-inference/fig3_coverage.png)

正态 n=30：EL 0.9219，Wald 0.9313，bootstrap 0.9250；n=240：EL 0.9375，Wald 0.9563，bootstrap 0.9469。EL 在所有样本量上都比 Wald 低约 1.5 到 2 个百分点。宽度解释了这个差额：n=30 时 EL 平均宽 0.6766，Wald 0.7042，bootstrap 0.6741；EL 用更窄的区间拿到了更低的覆盖，属于典型的过度自信。

偏斜情形更刺眼。n=30 时 EL 宽 0.8448，Wald 只有 0.6852，bootstrap 0.6554，但三者覆盖分别是 0.9125、0.9125、0.9031：EL 多付 23% 的宽度，一分覆盖优势都没拿到。直到 n=60，EL 宽度降到 0.4838 略小于 Wald 的 0.4964，覆盖 0.9500 才追平 bootstrap 的 0.9563 并接近 Wald 的 0.9625。厚尾 t(5) 下 EL 也没有惊喜，n=30 覆盖 0.9250 对 Wald 0.9438，n=240 是 0.9406 对 0.9625。

结论要说透：Wilks 阈值不是免费午餐，它把方差估计的不确定性从区间宽度里挪走了，却没有凭空消灭。样本小、分布偏时，这部分不确定性会以覆盖不足的形式回来。

## 五、它真正值钱的地方是形状

区间宽度和覆盖讲的是平均表现。EL 的设计初衷是让数据决定形状，这一点在偏斜样本上可以量化。

![图4：偏斜样本下经验似然区间的左右臂长度与bootstrap对比](/images/empirical-likelihood-wilks-inference/fig4_asymmetry.png)

移位指数分布的真实均值是 0，右尾更长，所以样本的合理区间应该向右多伸一点。n=60 时 EL 左臂平均 0.2071、右臂 0.2796，比值 1.3485；bootstrap 比值 1.1152，Wald 恒等于 1。n=120 时 EL 比值 1.2724，bootstrap 1.1049。n=30 时两者分别是 1.1458 与 1.1358，差距不大——点数太少，任何方法都难以稳定识别偏斜方向。

不对称带来的直接后果是功效的方向性。样本量 100、真均值平移 $\delta$：

![图5：正态与偏斜样本下三种区间的检验功效曲线](/images/empirical-likelihood-wilks-inference/fig5_power.png)

正态样本上，$\delta=-0.30$ 时 EL 功效 0.8656，Wald 与 bootstrap 都是 0.8406；$\delta=+0.30$ 时 EL 0.8281 对 Wald 0.8063。偏斜样本上差距被拉开：$\delta=+0.30$ 时 EL 0.9594、Wald 0.8875、bootstrap 0.9188；$\delta=+0.20$ 时是 0.6500 对 0.5156。但换到左尾，$\delta=-0.30$ 时 EL 掉到 0.6969，Wald 0.7406、bootstrap 0.7531。

这不是缺点，而是权衡被显式化了：EL 把拒绝域按数据形状重新分配，右尾更敏感、左尾更迟钝。做策略研究时如果你只关心收益分布向上偏移的能力，这种分配可能正合心意；如果你要双向严谨的假设检验，就得为左尾多留点预算。

## 六、怎么用在量化研究里

三条可执行的建议。第一，EL 适合样本短、分布明显偏斜且你愿意接受一点覆盖折价的场景，例如事件窗口内的异常收益均值；它给出的非对称区间比对称 Wald 更贴近收益的真实形状。第二，不要把它当成厚尾万灵药：t(5) 上 EL 的覆盖没有超过 Wald，尾部风险更该交给专门方法。第三，报告时同时给出经验分位校准，或者明说用的是渐近 Wilks 阈值——第三节显示同一个方法在偏斜 n=30 上的真实误报接近 10%，这个信息不该被 95% 三个字盖掉。

## 七、把经验似然放进研究流程

实操时我把它当作三层工具，而不是一个替代 Wald 的按钮。第一层是诊断：拿到一批因子收益，先在候选均值上画出 $\ell(\mu)$ 曲线，看它的最低点、曲率和左右不对称程度。曲线本身比一个区间端点包含更多信息——它告诉你数据愿意为哪个均值付出多大代价，也告诉你代价是不是对称上升的。图1 里正态分布是标准抛物线，偏斜分布左右斜率不同，这种视觉诊断比一个数字更能提示后续建模方向。

第二层是校准。如果样本量小于 100，或者收益分布明显偏斜，就别直接用 3.841。做法是从经验分布重抽 B 次，在每个重抽样本上算 $\ell^*(\hat\mu)$，取其 95% 分位替代卡方阈值。代价是每重抽一次都要解一次乘子方程，代码里的求解器已经向量化，几百次重抽在普通笔记本上仍在秒级。这一步把偏斜 n=30 的名义误报从 0.0967 拉回来的思路，与第三节的实测分位数是同一回事，只是校准值来自当前样本而不是实验设定。

第三层是报告口径。给出区间时同时标注用的是渐近 Wilks 阈值还是校准阈值，并把左右端点分别列出——对称区间只有一个半宽，非对称区间必须写两个数。做策略复盘时还要记住功效的方向性：图5 显示偏斜样本向右偏移时 EL 更敏感，向左偏移时更迟钝，把这种不对称当成免费的精度提升会摔跟头。

代码从 $p_i$ 的乘子方程一路写到功效曲线，所有随机数来自固定种子，重跑即得文中数字。需要调的是样本量与分布设定；求解器本身对分布形式没有假设，这正是经验似然的价值所在。

```python
# -*- coding: utf-8 -*-
"""Empirical likelihood for a mean: Wilks calibration, coverage and power.

Single source of truth for every number and figure in the article.

The empirical likelihood (EL) ratio for a candidate mean mu is
    R(mu) = max prod_i p_i  s.t. sum p_i = 1, sum p_i x_i = mu,
whose Lagrange solution satisfies sum (x_i-mu)/(1+lam*(x_i-mu)) = 0.
Wilks' theorem gives  w(mu) = -2 log R(mu) -> chi2(1) at the true mu,
so a 95% region needs no variance estimate: {mu : w(mu) <= chi2_1(0.95)}.

We compare EL with the Wald t-interval and the percentile bootstrap on
three truth-known distributions (normal, shifted exponential = skew,
standardized t(5) = heavy tail), four sample sizes, and a power experiment.
All randomness uses one fixed seed; every number printed is recomputed.
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

SEED = 20261122
CONF = 0.95
Q = float(stats.chi2.ppf(CONF, 1))
Z = float(stats.norm.ppf(0.5 + CONF / 2.0))
R = 320            # Monte Carlo replications per cell
G = 61             # mu grid points for interval extraction
B_BOOT = 199       # bootstrap resamples per replication
NS = [30, 60, 120, 240]
SIZES = {"normal": 1.0, "skew": 1.0, "t5": 1.0}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "empirical-likelihood-wilks-inference")
os.makedirs(_OUT, exist_ok=True)


# ------------------------------------------------------------------ EL solver
def solve_lambda(d):
    """Solve sum(d/(1+lam*d)) = 0 with 1+lam*d > 0, vectorised on leading axes."""
    d = np.asarray(d, dtype=float)
    mx = np.max(d, axis=-1)
    mn = np.min(d, axis=-1)
    lo = -1.0 / np.where(mx > 0, mx, np.inf)
    hi = -1.0 / np.where(mn < 0, mn, -np.inf)
    lam = np.zeros_like(lo)
    for _ in range(32):
        z = 1.0 + lam[..., None] * d
        g = np.sum(d / z, axis=-1)
        gp = -np.sum((d / z) ** 2, axis=-1)
        mid = 0.5 * (lo + hi)
        cand = lam - g / gp
        bad = (~np.isfinite(cand)) | (cand <= lo) | (cand >= hi)
        nxt = np.where(bad, mid, cand)
        lo = np.where(g > 0, lam, lo)
        hi = np.where(g <= 0, lam, hi)
        eps = 1e-12 * (hi - lo)
        lam = np.minimum(np.maximum(nxt, lo + eps), hi - eps)
    return lam


def el_stat(x, mu):
    """w(mu)=-2 sum log(1+lam*(x-mu)); x (...,n), mu (...,) or scalar."""
    x = np.asarray(x, dtype=float)
    d = x - np.asarray(mu)[..., None]
    lam = solve_lambda(d)
    # R(mu)/R(muhat)=prod 1/(1+lam*d); AM-GM => prod(1+lam*d) >= 1, so w >= 0
    return 2.0 * np.sum(np.log1p(lam[..., None] * d), axis=-1)


def el_interval(x, conf=CONF, span=5.0):
    """Interval from a sample-specific mu grid; returns (lo, hi, width, missing)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    muhat = x.mean(axis=-1)
    se = x.std(axis=-1, ddof=1) / np.sqrt(n)
    grid = muhat[..., None] + np.linspace(-span, span, G)[None, :] * se[..., None]
    w = el_stat(x[:, None, :], grid)
    ok = w <= Q
    idx = np.argmin(w, axis=-1)
    lo_i = np.full(x.shape[:-1], -1, dtype=int)
    hi_i = np.full(x.shape[:-1], -1, dtype=int)
    for i in range(len(lo_i)):
        good = np.flatnonzero(ok[i])
        if good.size:
            lo_i[i] = good[0]
            hi_i[i] = good[-1]
    missing = int(np.sum(lo_i < 0))
    lo = np.where(lo_i >= 0, np.take_along_axis(grid, lo_i[..., None], -1)[..., 0], np.nan)
    hi = np.where(hi_i >= 0, np.take_along_axis(grid, hi_i[..., None], -1)[..., 0], np.nan)
    return lo, hi, hi - lo, missing


def boot_interval(x, conf=CONF):
    """Percentile bootstrap interval for the mean, x shape (R,n)."""
    rng = np.random.default_rng(SEED + 77)
    lo, hi = [], []
    for start in range(0, len(x), 80):
        chunk = x[start:start + 80]
        idx = rng.integers(0, chunk.shape[1], size=(len(chunk), B_BOOT, chunk.shape[1]))
        bm = chunk[np.arange(len(chunk))[:, None, None], idx].mean(axis=-1)
        a = (1 - conf) / 2
        lo.append(np.quantile(bm, a, axis=-1))
        hi.append(np.quantile(bm, 1 - a, axis=-1))
    lo = np.concatenate(lo)
    return lo, np.concatenate(hi)


def sample(scenario, n, rng):
    if scenario == "normal":
        return rng.standard_normal(n)
    if scenario == "skew":
        return rng.exponential(1.0, n) - 1.0   # mean 0, right-skew
    if scenario == "t5":
        z = rng.standard_t(5, size=n)
        return z / np.sqrt(5.0 / 3.0)          # mean 0, unit variance
    raise ValueError(scenario)


def truth(scenario):
    return 0.0


# ------------------------------------------------------------------ experiments
def exp_wilks():
    """Distribution of w(mu_true) against chi2(1): size and Bartlett factor."""
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for scenario in ["normal", "skew", "t5"]:
        for n in [30, 120]:
            x = np.stack([sample(scenario, n, rng) for _ in range(600)])
            w = el_stat(x, np.full(len(x), truth(scenario)))
            q95 = float(np.quantile(w, 0.95))
            out[f"{scenario}_n{n}"] = {
                "e_w": round(float(w.mean()), 6),
                "bartlett_factor": round(float(w.mean() / Q), 6),
                "q50": round(float(np.quantile(w, 0.50)), 6),
                "q90": round(float(np.quantile(w, 0.90)), 6),
                "q95": round(q95, 6),
                "q99": round(float(np.quantile(w, 0.99)), 6),
                "size_el": round(float(np.mean(w > Q)), 6),
            }
    return {"chi2_1_q95": round(Q, 6), "cells": out, "chi2_ref": {
        "q50": round(float(stats.chi2.ppf(0.5, 1)), 6),
        "q90": round(float(stats.chi2.ppf(0.9, 1)), 6),
        "q95": round(Q, 6),
        "q99": round(float(stats.chi2.ppf(0.99, 1)), 6),
    }}


def exp_coverage():
    rng = np.random.default_rng(SEED + 2)
    out = {}
    for scenario in ["normal", "skew", "t5"]:
        for n in NS:
            x = np.stack([sample(scenario, n, rng) for _ in range(R)])
            mu = truth(scenario)
            elo, ehi, ew, missing = el_interval(x)
            se = x.std(axis=-1, ddof=1) / np.sqrt(n)
            m = x.mean(axis=-1)
            wlo, whi = m - Z * se, m + Z * se
            blo, bhi = boot_interval(x)
            cov_el = np.mean((elo <= mu) & (mu <= ehi))
            cov_w = np.mean((wlo <= mu) & (mu <= whi))
            cov_b = np.mean((blo <= mu) & (mu <= bhi))
            out[f"{scenario}_n{n}"] = {
                "el": round(float(cov_el), 6),
                "wald": round(float(cov_w), 6),
                "boot": round(float(cov_b), 6),
                "el_width": round(float(np.nanmean(ew)), 6),
                "wald_width": round(float(np.mean(whi - wlo)), 6),
                "boot_width": round(float(np.mean(bhi - blo)), 6),
                "el_missing_grid": missing,
            }
    return out


def exp_shape():
    """One representative sample per distribution: the EL ratio curve is data-shaped."""
    rng = np.random.default_rng(SEED + 3)
    out = {}
    for scenario in ["normal", "skew", "bimodal"]:
        n = 60
        if scenario == "bimodal":
            x = np.concatenate([rng.normal(-1.2, 0.45, n // 2), rng.normal(1.2, 0.45, n // 2)])
        else:
            x = sample(scenario, n, rng)
        muhat = float(x.mean())
        se = float(x.std(ddof=1) / np.sqrt(n))
        grid = np.linspace(muhat - 3.2 * se, muhat + 3.2 * se, 241)
        w = el_stat(np.broadcast_to(x, (len(grid), n)), grid)
        inside = w <= Q
        if np.any(inside):
            gi = np.flatnonzero(inside)
            lo, hi = float(grid[gi[0]]), float(grid[gi[-1]])
        else:
            lo, hi = float("nan"), float("nan")
        out[scenario] = {
            "mean": round(muhat, 6), "se": round(se, 6),
            "el_lo": round(lo, 6), "el_hi": round(hi, 6),
            "wald_lo": round(muhat - Z * se, 6), "wald_hi": round(muhat + Z * se, 6),
            "left_len": round(muhat - lo, 6), "right_len": round(hi - muhat, 6),
            "grid": [round(float(v), 8) for v in grid[::6]],
            "w": [round(float(v), 8) for v in w[::6]],
        }
    return out


def exp_asymmetry():
    """Average left/right arm lengths for the right-skewed distribution."""
    rng = np.random.default_rng(SEED + 4)
    from scipy.stats import norm
    out = {}
    for n in [30, 60, 120]:
        x = np.stack([sample("skew", n, rng) for _ in range(R)])
        lo, hi, wd, missing = el_interval(x)
        m = x.mean(axis=-1)
        se = x.std(axis=-1, ddof=1) / np.sqrt(n)
        blo, bhi = boot_interval(x)
        ratio_el = float(np.nanmean((hi - m) / (m - lo)))
        ratio_b = float(np.nanmean((bhi - m) / (m - blo)))
        out[f"n{n}"] = {
            "el_left": round(float(np.nanmean(m - lo)), 6),
            "el_right": round(float(np.nanmean(hi - m)), 6),
            "el_right_left_ratio": round(ratio_el, 6),
            "boot_right_left_ratio": round(ratio_b, 6),
            "wald_symmetric": True,
            "el_missing_grid": missing,
        }
    return out


def exp_power():
    rng = np.random.default_rng(SEED + 5)
    deltas = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]
    out = {}
    for scenario in ["normal", "skew"]:
        rows = {}
        for d in deltas:
            x = np.stack([sample(scenario, 100, rng) + d for _ in range(R)])
            elo, ehi, _, _ = el_interval(x)
            m = x.mean(axis=-1)
            se = x.std(axis=-1, ddof=1) / np.sqrt(100)
            wlo, whi = m - Z * se, m + Z * se
            blo, bhi = boot_interval(x)
            rows[f"{d:+.2f}"] = {
                "el": round(float(np.nanmean((elo > 0) | (ehi < 0))), 6),
                "wald": round(float(np.mean((wlo > 0) | (whi < 0))), 6),
                "boot": round(float(np.mean((blo > 0) | (bhi < 0))), 6),
            }
        out[scenario] = rows
    return {"delta": deltas, "cells": out}


# ------------------------------------------------------------------ figures
def plot_shape(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, (name, label) in zip(ax, [("normal", "Normal"), ("skew", "Shifted exponential"),
                                     ("bimodal", "Bimodal mixture")]):
        d = e[name]
        a.plot(d["grid"], d["w"], color="#b23a2b", lw=2)
        a.axhline(Q, color="#2b5d8a", ls="--", lw=1.5, label=f"chi2(1) 95% = {Q:.3f}")
        a.axvline(0, color="0.6", lw=1)
        a.set_title(label + ": data-shaped EL ratio")
        a.set_xlabel("candidate mean mu"); a.set_ylabel("w(mu)")
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_el_shape.png"), dpi=115)
    plt.close(fig); return "fig1_el_shape.png"


def plot_wilks(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ref = e["chi2_ref"]
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    keys = [("normal_n30", "Normal n=30"), ("skew_n120", "Skew n=120"), ("t5_n120", "t(5) n=120")]
    for a, (k, label) in zip(ax, keys):
        c = e["cells"][k]
        obs = [c["q50"], c["q90"], c["q95"], c["q99"]]
        theo = [ref["q50"], ref["q90"], ref["q95"], ref["q99"]]
        a.plot(theo, theo, "k--", lw=1)
        a.plot(theo, obs, "o-", color="#b23a2b")
        a.set_title("%s\nE[w]=%.2f, factor=%.2f, size=%.3f" %
                    (label, c["e_w"], c["bartlett_factor"], c["size_el"]))
        a.set_xlabel("chi2(1) quantile"); a.set_ylabel("empirical w quantile")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_wilks_qq.png"), dpi=115)
    plt.close(fig); return "fig2_wilks_qq.png"


def plot_coverage(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, (sc, lab) in zip(ax, [("normal", "Normal"), ("skew", "Skew"), ("t5", "t(5)")]):
        for meth, col, mk in [("el", "#b23a2b", "o"), ("wald", "#2b5d8a", "s"),
                              ("boot", "#2e7d55", "^")]:
            ys = [e[f"{sc}_n{n}"][meth] for n in NS]
            a.plot(NS, ys, marker=mk, color=col, label=meth)
        a.axhline(CONF, color="0.4", ls=":")
        a.set_ylim(0.72, 1.01); a.set_title(lab + " coverage")
        a.set_xlabel("n"); a.set_ylabel("95% CI coverage"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_coverage.png"), dpi=115)
    plt.close(fig); return "fig3_coverage.png"


def plot_asym(e):
    ns = [30, 60, 120]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(ns, [e[f"n{n}"]["el_left"] for n in ns], "o-", color="#2b5d8a", label="EL left arm")
    ax[0].plot(ns, [e[f"n{n}"]["el_right"] for n in ns], "o-", color="#b23a2b", label="EL right arm")
    ax[0].set_title("Skew: EL interval is asymmetric"); ax[0].set_xlabel("n")
    ax[0].set_ylabel("mean arm length"); ax[0].legend()
    ax[1].plot(ns, [e[f"n{n}"]["el_right_left_ratio"] for n in ns], "o-",
               color="#b23a2b", label="EL")
    ax[1].plot(ns, [e[f"n{n}"]["boot_right_left_ratio"] for n in ns], "s-",
               color="#2e7d55", label="bootstrap")
    ax[1].axhline(1.0, color="0.4", ls="--", label="Wald (=1)")
    ax[1].set_title("Right / left arm length"); ax[1].set_xlabel("n")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_asymmetry.png"), dpi=115)
    plt.close(fig); return "fig4_asymmetry.png"


def plot_power(e):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ds = e["delta"]
    for a, (sc, lab) in zip(ax, [("normal", "Normal"), ("skew", "Skew")]):
        for meth, col in [("el", "#b23a2b"), ("wald", "#2b5d8a"), ("boot", "#2e7d55")]:
            a.plot(ds, [e["cells"][sc][f"{d:+.2f}"][meth] for d in ds],
                   marker="o", color=col, label=meth)
        a.axhline(0.05, color="0.4", ls=":")
        a.set_title(lab + " power, n=100"); a.set_xlabel("true shift")
        a.set_ylabel("reject H0: mu=0"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_power.png"), dpi=115)
    plt.close(fig); return "fig5_power.png"


def main():
    r = {"setup": {"seed": SEED, "conf": CONF, "chi2_q95": round(Q, 6),
                   "R": R, "G": G, "B_boot": B_BOOT, "n": NS},
         "wilks": exp_wilks(), "shape": exp_shape(),
         "coverage": exp_coverage(), "asymmetry": exp_asymmetry(),
         "power": exp_power()}
    r["figs"] = [plot_shape(r["shape"]), plot_wilks(r["wilks"]),
                 plot_coverage(r["coverage"]), plot_asym(r["asymmetry"]),
                 plot_power(r["power"])]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

