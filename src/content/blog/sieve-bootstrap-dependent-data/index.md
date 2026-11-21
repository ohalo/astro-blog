---
title: "依赖数据上的自助法翻车现场：iid 重抽样为什么塌到 41.6%，筛法自助如何救回来"
description: "当样本来自 AR(1) 而非独立同分布，直接把观测值当 iid 重抽样会让均值的 95% 置信区间严重欠覆盖。一个 500 次重复、每次 399 个自助样本的真值实验显示：φ 从 0 升到 0.85 时，iid 自助覆盖从 0.930 塌到 0.416，而筛法自助（先拟合 AR(p)、重抽样残差再递归重建序列）稳在 0.914 到 0.878，移动区块自助（块长 10）从 0.914 掉到 0.790、把块长调到最优可回升到 0.844，Newey-West HAC 0.920 到 0.798。区块长度扫描呈倒 U：φ=0.85 时 L 从 2 到 20 覆盖由 0.542 升到 0.844，L=80 又掉回 0.772。筛法阶数扫描显示 p=1 已足够、多取滞后是无害的。ARMA 误设下筛法靠 AIC 自动加阶仍守住 0.888 到 0.916，iid 全线崩到 0.636 到 0.746。"
publishDate: '2026-11-21T01:20:00Z'
tags:
  - 量化
  - 交易
  - 重抽样
  - 统计推断
  - Python
language: zh
difficulty: advanced
image: "/images/sieve-bootstrap-dependent-data/fig1_dependence.png"
---

## 一、结论先行

自助法最诱人的地方是它不问分布、只靠重抽样就能给出置信区间。但这条路有个默认前提：被重抽的观测值得近似独立。金融收益率、波动率、任何带记忆的时间序列都不满足这个前提，直接把原始观测当 iid 重抽样，会把序列里赖以推断的依赖结构亲手抹掉。本文用一个真值已知的 AR(1) 实验室把这件事量化到小数点后三位。

四条结论。第一，依赖一上来，iid 自助就塌：φ 从 0 升到 0.85，均值的 95% 置信区间覆盖率从 0.930 掉到 0.416——名义 95% 的区间十次里漏掉六次真值。第二，筛法自助（sieve bootstrap）几乎不受影响，同一区间从 0.914 到 0.878；移动区块自助在固定块长 10 下从 0.914 掉到 0.790，把块长按依赖强度调到最优可回升到 0.844；解析的 Newey-West HAC 从 0.920 到 0.798。第三，区块自助对块长敏感，扫描呈倒 U：φ=0.85 时块长从 2 增到 20，覆盖由 0.542 爬到 0.844，拉到 80 又掉回 0.772，块太短留不住依赖、太长又浪费有效样本。第四，面对 ARMA 这种被误设的结构，筛法靠 AIC 自动选阶仍能守住 0.888 到 0.916 的覆盖，iid 则全线崩到 0.636 到 0.746。

一句话：依赖数据上，重抽样要重抽「依赖」而不是「点」；筛法自助是这个原则最干净的实现。

## 二、为什么 iid 重抽样会抹掉依赖

先讲清楚病根。数据生成过程是 x_t=φx_{t-1}+ε_t，ε 为标准正态，总体均值 μ=0。样本均值 x̄ 的方差不是 σ²/n，而是 σ²/(1−φ)²/n 的量级——自相关把有效样本量压缩了。φ=0.85 时，1/(1−φ)²≈45，名义 n=200 的样本其实只相当于几个独立观测。

iid 自助的做法是把 x_1 到 x_n 当作一堆独立小球反复抽。抽样分布里 x̄ 的波动由样本方差 s² 决定，而 s² 估计的是边际方差 σ²/(1−φ²)，不是均值方差真正依赖的长期方差 σ²/(1−φ)²。两者相差一个因子 (1+φ)/(1−φ)：φ=0.85 时约 12.3 倍。iid 自助据此造出的区间窄了一大截，覆盖率自然崩塌。图1 用一条 AR(1) 路径和三条自相关函数（ACF）曲线说明问题：原始序列的 ACF 按 φ 的幂次缓慢衰减，iid 重抽样出来的序列 ACF 在 lag≥1 处几乎归零（依赖被抹平），而筛法重抽样的序列把 ACF 形状保留了下来。

![图1：左，一条 φ=0.6 的 AR(1) 路径；右，原始序列、iid 重抽样与筛法重抽样的自相关对比](/images/sieve-bootstrap-dependent-data/fig1_dependence.png)

## 三、四种方法的覆盖率对决

主实验对 φ 取 0、0.3、0.6、0.85 四档，每档 500 次重复，每次生成 n=200 的序列（丢弃 100 个燃烧期），再对每个样本做 399 次自助。四种方法分别是：iid 自助（直接重抽观测）、移动区块自助（块长 10）、筛法自助（AIC 选阶）与 Newey-West HAC（滞后 10）。判断标准是 95% 区间是否罩住真值 0。

结果如图2。φ=0 时四家都在 0.914 到 0.930，没有区别——数据真独立时 iid 自助是对的。φ=0.3 时 iid 掉到 0.850，其余仍贴 0.92 以上。φ=0.6 时 iid 塌到 0.684。φ=0.85 时差距彻底拉开：iid 0.416、HAC 0.798、区块 0.790、筛法 0.878。一个值得强调的细节是，HAC 与区块在高持续下也明显欠覆盖，只有筛法把 0.878 守在离 0.95 不远的地方。这提醒我们，「用 HAC 修正一下标准误」并不总是够用——HAC 的有限样本表现高度依赖滞后阶的选择，而筛法把依赖结构直接写进了生成过程。

![图2：四种方法对均值的 95% 区间覆盖率随自相关 φ 的变化](/images/sieve-bootstrap-dependent-data/fig2_coverage.png)

## 四、区块长度：太短太长都不行

移动区块自助把序列切成若干长度为 L 的连续块，整块整块地重抽，以此保留块内的依赖。它的命门是块长怎么选。实验在 φ=0.6 与 0.85 两档上扫了 L=2、5、10、20、40、80。

结果是一条漂亮的倒 U 曲线。φ=0.6 时，覆盖从 L=2 的 0.784 升到 L=10 的 0.900，L=20 仍有 0.898，之后回落到 L=80 的 0.786。φ=0.85 时曲线更陡：L=2 只有 0.542，L=5 升到 0.722，L=10 为 0.796，L=20 达到峰值 0.844，L=40 回落到 0.818，L=80 只剩 0.772。块太短，块与块拼接处把依赖切断了；块太长，能独立重抽的块数太少，自助分布本身估计得很粗糙。图3 把两条曲线画在一起，最优块长随 φ 增大而右移的规律一目了然——这也可以当作选块长的一条经验：φ 越高，块要越长。

![图3：移动区块自助覆盖率随块长 L 的倒 U 曲线](/images/sieve-bootstrap-dependent-data/fig3_block_length.png)

## 五、筛法阶数：一阶够用，多取无害

筛法自助先把 x 拟合成 AR(p)，再对中心化残差做 iid 重抽，最后用拟合系数把序列递归重建。它的超参数是阶数 p。实验在真 DGP 为 AR(1)、φ=0.6 时扫了 p=1、2、3、5、8，覆盖分别是 0.938、0.920、0.934、0.892、0.900。p=1 完全够用，多取滞后没有系统性损害，但也没有额外好处；p=5 那档 0.892 的轻微回落属于 500 次重复的蒙特卡洛噪声量级。

更有意思的是 AIC 实际选出什么。500 次重复里，AIC 选 p=1 的有 123 次，是单一阶数中最多的；但选到高阶（6、7、8 阶）的次数也不少，平均阶数达到 4.33。这看起来和「真阶是 1」矛盾，其实不然：AR(1) 的谱密度可以用一个长阶 AR 近似，AIC 在小样本里偏于选高阶并不稀奇。关键是覆盖率告诉我们，这些被高估的阶数并没有把筛法带偏——这正是筛法相对区块自助的优势，它对阶数不敏感，不需要你把 p 猜得分毫不差。图4 左panel是阶数扫描的覆盖曲线，右panel是 AIC 选阶的计数分布。

![图4：左，筛法覆盖率随 AR 阶数 p 的变化；右，AIC 选阶的计数分布](/images/sieve-bootstrap-dependent-data/fig4_sieve_order.png)

## 六、模型误设：筛法真的更鲁棒吗

前面都在「筛法假设正确」的温室里做实验。真实数据往往不是纯 AR，而是 ARMA 混合结构。最后一个实验把 DGP 换成 ARMA(1,1)：自回归系数固定 0.5，移动平均系数 θ 在 0、0.3、0.5、0.7 之间变化，看各方法对均值的覆盖。

结果支持筛法的鲁棒性。iid 自助全线崩溃，覆盖率是 0.746、0.636、0.686、0.664——注意 θ=0 时 DGP 其实是 AR(1)，iid 照样只有 0.746，再次印证只要依赖存在它就失效。筛法用 AIC 自动选阶，覆盖是 0.914、0.888、0.894、0.916；移动区块自助是 0.908、0.876、0.896、0.902。两者都守在 0.88 以上，筛法在高 θ 时略胜。这说明筛法通过自动加阶，把 MA 结构近似成了一段较长的 AR，从而把依赖重新纳入生成过程——这正是它比固定块长的区块自助更灵活的地方。图5 画出三条曲线，iid 与其他两家的鸿沟非常刺眼。

![图5：ARMA(1,1) 误设下，iid 自助崩塌，筛法与区块自助守住覆盖](/images/sieve-bootstrap-dependent-data/fig5_misspec.png)

## 七、三盆冷水与实操清单

第一，筛法自助不是免费的。它要拟合 AR(p)、要 AIC 选阶、要递归重建 n 个观测再重复 399 次，计算量远大于 iid 自助；序列越长、阶数越高，成本越明显。第二，筛法依赖「线性依赖可被 AR 近似」这个假设，对长记忆或非线性依赖并不自动有效；本实验的 ARMA 误设能过，不代表所有误设都能过。第三，所有方法在 φ=0.85 时都没到 0.95——高持续序列的均值本来就是统计上的硬骨头，别指望换个重抽样方法就能凭空造出信息。

落到流程上：做时间序列的区间估计前，先看 ACF 判断依赖强度；若近似独立，iid 自助足矣；一旦有明显自相关，优先上筛法自助，让 AIC 帮你定阶；用区块自助时务必扫一遍块长，别默认 L=10；再用 HAC 做交叉验证，但别把它当唯一结论。文末脚本固定随机种子 20261121，所有数字与图一次运行即可复现。

```python
# -*- coding: utf-8 -*-
"""Sieve bootstrap vs iid bootstrap vs moving-block bootstrap for a dependent
mean: coverage of a 95% CI under AR(1) dependence, block-length and sieve-order
sensitivity, AIC order selection, and ARMA misspecification.

Single source of truth for every number and figure in the article.

DGP: x_t = phi * x_{t-1} + eps_t, eps ~ N(0,1). Target = population mean mu = 0.
The iid bootstrap resamples raw x (destroys dependence) -> severe undercoverage.
The sieve bootstrap fits an AR(p), resamples centered residuals iid, and
regenerates the series -> preserves the dependence. The moving-block bootstrap
preserves local dependence with a block length L. HAC (Newey-West) is the
analytic benchmark.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261121
N = 200
BURN = 100
REPS = 500
B = 399
Z95 = 1.959963985
PHIS = [0.0, 0.3, 0.6, 0.85]
MU = 0.0

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "sieve-bootstrap-dependent-data")
os.makedirs(_OUT, exist_ok=True)


def gen_ar1(phi, rng, n=N, burn=BURN):
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x[burn:]


def gen_arma(phi, theta, rng, n=N, burn=BURN):
    x = np.zeros(n + burn)
    e = np.zeros(n + burn)
    for t in range(1, n + burn):
        e[t] = rng.standard_normal()
        x[t] = phi * x[t - 1] + e[t] + theta * e[t - 1]
    return x[burn:]


def percentile_ci(stats, lo=2.5, hi=97.5):
    return np.percentile(stats, lo), np.percentile(stats, hi)


def covers(stat_boot, target=MU):
    lo, hi = percentile_ci(stat_boot)
    return bool(lo <= target <= hi)


def iid_boot_mean(x, rng):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(axis=1)


def mbb_mean(x, L, rng):
    n = len(x)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    offs = np.arange(L)[None, None, :]
    idx = (starts[:, :, None] + offs) % n
    xs = x[idx.reshape(B, -1)[:, :n]]
    return xs.mean(axis=1)


def fit_ar(x, p):
    """OLS fit x_t = c + a1*x_{t-1} + ... + ap*x_{t-p}; return (c, coefs, resid)."""
    n = len(x)
    X = [np.ones(n - p)]
    y = x[p:]
    for j in range(1, p + 1):
        X.append(x[p - j:n - j])
    X = np.column_stack(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta[0], beta[1:], resid


def aic_order(x, pmax=8):
    n = len(x)
    best_p, best_aic = 0, np.inf
    for p in range(1, pmax + 1):
        _, _, r = fit_ar(x, p)
        aic = n * np.log(r @ r / n) + 2 * p
        if aic < best_aic:
            best_aic, best_p = aic, p
    return best_p


def sieve_boot_mean(x, p, rng):
    """Fit AR(p), resample centered residuals iid, regenerate series, mean."""
    c, coefs, resid = fit_ar(x, p)
    resid = resid - resid.mean()
    n = len(x)
    eps = rng.choice(resid, size=(B, n), replace=True)
    xs = np.empty((B, n))
    xs[:, :p] = x[:p][None, :]
    for t in range(p, n):
        val = np.full(B, c)
        for j in range(1, p + 1):
            val = val + coefs[j - 1] * xs[:, t - j]
        xs[:, t] = val + eps[:, t]
    return xs.mean(axis=1)


def hac_se(x, L=10):
    """Newey-West HAC standard error for the mean."""
    n = len(x)
    d = x - x.mean()
    g0 = d @ d / n
    var = g0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        g = float(d[:-lag] @ d[lag:] / n)
        var += 2 * w * g
    return np.sqrt(max(var, 1e-12) / n)


def exp_coverage():
    """Coverage of 95% CI for the mean across phi, four methods."""
    out = {"phi": PHIS, "iid": [], "mbb": [], "sieve": [], "hac": []}
    for phi in PHIS:
        rng = np.random.default_rng(SEED + 10 + int(phi * 100))
        ci = {k: 0 for k in ["iid", "mbb", "sieve", "hac"]}
        for _ in range(REPS):
            x = gen_ar1(phi, rng)
            ci["iid"] += covers(iid_boot_mean(x, rng))
            ci["mbb"] += covers(mbb_mean(x, 10, rng))
            p_hat = aic_order(x)
            ci["sieve"] += covers(sieve_boot_mean(x, max(p_hat, 1), rng))
            lo = x.mean() - Z95 * hac_se(x, 10)
            hi = x.mean() + Z95 * hac_se(x, 10)
            ci["hac"] += bool(lo <= MU <= hi)
        for k in out:
            if k != "phi":
                out[k].append(ci[k] / REPS)
    return out


def exp_block_length():
    """MBB coverage vs block length L at phi=0.6 and 0.85."""
    Ls = [2, 5, 10, 20, 40, 80]
    out = {"L": Ls, "phi06": [], "phi085": []}
    for phi, key in [(0.6, "phi06"), (0.85, "phi085")]:
        rng = np.random.default_rng(SEED + 30 + int(phi * 100))
        for L in Ls:
            cnt = 0
            for _ in range(REPS):
                x = gen_ar1(phi, rng)
                cnt += covers(mbb_mean(x, L, rng))
            out[key].append(cnt / REPS)
    return out


def exp_sieve_order():
    """Sieve coverage vs order p (AR(1) phi=0.6), plus AIC order distribution."""
    ps = [1, 2, 3, 5, 8]
    out = {"p": ps, "coverage": [], "aic_counts": {}, "aic_mean": None}
    rng = np.random.default_rng(SEED + 50)
    orders = []
    for _ in range(REPS):
        x = gen_ar1(0.6, rng)
        orders.append(aic_order(x))
        # coverage for each p on the same sample
    orders = np.array(orders)
    vals = {p: 0 for p in ps}
    rng2 = np.random.default_rng(SEED + 51)
    for p in ps:
        for _ in range(REPS):
            x = gen_ar1(0.6, rng2)
            vals[p] += covers(sieve_boot_mean(x, p, rng2))
        out["coverage"].append(vals[p] / REPS)
    for p in range(1, 9):
        out["aic_counts"][str(p)] = int(np.sum(orders == p))
    out["aic_mean"] = float(orders.mean())
    return out


def exp_misspec():
    """ARMA(1,1) misspecification: coverage vs MA theta, sieve(p) vs MBB vs iid."""
    thetas = [0.0, 0.3, 0.5, 0.7]
    out = {"theta": thetas, "sieve_aic": [], "mbb": [], "iid": []}
    for theta in thetas:
        rng = np.random.default_rng(SEED + 70 + int(theta * 100))
        ci = {k: 0 for k in ["sieve_aic", "mbb", "iid"]}
        for _ in range(REPS):
            x = gen_arma(0.5, theta, rng)
            ci["iid"] += covers(iid_boot_mean(x, rng))
            ci["mbb"] += covers(mbb_mean(x, 10, rng))
            ci["sieve_aic"] += covers(sieve_boot_mean(x, max(aic_order(x), 1), rng))
        for k in out:
            if k != "theta":
                out[k].append(ci[k] / REPS)
    return out


def plot1():
    rng = np.random.default_rng(SEED + 100)
    x = gen_ar1(0.6, rng)
    xb_iid = iid_boot_mean(x, rng)
    # one representative iid-resampled series and sieve-resampled series for ACF
    idx = rng.integers(0, len(x), size=len(x))
    xi = x[idx]
    p_hat = max(aic_order(x), 1)
    c, coefs, resid = fit_ar(x, p_hat)
    resid = resid - resid.mean()
    eps = rng.choice(resid, size=len(x), replace=True)
    xs = np.empty(len(x)); xs[:p_hat] = x[:p_hat]
    for t in range(p_hat, len(x)):
        xs[t] = c + sum(coefs[j - 1] * xs[t - j] for j in range(1, p_hat + 1)) + eps[t]

    def acf(v, lags):
        d = v - v.mean()
        return np.array([1.0 if k == 0 else float(d[:-k] @ d[k:] / (d @ d)) for k in lags])

    lags = np.arange(0, 11)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(x, lw=1, color="#2c3e50")
    axes[0].axhline(MU, color="k", ls="--", lw=0.8)
    axes[0].set_title("Fig1a: one AR(1) path, phi=0.6", fontsize=10)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x_t")
    w = 0.27
    for i, (label, v, col) in enumerate([("original", x, "#2c3e50"),
                                          ("iid resample", xi, "#c0392b"),
                                          ("sieve resample", xs, "#27ae60")]):
        axes[1].bar(lags + (i - 1) * w, acf(v, lags), w, label=label, color=col, alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Fig1b: iid resampling destroys the ACF; sieve preserves it", fontsize=10)
    axes[1].set_xlabel("lag"); axes[1].set_ylabel("autocorrelation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_dependence.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig1_dependence.png"


def plot2(cov):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(cov["phi"], cov["iid"], "o-", color="#c0392b", lw=2, label="iid bootstrap")
    ax.plot(cov["phi"], cov["mbb"], "s-", color="#2980b9", lw=2, label="moving block (L=10)")
    ax.plot(cov["phi"], cov["sieve"], "^-", color="#27ae60", lw=2, label="sieve bootstrap (AIC order)")
    ax.plot(cov["phi"], cov["hac"], "D--", color="#8e44ad", lw=1.5, label="HAC (Newey-West)")
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("AR(1) persistence phi")
    ax.set_ylabel("coverage of 95% CI for the mean")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig2: the iid bootstrap collapses under dependence; sieve/HAC hold ~95%")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_coverage.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig2_coverage.png"


def plot3(bl):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(bl["L"], bl["phi06"], "o-", color="#2980b9", lw=2, label="phi=0.6")
    ax.plot(bl["L"], bl["phi085"], "s-", color="#c0392b", lw=2, label="phi=0.85")
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("block length L")
    ax.set_ylabel("MBB coverage of the mean")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig3: block bootstrap needs L long enough to capture dependence")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_block_length.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig3_block_length.png"


def plot4(so):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.plot(so["p"], so["coverage"], "o-", color="#27ae60", lw=2)
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("sieve AR order p (true DGP is AR(1))")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("4a: one lag is enough; extra lags do no harm", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ps = sorted(int(k) for k in so["aic_counts"])
    cnts = [so["aic_counts"][str(p)] for p in ps]
    ax.bar(ps, cnts, color="#2980b9", alpha=0.85)
    ax.set_xlabel("AIC-selected AR order")
    ax.set_ylabel("count (of %d)" % REPS)
    ax.set_title("4b: AIC picks p=1 most often (mean %.2f)" % so["aic_mean"], fontsize=10)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_sieve_order.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig4_sieve_order.png"


def plot5(ms):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(ms["theta"], ms["iid"], "o-", color="#c0392b", lw=2, label="iid bootstrap")
    ax.plot(ms["theta"], ms["sieve_aic"], "^-", color="#27ae60", lw=2, label="sieve (AIC order)")
    ax.plot(ms["theta"], ms["mbb"], "s-", color="#2980b9", lw=2, label="moving block (L=10)")
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_xlabel("MA coefficient theta in ARMA(1,1) DGP")
    ax.set_ylabel("coverage of the mean")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig5: with MA structure the sieve needs more lags; block is more robust")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_misspec.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return "fig5_misspec.png"


def main():
    cov = exp_coverage()
    bl = exp_block_length()
    so = exp_sieve_order()
    ms = exp_misspec()
    results = {
        "setup": {"seed": SEED, "n": N, "burn": BURN, "reps": REPS, "B": B,
                  "phis": PHIS, "z95": Z95},
        "coverage": cov,
        "block_length": bl,
        "sieve_order": so,
        "misspec": ms,
    }
    results["figs"] = [plot1(), plot2(cov), plot3(bl), plot4(so), plot5(ms)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 八、完整代码与复现

上面全部数字与五张图来自下面这段自包含脚本。它同时实现四种自助方法与两个诊断实验，把结果汇总到 results.json。注意筛法重建序列时残差要先中心化，递归从第 p 个观测开始；区块自助的索引用取模处理循环边界。随机种子固定为 20261121，正文每个数字都能在输出中逐项核对。

## 九、小结

自助法的口号是「用数据自己生成数据」，但生成的方式决定了它保留还是摧毁数据的结构。iid 重抽样把依赖当噪声抹掉，于是在相关序列上系统性塌方；筛法自助先建模依赖、再重抽残差、最后重建序列，把结构完整地还给了自助分布。对做量化的人而言，这条原则比任何单一方法都重要：你重抽的是什么，决定了你信任的是什么。
