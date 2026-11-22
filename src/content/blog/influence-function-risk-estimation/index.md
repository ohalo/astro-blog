---
title: "影响函数与刀切法：一个极端交易日如何撬动你的 VaR，以及分位数误差棒为什么会骗你"
description: "用一个真值已知的实验把风险估计的稳健性拆开看：影响函数（IF）描述单个观测对估计量的边际影响。均值的影响函数无界、随损失线性增长，中位数与 VaR 的影响函数有界、被按在分位数附近，ES 的影响函数在左尾以 1/q=20 的斜率发散。往 200 天样本里塞一个 -20 的极端损失，均值被拉动 -0.1003、中位数纹丝不动 -0.0020、ES 暴跌 -1.6242。刀切法对光滑泛函（均值、ES）给出的标准误与影响函数渐近值吻合到 0.1%，但对分位数严重失真：正态样本下 VaR 的刀切标准误只有真值的 49%、中位数只有 30%，t(3) 厚尾下反而放大到 188% 与 271%。用影响函数标准误构造 95% 置信区间，均值与中位数覆盖率贴着 95%，n=50 时 ES 只有 69.9%、VaR 85.6%，厚尾下 ES 更低到 64.4%。结论：尾部风险估计的标准误不能随手用刀切法，分位数尤其要命。"
publishDate: '2026-11-22T01:10:00Z'
tags:
  - 量化
  - 风险管理
  - 稳健统计
  - 影响函数
  - Python
language: zh
difficulty: advanced
image: "/images/influence-function-risk-estimation/fig2_single_outlier.png"
---

## 一、结论先行

风控报表上每个数字都带一个误差棒，但误差棒本身也可能是错的。本文讨论一个被长期忽略的问题：当收益分布有极端点时，风险度量的标准误该怎么算，以及我们常用的刀切法（jackknife）会不会给出虚假的精度。

用一个真值已知的实验把事情拆开。核心工具是影响函数（Influence Function, IF）——它衡量把一个观测点 infinitesimal 地掺进分布时，某个估计量会朝哪个方向动、动多快。四个常见风险度量给出了截然不同的答案。均值的影响函数无界，跟着损失线性增长；中位数与 VaR 的影响函数有界，被死死按在分位数附近；ES（预期短缺）的影响函数在左尾以 $1/q$ 的斜率发散——在 5% 的分位水平上，这个斜率恰好是 20。

由此得到五个结论。第一，单个极端点对不同度量的破坏力天差地别。第二，刀切法对光滑泛函很准，对分位数却系统性失真。第三，用影响函数构造的置信区间，覆盖率在均值上很稳，在尾部风险上小样本严重不足。第四，厚尾会把 ES 的渐近方差放大到均值的上百倍。第五，1% 的污染就足以让 ES 偏掉 1.65 个标准差，而中位数几乎无感。所有数字来自同一段脚本，附在文末，可逐项复现。

## 二、影响函数：给估计量做一次压力测试

把经验分布 $F_n$ 想象成一个装满观测的罐子。现在往里掺一点点污染物——在某个点 $x$ 上加一丁点权重 $\varepsilon$，其余点相应减少。估计量 $T(F_n)$ 会怎么变？把变化量除以 $\varepsilon$ 并令 $\varepsilon \to 0$，得到的就是影响函数 $\mathrm{IF}(x)$。它是估计量对单个观测的敏感度，也是渐近方差的钥匙：任意估计量的渐近方差都等于 $\mathbb{E}[\mathrm{IF}^2]/n$。

四个风险度量的影响函数有清晰的解析形式。设 $q=0.05$ 为左尾水平，$\xi = F^{-1}(q)$ 为分位数，$\mathrm{ES}_q = \mathbb{E}[X\mid X\le \xi]$：

$$
\mathrm{IF}_{\text{mean}}(x) = x - \mu
$$

$$
\mathrm{IF}_{\text{VaR}}(x) = \frac{q - \mathbf{1}\{x\le\xi\}}{f(\xi)}
$$

$$
\mathrm{IF}_{\text{ES}}(x) = (\xi - \mathrm{ES}_q) + \frac{1}{q}\,\mathbf{1}\{x\le\xi\}\,(x-\xi)
$$

中位数是 VaR 在 $q=0.5$ 的特例。关键在最后一行：当 $x$ 落进左尾（$x\le\xi$），影响函数含一项 $(x-\xi)/q$，随着 $x$ 越负而无界下降。VaR 的影响函数则是一个有界的阶梯——不管 $x$ 多极端，它最多把概率质量从 $q$ 改成 $q-1$，除以密度就是个固定跳幅。这一正一负的差别，是后文所有反直觉现象的源头。

先用有限差分验证公式。取 20 万个正态样本，把其中 0.1% 的点替换成探针值 $x_0$，看估计量实际变化除以污染比例。结果：ES 影响函数的实测斜率是 20.000，与理论 $1/q = 20$ 逐位吻合；均值、VaR、ES 三条解析曲线与有限差分的最大绝对偏差分别只有 0.131、0.305、0.395，相对 20 量级的斜率可以忽略。地基是稳的。

![图1：均值、VaR、ES 的影响函数形状——有限差分与解析公式对拍](/images/influence-function-risk-estimation/fig1_if_shape.png)

图1 从左到右分别是均值、VaR、ES。均值是一条直线，越极端影响越大；VaR 是两级台阶，探针对它要么没影响要么是个固定跳幅；ES 在阈值左侧是陡峭的斜线，斜率正是 20。三条曲线里只有 ES 会「记住」损失到底有多深。

## 三、一个极端交易日能撬动什么

把抽象的影响函数变成看得见的冲击。取 200 天的标准正态收益样本，往里追加一个极端损失 $x_0$，从 -2 一路扫到 -20，看四个估计量各被拉动多少。

![图2：往 200 天样本里注入单个极端损失，四个风险度量的反应](/images/influence-function-risk-estimation/fig2_single_outlier.png)

均值被拉动的幅度是 $(x_0-\bar{x})/201$，从 $x_0=-2$ 时的 -0.0107 线性增长到 $x_0=-20$ 时的 -0.1003——分布每多一个爆炸日，日均收益就被拖走一截。中位数从头到尾只动了 -0.0020，完全不受影响：一个点在 201 个样本里翻不起浪，只要它没跨过中位数的位置。VaR 也几乎不动（全程 -0.0010），同样是分位数的有界性在起作用。真正崩掉的是 ES：从 -2 时的 +0.0122 一路跌到 -20 时的 -1.6242，而且斜率恒定——每多亏一个单位，ES 就被多拖 0.0895 个单位。

这就是 ES 的诅咒。它被定义为尾部损失的平均值，天然把最坏的那些点纳入计算，于是每个极端日都能直接改写它。用 ES 做风控限额，等于把系统暴露在单个黑天鹅的直接影响之下；用中位数或 VaR，则对这种单点冲击免疫，但代价是对尾部深度视而不见。

## 四、刀切法的标准误：光滑处很准，分位数处翻车

标准误最直观的估法是刀切法：依次删掉每个观测，重算 $n$ 次估计量，再用这 $n$ 个值的离散度推出标准误。它对光滑泛函有理论保证，但对分位数呢？本文把刀切法、影响函数渐近公式、自助法三种标准误放在一起对拍。

![图3：影响函数、刀切法、自助法三种标准误的对比](/images/influence-function-risk-estimation/fig3_se_methods.png)

先看好消息。对均值和 ES 这两个光滑泛函，三种方法高度一致：正态样本下均值的三种标准误是 0.04499、0.04504、0.04505，ES 是 0.11627、0.11632、0.11687，刀切与影响函数的比值都是 1.000。厚尾 t(3) 下同样吻合——ES 的三种值 0.37609、0.37593、0.37064。刀切法在这些地方是可信的。

再看坏消息。分位数彻底翻车。正态样本下，VaR 的影响函数标准误是 0.06837，刀切法只给出 0.03376，只有真值的 49%；中位数更夸张，真值 0.06323，刀切 0.01913，只有 30%。厚尾 t(3) 下方向反过来：VaR 的刀切标准误 0.40722 是渐近值 0.21701 的 188%，中位数 0.09802 是 0.03616 的 271%。自助法始终贴着影响函数，说明问题出在刀切法本身。

原因并不神秘。分位数是次序统计量，删掉一个点往往只让估计值在相邻两个次序值之间跳一格，这 $n$ 个「留一估计」的分布根本不代表抽样分布——它们挤在极窄的几个离散值上（正态情形系统性低估），而在厚尾里又被少数极端次序值拉出长尾（系统性高估）。分位数的刀切方差估计在理论上就不一致，这不是样本量不够能救回来的。

## 五、置信区间覆盖率：尾部风险的小样本困境

既然影响函数给出了渐近标准误，用它构造 95% 正态近似置信区间，覆盖率有多靠谱？这里用真值已知的分布做 1600 次重复，样本量取 50、200、1000。

![图4：基于影响函数标准误的 95% 置信区间覆盖率](/images/influence-function-risk-estimation/fig4_coverage.png)

均值与中位数在所有设定下都贴着 95%：正态 n=1000 时分别是 95.5% 与 95.6%，厚尾 t(3) 下是 95.0% 与 94.7%。分位数与尾部风险就没这么体面了。正态 n=50 时 VaR 只有 85.6%、ES 只有 69.9%；样本涨到 1000，VaR 回到 94.8%、ES 回到 93.2%。厚尾 t(3) 更糟，n=50 时 ES 覆盖率塌到 64.4%，n=1000 也只有 89.6%。

实践含义很直接：如果你用 1000 天数据估一个 VaR 或 ES，再报一个基于渐近正态的 95% 误差棒，这个区间的真实覆盖水平其实是 90% 上下，比声称的乐观。样本越短、尾部越厚，水分越大。尾部风险的区间估计应该优先考虑自助法或专门的极值理论方法，而不是拿一个渐近公式直接套。

## 六、厚尾与污染：ES 的方差为什么爆炸

最后一个维度是分布本身的尾部厚度。让 t 分布的自由度从 2.5 变到 30，看 ES 与均值的渐近方差因子 $\mathbb{E}[\mathrm{IF}^2]$ 怎么变。

![图5：厚尾下 ES 的渐近方差相对均值的倍数](/images/influence-function-risk-estimation/fig5_tail.png)

自由度 30 时，ES 的渐近方差因子 10.52，均值 1.281，ES 是均值的 8.2 倍。自由度降到 3，ES 飙到 671.79，均值只有 14.77，倍数 45.5。自由度 2.5（方差已接近发散）时，ES 高达 3159.44，是均值的 127.3 倍。尾部越厚，ES 的标准误相对均值越发不可收拾——这正是它把极端点全部纳入的代价。

污染实验把这点钉死。往 1000 个正态样本里掺 -10 的极端值，污染比例 1% 时，均值偏 -0.100、中位数只偏 -0.005、VaR 偏 -0.028、ES 却偏了 -1.654。比例升到 5%，ES 偏 -7.854，而中位数仍只偏 -0.065。ES 对污染的比例敏感度是均值的数十倍，中位数则近乎免疫。

## 七、给量化从业者的三条建议

第一，报风险数字必须配误差棒，但要知道误差棒是怎么来的。均值、波动率这类光滑泛函用刀切法没有问题；VaR、中位数这类分位数，请直接放弃刀切法，改用自助法或直接基于影响函数。第二，ES 比 VaR 更能刻画尾部深度，但它对极端点和厚尾极度敏感——小样本下它的置信区间远比你以为的宽，别拿点估计当精确值。第三，如果你的策略对 ES 限额敏感，务必检查样本里有没有少数几天在主导整个估计；影响函数图就是最好的诊断工具，它能直接告诉你哪几天在撬动你的风控指标。

## 八、完整代码与复现

上面全部数字与五张图来自下面这段自包含脚本。它用固定种子 20261122，依次做影响函数的有限差分验证、单个极端点注入、三种标准误对拍、置信区间覆盖率的蒙特卡洛、厚尾与污染压力测试。四个风险度量都用排序后的经验分布直接计算，密度用 Silverman 规则的核密度估计。正文每个数字都能在脚本输出中找到。

```python
# -*- coding: utf-8 -*-
"""Influence functions + delete-1 jackknife for risk-measure inference.

Single source of truth for every number and figure in the article.

Truth-known Monte-Carlo experiments on Gaussian and standardized t(3) returns.
Risk measures (LEFT tail, q=0.05):
    mean, median, VaR_q = F^{-1}(q), ES_q = E[X | X <= VaR_q].

Influence functions (IF) — the derivative of a functional T at F when a single
point x is contaminated by weight eps:
    IF_mean(x)   = x - mu
    IF_med(x)    = (0.5 - 1{x <= m}) / f(m)
    IF_VaR(x)    = (q - 1{x <= xi}) / f(xi),   xi = F^{-1}(q)
    IF_ES(x)     = (xi - ES) + (1/q) * 1{x <= xi} * (x - xi)
The last one is UNBOUNDED in the left tail (slope 1/q), while VaR's IF is
bounded — the key contrast the article documents.

Asymptotic SE from IF:  SE = sqrt(E[IF^2] / n).  We compare it with the
delete-1 jackknife SE and the iid bootstrap SE, then check finite-sample
coverage of the normal-approximation CI, and tail-heavy / contamination stress.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, special

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261122
Q = 0.05
CONF = 0.95
Z = float(special.ndtri(0.5 + CONF / 2.0))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "influence-function-risk-estimation")
os.makedirs(_OUT, exist_ok=True)


# ---------------------------------------------------------------- risk measures
def measures(x, q=Q):
    x = np.asarray(x, dtype=float)
    xm = np.sort(x)
    n = xm.size
    var = float(np.quantile(xm, q))
    tail = xm[xm <= var]
    es = float(tail.mean()) if tail.size else var
    return {
        "mean": float(xm.mean()),
        "median": float(np.median(xm)),
        "var": var,
        "es": es,
    }


def density_at(x, a):
    """Silverman-rule KDE density of x evaluated at scalar a."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = 0.9 * min(np.std(x, ddof=1), stats.iqr(x) / 1.349) * n ** (-0.2)
    h = max(h, 1e-6)
    return float(np.mean(np.exp(-0.5 * ((x - a) / h) ** 2)) / (h * np.sqrt(2 * np.pi)))


def if_values(x, q=Q):
    """Empirical IF at each observation for the four measures."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = measures(x, q)
    med, var, es = m["median"], m["var"], m["es"]
    f_med = density_at(x, med)
    f_var = density_at(x, var)
    ind_med = (x <= med).astype(float)
    ind_var = (x <= var).astype(float)
    return {
        "mean": x - m["mean"],
        "median": (0.5 - ind_med) / f_med,
        "var": (q - ind_var) / f_var,
        "es": (var - es) + (1.0 / q) * ind_var * (x - var),
    }


def se_if(x, q=Q):
    ifs = if_values(x, q)
    return {k: float(np.sqrt(np.mean(v ** 2) / x.size)) for k, v in ifs.items()}


def se_jack(x, q=Q):
    x = np.asarray(x, dtype=float)
    n = x.size
    names = ["mean", "median", "var", "es"]
    acc = {k: [] for k in names}
    for i in range(n):
        t = measures(np.delete(x, i), q)
        for k in names:
            acc[k].append(t[k])
    # delete-1 jackknife: var = (n-1)/n * sum_i (t_(i)-mean)^2 = (n-1)*var(t_(i))
    return {k: float(np.sqrt((n - 1) * np.var(acc[k], ddof=0))) for k in names}


def se_boot(x, q=Q, b=1600, rng=None):
    rng = rng or np.random.default_rng(0)
    n = x.size
    names = ["mean", "median", "var", "es"]
    acc = {k: np.empty(b) for k in names}
    for r in range(b):
        xs = x[rng.integers(0, n, n)]
        t = measures(xs, q)
        for k in names:
            acc[k][r] = t[k]
    return {k: float(np.std(acc[k], ddof=1)) for k in names}


# ---------------------------------------------------------------- exp 1: IF shape
def exp_if_shape():
    rng = np.random.default_rng(SEED)
    n = 200000
    base = rng.standard_normal(n)
    eps = 0.001
    k = int(round(eps * n))          # number of points replaced by the probe x0
    probes = np.linspace(-8.0, 4.0, 25)
    clean = measures(base)
    rows = []
    xi = float(stats.norm.ppf(Q))
    es_true = -float(stats.norm.pdf(xi)) / Q
    for x0 in probes:
        cont = base.copy()
        cont[:k] = x0
        mc = measures(cont)
        rows.append({
            "x0": float(x0),
            "mean": (mc["mean"] - clean["mean"]) / eps,
            "median": (mc["median"] - clean["median"]) / eps,
            "var": (mc["var"] - clean["var"]) / eps,
            "es": (mc["es"] - clean["es"]) / eps,
        })
    # analytic lines for mean / VaR / ES (median handled by finite diff only)
    analytic = {
        "mean_a": [float(r["x0"]) for r in rows],
        "var_a": [(Q - (1.0 if r["x0"] <= xi else 0.0)) / float(stats.norm.pdf(xi)) for r in rows],
        "es_a": [(xi - es_true) + (1.0 / Q) * (1.0 if r["x0"] <= xi else 0.0) * (r["x0"] - xi) for r in rows],
    }
    # agreement metrics (finite-diff vs analytic)
    fd_mean = np.array([r["mean"] for r in rows])
    fd_var = np.array([r["var"] for r in rows])
    fd_es = np.array([r["es"] for r in rows])
    return {
        "eps": eps,
        "xi": xi,
        "es_true": es_true,
        "rows": rows,
        "analytic": analytic,
        "max_abs_err": {
            "mean": float(np.max(np.abs(fd_mean - np.array(analytic["mean_a"])))),
            "var": float(np.max(np.abs(fd_var - np.array(analytic["var_a"])))),
            "es": float(np.max(np.abs(fd_es - np.array(analytic["es_a"])))),
        },
        "es_slope_fd": float((fd_es[0] - fd_es[6]) / (probes[0] - probes[6])),
    }


# ---------------------------------------------------------------- exp 2: single outlier
def exp_single_outlier():
    rng = np.random.default_rng(SEED + 1)
    n = 200
    base = rng.standard_normal(n)
    clean = measures(base)
    xs = np.linspace(-2.0, -20.0, 19)
    mean0 = clean["mean"]
    rows = []
    for x0 in xs:
        cont = np.append(base, x0)          # one extra extreme observation
        mc = measures(cont)
        rows.append({
            "x0": float(x0),
            "d_mean": mc["mean"] - mean0,
            "d_median": mc["median"] - clean["median"],
            "d_var": mc["var"] - clean["var"],
            "d_es": mc["es"] - clean["es"],
        })
    theo_mean = [(x - mean0) / (n + 1) for x in xs]
    return {"n": n, "rows": rows, "theo_mean": theo_mean,
            "clean": clean}


# ---------------------------------------------------------------- exp 3: SE methods
def exp_se_methods():
    rng = np.random.default_rng(SEED + 2)
    n = 500
    out = {}
    for dist, gen in [("normal", "normal"), ("t3", "t3")]:
        if gen == "normal":
            x = rng.standard_normal(n)
        else:
            t = rng.standard_t(3.0, n)
            x = t / np.sqrt(3.0 / (3.0 - 2.0))   # standardize to unit variance
        s_if = se_if(x)
        s_jk = se_jack(x)
        s_bs = se_boot(x, b=1600, rng=np.random.default_rng(SEED + 20))
        out[dist] = {"if": s_if, "jack": s_jk, "boot": s_bs}
    return {"n": n, "methods": out}


# ---------------------------------------------------------------- exp 4: coverage
def _oracle_se_if(x, dist):
    """SE from IF using the TRUE density at the true quantile (best case)."""
    n = x.size
    xs = np.sort(x)
    m = measures(xs)
    if dist == "normal":
        xi = float(stats.norm.ppf(Q))
        fx = float(stats.norm.pdf(xi))
        fmed = float(stats.norm.pdf(0.0))
        es_true = -fx / Q
    else:
        # standardized t(3): scale s = sqrt(3/(3-2)) applied to t
        s = np.sqrt(3.0 / 1.0)
        xi_t = float(stats.t.ppf(Q, 3.0))
        xi = s * xi_t
        fx = float(stats.t.pdf(xi_t, 3.0)) / s
        fmed = float(stats.t.pdf(0.0, 3.0)) / s
        es_true = -(s * (3.0 + xi_t ** 2) / (3.0 - 1.0) * float(stats.t.pdf(xi_t, 3.0))) / Q
    ind_var = (xs <= xi).astype(float)
    ind_med = (xs <= 0.0).astype(float)
    if_var = (Q - ind_var) / fx
    if_med = (0.5 - ind_med) / fmed
    if_es = (xi - es_true) + (1.0 / Q) * ind_var * (xs - xi)
    if_mean = xs - 0.0
    return {
        "mean": float(np.sqrt(np.mean(if_mean ** 2) / n)),
        "median": float(np.sqrt(np.mean(if_med ** 2) / n)),
        "var": float(np.sqrt(np.mean(if_var ** 2) / n)),
        "es": float(np.sqrt(np.mean(if_es ** 2) / n)),
    }


def exp_coverage():
    rng = np.random.default_rng(SEED + 3)
    reps = 1600
    ns = [50, 200, 1000]
    names = ["mean", "median", "var", "es"]
    out = {}
    for dist in ["normal", "t3"]:
        out[dist] = {}
        for n in ns:
            hit = {k: 0 for k in names}
            half_mean = {k: 0.0 for k in names}
            for _ in range(reps):
                if dist == "normal":
                    x = rng.standard_normal(n)
                else:
                    x = rng.standard_t(3.0, n) * np.sqrt(3.0)
                est = measures(x)
                se = _oracle_se_if(x, dist)
                for k in names:
                    half = Z * se[k]
                    half_mean[k] += half
                    if abs(est[k] - _truth(k, dist)) <= half:
                        hit[k] += 1
            out[dist][str(n)] = {
                "cover": {k: hit[k] / reps for k in names},
                "half_width": {k: half_mean[k] / reps for k in names},
            }
    return {"reps": reps, "ns": ns, "coverage": out}


def _truth(k, dist):
    if dist == "normal":
        xi = float(stats.norm.ppf(Q))
        return {
            "mean": 0.0, "median": 0.0, "var": xi,
            "es": -float(stats.norm.pdf(xi)) / Q,
        }[k]
    s = np.sqrt(3.0)
    xi_t = float(stats.t.ppf(Q, 3.0))
    xi = s * xi_t
    es = -(s * (3.0 + xi_t ** 2) / 2.0 * float(stats.t.pdf(xi_t, 3.0))) / Q
    return {"mean": 0.0, "median": 0.0, "var": xi, "es": es}[k]


# ---------------------------------------------------------------- exp 5: tail heaviness
def exp_tail_heaviness():
    rng = np.random.default_rng(SEED + 4)
    n = 400
    nus = [2.5, 3.0, 5.0, 10.0, 30.0]
    rows = []
    for nu in nus:
        x = rng.standard_t(nu, n) * np.sqrt(nu / (nu - 2.0))
        ifs = if_values(x)
        rows.append({
            "nu": nu,
            "mean_if2": float(np.mean(ifs["mean"] ** 2)),
            "es_if2": float(np.mean(ifs["es"] ** 2)),
            "ratio": float(np.mean(ifs["es"] ** 2) / np.mean(ifs["mean"] ** 2)),
        })
    return {"n": n, "rows": rows}


# ---------------------------------------------------------------- exp 6: contamination
def exp_contamination():
    rng = np.random.default_rng(SEED + 5)
    n = 1000
    base = rng.standard_normal(n)
    clean = measures(base)
    eps_grid = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
    rows = []
    for eps in eps_grid:
        k = int(round(eps * n))
        cont = base.copy()
        if k:
            cont[:k] = -10.0
        mc = measures(cont)
        rows.append({
            "eps": eps,
            "mean": mc["mean"] - clean["mean"],
            "median": mc["median"] - clean["median"],
            "var": mc["var"] - clean["var"],
            "es": mc["es"] - clean["es"],
        })
    return {"n": n, "x0": -10.0, "rows": rows}


# ---------------------------------------------------------------- plots
def plot1(r):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rows = r["rows"]
    xs = [q["x0"] for q in rows]
    ax[0].plot(xs, [q["mean"] for q in rows], "o-", color="#c0392b", label="FD mean IF")
    ax[0].plot(xs, r["analytic"]["mean_a"], "--", color="#333", label="theory x")
    ax[0].set_title("Mean IF: unbounded, linear"); ax[0].legend(fontsize=8)
    ax[1].plot(xs, [q["var"] for q in rows], "o-", color="#2980b9", label="FD VaR IF")
    ax[1].plot(xs, r["analytic"]["var_a"], "--", color="#333", label="theory")
    ax[1].set_title("VaR IF: bounded step"); ax[1].legend(fontsize=8)
    ax[2].plot(xs, [q["es"] for q in rows], "o-", color="#8e44ad", label="FD ES IF")
    ax[2].plot(xs, r["analytic"]["es_a"], "--", color="#333", label="theory")
    ax[2].set_title("ES IF: unbounded in left tail"); ax[2].legend(fontsize=8)
    for a in ax:
        a.axhline(0, color="grey", lw=0.6); a.set_xlabel("contaminating point x0")
        a.grid(alpha=0.3)
    fig.suptitle("Finite-difference empirical IF vs analytic formula")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_if_shape.png"), dpi=120)
    plt.close(fig); return "fig1_if_shape.png"


def plot2(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [q["x0"] for q in r["rows"]]
    ax.plot(xs, [q["d_mean"] for q in r["rows"]], "o-", color="#c0392b", label="mean")
    ax.plot(xs, [q["d_median"] for q in r["rows"]], "s-", color="#27ae60", label="median")
    ax.plot(xs, [q["d_var"] for q in r["rows"]], "^-", color="#2980b9", label="VaR95")
    ax.plot(xs, [q["d_es"] for q in r["rows"]], "d-", color="#8e44ad", label="ES95")
    ax.plot(xs, r["theo_mean"], "--", color="#c0392b", alpha=0.5, label="mean theory (x-mu)/(n+1)")
    ax.set_xlabel("single injected loss x0"); ax.set_ylabel("change in estimate")
    ax.set_title("One extreme trading day: who moves, who stands still")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.axhline(0, color="grey", lw=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_single_outlier.png"), dpi=120)
    plt.close(fig); return "fig2_single_outlier.png"


def plot3(r):
    names = ["mean", "median", "var", "es"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, dist in enumerate(["normal", "t3"]):
        d = r["methods"][dist]
        x = np.arange(4); w = 0.26
        for i, (m, c) in enumerate([("if", "#2c3e50"), ("jack", "#e67e22"), ("boot", "#16a085")]):
            vals = [d[m][k] for k in names]
            ax[j].bar(x + (i - 1) * w, vals, w, label=m, color=c)
        ax[j].set_xticks(x); ax[j].set_xticklabels(names)
        ax[j].set_title(f"SE comparison ({dist}, n={r['n']})")
        ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3, axis="y")
    fig.suptitle("IF-asymptotic vs delete-1 jackknife vs iid bootstrap")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_se_methods.png"), dpi=120)
    plt.close(fig); return "fig3_se_methods.png"


def plot4(r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["mean", "median", "var", "es"]
    cols = {"mean": "#c0392b", "median": "#27ae60", "var": "#2980b9", "es": "#8e44ad"}
    for j, dist in enumerate(["normal", "t3"]):
        for k in names:
            vals = [r["coverage"][dist][str(n)]["cover"][k] for n in r["ns"]]
            ax[j].plot(r["ns"], vals, "o-", color=cols[k], label=k)
        ax[j].axhline(CONF, color="grey", ls="--", lw=1, label="95% target")
        ax[j].set_ylim(0.4, 1.02); ax[j].set_xlabel("sample size n")
        ax[j].set_title(f"95% CI coverage ({dist})")
        ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3)
    fig.suptitle("Normal-approx CI built from IF standard errors")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_coverage.png"), dpi=120)
    plt.close(fig); return "fig4_coverage.png"


def plot5(r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = [q["nu"] for q in r["rows"]]
    ax[0].plot(xs, [q["mean_if2"] for q in r["rows"]], "o-", color="#c0392b", label="mean")
    ax[0].plot(xs, [q["es_if2"] for q in r["rows"]], "o-", color="#8e44ad", label="ES")
    ax[0].set_xlabel("t degrees of freedom"); ax[0].set_ylabel("E[IF^2]")
    ax[0].set_title("Asymptotic variance factor"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(xs, [q["ratio"] for q in r["rows"]], "o-", color="#e67e22")
    ax[1].set_xlabel("t degrees of freedom"); ax[1].set_ylabel("ES IF^2 / mean IF^2")
    ax[1].set_title("ES asymptotic variance blow-up vs mean")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Tail heaviness makes ES far more variable than the mean")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_tail.png"), dpi=120)
    plt.close(fig); return "fig5_tail.png"


def main():
    e1 = exp_if_shape()
    e2 = exp_single_outlier()
    e3 = exp_se_methods()
    e4 = exp_coverage()
    e5 = exp_tail_heaviness()
    e6 = exp_contamination()
    results = {
        "setup": {"seed": SEED, "q": Q, "conf": CONF, "z": Z},
        "if_shape": e1,
        "single_outlier": e2,
        "se_methods": e3,
        "coverage": e4,
        "tail": e5,
        "contamination": e6,
    }
    results["figs"] = [plot1(e1), plot2(e2), plot3(e3), plot4(e4), plot5(e5)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 九、小结

影响函数是理解风险估计稳健性的统一透镜：它解释了为什么均值怕极端点、VaR 不怕、ES 最怕，也解释了为什么刀切法在光滑处灵验、在分位数处失灵。尾部风险的误差棒不能偷懒——分位数别用刀切法，ES 的小样本区间要留足余地，厚尾下它的方差能比均值大两个数量级。看懂影响函数，才知道你的风控数字究竟有多可信。
