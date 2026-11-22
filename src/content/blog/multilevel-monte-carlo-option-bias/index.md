---
title: "多层蒙特卡洛 MLMC：欧拉离散偏差怎么一层层拆掉，以及不连续收益为什么把省下的算力吃回去"
description: "在 S0=1、r=0.05、sigma=0.2、T=1 的 GBM 上，把欧拉格式的弱偏差与统计误差分开算：连续看涨的偏差从 l=0 的 -0.00259073 按 h^1.0942 衰减，数字期权的偏差从 0.03666038 按 h^1.0013 衰减，但层间方差的衰减率分别是 h^0.9578 与 h^0.4229。用同一组细网格布朗增量耦合相邻层，按 Neyman 分配样本：目标 RMSE 0.000625 时看涨期权 MLMC 只需 194461 个正态数，朴素 MC 要 1726616 个（省 8.88 倍）；数字期权只省 5.13 倍，成本指数从理想的 -2 退化到 -2.7185。40 次独立重跑显示两者 RMSE 接近而成本差三倍以上，附完整代码与五张真实计算图。"
publishDate: '2026-11-22T11:20:00Z'
tags:
  - 量化
  - 蒙特卡洛
  - 多层方法
  - 期权定价
  - Python
language: zh
difficulty: advanced
image: "/images/multilevel-monte-carlo-option-bias/fig1_weak_bias.png"
---

## 一、结论先行

期权定价里的蒙特卡洛误差有两笔账。一笔是统计误差，按 $N^{-1/2}$ 掉；另一笔是离散偏差，欧拉格式的时间步长 $h$ 砍一半，偏差也大致砍一半。想把总误差降到 $\varepsilon$，朴素做法要在最细网格上同时付两笔钱：样本量按 $\varepsilon^{-2}$ 涨，步长按 $\varepsilon$ 缩，总成本约 $\varepsilon^{-3}$。

多层蒙特卡洛（MLMC）的核心恒等式是望远镜求和：$\mathbb{E}[P_L]=\mathbb{E}[P_0]+\sum_{l=1}^L \mathbb{E}[P_l-P_{l-1}]$。粗网格层用少量样本就能估准，细网格层的修正项 $P_l-P_{l-1}$ 方差很小，也只需少量样本。只要相邻层用同一组布朗增量耦合，总成本可以压到 $\varepsilon^{-2}(\log\varepsilon)^2$ 量级。

在这套 GBM 参数上（$S_0=1$、$r=0.05$、$\sigma=0.2$、$T=1$，Black-Scholes 看涨 0.10450584、数字期权 0.53232482），实测结果分三层。第一层是弱偏差：看涨从 l=0 的 -0.00259073 衰减到 l=8 的 -0.00000391，拟合斜率 1.0942；数字期权从 0.03666038 降到 0.00018073，斜率 1.0013。第二层是层间方差：看涨的 $\mathrm{Var}(P_l-P_{l-1})$ 从 0.00023834 降到 0.00000230，斜率 0.9578；数字期权只从 0.01728731 降到 0.00240688，斜率 0.4229。第三层是省钱效果：目标 $\varepsilon=0.000625$ 时看涨期权 MLMC 成本 194461，朴素 MC 1726616，省 8.88 倍；数字期权只省 5.13 倍，成本指数退化到 -2.7185。所有数字来自文末同一段代码。

## 二、先把离散偏差量准

欧拉格式为 $S_{t+\Delta t}=S_t(1+r\Delta t+\sigma\sqrt{\Delta t}\,Z)$，层数 $l$ 对应 $m=2^l$ 步、$h=2^{-l}$。朴素蒙特卡洛在这个格式上跑最细网格，均值收敛到的是带偏差的欧拉价格，不是 Black-Scholes 价格。

直接拿样本均值减 BS 价格来量偏差，会被蒙特卡洛噪声淹没：细网格层上 payoff 的标准误约 0.00027，而 l=8 的真实偏差只有几百万分之一。这里用了一个系数为 1 的控制变量：同一组底层正态增量既生成欧拉终值，也生成精确 GBM 终值 $S_T=S_0\exp[(r-\sigma^2/2)T+\sigma\sqrt{T}\,Z]$。精确 payoff 的期望恰是 BS 价格，于是 $\mathbb{E}[P^{\text{Euler}}-P^{\text{exact}}]$ 就是偏差，而两者高度相关，差值方差远小于单独估计。

![图1：欧拉格式的弱偏差随网格步长衰减，两条曲线斜率都接近 1](/images/multilevel-monte-carlo-option-bias/fig1_weak_bias.png)

看涨期权 l=0 偏差 -0.00259073，l=2 是 -0.00041048，l=4 是 -0.00007637，l=8 是 -0.00000391，log-log 斜率 1.0942。数字期权更慢：l=0 偏差 0.03666038，l=3 还有 0.00423773，l=8 才降到 0.00018073，斜率 1.0013。两条曲线都验证了一阶弱收敛，但常数项差了一个数量级——这已经预告了后面的算力分化。

## 三、耦合决定层间方差

望远镜恒等式只有在 $P_l-P_{l-1}$ 的方差足够小时才有用。耦合方法是：先生成 $2^l$ 个细网格正态增量，把相邻两个按 $(Z_{2j}+Z_{2j+1})/\sqrt{2}$ 合并成粗网格增量。两份路径用完全相同的随机源， payoff 的连续性使差值主要由离散误差决定。

![图2：相邻层差值的方差与均值随层数变化](/images/multilevel-monte-carlo-option-bias/fig2_level_variance.png)

看涨期权差值方差从 l=1 的 0.00023834 近乎线性地降到 l=8 的 0.00000230，斜率 0.9578，贴合理论的一阶衰减。数字期权就难看得多：l=1 是 0.01728731，l=8 还有 0.00240688，斜率只有 0.4229。原因是示性 payoff 在行权价附近跳变，细网格上欧拉终值只要跨过 $K=1$ 这道线，两份路径的 payoff 就差出一整个折现单位；小步长只缩小了跨线的概率，没有缩小跨线时的跳幅。

层间均值差也印证了这一点：看涨期权各层的 $|E[P_l-P_{l-1}]|$ 随 $h$ 快速趋零，数字期权的高层修正项虽然变小，却始终带着可观的噪声。换句话说，MLMC 省不省钱，取决于 payoff 对扰动是否连续——这是财务建模里经常被忽略的结构条件。

## 四、样本怎么分才不浪费

给定目标 RMSE $\varepsilon$，把预算拆成统计误差不超过 $\varepsilon^2/2$、偏差不超过 $\varepsilon/2$。层数取满足实测偏差 $\le\varepsilon/2$ 的最小 $l$；各层按 Neyman 分配：

$$
N_l=\frac{2}{\varepsilon^2}\sqrt{\frac{V_l}{C_l}}\sum_{k=0}^L\sqrt{V_k C_k},\qquad C_l=2^l
$$

其中 $V_0$ 用最粗层 payoff 方差，$l\ge1$ 用上节测得的层间方差。

![图3：最优样本分配与等样本分配在同一成本下的精度对比](/images/multilevel-monte-carlo-option-bias/fig3_allocation.png)

以 $\varepsilon=0.0025$ 为例，看涨期权取 l=1，最优分配是 $N_0=6088$、$N_1=522$，统计标准差 0.00176753；把同样 7132 个正态数的成本平均分给两层，标准差升到 0.00263295，劣化 49%。数字期权同样目标需要 l=5，最优分配 $N_0=97292$、$N_1=19402$，标准差 0.00176776 对平均分配的 0.00227424。粗层承担大部分抽样，因为它的单点成本最低；细层只做小修小补。

注意，当 $\varepsilon$ 很松时（看涨 $\varepsilon\ge0.01$），l=0 的偏差已经低于 $\varepsilon/2$，根本没有可拆的层，MLMC 与朴素 MC 在这个设定下没有区别。多层不是免费的宗教信仰，它只在偏差逼你细化网格时才开始工作。

## 五、40 次重跑：省的是成本，不是误差

理论分配给出的标准差只是预算。实际估出来的价格偏差和方差是否相符？对每种 payoff 独立跑 40 次完整估计，朴素对照在相同最细层上取统计标准差 $\varepsilon/2$ 的样本量。

![图4：40 次独立估计的分布与 Black-Scholes 参考价](/images/multilevel-monte-carlo-option-bias/fig4_estimator.png)

看涨期权在 $\varepsilon=0.0025$、l=1 下：MLMC 40 次均值 0.10338351，RMSE 0.00191355；朴素 MC 均值 0.10354620，RMSE 0.00187328。两者都比 BS 的 0.10450584 低约 0.001，是 l=1 的离散偏差 -0.00100426 在起作用，方法之间 RMSE 几乎打平；但 MLMC 成本 7132，朴素 MC 24068，便宜 3.37 倍。诚实说法是：同样精度，少花三倍算力，而不是 MLMC 把误差压得更低。

数字期权需要 l=5：MLMC 均值 0.53303594、RMSE 0.00176852，朴素 MC 均值 0.53387003、RMSE 0.00188892，参考价 0.53232482；成本 1410788 对 4562336，省 3.23 倍。这一组 MLMC 的 RMSE 略低，但差距在 40 次重复的抽样波动范围内，不宜过度解读。

## 六、成本指数与边界

把 $\varepsilon$ 从 0.01 扫到 0.000625，用实测偏差和方差分别规划两种方法所需的正态数总数。

![图5：目标精度对成本的双对数曲线](/images/multilevel-monte-carlo-option-bias/fig5_cost.png)

看涨期权上，朴素 MC 成本指数 -2.7603，MLMC -2.2711；$\varepsilon$ 每减半，MLMC 的相对优势从 3.37 倍扩大到 8.88 倍。数字期权上，朴素是 -3.0007，MLMC 退化到 -2.7185，优势只从 2.35 倍爬到 5.13 倍。理想 MLMC 的指数接近 -2，看涨已经很接近，数字期权被不连续 payoff 拖住。

三条边界值得写进工具箱。第一，弱偏差必须先量准；用独立样本在细层上减 BS，噪声会把一阶斜率淹掉，本文用精确 GBM payoff 做同路径控制变量才拿到 1.0942 与 1.0013。第二，$V_l$ 的衰减率是方法能否省钱的判据；先跑几层 pilot 再决定是否上 MLMC，不要默认适用。第三，本文的欧拉格式是一阶弱收敛；换成 Milstein 或更高阶格式会改变偏差常数和斜率，但耦合与分配公式不变。

## 七、从期权定价回到定价引擎

把这套方法搬进实际的定价或风控引擎，流程可以拆成四步，每一步都有明确的验收数字。先跑 pilot：用十万到二十万条共同路径，在 l=0 到 l=8 上同时记录欧拉 payoff 与精确 payoff 的差值，画出图1 的偏差曲线。如果 log-log 斜率显著低于 0.8，说明格式实现可能有误，或者 payoff 的奇点正在主导误差，先修实现再谈加速。

再做层间方差诊断。图2 给出的斜率是可预期的省钱上限：看涨 0.9578 意味着每细化一层方差近乎减半，数字期权 0.4229 意味着减半速度慢得多。工程上可以把这个斜率写进监控面板，一旦某新产品族的斜率跌破 0.5 就自动退回朴素 MC 或改用专门的平滑处理，避免把算力浪费在收不回来的细层上。

第三步是预算规划。目标精度 $\varepsilon$ 要先换成业务语言：是价格小数点后第三位的绝对误差，还是相对名义本金的风险容忍度？本文用绝对 RMSE，所以 $\varepsilon=0.0025$ 在 $S_0=1$ 上是 0.25% 的量级。选定 $\varepsilon$ 后再按第四节公式分配各层样本；不要把统计精度目标与偏差目标混为一谈，第五节里两种方法的 RMSE 接近但都带着 l=1 的离散偏差，就是最直接的提醒。

最后是验证。用独立于 pilot 的随机种子重跑几十次完整估计，把 RMSE、均值和成本一起拿去和朴素 MC 对拍，而不是只看单次价格是否贴近参考价。本文第四、五节的数字就是按这个口径出具的：MLMC 的优势是成本，不是神迹；把它讲成后者，迟早在奇异 payoff 上还债。

```python
# -*- coding: utf-8 -*-
"""Multilevel Monte Carlo for Euler-discretised option prices.

Single source of truth for every number and figure in the article.

GBM with S0=1, r=0.05, sigma=0.2, T=1. Euler scheme at mesh h=2^-l has
first-order weak bias, so plain Monte Carlo at the finest mesh pays for both
statistical error and discretisation bias. MLMC telescoping

    E[P_L] = E[P_0] + sum_{l=1..L} E[P_l - P_{l-1}]

couples each pair through the same Brownian increments (coarse increments are
sums of fine pairs). Levels with small variance need few samples, so the total
cost grows like eps^-2 (log eps)^2 instead of eps^-3.

We measure weak bias, level variances, optimal allocation, actual estimator
RMSE, and the cost exponent for both a continuous call and a discontinuous
cash-or-nothing digital. Black-Scholes closed forms are the reference prices.
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
S0, K, R, SIG, T = 1.0, 1.0, 0.05, 0.20, 1.0
LMAX = 8
EPS_LIST = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
EPS_DEMO = 0.0025
N_REPS = 40

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "multilevel-monte-carlo-option-bias")
os.makedirs(_OUT, exist_ok=True)


# ------------------------------------------------------------------- benchmark
def bs_price(kind):
    d1 = (np.log(S0 / K) + (R + 0.5 * SIG ** 2) * T) / (SIG * np.sqrt(T))
    d2 = d1 - SIG * np.sqrt(T)
    if kind == "call":
        return float(S0 * stats.norm.cdf(d1) - K * np.exp(-R * T) * stats.norm.cdf(d2))
    if kind == "digital":
        return float(np.exp(-R * T) * stats.norm.cdf(d2))
    raise ValueError(kind)


def payoff(kind, ST):
    disc = np.exp(-R * T)
    if kind == "call":
        return disc * np.maximum(ST - K, 0.0)
    return disc * (ST > K).astype(float)


def euler_from_increments(z):
    """z (n,m) standard normals; Euler steps with m steps over T."""
    m = z.shape[1]
    dt = T / m
    s = np.full(z.shape[0], S0)
    for j in range(m):
        s = s * (1.0 + R * dt + SIG * np.sqrt(dt) * z[:, j])
    return s


def single_level(n, m, rng):
    out = []
    for start in range(0, n, 50000):
        b = min(50000, n - start)
        out.append(euler_from_increments(rng.standard_normal((b, m))))
    return np.concatenate(out)


def level_pair(n, l, rng, kind):
    m = 2 ** l
    z = rng.standard_normal((n, m))
    pf = payoff(kind, euler_from_increments(z))
    coz = (z[:, 0::2] + z[:, 1::2]) / np.sqrt(2.0)
    pc = payoff(kind, euler_from_increments(coz))
    d = pf - pc
    return float(np.mean(d)), float(np.var(d, ddof=1))


def pilot(kind, rng):
    """Weak bias and level variances from ONE shared fine Brownian path set.

    All levels are coarsenings of the same 2^LMAX-step increments, so the
    bias differences across levels are coupled and Monte-Carlo noise largely
    cancels. This is essential: independent level samples drown a O(h) call
    bias under O(n^-1/2) noise at fine meshes.
    """
    ref = bs_price(kind)
    n = 200000
    payoff_var = {}
    bias_rows, var_rows = [], []
    sums = np.zeros(LMAX + 1)       # sum of Euler - exact-GBM payoff (CV with beta=1)
    sumsq = np.zeros(LMAX + 1)
    raw_mean = np.zeros(LMAX + 1)
    for start in range(0, n, 40000):
        b = min(40000, n - start)
        z = rng.standard_normal((b, 2 ** LMAX))
        zsum = z.sum(axis=1) / np.sqrt(2 ** LMAX)
        exact_st = S0 * np.exp((R - 0.5 * SIG ** 2) * T + SIG * np.sqrt(T) * zsum)
        exact_p = payoff(kind, exact_st)
        for l in range(LMAX + 1):
            g = 2 ** (LMAX - l)
            zc = z.reshape(b, 2 ** l, g).sum(axis=2) / np.sqrt(g)
            p = payoff(kind, euler_from_increments(zc))
            diff = p - exact_p
            sums[l] += diff.sum()
            sumsq[l] += np.sum(diff ** 2)
            raw_mean[l] += p.sum()
            payoff_var.setdefault(str(l), []).append(p.var(ddof=1))
    for l in range(LMAX + 1):
        bias = sums[l] / n
        se_diff = np.sqrt(max(sumsq[l] / n - bias ** 2, 0) / n)
        bias_rows.append({"l": l, "h": round(2.0 ** -l, 8),
                          "mean": round(float(raw_mean[l] / n), 8),
                          "bias": round(float(bias), 8),
                          "se_shared": round(float(se_diff), 8),
                          "n": int(n)})
        payoff_var[str(l)] = round(float(np.mean(payoff_var[str(l)])), 10)
        v = None
        if l >= 1:
            diffs = []
            for start in range(0, n, 20000):
                z = rng.standard_normal((min(20000, n - start), 2 ** l))
                pf = payoff(kind, euler_from_increments(z))
                coz = (z[:, 0::2] + z[:, 1::2]) / np.sqrt(2.0)
                pc = payoff(kind, euler_from_increments(coz))
                diffs.append(pf - pc)
            d = np.concatenate(diffs)
            v = {"l": l, "mean_diff": round(float(d.mean()), 8),
                 "var_diff": round(float(d.var(ddof=1)), 10), "n": int(n)}
        var_rows.append(v)
    return {"ref": round(ref, 8), "payoff_var": payoff_var,
            "bias": bias_rows, "variance": var_rows}


def fit_slope(xs, ys):
    a = np.polyfit(np.log(np.asarray(xs, float)), np.log(np.abs(np.asarray(ys, float))), 1)
    return {"slope": round(float(a[0]), 6), "intercept": round(float(a[1]), 6)}


def choose_level(bias_rows, eps):
    """Smallest level whose measured bias is at most eps/2."""
    for row in bias_rows:
        if abs(row["bias"]) <= eps / 2.0:
            return row["l"]
    return None


def allocate(var_rows, level, eps, v0):
    """Standard MLMC Neyman allocation: N_l = 2/eps^2 * sqrt(V_l/C_l)*sum sqrt(V_k C_k)."""
    levels = list(range(0, level + 1))
    vv = np.array([v0 if l == 0 else var_rows[l]["var_diff"] for l in levels], float)
    cost_l = np.array([2.0 ** l for l in levels], float)
    s = float(np.sum(np.sqrt(vv * cost_l)))
    ns = (2.0 / eps ** 2) * np.sqrt(vv / cost_l) * s
    ns = np.maximum(np.ceil(ns), 100.0)
    stat_var = float(np.sum(vv / ns))
    cost = float(np.sum(ns * cost_l))
    return {"level": level, "N": {str(l): int(ns[i]) for i, l in enumerate(levels)},
            "stat_var": round(stat_var, 10), "stat_sd": round(float(np.sqrt(stat_var)), 8),
            "cost": round(cost, 2)}


def mlmc_sample(level, ns, rng, kind):
    total = 0.0
    z0 = rng.standard_normal((int(ns["0"]), 1))
    p0 = payoff(kind, euler_from_increments(z0))
    total += float(p0.mean())
    for l in range(1, level + 1):
        n = int(ns[str(l)])
        out = []
        for start in range(0, n, 40000):
            b = min(40000, n - start)
            z = rng.standard_normal((b, 2 ** l))
            pf = payoff(kind, euler_from_increments(z))
            coz = (z[:, 0::2] + z[:, 1::2]) / np.sqrt(2.0)
            pc = payoff(kind, euler_from_increments(coz))
            out.append(pf - pc)
        total += float(np.concatenate(out).mean())
    return total


def plain_sample(level, eps, var_L, rng, kind):
    n = int(np.ceil(4.0 * var_L / eps ** 2))     # sd <= eps/2
    out = []
    for start in range(0, n, 40000):
        b = min(40000, n - start)
        out.append(payoff(kind, single_level(b, 2 ** level, rng)))
    return float(np.concatenate(out).mean()), n


# ------------------------------------------------------------------ experiments
def exp_bias_variance():
    res = {}
    for kind in ["call", "digital"]:
        p = pilot(kind, rng=np.random.default_rng(SEED + (0 if kind == "call" else 1)))
        br = p["bias"]
        vr = [v for v in p["variance"] if v is not None]
        hs = [row["h"] for row in br[1:]]
        bs = [row["bias"] for row in br[1:]]
        vs = [v["var_diff"] for v in vr]
        res[kind] = {
            "ref": p["ref"], "bias": br, "variance": p["variance"],
            "payoff_var": p["payoff_var"],
            "weak_slope": fit_slope(hs, bs),
            "var_slope": fit_slope(hs, vs),
            "max_abs_bias": round(max(abs(x) for x in bs), 8),
        }
    return res


def exp_cost(pv):
    res = {}
    for kind in ["call", "digital"]:
        rows = []
        for eps in EPS_LIST:
            lev = choose_level(pv[kind]["bias"], eps)
            if lev is None or lev < 1:
                rows.append({"eps": eps, "level": lev, "mlmc_cost": None, "plain_cost": None})
                continue
            a = allocate(pv[kind]["variance"], lev, eps, pv[kind]["payoff_var"]["0"])
            # plain MC needs Var(P_L) at the same mesh; estimate from pilot sample variance
            var_payoff = pv[kind]["payoff_var"].get(str(lev))
            n_plain = int(np.ceil(4.0 * var_payoff / eps ** 2))
            plain_cost = n_plain * (2.0 ** lev)
            rows.append({"eps": eps, "level": lev, "mlmc_cost": round(a["cost"], 2),
                         "plain_cost": round(plain_cost, 2),
                         "speedup": round(plain_cost / a["cost"], 3),
                         "mlmc_stat_sd": a["stat_sd"],
                         "bias_at_level": abs(pv[kind]["bias"][lev]["bias"]),
                         "N_total": int(sum(a["N"].values()))})
        xs = [r["eps"] for r in rows if r["mlmc_cost"]]
        ym = [r["mlmc_cost"] for r in rows if r["mlmc_cost"]]
        yp = [r["plain_cost"] for r in rows if r["plain_cost"]]
        res[kind] = {"rows": rows,
                     "mlmc_cost_slope": fit_slope(xs, ym) if xs else None,
                     "plain_cost_slope": fit_slope(xs, yp) if xs else None}
    return res


def exp_validation(pv):
    out = {}
    for kind in ["call", "digital"]:
        lev = choose_level(pv[kind]["bias"], EPS_DEMO)
        rng_k = np.random.default_rng(SEED + (20 if kind == "call" else 21))
        ak = allocate(pv[kind]["variance"], lev, EPS_DEMO, pv[kind]["payoff_var"]["0"])
        ml = [mlmc_sample(lev, ak["N"], rng_k, kind) for _ in range(N_REPS)]
        var_L = pv[kind]["payoff_var"][str(lev)]
        pl, np_plain = [], None
        for _ in range(N_REPS):
            v, np_plain = plain_sample(lev, EPS_DEMO, var_L, rng_k, kind)
            pl.append(v)
        ref = pv[kind]["ref"]
        out[kind] = {
            "level": lev, "eps": EPS_DEMO, "N_plain": int(np_plain),
            "mlmc_mean": round(float(np.mean(ml)), 8),
            "plain_mean": round(float(np.mean(pl)), 8),
            "ref": ref,
            "mlmc_rmse": round(float(np.sqrt(np.mean((np.array(ml) - ref) ** 2))), 8),
            "plain_rmse": round(float(np.sqrt(np.mean((np.array(pl) - ref) ** 2))), 8),
            "mlmc_sd": round(float(np.std(ml, ddof=1)), 8),
            "plain_sd": round(float(np.std(pl, ddof=1)), 8),
            "mlmc_cost": round(ak["cost"], 2),
            "plain_cost": round(np_plain * 2.0 ** lev, 2),
            "n_reps": N_REPS,
            "estimates_mlmc": [round(float(v), 8) for v in ml],
            "estimates_plain": [round(float(v), 8) for v in pl],
        }
    return out


def exp_allocation(pv):
    lev = choose_level(pv["call"]["bias"], EPS_DEMO)
    out = {}
    for kind in ["call", "digital"]:
        a = allocate(pv[kind]["variance"], lev, EPS_DEMO, pv[kind]["payoff_var"]["0"])
        equal_n = int(a["cost"] / sum(2.0 ** l for l in range(0, lev + 1)))
        vv = [pv[kind]["payoff_var"]["0"]] + [pv[kind]["variance"][l]["var_diff"] for l in range(1, lev + 1)]
        eq_var = float(np.sum(np.array(vv, float) / equal_n))
        out[kind] = {"level": lev, "eps": EPS_DEMO, "N_opt": a["N"],
                     "N_equal": equal_n, "opt_sd": a["stat_sd"],
                     "equal_sd": round(float(np.sqrt(eq_var)), 8),
                     "opt_cost": round(a["cost"], 2),
                     "equal_cost": round(a["cost"], 2)}
    return out


# ------------------------------------------------------------------ figures
def plot_bias(pv):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = pv[kind]
        hs = [r["h"] for r in d["bias"][1:]]
        bs = [abs(r["bias"]) for r in d["bias"][1:]]
        a.loglog(hs, bs, "o-", color="#b23a2b")
        a.set_title("%s: weak bias slope %.2f" % (kind, d["weak_slope"]["slope"]))
        a.set_xlabel("mesh h = 2^-l"); a.set_ylabel("|E[P_l] - BS|")
        a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_weak_bias.png"), dpi=115)
    plt.close(fig); return "fig1_weak_bias.png"


def plot_variance(pv):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = pv[kind]
        lv = [v["l"] for v in d["variance"] if v is not None]
        vv = [v["var_diff"] for v in d["variance"] if v is not None]
        md = [abs(v["mean_diff"]) for v in d["variance"] if v is not None]
        a.loglog(lv, vv, "o-", color="#2b5d8a", label="Var(P_l-P_l-1)")
        a.loglog(lv, md, "s--", color="#2e7d55", label="|mean difference|")
        a.set_title("%s: Var slope %.2f" % (kind, d["var_slope"]["slope"]))
        a.set_xlabel("level l"); a.set_ylabel("level difference")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_level_variance.png"), dpi=115)
    plt.close(fig); return "fig2_level_variance.png"


def plot_alloc(al):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = al[kind]
        ls = sorted(int(x) for x in d["N_opt"])
        a.bar(np.array(ls) - 0.18, [d["N_opt"][str(l)] for l in ls], width=0.36,
              color="#b23a2b", label="optimal N_l")
        a.bar(np.array(ls) + 0.18, [d["N_equal"]] * len(ls), width=0.36,
              color="#2b5d8a", alpha=0.6, label="equal N_l")
        a.set_yscale("log"); a.set_title("%s: sd opt %.5f / equal %.5f" %
                                         (kind, d["opt_sd"], d["equal_sd"]))
        a.set_xlabel("level l"); a.set_ylabel("samples (log)"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_allocation.png"), dpi=115)
    plt.close(fig); return "fig3_allocation.png"


def plot_validation(va):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = va[kind]
        a.boxplot([d["estimates_mlmc"], d["estimates_plain"]],
                  tick_labels=["MLMC", "plain MC"])
        a.axhline(d["ref"], color="#2e7d55", ls="--", lw=1.5, label="BS %.5f" % d["ref"])
        a.set_title("%s: RMSE %.5f vs %.5f" % (kind, d["mlmc_rmse"], d["plain_rmse"]))
        a.set_ylabel("estimated price"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_estimator.png"), dpi=115)
    plt.close(fig); return "fig4_estimator.png"


def plot_cost(cost):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, kind in zip(ax, ["call", "digital"]):
        d = cost[kind]
        rows = [r for r in d["rows"] if r["mlmc_cost"]]
        es = [r["eps"] for r in rows]
        a.loglog(es, [r["plain_cost"] for r in rows], "s-", color="#2b5d8a", label="plain MC")
        a.loglog(es, [r["mlmc_cost"] for r in rows], "o-", color="#b23a2b", label="MLMC")
        a.set_title("%s: slope plain %.2f / MLMC %.2f" %
                    (kind, d["plain_cost_slope"]["slope"], d["mlmc_cost_slope"]["slope"]))
        a.set_xlabel("target RMSE eps"); a.set_ylabel("normal draws")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_cost.png"), dpi=115)
    plt.close(fig); return "fig5_cost.png"


def main():
    pv = exp_bias_variance()
    cost = exp_cost(pv)
    al = exp_allocation(pv)
    va = exp_validation(pv)
    r = {"setup": {"seed": SEED, "S0": S0, "K": K, "r": R, "sigma": SIG, "T": T,
                   "LMAX": LMAX, "eps_demo": EPS_DEMO, "n_reps": N_REPS,
                   "eps_list": EPS_LIST},
         "pilot": pv, "cost": cost, "allocation": al, "validation": va}
    r["figs"] = [plot_bias(pv), plot_variance(pv), plot_alloc(al),
                 plot_validation(va), plot_cost(cost)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

