---
title: "指数效用 CARA 组合与鞍点近似：为什么正态世界的需求等于均值方差、非正态尾部却会把确定性等价系统性算错"
description: "三正态混合日收益：均值 0.000780、标准差 0.00955676、偏度 -3.4898、超额峰度 52.8859。a=10 时正态需求 0.854031，鞍点最优 0.73，低配 0.124031；正态需求上的真实确定性等价 0.00029031，比鞍点最优 0.00030048 少 0.00001017，年化损失 25.618bp。正态公式却把同一个权重的确定性等价报成 0.00033307，高估 0.00004276，年化 107.76bp。样本量 1000 时经验 CGF 平均后悔 0.00008474 优于正态 0.00012554，样本量 60 时两者几乎打平。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T18:14:00Z'
tags:
  - 量化
  - 组合优化
  - 风险度量
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

指数效用是目前最干净的仓位框架之一：绝对风险厌恶系数 a 一旦给定，最优权重就有明确定义，不需要人为指定方差和收益的权衡比。但很多实现直接套用正态收益下的闭式解 w=mu/(a sigma²)，理由只是「收益近似正态」。这个近似在尾部会同时做两件坏事：把确定性等价（CE）算高，从而让仓位看起来比真实更有吸引力；把最优权重算高，从而在 crash 状态里承担超出风险预算的暴露。

本文用一组可复现的三正态混合收益把误差量化。混合分量为 92% 的 N(0.15%, 0.40%)、5% 的 N(-0.30%, 1.20%)、3% 的 N(-1.50%, 4.50%)。真实均值 0.000780、标准差 0.00955676，偏度 -3.4898、超额峰度 52.8859。取 a=10，正态闭式解给出 0.854031 的风险资产权重；直接最大化混合分布下的 CE，最优权重只有 0.73，差 0.124031。

更重要的是误差的方向。正态公式在 0.854031 上报出 CE 为 0.00033307，而同一个权重在真实混合分布下的 CE 只有 0.00029031，高估 0.00004276，按 252 天折成年化约 107.76bp。把权重从正态解换到鞍点解，日 CE 只提升 0.00001017，年化 25.618bp。两个数字差别很大，因为它们回答的问题不同：前者是估值口径错误，后者是决策修正收益。

![三正态混合收益密度](/images/exponential-utility-saddlepoint-portfolio/fig1_mixture_density.png)

## 二、确定性等价就是负阶累积量

CARA 效用写成 $u(x)=-\exp(-ax)/a$，其中 a>0。确定性等价 CE 定义为与不确定收益具有相同效用的确定金额：

$$
u(CE)=E[u(X)],\quad CE=-\frac{1}{a}\ln E[\exp(-aX)].
$$

令 $K(t)=\ln E[\exp(tX)]$ 为累积母函数，就有 $CE=-K(-a)/a$。这一步是鞍点方法的入口：CE 不需要求完整分布，只需要在 t=-a 处读出累积母函数。正态世界中 $K(t)=\mu t+\sigma^2 t^2/2$，代回得到 $CE=w\mu-a w^2\sigma^2/2$，一阶条件正是 $w=\mu/(a\sigma^2)$。也就是说，熟悉的均值方差需求只是 CARA 在正态假设下的一个特例，不是指数效用本身的要求。

对三正态混合，累积母函数可以解析写出：

$$
K_w(t)=\ln\sum_k p_k\exp\left(tw\mu_k+\frac{t^2w^2\sigma_k^2}{2}\right).
$$

把 t=-a 代入并对 w 做一维网格搜索，就得到不依赖正态近似的 CE 与最优权重。本文称这条路径为鞍点 CE；当混合参数已知时它是精确的，当参数和分布需要从样本估计时，它变成经验累积母函数的插入式估计。

实现时要先做指数项的中心化。当 a=10、w 接近 1 时，崩盘分量的指数参数会落到十几个单位，直接求 exp 虽然有定义，但不同分量之间相差多个数量级，浮点求和会丢掉小分量。代码里先减去各分量指数项的最大值，算完加权和再把最大值加回对数，结果与直接计算在数学上完全相同，却能稳定扫到 a=40。任何 CARA 实现都应该先过这一关，否则所谓的最优权重可能只是溢出保护的产物。

## 三、正态公式高估的是哪一块

只看权重差 0.124031，容易以为误差不大。真正的问题在 CE 曲线：正态 CE 是开口向下的抛物线，顶点在 0.854031；混合分布的 CE 在左侧更平缓、在右侧被 3% 的崩盘分量迅速压低，顶点移到 0.73。两条曲线在 0.73 处的相差约 0.00002557，在 0.854031 处扩大到 0.00004276。

![正态 CE 与混合分布 CE 的对比](/images/exponential-utility-saddlepoint-portfolio/fig2_ce_curves.png)

崩盘分量只占 3%，为什么能把 CE 拉低 10% 以上？因为 CARA 的期望里含有 $\exp(-aX)$，它对负尾的权重远大于对均值的权重。在 w=0.854031 时，崩盘状态下的组合损失乘以 a 后进入指数函数，贡献了不成比例的负效用；正态分布把这部分概率质量摊到两侧，等价于假设极端亏损更温和、更对称。偏度 -3.4898 与超额峰度 52.8859 就是这种不对性的数值画像。

本文用 30 万条独立抽样做蒙特卡洛侧验：鞍点最优权重处的解析 CE 是 0.00030048，蒙特卡洛给出 0.0003205，相差约 1.3 个蒙特卡洛标准误，没有显著方向偏差。累积母函数在 t 从 -15 到 0 的 13 个格点上，蒙特卡洛与解析值的最大绝对差为 0.000397。验算的意义不是让蒙特卡洛取代公式，而是确认数值稳定化没有在指数项上溢出或丢项。

## 四、风险厌恶越高，差距不一定越大

把 a 从 2 扫到 40，低风险厌恶端两个解都被仓位上界 1.5 截住，差距消失：a=2 时正态与鞍点都是 1.5。a=5 时是 1.5 对 1.46，差距 0.04。a=10 时差距扩大到 0.124031；a=20 时 0.427015 对 0.365；a=40 时 0.213508 对 0.18，绝对差又缩到 0.0335。

![最优权重随风险厌恶的变化](/images/exponential-utility-saddlepoint-portfolio/fig3_weights_vs_aversion.png)

这个非单调形状说明「风险厌恶越大，正态近似越危险」并不成立。真正决定误差的是权重所在位置与尾部的相互作用：仓位接近上界时，两个方法都撞到同一个约束；仓位很小时，尾部损失被 w 缩放后不足以主导 CE；只有中间区间才让分布形状充分发声。因此报告模型风险时要同时说明 a 和约束边界，单独给一个系数值没有意义。

网格步长取 0.005，上界设为 1.5，相当于允许最多 50% 的杠杆。若把上界放宽，a=2 时两边都会继续向右移动，结论会不会保持需要重新算；若把步长放大到 0.05，0.124031 的权重差会被四舍五入成 0.1，年化后悔也会被抹掉一小部分。本文把所有搜参设置写入结果文件，就是为了避免读者把数值精度误当成模型结论。

## 五、用样本估计时，鞍点不是免费午餐

已知分布时鞍点解占优，但实战中连混合分量都要估。本文做 240 次重复试验，分别用 60、250、1000 天样本估计两种需求，再放到真实分布上计算 CE 后悔（真实最优 CE 减去该权重下的真实 CE）。

样本量 1000 时，正态插件的平均后悔 0.00012554，经验鞍点为 0.00008474，鞍点胜出比例 52.92%；样本量 250 时是 0.00025195 对 0.00022257，胜率降到 32.5%；样本量 60 时两者几乎打平（0.0003334 对 0.00032877），胜率只有 13.33%。平均估计权重也显示同样趋势：1000 天时正态给出 0.888614、经验鞍点 0.778438，后者更接近真值 0.73；60 天时两者都在 0.95 附近，尾部样本太少，经验累积母函数没有信息优势。

![有限样本估计权重与后悔](/images/exponential-utility-saddlepoint-portfolio/fig4_finite_sample.png)

这条边界比理论结论更值得记住：经验 CE 的估计依赖 $\exp(-awr)$ 的样本均值，若样本里没有出现崩盘状态，它甚至不知道崩盘存在。小样本下改用结构化的混合模型或加入尾部先验，往往比直接套经验累积母函数更稳。鞍点的价值在于提供正确的目标函数，不保证有限样本估计自动变好。

## 六、落地检查清单

第一，任何 CARA 仓位系统都要同时输出 CE 的两种口径：正态公式值和尾部模型值；两者相差超过风险预算时，直接拒绝正态配置。第二，最优权重必须与风险厌恶系数、仓位上下界一起报告，避免在约束边界上误读模型差异。第三，若使用历史数据估计，先数一数样本里有多少天落在极端状态，再决定用经验累积母函数还是参数化混合分布。

需要说明边界。本文的混合参数、风险厌恶系数与样本量都是为揭示机制而设，不是市场标定结果；没有考虑交易成本、杠杆融资约束与多资产相关结构；CE 是效用口径的收益，不等于可实现年化收益，不能直接和夏普比率比较。它回答的是「给定效用函数，哪套权重更少违背风险偏好」，仅此而已。

另一个容易被忽略的口径是无风险利率。本文把 R 定义为超额收益，现金部分效用为线性项，不参与最优权重的一阶条件；若改用总财富效用并引入随机利率，CARA 的可加性会被打破，需要联立求解现金与风险资产。读者把这套代码接到真实账户前，应当先确认自己 feed 进来的是超额收益还是总收益。

![累积母函数与蒙特卡洛验算](/images/exponential-utility-saddlepoint-portfolio/fig5_saddlepoint_check.png)

## 七、完整可运行代码

下面的代码生成全部数据、五张图与 results.json，可独立运行。

```python
# -*- coding: utf-8 -*-
"""CARA portfolio certainty equivalents: normal plug-in versus mixture saddlepoint."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261129
A = 10.0
P = np.array([0.92, 0.05, 0.03])
MU = np.array([0.0015, -0.0030, -0.0150])
SIG = np.array([0.0040, 0.0120, 0.0450])
N_MC = 300000
N_GRID = 301
TRIALS = 240
SIZES = (60, 250, 1000)
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "exponential-utility-saddlepoint-portfolio")


def mixture_moments():
    m = float(P @ MU)
    e2 = float(P @ (SIG ** 2 + MU ** 2))
    e3 = float(P @ (3 * MU * SIG ** 2 + MU ** 3))
    e4 = float(P @ (3 * SIG ** 4 + 6 * MU ** 2 * SIG ** 2 + MU ** 4))
    var = e2 - m * m
    m3 = e3 - 3 * m * e2 + 2 * m ** 3
    m4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m ** 4
    return m, var, m3 / var ** 1.5, m4 / var ** 2 - 3.0


def cgf(w, t):
    z = t * w * MU + 0.5 * (t * w * SIG) ** 2
    zmax = z.max()
    z -= zmax
    return float(np.log(P @ np.exp(z)) + zmax)


def ce_saddle(w, a=A):
    return -cgf(w, -a) / a


def ce_normal(w, m, var, a=A):
    return w * m - 0.5 * a * w * w * var


def ce_empirical(w, sample, a=A):
    z = -a * w * sample
    zmax = z.max()
    return -float(np.log(np.mean(np.exp(z - zmax))) + zmax) / a


def sample_draws(rng, n):
    k = rng.choice(len(P), size=n, p=P)
    return MU[k] + SIG[k] * rng.standard_normal(n)


def best_weight(ce_fn, grid):
    vals = np.array([ce_fn(float(w)) for w in grid])
    return float(grid[int(np.argmax(vals))]), float(vals.max())


def main():
    os.makedirs(IMG, exist_ok=True)
    m, var, skew, kurt = mixture_moments()
    rng = np.random.default_rng(SEED)
    draws = sample_draws(rng, N_MC)
    fine = np.linspace(0.0, 1.5, N_GRID)

    w_norm_true = float(np.clip(m / (A * var), 0.0, 1.5))
    w_sad, ce_sad = best_weight(lambda w: ce_saddle(w), fine)
    ce_norm_best = ce_normal(w_norm_true, m, var)
    ce_norm_at_sad = ce_normal(w_sad, m, var)
    ce_sad_at_norm = ce_saddle(w_norm_true)
    ce_mc_sad = -float(np.log(np.mean(np.exp(-A * w_sad * draws)))) / A
    ce_mc_norm = -float(np.log(np.mean(np.exp(-A * w_norm_true * draws)))) / A
    mc_se = float(np.std(np.exp(-A * w_sad * draws)) /
                  (np.sqrt(N_MC) * np.mean(np.exp(-A * w_sad * draws))))

    a_grid = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
    a_sweep = {}
    for a in a_grid:
        wn = float(np.clip(m / (a * var), 0.0, 1.5))
        ws, _ = best_weight(lambda w, aa=a: ce_saddle(w, aa), fine)
        a_sweep[str(a)] = {"w_normal": round(wn, 6), "w_saddle": round(ws, 6),
                           "ce_normal": round(ce_normal(wn, m, var, a), 8),
                           "ce_saddle": round(ce_saddle(ws, a), 8)}

    coarse = np.linspace(0.0, 1.5, 61)
    estimation = {}
    for n in SIZES:
        trial_rng = np.random.default_rng(SEED + n)
        wn_list, ws_list, reg_n, reg_s = [], [], [], []
        wins = 0
        for _ in range(TRIALS):
            s = sample_draws(trial_rng, n)
            mh = float(s.mean())
            vh = float(s.var(ddof=1))
            whn = float(np.clip(mh / (A * vh), 0.0, 1.5))
            whs, _ = best_weight(lambda w, ss=s: ce_empirical(w, ss), coarse)
            wn_list.append(whn)
            ws_list.append(whs)
            rn = ce_sad - ce_saddle(whn)
            rs = ce_sad - ce_saddle(whs)
            reg_n.append(rn)
            reg_s.append(rs)
            wins += int(rs < rn)
        estimation[str(n)] = {
            "mean_w_normal": round(float(np.mean(wn_list)), 6),
            "mean_w_saddle": round(float(np.mean(ws_list)), 6),
            "mean_regret_normal": round(float(np.mean(reg_n)), 8),
            "mean_regret_saddle": round(float(np.mean(reg_s)), 8),
            "median_regret_normal": round(float(np.median(reg_n)), 8),
            "median_regret_saddle": round(float(np.median(reg_s)), 8),
            "saddle_win_rate": round(wins / TRIALS, 6),
        }

    t_grid = np.linspace(-A * 1.5, 0.0, 13)
    k_exact = np.array([cgf(w_sad, float(t)) for t in t_grid])
    k_mc = np.array([np.log(np.mean(np.exp(t * w_sad * draws))) for t in t_grid])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(draws, bins=180, density=True, color="#b9c8dc", edgecolor="none")
    xs = np.linspace(-0.08, 0.04, 600)
    for i in range(len(P)):
        ax.plot(xs, P[i] / (SIG[i] * np.sqrt(2 * np.pi)) *
                np.exp(-0.5 * ((xs - MU[i]) / SIG[i]) ** 2),
                lw=1.5, label="component " + str(i + 1))
    ax.axvline(m, color="#c0392b", ls="--", lw=1.2, label="mixture mean")
    ax.set_title("Three-normal mixture of daily excess returns")
    ax.set_xlabel("daily excess return")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_mixture_density.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(fine, [ce_normal(float(w), m, var) for w in fine],
            lw=1.6, color="#3b5b8c", label="normal plug-in CE")
    ax.plot(fine, [ce_saddle(float(w)) for w in fine],
            lw=1.6, color="#c0392b", label="mixture saddlepoint CE")
    ax.plot(fine, [-np.log(np.mean(np.exp(-A * w * draws))) / A for w in fine],
            lw=1.0, color="#7f8c8d", label="Monte Carlo CE")
    ax.axvline(w_norm_true, color="#3b5b8c", ls="--", lw=1.0)
    ax.axvline(w_sad, color="#c0392b", ls="--", lw=1.0)
    ax.set_title("Certainty equivalent versus portfolio weight")
    ax.set_xlabel("weight on risky asset")
    ax.set_ylabel("daily CE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_ce_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(a_grid, [a_sweep[str(a)]["w_normal"] for a in a_grid], "o-",
            color="#3b5b8c", label="normal weight")
    ax.plot(a_grid, [a_sweep[str(a)]["w_saddle"] for a in a_grid], "o-",
            color="#c0392b", label="saddlepoint weight")
    ax.fill_between(a_grid, [a_sweep[str(a)]["w_normal"] for a in a_grid],
                    [a_sweep[str(a)]["w_saddle"] for a in a_grid], alpha=0.15, color="#c0392b")
    ax.set_title("Optimal weight versus risk aversion")
    ax.set_xlabel("CARA coefficient a")
    ax.set_ylabel("weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_weights_vs_aversion.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ns = [str(n) for n in SIZES]
    axes[0].plot(ns, [estimation[n]["mean_w_normal"] for n in ns], "o-",
                 color="#3b5b8c", label="normal plug-in")
    axes[0].plot(ns, [estimation[n]["mean_w_saddle"] for n in ns], "o-",
                 color="#c0392b", label="empirical CE")
    axes[0].axhline(w_sad, ls="--", lw=1.0, color="0.35", label="true optimum")
    axes[0].set_title("Mean estimated weight")
    axes[0].set_xlabel("sample size")
    axes[0].set_ylabel("weight")
    axes[0].legend()
    axes[1].bar(np.arange(3) - 0.18, [estimation[n]["mean_regret_normal"] for n in ns],
                width=0.36, color="#3b5b8c", label="normal")
    axes[1].bar(np.arange(3) + 0.18, [estimation[n]["mean_regret_saddle"] for n in ns],
                width=0.36, color="#c0392b", label="saddlepoint")
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(ns)
    axes[1].set_title("Out-of-sample CE regret")
    axes[1].set_xlabel("sample size")
    axes[1].set_ylabel("daily CE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_finite_sample.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_grid, k_exact, "o-", color="#c0392b", label="analytic CGF")
    axes[0].plot(t_grid, k_mc, "s--", color="#7f8c8d", label="MC CGF")
    axes[0].set_title("Cumulant generating function check")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("K(t)")
    axes[0].legend()
    axes[1].plot(t_grid, k_mc - k_exact, "o-", color="#2c6e7f")
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_title("MC minus analytic CGF")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("error")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_saddlepoint_check.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "a": A,
        "mix_p": P.tolist(),
        "mix_mu": MU.tolist(),
        "mix_sigma": SIG.tolist(),
        "n_mc": N_MC,
        "trials": TRIALS,
        "true_mean": round(m, 8),
        "true_var": round(var, 10),
        "true_sd": round(float(np.sqrt(var)), 8),
        "skewness": round(float(skew), 6),
        "excess_kurtosis": round(float(kurt), 6),
        "w_normal": round(w_norm_true, 6),
        "w_saddle": round(w_sad, 6),
        "ce_normal_best": round(ce_norm_best, 8),
        "ce_normal_at_saddle": round(ce_norm_at_sad, 8),
        "ce_saddle_at_normal": round(ce_sad_at_norm, 8),
        "ce_saddle_best": round(ce_sad, 8),
        "ce_mc_at_saddle": round(ce_mc_sad, 8),
        "ce_mc_at_normal": round(ce_mc_norm, 8),
        "mc_relative_se": round(mc_se, 8),
        "weight_gap": round(w_norm_true - w_sad, 6),
        "ce_regret_of_normal": round(ce_sad - ce_sad_at_norm, 8),
        "ce_regret_annual_bp": round((ce_sad - ce_sad_at_norm) * 252 * 10000, 3),
        "a_sweep": a_sweep,
        "estimation": estimation,
        "cgf_max_abs_error": round(float(np.max(np.abs(k_mc - k_exact))), 6),
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```

