---
title: "回测不显著，是因子没用还是样本不够？最小可检测 IC 的现场计算"
description: "一个真 IC=0.02 的因子，250 天、每天 50 只股票的回测只有 0.5933 的概率把它检出来——40.67% 的回测会误判它无效。本文在真值已知的横截面上现场对拍功效曲线：解析式给出的 80% 功效最小可检测 IC 是 0.025399，蒙特卡洛插值给 0.025939，两者差 0.00054；加天数与加股票都按 T×N 的平方根摊薄门槛，T 从 60 加到 1000 天把 MDE 从 0.052426 压到 0.012667。三盆冷水：挖 1000 个因子时 Bonferroni 把门槛抬到 0.044899（1.7677 倍）、真 IC=0.04 的检出率从 0.9933 掉到 0.6073；条件于显著的 IC 点估计被抬高 1.2961 倍而独立样本外仍是 0.019545；重叠持有期让 IC 序列出现 0.6029 的一阶自相关，朴素 t 检验假阳性从 0.048 涨到 0.338，Newey-West HAC 修到 0.0893。厚尾不降反升功效（t(3) 0.7013），但 IC 均值被抬到 0.022034、检验 size 掉到 0.0362，拒绝率更高不等于检验更好。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-12T09:10:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 因子研究
  - Python
language: zh
difficulty: advanced
---

## 一、把「p 值大于 0.05」翻译成一个可计算的量

因子回测报告里最常见的一句结论是：Rank-IC 均值 0.02，t 值 1.4，不显著，因子无效。这句话把两件完全不同的事混在了一起——「没有证据表明有效」和「有证据表明无效」。区分它们的量叫统计功效：当因子确实有效时，这套检验有多大概率把它检出来。功效低的时候，不显著是默认结果，跟因子质量无关。

本站此前几篇讨论的是另一侧的误差。`anytime-valid-backtest-e-process` 讲的是反复偷看 p 值如何把 5% 的假阳性偷成 42%，`false-discovery-rate-fdr` 与 `deflated-sharpe-ratio` 讲的是多因子筛选后的假阳性折扣，`bootstrap-ic-confidence-interval` 给 IC 配了置信区间。这些都是 I 类错误。本文补上 II 类错误这一半：把「样本量够不够」变成一个能算出来的数——最小可检测效应（minimum detectable effect，MDE），也就是达到 80% 功效所需的最小真实 IC。

## 二、真值已知的横截面，以及一条 2.21 倍标准差的位移

设定一个完全可控的生成过程：每个交易日 $t$ 取 $N=50$ 只股票，标准化因子暴露 $f_{i,t}\sim N(0,1)$，收益 $r_{i,t}=\rho f_{i,t}+\sqrt{1-\rho^2}\,z_{i,t}$，$z_{i,t}\sim N(0,1)$ 且与 $f$ 独立。于是横截面的总体 Pearson 相关恰好等于 $\rho$，这就是真 IC。每日观测到的 $\widehat{IC}_t$ 是样本相关，在 $H_0:\rho=0$ 下抽样标准差约 $1/\sqrt{N-1}=0.142857$。

回测 $T=250$ 天，得到 250 个日度 IC，用单样本 t 检验判断均值是否为零，双侧水平 5%，临界值 $t_{0.975,249}=1.969537$。检验统计量可写成 $\widehat{\rho}/(\sigma_{IC}/\sqrt{T})$，其非中心量 $\delta=\rho\sqrt{T(N-1)}$。真 IC=0.02 时 $\delta=0.02\times\sqrt{250\times49}=2.2136$，只比临界值高 0.244 个标准差——功效因此只有 $\Phi(0.244)\approx0.5964$。

图 1 把这件事画在同一把标准尺上：日度 IC 除以其理论标准差、250 天均值 IC 除以其自身理论标准差，两条直方图的宽度几乎一样，后者的中心右移 2.21 个单位。信号自始至终都在，只是被 0.14 量级的日度噪声压着。

![代表性 IC 路径与标准化后的日度 IC、均值 IC 分布](/images/statistical-power-minimum-detectable-ic/fig1_ic_paths_and_noise.png)

## 三、功效曲线与 MDE：0.025399

对 12 档真 IC，用同一批 $(f,z)$ 随机数（共同随机数，避免曲线被抽样噪声污染）跑 1500 条独立回测，统计拒绝率。实测功效从 $\rho=0$ 的 0.0493（名义 5%，验证检验本身没坏）上升到 0.005 的 0.0900、0.01 的 0.1953、0.02 的 0.5933、0.03 的 0.8980，0.04 时 0.9933。解析正态近似给出的对应值是 0.0489、0.0842、0.1952、0.5964、0.9116、0.9930，全档最大偏差 0.014——解析式不是拟合出来的，是独立算出来的，两者对上了才敢用它做外推。

把 80% 功效代入反解，$\mathrm{MDE}=(t_{0.975,249}+z_{0.80})/\sqrt{T(N-1)}=(1.969537+0.841621)/110.68=0.025399$。蒙特卡洛曲线线性插值给 0.025939，差 0.00054。换句话说：**一年日频、50 只股票的回测，真实 IC 低于 0.0254 时你连八成把握都没有**。反过来，一个业内公认不错、真 IC=0.02 的因子，有 40.67% 的概率被这套流程判成无效。

![功效曲线的解析式与蒙特卡洛对拍，及 80% 功效对应的 MDE](/images/statistical-power-minimum-detectable-ic/fig2_power_curve_mde.png)

## 四、加天数还是加股票：决定门槛的是 T×N

MDE 的分母是 $\sqrt{T(N-1)}$，所以延长回测与拓宽横截面在数学上等价，都以平方根速度摊薄门槛。$T=60$ 天时 MDE 是 0.052426，125 天 0.036044，250 天 0.025399，500 天 0.017929，1000 天 0.012667；固定 $T=250$ 而把 $N$ 从 25 加到 200，MDE 从 0.036292 降到 0.012603。图 3 里每个 MC 锚点都单独重跑了功效曲线再插值，与解析式最大差 0.003036，主要来自 $N=200$ 时网格步长偏粗。

这个「等价」有重要的实务含义：把回测从一年拉到四年听起来免费，但四年里的区制变化、因子衰减、成分股调整都在破坏「同一个 $\rho$」这个前提；而把横截面从 50 扩到 200 只，代价是纳入流动性差、噪声大的小票，$\rho$ 本身可能被稀释。公式里两者对称，现实里成本结构完全不同。另外注意 T 与 N 的乘积决定有效信息量，这也是为什么日内或横截面高频数据的 IC 检验看起来总是「显著」——不是因子更强，是 $T\times N$ 大得多。

![MDE 随回测长度与横截面宽度的平方根缩放，附 MC 锚点](/images/statistical-power-minimum-detectable-ic/fig3_mde_scaling_T_N.png)

## 五、挖 1000 个因子，门槛被抬到 0.044899

真实研究不会只检验一个因子。设一共试 $M$ 个候选、其中只有 1 个真有信号（真 IC=0.04），其余全是噪声。Bonferroni 把单次水平压到 $0.05/M$，$M=100$ 时 MDE 从 0.025399 涨到 0.039472，$M=1000$ 时 0.044899，是单次检验的 1.7677 倍。对应的检出率：真 IC=0.04 在 $M=1$ 时有 0.9933 的把握被抓到，$M=100$ 降到 0.7993，$M=1000$ 只剩 0.6073。

一个值得注意的细节：在「只有 1 个真信号」的设定下，BH-FDR 的检出率几乎与 Bonferroni 重合（$M=1000$ 时同为 0.6073），FDR 控制的放宽没有转化为功效。原因是 BH 的阈值依赖排序后 p 值的整体分布，稀疏真信号时它退化成接近 Bonferroni 的严格门槛。这与 `false-discovery-rate-fdr` 里「1000 个因子中 95 个显著」的密集设定不同——**功效不是校正方法的属性，是「试了多少个」和「真信号有多稀疏」的联合属性**。

![多重检验下 Bonferroni 的 MDE 膨胀、检出率与 FWER](/images/statistical-power-minimum-detectable-ic/fig4_multiple_testing.png)

## 六、三盆冷水

**第一盆：显著本身就是一个有偏的筛选器。** 真 IC=0.02 时，全部 1500 次回测的 $\widehat{\rho}$ 均值 0.019782（无偏），但只看 p<0.05 的那 59.33%，均值变成 0.025640，抬高 1.2961 倍。用一批完全独立的新数据重新估计，全部样本 0.019502、显著组 0.019545，都贴近真值 0.02——高估纯粹来自条件化，不是这些回测运气好。由此还能拆穿事后功效：把观测到的 $\widehat{\rho}$ 代回功效公式，显著组平均 0.7653、不显著组 0.2695，而真功效恒为 0.5933。拿「不显著组事后功效低」论证因子无效，是拿结果解释结果的循环论证。

**第二盆：重叠持有期会让 IC 序列相关，朴素 t 检验直接失效。** 若「下期收益」实际是 5 日累计收益而每天采样一次，相邻两天的 IC 共享 4 天收益，IC 序列自带 AR(1) 依赖。这里用前面横截面实测的边际标准差 0.143006 校准，只改变依赖结构：一阶自相关 0.6029 时，长期方差是 iid 假设的 $(1+\phi)/(1-\phi)=4$ 倍，朴素 t 检验的假阳性从 0.048 依次涨到 0.108、0.2013、0.338。换成 Newey-West HAC（Bartlett 核、10 阶滞后）修到 0.0673、0.0700、0.0727、0.0893——方向对了但没精确回到 5%，有限样本下 HAC 自身也有偏差，这一点不该粉饰。它与 `block-bootstrap-coverage-block-length` 是同一个病根的两种表现：序列依赖让朴素标准误系统性偏小。

**第三盆：厚尾不是主犯，而且方向和我们预设的相反。** 把横截面冲击换成单位方差的 t(8)、t(5)、t(3)，日度 IC 标准差几乎不变（0.142661 到 0.142670），功效却从正态的 0.6100 走到 0.5850、0.6088、0.7013，t(3) 反而高 0.0913。原因不是检验变强：t(3) 下 IC 均值被抬到 0.022034（正态 0.020267），少数极端股票主导了样本 Pearson 相关，同时检验 size 掉到 0.0362。**拒绝率更高、区间却更不可信**——这提醒我们别把厚尾下的「更容易显著」当成好消息。分布形状真正致命的地方在推断的另一侧，也就是本批姊妹篇里高斯 QMLE 的标准误。

![三盆冷水：winner's curse、序列依赖下的 size 膨胀、厚尾下不降反升的功效](/images/statistical-power-minimum-detectable-ic/fig5_caveats.png)

## 七、完整可运行代码

下面这段代码就是本文全部数字的来源：它自己生成横截面数据、算功效、跑所有扫描、画出五张图并写出 results.json。直接运行即可复现，无需任何外部数据文件。

```python
"""Statistical power of a factor IC backtest and the minimum detectable IC.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_stat_power_20261112.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
Every number quoted in the article comes from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as student_t

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261112
T = 250              # trading days in one backtest year
N = 50               # names in the cross-section
ALPHA = 0.05         # two-sided test level
TARGET = 0.80        # desired power
R_MAIN = 1500        # independent backtest replications
R_TAIL = 800
R_N = 600

Z975 = float(norm.ppf(1 - ALPHA / 2))
Z80 = float(norm.ppf(TARGET))
SIGMA_IC = 1.0 / np.sqrt(N - 1)          # sampling sd of one daily Pearson IC under H0


def crit_t(df, level=ALPHA):
    return float(student_t.ppf(1 - level / 2, df))


MDE_ANALYTIC = (crit_t(T - 1) + Z80) * SIGMA_IC / np.sqrt(T)


# ---------- cross-sectional DGP with a known true IC ----------
def make_pool(rng, R, T_, N_, dtype=np.float32):
    """Independent standardised factor scores and idiosyncratic shocks."""
    f = rng.standard_normal((R, T_, N_), dtype=dtype)
    z = rng.standard_normal((R, T_, N_), dtype=dtype)
    return f, z


def ic_sequences(f, z, rho, phi=0.0, sigma_eta=0.0, seed=None):
    """Daily Pearson IC from r_i = rho_t * f_i + sqrt(1-rho_t^2) * z_i.

    rho is the true cross-sectional IC (constant unless phi>0, in which case
    the true IC itself follows a zero-mean AR(1) around rho). Returns (R, T).
    """
    R, T_, N_ = f.shape
    if phi > 0.0:
        rg = np.random.default_rng(seed)
        eta = rg.normal(0.0, sigma_eta, size=(R, T_)).astype(np.float32)
        cur = np.zeros(R, dtype=np.float32)
        path = np.empty((R, T_), dtype=np.float32)
        for t in range(T_):
            cur = phi * cur + eta[:, t]
            path[:, t] = rho + cur
        rho_t = np.clip(path[..., None], -0.95, 0.95).astype(np.float32)
    else:
        rho_t = np.float32(rho)
    r = rho_t * f + np.sqrt(1.0 - rho_t ** 2) * z
    fc = f - f.mean(axis=-1, keepdims=True)
    rc = r - r.mean(axis=-1, keepdims=True)
    num = (fc * rc).sum(axis=-1)
    den = np.sqrt((fc * fc).sum(axis=-1) * (rc * rc).sum(axis=-1))
    return (num / den).astype(np.float64)


def tstat(ic):
    """Two-sided one-sample t statistic for mean(IC)=0, per replication."""
    m = ic.mean(axis=1)
    s = ic.std(axis=1, ddof=1)
    return m / (s / np.sqrt(ic.shape[1]))


def tstat_hac(ic, max_lags=10):
    """Same t statistic but with a Newey-West (Bartlett) long-run variance.

    Under AR(1) dependence the naive s^2/T underestimates Var(mean(IC));
    the HAC estimate restores the correct size.
    """
    m = ic.mean(axis=1)
    x = ic - m[:, None]
    T_ = ic.shape[1]
    g0 = (x * x).sum(axis=1) / T_
    lr = g0.copy()
    for lag in range(1, max_lags + 1):
        w = 1.0 - lag / (max_lags + 1.0)
        gl = (x[:, lag:] * x[:, :-lag]).sum(axis=1) / T_
        lr += 2.0 * w * gl
    lr = np.maximum(lr, g0 * 1e-6)
    return m / np.sqrt(lr / T_)


def power_at(ic, thresh):
    return float(np.mean(np.abs(tstat(ic)) > thresh))


def analytic_power(rho, T_, N_, thresh=None):
    # Normal approximation to the non-central t, evaluated at the SAME
    # critical value used by the Monte Carlo test.
    c = crit_t(T_ - 1) if thresh is None else thresh
    delta = rho * np.sqrt(T_ * (N_ - 1))
    return float(norm.cdf(delta - c) + norm.cdf(-delta - c))


def mde_analytic(T_, N_, level=ALPHA):
    return (crit_t(T_ - 1, level) + Z80) / np.sqrt(T_ * (N_ - 1))


def mde_mc(rhos, powers):
    """Linearly interpolate the true IC at which empirical power hits TARGET."""
    for i in range(len(rhos) - 1):
        if powers[i] <= TARGET <= powers[i + 1]:
            w = (TARGET - powers[i]) / (powers[i + 1] - powers[i])
            return float(rhos[i] + w * (rhos[i + 1] - rhos[i]))
    return float("nan")


def run_all():
    rng = np.random.default_rng(SEED)
    stats = {}
    figs = {}

    stats["design"] = {
        "T": T, "N": N, "alpha": ALPHA, "target_power": TARGET,
        "R_main": R_MAIN, "R_tail": R_TAIL, "R_n": R_N, "seed": SEED,
        "z_975": round(Z975, 6), "z_80": round(Z80, 6),
        "sigma_ic_theory": round(float(SIGMA_IC), 6),
    }

    # ---- common random numbers: one (f,z) pool reused by every rho ----
    f, z = make_pool(rng, R_MAIN, T, N)
    crit = crit_t(T - 1)
    stats["design"]["t_crit_249df"] = round(float(crit), 6)

    # ---- Figure 1 + Figure 2: representative paths, power curve, MDE ----
    rhos = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035,
            0.04, 0.05, 0.06, 0.08]
    mc_powers, an_powers = [], []
    ic0 = ic_sequences(f, z, 0.0)
    for rho in rhos:
        ic = ic0 if rho == 0.0 else ic_sequences(f, z, rho)
        mc_powers.append(power_at(ic, crit))
        an_powers.append(analytic_power(rho, T, N))
        if abs(rho - 0.02) < 1e-12:
            ic_002 = ic
            figs["ic0"] = ic0[0]
            figs["ic002"] = ic[0]
            figs["ic0_all"] = ic0
            figs["tvals0"] = tstat(ic0)
            figs["mean_ic002"] = ic.mean(axis=1)
    figs["rhos"] = rhos
    figs["mc_powers"] = mc_powers
    figs["an_powers"] = an_powers
    figs["t_crit"] = crit

    mde_mc_val = mde_mc(rhos, mc_powers)
    stats["power_curve"] = {
        "rho": rhos,
        "mc_power": [round(v, 4) for v in mc_powers],
        "analytic_power": [round(v, 4) for v in an_powers],
        "max_abs_gap": round(float(np.max(np.abs(np.array(mc_powers) - np.array(an_powers)))), 4),
        "mde_mc": round(mde_mc_val, 6),
        "mde_analytic": round(float(MDE_ANALYTIC), 6),
    }
    stats["headline"] = {
        "size_at_rho0": round(mc_powers[0], 4),
        "power_at_rho_002": round(mc_powers[4], 4),
        "analytic_power_at_rho_002": round(an_powers[4], 4),
        "mde_80": round(float(MDE_ANALYTIC), 6),
        "mde_80_mc": round(mde_mc_val, 6),
        "miss_rate_at_rho_002": round(1 - mc_powers[4], 4),
    }

    # ---- Figure 3: MDE scaling in T and N, analytic curves + MC anchors ----
    T_grid = [60, 125, 250, 500, 1000]
    mde_T = [mde_analytic(t, N) for t in T_grid]
    T_anchor = [60, 125, 250]
    mc_T = []
    for t in T_anchor:
        sub_rhos = np.linspace(0.0, 0.12, 13)
        pw = [power_at(ic_sequences(f[:, :t, :], z[:, :t, :], r),
                       crit_t(t - 1)) for r in sub_rhos]
        mc_T.append(mde_mc(list(sub_rhos), pw))
    rng_n = np.random.default_rng(SEED + 101)
    N_grid = [25, 50, 100, 200]
    mde_N = [mde_analytic(T, n) for n in N_grid]
    mc_N = []
    for n in N_grid:
        fn, zn = make_pool(rng_n, R_N, T, n)
        sub_rhos = np.linspace(0.0, 0.10, 11)
        pw = [power_at(ic_sequences(fn, zn, r), crit) for r in sub_rhos]
        mc_N.append(mde_mc(list(sub_rhos), pw))
    # The analytic MDE is checked against independent MC anchors at every
    # T and N setting; report the largest discrepancy instead of assuming it.
    gap_T = [abs(mde_analytic(t, N) - mc) for t, mc in zip(T_anchor, mc_T)]
    gap_N = [abs(mde_analytic(T, n) - mc) for n, mc in zip(N_grid, mc_N)]
    stats["mde_scaling"] = {
        "T_grid": T_grid, "mde_T": [round(v, 6) for v in mde_T],
        "T_anchor": T_anchor, "mde_T_mc": [round(v, 6) for v in mc_T],
        "N_grid": N_grid, "mde_N": [round(v, 6) for v in mde_N],
        "mde_N_mc": [round(v, 6) for v in mc_N],
        "mde_at_T60_N50": round(mde_analytic(60, 50), 6),
        "analytic_vs_mc_max_gap": round(float(max(max(gap_T), max(gap_N))), 6),
    }
    figs["T_grid"], figs["mde_T"], figs["T_anchor"], figs["mc_T"] = T_grid, mde_T, T_anchor, mc_T
    figs["N_grid"], figs["mde_N"], figs["mc_N"] = N_grid, mde_N, mc_N

    # ---- Figure 4: multiple testing ----
    M_grid = [1, 5, 20, 100, 500, 1000]
    bonf_mde, bonf_fwer, bonf_detect, bh_detect, bh_fdr = [], [], [], [], []
    rg_m = np.random.default_rng(SEED + 202)
    ic_null = ic0                       # (R_main, T) under H0
    ic_true = ic_sequences(f, z, 0.04)  # one genuinely predictive factor
    df_t = T - 1
    p_null = 2.0 * student_t.sf(np.abs(tstat(ic_null)), df_t)
    p_true = 2.0 * student_t.sf(np.abs(tstat(ic_true)), df_t)
    R_m = R_MAIN
    for M in M_grid:
        lvl = ALPHA / M
        bonf_mde.append(mde_analytic(T, N, lvl))
        # one true factor + (M-1) nulls, paired across replications
        idx = rg_m.integers(0, R_m, size=(R_m, max(M - 1, 0)))
        pn = p_null[idx] if M > 1 else np.empty((R_m, 0))
        rej_null = pn < lvl
        bonf_fwer.append(float(np.mean(rej_null.any(axis=1))))
        bonf_detect.append(float(np.mean(p_true < lvl)))
        # Benjamini-Hochberg at FDR level ALPHA on the same family
        p_all = np.concatenate([p_true[:, None], pn], axis=1)
        ps = np.sort(p_all, axis=1)
        k = np.arange(1, M + 1)
        passed = ps <= (k / M) * ALPHA
        kmax = np.where(passed.any(axis=1),
                        np.max(np.where(passed, k, 0), axis=1), 0)
        cutoff = ps[np.arange(R_m), np.maximum(kmax - 1, 0)]
        rej_bh = (p_all <= cutoff[:, None]) & (kmax[:, None] > 0)
        bh_detect.append(float(np.mean(rej_bh[:, 0])))
        n_false = rej_bh[:, 1:].sum(axis=1)
        n_rej = np.maximum(rej_bh.sum(axis=1), 1)
        bh_fdr.append(float(np.mean(np.where(kmax > 0, n_false / n_rej, 0.0))))
    stats["multiple_testing"] = {
        "M": M_grid,
        "bonferroni_alpha": [ALPHA / M for M in M_grid],
        "bonferroni_mde": [round(v, 6) for v in bonf_mde],
        "bonferroni_fwer": [round(v, 4) for v in bonf_fwer],
        "bonferroni_power_rho004": [round(v, 4) for v in bonf_detect],
        "bh_power_rho004": [round(v, 4) for v in bh_detect],
        "bh_false_discovery_prop": [round(v, 4) for v in bh_fdr],
        "mde_inflation_M100": round(bonf_mde[3] / bonf_mde[0], 4),
        "mde_inflation_M1000": round(bonf_mde[5] / bonf_mde[0], 4),
    }
    figs["M_grid"], figs["bonf_mde"] = M_grid, bonf_mde
    figs["bonf_fwer"], figs["bonf_detect"], figs["bh_detect"] = bonf_fwer, bonf_detect, bh_detect

    # ---- Figure 5a: winner's curse + post-hoc power (rho_true = 0.02) ----
    hat = figs["mean_ic002"]
    sig = np.abs(tstat(ic_002)) > crit
    # A genuinely independent out-of-sample panel, same true IC, fresh shocks.
    rng_oos = np.random.default_rng(SEED + 303)
    f_oos, z_oos = make_pool(rng_oos, R_MAIN, T, N)
    oos = ic_sequences(f_oos, z_oos, 0.02).mean(axis=1)
    posthoc = norm.cdf(hat * np.sqrt(T * (N - 1)) - Z975)
    stats["winners_curse"] = {
        "rho_true": 0.02,
        "unconditional_mean_hat": round(float(hat.mean()), 6),
        "conditional_mean_hat_given_significant": round(float(hat[sig].mean()), 6),
        "inflation_factor": round(float(hat[sig].mean() / hat.mean()), 4),
        "significant_fraction": round(float(sig.mean()), 4),
        "oos_mean_ic_all": round(float(oos.mean()), 6),
        "oos_mean_ic_significant": round(float(oos[sig].mean()), 6),
        "posthoc_power_all": round(float(posthoc.mean()), 4),
        "posthoc_power_given_significant": round(float(posthoc[sig].mean()), 4),
        "posthoc_power_given_nonsignificant": round(float(posthoc[~sig].mean()), 4),
        "true_power": round(mc_powers[4], 4),
    }
    figs["hat"], figs["sig"], figs["posthoc"] = hat, sig, posthoc

    # ---- Figure 5b: overlapping horizons make the IC series AR(1) ----
    # Real mechanism: when the "forward return" is an h-day cumulative return
    # measured every day, adjacent IC values share h-1 days of returns. We
    # model the resulting daily IC series directly as a stationary AR(1) whose
    # marginal sd is CALIBRATED to the cross-sectional simulation above (not
    # assumed), so the only thing that changes is the dependence structure.
    sigma_measured = float(ic0.std())
    rg_s = np.random.default_rng(SEED + 404)
    eps_s = rg_s.standard_normal((R_MAIN, T))
    phi_grid = [0.0, 0.2, 0.4, 0.6]
    size_phi, size_phi_hac, ac_measured = [], [], []
    for phi in phi_grid:
        v = np.empty((R_MAIN, T))
        cur = eps_s[:, 0].copy()
        v[:, 0] = cur
        for t in range(1, T):
            cur = phi * cur + np.sqrt(1.0 - phi ** 2) * eps_s[:, t]
            v[:, t] = cur
        ic_p = sigma_measured * v                       # H0: true IC = 0
        size_phi.append(power_at(ic_p, crit))
        size_phi_hac.append(float(np.mean(np.abs(tstat_hac(ic_p)) > crit)))
        ac_measured.append(float(np.corrcoef(ic_p[:, :-1].ravel(),
                                             ic_p[:, 1:].ravel())[0, 1]))
    stats["serial_dependence"] = {
        "phi": phi_grid, "empirical_size": [round(v, 4) for v in size_phi],
        "empirical_size_hac_corrected": [round(v, 4) for v in size_phi_hac],
        "measured_lag1_autocorr": [round(v, 4) for v in ac_measured],
        "marginal_sd_calibrated": round(sigma_measured, 6),
        "nominal": ALPHA,
        "theoretical_variance_inflation_phi06": round((1 + 0.6) / (1 - 0.6), 4),
    }
    figs["phi_grid"], figs["size_phi"], figs["size_phi_hac"] = (
        phi_grid, size_phi, size_phi_hac)

    # ---- Figure 5c: heavy-tailed cross-sectional shocks ----
    nu_grid = [1e9, 8.0, 5.0, 3.0]
    tail_power, tail_sigma, tail_size, tail_mean, tail_kurt = [], [], [], [], []
    for nu in nu_grid:
        rg_t = np.random.default_rng(SEED + int(nu) if nu < 1e8 else SEED + 999)
        if nu > 1e8:
            ft, zt = f[:R_TAIL], z[:R_TAIL]
        else:
            ft = rg_t.standard_normal((R_TAIL, T, N), dtype=np.float32)
            zt = rg_t.standard_t(nu, size=(R_TAIL, T, N)).astype(np.float32)
            zt /= np.sqrt(nu / (nu - 2.0))
        ic_h0 = ic_sequences(ft, zt, 0.0)
        ic_h1 = ic_sequences(ft, zt, 0.02)
        tail_size.append(power_at(ic_h0, crit))
        tail_power.append(power_at(ic_h1, crit))
        tail_sigma.append(float(ic_h0.std()))
        tail_mean.append(float(ic_h1.mean()))
        tail_kurt.append(float(np.mean((ic_h0 / ic_h0.std()) ** 4)))
    stats["heavy_tail"] = {
        "nu": ["normal", "8", "5", "3"],
        "daily_ic_sd": [round(v, 6) for v in tail_sigma],
        "mean_daily_ic_at_true_002": [round(v, 6) for v in tail_mean],
        "excess_kurtosis_daily_ic": [round(v - 3.0, 4) for v in tail_kurt],
        "empirical_size": [round(v, 4) for v in tail_size],
        "power_at_rho_002": [round(v, 4) for v in tail_power],
        "power_change_t3": round(tail_power[3] - tail_power[0], 4),
        "interpretation": "heavy tails do NOT lower power here; they distort "
                          "the IC estimator and the size, so a higher rejection "
                          "rate is not evidence of a better test",
    }
    figs["nu_labels"], figs["tail_power"], figs["tail_sigma"] = (
        ["normal", "t(8)", "t(5)", "t(3)"], tail_power, tail_sigma)

    return stats, figs


# ---------- figures ----------
def make_figures(g):
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    # fig 1: signal buried in noise
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(g["ic0"], lw=0.55, color="#94a3b8", label="true IC = 0")
    ax[0].plot(g["ic002"], lw=0.55, color="#d62728", alpha=0.8, label="true IC = 0.02")
    ax[0].axhline(0, color="#111827", lw=0.8)
    ax[0].set_xlabel("trading day"); ax[0].set_ylabel("daily cross-sectional IC")
    ax[0].set_title("One-year IC paths (N=50 names/day)")
    ax[0].legend(frameon=False, fontsize=8)
    # Both histograms are put on the SAME standardised scale: daily IC is
    # divided by its theoretical sd, and the 250-day mean IC by its own
    # theoretical sd. No visual stretching or "x20" trick.
    bins = np.linspace(-4, 6, 60)
    daily_std = g["ic0_all"].ravel() / SIGMA_IC
    mean_std = g["mean_ic002"] / (SIGMA_IC / np.sqrt(T))
    ax[1].hist(daily_std, bins=bins, density=True, color="#94a3b8",
               alpha=0.75, label="daily IC / sd(IC)")
    ax[1].hist(mean_std, bins=bins, density=True, histtype="step",
               color="#d62728", lw=1.8, label="mean IC / sd(mean IC)")
    xs = np.linspace(-4, 6, 200)
    ax[1].plot(xs, norm.pdf(xs), "--", color="#111827", lw=1.0, label="N(0,1)")
    ax[1].axvline(g["t_crit"], color="#16a34a", ls=":", lw=1.5,
                  label=f"critical t = {g['t_crit']:.3f}")
    ax[1].set_xlabel("standardised units"); ax[1].set_ylabel("density")
    ax[1].set_title("Standardised: signal shifts the mean by 2.21 sd")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig1_ic_paths_and_noise.png"); plt.close(fig)

    # fig 2: power curve + MDE
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    r = np.array(g["rhos"])
    ax.plot(r, g["an_powers"], "-", color="#2563eb", lw=1.8, label="analytic (normal approx.)")
    ax.plot(r, g["mc_powers"], "o", color="#d62728", ms=5, label="Monte Carlo (1500 backtests)")
    ax.axhline(TARGET, color="#111827", ls="--", lw=1.1, label="80% power target")
    ax.axvline(MDE_ANALYTIC, color="#16a34a", ls=":", lw=1.6,
               label=f"MDE = {MDE_ANALYTIC:.4f}")
    ax.axhline(ALPHA, color="#94a3b8", ls=":", lw=1.0)
    ax.set_xlabel("true cross-sectional IC"); ax.set_ylabel("P(reject H0: mean IC = 0)")
    ax.set_title(f"Power of a {T}-day IC t-test (N={N}, two-sided 5%)")
    ax.set_ylim(-0.02, 1.02); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig2_power_curve_mde.png"); plt.close(fig)

    # fig 3: MDE vs T and N
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    ax[0].plot(g["T_grid"], g["mde_T"], "-o", color="#2563eb", ms=4,
               label="analytic  (z.975+z.80)/sqrt(T(N-1))")
    ax[0].plot(g["T_anchor"], g["mc_T"], "s", color="#d62728", ms=6, label="MC anchors")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("backtest length T (days)"); ax[0].set_ylabel("minimum detectable IC")
    ax[0].set_title("Lengthening the backtest (N=50)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(g["N_grid"], g["mde_N"], "-o", color="#2563eb", ms=4, label="analytic")
    ax[1].plot(g["N_grid"], g["mc_N"], "s", color="#d62728", ms=6, label="MC anchors")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("cross-section width N (names/day)"); ax[1].set_ylabel("minimum detectable IC")
    ax[1].set_title("Widening the cross-section (T=250)"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig3_mde_scaling_T_N.png"); plt.close(fig)

    # fig 4: multiple testing
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    M = np.array(g["M_grid"])
    ax[0].plot(M, g["bonf_mde"], "-o", color="#d62728", ms=5, label="Bonferroni MDE")
    ax[0].axhline(g["bonf_mde"][0], color="#2563eb", ls="--", lw=1.2, label="single-test MDE")
    ax[0].set_xscale("log"); ax[0].set_xlabel("number of candidate factors M")
    ax[0].set_ylabel("minimum detectable IC")
    ax[0].set_title("Searching more factors raises the bar")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(M, g["bonf_detect"], "-o", color="#d62728", ms=5, label="Bonferroni power")
    ax[1].plot(M, g["bh_detect"], "-s", color="#16a34a", ms=5, label="BH-FDR power")
    ax[1].plot(M, g["bonf_fwer"], "-^", color="#94a3b8", ms=5, label="Bonferroni FWER")
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of candidate factors M")
    ax[1].set_ylabel("probability"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("Detection of one true factor (true IC = 0.04)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_multiple_testing.png"); plt.close(fig)

    # fig 5: three caveats
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax[0].hist(g["hat"][g["sig"]], bins=28, color="#d62728", alpha=0.8,
               label="significant backtests")
    ax[0].axvline(0.02, color="#111827", ls="--", lw=1.4, label="true IC = 0.02")
    ax[0].axvline(float(g["hat"][g["sig"]].mean()), color="#2563eb", ls=":", lw=1.6,
                  label="conditional mean")
    ax[0].set_xlabel("estimated mean IC"); ax[0].set_ylabel("count")
    ax[0].set_title("Winner's curse: conditioning on p<0.05")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(g["phi_grid"], g["size_phi"], "-o", color="#d62728", ms=5,
               label="naive t-test size")
    ax[1].plot(g["phi_grid"], g["size_phi_hac"], "-s", color="#16a34a", ms=5,
               label="Newey-West HAC size")
    ax[1].axhline(ALPHA, color="#111827", ls="--", lw=1.1, label="nominal 5%")
    ax[1].set_xlabel("AR(1) coefficient of the daily IC series")
    ax[1].set_ylabel("P(reject | H0 true)")
    ax[1].set_title("Overlapping horizons break the naive t-test")
    ax[1].legend(frameon=False, fontsize=7.5)
    x = np.arange(4)
    ax[2].bar(x, g["tail_power"], color=["#2563eb", "#60a5fa", "#f59e0b", "#d62728"])
    ax[2].set_xticks(x); ax[2].set_xticklabels(g["nu_labels"])
    ax[2].axhline(g["tail_power"][0], color="#111827", ls="--", lw=1.0)
    ax[2].set_ylabel("power at true IC = 0.02")
    ax[2].set_title("Heavy tails do NOT lower power (size also drifts)")
    for i, v in enumerate(g["tail_power"]):
        ax[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax[2].set_ylim(0, max(g["tail_power"]) * 1.22)
    fig.tight_layout(); fig.savefig(OUT / "fig5_caveats.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("MDE(80%, analytic) =", round(float(MDE_ANALYTIC), 6))
    print("power at true IC=0.02 =", stats["headline"]["power_at_rho_002"])
    print("Bonferroni MDE inflation M=1000 =",
          stats["multiple_testing"]["mde_inflation_M1000"])
    print("Winner's curse inflation =", stats["winners_curse"]["inflation_factor"])
    print("size under phi=0.6 =", stats["serial_dependence"]["empirical_size"][-1])
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
```

## 八、结论与边界

把结论压成三句话。第一，回测前先算 MDE：本文设定下 0.025399，任何小于它的「不显著」都不构成否定证据。第二，因子库越大、持有期越重叠，门槛越高、朴素 p 值越不可信；Bonferroni 与 HAC 是必要但都不完美的修正。第三，显著样本的 IC 天然被高估，样本外回归真值是常态而不是策略失效。

边界也要说清：这里假设每日 IC 独立同分布、真 IC 恒定、无交易成本、无因子择时，且用 Pearson 相关而非 Rank-IC；真实市场中 IC 时变、因子有行业暴露、收益有横截面相关，有效样本量会比 $\sqrt{T(N-1)}$ 更小。解析功效用正态近似、MC 用 t(249) 临界值，全档最大差 0.014，属于有限样本近似误差而非模型失误。厚尾扫描用 $R=800$、其余用 $R=1500$，覆盖度一类比例的标准误约 0.01 量级，图上的小数第三位不必当真。功效分析必须在看数据之前做——事后再算，就已经掉进第六节那个循环了。
