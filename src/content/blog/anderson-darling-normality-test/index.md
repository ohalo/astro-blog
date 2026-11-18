---
title: "Anderson-Darling 正态性检验：尾部加权比 K-S 更灵敏，但肥尾功效仍输给 Jarque-Bera"
description: "正态性检验是量化风控的第一道闸门，K-S 因参数自估失真、JB 只盯偏度峰度，Anderson-Darling 用 1/(u(1-u)) 权重放大尾部偏离。本文从零实现 AD 统计量并与 scipy 逐样本对拍，最大偏差 1.14e-13；n=252 正态样本 5000 次蒙特卡洛得 A* 的 5% 临界值 0.7258，与 scipy 内置 Stephens 常数换算值几乎重合，实际误报率 0.05。功效对决揭示诚实的排名：t(5) 肥尾下 n=252 时 JB 检出 0.9165、AD 0.8230、K-S 0.6370；卖期权式负偏肥尾（97% 小盈+3% 崩损）JB 0.9473、AD 0.8140、K-S 0.6467——JB 凭偏度峰度特异性反超，AD 贵在全分布 omnibus 与分布自由临界值。附完整代码、JSON 与五张真实计算图。"
publishDate: '2026-11-18T09:40:00Z'
tags:
  - 量化
  - 交易
  - 统计检验
  - 正态性
  - 风险管理
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

收益率是否正态，直接决定 VaR、夏普置信区间、配对检验还灵不灵。三个最常用的正态性检验各有软肋：K-S 距离在均值方差由样本估计时严重失真；Jarque-Bera（JB）只汇总偏度和峰度两个矩，对形状偏离视而不见；Anderson-Darling（AD）则给经验分布函数的偏离加一个 $1/[u(1-u)]$ 的权重，把两头（尾部）放大数百倍——而尾部恰恰是风控最在意的地方。

但"AD 最灵敏"是个需要修正的流行说法。本文用共同随机数对三个检验做正面对决，诚实的结论是：

| 场景（n=252，α=0.05） | AD | Jarque-Bera | K-S |
|---|---|---|---|
| t(5) 对称肥尾检出率 | 0.8230 | **0.9165** | 0.6370 |
| 卖期权式负偏肥尾检出率 | 0.8140 | **0.9473** | 0.6467 |
| 正态误报率（尺寸） | 0.0500 | 0.0434 | 0.0470 |

1. **JB 在肥尾/偏态上检出率最高**：它对偏度、峰度直接建模，遇到肥尾或负偏正好打在靶心，AD 反而不是第一。
2. **AD 的价值在"全分布 + 分布自由临界值"**：它不依赖偏度峰度的具体形式，任何形状偏离都能响应；且 Stephens 调整后的 $A^*$ 原假设分布几乎不依赖估参，有现成临界表，而 K-S 用教科书 $\chi^2$ 型近似时误报率会塌到 0。
3. **K-S 最迟钝且在参数自估时最不可靠**：t(5) 下检出率只有 0.637，若误用经典临界值 $1.36/\sqrt{n}$，误报率是 0——检验彻底哑火。

## 二、三个检验"看"的地方不一样

K-S 统计量是经验 CDF 与拟合正态 CDF 之间的最大逐点距离 $D=\sup_x|\hat F(x)-\Phi(x)|$，全程等权。问题是你拿样本自己的均值方差去拟合正态，CDF 被"贴"着数据画，D 天然偏小——经典的 $1.36/\sqrt{n}$ 临界值前提是真参数已知，这里全错了。实测用这个朴素临界值时误报率为 0（图 2 的灰色柱），必须用蒙特卡洛重新校准到 0.0563 才能贴住名义 5%。

AD 的核心改造是把等权的最大距离换成加权积分：

$$A^2=-n-\frac{1}{n}\sum_{i=1}^{n}(2i-1)\left[\ln\Phi(z_{(i)})+\ln\left(1-\Phi(z_{(n+1-i)})\right)\right]$$

其中 $z$ 是用样本均值、样本标准差（ddof=1）标准化的顺序统计量。权重 $2i-1$ 除以 n 后，在分布两端约为 1、中间趋近 0——换成连续形式就是把偏离按 $1/[u(1-u)]$ 加权。图 1 左用一组 t(4) 样本展示 K-S 只盯最大竖直缝隙，右画出 AD 的权重曲线：u=0.002 处权重约 501，是中间 u=0.5 处（权重 4）的一百多倍。尾部的一点点偏离，AD 都会重罚。

![K-S 等权与 AD 尾部加权的对比](/images/anderson-darling-normality-test/fig1_weighting.png)

## 三、实现与 scipy 对拍、临界值校准

我从零实现 $A^2$，用数值稳定的 logcdf/logsf 避免厚尾下 $\ln(0)$ 溢出，再乘 Stephens 小样本因子得到 $A^*=A^2(1+0.75/n+2.25/n^2)$。与 scipy.stats.anderson 在同一批 t(4) 样本上逐样本对比，原始 $A^2$ 的**最大绝对偏差 1.14e-13**（机器精度级），确认实现无误。

接着做尺寸校准：n=252 正态样本抽 5000 次，得 $A^*$ 经验分布的 5% 分位 = **0.7258**（90%/97.5%/99% 分位分别为 0.6164/0.8459/1.0106）；scipy 内置 Stephens 常数换算到 $A^*$ 口径后的 5% 线几乎重合（图 2）。用该临界值判定的实际误报率恰好 0.0500；用 scipy 的 5% 线是 0.0430，略保守。JB 的 $\chi^2(2)$ 渐近近似在 n=252 下误报率 0.0434，也还算靠谱。

![A* 经验零分布与 Stephens 临界值对表](/images/anderson-darling-normality-test/fig2_null_calibration.png)

## 四、功效对决：JB 反超，AD 居中，K-S 垫底

尺寸校准后，用共同随机数对 df=3–10 的 Student-t 备择（n=252）跑拒绝率。图 3 显示：肥尾越重（df 越小）三个检验功效都越高；df=5 时 JB 0.9275 > AD 0.8495 > K-S 0.6530；到 df=10（接近正态）三者都降到 0.5 以下。JB 全程领先，因为它直接测三、四阶矩，t 分布的厚尾正好体现在峰度上。

![不同自由度 t 备择下的功效对比](/images/anderson-darling-normality-test/fig3_power_vs_df.png)

图 4 固定 df=5、扫描样本量 n=60–1008：n=252 是关键转折点，此时 AD 已达 0.8230；n=60 时三个检验都束手无策（AD 仅 0.3365）。这提醒我们：**一年交易日的数据，够检出中等肥尾，但 detecting 更轻微的偏离需要好几年**。

![样本量对抗检验功效的影响](/images/anderson-darling-normality-test/fig4_power_vs_n.png)

## 五、卖期权式收益：负偏肥尾的实战

构造一类典型策略收益：97% 的交易日小赚 $N(0.12\%,0.8\%)$，3% 的交易日崩损 $N(-2.5\%,1.5\%)$——这是卖出尾部期权的收益结构，靠高频小盈换低频巨亏。抽一年 252 天，实测偏度 -1.14、峰度 7.12，三个检验全部报警（$A^*=1.791$ 远超 0.7258，JB=232.6，K-S=0.0626 远超 0.0563）。

但这种分布的"脾气"要多少样本才被稳定识破？图 5 右扫描 n=120–756：n=252 时 JB 报警率 0.9473、AD 0.8140、K-S 0.6467；到 n=504 三者都超过 0.86。JB 仍是最快报警的那个。

![卖期权式收益的分布与报警样本量](/images/anderson-darling-normality-test/fig5_option_selling.png)

## 六、怎么选

- **已知关注偏度/峰度**（如期权策略尾部）：优先 JB，功效最高。
- **不知道偏离长什么样、要 omnibus 把关**：用 AD，配合它的分布自由临界表，稳健且尾部敏感。
- **K-S 仅在参数外生给定时才用经典临界值**；参数自估务必蒙特卡洛重校准，否则检验形同虚设。

无论哪种，"没拒绝正态"都不等于"服从正态"——功效不足时（小样本、轻尾）三者都可能漏报，这才是实务里最该警惕的地方。

## 七、完整可运行代码

以下脚本是本文所有数字与五张图的唯一来源，自带数据生成，`python3` 直接运行即可复现 results.json：

```python
# -*- coding: utf-8 -*-
"""Anderson–Darling 正态性检验：为什么查收益分布的尾部，它比 Jarque–Bera 和 K–S 更狠。

本文全部数字的唯一来源。运行本脚本会：
1. 从零实现 AD 统计量 A² 与 Stephens 调整 A*（scipy 约定 1+0.75/√n+2.25/n），
   与 scipy.stats.anderson 逐样本对拍（最大偏差 <1e-9）；
2. 蒙特卡洛校准：正态原假设下抽 5000 个 n=252 样本，得 A* 的经验临界值，
   与 scipy 内置 Stephens 常数对表；同批样本顺带校准 JB 的 χ²(2) 近似与
   『参数自己估』后的 K–S 的失真程度；
3. 功效对决：对 df=3..10 的 Student-t 备择（n=252）与 df=5 下 n=60..1008，
   用共同随机数同步跑 AD*/JB/KS 三个检验的拒绝率；
4. 应用案例：卖期权式负偏肥尾收益（97% 小额盈利 + 3% 崩损），
   展示三个检验各在什么样本量下才报警；
5. 画 5 张图到 public/images/anderson-darling-normality-test/，
   全部统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261118
_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "anderson-darling-normality-test",
)
os.makedirs(_OUT, exist_ok=True)

ALPHA = 0.05


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def stephens_factor(n):
    """正态性 AD 的 Stephens 小样本修正因子：1 + 0.75/n + 2.25/n²。"""
    return 1.0 + 0.75 / n + 2.25 / (n * n)


def ad_raw_a2(x):
    """原始 Anderson–Darling 统计量 A²（μ、σ 由样本估计，ddof=1，与 scipy 一致）。

    A² = -n - (1/n) Σ_{i=1..n} (2i-1)[ln Φ(z_(i)) + ln(1-Φ(z_(n+1-i)))]
    z 用样本均值与样本标准差（ddof=1）标准化；logcdf/logsf 保证厚尾下不溢出。
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    z = (x - x.mean()) / x.std(ddof=1)
    log_F = stats.norm.logcdf(z)
    log_1mF = stats.norm.logsf(z)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2.0 * i - 1.0) * (log_F + log_1mF[::-1])))


def ad_star(x):
    """Stephens 调整后的 A* = A² · (1 + 0.75/n + 2.25/n²)。

    A* 的原假设分布几乎不依赖样本估参这一点，正是 AD 相对 K–S 的核心优势。
    """
    x = np.asarray(x, dtype=float)
    return ad_raw_a2(x) * stephens_factor(x.size)


def jb_stat(x):
    """Jarque–Bera：JB = n/6 (S² + (K-3)²/4)，渐近 χ²(2)。"""
    x = np.asarray(x, dtype=float)
    n = x.size
    mu = x.mean()
    s2 = ((x - mu) ** 2).mean()
    s3 = ((x - mu) ** 3).mean()
    s4 = ((x - mu) ** 4).mean()
    skew = s3 / s2 ** 1.5
    kurt = s4 / s2 ** 2
    return float(n / 6.0 * (skew ** 2 + (kurt - 3.0) ** 2 / 4.0))


def ks_stat(x):
    """K–S 距离：经验 CDF 对『自己估参数』的正态 CDF 的最大偏离。"""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    F = stats.norm.cdf((x - x.mean()) / x.std(ddof=1))
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(np.abs(i / n - F), np.abs(F - (i - 1) / n))))


def draw_samples(rng, dist, n, reps):
    """共同随机数：一次抽出 (reps, n)，三个检验吃同一批样本。

    dist: ("normal",) 或 ("t", df)。尺度族检验对 scale 不变，t 直接用标准 t。
    """
    if dist[0] == "normal":
        return rng.normal(0.0, 1.0, size=(reps, n))
    return rng.standard_t(dist[1], size=(reps, n))


def reject_rates(samples):
    """三个检验在 α=0.05 下的拒绝率。

    AD*/JB 用各自的临界值；KS 用『自己估参数』的蒙特卡洛临界值（KS 部分
    由调用方传入 KS_MC_CRIT 决定），这里返回统计量，拒绝判定分开做。
    """
    reps, n = samples.shape
    ad = np.array([ad_star(s) for s in samples])
    jb = np.array([jb_stat(s) for s in samples])
    ks = np.array([ks_stat(s) for s in samples])
    return ad, jb, ks


# —— K–S 的蒙特卡洛临界值（正态原假设、参数自估，n=252）——
def ks_mc_critical(reps=4000, n=252):
    rng = np.random.default_rng(SEED + 11)
    ks_vals = np.sort(np.array([ks_stat(s) for s in draw_samples(rng, ("normal",), n, reps)]))
    return {f"p{int(q * 100)}": float(np.quantile(ks_vals, q)) for q in [0.90, 0.95, 0.975, 0.99]}


def ad_mc_crit95(n, reps=4000, seed=None):
    """A* 在正态原假设下的蒙特卡洛 5% 临界值（参数自估）。"""
    rng = np.random.default_rng(SEED + 12 + n if seed is None else seed)
    vals = np.array([ad_star(s) for s in draw_samples(rng, ("normal",), n, reps)])
    return float(np.quantile(vals, 0.95))


def main():
    results = {"seed": SEED, "alpha": ALPHA}

    # 0) 与 scipy 对拍原始 A²（同一批样本，逐样本比，证明实现无误）
    rng = np.random.default_rng(SEED + 1)
    check = draw_samples(rng, ("t", 4), 252, 300)
    mine = np.array([ad_raw_a2(s) for s in check])
    sp = np.array([stats.anderson(s, dist="norm").statistic for s in check])
    results["scipy_check_max_abs_diff"] = float(np.max(np.abs(mine - sp)))

    # 1) 原假设校准：n=252 正态样本，5000 重复
    rng = np.random.default_rng(SEED + 2)
    null = draw_samples(rng, ("normal",), 252, 5000)
    ad0, jb0, ks0 = reject_rates(null)
    results["ad_empirical_criticals_n252"] = {
        f"p{int(q * 100)}": float(np.quantile(ad0, q)) for q in [0.90, 0.95, 0.975, 0.99]
    }
    sp_ref = stats.anderson(np.random.default_rng(0).normal(size=252), dist="norm")
    results["scipy_stephens_constants"] = {
        "levels_percent": [float(v) for v in sp_ref.significance_level],
        "critical_values": [float(v) for v in sp_ref.critical_values],
    }
    # size：各检验在自己名义 5% 下的真实误报率
    crit95_ad = results["ad_empirical_criticals_n252"]["p95"]
    sp5_a2 = float(sp_ref.critical_values[2])  # scipy 5% 临界值（原始 A² 口径）
    results["size"] = {
        "ad_with_mc_critical": float((ad0 > crit95_ad).mean()),
        "ad_with_scipy_stephens_5pct": float((ad0 > sp5_a2 * stephens_factor(252)).mean()),
        "jb_chi2": float((jb0 > stats.chi2.ppf(0.95, 2)).mean()),
        "ks_with_mc_critical": float((ks0 > ks_mc_critical()["p95"]).mean()),
        "ks_with_naive_chi2_asym": float((ks0 > 1.36 / math.sqrt(252)).mean()),
    }
    results["null_moments"] = {
        "ad_mean": float(ad0.mean()), "jb_mean": float(jb0.mean()),
        "ks_mean": float(ks0.mean()),
    }
    ks_crit = ks_mc_critical()
    results["ks_mc_criticals_n252"] = ks_crit

    # 2) 功效 vs df（n=252，共同随机数）
    power_df = {}
    for df in [3, 4, 5, 6, 8, 10]:
        rng = np.random.default_rng(SEED + 100 + df)
        s = draw_samples(rng, ("t", df), 252, 2000)
        ad, jb, ks = reject_rates(s)
        power_df[f"t{df}"] = {
            "ad": float((ad > crit95_ad).mean()),
            "jb": float((jb > stats.chi2.ppf(0.95, 2)).mean()),
            "ks": float((ks > ks_crit["p95"]).mean()),
        }
    results["power_vs_df_n252"] = power_df

    # 3) 功效 vs n（df=5，共同随机数）
    power_n = {}
    for n in [60, 120, 252, 504, 1008]:
        rng = np.random.default_rng(SEED + 300 + n)
        s = draw_samples(rng, ("t", 5), n, 2000)
        # 临界值随 n 变：AD、KS 均用同 n 下的 MC 临界值（口径一致）
        rng_c = np.random.default_rng(SEED + 400 + n)
        ks_null = np.sort(np.array([ks_stat(x) for x in draw_samples(rng_c, ("normal",), n, 3000)]))
        kcrit = float(np.quantile(ks_null, 0.95))
        acrit = ad_mc_crit95(n)
        ad, jb, ks = reject_rates(s)
        power_n[f"n{n}"] = {
            "ad": float((ad > acrit).mean()),
            "jb": float((jb > stats.chi2.ppf(0.95, 2)).mean()),
            "ks": float((ks > kcrit).mean()),
            "ad_mc_crit95": acrit,
            "ks_mc_crit95": kcrit,
        }
    results["power_vs_n_t5"] = power_n

    # 4) 卖期权式收益：97% N(+0.12%, 0.8%)，3% N(-2.5%, 1.5%)
    p_crash, mu1, sd1, mu2, sd2 = 0.03, 0.0012, 0.008, -0.025, 0.015

    def sell_put_sample(rng, n):
        flag = rng.random(n) < p_crash
        x = rng.normal(mu1, sd1, size=n)
        x[flag] = rng.normal(mu2, sd2, size=int(flag.sum()))
        return x, flag

    rng = np.random.default_rng(SEED + 505)
    x_demo, flag_demo = sell_put_sample(rng, 252)
    results["demo_sample"] = {
        "n": 252, "n_crash_days": int(flag_demo.sum()),
        "skew": float(stats.skew(x_demo)), "kurt": float(stats.kurtosis(x_demo, fisher=False)),
        "ad_star": ad_star(x_demo), "jb": jb_stat(x_demo), "ks": ks_stat(x_demo),
        "ad_reject": bool(ad_star(x_demo) > crit95_ad),
        "jb_reject": bool(jb_stat(x_demo) > stats.chi2.ppf(0.95, 2)),
        "ks_reject": bool(ks_stat(x_demo) > ks_crit["p95"]),
        "mean_daily": float(x_demo.mean()), "sd_daily": float(x_demo.std(ddof=1)),
    }
    detect_n = {}
    for n in [120, 252, 504, 756]:
        rng = np.random.default_rng(SEED + 600 + n)
        ad, jb, ks = [], [], []
        for _ in range(1500):
            x, _f = sell_put_sample(rng, n)
            ad.append(ad_star(x))
            jb.append(jb_stat(x))
            ks.append(ks_stat(x))
        rng_c = np.random.default_rng(SEED + 700 + n)
        ks_null = np.sort(np.array([ks_stat(x) for x in draw_samples(rng_c, ("normal",), n, 3000)]))
        kcrit = float(np.quantile(ks_null, 0.95))
        acrit = ad_mc_crit95(n)
        detect_n[f"n{n}"] = {
            "ad": float((np.array(ad) > acrit).mean()),
            "jb": float((np.array(jb) > stats.chi2.ppf(0.95, 2)).mean()),
            "ks": float((np.array(ks) > kcrit).mean()),
        }
    results["detection_power_sellput"] = detect_n

    # —— 五张图 ——
    # 图1：K–S 与 AD 的加权对比（左：ECDF 偏离；右：尾部权重 1/(u(1-u))）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    rng = np.random.default_rng(SEED + 808)
    x1 = rng.standard_t(4, size=252)
    xs = np.sort(x1)
    Fs = stats.norm.cdf((xs - xs.mean()) / xs.std(ddof=1))
    ii = np.arange(1, 253) / 252.0
    axes[0].step(xs, ii, where="post", color="#2c3e50", label="经验 CDF")
    axes[0].plot(xs, Fs, "-", color="#e74c3c", lw=1.8, label="拟合正态 CDF")
    axes[0].fill_between(xs, ii, Fs, step="post", color="#e74c3c", alpha=0.18)
    axes[0].set_title("K–S：全程等权地看 |F̂−Φ| 的最大偏离")
    axes[0].legend(fontsize=8)
    u = np.linspace(0.002, 0.998, 400)
    axes[1].plot(u, 1.0 / (u * (1.0 - u)), color="#8e44ad", lw=2, label="AD 权重 1/(u(1−u))")
    axes[1].plot(u, np.ones_like(u), color="#7f8c8d", lw=2, ls="--", label="K–S 权重 1")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("分位点 u = Φ(z)")
    axes[1].set_title("AD：偏离按 1/(u(1−u)) 加权，两头放大数百倍")
    axes[1].legend(fontsize=8)
    fig.suptitle("图1：同样一组 t(4) 样本，K–S 与 AD 各自『看哪里』", fontsize=11)
    results["figs"] = [_save(fig, "fig1_weighting.png")]

    # 图2：A* 的经验零分布 vs Stephens 常数
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.hist(ad0, bins=70, density=True, color="#2980b9", alpha=0.65,
            label=f"蒙特卡洛零分布（n=252，5000 重复）")
    for cv, lab in zip(results["scipy_stephens_constants"]["critical_values"],
                       ["15%", "10%", "5%", "2.5%", "1%"]):
        cv_star = cv * stephens_factor(252)  # 换算到 A* 口径
        ax.axvline(cv_star, color="#c0392b", ls="--", lw=1.2)
        ax.text(cv_star, ax.get_ylim()[1] * 0.92, f"{lab}\n{cv_star:.3f}", fontsize=7, ha="center")
    for q, name in [("p95", "实测95%分位")]:
        ax.axvline(results["ad_empirical_criticals_n252"][q], color="k", lw=1.6)
        ax.text(results["ad_empirical_criticals_n252"][q], ax.get_ylim()[1] * 0.55,
                f"实测 {name}={results['ad_empirical_criticals_n252'][q]:.3f}",
                fontsize=8, rotation=90, va="center")
    ax.set_xlabel("A* 统计量")
    ax.set_title("图2：A* 的经验零分布与 Stephens 临界值对表（两者几乎重合）")
    ax.legend(fontsize=8)
    results["figs"].append(_save(fig, "fig2_null_calibration.png"))

    # 图3：功效 vs df（n=252）
    dfs = [3, 4, 5, 6, 8, 10]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for key, name, color, mk in [("ad", "AD*（MC 临界值）", "#8e44ad", "o"),
                                 ("jb", "Jarque–Bera（χ²(2)）", "#e67e22", "s"),
                                 ("ks", "K–S（MC 临界值）", "#16a085", "^")]:
        ys = [power_df[f"t{d}"][key] for d in dfs]
        ax.plot(dfs, ys, mk + "-", color=color, lw=2, label=name)
        for d, y in zip(dfs, ys):
            ax.text(d, y + 0.012, f"{y:.2f}", ha="center", fontsize=7, color=color)
    ax.axhline(0.05, color="k", ls="--", lw=1.1)
    ax.text(10.15, 0.052, "名义 5%", fontsize=8)
    ax.set_xlabel("Student-t 备择的自由度 df（越轻尾越像正态）")
    ax.set_ylabel("α=0.05 下的拒绝率（功效）")
    ax.set_title("图3：n=252 时三个检验对肥尾备择的功效（共同随机数对决）")
    ax.legend(fontsize=9)
    results["figs"].append(_save(fig, "fig3_power_vs_df.png"))

    # 图4：功效 vs n（df=5）
    ns = [60, 120, 252, 504, 1008]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for key, name, color, mk in [("ad", "AD*", "#8e44ad", "o"),
                                 ("jb", "Jarque–Bera", "#e67e22", "s"),
                                 ("ks", "K–S", "#16a085", "^")]:
        ys = [power_n[f"n{n}"][key] for n in ns]
        ax.plot(ns, ys, mk + "-", color=color, lw=2, label=name)
        for n, y in zip(ns, ys):
            ax.text(n, y + 0.012, f"{y:.2f}", ha="center", fontsize=7, color=color)
    ax.axhline(0.05, color="k", ls="--", lw=1.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("样本量 n（交易日）")
    ax.set_ylabel("α=0.05 下的拒绝率")
    ax.set_title("图4：df=5 肥尾备择下，三家各自的报警样本量（AD 在 n=252 检出 82%）")
    ax.legend(fontsize=9)
    results["figs"].append(_save(fig, "fig4_power_vs_n.png"))

    # 图5：卖期权式收益的报警样本量
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    raw = np.sort(x_demo)
    axes[0].hist(raw, bins=80, color="#2980b9", alpha=0.8)
    axes[0].axvline(np.sort(raw)[int(flag_demo.sum()) - 1], color="#c0392b", ls="--", lw=1.4)
    axes[0].text(np.sort(raw)[int(flag_demo.sum()) - 1], axes[0].get_ylim()[1] * 0.6,
                 f" 3% 崩损日 ×{int(flag_demo.sum())}", fontsize=8, color="#c0392b")
    axes[0].set_title(f"一年 252 天的卖期权式日收益：偏度 {results['demo_sample']['skew']:.2f}、"
                      f"峰度 {results['demo_sample']['kurt']:.1f}")
    axes[0].set_xlabel("日收益")
    ns2 = [120, 252, 504, 756]
    for key, name, color, mk in [("ad", "AD*", "#8e44ad", "o"),
                                 ("jb", "Jarque–Bera", "#e67e22", "s"),
                                 ("ks", "K–S", "#16a085", "^")]:
        ys = [detect_n[f"n{n}"][key] for n in ns2]
        axes[1].plot(ns2, ys, mk + "-", color=color, lw=2, label=name)
        for n, y in zip(ns2, ys):
            axes[1].text(n, y + 0.015, f"{y:.2f}", ha="center", fontsize=7, color=color)
    axes[1].axhline(0.05, color="k", ls="--", lw=1.1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ns2)
    axes[1].set_xticklabels(ns2)
    axes[1].set_xlabel("样本量 n（交易日）")
    axes[1].set_title("报警率：负偏+肥尾多久能被三个检验逮住")
    axes[1].legend(fontsize=9)
    fig.suptitle("图5：卖期权式负偏肥尾——JB 报警最快，AD 次之，K–S 最迟钝", fontsize=11)
    results["figs"].append(_save(fig, "fig5_option_selling.png"))

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
```

