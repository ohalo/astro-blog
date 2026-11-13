---
title: "β 测不准，溢价就打折：两阶段因子回归的测量误差、衰减偏倚与 Shanken 修正"
description: "因子暴露是估出来的，两阶段回归的推断从第一步就在漏气。真值已知的模拟现场对拍：因子带测量噪声时 OLS β 沿 β/(1+λ) 精确衰减，λ=1 时 0.8 被打到 0.3997；N=100、T=120、特异波动 1.2 时，生成的回归元把截面斜率从 0.5 压到 0.4124（衰减因子 0.8389），偏倚恰好等于 1 个标准差。Shanken 修正只治标准误：覆盖率从 0.8913 抬到 0.9100，治不了衰减；修正因子实测 1+λ² 的平方根在 λ=1 时 1.3639 对理论 1.4142，因为 λ̂ 自己也在衰减。最贵的代价是伪 alpha：衰减的 beta 制造出与真实 β 成正比的假异常（斜率 0.0846 对理论 0.0805），|t|>1.96 的资产占比从名义 5% 涨到实测 16.78%，膨胀 3.4 倍。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-13T07:10:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 因子模型
  - Python
language: zh
difficulty: advanced
---

## 一、问题：第二阶段的斜率从第一步就已经不是溢价了

因子模型的标准检验流程是两步走：第一步对每个资产做时间序列回归，估出因子暴露 βᵢ；第二步拿这些 βᵢ 当回归元，做截面回归估计风险溢价 λ。几乎所有人都默认第二步的输出可以按普通 OLS 解读——标准误、置信区间、t 值，一路照搬教科书公式。

问题在于，第二步的回归元不是上帝给的 βᵢ，而是第一步估出来的 β̂ᵢ。回归元自己带着估计噪声，这在计量经济学里叫「生成的回归元」（generated regressors）：经典 OLS 推断的全部前提——回归元与误差独立、回归元没有测量误差——从第一步开始就已经不成立。这篇文章用四个可以现场复算的实验，把漏气的位置和漏气量逐个量出来。

## 二、单资产：β 沿 β/(1+λ) 精确衰减

先看最干净的一层。设真实因子 f、观测因子 f̃ = f + u，测量噪声方差与因子方差之比记作 λ = Var(u)/Var(f)。资产收益 r = βf + e。对 f̃ 做 OLS，斜率的概率极限是

$$\mathrm{plim}\ \hat\beta = \frac{\mathrm{Cov}(r,\tilde f)}{\mathrm{Var}(\tilde f)} = \frac{\beta\,\mathrm{Var}(f)}{\mathrm{Var}(f)+\mathrm{Var}(u)} = \frac{\beta}{1+\lambda}$$

这是最经典的衰减公式：测量噪声把 β 按固定比例往零打折，噪声越大、打折越狠，但永远不会变号。

![图1 单资产 OLS β 沿理论衰减线 β/(1+λ) 精确下滑](/images/beta-attenuation-errors-in-variables/fig1_eiv_attenuation.png)

图 1 是 400 次重复的模拟（T=1000，真 β=0.8）。六个噪声档位下，模拟均值与理论线的最大相对偏差只有 0.0008：λ=0.5 时 β̂ 均值 0.5331 对理论 0.5333，λ=1.0 时 0.3997 对 0.4000。衰减不是近似，是精确的力学——只要测量噪声方差已知，打折比例就能精确算出来。麻烦在于实际中 Var(u) 不可观测，你甚至不知道自己的 β 被打了多少折。

## 三、两阶段：生成的回归元让截面斜率打折

单资产衰减还只是热身。真正的两阶段场景里，噪声进入的方式更隐蔽：β̂ᵢ = βᵢ + ηᵢ，ηᵢ 的方差约等于 σ²ₑ/(T·Var(f))——特异波动越大、样本越短，β̂ᵢ 越不准。第二步截面回归 yᵢ = λβᵢ 的斜率于是变成

$$\mathrm{plim}\ \hat\lambda_{CS} = \lambda\cdot\frac{\mathrm{Var}(\beta)}{\mathrm{Var}(\beta)+\mathrm{Var}(\eta)}$$

截图面 β 的离散度与估计噪声方差之比决定打折幅度：β 离散度越小（大家暴露差不多），或者特异波动越大、窗口越短，衰减越狠。

![图2 两阶段截面斜率随特异波动衰减，模拟与理论吻合](/images/beta-attenuation-errors-in-variables/fig2_cs_slope_attenuation.png)

图 2 的设定：N=100 个资产、T=120 天、βᵢ~N(1, 0.25²)、真溢价 λ=0.5，扫特异波动 σₑ 从 0.2 到 1.0，每档 500 次重复。绿色方块（因变量用真实期望收益 λβ）与理论线几乎重合，最大偏差 0.0006；紫色圆点（因变量用样本均值 r̄，多一层均值估计噪声）均值与前者几乎一样，但散布大得多——均值估计噪声不改变衰减的中心，只加大方差。σₑ=1.0 时斜率已经从 0.5 掉到 0.4405，折掉 12%。

有一个反直觉的细节值得单独说：衰减比例的分母里进的是 β 的**截面离散度** Var(β)，不是 β 的均值。这意味着同一套流程在宽截面上相对可靠，在窄截面上几乎失效。如果你的 100 个资产全是同行业的、暴露挤在 0.9 到 1.1 之间，Var(β) 会小到 β̂ 的噪声可以主导分母——哪怕每个 β 的时序回归 R² 都很漂亮，第二步的斜率照样被大幅打折。反过来，刻意选择暴露分散的截面，等价于免费放大信号的分母。多数人调两阶段回归时只盯着 T 够不够长，很少意识到截面的选择本身就是估计精度的一部分。

另外注意衰减的对称性：它把斜率往零打折，但不改变符号，也不改变排序。所以基于 β̂ 的排序、分组、多空划分大体上还是对的——错的是给排序结果标价的那个 λ。这就是为什么衰减偏倚在排序类研究里容易潜伏多年：排序结论看起来都对，溢价数字却系统性偏低，而偏低本身不会触发任何报警。

## 四、Shanken 修正管标准误，不管衰减偏倚

Shanken（1992）的修正在业内名气很大，但它到底修的是什么，经常被说错。它修的是**标准误**：Fama-MacBeth 朴素标准误把 β̂ᵢ 当成真值，忽略了回归元噪声带来的额外不确定性，修正因子是 c = 1 + λ′Σ_f⁻¹λ（单因子即 1 + λ²/σ²_f），标准误乘 √c。

![图3 Shanken 修正抬升覆盖率，但治不了衰减偏倚](/images/beta-attenuation-errors-in-variables/fig3_shanken_coverage.png)

图 3 的蒙特卡洛（800 次重复，λ=0.5）：朴素 SE 的 95% 置信区间覆盖率在 σₑ=0.5/0.8/1.2 时分别为 0.9488、0.9338、0.8913，Shanken 修正后抬到 0.9700、0.9513、0.9100。方向对了，但请注意右图：σₑ=1.2 时 λ̄ 的均值是 0.4124，离真值 0.5 差了 0.0876，而经验标准差只有 0.0877——**偏倚恰好等于 1 个标准差**。这时无论标准误修得多准，以 λ̂ 为中心的置信区间都不可能同时盖住真值，覆盖率封死在 0.91 附近。Shanken 修正是方差修正，衰减偏倚是 O(1/T) 的中心偏倚，两者是两种病。

![图4 Shanken 修正因子随溢价水平放大，实测略低于理论](/images/beta-attenuation-errors-in-variables/fig4_shanken_se_inflation.png)

图 4 把修正因子本身摊开看（σₑ=0.8，扫溢价 λ 从 0 到 1.0）：实测的 Shanken/朴素 SE 之比是 1.0036、1.0308、1.1049、1.2180、1.3639，对理论 √(1+λ²) = 1、1.0308、1.1180、1.25、1.4142。前三档几乎重合，λ=1.0 时实测比理论低 0.0503——原因很诚实：修正公式里用的是估计出来的 λ̂，而 λ̂ 自己已经被衰减打折（1.0 打到约 0.92），平方之后差距更明显。溢价越高、修正越重要，同时修正本身也越保守。

把三个实验串起来读，两阶段回归的误差结构其实分了三层。第一层是衰减：斜率的中心从 λ 移到 kλ，k<1，这一层 Shanken 完全不管。第二层是方差低估：朴素 FM 标准误把 β̂ 当真值，忽略每个截面上回归元都在抖，这一层 Shanken 用 √c 修复。第三层是 FM 时间序列自身的有限样本噪声，检验统计量的分布在小 N、短 T 时偏离渐近正态。实务里最常见的错误是把第二层的修复误当成对第一层的修复——覆盖率从 0.89 抬到 0.91 之后，人容易产生「修正过了，数字可信」的错觉，而此时点估计离真值的距离仍然是标准差的一倍。判断自己在哪一层，看的不是覆盖率，而是 λ̄ 与理论溢价之间的绝对距离除以经验标准差这个量：它接近零，才轮到标准误登场。

## 五、最贵的代价：衰减的 beta 制造假 alpha

前面都是溢价估计的失准，实务里更疼的伤口在 alpha。定价检验的 alpha 是 α̂ᵢ = r̄ᵢ − λ̂β̂ᵢ。当 β̂ 系统性偏小（衰减因子 k<1）而 λ̂ 也跟着偏小时，两者错配的残差不会归零：展开期望可得 E[α̂ᵢ] ≈ λ(1−k)βᵢ——**伪 alpha 与真实 β 成正比**。回归出来的东西长什么样？一个「低 beta 异常」：高 β 资产全体带着负 alpha、低 β 资产全体带着正 alpha，换算成年化就是一条看似可交易的假曲线。

![图5 伪 alpha 与真实 β 成正比，把假发现率抬到名义的 3.4 倍](/images/beta-attenuation-errors-in-variables/fig5_spurious_alpha_fdr.png)

图 5 的设定（σₑ=1.2、T=120、λ=0.5，500 次重复）：左图是单次重复的 α̂ᵢ 对真实 βᵢ 的散点，理论斜率 λ(1−k) = 0.0805（k=0.8389），模拟均值 0.0846、标准差 0.0293，与理论相容。右图更扎眼：研究者按惯例用 σ̂ₑ/√T 算每个资产 alpha 的 t 值（忽略 β̂ 噪声），|t|>1.96 的资产占比均值 0.1678——名义 5% 的检验实际假发现率 16.78%，膨胀 3.4 倍。而且这个膨胀对每个资产是同向的：它不是把 5% 的假阳性随机撒得到处都是，而是集中在 β 高的资产上，恰好制造出「低 beta 显著跑赢」的系统性假象。

## 六、怎么办，以及没写进结论的部分

三件事值得写进检查清单。第一，报告截面回归结果时同时报告第一阶段 β̂ 的估计精度：特异波动 ÷ √T 这个量，决定了你的溢价被打了多少折。第二，Shanken 修正要照做，但要知道它只修方差；T 短、特异波动大时，中心偏倚主导，修正后覆盖率依然不到 0.95——此时该做的是拉长窗口或换更离散的 β 截面，而不是继续调标准误。第三，alpha 检验前先做反向诊断：把 α̂ 对 β̂（或更稳妥地对外生工具变量）回归，斜率显著非零就是衰减在报警。

本文的边界也要说清楚：所有实验都是单因子、正态噪声、β 截面正态的设计；ηᵢ 与均值估计噪声 r̄ᵢ−λβᵢ 来自同一路残差，两者相关会让小样本里斜率分布偏斜，这正是图 2 紫色点散布明显更大的原因之一；多因子情形的 Shanken 修正因子是矩阵式 c = 1 + λ′Σ_f⁻¹λ，本文只演示了单因子标量版。参考文献：Shanken (1992, RFS) 的修正与 Fama-MacBeth (1973) 的原始框架；测量误差衰减的经典出处是 Frisch（1934）以来的 errors-in-variables 文献。

```python
# -*- coding: utf-8 -*-
"""两阶段因子回归的测量误差：beta 衰减与 Shanken 修正。

本文所有数字的唯一来源。运行本脚本会：
1. 现场生成全部模拟数据（固定随机种子，可复现）；
2. 画 5 张图到 public/images/beta-attenuation-errors-in-variables/；
3. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261113
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "beta-attenuation-errors-in-variables"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


def _ols_slope(x, y):
    """带截距 OLS 斜率。x: (n,), y: (n,)。"""
    xm = x - x.mean()
    ym = y - y.mean()
    return float(xm @ ym / (xm @ xm))


# ---------------------------------------------------------------- 实验 1
# 单资产 EIV：因子带测量噪声时，OLS beta 以 1/(1+lam) 衰减
def exp1_attenuation(reps=400, T=1000, beta_true=0.8, sigma_e=0.5):
    rng = np.random.default_rng(SEED)
    lams = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]
    means, sds = [], []
    for lam in lams:
        draws = np.empty(reps)
        for k in range(reps):
            f = rng.standard_normal(T)
            u = np.sqrt(lam) * rng.standard_normal(T)
            e = sigma_e * rng.standard_normal(T)
            r = beta_true * f + e
            draws[k] = _ols_slope(f + u, r)
        means.append(float(draws.mean()))
        sds.append(float(draws.std(ddof=1)))
    theory = [beta_true / (1.0 + lam) for lam in lams]
    max_dev = max(abs(m - t) for m, t in zip(means, theory))
    rel_dev = max(abs(m - t) / t for m, t in zip(means, theory))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        lams, means, yerr=sds, fmt="o", color="#1f6feb", capsize=3,
        label="模拟均值 ± 1 标准差（400 次重复）",
    )
    ax.plot(lams, theory, "-", color="#d1242f", lw=1.8,
            label=r"理论衰减 $\beta/(1+\lambda)$")
    ax.set_xlabel(r"测量噪声方差比 $\lambda=\mathrm{Var}(u)/\mathrm{Var}(f)$")
    ax.set_ylabel(r"OLS $\hat\beta$")
    ax.set_title(r"图1 因子测量误差使 OLS $\beta$ 沿 $\beta/(1+\lambda)$ 精确衰减")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig1_eiv_attenuation.png")
    return {
        "lams": lams,
        "beta_hat_mean": means,
        "beta_hat_sd": sds,
        "theory": theory,
        "beta_true": beta_true,
        "T": T,
        "reps": reps,
        "max_abs_dev": max_dev,
        "max_rel_dev": rel_dev,
    }


# ---------------------------------------------------------------- 实验 2
# 两阶段截面回归：生成的回归元（第一阶段估计的 beta）带来斜率衰减
def exp2_cs_slope(reps=500, N=100, T=120, lam_premium=0.5, beta_mean=1.0,
                  beta_sd=0.25):
    rng = np.random.default_rng(SEED + 1)
    sigmas_e = [0.2, 0.4, 0.6, 0.8, 1.0]
    m1, m2, s1, s2 = [], [], [], []
    var_beta = beta_sd ** 2
    for se in sigmas_e:
        v_eta = se ** 2 / T  # Var(f)=1
        k = var_beta / (var_beta + v_eta)
        d1, d2 = np.empty(reps), np.empty(reps)
        for rep in range(reps):
            beta = rng.normal(beta_mean, beta_sd, N)
            f = rng.standard_normal(T)
            e = se * rng.standard_normal((N, T))
            r = lam_premium * beta[:, None] + beta[:, None] * f[None, :] + e
            # 第一阶段：逐资产时序回归（带截距）
            fc = f - f.mean()
            denom = fc @ fc
            beta_hat = (r @ fc) / denom  # 斜率（截距吸收 lam*beta）
            rbar = r.mean(axis=1)
            d1[rep] = _ols_slope(beta_hat, rbar)
            d2[rep] = _ols_slope(beta_hat, lam_premium * beta)
        m1.append(float(d1.mean())); s1.append(float(d1.std(ddof=1)))
        m2.append(float(d2.mean())); s2.append(float(d2.std(ddof=1)))
    theory = [
        lam_premium * var_beta / (var_beta + se ** 2 / T) for se in sigmas_e
    ]
    max_dev = max(abs(m - t) for m, t in zip(m2, theory))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(sigmas_e, m2, yerr=s2, fmt="o", color="#1f6feb", capsize=3,
                label="因变量=真实期望收益 λβ（500 次重复）")
    ax.errorbar(sigmas_e, m1, yerr=s1, fmt="s", color="#8250df", capsize=3,
                label="因变量=样本均值 r̄（多一层均值估计噪声）")
    ax.plot(sigmas_e, theory, "--", color="#d1242f", lw=1.8,
            label=r"理论 $\lambda\,\mathrm{Var}(\beta)/(\mathrm{Var}(\beta)+\sigma_e^2/T)$")
    ax.axhline(lam_premium, color="gray", lw=1, ls=":")
    ax.text(sigmas_e[0], lam_premium + 0.004, "真风险溢价 λ=0.5", fontsize=9,
            color="gray")
    ax.set_xlabel(r"特异波动 $\sigma_e$")
    ax.set_ylabel("第二阶段截面斜率")
    ax.set_title("图2 生成的回归元让截面斜率朝零衰减，且随特异波动加重")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig2_cs_slope_attenuation.png")
    return {
        "sigmas_e": sigmas_e,
        "slope_mean_rbar": m1,
        "slope_sd_rbar": s1,
        "slope_mean_true": m2,
        "slope_sd_true": s2,
        "theory": theory,
        "lam_premium": lam_premium,
        "N": N,
        "T": T,
        "beta_sd": beta_sd,
        "reps": reps,
        "max_abs_dev_true": max_dev,
    }


# ---------------------------------------------------------------- FM 工具
def _fama_macbeth(r, beta_hat):
    """逐期截面 OLS 斜率（带截距）。r: (N,T), beta_hat: (N,)。"""
    N, T = r.shape
    x = np.vstack([np.ones(N), beta_hat])
    inv = np.linalg.inv(x @ x.T)
    lams = np.empty(T)
    for t in range(T):
        lams[t] = (inv @ x @ r[:, t])[1]
    return lams


# ---------------------------------------------------------------- 实验 3
# Shanken 修正前后的 95% 置信区间覆盖率（蒙特卡洛）
def exp3_shanken_coverage(reps=800, N=100, T=120, lam_true=0.5,
                          beta_mean=1.0, beta_sd=0.25):
    rng = np.random.default_rng(SEED + 2)
    sigmas_e = [0.5, 0.8, 1.2]
    out = {"sigmas_e": sigmas_e, "coverage_naive": [], "coverage_shanken": [],
           "sd_emp": [], "se_naive_avg": [], "se_shanken_avg": [],
           "lam_mean": [], "reps": reps, "N": N, "T": T,
           "lam_true": lam_true, "beta_sd": beta_sd}
    for se in sigmas_e:
        cov_n = cov_s = 0
        se_n_all, se_s_all, lam_bar_all = np.empty(reps), np.empty(reps), np.empty(reps)
        for rep in range(reps):
            beta = rng.normal(beta_mean, beta_sd, N)
            f = rng.standard_normal(T)
            e = se * rng.standard_normal((N, T))
            r = lam_true * beta[:, None] + beta[:, None] * f[None, :] + e
            fc = f - f.mean()
            beta_hat = (r @ fc) / (fc @ fc)
            lams_t = _fama_macbeth(r, beta_hat)
            lam_bar = lams_t.mean()
            se_n = lams_t.std(ddof=1) / np.sqrt(T)
            c = 1.0 + lam_bar ** 2 / f.var(ddof=1)
            se_s = se_n * np.sqrt(c)
            lam_bar_all[rep] = lam_bar
            se_n_all[rep] = se_n
            se_s_all[rep] = se_s
            if abs(lam_bar - lam_true) < 1.96 * se_n:
                cov_n += 1
            if abs(lam_bar - lam_true) < 1.96 * se_s:
                cov_s += 1
        out["coverage_naive"].append(cov_n / reps)
        out["coverage_shanken"].append(cov_s / reps)
        out["sd_emp"].append(float(lam_bar_all.std(ddof=1)))
        out["se_naive_avg"].append(float(se_n_all.mean()))
        out["se_shanken_avg"].append(float(se_s_all.mean()))
        out["lam_mean"].append(float(lam_bar_all.mean()))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    x = np.arange(len(sigmas_e))
    w = 0.36
    ax.bar(x - w / 2, out["coverage_naive"], w, color="#9a6700",
           label="Fama-MacBeth 朴素 SE")
    ax.bar(x + w / 2, out["coverage_shanken"], w, color="#d1242f",
           label="Shanken 修正 SE")
    ax.axhline(0.95, color="gray", ls=":", lw=1.5)
    ax.text(-0.45, 0.955, "名义 95%", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigmas_e])
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel(r"特异波动 $\sigma_e$")
    ax.set_ylabel(r"$\hat\lambda$ 的 95% CI 覆盖率")
    ax.set_title("图3a Shanken 修正抬升覆盖率，但治不了衰减偏倚")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.plot(x, out["sd_emp"], "o-", color="#1f6feb", label="经验标准差")
    ax.plot(x, out["se_naive_avg"], "s--", color="#9a6700", label="朴素 SE 均值")
    ax.plot(x, out["se_shanken_avg"], "^-", color="#d1242f",
            label="Shanken SE 均值")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigmas_e])
    ax.set_xlabel(r"特异波动 $\sigma_e$")
    ax.set_ylabel(r"$\hat\lambda$ 的离散度")
    ax.set_title(r"图3b $\sigma_e=1.2$ 时偏倚 $\approx$ 1 个标准差，主导失准")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig3_shanken_coverage.png")
    return out


# ---------------------------------------------------------------- 实验 4
# Shanken 因子的来源：c = 1 + λ²/σ_f²，随溢价水平放大
def exp4_se_inflation(reps=800, N=100, T=120, sigma_e=0.8,
                      beta_mean=1.0, beta_sd=0.25):
    rng = np.random.default_rng(SEED + 3)
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    sd_emp, se_n_avg, se_s_avg = [], [], []
    for lam_true in lams:
        lam_bar_all = np.empty(reps)
        se_n_all = np.empty(reps)
        se_s_all = np.empty(reps)
        for rep in range(reps):
            beta = rng.normal(beta_mean, beta_sd, N)
            f = rng.standard_normal(T)
            e = sigma_e * rng.standard_normal((N, T))
            r = lam_true * beta[:, None] + beta[:, None] * f[None, :] + e
            fc = f - f.mean()
            beta_hat = (r @ fc) / (fc @ fc)
            lams_t = _fama_macbeth(r, beta_hat)
            se_n = lams_t.std(ddof=1) / np.sqrt(T)
            lam_bar = lams_t.mean()
            lam_bar_all[rep] = lam_bar
            se_n_all[rep] = se_n
            se_s_all[rep] = se_n * np.sqrt(1.0 + lam_bar ** 2 / f.var(ddof=1))
        sd_emp.append(float(lam_bar_all.std(ddof=1)))
        se_n_avg.append(float(se_n_all.mean()))
        se_s_avg.append(float(se_s_all.mean()))
    theory = [float(np.sqrt(1.0 + lam ** 2)) for lam in lams]
    ratio = [s / n for s, n in zip(se_s_avg, se_n_avg)]
    max_dev = max(abs(rt - t) for rt, t in zip(ratio, theory))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(lams, sd_emp, "o-", color="#1f6feb", label="经验标准差（800 次重复）")
    ax.plot(lams, se_n_avg, "s--", color="#9a6700", label="Fama-MacBeth 朴素 SE 均值")
    ax.plot(lams, se_s_avg, "^-", color="#d1242f",
            label=r"Shanken 修正 SE 均值（$\times\sqrt{1+\hat\lambda^2}$）")
    ax.set_xlabel(r"真风险溢价 $\lambda$")
    ax.set_ylabel(r"$\hat\lambda$ 的标准误")
    ax.set_title(r"图4 Shanken 因子 $c=1+\lambda^2/\sigma_f^2$：溢价越高，低估越狠")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig4_shanken_se_inflation.png")
    return {
        "lams": lams, "sd_emp": sd_emp, "se_naive_avg": se_n_avg,
        "se_shanken_avg": se_s_avg, "ratio_shanken_over_naive": ratio,
        "theory_c": theory, "sigma_e": sigma_e, "reps": reps,
        "max_abs_dev_ratio": max_dev,
    }


# ---------------------------------------------------------------- 实验 5
# 衰减的 beta 制造假 alpha：与真实 beta 成正比的伪异常 + 假发现率膨胀
def exp5_spurious_alpha(reps=500, N=100, T=120, lam_true=0.5, sigma_e=1.2,
                        beta_mean=1.0, beta_sd=0.25):
    rng = np.random.default_rng(SEED + 4)
    var_beta = beta_sd ** 2
    v_eta = sigma_e ** 2 / T
    k = var_beta / (var_beta + v_eta)
    theory_slope = lam_true * (1.0 - k)
    slopes = np.empty(reps)
    fdr = np.empty(reps)
    for rep in range(reps):
        beta = rng.normal(beta_mean, beta_sd, N)
        f = rng.standard_normal(T)
        e = sigma_e * rng.standard_normal((N, T))
        r = lam_true * beta[:, None] + beta[:, None] * f[None, :] + e
        fc = f - f.mean()
        denom = fc @ fc
        beta_hat = (r @ fc) / denom
        resid = r - r.mean(axis=1, keepdims=True) - beta_hat[:, None] * fc[None, :]
        sig_e = resid.std(axis=1, ddof=2)
        lams_t = _fama_macbeth(r, beta_hat)
        lam_bar = lams_t.mean()
        rbar = r.mean(axis=1)
        alpha = rbar - lam_bar * beta_hat
        slopes[rep] = _ols_slope(beta, alpha)  # 诊断用真实 beta
        tstat = alpha / (sig_e / np.sqrt(T))
        fdr[rep] = float((np.abs(tstat) > 1.96).mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    beta_show = rng.normal(beta_mean, beta_sd, N)
    f_show = rng.standard_normal(T)
    e_show = sigma_e * rng.standard_normal((N, T))
    r_show = lam_true * beta_show[:, None] + beta_show[:, None] * f_show[None, :] + e_show
    fc = f_show - f_show.mean()
    bh = (r_show @ fc) / (fc @ fc)
    lams_t = _fama_macbeth(r_show, bh)
    alpha_show = r_show.mean(axis=1) - lams_t.mean() * bh
    ax.scatter(beta_show, alpha_show, s=14, color="#1f6feb", alpha=0.6)
    xs = np.linspace(beta_show.min(), beta_show.max(), 10)
    ax.plot(xs, theory_slope * xs, color="#d1242f", lw=1.8,
            label=r"理论斜率 $\lambda(1-k)$")
    ax.set_xlabel(r"真实 $\beta_i$")
    ax.set_ylabel(r"估计的 $\hat\alpha_i$")
    ax.set_title("图5a 衰减的 beta 制造与 β 成正比的伪 alpha（单次重复）")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(fdr, bins=24, color="#1f6feb", alpha=0.75)
    ax.axvline(0.05, color="gray", ls=":", lw=1.5)
    ax.text(0.052, ax.get_ylim()[1] * 0.9, "名义 5%", fontsize=9, color="gray")
    ax.axvline(float(fdr.mean()), color="#d1242f", lw=1.8)
    ax.text(float(fdr.mean()) + 0.002, ax.get_ylim()[1] * 0.78,
            f"实测 {fdr.mean():.3f}", fontsize=9, color="#d1242f")
    ax.set_xlabel("单次重复中 |t|>1.96 的资产占比")
    ax.set_ylabel("重复次数")
    ax.set_title("图5b 伪 alpha 把 |t|>1.96 比例抬到远超 5%")
    ax.grid(alpha=0.3)
    _save(fig, "fig5_spurious_alpha_fdr.png")
    return {
        "sigma_e": sigma_e, "T": T, "lam_true": lam_true,
        "attenuation_k": k, "theory_alpha_slope": theory_slope,
        "alpha_slope_mean": float(slopes.mean()),
        "alpha_slope_sd": float(slopes.std(ddof=1)),
        "reps": reps,
        "fdr_mean": float(fdr.mean()),
        "fdr_sd": float(fdr.std(ddof=1)),
        "nominal": 0.05,
    }


def main():
    r1 = exp1_attenuation()
    r2 = exp2_cs_slope()
    r3 = exp3_shanken_coverage()
    r4 = exp4_se_inflation()
    r5 = exp5_spurious_alpha()
    results = {"exp1": r1, "exp2": r2, "exp3": r3, "exp4": r4, "exp5": r5}
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

