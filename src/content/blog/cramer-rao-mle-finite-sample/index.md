---
title: "Cramér–Rao 下界不是你能达到的精度：Student-t 自由度估计的有限样本偏倚与 VaR 误差"
description: "厚尾建模里自由度 nu 看着只是一个参数，实则几乎不可估。本文用数值积分、40 万次蒙特卡洛与观测 Hessian 三条独立路径现场计算 t 分布关于 nu 的 Fisher 信息，解析 score 与中心差分逐点最大偏差 2.98e-11；I(5)=0.003023、I(10)=0.000250，n=252 的 Cramér–Rao 下界从 nu=3 的 0.2346 暴涨到 nu=10 的 15.8485（67.5 倍）。1500 次 MLE 重复：n=60 时 Var/CRLB 达 1289 倍、5.13% 样本发散到边界，n=252 降到 3.47 倍，n=1000 才收到 1.19 倍。诚实的反转：Wald 区间覆盖率 0.962 反而略高于 profile 似然比区间的 0.9587，因为 nu_hat 右偏使 I(nu_hat) 变小、区间自动放宽；但 Wald 有 0.47% 给出负下界，似然比有 0.33% 完全没有上界。更贵的代价是尺度与尾厚度混淆：真实尺度从 0.8 到 2.0（估计时一律错设为 1），nu_hat 中位数从 12.235 掉到 1.422，99% VaR 从 2.694 抬到 12.603；即使尺度正确，单条 252 日样本推出的 VaR 5%-95% 带宽仍有 2.859-3.878（真值 3.365）。附完整代码与五张真实计算图。"
publishDate: '2026-11-18T11:20:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 厚尾风险
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

给日收益套一个 Student-t 分布，自由度 $\nu$ 常被当成「顺手估一下」的参数。本文用真值 $\nu_0=5$ 的模拟把这件事拆开，结论不太客气：**$\nu$ 是整条分布里最难估的量之一，而它的估计误差会直接变成 99% VaR 的真金白银误差**。

四条要点：

1. **信息量随 $\nu$ 增大而塌缩**。$I(3)=0.016911$、$I(5)=0.003023$、$I(10)=0.000250$、$I(50)=5.21\times10^{-7}$。$n=252$ 时 $\mathrm{Var}(\hat\nu)$ 的 Cramér–Rao 下界从 $\nu=3$ 的 0.2346 涨到 $\nu=10$ 的 15.8485，整整 67.5 倍。分布越接近正态，越难判断它到底有多厚尾。
2. **下界是渐近极限，不是小样本精度**。1500 次 MLE 重复里，$n=60$ 的 $\mathrm{Var}/\mathrm{CRLB}=1288.95$ 倍（5.13% 的样本把 $\hat\nu$ 推到数值边界 100 以上），$n=252$ 降到 3.47 倍，$n=1000$ 才收到 1.19 倍。
3. **一个与预期相反的诚实结果**：Wald 区间覆盖率 0.962，反而略高于 profile 似然比区间的 0.9587，两者都盖住了名义 95%。原因不是 Wald 二次近似变好了，而是 $\hat\nu$ 右偏时 $I(\hat\nu)$ 变小、区间自动放宽，误打误撞补偿了偏斜。
4. **最贵的代价是尺度混淆**。真实尺度从 0.8 变到 2.0（估计时一律错设为 1），$\hat\nu$ 中位数从 12.235 掉到 1.422，推出的 99% VaR 从 2.694 抬到 12.603——近 4.7 倍。

## 二、Fisher 信息：三条独立路径现场对拍

标准 Student-t（位置 0、尺度 1）的对数密度与关于 $\nu$ 的 score 为：

$$\log f(x;\nu)=\log\Gamma\!\left(\tfrac{\nu+1}{2}\right)-\log\Gamma\!\left(\tfrac{\nu}{2}\right)-\tfrac12\log(\nu\pi)-\tfrac{\nu+1}{2}\log\!\left(1+\tfrac{x^2}{\nu}\right)$$

$$s(x;\nu)=\tfrac12\psi\!\left(\tfrac{\nu+1}{2}\right)-\tfrac12\psi\!\left(\tfrac{\nu}{2}\right)-\tfrac1{2\nu}
-\tfrac12\log\!\left(1+\tfrac{x^2}{\nu}\right)+\tfrac{\nu+1}{2}\cdot\frac{x^2/\nu^2}{1+x^2/\nu}$$

其中 $\psi$ 是 digamma 函数。$I(\nu)=E[s^2]$ 是 Cramér–Rao 下界 $\mathrm{CRLB}=1/(nI(\nu))$ 的分母。为了不靠记忆里的闭式公式，本文用三条独立路径算它：对 $s^2f$ 做无穷区间数值积分、40 万次蒙特卡洛取 $s^2$ 均值、以及 40 万样本上对 $\log f$ 做中心二阶差分取 $-E[\ell'']$。

先把公式本身钉死：解析 score 与中心差分一阶导在 20 万个点上逐点比对，最大绝对偏差 $2.98\times10^{-11}$；40 万样本上 $E[s]$ 最大绝对值 $1.31\times10^{-4}$，符合零均值 score 的抽样噪声量级。三条信息路径的最大偏差：积分对蒙特卡洛 $1.12\times10^{-4}$、积分对 Hessian $1.70\times10^{-5}$，且 $E[s]$ 最大 $1.31\times10^{-4}$。

![图1：Fisher 信息三路对拍与 Cramér–Rao 下界](/images/cramer-rao-mle-finite-sample/fig1_fisher_crlb.png)

这里要交代一个**诚实的数值边界**：$\nu=50$ 时 $I$ 只有 $5.21\times10^{-7}$，40 万次蒙特卡洛在这个量级上的相对抽样误差会到几个百分点（Hessian 路径给 $4.62\times10^{-7}$）。这不是公式分歧，是蒙特卡洛精度；正文引用的 $I(\nu)$ 一律以数值积分为准。$\nu\le10$ 的实务区间里三路吻合都在 1% 左右。

## 三、下界只是渐近极限：MLE 的有限样本偏倚

固定 $\nu_0=5$，在 $n=60/252/1000$ 上各跑 1500 次极大似然（一维有界优化，$\nu\in[1.01,400]$），结果：

| $n$ | $\hat\nu$ 均值 | 中位数 | 方差 | CRLB | 方差/CRLB | 发散率 |
|---|---|---|---|---|---|---|
| 60 | 26.6428 | 5.4178 | 7107.34 | 5.5140 | **1288.95** | 5.13% |
| 252 | 5.4902 | 5.0917 | 4.5589 | 1.3129 | 3.47 | 0% |
| 1000 | 5.0904 | 5.0162 | 0.3951 | 0.3308 | 1.19 | 0% |

![图3：三档样本量下 nu_hat 的右偏长尾分布](/images/cramer-rao-mle-finite-sample/fig3_mle_distributions.png)

$n=60$ 的均值 26.6428 与中位数 5.4178 差了近五倍，是典型的**右偏长尾**：多数样本落在真值附近，少数样本看起来「太像正态」，使似然在 $\nu\to\infty$ 方向单调趋平，$\hat\nu$ 一直顶到数值上界 400。这些发散样本把方差抬到 7107，也把 $n\cdot\mathrm{Var}$ 推到 42.6 万。图 4 左子图用对数轴把三档与下界常数画在一起，收敛路径一目了然。

**必须说清一个定理前提**：经典 Cramér–Rao 不等式约束的是**无偏估计量**的方差，而小样本 MLE 明显有偏（$n=60$ 偏倚 21.6428、$n=252$ 偏倚 0.4902）。所以 1289 倍不是「违反了下界」，而是定理的前提根本不成立；严谨比较应该用含 $\partial E[\hat\nu]/\partial\nu$ 的广义下界或均方误差。本文坚持报方差比，是为了量化实践中「按教科书下界估精度会乐观多少倍」，不是拿它当反例。

## 四、一个诚实的反转：Wald 区间这次没崩

按常见叙事，profile 似然右偏、Wald 的二次近似对称，Wald 区间应该严重欠覆盖。本文实测不是这样。

![图2：同一份样本的 profile 似然曲线与两种区间](/images/cramer-rao-mle-finite-sample/fig2_profile_skewness.png)

取一条 $n=252$ 样本，$\hat\nu=6.2124$。真实 profile 似然在 $\nu=60$ 处只比峰值低 9.931，而 Wald 的二次近似在同一处低了 514.275——**外推局部曲率会把下降速度高估约 52 倍**，这正是似然曲线右偏长尾的直接证据。相应地，Wald 区间是 $[2.9253,\,9.4995]$，似然比区间是 $[3.8322,\,11.7606]$，后者明显右移且更宽。

![图4：方差收敛与两种区间的真实覆盖率](/images/cramer-rao-mle-finite-sample/fig4_variance_coverage.png)

但 1500 次重复的覆盖率是 Wald **0.962**、似然比 **0.9587**，都略高于名义 0.95，Wald 甚至还高一点。机制不神秘：$I(\nu)$ 随 $\nu$ 增大而减小，$\hat\nu$ 右偏时 Wald 用的 $I(\hat\nu)$ 偏小，标准误自动变大（Wald 中位宽 4.6322、似然比中位宽 5.1197），偏斜与宽度互相抵消。

所以别急着宣布「Wald 没问题」。两种构造各有约 0.5% 的样本给出**物理上荒谬的边界**：Wald 有 0.47% 的下界为负（$\nu>0$ 是参数域硬约束），似然比有 0.33% 根本没有有限上界（似然在右端一直平到搜索边界）。这两类样本在自动化风控流水线里最危险，因为它们不会报错，只会吐出一个看起来正常的数。

## 五、尺度与尾厚度混淆：误差怎么传导到 VaR

把真实尺度 $\sigma_{\text{true}}$ 从 0.8 扫到 2.0，生成 $\sigma_{\text{true}}\cdot t(5)$ 的样本，但估计时**刻意把尺度固定为 1**（错设），每个设定 300 次重复：

| $\sigma_{\text{true}}$ | 0.8 | 1.0 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| $\hat\nu$ 中位数 | 12.235 | 5.208 | 3.261 | 2.153 | 1.422 |
| 均值 99% VaR | 2.694 | 3.331 | 4.317 | 6.343 | **12.603** |

尺度被低估一点，模型就只能用「更厚的尾」去解释同样宽的散布；尺度被高估，反过来把尾部显得更薄。真值 5 只在 $\sigma=1.0$ 附近被还原。这个实验是刻意错设以隔离效应，实务中应联合估计 $(\mu,\sigma,\nu)$；但联合估计同样存在三者耦合，只是混淆换了形式。

![图5：尺度混淆与 nu_hat 抽样噪声传导到 99% VaR](/images/cramer-rao-mle-finite-sample/fig5_scale_var.png)

即使尺度完全正确，$\hat\nu$ 的抽样噪声也不便宜：$t(5)$ 的真实 99% 分位是 3.3649，1500 次重复推出的均值 3.3562（相对偏差 $-0.26\%$）、中位数 3.3398，看起来几乎无偏；但单条 252 日样本的 5%–95% 带宽是 $[2.8589,\,3.8784]$，相对真值约 $-15\%$ 到 $+15\%$，53.4% 的样本低估了尾部风险。**均值无偏不等于单次估计可靠**——风控额度是按手上那一份样本算的，不是按 1500 次的平均。

## 六、边界与实战建议

- 别用 $\hat\nu$ 的点估计直接定 VaR 限额。要么给区间（优先似然比，并显式检查上界是否有限），要么在压力测试里把 $\nu$ 扫一遍。
- 估计 $\nu$ 前先把波动率模型定稳。本文的尺度混淆幅度远大于纯抽样噪声，波动率与尾厚度不可辨识是最先该排除的系统性误差。
- 短样本（$n\le60$）里 $\nu$ 常常根本不可辨识，5.13% 的发散率不是优化器坏了，是数据里没有信息。此时与其报 $\hat\nu=400$，不如直接声明「无法区分正态与厚尾」。
- 本文结论限定在 $\nu_0=5$、单参数、位置尺度已知或错设的设定；联合估计、条件异方差模型、以及 $\nu$ 接近 2（方差临界）时行为会不同，需要另做实验。
- 与本站《夏普比率的置信区间》《Anderson-Darling 正态性检验》以及本批《三大检验渐近等价》同属统计推断进阶系列：一篇修标准误，一篇选检验统计量，一篇讲参数区间，这一篇讲参数本身的信息量上限。

代码用固定种子 20261118，自包含可运行，会重画五张图并把全部统计量写入同目录 `results.json`。

## 七、完整可运行代码

```python
# -*- coding: utf-8 -*-
"""Cramér–Rao 下界与 Student-t 自由度的 MLE：有限样本偏倚、偏斜 profile 似然与 VaR 传导。

本文全部数字的唯一来源。运行本脚本会：
1. 用数值积分、40 万次蒙特卡洛、观测 Hessian 三条独立路径对拍 t 分布自由度 nu 的
   单观测 Fisher 信息 I(nu)，顺带核对 E[score]=0，给出 CRLB = 1/(n·I(nu))；
2. 在 nu0=5、n=60/252/1000 上各跑 1500 次 MLE，量化有限样本偏倚、方差相对
   CRLB 的膨胀以及估计发散到边界的比例；
3. 对同一批 n=252 样本算 Wald 区间与 profile 似然比区间的真实覆盖率；
4. 把真实尺度设错（sigma_true 扫描），展示波动率与尾厚度之间的混淆；
5. 把 nu_hat 的抽样分布传导到 99% VaR，量化尾部风险被低估的幅度；
6. 画 5 张图到 public/images/cramer-rao-mle-finite-sample/，
   全部统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, special, stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261118
_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "cramer-rao-mle-finite-sample",
)
os.makedirs(_OUT, exist_ok=True)

NU0 = 5.0
ALPHA = 0.05
CRIT = float(stats.chi2.ppf(1 - ALPHA, 1))
Z95 = float(stats.norm.ppf(1 - ALPHA / 2))


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


def logpdf_t(x, nu):
    """标准 Student-t（位置 0、尺度 1）的对数密度，对 nu 求导用。"""
    x = np.asarray(x, dtype=float)
    return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - 0.5 * (nu + 1) * np.log1p(x * x / nu))


def score_nu(x, nu):
    """∂/∂nu log f(x;nu)：digamma 项 + 常数项 + log(1+x²/nu) 项 + 有理项。"""
    x = np.asarray(x, dtype=float)
    q = x * x / nu
    return (0.5 * special.digamma((nu + 1) / 2)
            - 0.5 * special.digamma(nu / 2)
            - 0.5 / nu - 0.5 * np.log1p(q)
            + 0.5 * (nu + 1) * (x * x / (nu * nu)) / (1.0 + q))


def fisher_quad(nu):
    """I(nu) = E[score²]，用数值积分直接算（第一条独立路径）。"""
    def f(x):
        s = _scalar(score_nu(x, nu))
        lp = _scalar(logpdf_t(x, nu))
        return s * s * math.exp(lp)
    val, _ = integrate.quad(f, -np.inf, np.inf, limit=400, epsabs=1e-9)
    return float(val)


def neg_ell(nu, x2):
    """-sum log f(x_i;nu)，只依赖充分统计量 x² 数组。"""
    n = x2.size
    c = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
         - 0.5 * np.log(nu * np.pi))
    return float(0.5 * (nu + 1) * np.log1p(x2 / nu).sum() - n * float(c))


def mle_nu(x):
    x2 = np.asarray(x, dtype=float) ** 2
    r = optimize.minimize_scalar(lambda v: neg_ell(v, x2),
                                 bounds=(1.01, 400.0), method="bounded",
                                 options={"xatol": 1e-7})
    return float(r.x), float(r.fun)


def lr_interval(x2, nu_hat, ell_hat):
    """profile 似然比区间：{nu: 2[ell(nu_hat)-ell(nu)] <= chi2_0.95(1)}。"""
    def dev(v):
        return 2.0 * (neg_ell(v, x2) - ell_hat) - CRIT

    left = 1.01
    if dev(1.01) > 0:
        left = float(optimize.brentq(dev, 1.01, nu_hat, xtol=1e-8))
    right = math.inf
    if dev(400.0) > 0:
        right = float(optimize.brentq(dev, nu_hat, 400.0, xtol=1e-8))
    return left, right


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "nu0": NU0, "alpha": ALPHA, "chi2_crit": CRIT, "figs": []}

    # ---- 实验 1：Fisher 信息三路对拍 ----
    nus = [2.5, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0]
    # 确定性物理校验：解析 score 与中心差分一阶导逐点对拍（独立随机流，不扰动主 rng）
    x_chk = stats.t.rvs(5.0, size=200000, random_state=np.random.default_rng(SEED + 1))
    h_chk = 1e-4
    num_score = (logpdf_t(x_chk, 5.0 + h_chk) - logpdf_t(x_chk, 5.0 - h_chk)) / (2 * h_chk)
    score_dev = float(np.max(np.abs(num_score - score_nu(x_chk, 5.0))))
    fish = {}
    for nu in nus:
        Iq = fisher_quad(nu)
        xb = stats.t.rvs(nu, size=400000, random_state=rng)
        sc = score_nu(xb, nu)
        Imc = float(np.mean(sc ** 2))
        Escore = float(np.mean(sc))
        h = 1e-2
        Ih = float(-np.mean((logpdf_t(xb, nu + h) - 2 * logpdf_t(xb, nu)
                             + logpdf_t(xb, nu - h)) / h ** 2))
        fish[f"nu{nu}"] = {
            "I_quad": Iq, "I_mc": Imc, "I_hess": Ih, "E_score": Escore,
            "crlb_n252": 1.0 / (252 * Iq),
        }
    results["fisher_info"] = {
        "nus": nus, "rows": fish,
        "max_abs_dev_quad_vs_mc": max(abs(fish[f"nu{v}"]["I_quad"] - fish[f"nu{v}"]["I_mc"]) for v in nus),
        "max_abs_dev_quad_vs_hess": max(abs(fish[f"nu{v}"]["I_quad"] - fish[f"nu{v}"]["I_hess"]) for v in nus),
        "max_abs_E_score": max(abs(fish[f"nu{v}"]["E_score"]) for v in nus),
        "crlb_ratio_nu10_over_nu3": fish["nu10.0"]["crlb_n252"] / fish["nu3.0"]["crlb_n252"],
        "score_max_abs_dev": score_dev,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    ax = axes[0]
    ax.plot(nus, [fish[f"nu{v}"]["I_quad"] for v in nus], "-", color="#2c3e50", lw=2.2,
            label="数值积分 E[score²]")
    ax.plot(nus, [fish[f"nu{v}"]["I_mc"] for v in nus], "o", color="#c0392b", ms=6,
            label="40 万次蒙特卡洛")
    ax.plot(nus, [fish[f"nu{v}"]["I_hess"] for v in nus], "s", color="#2980b9", ms=5,
            label="观测 Hessian")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("自由度 ν")
    ax.set_ylabel("单观测 Fisher 信息 I(ν)")
    ax.set_title("三条独立路径对拍：ν 越大信息越少", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(nus, [fish[f"nu{v}"]["crlb_n252"] for v in nus], "o-", color="#e67e22", lw=2.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("自由度 ν")
    ax.set_ylabel("n=252 时 Var(ν̂) 的 Cramér–Rao 下界")
    ax.set_title("厚尾参数天生难估：ν=10 的下界是 ν=3 的"
                 f" {results['fisher_info']['crlb_ratio_nu10_over_nu3']:.1f} 倍", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("图1：Fisher 信息与 Cramér–Rao 下界的现场计算", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig1_fisher_crlb.png"))

    # ---- 实验 2：MLE 有限样本分布 ----
    ns = [60, 252, 1000]
    reps = 1500
    mc = {}
    samples_252 = []
    hats_by_n = {}
    for n in ns:
        X = stats.t.rvs(NU0, size=(reps, n), random_state=rng)
        hats = []
        ells = []
        for i in range(reps):
            v, e = mle_nu(X[i])
            hats.append(v)
            ells.append(e)
            if n == 252:
                samples_252.append((X[i] ** 2, v, e))
        hats = np.array(hats)
        hats_by_n[n] = hats
        I0 = fish["nu5.0"]["I_quad"]
        crlb = 1.0 / (n * I0)
        mc[f"n{n}"] = {
            "mean": float(hats.mean()), "median": float(np.median(hats)),
            "std": float(hats.std(ddof=1)), "var": float(hats.var(ddof=1)),
            "q05": float(np.quantile(hats, 0.05)), "q95": float(np.quantile(hats, 0.95)),
            "crlb": crlb, "var_over_crlb": float(hats.var(ddof=1) / crlb),
            "diverge_rate_over100": float((hats > 100).mean()),
            "bias": float(hats.mean() - NU0),
        }
    results["mle_mc"] = {"reps": reps, "rows": mc}

    # 图 3：三档 n 的 ν̂ 分布
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9), sharey=True)
    for ax, n in zip(axes, ns):
        row = mc[f"n{n}"]
        hats_draw = hats_by_n[n]
        ax.hist(np.clip(hats_draw, 0, 40), bins=60, range=(0, 40),
                color="#2980b9", alpha=0.75)
        ax.axvline(NU0, color="#c0392b", ls="--", lw=1.8, label=f"真值 ν0={NU0:.0f}")
        ax.axvline(row["mean"], color="#e67e22", ls="-", lw=1.6,
                   label=f"均值 {row['mean']:.2f}")
        ax.axvline(row["median"], color="#16a085", ls=":", lw=1.6,
                   label=f"中位数 {row['median']:.2f}")
        ax.set_xlabel("ν̂")
        ax.set_title(f"n={n}，发散率 {row['diverge_rate_over100']:.3f}\n"
                     f"Var/CRLB = {row['var_over_crlb']:.2f} 倍", fontsize=10)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("频数（1500 次重复）")
    fig.suptitle("图3：小样本下 ν̂ 右偏长尾，均值被少数发散样本拖高", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig3_mle_distributions.png"))

    # ---- 实验 3：Wald 区间 vs profile 似然比区间 ----
    igrid = np.geomspace(1.5, 400.0, 200)
    ivals = np.array([fisher_quad(v) for v in igrid])

    def I_of(v):
        return float(np.interp(v, igrid, ivals))

    wald_cov = 0
    lr_cov = 0
    wald_neg = 0
    lr_infinite = 0
    wald_widths = []
    lr_widths = []
    for x2, vhat, ehat in samples_252:
        se = 1.0 / math.sqrt(252 * I_of(vhat))
        wlo, whi = vhat - Z95 * se, vhat + Z95 * se
        if wlo < 0:
            wald_neg += 1
        wald_cov += int(wlo <= NU0 <= whi)
        wald_widths.append(whi - wlo)
        llo, lhi = lr_interval(x2, vhat, ehat)
        if math.isinf(lhi):
            lr_infinite += 1
        else:
            lr_widths.append(lhi - llo)
        lr_cov += int(llo <= NU0 <= lhi)
    results["intervals"] = {
        "n": 252, "reps": reps,
        "wald_coverage": wald_cov / reps, "lr_coverage": lr_cov / reps,
        "wald_negative_lower_rate": wald_neg / reps,
        "lr_unbounded_upper_rate": lr_infinite / reps,
        "wald_median_width": float(np.median(wald_widths)),
        "lr_median_bounded_width": float(np.median(lr_widths)),
    }

    # 图 2：一份样本的 profile 似然曲线与两种区间
    x2_demo, vh_demo, eh_demo = samples_252[0]
    grid = np.linspace(1.5, 60.0, 400)
    prof = np.array([-(neg_ell(v, x2_demo) - eh_demo) for v in grid])
    quad_app = -0.5 * 252 * I_of(vh_demo) * (grid - vh_demo) ** 2
    wlo_d, whi_d = vh_demo - Z95 / math.sqrt(252 * I_of(vh_demo)), vh_demo + Z95 / math.sqrt(252 * I_of(vh_demo))
    llo_d, lhi_d = lr_interval(x2_demo, vh_demo, eh_demo)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(grid, prof, "-", color="#2c3e50", lw=2.3, label="profile 对数似然（减最大值）")
    ax.plot(grid, quad_app, "--", color="#c0392b", lw=1.8, label="Wald 的二次近似")
    ax.axhline(-CRIT / 2, color="k", ls=":", lw=1.4, label="95% 似然比阈值 −χ²/2")
    ax.axvspan(max(0.0, wlo_d), whi_d, color="#c0392b", alpha=0.13)
    ax.axvspan(llo_d, min(lhi_d, 60.0), color="#2980b9", alpha=0.13)
    ax.axvline(NU0, color="#16a085", ls="-", lw=1.7, label=f"真值 ν0={NU0:.0f}")
    ax.text(0.02, 0.08, f"Wald 区间 [{wlo_d:.2f}, {whi_d:.2f}]\n"
                        f"似然比区间 [{llo_d:.2f}, {lhi_d if math.isfinite(lhi_d) else 400:.2f}]",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="#bdc3c7", alpha=0.9))
    ax.set_xlabel("自由度 ν")
    ax.set_ylabel("ℓ(ν) − ℓ(ν̂)")
    ax.set_title(f"图2：同一份 n=252 样本，ν̂={vh_demo:.2f}——似然曲线右偏且二次近似不对称", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    # 二次近似在大范围内远高于真实曲率，若按它的量级设 y 轴会把真实曲线压成平线。
    # 这里以真实 profile 曲线的下界为准，二次近似超出画面部分由 matplotlib 裁剪。
    ylo = max(-30.0, float(prof.min()) * 1.15)
    ax.set_ylim(ylo, 0.6)
    ax.text(0.98, 0.34, "二次近似在右端已跌出画面：\n局部曲率外推到全局会高估下降速度",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="#c0392b")
    fig.tight_layout()
    results["figs"].append(_save(fig, "fig2_profile_skewness.png"))
    results["fig2_demo"] = {
        "nu_hat": float(vh_demo),
        "prof_at_nu60": float(prof[-1]),
        "quad_at_nu60": float(quad_app[-1]),
        "wald_interval": [float(wlo_d), float(whi_d)],
        "lr_interval": [float(llo_d), 400.0 if math.isinf(lhi_d) else float(lhi_d)],
        "y_axis_lower": ylo,
    }

    # 图 4：方差膨胀收敛 + 覆盖率对比
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    ax.plot(ns, [mc[f"n{n}"]["var"] * n for n in ns], "o-", color="#c0392b", lw=2.1,
            label="实测 n·Var(ν̂)")
    ax.axhline(1.0 / fish["nu5.0"]["I_quad"], color="k", ls="--", lw=1.6,
               label=f"CRLB 常数 1/I(ν0) = {1.0 / fish['nu5.0']['I_quad']:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(200.0, 1e6)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("样本量 n")
    ax.set_ylabel("n·Var(ν̂)（对数轴）")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("MLE 方差向下界收敛，小样本远达不到", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    vals = [results["intervals"]["wald_coverage"], results["intervals"]["lr_coverage"]]
    bars = ax.bar(["Wald 区间", "profile 似然比区间"], vals,
                  color=["#c0392b", "#2980b9"], alpha=0.85, width=0.5)
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=1.4, label="名义 0.95")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("真值覆盖率")
    ax.set_title(f"Wald 下界为负的比例 {results['intervals']['wald_negative_lower_rate']:.3f}，"
                 f"\n似然比上界无穷的比例 {results['intervals']['lr_unbounded_upper_rate']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("图4：两种区间都盖住了 95%，但各有约 0.5% 给出荒谬边界", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig4_variance_coverage.png"))

    # ---- 实验 4：波动率与尾厚度混淆 ----
    sigmas = [0.8, 1.0, 1.2, 1.5, 2.0]
    reps_s = 300
    conf = {}
    var_true = float(stats.t.ppf(0.99, NU0))
    hats_252 = np.array([v for _, v, _ in samples_252])
    var_hats = stats.t.ppf(0.99, hats_252)
    for s in sigmas:
        X = s * stats.t.rvs(NU0, size=(reps_s, 252), random_state=rng)
        hs = np.array([mle_nu(X[i])[0] for i in range(reps_s)])
        conf[f"sigma{s}"] = {
            "mean_nu_hat": float(hs.mean()), "median_nu_hat": float(np.median(hs)),
            "bias": float(hs.mean() - NU0),
            "mean_var99": float(np.mean(stats.t.ppf(0.99, hs))),
        }
    results["scale_confusion"] = {
        "sigmas": sigmas, "reps": reps_s, "rows": conf,
        "var99_true": var_true,
        "mc_var99": {
            "mean": float(np.mean(var_hats)),
            "median": float(np.median(var_hats)),
            "q05": float(np.quantile(var_hats, 0.05)),
            "q95": float(np.quantile(var_hats, 0.95)),
            "underestimate_rate": float((var_hats < var_true).mean()),
            "rel_bias": float(np.mean(var_hats) / var_true - 1.0),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    ax.plot(sigmas, [conf[f"sigma{s}"]["median_nu_hat"] for s in sigmas],
            "o-", color="#8e44ad", lw=2.1, label="ν̂ 中位数")
    ax.plot(sigmas, [conf[f"sigma{s}"]["mean_nu_hat"] for s in sigmas],
            "s--", color="#e67e22", lw=1.7, label="ν̂ 均值")
    ax.axhline(NU0, color="k", ls="--", lw=1.4, label=f"真值 ν0={NU0:.0f}")
    ax.set_xlabel("真实尺度 σ_true（估计时一律错设为 1）")
    ax.set_ylabel("ν̂")
    ax.set_title("尺度设大一点，模型就用更厚的尾去解释", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(var_hats, bins=70, range=(2.0, 5.0), color="#2980b9", alpha=0.75)
    ax.axvline(var_true, color="#c0392b", ls="--", lw=1.9,
               label=f"真值 99% VaR = {var_true:.3f}")
    ax.axvline(np.mean(var_hats), color="#e67e22", ls="-", lw=1.7,
               label=f"估计均值 {np.mean(var_hats):.3f}")
    ax.set_xlabel("由 ν̂ 推出的 99% VaR")
    ax.set_ylabel("频数")
    mc_v = results["scale_confusion"]["mc_var99"]
    ax.axvline(mc_v["q05"], color="#8e44ad", ls=":", lw=1.6, label="5% / 95% 分位")
    ax.axvline(mc_v["q95"], color="#8e44ad", ls=":", lw=1.6)
    ax.set_title(f"均值只偏 {mc_v['rel_bias']:+.2%}，但 5%–95% 带宽 "
                 f"[{mc_v['q05']:.2f}, {mc_v['q95']:.2f}]", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("图5：ν̂ 的抽样噪声如何变成真金白银的 VaR 误差", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig5_scale_var.png"))

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
```


## 八、结果文件

五张图与 `results.json` 位于 `/images/cramer-rao-mle-finite-sample/`，包含 Fisher 信息三路对拍、三档样本量的 MLE 分布、两种区间覆盖率、尺度扫描与 VaR 分位数的全部条目，可逐项核对正文表格。
