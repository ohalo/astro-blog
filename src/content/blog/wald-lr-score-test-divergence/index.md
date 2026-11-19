---
title: "三大检验渐近等价，换个坐标系结论就翻转：Wald、似然比与 Score 的有限样本实测"
description: "Wald、似然比 LR 与 Score/LM 三大检验在教科书里渐近等价，但同一份数据可以给出不同判决。本文在 AR(1) 条件高斯似然下把 sigma^2 profile 掉，解析得到三者都是同一个充分统计量之比 u 的函数：LM = n·u/(1+u) ≤ LR = n·log(1+u) ≤ Wald = n·u，20000 次蒙特卡洛零违反。实测 n=252、原假设为真时 size 依次 0.05365 / 0.05265 / 0.0516，分歧仅 0.2 个百分点；n=30 时 1.28 个百分点，n=1000 收敛到 0.017 个百分点，功效曲线三家最大差 0.0073——渐近等价在这个设定里相当准。真正的坑与样本量无关：用一族光滑正则重参数化 phi = psi/(1-a·psi) 重跑，LR 恒 2.9838、Fisher 版 LM 恒 2.9662，Wald 却从 3.0015 涨到 30.1447，4000 条样本里结论翻转比例升到 26.85%；把 LM 的 Fisher 信息换成观测 Hessian，二阶项让信息在 a≥5 变负，统计量出现 -27.4409 的荒谬值。另给厚尾与 GARCH 条件异方差对照：t(5) 让三家一起保守到 0.0475-0.0485，条件异方差才把 size 顶到 0.0863-0.0898。附完整代码与五张真实计算图。"
publishDate: '2026-11-18T11:10:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 假设检验
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

计量教科书会说：Wald 检验、似然比检验（LR）与 Score 检验（也叫 LM、拉格朗日乘子检验）在大样本下都收敛到同一个 $\chi^2$ 分布，用谁都行。这句话有两个前提常被忽略——**样本要够大，参数坐标系不能乱换**。本文用一条 $AR(1)$ 收益序列把这两件事拆开算清楚。

所有数字来自 252 个交易日、$\phi_0=0$ 的条件高斯似然蒙特卡洛（重复 20000 次），$\chi^2(1)$ 的 5% 临界值取 3.8415：

| 统计量 | 解析形式 | 经验 95% 分位 | 实际 size |
|---|---|---|---|
| Wald | $n\,u$ | 3.9475 | **0.05365** |
| LR | $n\log(1+u)$ | 3.9169 | 0.05265 |
| LM / Score | $n\,u/(1+u)$ | 3.8866 | **0.05160** |

四个结论：

1. 三者不是「差不多相等」，而是**同一个量 $u$ 的三个不同函数**，因此恒有 $\mathrm{LM}\le\mathrm{LR}\le\mathrm{Wald}$。20000 次重复里违反次数为 0，$LM-LR$ 最大仅 $4.02\times10^{-14}$、$LR-W$ 最大 $4.93\times10^{-14}$，纯粹是浮点尾数。
2. 在似然设定正确的线性模型里，**渐近等价其实相当好用**。Wald 与 LM 的 size 之差在 $n=30$ 时是 1.283 个百分点，$n=252$ 缩到 0.083 个百分点，$n=1000$ 只剩 0.017 个百分点；共同随机数下的功效曲线三家最大差只有 0.0073。
3. 但 Wald 有一个**与样本量无关的结构性缺陷**：它依赖参数的坐标系。固定一份样本、固定同一个原假设，只把 $\phi$ 换成一族光滑可逆的 $\psi$，Wald 从 3.0015 一路涨到 30.1447，LR 与 Fisher 信息版 LM 纹丝不动。4000 条样本里判决翻转的比例最高达 26.85%。
4. LM 的「不变」是有条件的。若把它的 Fisher 信息换成观测 Hessian，重参数化的二阶项会让信息在 $a\ge5$ 变成负数，统计量算出 $-27.4409$ 这种没有意义的数。

## 二、把方差 profile 掉：三个检验原来是同一个数的三个函数

设 $x_t=\phi x_{t-1}+\varepsilon_t$、$\varepsilon_t\sim N(0,\sigma^2)$，检验 $H_0:\phi=\phi_0$。给定初值后，条件对数似然里的 $\sigma^2$ 可以 profile 掉，剩下全部信息压在三个充分统计量里：$S_0=\sum x_{t-1}^2$、$S_1=\sum x_{t-1}x_t$、$S_2=\sum x_t^2$。于是 $\hat\phi=S_1/S_0$，无约束残差平方和 $\mathrm{RSS}(\hat\phi)=S_2-S_1^2/S_0$，而约束残差平方和满足平方和分解：

$$
\mathrm{RSS}(\phi_0)=\mathrm{RSS}(\hat\phi)+(\hat\phi-\phi_0)^2S_0
$$

定义 $u=(\hat\phi-\phi_0)^2S_0/\mathrm{RSS}(\hat\phi)$，则 $\mathrm{RSS}(\phi_0)/\mathrm{RSS}(\hat\phi)=1+u$。把 $\hat\sigma^2=\mathrm{RSS}(\hat\phi)/n$、$\tilde\sigma^2=\mathrm{RSS}(\phi_0)/n$ 分别代进三个统计量的标准定义，得到三条闭式：

$$
W=\frac{(\hat\phi-\phi_0)^2S_0}{\hat\sigma^2}=n\,u,\qquad
\Lambda=n\log\frac{\tilde\sigma^2}{\hat\sigma^2}=n\log(1+u),\qquad
LM=\frac{(\hat\phi-\phi_0)^2S_0}{\tilde\sigma^2}=\frac{n\,u}{1+u}
$$

因为对 $u\ge0$ 恒有 $\frac{u}{1+u}\le\log(1+u)\le u$，三者的大小顺序是**代数恒等式**，不是抽样噪声。这也解释了为什么 Wald 永远最激进、LM 永远最保守：它们把同一个证据强度 $u$ 放大或压缩了。

![图1：三个统计量在原假设下的经验分布与 chi2(1) 密度](/images/wald-lr-score-test-divergence/fig1_h0_distributions.png)

图 1 把 20000 次重复的尾部直方图叠在 $\chi^2(1)$ 理论密度上。三条经验 95% 分位（3.9475 / 3.9169 / 3.8866）都在名义临界 3.8415 附近但方向确定，对应的 size 依次 0.05365 / 0.05265 / 0.05160。差异不大，却不是随机波动——它是同一条不等式在分布尾部的投影。

![图2：三个统计量作为 u 的解析函数与跨线位置](/images/wald-lr-score-test-divergence/fig2_analytic_ordering.png)

图 2 更直接：把三者画成 $u$ 的函数，跨过临界线所需的 $u$ 分别是 0.015244（Wald）、0.015361（LR）、0.015480（LM）。也就是说，只要证据强度落在这三个数之间，同一份样本、同一个模型、同一个 5% 门槛，三家就会给出不同判决。灰色分歧带随 $u$ 单调变宽，这是有限样本分歧的全部来源。

## 三、渐近等价到底有多准：样本量、厚尾与条件异方差

扫 $n=30,60,120,252,504,1000$，每档 6000 次重复，三种新息都用**同一批标准化新息**（共同随机数），避免把初始化噪声误读成方法差异。

![图3：size 随样本量收敛，三种新息对照](/images/wald-lr-score-test-divergence/fig3_size_vs_n.png)

正态新息下，$n=30$ 时 Wald / LR / LM 的 size 是 0.05517 / 0.04883 / 0.04233，$n=252$ 是 0.05200 / 0.05150 / 0.05117，$n=1000$ 三家收敛到 0.05467 / 0.05450 / 0.05450。分歧确实随 $n$ 消失，但注意 $n=1000$ 时三家一起略高于 0.05——这是 6000 次重复的二项抽样噪声（标准误约 0.0028）叠加有限样本偏差，不是 $\chi^2$ 近似失效。

两个容易搞反的对照值得单列：

- **厚尾不是死穴**。把新息换成标准化 $t(5)$（方差仍为 1），$n=252$ 的 size 是 0.04850 / 0.04783 / 0.04750，三家一起变保守而非膨胀。原因是 OLS 的 $\hat\phi$ 在厚尾下仍然无偏，而 $\tilde\sigma^2$ 的估计偏大，把三个统计量一起压小了。
- **条件异方差才是**。换成 GARCH(1,1)（$\omega=0.05$、$a=0.10$、$b=0.85$，持久性 0.95、无条件方差仍为 1），$n=252$ 的 size 跳到 0.08983 / 0.08733 / 0.08633，$n=1000$ 进一步到 0.09717 / 0.09717 / 0.09650。**样本越多越糟**，因为误设的同方差似然不会随 $n$ 自动变对，得靠 HAC / White 标准误去修。这一点与本站《夏普比率的置信区间》里 $\phi=0.4$ 把 i.i.d. 覆盖率打到 0.7995 是同一类病，只是换了个统计量。

![图5：共同随机数下的功效曲线与三家功效差](/images/wald-lr-score-test-divergence/fig5_power_curve.png)

图 5 给出 $\phi$ 从 0 扫到 0.30 的功效。$\phi=0.15$ 时三家是 0.6737 / 0.6700 / 0.6663，$\phi=0.30$ 时都到 0.997 以上，全程最大差 0.0073。这里要诚实交代一个**没有出现**的现象：文献里常提到 Score 统计量在远离原假设时可能功效非单调，因为它在 $H_0$ 处评估信息矩阵；本文的 $AR(1)$ 设定下 LM 均值从 0.987 单调升到 23.27，远没触及 $n=252$ 的理论上界，功效曲线也一路上升。别把这个结论外推到强非线性或大偏离场景。

## 四、坐标依赖：换参数化就能翻转判决

这是全文最该记住的一节。取一族光滑、在原假设处导数非零（$g'(0)=1$，满足正则条件）的重参数化：

$$
\phi=g_a(\psi)=\frac{\psi}{1-a\psi},\qquad \psi=\frac{\phi}{1+a\phi},\qquad a\ge0
$$

$a=0$ 就是原始 $\phi$ 坐标系。$H_0:\phi=0$ 与 $H_0:\psi=0$ 是**同一个假设**，profile 似然的最大值也一模一样，所以 LR 不变；链式法则下 $U(\psi_0)=g'(\psi_0)U(\phi_0)$ 与 $I(\psi_0)=g'(\psi_0)^2I(\phi_0)$ 中 $g'$ 抵消，所以 Fisher 信息版 LM 也不变。Wald 却要用 $\hat\psi$ 处的信息，多出一个 $(1+a\hat\phi)^2$ 因子：

$$
W_\psi=W_\phi\,(1+a\hat\phi)^2
$$

![图4：重参数化下只有 Wald 变化，翻转率随 a 上升](/images/wald-lr-score-test-divergence/fig4_reparameterization.png)

取一条 $\phi_{\text{true}}=0.15$ 的样本，$\hat\phi=0.10846$、$W_\phi=3.0015$（差一点点就跨过 3.8415）。把 $a$ 从 0 扫到 20：

| $a$ | Wald($\psi$) | LR($\psi$) | LM  Fisher 版 | LM 观测信息版 | 判决翻转率 |
|---|---|---|---|---|---|
| 0 | 3.0015 | 2.9838 | 2.9662 | 3.0377 | 0 |
| 2 | 4.4448 | 2.9838 | 2.9662 | 5.4662 | 10.48% |
| 5 | 7.1394 | 2.9838 | 2.9662 | **-27.4409** | 17.60% |
| 20 | **30.1447** | 2.9838 | 2.9662 | -0.8823 | **26.85%** |

LR 是**数值重跑**得到的：每个 $a$ 都以 $\psi$ 为参数重新极大化同一条 profile 似然，与 $\phi$ 坐标系的最大偏差仅 $4.00\times10^{-14}$；Fisher 版 LM 最大偏差 $1.30\times10^{-7}$（数值一阶差分精度）。Wald 与解析式 $(1+a\hat\phi)^2$ 的比值全程 1.000000，$\hat\psi$ 的数值解与闭式解逐位一致。

观测信息版 LM 为什么崩？链式展开里 $\ell''_{\psi\psi}=g''(\psi_0)\ell'_\phi+g'(\psi_0)^2\ell''_{\phi\phi}$，$g''(0)=2a$ 这一项不抵消。$a=5$ 时观测信息从 246.235 掉到 $-27.258$，$a=20$ 掉到 $-847.738$——分母变负，统计量自然算出负数。所以「LM 检验不变」这句话，严格说是**用期望 Fisher 信息构造的 Score 检验**才成立。这正是 Buse（1982）与 Engle（1984）在综述三大检验时反复强调的区别。

## 五、实战取舍与边界

- **要用重参数化、非线性参数或边界附近的参数，就用 LR 或 Fisher 信息版 LM**，别用 Wald。本文的翻转率 26.85% 不是极端构造：$a=2$ 这样一个温和的坐标变换就有 10.48%。
- **小样本别硬套 $\chi^2$**。$n=30$ 时三家 size 从 0.042 到 0.055 散开，线性模型里其实有精确的 $F$ / $t$ 检验可用。
- **似然误设时三家一起错**，而且样本越多越不收敛（GARCH 对照）。先修标准误，再谈选哪个检验。
- **别把本文的「分歧很小」读成「可以随便用」**。分歧小只发生在正确设定的线性高斯、参数化固定、原假设为内点的窄场景；本文的核心反例恰恰说明换一个前提结论就能翻。
- 与本站《夏普比率的置信区间》《Anderson-Darling 正态性检验》同属统计推断进阶系列：前者讲 HAC 与区块自助怎么修标准误，后者讲检验统计量在厚尾下的功效排序，这一篇讲检验统计量本身从哪来、什么时候会失效。

所有实验用固定随机种子 20261118，代码块自包含、可直接运行；它会重画五张图并把全部统计量写进同一目录的 `results.json`。图表与正文的每一个数字都来自这次真实计算。

## 六、完整可运行代码

```python
# -*- coding: utf-8 -*-
"""Wald / 似然比 / Score 三大渐近等价检验：有限样本下为何给出不同结论。

本文全部数字的唯一来源。运行本脚本会：
1. 在 AR(1) 条件高斯似然下把 sigma^2 profile 掉，解析得到三个统计量都是
   同一个量 u 的函数：LM = n*u/(1+u) <= LR = n*log(1+u) <= Wald = n*u；
2. 用 20000 次 H0 蒙特卡洛核对三者的经验分布、95% 分位与实际 size；
3. 扫描 n=30..1000 与正态/t(5) 两种新息，看 size 随样本量收敛的速度；
4. 用一族光滑正则重参数化 phi = psi/(1-a*psi) 重跑三个检验，
   核对 LR 与 Fisher 版 LM 不变、Wald 随 a 变化（解析比 (1+a*phi_hat)^2），
   并统计结论翻转率；
5. 共同随机数下画功效曲线，观察 Score 检验在远离原假设时的非单调性；
6. 画 5 张图到 public/images/wald-lr-score-test-divergence/，
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
from scipy.optimize import minimize_scalar

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261118
_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "wald-lr-score-test-divergence",
)
os.makedirs(_OUT, exist_ok=True)

ALPHA = 0.05
CRIT = float(stats.chi2.ppf(1 - ALPHA, 1))  # 3.841458820694124
BURN = 60


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def innovations(nrep, total, rng, innov="normal", df=5):
    if innov == "normal":
        return rng.standard_normal((nrep, total))
    if innov == "t5":
        return rng.standard_t(df, size=(nrep, total)) / math.sqrt(df / (df - 2.0))
    if innov == "garch":
        # GARCH(1,1) 条件异方差：omega=0.05, a=0.10, b=0.85，
        # 持久性 0.95、无条件方差 omega/(1-a-b)=1.0，与前两种新息同尺度。
        z = rng.standard_normal((nrep, total))
        s2 = np.ones(nrep)
        ep2 = np.ones(nrep)
        E = np.empty((nrep, total))
        om, ga, gb = 0.05, 0.10, 0.85
        for t in range(total):
            s2 = om + ga * ep2 + gb * s2
            e = np.sqrt(s2) * z[:, t]
            E[:, t] = e
            ep2 = e * e
        return E
    raise ValueError(innov)


def simulate(phi, n, E, burn=BURN):
    """给定同一批新息 E（共同随机数），模拟 AR(1) 并返回三个充分统计量。

    S0 = sum x_{t-1}^2, S1 = sum x_{t-1} x_t, S2 = sum x_t^2。
    丢弃 burn 步预热，避免初值 0 带来的非平稳偏差。
    """
    total = E.shape[1]
    x = np.zeros(E.shape[0])
    for t in range(total):
        x = phi * x + E[:, t]
        if t == burn - 1:
            x0 = x.copy()
        if t >= burn:
            if t == burn:
                s0 = x0 * x0
                s1 = x0 * x
                s2 = x * x
            else:
                s0 = s0 + prev * prev
                s1 = s1 + prev * x
                s2 = s2 + x * x
        prev = x
    return s0, s1, s2


def three_stats(s0, s1, s2, n, phi0=0.0):
    """profile 掉 sigma^2 后三个检验统计量的解析式。"""
    ph = s1 / s0
    rss_u = s2 - s1 * s1 / s0
    rss_r = s2 - 2.0 * phi0 * s1 + phi0 * phi0 * s0
    u = (ph - phi0) ** 2 * s0 / rss_u
    W = n * u
    LR = n * np.log(rss_r / rss_u)
    LM = n * (ph - phi0) ** 2 * s0 / rss_r
    return ph, W, LR, LM, rss_u, rss_r


def g_a(psi, a):
    return psi / (1.0 - a * psi)


def psi_of_phi(phi, a):
    return phi / (1.0 + a * phi)


def reparam_wald(phi_hat, s0, sig2_u, a):
    """Wald 在 psi 坐标系下的值：I_psi(psi_hat) = g'(psi_hat)^2 * s0 / sig2_u。

    g'(psi) = 1/(1-a*psi)^2，代入 psi_hat 得 (1+a*phi_hat)^2。
    """
    ps_hat = psi_of_phi(phi_hat, a)
    gp = 1.0 / (1.0 - a * ps_hat) ** 2
    return (ps_hat ** 2) * (gp ** 2) * s0 / sig2_u


def obs_info_lm(x_series, a, phi0=0.0):
    """用观测信息（profile 似然的数值 Hessian）算的 LM，检验不变性边界。"""
    n = len(x_series) - 1
    xl = x_series[:-1]
    xr = x_series[1:]
    s0 = float((xl * xl).sum())
    s1 = float((xl * xr).sum())
    s2 = float((xr * xr).sum())
    rss_r = s2 - 2.0 * phi0 * s1 + phi0 * phi0 * s0
    sig2_r = rss_r / n

    def ell(psi):
        ph = g_a(psi, a)
        rss = s2 - 2.0 * ph * s1 + ph * ph * s0
        return -0.5 * n * math.log(rss / n)

    h = 1e-5
    d1 = (ell(h) - ell(-h)) / (2.0 * h)
    d2 = (ell(h) - 2.0 * ell(0.0) + ell(-h)) / (h * h)
    info_obs = -d2
    info_fisher = s0 / sig2_r  # g'(0)=1，链式后不变
    return float(d1 * d1 / info_obs), float(d1 * d1 / info_fisher), float(info_obs)


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "alpha": ALPHA, "chi2_crit": CRIT, "figs": []}

    # ---- 实验 1：H0 下三统计量的经验分布 ----
    n0, rep0 = 252, 20000
    E0 = innovations(rep0, n0 + BURN, rng)
    s0, s1, s2 = simulate(0.0, n0, E0)
    ph, W, LR, LM, rss_u, rss_r = three_stats(s0, s1, s2, n0)
    dist = {}
    for key, v in [("wald", W), ("lr", LR), ("lm", LM)]:
        dist[key] = {
            "mean": float(v.mean()),
            "q95": float(np.quantile(v, 0.95)),
            "q99": float(np.quantile(v, 0.99)),
            "size": float((v > CRIT).mean()),
        }
    results["h0_distribution"] = {
        "n": n0, "reps": rep0, **dist,
        "ordering_violations": int(((LM > LR + 1e-9) | (LR > W + 1e-9)).sum()),
        "max_lm_minus_lr": float((LM - LR).max()),
        "max_lr_minus_wald": float((LR - W).max()),
    }

    # 图1：三统计量尾部直方图 + chi2(1) 密度 + 经验 95% 分位
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharex=True)
    grid = np.linspace(0.001, 9.0, 400)
    dens = stats.chi2.pdf(grid, 1)
    for ax, (key, v, label, color) in zip(
        axes,
        [("wald", W, "Wald = n·u", "#c0392b"),
         ("lr", LR, "LR = n·log(1+u)", "#e67e22"),
         ("lm", LM, "LM = n·u/(1+u)", "#2980b9")],
    ):
        ax.hist(v, bins=110, range=(0, 9), density=True, color=color, alpha=0.55)
        ax.plot(grid, dens, "k-", lw=1.5, label="χ²(1) 理论密度")
        ax.axvline(CRIT, color="k", ls="--", lw=1.3, label=f"名义临界 {CRIT:.3f}")
        q = dist[key]["q95"]
        ax.axvline(q, color=color, ls="-", lw=2.0, label=f"经验95%分位 {q:.3f}")
        ax.set_title(f"{label}\n实际 size = {dist[key]['size']:.4f}", fontsize=10)
        ax.set_xlabel("统计量取值")
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 0.85)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("密度")
    fig.suptitle("图1：同一个 u 的三种函数——Wald 最激进、LM 最保守，位移方向确定", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig1_h0_distributions.png"))

    # ---- 实验 2：三个统计量作为 u 的解析函数 ----
    us = np.linspace(0, 0.06, 300)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(us, n0 * us / (1 + us), "-", color="#2980b9", lw=2.2, label="LM = n·u/(1+u)")
    ax.plot(us, n0 * np.log(1 + us), "-", color="#e67e22", lw=2.2, label="LR = n·log(1+u)")
    ax.plot(us, n0 * us, "-", color="#c0392b", lw=2.2, label="Wald = n·u")
    ax.fill_between(us, n0 * us / (1 + us), n0 * us, color="#95a5a6", alpha=0.18,
                    label="LM 与 Wald 的分歧带")
    ax.axhline(CRIT, color="k", ls="--", lw=1.4, label=f"χ²(1) 5% 临界 {CRIT:.3f}")
    crossing = {}
    for name, f, c in [("Wald", lambda u: n0 * u, "#c0392b"),
                       ("LR", lambda u: n0 * math.log(1 + u), "#e67e22"),
                       ("LM", lambda u: n0 * u / (1 + u), "#2980b9")]:
        lo, hi = 0.0, 0.06
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid) < CRIT:
                lo = mid
            else:
                hi = mid
        crossing[name] = float((lo + hi) / 2)
        ax.plot([crossing[name]], [CRIT], "o", color=c, ms=8)
    ax.text(0.02, 0.96,
            "三者跨过 5% 临界线所需的 u：\n"
            f"Wald  {crossing['Wald']:.6f}\n"
            f"LR      {crossing['LR']:.6f}\n"
            f"LM      {crossing['LM']:.6f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#bdc3c7", alpha=0.92))
    results["crossing_u"] = crossing
    ax.set_xlabel("u = (φ̂−φ0)²·Σx²/RSS(φ̂)（与参数化无关的充分统计量之比）")
    ax.set_ylabel("统计量取值（n=252）")
    ax.set_title("图2：LM ≤ LR ≤ Wald 是恒等式，不是抽样误差——三者跨临界线的位置必然不同")
    ax.legend(fontsize=8)
    fig.tight_layout()
    results["figs"].append(_save(fig, "fig2_analytic_ordering.png"))

    # ---- 实验 3：size 随 n 收敛；正态 / t(5) / GARCH(1,1) ------
    ns = [30, 60, 120, 252, 504, 1000]
    rep_s = 6000
    size_scan = {}
    for innov in ["normal", "t5", "garch"]:
        per = {}
        for n in ns:
            E = innovations(rep_s, n + BURN, rng, innov=innov)
            ss0, ss1, ss2 = simulate(0.0, n, E)
            _, w, l, m, _, _ = three_stats(ss0, ss1, ss2, n)
            per[f"n{n}"] = {
                "wald": float((w > CRIT).mean()),
                "lr": float((l > CRIT).mean()),
                "lm": float((m > CRIT).mean()),
            }
        size_scan[innov] = per
    results["size_vs_n"] = size_scan
    results["size_gaps"] = {
        "gap_wald_minus_lm_n30": size_scan["normal"]["n30"]["wald"] - size_scan["normal"]["n30"]["lm"],
        "gap_wald_minus_lm_n252": size_scan["normal"]["n252"]["wald"] - size_scan["normal"]["n252"]["lm"],
        "gap_wald_minus_lm_n1000": size_scan["normal"]["n1000"]["wald"] - size_scan["normal"]["n1000"]["lm"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharey=False)
    for ax, innov, title in zip(
        axes, ["normal", "t5", "garch"],
        ["正态新息（似然设定正确）", "t(5) 厚尾（尺度仍正确）", "GARCH(1,1) 条件异方差"]):
        for key, label, color, mk in [("wald", "Wald", "#c0392b", "o"),
                                     ("lr", "LR", "#e67e22", "s"),
                                     ("lm", "LM / Score", "#2980b9", "^")]:
            ys = [size_scan[innov][f"n{n}"][key] for n in ns]
            ax.plot(range(len(ns)), ys, mk + "-", color=color, lw=1.9, label=label)
        ax.axhline(ALPHA, color="k", ls="--", lw=1.3, label="名义 0.05")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns)
        ax.set_xlabel("样本量 n")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("H0 实际拒绝率")
    fig.suptitle("图3：分歧只在小样本显著；厚尾让三家一起保守，异方差才把 size 顶上去", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig3_size_vs_n.png"))

    # ---- 实验 4：重参数化不变性 ----
    # 固定一条 phi_true=0.15 的样本（保留原始序列以便数值重跑 profile 似然）
    n1, rep1 = 252, 4000
    E1 = innovations(rep1, n1 + BURN, rng)
    ss0, ss1, ss2 = simulate(0.15, n1, E1)
    ph1, w1, l1, m1, ru1, rr1 = three_stats(ss0, ss1, ss2, n1)
    sig2u1 = ru1 / n1
    # 取一条 phi_hat 明显为正、Wald 恰好未过临界线的样本
    cand = np.where((ph1 > 0.08) & (w1 < CRIT))[0]
    idx = int(cand[np.argmin(np.abs(w1[cand] - 3.0))])
    total = E1.shape[1]
    xs = np.zeros(total + 1)
    for t in range(total):
        xs[t + 1] = 0.15 * xs[t] + E1[idx, t]
    x_demo = xs[BURN:]
    d0, d1, d2 = simulate(0.15, n1, E1[idx:idx + 1])
    p_demo, W_demo, LR_demo, LM_demo, ru_demo, rr_demo = three_stats(d0, d1, d2, n1)
    su_demo = ru_demo / n1

    a_grid = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    rep_rows = []
    flip_rates = []
    for a in a_grid:
        Wp = float(reparam_wald(p_demo, d0, su_demo, a)[0])
        # 数值重跑：以 psi 为参数极大化同一条 profile 似然
        def neg_ell(psi):
            ph = g_a(psi, a)
            rss = d2 - 2.0 * ph * d1 + ph * ph * d0
            return 0.5 * n1 * math.log(float((rss / n1)[0]))
        opt = minimize_scalar(neg_ell, bounds=(-0.4, 0.4), method="bounded",
                              options={"xatol": 1e-12})
        LRp = 2.0 * (neg_ell(0.0) - opt.fun)
        LM_obs, LM_fisher, info_obs = obs_info_lm(x_demo, a)
        # MC 翻转率：phi 坐标系不拒绝、psi 坐标系拒绝
        Wp_all = reparam_wald(ph1, ss0, sig2u1, a)
        flip = float(((w1 < CRIT) & (Wp_all > CRIT)).mean())
        flip_rates.append(flip)
        rep_rows.append({
            "a": a, "wald_psi": Wp, "lr_psi": float(LRp),
            "lm_fisher_psi": LM_fisher, "lm_obs_psi": LM_obs,
            "obs_info": info_obs,
            "wald_ratio_to_closed_form": float(Wp / (W_demo[0] * (1.0 + a * p_demo[0]) ** 2)),
            "psi_hat_numeric": float(opt.x), "psi_hat_closed": float(psi_of_phi(p_demo[0], a)),
            "flip_rate": flip,
        })
    results["reparam"] = {
        "n": n1, "reps_mc": rep1, "phi_true": 0.15,
        "demo": {
            "phi_hat": float(p_demo[0]), "wald_phi": float(W_demo[0]),
            "lr_phi": float(LR_demo[0]), "lm_phi": float(LM_demo[0]),
            "sample_index": idx,
        },
        "a_grid": a_grid, "rows": rep_rows,
        "lr_max_dev": max(abs(r["lr_psi"] - float(LR_demo[0])) for r in rep_rows),
        "lm_fisher_max_dev": max(abs(r["lm_fisher_psi"] - float(LM_demo[0])) for r in rep_rows),
        "lm_obs_max_dev": max(abs(r["lm_obs_psi"] - float(LM_demo[0])) for r in rep_rows),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    ax = axes[0]
    ax.plot(a_grid, [r["wald_psi"] for r in rep_rows], "o-", color="#c0392b", lw=2.1,
            label="Wald(ψ) 随坐标系变化")
    ax.plot(a_grid, [r["lr_psi"] for r in rep_rows], "s-", color="#e67e22", lw=1.8,
            label="LR(ψ) 数值重跑")
    ax.plot(a_grid, [r["lm_fisher_psi"] for r in rep_rows], "^-", color="#2980b9", lw=1.8,
            label="LM(ψ) Fisher 信息版")
    ax.plot(a_grid, [r["lm_obs_psi"] for r in rep_rows], "d--", color="#8e44ad", lw=1.5,
            label="LM(ψ) 观测信息版")
    ax.axhline(CRIT, color="k", ls="--", lw=1.3, label="χ²(1) 5% 临界")
    ax.set_xlabel("重参数化旋钮 a：φ = ψ/(1−aψ)")
    ax.set_ylabel("统计量取值")
    ax.set_title(f"同一份样本 φ̂={p_demo[0]:.4f}，Wald(φ)={W_demo[0]:.3f}", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.plot(a_grid, flip_rates, "o-", color="#16a085", lw=2.2)
    ax.set_xlabel("重参数化旋钮 a")
    ax.set_ylabel("结论翻转的比例")
    ax.set_title("φ_true=0.15 的 4000 条样本里，\n「φ 系不拒绝、ψ 系拒绝」的占比", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(flip_rates) * 1.25 + 0.01)
    fig.suptitle("图4：换坐标系只改 Wald——同一数据、同一 H0、结论可以翻转", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig4_reparameterization.png"))

    # ---- 实验 5：共同随机数下的功效曲线 ----
    phis = np.round(np.linspace(0.0, 0.30, 13), 4)
    rep_p = 6000
    Ep = innovations(rep_p, n0 + BURN, rng)
    power = {}
    for phi in phis:
        ps0, ps1, ps2 = simulate(float(phi), n0, Ep)
        _, wp, lp, mp, _, _ = three_stats(ps0, ps1, ps2, n0)
        power[f"phi{phi:.4f}"] = {
            "wald": float((wp > CRIT).mean()),
            "lr": float((lp > CRIT).mean()),
            "lm": float((mp > CRIT).mean()),
            "mean_lm": float(mp.mean()),
        }
    gaps = [power[f"phi{p:.4f}"]["wald"] - power[f"phi{p:.4f}"]["lm"] for p in phis]
    lm_last = power[f"phi{phis[-1]:.4f}"]["mean_lm"]
    results["power_curve"] = {
        "phis": [float(p) for p in phis], **power,
        "max_gap_wald_minus_lm": float(max(gaps)),
        "lm_mean_at_max_phi": lm_last,
        "lm_theoretical_upper_bound": n0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    for key, label, color, mk in [("wald", "Wald", "#c0392b", "o"),
                                  ("lr", "LR", "#e67e22", "s"),
                                  ("lm", "LM / Score", "#2980b9", "^")]:
        ys = [power[f"phi{p:.4f}"][key] for p in phis]
        ax.plot(phis, ys, mk + "-", color=color, lw=2.0, label=label)
    ax.axhline(ALPHA, color="k", ls="--", lw=1.2)
    ax.set_xlabel("真实 φ（H0: φ=0）")
    ax.set_ylabel("拒绝率 / 功效")
    ax.set_title(f"n={n0}，6000 条共同随机数路径", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(phis, gaps, "o-", color="#8e44ad", lw=2.1, label="Wald 功效 − LM 功效")
    ax.axhline(0.0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("真实 φ（H0: φ=0）")
    ax.set_ylabel("功效差")
    ax.set_title(f"三家功效最大差仅 {max(gaps):.4f}；\n"
                 f"LM 均值 {lm_last:.2f} 远未触及理论上界 n={n0}", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("图5：正确设定的线性模型里，三家的功效几乎重合", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig5_power_curve.png"))

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
```


## 七、结果文件

五张图与 `results.json` 都在 `/images/wald-lr-score-test-divergence/`。结果文件包含原假设分布、样本量扫描、重参数化逐点数值、翻转率与功效曲线的全部条目，可以直接逐项核对正文表格。
