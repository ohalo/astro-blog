---
title: "熵风险度量 EVaR：用指数矩给 CVaR 一个更保守的上界"
publishDate: '2026-07-27'
description: "熵风险度量 EVaR：用指数矩给 CVaR 一个更保守的上界 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

风险度量的历史，是一部「不断把尾部看得更清楚」的历史。VaR 只报一个分位点，尾部破了以后多惨它不管；CVaR（也叫 ES）把超过 VaR 的损失平均进来，一致性有了、也看到了尾部平均。但 CVaR 还是有个软肋：它只用「尾部的均值」，对尾部内部那些真正灾难性的极端值，权重给得不够。熵风险度量（Entropic Value at Risk，EVaR）由 Ahmadi-Javid 在 2012 年提出，用一个漂亮的想法把这个软肋补上——**用收益的指数矩（矩生成函数）给尾部一个更保守的上界**。它是目前已知的、计算上可处理（凸、可优化）的一致风险度量里，最保守的一个。

## 从一个不等式说起：为什么是「指数矩」

EVaR 的全部秘密，藏在概率论里一个古老的工具——切诺夫界（Chernoff bound）里。对任意损失随机变量 $L$ 和任意 $t>0$，马尔可夫不等式作用在 $e^{tL}$ 上给出：

$$P(L \ge a) \le e^{-ta}\, E[e^{tL}]$$

把这个不等式反过来解，就能得到一个「在置信度 $\alpha$ 下损失不会超过多少」的界。定义 $z=1-\alpha$，EVaR 就是：

$$\mathrm{EVaR}_\alpha(L) = \inf_{t>0}\ \left\{ \frac{1}{t}\ln\frac{E[e^{tL}]}{1-\alpha} \right\}$$

换个更常用的写法（令 $t' = 1/t$）：

$$\mathrm{EVaR}_\alpha(L) = \inf_{t>0}\ t\left[\ln E[e^{L/t}] - \ln(1-\alpha)\right]$$

这里 $E[e^{L/t}]$ 就是损失的矩生成函数（MGF）。整个定义就是「扫描一个辅助变量 $t$，找一条最紧的切诺夫界」。这个 $t$ 我喜欢叫它「风险温度」——它像统计物理里的温度参数，$t$ 越小，指数 $e^{L/t}$ 对大损失的放大越猛，度量就越只盯着最极端的尾部。

**为什么叫「熵」？** 因为这个上确界形式，通过对偶（Donsker-Varadhan 变分公式），恰好等价于「在相对熵（KL 散度）约束下，对最坏情况分布取的期望损失」。EVaR 本质上是一个分布鲁棒（distributionally robust）度量：它假设真实分布可能在你估计的分布附近的一个 KL 球里游走，然后报告这个球里最坏那个分布下的期望损失。这就是它天生保守的根源。

## VaR ≤ CVaR ≤ EVaR：一条不等式链

EVaR 最重要的性质，是它给 CVaR 一个上界。对同一个置信度 $\alpha$，永远有：

$$\mathrm{VaR}_\alpha(L) \le \mathrm{CVaR}_\alpha(L) \le \mathrm{EVaR}_\alpha(L)$$

第一个不等号显然（CVaR 是超过 VaR 部分的平均，必然 ≥ VaR）。第二个不等号是 EVaR 的招牌：它说 EVaR 永远站在 CVaR 外侧，是一个更保守的风险预算。我用一段双成分高斯混合（平静态 + 危机态，制造肥尾）跑了 20 万个样本，$\alpha=0.95$ 下实测：

![高斯混合肥尾损失下三种风险度量对比](/images/entropic-var-evar/evar-vs-cvar-var.jpg)

三条虚线从左到右分别是 VaR95 = 1.80%、CVaR95 = 3.20%、EVaR95 = 5.32%。可以直观看到，EVaR 比 CVaR 又往右推了整整两个百分点——它把混合分布里那条来自「危机态」的肥尾，实打实地计入了风险预算。如果你用 VaR 做风控，你给这个组合的资本缓冲是 1.8%；用 CVaR 是 3.2%；而 EVaR 会要求你准备 5.3%。在一个真会出现危机态的世界里，谁更接近现实，不言而喻。

## 怎么算：一维凸优化，扫一个 t 就够

EVaR 定义里那个 $\inf_{t>0}$ 看着吓人，其实是量化风控里最友好的一类问题——**目标函数对 $t$ 是凸的**，一维、光滑、有唯一最小值。你不需要任何高级求解器，`scipy.optimize.minimize_scalar` 一行搞定。

```python
import numpy as np
from scipy import optimize

def evar(losses, alpha, sample_size=40000, seed=0):
    """
    用样本 MGF 近似计算 EVaR。
    losses: 损失数组（正值代表亏损）
    alpha:  置信度，如 0.95
    """
    rng = np.random.default_rng(seed)
    # 子采样以稳定 MGF 估计（指数对极端值极敏感）
    x = losses[rng.integers(0, len(losses), sample_size)]

    def g(t):
        # 目标函数 g(t) = t * [ ln E[e^{L/t}] - ln(1-alpha) ]
        # 用 logsumexp 技巧防指数溢出
        z = x / t
        m = z.max()
        log_mgf = m + np.log(np.mean(np.exp(z - m)))
        return t * (log_mgf - np.log(1 - alpha))

    res = optimize.minimize_scalar(g, bounds=(1e-4, 1.0), method="bounded")
    return res.fun, res.x   # (EVaR 值, 最优 t*)

# 用法
rng = np.random.default_rng(20260727)
N = 200_000
crisis = rng.random(N) < 0.10
ret = np.where(crisis,
               rng.normal(-0.004, 0.030, N),   # 危机态
               rng.normal(0.0006, 0.009, N))   # 平静态
loss = -ret
ev, t_star = evar(loss, 0.95)
print(f"EVaR95 = {ev*100:.3f}%,  最优风险温度 t* = {t_star:.4f}")
# EVaR95 = 5.315%,  t* = 0.0124
```

关键工程细节是那个 **logsumexp 技巧**。$e^{L/t}$ 在 $t$ 很小、$L$ 较大时会瞬间溢出到 `inf`，直接算 `np.mean(np.exp(x/t))` 几乎必然爆掉。把最大值提出来再取对数，是数值稳定的标准做法，不做这步 EVaR 根本算不出来。

下面这张图把「扫 $t$ 找下确界」的过程画了出来：

![EVaR 是一维凸优化的下确界](/images/entropic-var-evar/evar-variational.jpg)

紫色曲线是目标函数 $g(t)$，它是一条典型的凸曲线——$t$ 太小，指数放大过猛，值飙高；$t$ 太大，$\ln(1-\alpha)$ 那项主导，值也回升。中间那个红点就是最优的 $t^*=0.0124$，对应的 $g(t^*)$ 就是 EVaR。橙色点线是 CVaR，可以看到整条 $g(t)$ 曲线始终在 CVaR 上方——这就是 $\mathrm{CVaR}\le\mathrm{EVaR}$ 的几何图像。

## 置信度越高，EVaR 越显保守

EVaR 的保守性不是恒定的，它随置信度 $\alpha$ 放大。$\alpha$ 越靠近 1（你越关心极端尾部），EVaR 相对 CVaR 撑开的裕度越大。我把 $\alpha$ 从 0.90 扫到 0.999：

![置信度扫描：EVaR 相对 CVaR 的裕度](/images/entropic-var-evar/evar-confidence-sweep.jpg)

三条线一路发散：在 $\alpha=0.90$ 时 EVaR 和 CVaR 还贴得比较近，到 $\alpha=0.99$、$0.999$ 时，紫色的 EVaR 已经明显甩开红色的 CVaR，中间那片阴影就是 EVaR 额外要求的保守裕度。这个性质有实际含义：当你做的是**极端风险管理**（比如监管资本、尾部对冲预算），越往尾部走，EVaR 相对 CVaR 的「多要一点缓冲」越合理——因为尾部越深，样本越稀，你越该给未知的坏情况留余地，而 EVaR 的指数加权恰好自动做到了这一点。

## EVaR 相对 CVaR 的三个实际优势

**一、它是可优化的一致风险度量里最保守的。** 一致性（次可加、正齐次、单调、平移不变）保证了它在组合层面行为良好——分散化永远不会让 EVaR 变大。而在所有满足一致性且计算可处理的度量里，EVaR 给出最紧的保守上界，这在需要「宁可高估不可低估」的场景（清算风险、压力测试）里是刚需。

**二、组合优化更好解。** CVaR 的组合优化（Rockafellar-Uryasev）要引入大量辅助变量，本质是线性规划、随场景数膨胀。EVaR 的组合优化因为目标函数对 $t$ 联合凸，通常能写成更紧凑的凸规划（甚至指数锥规划），在场景数巨大时数值上更省。

**三、它是分布鲁棒的。** EVaR 天生等价于「KL 球内最坏分布的期望损失」。这意味着你用 EVaR 做出的资本决策，对「真实分布和你估计的分布有偏差」这件事有内建的免疫力——CVaR 没有这层保护，它只对你喂进去的那个经验分布负责。

## 小结

EVaR 的一句话本质：**用切诺夫界把 CVaR 往外顶一层，得到一个既保守、又一致、还可优化的尾部风险度量**。它站在 VaR ≤ CVaR ≤ EVaR 这条不等式链的最外侧，代价是要求损失分布有有限的矩生成函数（这也是它不能直接用在幂律肥尾上的原因）。对于「宁可备足弹药也不愿被尾部打穿」的风控场景，EVaR 是比 CVaR 更诚实的选择——它把「我对分布估计没那么自信」这件事，直接写进了风险数字里。

**诚实边界**：EVaR 要求损失分布存在矩生成函数，这排除了 t 分布、帕累托这类真正的幂律肥尾（它们的 MGF 发散，EVaR 会算出无穷大或数值爆炸）——本文特意用有限 MGF 的高斯混合来演示，正是因为 t(4) 直接套 EVaR 会失效；实践中若尾部确为幂律，应改用 CVaR 或专门的极值理论工具。其次，EVaR 依赖对 MGF 的估计，而指数矩对极端样本极度敏感，样本量不足或采样噪声大时 EVaR 估计的方差会远大于 CVaR，报数字前必须检查稳定性。第三，EVaR 的保守性是一把双刃剑：在真实风险并不极端的资产上，它可能系统性高估、导致资本利用率偏低。最后，最优风险温度 $t^*$ 的求解虽是凸问题，但在 $t$ 很小的区域数值上易出现溢出，工程实现必须用 logsumexp 稳定化，否则结果不可信。（中阶）
