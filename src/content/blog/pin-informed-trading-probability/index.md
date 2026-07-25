---
title: "PIN 知情交易概率：从订单流不平衡里估计谁在偷跑"
publishDate: '2026-07-26'
description: "PIN 知情交易概率：从订单流不平衡里估计谁在偷跑 - EKOP 混合泊松模型 MLE 复现：300 只股票估计 PIN 与真实值相关 0.99，五分组年化 0.0%→5.7%，多空 t=4.89；但 PIN 与交易强度相关 -0.39，高 PIN 溢价可能只是非流动性换了个马甲 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

![EKOP 混合结构散点](/images/pin-informed-trading-probability/ekop-scatter.png)

市场里总有人比你先知道。财报泄露、并购内幕、大股东减持计划——这些信息落到盘口上，不会举着牌子自我介绍，但会留下统计痕迹：**知情交易者只朝一个方向下单**。好消息日买单异常多，坏消息日卖单异常多，无消息日买卖大致平衡。1996 年 Easley、Kiefer、O'Hara、Paperman（EKOP）把这个直觉写成了一个混合泊松模型，产出了微观结构领域被引用最多的指标之一：**PIN（Probability of INformed trading），任意一笔到达的订单来自知情交易者的概率**。

它的野心不小：不需要知道谁在交易、不需要监管数据，只用每天的买单笔数和卖单笔数两列数，就能反推出"这只股票的订单流里有多大比例是带信息的"。这篇文章把 EKOP 模型从数据生成到最大似然估计完整复现一遍，然后讨论它在 2009 年之后声誉受损的原因——高 PIN 溢价到底是信息风险定价，还是非流动性换了个马甲。

## EKOP 模型：一棵三分支的树

模型假设每个交易日开盘前，大自然掷两次骰子：

1. 以概率 $\alpha$ 发生信息事件（有私有信息产生），以 $1-\alpha$ 无事发生；
2. 若有事件，以概率 $\delta$ 是坏消息，$1-\delta$ 是好消息。

订单到达服从泊松过程：不知情的买方和卖方分别以强度 $\varepsilon_b$、$\varepsilon_s$ 全天候下单（他们因流动性需求交易，不管有没有消息）；知情交易者只在事件日出现，以强度 $\mu$ **单向**下单——好消息日全部买入，坏消息日全部卖出。于是一天的（买单数 $B$，卖单数 $S$）来自三个泊松分布之一：

- 无消息日：$B\sim Poisson(\varepsilon_b)$，$S\sim Poisson(\varepsilon_s)$
- 好消息日：$B\sim Poisson(\varepsilon_b+\mu)$，$S\sim Poisson(\varepsilon_s)$
- 坏消息日：$B\sim Poisson(\varepsilon_b)$，$S\sim Poisson(\varepsilon_s+\mu)$

PIN 就是知情订单流占总订单流的期望比例：

$$PIN = \frac{\alpha\mu}{\alpha\mu + \varepsilon_b + \varepsilon_s}$$

文首散点图是 250 个模拟交易日的 $(B,S)$ 分布：三团点清晰可辨——灰色的无消息日贴着对角线，绿色好消息日向右偏（买单多），红色坏消息日向上偏（卖单多）。**估计 PIN 的本质，就是在没有颜色标签的情况下把这三团点分开**。

## 最大似然估计

单日似然是三个泊松分支的加权和，全样本对数似然对五个参数 $(\alpha,\delta,\mu,\varepsilon_b,\varepsilon_s)$ 求最大：

```python
import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
from scipy.optimize import minimize

def neg_loglik(params, B, S):
    a, d, mu, eb, es = params
    if not (0 < a < 1 and 0 < d < 1 and min(mu, eb, es) > 0):
        return 1e12
    l0 = np.log(1-a)     + poisson.logpmf(B, eb)      + poisson.logpmf(S, es)
    l1 = np.log(a*d)     + poisson.logpmf(B, eb)      + poisson.logpmf(S, es+mu)
    l2 = np.log(a*(1-d)) + poisson.logpmf(B, eb+mu)   + poisson.logpmf(S, es)
    return -np.sum(logsumexp(np.vstack([l0, l1, l2]), axis=0))

def estimate_pin(B, S):
    best = None
    for a0 in (0.2, 0.4):                      # 多起点，避开局部极值
        for mu0 in (np.mean(B+S)*0.3, np.mean(B+S)*0.6):
            x0 = [a0, 0.5, mu0, B.mean()*0.8, S.mean()*0.8]
            res = minimize(neg_loglik, x0, args=(B, S), method="Nelder-Mead",
                           options={"maxiter": 4000})
            if best is None or res.fun < best.fun:
                best = res
    a, d, mu, eb, es = best.x
    return a*mu / (a*mu + eb + es)
```

两个工程细节决定成败。第一，**必须用 `logpmf` + `logsumexp`**：泊松强度动辄几百，直接算 pmf 再相乘会下溢成零，这是 EKOP 估计在高换手股票上大面积失败的技术根源（Easley-Hvidkjaer-O'Hara 2010 和 Lin-Ke 2011 各给过一套因子化重写，本质都是在对数域做稳定化）。第二，**多起点优化**：似然面有平坦区和局部极值，单起点 Nelder-Mead 经常停在 $\alpha$ 贴边界的退化解上。

单只股票的验证：真实参数 $\alpha=0.35,\mu=180,\varepsilon_b=350,\varepsilon_s=330$（真实 PIN=0.085），250 天数据的 MLE 估出 0.082，五个参数全部落在真值附近。

![PIN 估计 vs 真实](/images/pin-informed-trading-probability/pin-estimation.png)

横截面上的表现更能说明问题：300 只股票，参数各自随机（$\alpha\in[0.1,0.6]$，$\mu\in[60,260]$，不知情强度 $\varepsilon\in[150,600]$），每只用 250 天估计。**估计 PIN 与真实 PIN 的横截面相关 0.99**——在模型设定正确、买卖方向无误分类的理想世界里，PIN 的可估性没有问题。记住这个前提，后面要收回来。

## PIN 溢价的横截面回测

Easley-Hvidkjaer-O'Hara (2002) 的核心实证主张：高 PIN 股票有更高的期望收益，因为持有它们的不知情投资者承担了"总是和知情者做对手方"的逆向选择风险，要求补偿。我们在合成市场里植入这个溢价（月度收益对真实 PIN 的暴露系数为正），然后按**估计 PIN** 五分组、月度再平衡跑 15 年：

![PIN 五分组年化收益](/images/pin-informed-trading-probability/quintile-returns.png)

五分组年化从 G1 的 0.0% 升到 G5 的 5.7%，多空组合月均 0.46%（t=4.89），年化 5.7%，Sharpe 1.26。

![多空净值](/images/pin-informed-trading-probability/long-short-nav.png)

先声明合成数据的两处美化：Sharpe 1.26 受益于特异波动完全独立的设定，真实市场高 PIN 股票的共同暴露会把它压低一半以上；分组收益的单调性也比真实数据干净——真实研究里 PIN 溢价集中在小盘股，中间组经常搅成一团。这张图证明的是流程正确性，不是可实现收益。

## 四个真正的问题

**一，PIN 溢价可能是非流动性的马甲。** 这是 Duarte-Young (2009) 的著名批评，也是 PIN 声誉的转折点。我们的合成横截面里有个不显眼的数字：**估计 PIN 与总交易强度的相关是 -0.39**——这不是估计误差，真实 PIN 与强度的相关同样是 -0.4，因为 PIN 的分母就是总订单流。换句话说，高 PIN 股票几乎必然是低换手、低流动性的股票。Duarte-Young 把 PIN 拆成"信息不对称成分"和"与流动性相关的成分"后发现，**被定价的主要是后者**。你以为在赚信息风险的钱，其实大概率又绕回了 Amihud 那条非流动性溢价的路。做 PIN 研究，与 size、换手、ILLIQ 的双重排序不是加分项，是及格线。

**二，买卖方向分类是整个链条最脆弱的一环。** 模型的输入是"买单数"和"卖单数"，但逐笔数据不带方向标签，实证中全靠 Lee-Ready 算法（成交价高于中间价算买、低于算卖）近似。高频时代报价更新快于成交回报，Lee-Ready 的误分类率可达 15-20%，而误分类直接稀释 $B$、$S$ 的不平衡，**系统性低估 PIN**。用 tick rule 的更糟。这层噪声在模型里完全没有建模。

**三，日频泊松假设与真实订单流相去甚远。** 真实订单流有强烈的日内 U 型、聚集性（一笔大单拆成上百笔子单）和跨日相关，都不是常数强度泊松。后续的 DY 模型、GPIN 模型加了对称订单流冲击项和动态强度来补救，代价是参数从 5 个膨胀到 8 个以上，似然面更病态。**模型误设时 MLE 照样收敛并给出一个体面的数字**——这比不收敛更危险。

**四，A 股的适用性要打折扣。** A 股逐笔成交自带买卖方向标记（内外盘），跳过了 Lee-Ready 这层噪声，这是好消息。坏消息是涨跌停制度：涨停日卖单几乎消失、跌停日买单几乎消失，产生的极端 $(B,S)$ 不平衡与私有信息无关，却会把 $\alpha$ 和 $\mu$ 一起推高。触板日必须剔除或单独建模，否则高 PIN 组会富集连板妖股——和 ILLIQ 因子在 A 股遇到的是同一类问题。

## 结语

PIN 是那种"想法比数字更长寿"的指标。作为一个具体的可交易因子，它在 Duarte-Young 之后已经很难独立于流动性讲出增量故事；但作为一个建模范式——**用混合分布从粗糙的公开数据里反推不可观测的交易者构成**——它是后来 VPIN、GPIN、OWR 等一整条研究线的起点。用它的正确姿势：不当 alpha 用，当风险维度用；估计时在对数域做数值稳定化、多起点优化、剔除极端流动性事件日；解释结果前，先和非流动性因子做完正交化再开口。

*本文使用合成数据演示方法论，所有收益数字不构成任何投资建议。*
