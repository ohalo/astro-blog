---
title: "Easley-O'Hara 信息模型：信息事件如何驱动订单流与价差"
publishDate: '2026-07-29'
description: "Easley-O'Hara 序贯交易模型详解：信息事件树、做市商贝叶斯定价、PIN 与价差的关系及 Python 模拟 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 先说结论

Easley 和 O'Hara 在 1992 年提出的序贯交易模型回答了市场微观结构最核心的问题之一：**为什么会有买卖价差，以及价差的宽度由什么决定**。答案是：价差本质上是做市商对"知情交易风险"收取的保险费。市场里知情交易者越多、信息事件越频繁，做市商被"割"的概率越大，挂出的价差就越宽。

这个模型是 PIN（知情交易概率）指标的理论地基，也是理解 VPIN、订单流毒性等现代流动性风险指标的前提。这篇文章把模型结构、做市商的贝叶斯定价逻辑讲透，并用 Python 完整复现模型的三个核心结论。

## 模型设定：一棵信息事件树

模型把每个交易日抽象成一次自然的抽签，涉及五个参数：

- $\alpha$：当天发生**信息事件**的概率；
- $\delta$：信息事件是**坏消息**的概率；
- $\mu$：知情交易者的订单到达强度（泊松）；
- $\varepsilon_b, \varepsilon_s$：不知情交易者的买、卖到达强度。

事件树长这样：

![Easley-O'Hara 信息事件树](/images/easley-ohara-microstructure/eo-event-tree.png)

三种状态下的订单流构成完全不同：

| 状态 | 概率 | 买单强度 | 卖单强度 |
|------|------|---------|---------|
| 无事件 | $1-\alpha$ | $\varepsilon_b$ | $\varepsilon_s$ |
| 好消息 | $\alpha(1-\delta)$ | $\varepsilon_b + \mu$ | $\varepsilon_s$ |
| 坏消息 | $\alpha\delta$ | $\varepsilon_b$ | $\varepsilon_s + \mu$ |

关键假设是**知情交易者只在有利可图的方向交易**：知道好消息就只买，知道坏消息就只卖。不知情交易者则因为流动性需求随机买卖，方向与信息无关。

用 Python 模拟 2000 个交易日，把每天的（买单数 B，卖单数 S）画在平面上，三种状态会自然分成三簇：

```python
import numpy as np
rng = np.random.default_rng(7)

alpha, delta, mu = 0.3, 0.4, 60      # 事件概率/坏消息概率/知情强度
eb, es = 50, 50                       # 不知情买卖强度
days = 2000

is_event = rng.random(days) < alpha
is_good  = rng.random(days) < (1 - delta)

B = rng.poisson(eb, days).astype(float)
S = rng.poisson(es, days).astype(float)
B[is_event & is_good]  += rng.poisson(mu, (is_event & is_good).sum())
S[is_event & ~is_good] += rng.poisson(mu, (is_event & ~is_good).sum())
```

![订单流三簇分布](/images/easley-ohara-microstructure/eo-orderflow-clusters.png)

无事件日聚在中间（B≈S≈50），好消息日整体右移（买单激增），坏消息日整体上移（卖单激增）。**这个分簇结构就是 PIN 估计的统计基础**——极大似然估计做的事情，本质上就是从历史 (B, S) 散点里反推出这三簇的位置和权重。

## 做市商的定价：价差 = 逆向选择保险费

做市商知道模型结构和参数，但不知道今天抽到了哪个状态。设资产真值在好消息时为 $\overline{V}$，坏消息时为 $\underline{V}$，无事件时为 $V^*$。

做市商的报价规则是**条件期望定价**：

- 卖价（ask）= 在"下一单是买单"条件下的期望真值；
- 买价（bid）= 在"下一单是卖单"条件下的期望真值。

为什么买单会推高期望值？贝叶斯逻辑：买单更可能来自知道好消息的知情者，所以观察到买单后，"今天是好消息日"的后验概率上升。做市商必须把价格挂在条件期望上，否则会被知情者系统性地套利。

开盘时刻（尚无成交）的价差有一个非常干净的表达式。在对称情形（$\delta = 0.5$，$\varepsilon_b = \varepsilon_s = \varepsilon$）下：

$$
\text{Spread} = \frac{\alpha\mu}{\alpha\mu + 2\varepsilon} \times (\overline{V} - \underline{V}) = \text{PIN} \times (\overline{V} - \underline{V})
$$

第一项正是 PIN——任意一笔成交来自知情者的概率；第二项是信息的价值幅度。**价差与知情交易风险严格成正比**：

![PIN 与价差的关系](/images/easley-ohara-microstructure/eo-pin-spread.png)

这个公式的深意在于：即使做市商没有任何库存成本、没有任何手续费，只要存在信息不对称，价差就必然为正。这是 Glosten-Milgrom 逆向选择思想在动态框架下的延伸。

## 日内学习：报价如何收敛到真值

模型最漂亮的部分是日内动态。做市商对三种状态维护一个后验概率向量，每观察到一笔成交就做一次贝叶斯更新：

```python
VH, VL, V0 = 101.0, 99.0, 100.0

# 三种状态的 (买单强度, 卖单强度)
lam = {
    "none": (eb,      es),
    "good": (eb + mu, es),
    "bad":  (eb,      es + mu),
}
states = ["none", "good", "bad"]
post = np.array([1 - alpha, alpha * (1 - delta), alpha * delta])
values = np.array([V0, VH, VL])

def update(post, is_buy):
    """观察一笔成交后的贝叶斯更新"""
    lik = np.empty(3)
    for i, s in enumerate(states):
        lb, ls = lam[s]
        total = lb + ls
        lik[i] = (lb if is_buy else ls) / total
    post = post * lik
    return post / post.sum()

# 模拟一个"坏消息日"：按 bad 状态的强度生成 200 笔成交
mid_path = [post @ values]
p_buy_bad = eb / (eb + es + mu)
for _ in range(200):
    is_buy = rng.random() < p_buy_bad
    post = update(post, is_buy)
    mid_path.append(post @ values)
```

在坏消息日，卖单源源不断到来，做市商对"坏消息"状态的后验概率不断上升，中间价从无条件期望 100 一路收敛到真值 99：

![做市商贝叶斯学习路径](/images/easley-ohara-microstructure/eo-bayesian-quotes.png)

三个值得注意的性质：

1. **收敛不是单调的**：不知情买单会短暂把价格拉回去，形成锯齿；但大数定律保证长期方向正确。
2. **收敛速度由 $\mu / \varepsilon$ 决定**：知情者相对越活跃，订单流的信息含量越高，价格发现越快。信息最终**通过交易本身**进入价格，这就是"价格发现"的微观机制。
3. **价差随成交递减**：随着后验越来越确定，逆向选择风险下降，买卖价差在日内逐步收窄——这与实证中"开盘价差最宽、随后衰减"的日内 U 型价差形态一致。

## 模型的推论与实证含义

**推论一：无交易也是信息。** 在扩展版模型（Easley-O'Hara 1992）中，做市商还会从"没有成交"中学习——长时间无单到达，提高了"今天无信息事件"的后验，价差随之收窄。这解释了为什么清淡但平静的市场价差反而不宽。

**推论二：订单不平衡预测短期收益。** 既然买卖不平衡反映知情交易方向，$B - S$ 就应该与后续价格变动正相关。这是订单流不平衡（OFI）类高频因子的理论出处。

**推论三：PIN 是可估计的风险因子。** 五个参数 $(\alpha, \delta, \mu, \varepsilon_b, \varepsilon_s)$ 可以用每日买卖单数序列做极大似然估计，得到 PIN。Easley 等人后续的实证发现高 PIN 股票有更高的预期收益——信息风险要求补偿，尽管这一结论在学术界仍有争议（可能与规模、流动性因子混杂）。

## 局限性

模型的简化不能忽视：知情者被假设为无策略的"强度型"交易者，不会拆单、不会择时隐藏自己，而现实中的知情交易者一定会用算法把自己伪装成噪声；参数在日内被假设为常数，无法刻画盘中突发信息；泊松到达也无法产生真实市场的成交聚集（clustering）。这些缺陷推动了后续 VPIN（用成交量时钟替代日历时钟）和动态 PIN 模型的发展。

## 总结

Easley-O'Hara 模型用一棵三状态事件树和泊松订单流，把"信息如何进入价格"这个抽象问题变成了可计算、可估计的结构：价差是做市商向全体交易者收取的逆向选择保险费，其宽度等于 PIN 乘以信息价值；价格发现是做市商对订单流做贝叶斯学习的过程。理解了这个骨架，PIN、VPIN、订单流毒性这些流动性风险指标就不再是黑盒公式，而是同一套逻辑在不同工程约束下的变体。
