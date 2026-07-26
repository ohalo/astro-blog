---
title: "Glosten-Milgrom 序贯交易模型：用贝叶斯更新把知情交易写进买卖价差"
description: "为什么做市商要挂买卖价差？经典答案是"补偿库存风险"——但 Glosten 和 Milgrom 在 1985 年给出了更深刻的解释：价差的核心是补偿逆向选择。当你面对一个可能比你知道得多的对手方时，每次成交都在传递信息。模型用贝叶斯更新把这个问题形式化：做市商根据订单流不断修正对真实价值的信念，报出的 bid/ask 逐步收敛。本文从模型设定推到 Python 实现，模拟展示报价如何随交易序列收敛到真实价值、价差如何随知情交易者占比放大、以及逆向选择成分在价差中的主导地位。最后诚实拆解：方向标注问题、参数非平稳、以及模型对真实市场的简化假设（中阶）。"
publishDate: '2026-07-26'
tags:
  - 量化交易
  - 市场微观结构
  - Glosten-Milgrom
  - 知情交易
  - 买卖价差
  - 逆向选择
  - Python
language: Chinese
difficulty: intermediate
---

## 一个让做市商夜不能寐的问题

想象你是一个做市商，同时挂着买单和卖单。每笔成交都在告诉你一些信息：如果连续来了十笔买单，你是不是该把价格往上挪一挪？

答案是"应该"——但原因不是库存压力，而是**信息**。

连续买单可能意味着有人在知道一些你不知道的事情。那个不断买入的人，可能掌握了你还没有的消息。每一次和他成交，你都在"被割"——卖出的价格事后看总是太低。

这就是逆向选择的核心：**和你成交的对手方可能比你知道得多**。做市商必须在价差里预埋一块补偿，用来覆盖这个"被知情交易者收割"的期望损失。

Glosten 和 Milgrom 在 1985 年把这个直觉变成了一套可以用数学分析的形式模型。它至今仍是市场微观结构的基石之一。

## 模型设定：两个世界，两种交易者

### 两状态世界

假设资产真实价值 $V$ 只能取两个值：$V_H$（高价值状态）或 $V_L$（低价值状态）。做市商不知道哪个是真的，只有先验概率 $P(V = V_H) = P(V = V_L) = 0.5$。

有一小群**知情交易者**知道真实价值是哪个。他们根据这个私有信息选择交易方向：

- 如果 $V = V_H$：买入（因为当前价格低估了）
- 如果 $V = V_L$：卖出（因为当前价格高估了）

另一群是**噪声交易者**（或流动性交易者）：他们因为套保、调仓、赎回等非信息原因交易，方向随机，买入卖出概率各 50%。

### 交易者到达

每时刻来一个交易者，以概率 $\alpha$ 是知情的，以概率 $1 - \alpha$ 是噪声的。参数 $\alpha$ 是模型的关键：它度量了市场上知情交易的密度。

$\alpha$ 越高，订单流中包含的信息越多，做市商面临的信息不对称越严重。

### 贝叶斯更新

做市商观测到订单流（买单或卖单），根据贝叶斯公式更新对 $V_H$ 的后验概率。

以买单到达为例：

$$
P(V_H \mid \text{买}) = \frac{P(\text{买} \mid V_H) \cdot P(V_H)}{P(\text{买})}
$$

其中：

- $P(\text{买} \mid V_H) = \alpha + \frac{1-\alpha}{2} = \frac{1+\alpha}{2}$（知情者必买 + 噪声者一半概率买）
- $P(\text{买} \mid V_L) = \frac{1-\alpha}{2}$（知情者不买，噪声者一半概率买）
- $P(\text{买}) = P(\text{买} \mid V_H) \cdot P(V_H) + P(\text{买} \mid V_L) \cdot P(V_L)$

代入后得到后验概率，进而算出条件期望：

$$
E[V \mid \text{买}] = P(V_H \mid \text{买}) \cdot V_H + P(V_L \mid \text{买}) \cdot V_L
$$

同理可得 $E[V \mid \text{卖}]$。

### 报价规则

Glosten-Milgrom 的关键洞见：**bid 和 ask 是条件期望**，而不是中间价加减一个固定偏移。

- **买价（bid）**：卖单到达时的条件期望，即做市商愿意为接一个卖单支付的最高价
  $$\text{bid} = E[V \mid \text{卖}]$$
  
- **卖价（ask）**：买单到达时的条件期望，即做市商愿意卖出的最低价
  $$\text{ask} = E[V \mid \text{买}]$$

关键：ask > bid，因为买单信号推高期望、卖单信号压低期望。价差 = ask - bid 正是信息不对称的度量。

## Python 实现：模拟贝叶斯更新过程

我们用 Python 模拟完整的序贯交易过程，展示报价如何随订单流演化。

```python
import numpy as np

def glosten_milgrom_simulate(n_trades, alpha, V_low, V_high, prior=0.5, seed=None):
    """
    Glosten-Milgrom 模型模拟
    
    参数:
    - n_trades: 交易次数
    - alpha: 知情交易者占比
    - V_low: 低价值状态
    - V_high: 高价值状态
    - prior: V = V_high 的先验概率
    
    返回:
    - bid_arr, ask_arr, mid_arr: 报价序列
    - prob_arr: 后验概率序列
    - true_V: 真实价值
    """
    rng = np.random.default_rng(seed)
    
    # 真实价值（假设等概率高/低）
    true_V = V_high if rng.random() < 0.5 else V_low
    
    prob_V_high = prior  # 初始信念
    bid_list, ask_list, mid_list, prob_list = [], [], [], []
    
    for _ in range(n_trades):
        # 计算条件期望 E[V|卖] -> bid
        P_sell_given_H = 0.5 * (1 - alpha)  # V=V_high 时卖单概率
        P_sell_given_L = 0.5 * (1 + alpha)  # V=V_low 时卖单概率
        P_sell = prob_V_high * P_sell_given_H + (1 - prob_V_high) * P_sell_given_L
        prob_H_given_sell = prob_V_high * P_sell_given_H / P_sell if P_sell > 0 else prob_V_high
        bid = prob_H_given_sell * V_high + (1 - prob_H_given_sell) * V_low
        
        # 计算条件期望 E[V|买] -> ask
        P_buy_given_H = 0.5 * (1 + alpha)
        P_buy_given_L = 0.5 * (1 - alpha)
        P_buy = prob_V_high * P_buy_given_H + (1 - prob_V_high) * P_buy_given_L
        prob_H_given_buy = prob_V_high * P_buy_given_H / P_buy if P_buy > 0 else prob_V_high
        ask = prob_H_given_buy * V_high + (1 - prob_H_given_buy) * V_low
        
        bid_list.append(bid)
        ask_list.append(ask)
        mid_list.append((bid + ask) / 2)
        prob_list.append(prob_V_high)
        
        # 模拟交易者到达
        is_informed = rng.random() < alpha
        if is_informed:
            trade = 'buy' if true_V == V_high else 'sell'
        else:
            trade = 'buy' if rng.random() < 0.5 else 'sell'
        
        # 根据成交更新信念
        prob_V_high = prob_H_given_buy if trade == 'buy' else prob_H_given_sell
    
    return (
        np.array(bid_list), 
        np.array(ask_list), 
        np.array(mid_list), 
        np.array(prob_list), 
        true_V
    )

# 运行模拟
bid, ask, mid, prob, true_V = glosten_milgrom_simulate(
    n_trades=60, alpha=0.3, V_low=9.5, V_high=10.5, seed=42
)
print(f"真实价值: {true_V}")
print(f"初始中间价: {mid[0]:.4f}, 最终中间价: {mid[-1]:.4f}")
print(f"最终后验概率 P(V=V_high): {prob[-1]:.4f}")
```

核心逻辑：每次成交后，做市商用贝叶斯公式更新 $P(V = V_H)$，然后重新计算 bid/ask。

## 结果解读

### 报价的贝叶斯收敛

![做市商报价随交易序列的贝叶斯更新演化](/images/glosten-milgrom-model/bayesian-updating-evolution.png)

图中展示了 60 笔交易中 bid/ask/中间价的演化。核心观察：

1. **中间价逐步收敛到真实价值**。做市商一开始不知道 $V$ 是 9.5 还是 10.5，先验给的是 50/50。随着订单流积累，连续的买单（真实价值高时）或卖单（真实价值低时）逐步揭示信息。

2. **价差存在且非零**。即使在先验均匀的情况下，ask 依然高于 bid——这是信息不确定性的直接度量。

3. **收敛速度取决于 α**。知情交易者越多，订单流的信号越强，收敛越快。极端情况 α → 0（全噪声），订单流完全不传递信息，做市商的信念永远停在先验。

### 价差与知情交易者占比

![买卖价差随知情交易者占比的变化](/images/glosten-milgrom-model/spread-vs-alpha.png)

这张图回答了一个直观问题：知情交易者越多，价差越大吗？

答案是肯定的，而且关系近似线性。原因是：

- $\alpha$ 越高，买单更可能来自知情者（意味着 $V = V_H$），卖单同理。
- 条件期望 $E[V \mid \text{买}]$ 和 $E[V \mid \text{卖}]$ 之间的差距被拉大。
- 做市商必须用更宽的价差来覆盖更大的逆向选择风险。

实务含义：**信息不对称高的市场（如小盘股、财报前夕、并购传闻期），流动性天然更贵**。这不是做市商贪婪，而是博弈均衡的必然结果。

### 价差的成分分解

![买卖价差的成分分解](/images/glosten-milgrom-model/spread-decomposition.png)

这两张图进一步拆解：总价差中多少归因于逆向选择？

左图展示不同 $\alpha$ 下逆向选择成分的占比。当 $\alpha = 5\%$（知情交易者稀少），信息成分占比低；当 $\alpha = 60\%$，信息成分主导。

右图把总价差拆成两块柱子：逆向选择成分和非信息成分。可以看到：

- 低 $\alpha$ 市场：总价差窄，主要成分是订单处理成本（模型未显式建模但可类推）。
- 高 $\alpha$ 市场：总价差宽，且主要由逆向选择驱动。

## 诚实的边界

**第一，方向标注是隐含的。** 模型假设我们观测到"买单"或"卖单"。实盘数据要从报价-成交匹配推断方向（Lee-Ready 算法），错误率 10-15%。方向标错 = 贝叶斯更新公式里喂了噪声数据，估出来的信息成分会系统性偏低。

**第二，两状态假设是强简化。** 真实资产价值是连续随机变量，甚至可能服从跳跃扩散过程。模型用 $\{V_L, V_H\}$ 把问题变成二分类贝叶斯推断，方便分析但牺牲了精度。扩展版本（如连续信号模型）可以处理更丰富的分布，但估计难度显著上升。

**第三，α 非平稳且难以直接观测。** 知情交易者占比随时间、随事件剧烈波动。财报公告日前夕 $\alpha$ 跳升，公告后回落。用一个"全局 α"来描述市场，等于把高信息密度时段和低信息密度时段平均掉——可能两头都失真。

**第四，模型假设做市商是风险中性和竞争性的。** 真实做市商有库存约束、资本约束、风险厌恶，这些都会体现在报价里。Glosten-Milgrom 的价差纯粹来自信息不对称，没有考虑库存周期和风险补偿——这部分需要 Ho-Stoll 模型来补充。

**第五，序贯假设在高速市场中被打破。** 模型假设交易一个接一个到达，每次更新信念。真实市场同一毫秒可能有几十笔成交，做市商在更新报价前已经被多笔订单命中。批量成交下的信念更新需要更复杂的信号提取框架。

## 与 MRR 模型的关系

如果你读过本系列上一篇 [Madhavan-Richardson-Roomans 模型](/blog/madhavan-glosten-model)，会发现两者共享一个核心洞见：**逆向选择是价差的本质成分**。

区别在于视角：

- **Glosten-Milgrom（1985）**：静态信息结构 + 贝叶斯更新。适合分析"信息如何被价格吸收"的过程，回答"一次交易揭示多少信息"。
- **MRR（1997）**：成交数据的结构模型 + 矩条件估计。适合用历史数据反推参数，回答"这支股票的价差里有多少是信息成分"。

实务上两者互补：用 MRR 估计 $\theta$（信息成分），用 Glosten-Milgrom 理解 $\theta$ 的微观来源。

## 收尾

Glosten-Milgrom 模型用最简化的假设（两状态、两类交易者、序贯到达）提炼出市场微观结构的核心洞察：**价差的本质是对信息不对称的补偿**。做市商不是在和噪声交易者博弈，而是在和一个可能比他知道得多的对手方博弈。每一笔成交都在传递信息，报价的每一次调整都是在"学习"。

Python 模拟展示了这个过程：中间价随订单流收敛到真实价值，价差随知情交易者占比扩大，逆向选择成分在总价差中的主导地位。三个观察都有清晰的实务映射——从横截面比较哪些股票"有毒"，到判断什么时候该等一等再下单。

模型的诚实边界在于它太干净了：两状态世界、风险中性做市商、序贯交易。真实市场更脏、更乱、更难建模。但正是因为简化，我们才能看清信息流动的本质逻辑。在更复杂的扩展模型之前，先用 Glosten-Milgrom 把直觉建起来。

## 参考文献

- Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71-100.
- O'Hara, M. (1995). *Market Microstructure Theory*. Blackwell.
- Hasbrouck, J. (2007). *Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading*. Oxford University Press.
