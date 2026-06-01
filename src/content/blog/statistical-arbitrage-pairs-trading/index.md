---
title: "统计套利实战：用协整分析做配对交易"
publishDate: '2026-06-01'
description: "统计套利实战：用协整分析做配对交易 - halo的技术博客"
tags:
  - 量化交易
language: Chinese
---

统计套利是量化交易中最迷人的分支之一。不同于趋势跟踪或多因子选股这些押注于方向的策略，统计套利追求的是相对价值的均值回归——找到两只走势高度同步的股票，当它们的价差偏离历史均值时做多弱势方、做空强势方，等待价差回归时获利。听起来完美，做起来如何？让我们深入配对交易的全链路。

## 统计套利的核心思想

统计套利建立在两个基本假设之上。第一，某些资产之间存在长期稳定的统计关系，例如工商银行和建设银行的股价走势高度一致。第二，这种关系的暂时偏离最终会回归均值。

为什么这种偏离存在？原因很多：短期资金流动、市场情绪波动、订单簿深度差异……但长期来看，基本面的力量会把价差拉回均衡水平。统计套利者正是充当了"市场纠错者"的角色——他们不预测方向，而是赌"偏离会回归"。

这里的关键词是"长期稳定的统计关系"。如果两只股票的关系不稳定，你所谓的"偏离"可能只是结构性的永久变化，那么"回归"就不会发生。这也是统计套利最大的风险——**协整关系破裂**。

![配对交易原理图](/images/statistical-arbitrage-pairs-trading/pairs-trading-concept.jpg)

## 协整：比相关性更重要的概念

很多初学者用相关系数来找配对，这是一个常见的误区。**相关性高不等于适合配对交易**。

设想两只股票 A 和 B。A 每年涨 10%，B 每年涨 5%，两者的价格走势可能相关系数高达 0.95，但它们的价差在不断扩大——做多 B 做空 A 的组合会稳定亏损。

协整回答了更本质的问题：两只股票的价差是否围绕一个固定水平波动？如果价差在偏离后会回归，那么这两只股票就是协整的。

数学上，对 A 和 B 的对数价格做最小二乘回归：

$$\log(P_A) = \alpha + \beta \cdot \log(P_B) + \varepsilon$$

如果残差 $\varepsilon$ 是平稳的（单位根检验的 p 值 < 0.05），那么 A 和 B 就是协整的，$\beta$ 就是对冲比率。

## 实操：用 Python 做配对筛选

首先从同一行业中获取候选股票池。行业分类可以用申万一级行业，也可以用 SWS、GICS 等标准分类。为什么限定在同一行业？因为同行业的公司面临相似的商业环境，协整关系的经济基础更强。

以下是配对筛选的核心逻辑：

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

def find_pairs(prices_df, lookback=252, pvalue_threshold=0.05):
    """
    在给定的价格矩阵中找出协整的配对
    prices_df: DataFrame, columns为股票代码, values为收盘价
    """
    stocks = prices_df.columns
    pairs = []
    
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            s1, s2 = stocks[i], stocks[j]
            pair_data = pd.concat([prices_df[s1], prices_df[s2]], axis=1).dropna()
            
            if len(pair_data) < lookback:
                continue
            
            # 协整检验
            score, pvalue, _ = coint(pair_data[s1], pair_data[s2])
            
            if pvalue < pvalue_threshold:
                # 计算对冲比率
                hedge_ratio = np.polyfit(pair_data[s2], pair_data[s1], 1)[0]
                
                # 计算价差的半衰期
                spread = pair_data[s1] - hedge_ratio * pair_data[s2]
                halflife = estimate_halflife(spread)
                
                pairs.append({
                    'stock1': s1, 'stock2': s2,
                    'pvalue': pvalue,
                    'hedge_ratio': hedge_ratio,
                    'halflife': halflife,
                    'spread_std': spread.std()
                })
    
    return pd.DataFrame(pairs).sort_values('pvalue')
```

![协整配对筛选流程](/images/statistical-arbitrage-pairs-trading/cointegration-flow.jpg)

## 信号生成与交易执行

找到协整配对后，我们需要定义入场和出场信号。

**标准方法**：计算价差的 Z-Score（减均值除以标准差），当 Z-Score 超过 ±2 时入场，回归到 ±0.5 以内时平仓：

- Z-Score > +2：价差偏高 → 做空强势股（stock1），做多弱势股（stock2）
- Z-Score < -2：价差偏低 → 做多强势股（stock1），做空弱势股（stock2）

**动态均值与标准差**：不要用全历史区间的均值和标准差。使用滚动窗口（例如过去 60 个交易日）计算，这样模型能适应市场结构的变化。但窗口太长会反应迟钝，太短会噪音太多。一个经验值是使用价差半衰期的 3-5 倍作为窗口长度。

**止损**：统计套利最怕的是协整关系破裂。必须设置硬止损——通常当价差继续扩大到 Z-Score 的 3 倍标准差时无条件平仓。另一个可选方案是定期重新做协整检验，当 p 值上升超过 0.20 时，认为关系已失效，终止配对。

## 实盘中的挑战

配对交易在回测中往往表现极好，但实盘有三个"隐形杀手"：

**冲击成本**：配对交易涉及双向下单，尤其是在涨跌停板限制的 A 股市场，可能出现"能做多但做空不到"或者"能做空但券不够"的情况。融券成本也是不可忽视的——A 股的融券利率通常在 8% 以上，这会大幅侵蚀统计套利的微薄利润。

**结构性变化**：行业兼并、政策变化、公司基本面重大变动都可能打破历史协整关系。2015 年股灾期间，大量历史协整关系同时破裂，导致统计套利策略集体爆仓。这提醒我们：**分仓是必须的**——不要把大量资金集中在一个配对交易上。

**数据和执行延迟**：统计套利对执行速度有要求。当 Z-Score 信号触发时，价差可能在几秒钟内就被其他套利者抹平。在日频级别的配对交易中，这个问题较小；但如果你的策略是分钟级别或更短周期的，延迟就至关重要了。

## 总结

统计套利配对交易是一个优雅且有经济直觉支撑的策略框架。它的核心不在于多么复杂的数学模型，而在于三个关键：**找到真正协整的配对、合理设置入场离场信号、在协整关系破裂时果断止损**。

建议初学者从日频数据开始，用同一行业的大市值股票做配对（银行股、保险股、白酒股都是不错的起点），先用小资金验证信号的稳定性，再逐步扩大仓位。记住，统计套利的每一分利润都来自于你承担的系统风险——协整关系破裂的风险。你赚的不是无风险的"套利"，而是对这种风险的合理定价。