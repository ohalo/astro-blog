---
title: "配对交易与协整：统计套利的数学之美"
publishDate: '2026-06-13'
description: "配对交易与协整：统计套利的数学之美 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 当两只股票"形影不离"

想象一下，麦当劳和肯德基的股价走势应该很像吧？毕竟它们都卖炸鸡汉堡，面临相似的市场竞争和经济环境。如果某天麦当劳突然比肯德基涨得多很多，理性告诉我们：这种偏离可能不会持久，迟早会回归。

**配对交易（Pair Trading）**就是基于这种直觉的量化策略：找到两只价格走势高度相关的股票，当它们的价差（Spread）偏离历史均值时，做多被低估的、做空被高估的，等待价差回归获利。

但问题是：如何科学地判断"两只股票价格走势高度相关"？答案就是**协整（Cointegration）**。

![配对交易原理示意](/images/pair-trading-cointegration/pair_trading_concept.jpg)

## 相关性 ≠ 协整：一个关键区别

很多量化新手会混淆"相关性"和"协整性"，但它们天差地别：

**相关性（Correlation）**衡量的是**收益率**的线性关系。即使两只股票的价差越走越远，它们的日收益率可能依然高度相关。

**协整性（Cointegration）**衡量的是**价格序列**是否存在长期均衡关系。协整的两个价格序列，其线性组合（即价差）是平稳的（Stationary）。

用数学语言表达：
- 如果 $P_t^A$ 和 $P_t^B$ 都是非平稳的I(1)过程（比如随机游走）
- 但存在系数 $\beta$，使得 $P_t^A - \beta P_t^B = \epsilon_t$ 是平稳过程（I(0)）
- 那么 $P_t^A$ 和 $P_t^B$ 就是协整的

这意味着，虽然两只股票的价格各自都会"随机游走"，但它们的**相对价格**（价差）不会偏离太远，迟早会回归均值。

## Engle-Granger 两步法：检验协整的标准流程

如何检验两只股票是否协整？最经典的方法是 **Engle-Granger 两步法**：

### 第一步：OLS回归
用股票A的价格对股票B的价格做OLS回归：

$$P_t^A = \alpha + \beta P_t^B + \epsilon_t$$

得到残差序列 $\hat{\epsilon}_t = P_t^A - \hat{\alpha} - \hat{\beta} P_t^B$

### 第二步：ADF检验
对残差 $\hat{\epsilon}_t$ 做 **Augmented Dickey-Fuller (ADF) 检验**：
- 零假设 $H_0$：残差序列有单位根（非平稳）→ 不存在协整
- 备择假设 $H_1$：残差序列平稳 → 存在协整

如果ADF检验的p-value < 0.05（或更小），我们拒绝零假设，认为两只股票协整。

![协整检验流程图](/images/pair-trading-cointegration/cointegration_test.jpg)

## 实战：用Python实现配对交易策略

让我展示一个完整的配对交易策略框架：

```python
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

# 1. 下载数据
tickers = ['MCD', 'YUM']  # 麦当劳 vs 百胜餐饮（肯德基母公司）
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

# 2. 协整检验
score, p_value, _ = coint(data['MCD'], data['YUM'])
print(f"Cointegration test p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ 两只股票协整！可以配对交易")
    
    # 3. OLS回归获取对冲比例 beta
    X = data['YUM'].values.reshape(-1, 1)
    X = np.hstack([np.ones((len(X), 1)), X])  # 添加截距项
    model = OLS(data['MCD'].values, X).fit()
    beta = model.params[1]
    alpha = model.params[0]
    
    # 4. 计算价差（残差）
    spread = data['MCD'] - (alpha + beta * data['YUM'])
    
    # 5. 计算价差的Z-Score
    spread_mean = spread.rolling(window=20).mean()
    spread_std = spread.rolling(window=20).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 6. 交易信号
    entry_threshold = 2.0  # Z-Score超过2倍标准差时入场
    exit_threshold = 0.5   # Z-Score回归到0.5以内时出场
    
    signal = pd.Series(0, index=z_score.index)
    signal[z_score > entry_threshold] = -1  # 做空价差（做空MCD，做多YUM）
    signal[z_score < -entry_threshold] = 1  # 做多价差（做多MCD，做空YUM）
    
    # 7. 绘制结果
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(data['MCD'], label='MCD')
    axes[0].plot(data['YUM'], label='YUM')
    axes[0].set_title('Price Series')
    axes[0].legend()
    
    axes[1].plot(spread, label='Spread')
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Spread (Residual)')
    axes[1].legend()
    
    axes[2].plot(z_score, label='Z-Score')
    axes[2].axhline(y=entry_threshold, color='r', linestyle='--')
    axes[2].axhline(y=-entry_threshold, color='r', linestyle='--')
    axes[2].axhline(y=0, color='g', linestyle='-')
    axes[2].set_title('Z-Score of Spread')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
```

## 配对交易的实战要点

### 1. 如何筛选配对？

A股市场中，可以优先考虑以下类型的配对：
- **同一行业的不同公司**：比如招商银行 vs 平安银行、贵州茅台 vs 五粮液
- **产业链上下游**：比如铁矿石厂商 vs 钢铁厂
- **替代品**：比如可口可乐 vs 百事可乐

筛选流程：
1. 按行业分类，获取候选股票池
2. 计算所有组合的相关性，筛选高相关对（>0.7）
3. 对高相关对做协整检验，保留p-value < 0.05的配对
4. 计算价差的均值回归速度（Half-life），选择回归较快的

### 2. 交易信号的精细化

简单的Z-Score阈值策略容易受噪音干扰，可以考虑：
- **动态阈值**：根据价差的波动率调整入场阈值
- **确认信号**：Z-Score突破阈值后，等待1-2天确认再入场
- **时间止损**：如果价差在N天后仍未回归，强制平仓

### 3. 风险管理

配对交易虽然是市场中性策略，但仍需注意：
- **配比风险**：OLS估计的 $\beta$ 可能随时间变化，需要定期重新估计
- **结构性断裂**：公司并购、行业政策变化可能导致协整关系失效
- **流动性风险**：确保两只股票都有足够的成交量

## 配对交易的局限性

1. **协整关系可能失效**：市场结构变化、公司基本面变化都可能导致历史协整关系不再成立
2. **价差可能长期不回归**：理论上价差会均值回归，但实践中可能持续偏离数月甚至数年
3. **交易成本敏感**：配对交易通常需要频繁调仓，交易成本会显著侵蚀利润
4. **资金利用率低**：市场中性策略通常不做方向性押注，在单边行情中表现不佳

## 总结

配对交易是统计套利中最经典的策略之一，它将经济学直觉（"相关公司的股价应该同步"）转化为严谨的数学框架（协整检验）。

关键要点：
- ✅ 用**协整检验**而非相关性筛选配对
- ✅ 用**Z-Score**构建交易信号
- ✅ 定期**重新估计**对冲比例 $\beta$
- ✅ 严格**风险管理**，防范协整关系失效

在下一篇文章中，我们将深入探讨**价值因子的量化研究**——另一个量化投资的基石领域。

---

*免责声明：本文仅供学术交流，不构成投资建议。市场有风险，投资需谨慎。*
