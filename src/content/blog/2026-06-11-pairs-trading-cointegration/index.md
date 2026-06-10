---
title: "配对交易与协整分析：统计套利的核心技术"
publishDate: '2026-06-11'
description: "配对交易与协整分析：统计套利的核心技术 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言

统计套利是量化交易的重要分支，而配对交易（Pairs Trading）则是其中最经典的策略之一。本文将深入探讨如何利用协整分析构建稳健的配对交易策略。

## 配对交易的基本原理

配对交易的核心思想是寻找两个价格具有长期均衡关系的股票，当它们的价格偏离均衡时，通过做多低估股票、做空高估股票来获利。

### 为什么选择配对交易？

- **市场中性**：对冲了市场风险，专注于相对价值
- **低风险**：基于统计学原理，风险可控
- **适应性强**：在震荡市和趋势市中都能获利

## 协整分析：寻找长期均衡关系

### 什么是协整？

协整（Cointegration）是指两个或多个非平稳时间序列的线性组合是平稳的。在配对交易中，如果两只股票的价格序列是协整的，那么它们之间存在长期的均衡关系。

### 协整检验步骤

1. **单位根检验（ADF检验）**
   - 检验每只股票的价格序列是否非平稳
   - 原假设：序列有单位根（非平稳）

2. **协整检验（Engle-Granger检验或Johansen检验）**
   - 检验残差序列是否平稳
   - 如果残差平稳，则两只股票协整

3. **计算对冲比率（Hedge Ratio）**
   - 通过OLS回归得到：Price_A = α + β × Price_B + ε
   - β就是对冲比率

## Python实战：协整检验与配对选择

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf

# 下载股票数据
def download_data(ticker1, ticker2, start='2020-01-01'):
    stock1 = yf.download(ticker1, start=start)['Adj Close']
    stock2 = yf.download(ticker2, start=start)['Adj Close']
    return pd.DataFrame({ticker1: stock1, ticker2: stock2}).dropna()

# ADF检验
def adf_test(series, title=''):
    result = adfuller(series, autolag='AIC')
    print(f'\nADF Test: {title}')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    return result[1] <= 0.05

# 协整检验
def cointegration_test(s1, s2):
    score, pvalue, _ = coint(s1, s2)
    print(f'\nCointegration Test')
    print(f'Score: {score:.4f}')
    print(f'p-value: {pvalue:.4f}')
    return pvalue <= 0.05

# 示例：检验可口可乐和百事可乐
data = download_data('KO', 'PEP')
s1 = np.log(data['KO'])
s2 = np.log(data['PEP'])

print("=== 单位根检验 ===")
is_s1_stationary = adf_test(s1, 'KO')
is_s2_stationary = adf_test(s2, 'PEP')

if not is_s1_stationary and not is_s2_stationary:
    print("\n两只股票价格都是非平稳的，可以进行协整检验")
    is_cointegrated = cointegration_test(s1, s2)
    if is_cointegrated:
        print("✓ 两只股票存在协整关系！")
```

## 构建交易信号

### 计算价差的Z-Score

```python
from sklearn.linear_model import LinearRegression

# 计算对冲比率
def calculate_hedge_ratio(s1, s2):
    X = s2.values.reshape(-1, 1)
    y = s1.values
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]

# 计算残差和Z-Score
hedge_ratio = calculate_hedge_ratio(s1, s2)
spread = s1 - hedge_ratio * s2
z_score = (spread - spread.mean()) / spread.std()

# 交易信号
def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    signals = pd.DataFrame(index=z_score.index)
    signals['z_score'] = z_score
    
    # 1: 做多价差, -1: 做空价差, 0: 平仓
    signals['position'] = 0
    signals.loc[z_score < -entry_threshold, 'position'] = 1  # 做多
    signals.loc[z_score > entry_threshold, 'position'] = -1   # 做空
    signals.loc[abs(z_score) < exit_threshold, 'position'] = 0  # 平仓
    
    return signals
```

## 风险控制与资金管理

### 1. 止损策略

- **时间止损**：如果持仓超过N天仍未收敛，强制平仓
- **价格止损**：价差突破历史极值的一定比例时止损

### 2. 仓位管理

```python
# 基于波动率的仓位管理
def calculate_position_size(z_score, volatility, risk_per_trade=0.02):
    # 根据Z-Score的绝对值调整仓位
    signal_strength = min(abs(z_score) / 3.0, 1.0)  # 归一化到[0,1]
    
    # 根据波动率调整
    vol_adjustment = 0.2 / volatility  # 目标波动率20%
    
    # 最终仓位
    position_size = risk_per_trade * signal_strength * vol_adjustment
    return np.clip(position_size, 0, 0.1)  # 最大10%仓位
```

## 实战案例分析

### 案例1：中国银行股配对

选择工商银行（601398.SH）和建设银行（601939.SH）：

```python
# 回测结果示例
"""
策略表现：
- 年化收益率：12.3%
- 夏普比率：1.85
- 最大回撤：-8.2%
- 胜率：58.6%
- 平均持仓天数：8.5天
"""
```

### 案例2：美股科技股配对

选择微软（MSFT）和苹果（AAPL）：

```python
# 注意事项
"""
1. 美股盘前盘后交易需注意流动性
2. 科技股相关性在高波动市场可能失效
3. 建议加入基本面筛选条件
"""
```

## 常见问题与解决方案

### Q1: 协整关系断裂怎么办？

**A**: 定期重新检验协整关系（建议每月），如果发现断裂：
- 立即停止交易
- 平仓现有头寸
- 重新寻找新的配对

### Q2: 如何处理幸存者偏差？

**A**: 
- 使用存活股票和退市股票的数据
- 在回测中考虑交易成本
- 使用样本外数据验证

### Q3: 高频数据是否有效？

**A**: 
- 低频数据（日线）更适合配对交易
- 高频数据噪音大，交易成本过高
- 建议使用15分钟或1小时K线

## 总结

配对交易是一种经典的统计套利策略，核心在于：
1. 通过协整分析找到具有长期均衡关系的股票对
2. 利用价差的均值回归特性获利
3. 严格的风险管理和仓位控制

**关键要点**：
- ✓ 协整检验是策略成功的基础
- ✓ Z-Score阈值需要根据市场调整
- ✓ 止损和仓位管理不可或缺
- ✓ 定期监控协整关系的稳定性

## 参考资料

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
2. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*

![配对交易原理图](/images/2026-06-11-pairs-trading-cointegration/pairs_trading_concept.jpg)

*图1：配对交易的基本原理 - 利用价格偏离获利*

![Z-Score信号图](/images/2026-06-11-pairs-trading-cointegration/z_score_signals.png)

*图2：Z-Score交易信号示意图 - 入场和出场时机*
