---
title: "均值回归策略：捕捉价格偏离后的回归机会"
publishDate: '2026-06-04'
description: "均值回归策略 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 均值回归的数学基础

均值回归（Mean Reversion）是量化交易中最经典的策略之一，其核心假设是：**资产价格在长期会围绕某个均衡值波动，短期偏离后会回归均值**。

这一思想的数学基础可以追溯到1900年路易·巴舍利耶的随机游走理论，以及后来保罗·萨缪尔森对股票价格的几何布朗运动建模。但均值回归策略认为，真实市场价格并非完全随机游走，而是存在**可预测的偏离-回归周期**。

### 平稳性与单位根检验

判断一个时间序列是否具备均值回归特性，首先要检验其平稳性。常用方法包括：

**ADF检验（Augmented Dickey-Fuller Test）**

原假设 H₀：序列存在单位根（非平稳）
备择假设 H₁：序列平稳（均值回归）

ADF检验的统计量越负，拒绝原假设的证据越强。在实践中，p值小于0.05通常被认为是平稳的。

**Hurst指数**

Hurst指数 H 是判断序列特性的重要指标：
- H < 0.5：均值回归（anti-persistent）
- H = 0.5：随机游走
- H > 0.5：趋势持续（persistent）

均值回归策略偏好 H < 0.5 的资产。

## 均值回归策略的构建流程

### 1. 选择标的与观察期

均值回归策略适用于：
- 配对股票（如可口可乐 vs 百事可乐）
- ETF与其成分股
- 期货与其现货
- 同一行业内的相关性高的股票

观察期通常选择20-60个交易日，太短噪声大，太长均值可能已发生结构性变化。

### 2. 计算偏离度（Deviation）

最常用的偏离度度量是**Z-Score**：

Zₜ = (Pₜ - μ) / σ

其中：
- Pₜ：当前价格
- μ：观察期内均价值
- σ：观察期内标准差

**入场信号**：|Zₜ| > 2（价格偏离超过2个标准差）
**出场信号**：Zₜ 回归至 [-0.5, 0.5] 区间

### 3. 仓位管理与止损

均值回归策略的最大风险是**趋势持续而非回归**（"均值回归陷阱"）。因此必须设置：

- **时间止损**：若持仓超过N天（如10天）仍未回归，强制平仓
- **价格止损**：若价格继续偏离超过3-4个标准差，承认模型失效
- **仓位分配**：单次策略占用资金不超过总资金的5-10%

## 均值回归 vs 配对交易

很多初学者容易混淆这两个概念，核心区别在于：

| 维度 | 均值回归 | 配对交易 |
|------|---------|---------|
| 标的数量 | 单资产 | 两个高度相关资产 |
| 参考基准 | 历史均值 | 两个资产的价差/比率 |
| 假设 | 价格回归历史均值 | 价差回归历史均值 |
| 适用场景 | 波动较大的个股 | 高度相关的同行业股票 |

实践中，配对交易是均值回归的一种特殊形式，但加入了**对冲**概念，理论上市场中性更强。

## Python实战：构建简单的均值回归策略

下面是用Python实现的简化版均值回归策略回测框架：

```python
import pandas as pd
import numpy as np
from scipy import stats

def mean_reversion_backtest(price_series, lookback=20, z_entry=2.0, z_exit=0.5):
    """
    均值回归策略回测
    
    Parameters:
    - price_series: 价格序列
    - lookback: 滚动窗口天数
    - z_entry: 入场Z值阈值
    - z_exit: 出场Z值阈值
    """
    # 计算滚动Z-Score
    rolling_mean = price_series.rolling(window=lookback).mean()
    rolling_std = price_series.rolling(window=lookback).std()
    
    z_score = (price_series - rolling_mean) / rolling_std
    
    # 生成信号
    signals = pd.Series(0, index=price_series.index)
    signals[z_score < -z_entry] = 1   # 价格偏低，买入
    signals[z_score > z_entry] = -1   # 价格偏高，卖出
    signals[(z_score >= -z_exit) & (z_score <= z_exit)] = 0  # 出场
    
    # 计算收益
    returns = price_series.pct_change()
    strategy_returns = signals.shift(1) * returns
    
    # 性能指标
    cumulative_returns = (1 + strategy_returns).cumprod()
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    return {
        'cumulative_returns': cumulative_returns,
        'sharpe_ratio': sharpe,
        'signals': signals,
        'z_score': z_score
    }

# 使用示例
# result = mean_reversion_backtest(stock_price, lookback=20, z_entry=2.0, z_exit=0.5)
```

## 均值回归策略的实战陷阱

### 陷阱1：结构性断点

当资产基本面发生变化（如公司重组、行业政策变化），历史均值可能不再适用。此时均值回归会变成"接飞刀"。

**应对方法**：使用滚动窗口（如60天）而非全历史数据计算均值，使策略适应结构性变化。

### 陷阱2：趋势市场的持续偏离

在强趋势市场中（如科技股牛市），价格可能长期偏离均值而不回归。此时均值回归策略会频繁止损。

**应对方法**：加入趋势过滤器，如200日均线。当价格高于200日均线时，只允许做多均值回归（不做空）。

### 陷阱3：交易成本侵蚀收益

均值回归策略通常交易频率较高（每周可能多次交易），交易成本对收益影响显著。

**应对方法**：
- 提高Z值入场阈值（如从2.0提高到2.5）
- 加入持有期限制（至少持有N天再出场）
- 选择低佣金券商或使用期权对冲

## 进阶：Ornstein-Uhlenbeck模型

对于严格的均值回归建模，可以使用**Ornstein-Uhlenbeck（OU）过程**：

dXₜ = θ(μ - Xₜ)dt + σdWₜ

其中：
- θ：回归速度（越大回归越快）
- μ：长期均值
- σ：波动率
- Wₜ：维纳过程

通过最大似然估计可以拟合OU参数，进而计算**最优出场时机**（当预期回归收益 > 交易成本时持有，否则立即出场）。

## 总结

均值回归策略是量化交易中的基础策略，核心要点：

1. **先检验平稳性**（ADF检验、Hurst指数）
2. **用Z-Score量化偏离度**，而非简单看价格高低
3. **必须设置止损**（时间和价格双重止损）
4. **注意结构性变化**，避免用过期均值
5. **控制交易频率**，降低交易成本影响

对于初学者，建议先从配对交易入手（风险相对可控），再逐步尝试单资产均值回归策略。

![均值回归示意图](/images/2026-06-04-mean-reversion-strategy/mean_reversion_chart.jpg)

*上图：价格围绕均值波动，偏离超过2倍标准差时入场，回归时出场*

## 扩展阅读

- Ernest Chan《量化交易：如何建立自己的算法交易事业》
- Avellaneda & Lee《Statistical Arbitrage in the US Equities Market》
- 配对交易经典论文：Gatev, Goetzmann & Rouwenhorst (1999)

