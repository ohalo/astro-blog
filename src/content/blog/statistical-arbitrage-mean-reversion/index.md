---
title: "统计套利：均值回归策略"
description: "深入探讨统计套利的核心原理与均值回归策略的实践方法，从协整检验到配对交易，构建市场中性投资策略。"
publishDate: '2026-06-17'
language: Chinese
updatedDate: 2026-06-17
tags: ["统计套利", "均值回归", "配对交易", "协整分析"]
category: "量化交易"
keywords: ["统计套利", "均值回归", "配对交易", "协整", "市场中性"]
slug: "statistical-arbitrage-mean-reversion"
draft: false
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是一类基于量化分析的市场中性投资策略，其核心思想是利用资产价格之间的统计关系，通过多空对冲获取稳定收益。均值回归策略作为统计套利的重要分支，假设资产价格偏离长期均衡后会回归均值，从而捕捉价格修复的机会。

## 统计套利的理论基础

### 均值回归假设

均值回归是金融市场中普遍存在的现象。大多数金融资产的价格序列都呈现出"均值回归"特征：

1. **短期偏离**：资产价格会因为市场情绪、流动性冲击等因素短期偏离其内在价值
2. **长期回归**：随着市场有效性发挥作用，价格会逐渐回归到合理水平
3. **可预测性**：通过统计方法可以识别这种偏离并预测回归时机

### 统计套利的核心要素

一个完整的统计套利策略包含以下关键要素：

- **资产选择**：识别具有稳定关系的资产对或组合
- **信号生成**：判断当前价格是否偏离均衡水平
- **交易执行**：构建多空组合并实施交易
- **风险管理**：控制敞口、止损和仓位管理

## 配对交易：统计套利的经典范式

### 配对交易的基本原理

配对交易（Pairs Trading）是统计套利中最经典的策略。其基本思路是：

1. 找到两只具有长期协整关系的股票
2. 当价格比（或价差）偏离历史均值时，做多低估资产、做空高估资产
3. 等待价格比回归均值后平仓，获取差价收敛的收益

### 协整检验：寻找可靠的配对

协整（Cointegration）是配对交易的基础。只有存在协整关系的资产对，才能保证价格偏离后能够回归。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

class PairsTrading:
    """配对交易策略框架"""
    
    def __init__(self, stock1_data, stock2_data):
        """
        参数:
        - stock1_data: Series, 第一只股票的价格数据
        - stock2_data: Series, 第二只股票的价格数据
        """
        self.stock1 = stock1_data
        self.stock2 = stock2_data
        self.spread = None
        self.z_score = None
        
    def test_cointegration(self, significance=0.05):
        """
        进行协整检验
        
        返回:
        - is_cointegrated: bool, 是否存在协整关系
        - p_value: float, 检验p值
        - hedge_ratio: float, 对冲比率（回归系数）
        """
        # Engle-Granger 协整检验
        coint_result = coint(self.stock1, self.stock2)
        p_value = coint_result[1]
        
        # 计算对冲比率（通过OLS回归）
        model = OLS(self.stock1, self.stock2).fit()
        hedge_ratio = model.params[0]
        
        is_cointegrated = p_value < significance
        
        return is_cointegrated, p_value, hedge_ratio
    
    def calculate_spread(self, hedge_ratio):
        """
        计算价差（Spread）
        
        价差定义: spread = stock1 - hedge_ratio * stock2
        """
        self.spread = self.stock1 - hedge_ratio * self.stock2
        return self.spread
    
    def calculate_z_score(self, window=20):
        """
        计算价差的Z-score
        
        Z-score = (当前价差 - 均值) / 标准差
        """
        if self.spread is None:
            raise ValueError("请先调用 calculate_spread() 计算价差")
        
        rolling_mean = self.spread.rolling(window=window).mean()
        rolling_std = self.spread.rolling(window=window).std()
        
        self.z_score = (self.spread - rolling_mean) / rolling_std
        return self.z_score
    
    def generate_trading_signals(self, entry_threshold=2.0, 
                                 exit_threshold=0.5):
        """
        生成交易信号
        
        参数:
        - entry_threshold: float, 入场阈值（Z-score绝对值）
        - exit_threshold: float, 出场阈值（Z-score绝对值）
        
        返回:
        - signals: DataFrame, 包含多空信号的DataFrame
        """
        signals = pd.DataFrame(index=self.z_score.index)
        signals['z_score'] = self.z_score
        signals['position'] = 0
        
        # 生成信号
        for i in range(1, len(signals)):
            z = signals['z_score'].iloc[i]
            prev_z = signals['z_score'].iloc[i-1]
            
            # 入场信号
            if z < -entry_threshold:  # 价差过低，做多stock1，做空stock2
                signals['position'].iloc[i] = 1
            elif z > entry_threshold:  # 价差过高，做空stock1，做多stock2
                signals['position'].iloc[i] = -1
            
            # 出场信号
            elif abs(z) < exit_threshold:
                signals['position'].iloc[i] = 0
            
            # 保持前一期的仓位
            else:
                signals['position'].iloc[i] = signals['position'].iloc[i-1]
        
        return signals
    
    def backtest(self, signals, transaction_cost=0.001):
        """
        回测配对交易策略
        
        参数:
        - signals: DataFrame, 交易信号
        - transaction_cost: float, 交易成本（单边）
        
        返回:
        - returns: Series, 策略收益率
        - portfolio_value: Series, 组合价值
        """
        # 计算每日收益
        stock1_returns = self.stock1.pct_change()
        stock2_returns = self.stock2.pct_change()
        
        # 计算策略收益
        strategy_returns = (signals['position'].shift(1) * 
                           (stock1_returns - stock2_returns))
        
        # 计算交易成本
        position_change = signals['position'].diff().abs()
        trading_cost = position_change * transaction_cost
        
        # 净收益
        net_returns = strategy_returns - trading_cost
        
        # 计算累计收益
        cumulative_returns = (1 + net_returns).cumprod()
        
        return net_returns, cumulative_returns
```

### 实际应用示例

```python
# 示例：对中国平安和中国人寿进行配对交易
import yfinance as yf

# 下载数据
stock1 = yf.download('601318.SS', start='2020-01-01', end='2024-01-01')['Adj Close']
stock2 = yf.download('601628.SS', start='2020-01-01', end='2024-01-01')['Adj Close']

# 创建配对交易对象
pairs = PairsTrading(stock1, stock2)

# 协整检验
is_cointegrated, p_value, hedge_ratio = pairs.test_cointegration()
print(f"协整检验 p-value: {p_value:.4f}")
print(f"是否存在协整关系: {is_cointegrated}")
print(f"对冲比率: {hedge_ratio:.4f}")

if is_cointegrated:
    # 计算价差
    spread = pairs.calculate_spread(hedge_ratio)
    
    # 计算Z-score
    z_score = pairs.calculate_z_score(window=20)
    
    # 生成交易信号
    signals = pairs.generate_trading_signals(
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    # 回测
    returns, cumulative = pairs.backtest(signals, transaction_cost=0.001)
    
    # 绘制结果
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 子图1：价格序列
    axes[0].plot(stock1.index, stock1.values, label='中国平安')
    axes[0].plot(stock2.index, stock2.values, label='中国人寿')
    axes[0].set_title('价格序列')
    axes[0].legend()
    
    # 子图2：Z-score
    axes[1].plot(z_score.index, z_score.values)
    axes[1].axhline(y=2, color='r', linestyle='--', label='入场阈值')
    axes[1].axhline(y=-2, color='r', linestyle='--')
    axes[1].axhline(y=0.5, color='g', linestyle='--', label='出场阈值')
    axes[1].axhline(y=-0.5, color='g', linestyle='--')
    axes[1].set_title('Z-score')
    axes[1].legend()
    
    # 子图3：累计收益
    axes[2].plot(cumulative.index, cumulative.values)
    axes[2].set_title('策略累计收益')
    
    plt.tight_layout()
    plt.show()
```

## 多资产统计套利

### 统计套利的扩展

随着量化技术的发展，统计套利已从简单的配对交易扩展到多资产组合：

#### 1. 多因子模型

使用多个股票构建市场中性组合：

```python
class MultiAssetStatisticalArbitrage:
    """多资产统计套利框架"""
    
    def __init__(self, stock_data, factor_data):
        """
        参数:
        - stock_data: DataFrame, 多只股票的价格数据
        - factor_data: DataFrame, 因子数据（如行业因子、风格因子）
        """
        self.stock_data = stock_data
        self.factor_data = factor_data
        
    def calculate_residual_returns(self):
        """
        计算残差收益（剔除因子影响后的收益）
        """
        # 对每只股票进行因子回归
        residual_returns = pd.DataFrame()
        
        for stock in self.stock_data.columns:
            # OLS回归：股票收益 ~ 因子收益
            model = OLS(
                self.stock_data[stock].pct_change().dropna(),
                self.factor_data.loc[self.stock_data.index].dropna()
            ).fit()
            
            # 保存残差
            residual_returns[stock] = model.resid
            
        return residual_returns
    
    def construct_market_neutral_portfolio(self, residual_returns, 
                                          n_long=10, n_short=10):
        """
        构建市场中性组合
        
        参数:
        - residual_returns: DataFrame, 残差收益
        - n_long: int, 做多股票数量
        - n_short: int, 做空股票数量
        """
        # 计算每个时间点的残差收益排序
        rankings = residual_returns.rank(axis=1, ascending=False)
        
        # 选择做多和做空的股票
        long_stocks = rankings <= n_long
        short_stocks = rankings >= (len(rankings.columns) - n_short + 1)
        
        # 构建权重
        weights = pd.DataFrame(0, index=rankings.index, 
                              columns=rankings.columns)
        weights[long_stocks] = 1 / n_long
        weights[short_stocks] = -1 / n_short
        
        # 确保市场中性（权重和为0）
        weights = weights.sub(weights.mean(axis=1), axis=0)
        
        return weights
```

#### 2. 主成分分析（PCA）

使用PCA识别资产组合的主要风险因子：

```python
from sklearn.decomposition import PCA

def pca_statistical_arbitrage(stock_returns, n_components=5):
    """
    基于PCA的统计套利
    
    参数:
    - stock_returns: DataFrame, 股票收益率
    - n_components: int, 主成分数量
    """
    # 标准化收益率
    standardized_returns = (stock_returns - stock_returns.mean()) / stock_returns.std()
    
    # PCA分解
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized_returns.fillna(0))
    
    # 重构收益率（剔除主成分）
    reconstructed = pca.inverse_transform(principal_components)
    residual_returns = standardized_returns - reconstructed
    
    # 基于残差构建交易信号
    z_score = (residual_returns - residual_returns.rolling(20).mean()) / \
              residual_returns.rolling(20).std()
    
    signals = pd.DataFrame(0, index=z_score.index, 
                          columns=z_score.columns)
    signals[z_score > 2] = -1  # 做空
    signals[z_score < -2] = 1   # 做多
    
    return signals, residual_returns
```

## 风险管理与实战考虑

### 1. 模型风险

统计套利依赖统计模型的准确性，模型失效会导致策略亏损。

**应对措施**：
- 定期重新估计模型参数
- 使用滚动窗口或指数加权方法
- 设置模型失效预警指标

### 2. 执行风险

理论上的无风险套利在实际执行中面临诸多挑战：

```python
def calculate_implementation_shortfall(order_size, daily_volume, 
                                       price_impact=0.1):
    """
    计算执行缺口
    
    参数:
    - order_size: float, 订单大小
    - daily_volume: float, 日均成交量
    - price_impact: float, 价格冲击系数
    """
    # 计算市场影响
    market_impact = price_impact * (order_size / daily_volume) ** 2
    
    # 计算执行成本
    execution_cost = market_impact * order_size
    
    return execution_cost
```

### 3. 黑天鹅事件

统计套利在极端市场环境下可能失效，如2008年金融危机期间，许多统计套利策略出现巨额亏损。

**应对措施**：
- 设置严格的止损规则
- 控制单个策略的仓位上限
- 多元化策略组合

## 实证研究与绩效评估

### 策略评估指标

```python
def evaluate_statistical_arbitrage(returns, benchmark_returns=None):
    """评估统计套利策略"""
    
    # 基础指标
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 市场中性检验
    if benchmark_returns is not None:
        beta = np.cov(returns, benchmark_returns)[0, 1] / \
               np.var(benchmark_returns)
    else:
        beta = np.nan
    
    performance = {
        '总收益率': total_return,
        '年化收益率': annual_return,
        '年化波动率': annual_volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '市场Beta': beta
    }
    
    return performance
```

## 总结

统计套利：均值回归策略是量化投资中的重要组成部分。通过科学的统计方法和严格的风险管理，统计套利策略可以在市场波动中获取稳定的超额收益。

然而，统计套利并非"免费的午餐"。策略的成功实施需要：

1. **扎实的统计学基础**：理解协整、平稳性、残差分析等概念
2. **严谨的回测验证**：避免过度拟合和数据窥探偏差
3. **有效的风险管理**：控制杠杆、设置止损、分散投资
4. **持续的策略迭代**：市场环境变化要求策略不断进化

对于希望进入统计套利领域的量化投资者，建议从简单的配对交易开始，逐步扩展到多资产组合，并始终保持对风险的敬畏之心。

---

**参考文献**：

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.
3. Ganapathy, V. (2004). "Statistical Arbitrage and High-Frequency Data with an Application to Eurodollar Futures." *University of California, Berkeley*.
