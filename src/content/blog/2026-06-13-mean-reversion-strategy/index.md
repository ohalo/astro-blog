---
title: "均值回归交易策略：从统计学原理到实战"
publishDate: '2026-06-13'
description: "均值回归交易策略 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言

均值回归（Mean Reversion）是量化交易中最经典的策略之一，其核心假设是：资产价格在长期会围绕某个均衡值波动，当价格显著偏离均衡值时，未来会向均值回归。

与趋势跟踪策略相反，均值回归策略本质上是一种"逆势"策略——在价格下跌时买入，在价格上涨时卖出。这种策略在震荡市中表现优异，但在强趋势市中容易亏损。

本文将深入探讨均值回归的统计学原理、常用指标、策略构建方法以及实战中的注意事项。

## 统计学基础

### 平稳性检验

均值回归策略的前提是价格序列具有平稳性（Stationarity）。一个平稳的时间序列具有以下特征：

1. **均值恒定**：序列的均值不随时间变化
2. **方差恒定**：序列的方差不随时间变化
3. **协方差仅依赖于时间差**：序列的自协方差只与滞后期有关

### Augmented Dickey-Fuller检验

最常用的平稳性检验是ADF检验（Augmented Dickey-Fuller Test）。其原假设是"序列存在单位根（非平稳）"，备择假设是"序列平稳"。

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def adf_test(price_series):
    result = adfuller(price_series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if result[1] < 0.05:
        print("序列平稳，存在均值回归特性")
    else:
        print("序列非平稳，不存在均值回归特性")
```

### 半衰期（Half-life）

半衰期是衡量均值回归速度的重要指标，表示价格偏离均值后回归到一半所需时间。

对于Ornstein-Uhlenbeck过程：

$$
dx_t = \theta(\mu - x_t)dt + \sigma dW_t
$$

其中：
- $\theta$ 是均值回归速度
- $\mu$ 是长期均值
- $\sigma$ 是波动率

半衰期计算公式为：

$$
\text{Half-life} = \frac{\ln(2)}{\theta}
$$

```python
import numpy as np
import statsmodels.api as sm

def calculate_half_life(price_series):
    """计算均值回归的半衰期"""
    price_series = np.array(price_series)
    
    # 计算价格变化
    delta_y = np.diff(price_series)
    
    # 构建回归模型: delta_y = a + b * y_{t-1} + epsilon
    y_lag = price_series[:-1]
    X = sm.add_constant(y_lag)
    model = sm.OLS(delta_y, X)
    results = model.fit()
    
    # 提取回归系数
    theta = -results.params[1]  # b的负值
    
    if theta <= 0:
        return np.inf  # 不均值回归
    
    half_life = np.log(2) / theta
    return half_life
```

## 常用均值回归指标

### 1. 布林带（Bollinger Bands）

布林带由三条线组成：
- 中轨：N日移动平均线
- 上轨：中轨 + K倍标准差
- 下轨：中轨 - K倍标准差

**交易信号**：
- 价格触及或突破下轨 → 买入信号
- 价格触及或突破上轨 → 卖出信号
- 价格回归中轨 → 平仓信号

```python
import pandas as pd

def bollinger_bands(price, window=20, num_std=2):
    """计算布林带"""
    rolling_mean = price.rolling(window=window).mean()
    rolling_std = price.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, rolling_mean, lower_band

def bollinger_signal(price, upper, lower, mean):
    """生成布林带交易信号"""
    signals = pd.Series(index=price.index, dtype=str)
    
    # 买入信号：价格低于下轨
    signals[price < lower] = 'BUY'
    
    # 卖出信号：价格高于上轨
    signals[price > upper] = 'SELL'
    
    # 平仓信号：价格回到中轨附近
    signals[(price > mean * 0.98) & (price < mean * 1.02)] = 'CLOSE'
    
    return signals
```

### 2. RSI相对强弱指标

RSI衡量价格变动的速度和幅度，取值范围0-100。

**交易信号**：
- RSI < 30：超卖区域，考虑买入
- RSI > 70：超买区域，考虑卖出

```python
def calculate_rsi(price, window=14):
    """计算RSI指标"""
    delta = price.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def rsi_signal(rsi, oversold=30, overbought=70):
    """生成RSI交易信号"""
    signals = pd.Series(index=rsi.index, dtype=str)
    
    signals[rsi < oversold] = 'BUY'
    signals[rsi > overbought] = 'SELL'
    
    return signals
```

### 3. Z-Score标准化分数

Z-Score衡量当前价格偏离均值的标准差倍数：

$$
z_t = \frac{P_t - \mu}{\sigma}
$$

其中：
- $P_t$ 是当前价格
- $\mu$ 是N日移动平均
- $\sigma$ 是N日标准差

**交易信号**：
- Z-Score < -2：价格显著低于均值，买入
- Z-Score > 2：价格显著高于均值，卖出
- Z-Score回归0附近：平仓

```python
def calculate_z_score(price, window=20):
    """计算Z-Score"""
    rolling_mean = price.rolling(window=window).mean()
    rolling_std = price.rolling(window=window).std()
    
    z_score = (price - rolling_mean) / rolling_std
    
    return z_score

def z_score_signal(z_score, buy_threshold=-2, sell_threshold=2):
    """生成Z-Score交易信号"""
    signals = pd.Series(index=z_score.index, dtype=str)
    
    signals[z_score < buy_threshold] = 'BUY'
    signals[z_score > sell_threshold] = 'SELL'
    signals[(z_score >= buy_threshold * 0.5) & 
            (z_score <= sell_threshold * 0.5)] = 'CLOSE'
    
    return signals
```

## 配对交易（Pairs Trading）

配对交易是均值回归策略的经典应用，通过交易两个相关性高的资产来实现市场中性策略。

### 步骤1：寻找配对资产

使用距离法或协整检验寻找可配对资产：

```python
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(stocks_data):
    """寻找协整配对的股票"""
    n = len(stocks_data.columns)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            S1 = stocks_data.iloc[:, i]
            S2 = stocks_data.iloc[:, j]
            
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            
            if pvalue < 0.05:  # 显著协整
                pairs.append((stocks_data.columns[i], 
                             stocks_data.columns[j], pvalue))
    
    return pairs, score_matrix, pvalue_matrix
```

### 步骤2：计算价差并交易

```python
def pairs_trading_strategy(stock1, stock2, window=20):
    """配对交易策略"""
    # 计算对冲比例（使用滚动回归）
    hedge_ratio = []
    for i in range(window, len(stock1)):
        X = stock2[i-window:i]
        y = stock1[i-window:i]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        hedge_ratio.append(model.params[1])
    
    hedge_ratio = pd.Series(hedge_ratio, 
                           index=stock1.index[window:])
    
    # 计算价差
    spread = stock1[window:] - hedge_ratio * stock2[window:]
    
    # 计算Z-Score
    z_score = calculate_z_score(spread, window=20)
    
    # 生成交易信号
    signals = z_score_signal(z_score)
    
    return signals, spread, z_score
```

## 实战策略构建

### 多因子均值回归策略

结合多个指标提高信号可靠性：

```python
def multi_factor_mean_reversion(price, window=20):
    """多因子均值回归策略"""
    signals = pd.DataFrame(index=price.index)
    
    # 因子1：Z-Score
    z_score = calculate_z_score(price, window)
    signals['z_buy'] = (z_score < -2).astype(int)
    signals['z_sell'] = (z_score > 2).astype(int)
    
    # 因子2：布林带
    upper, middle, lower = bollinger_bands(price, window)
    signals['bb_buy'] = (price < lower).astype(int)
    signals['bb_sell'] = (price > upper).astype(int)
    
    # 因子3：RSI
    rsi = calculate_rsi(price)
    signals['rsi_buy'] = (rsi < 30).astype(int)
    signals['rsi_sell'] = (rsi > 70).astype(int)
    
    # 综合信号（至少2个因子同向）
    signals['buy_signal'] = ((signals['z_buy'] + 
                             signals['bb_buy'] + 
                             signals['rsi_buy']) >= 2).astype(int)
    
    signals['sell_signal'] = ((signals['z_sell'] + 
                              signals['bb_sell'] + 
                              signals['rsi_sell']) >= 2).astype(int)
    
    return signals
```

### 仓位管理

均值回归策略的仓位管理至关重要：

```python
def position_sizing(z_score, max_position=1.0):
    """基于Z-Score的仓位管理"""
    # Z-Score绝对值越大，仓位越重
    position = np.abs(z_score) / 4  # 归一化
    position = np.clip(position, 0, max_position)
    
    # 方向：负Z-Score做多，正Z-Score做空
    direction = -np.sign(z_score)
    
    return position * direction
```

## 风险控制

### 1. 止损策略

均值回归策略需要严格的止损：

```python
def mean_reversion_stop_loss(price, entry_price, stop_loss_rate=0.05):
    """均值回归止损策略"""
    loss = (price - entry_price) / entry_price
    
    # 做多止损：价格继续下跌超过5%
    # 做空止损：价格继续上涨超过5%
    stop_signal = (np.abs(loss) > stop_loss_rate).astype(int)
    
    return stop_signal
```

### 2. 最大持仓时间

防止策略在强趋势市中持续亏损：

```python
def max_holding_period_strategy(signals, max_days=10):
    """最大持仓时间控制"""
    position = 0
    holding_days = 0
    adjusted_signals = signals.copy()
    
    for i in range(len(signals)):
        if position != 0:
            holding_days += 1
        
        if signals.iloc[i] == 'BUY':
            position = 1
            holding_days = 0
        elif signals.iloc[i] == 'SELL':
            position = -1
            holding_days = 0
        elif holding_days >= max_days:
            adjusted_signals.iloc[i] = 'CLOSE'
            position = 0
            holding_days = 0
    
    return adjusted_signals
```

### 3. 市场环境过滤

均值回归策略在趋势市中表现较差，需要市场环境过滤：

```python
def market_regime_filter(price, window=60):
    """市场环境过滤：识别趋势市vs震荡市"""
    # 计算波动率
    volatility = price.rolling(window=window).std()
    vol_percentile = volatility.rank(pct=True)
    
    # 计算趋势强度（使用ADF检验p-value）
    trend_strength = []
    for i in range(window, len(price)):
        p_value = adfuller(price[i-window:i])[1]
        trend_strength.append(p_value)
    
    trend_strength = pd.Series(trend_strength, 
                              index=price.index[window:])
    
    # 过滤条件：
    # 1. 高波动时停止交易
    # 2. 强趋势时停止交易（p-value > 0.05表示非平稳）
    trade_signal = ((vol_percentile < 0.8) & 
                    (trend_strength < 0.05)).astype(int)
    
    return trade_signal
```

## 回测与分析

### 回测框架

```python
class MeanReversionBacktest:
    def __init__(self, price_data, initial_capital=100000):
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
    
    def run_backtest(self, signals):
        """运行回测"""
        portfolio_value = []
        
        for i in range(1, len(signals)):
            date = signals.index[i]
            price = self.price_data.iloc[i]
            signal = signals.iloc[i]
            
            # 执行交易
            if signal == 'BUY' and self.position == 0:
                # 买入
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                self.trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares
                })
            
            elif signal == 'SELL' and self.position == 0:
                # 卖空（需要支持卖空）
                pass
            
            elif signal == 'CLOSE' and self.position != 0:
                # 平仓
                revenue = self.position * price
                self.capital += revenue
                self.trades.append({
                    'date': date,
                    'action': 'CLOSE',
                    'price': price,
                    'shares': self.position
                })
                self.position = 0
            
            # 计算组合价值
            portfolio_value.append(
                self.capital + self.position * price
            )
        
        return portfolio_value
```

### 绩效指标

```python
def calculate_performance_metrics(portfolio_values, risk_free_rate=0.03):
    """计算策略绩效指标"""
    portfolio_values = np.array(portfolio_values)
    
    # 收益率
    total_return = (portfolio_values[-1] / 
                    portfolio_values[0] - 1) * 100
    
    # 年化收益率
    trading_days = len(portfolio_values)
    years = trading_days / 252
    annual_return = (1 + total_return/100) ** (1/years) - 1
    
    # 夏普比率
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = (np.mean(daily_returns) * 252 - 
                   risk_free_rate) / (np.std(daily_returns) * 
                                      np.sqrt(252))
    
    # 最大回撤
    cumulative = np.cummax(portfolio_values)
    drawdown = (portfolio_values - cumulative) / cumulative
    max_drawdown = np.min(drawdown) * 100
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }
```

## 实战注意事项

### 1. 交易成本

均值回归策略通常交易频繁，交易成本影响显著：

- **佣金**：选择低佣金券商
- **滑点**：使用限价单减少滑点
- **买卖价差**：避免交易价差过大的资产

### 2. 数据频率选择

- **日频数据**：适合中长期均值回归
- **分钟级数据**：适合日内均值回归
- **Tick数据**：适合高频均值回归

### 3. 参数优化与过拟合

```python
def walk_forward_optimization(price_data, train_window=252, 
                            test_window=63):
    """滚动窗口优化参数"""
    results = []
    
    for start in range(0, len(price_data) - train_window - 
                      test_window, test_window):
        # 训练期
        train_data = price_data[start:start + train_window]
        
        # 测试期
        test_data = price_data[start + train_window:
                              start + train_window + test_window]
        
        # 在训练期优化参数
        best_params = optimize_parameters(train_data)
        
        # 在测试期评估性能
        performance = evaluate_strategy(test_data, best_params)
        
        results.append(performance)
    
    return results
```

## 总结

均值回归策略是一种基于统计学原理的经典量化策略，其核心假设是价格会围绕均值波动。

**关键要点**：

1. **统计学基础**：使用ADF检验验证平稳性，计算半衰期衡量回归速度
2. **常用指标**：布林带、RSI、Z-Score都是有效的均值回归指标
3. **配对交易**：通过交易相关性高的资产对实现市场中性
4. **风险控制**：严格的止损、最大持仓时间、市场环境过滤
5. **参数优化**：使用滚动窗口优化，避免过拟合

**适用场景**：
- 震荡市
- 高波动资产
- 具有均值回归特性的资产（如波动率指数VIX）

**不适用场景**：
- 强趋势市
- 结构性变化的资产
- 低频数据（均值回归效应较弱）

均值回归策略可以作为量化组合的重要组成部分，但需要注意风险控制和参数优化，避免在市场环境变化时遭受重大损失。

![均值回归示意图](/images/2026-06-13-mean-reversion-strategy/mean-reversion-diagram.jpg)

*均值回归示意图：价格围绕均值波动，偏离后会回归*

![布林带指标](/images/2026-06-13-mean-reversion-strategy/bollinger-bands.jpg)

*布林带指标：价格触及上下轨时产生交易信号*
