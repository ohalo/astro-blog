---
title: "统计套利：均值回归策略"
description: "深入解析统计套利的核心原理——均值回归策略，从理论基础到Python实战，涵盖配对交易、协整检验、Z-score建模等关键技术"
publishDate: 2026-06-15
tags: ["统计套利", "均值回归", "配对交易", "协整检验", "量化策略"]
category: "量化交易"
cover: "/images/statistical-arbitrage-mean-reversion/cover.png"
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是量化投资领域的重要策略之一，其核心思想是利用资产价格之间的统计关系，通过构建多空组合来获取稳定的阿尔法收益。在众多统计套利方法中，**均值回归策略**以其坚实的理论基础和可验证的统计特性，成为对冲基金和量化团队的常用武器。

本文将深入探讨均值回归策略的理论基础、实现方法和实战技巧，并提供完整的Python代码示例。

## 一、均值回归的理论基础

### 1.1 有效市场假说与均值回归

有效市场假说（EMH）认为资产价格反映了所有可用信息，但在现实市场中，**短期价格往往偏离其内在价值**。均值回归理论认为，资产价格虽然短期可能偏离均衡水平，但长期来看会回归到均值（或趋势线）。

这意味着：
- **偏离即机会**：当价格显著偏离历史均值时，存在套利机会
- **可逆性**：偏离不会永久持续，价格终将回归
- **统计可量化**：可以用统计方法识别偏离程度

### 1.2 平稳性与协整

均值回归策略的前提是**价格序列具有平稳性**或**存在协整关系**。

**平稳性检验（ADF检验）**：

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller检验
    原假设：序列有单位根（非平稳）
    备择假设：序列平稳
    """
    print(f'ADF Test: {title}')
    result = adfuller(series, autolag='AIC')
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    
    if result[1] < 0.05:
        print("=> 序列平稳（p < 0.05）")
        return True
    else:
        print("=> 序列非平稳（p >= 0.05）")
        return False

# 示例：检验股价序列
# adf_test(stock_data['Close'], 'Stock Price')
```

**协整检验（Engle-Granger两步法）**：

```python
from statsmodels.tsa.stattools import coint
import yfinance as yf

def cointegration_test(series1, series2, title=''):
    """
    协整检验
    原假设：两个序列没有协整关系
    备择假设：存在协整关系
    """
    print(f'\nCointegration Test: {title}')
    score, pvalue, _ = coint(series1, series2)
    
    print(f'Cointegration Score: {score:.4f}')
    print(f'p-value: {pvalue:.4f}')
    
    if pvalue < 0.05:
        print("=> 存在协整关系（p < 0.05）")
        return True
    else:
        print("=> 不存在协整关系（p >= 0.05）")
        return False

# 示例：检验两只股票是否协整
# stock1 = yf.download('AAPL', start='2023-01-01')['Adj Close']
# stock2 = yf.download('MSFT', start='2023-01-01')['Adj Close']
# cointegration_test(stock1, stock2, 'AAPL vs MSFT')
```

## 二、配对交易策略

### 2.1 策略原理

配对交易（Pairs Trading）是最经典的均值回归策略：

1. **选股**：找到历史价格走势相似的两只股票（协整关系）
2. **建仓**：当价格偏离时，做多被低估的标的，做空被高估的标的
3. **平仓**：当价格回归时，平仓获利

### 2.2 价差建模

**方法1：简单价差**

```python
def simple_spread(series1, series2):
    """简单价差 = 股票1价格 - 股票2价格"""
    return series1 - series2
```

**方法2：对冲比例调整后的价差**

```python
import statsmodels.api as sm

def hedged_spread(series1, series2):
    """
    用线性回归计算对冲比例
    series1 = alpha + beta * series2 + error
    对冲价差 = series1 - beta * series2
    """
    X = sm.add_constant(series2)
    model = sm.OLS(series1, X).fit()
    beta = model.params[1]
    spread = series1 - beta * series2
    return spread, beta, model

# 示例
# spread, beta, model = hedged_spread(stock1, stock2)
# print(f'对冲比例 (beta): {beta:.4f}')
```

**方法3：Log价格差**

```python
def log_spread(series1, series2):
    """对数价格差 = ln(股票1) - ln(股票2)"""
    return np.log(series1) - np.log(series2)
```

### 2.3 Z-Score交易信号

```python
def calculate_zscore(spread, window=20):
    """
    计算价差的Z-Score
    Z-Score = (当前值 - 均值) / 标准差
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore

def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    生成交易信号
    Z-Score > entry_threshold: 做空价差（卖出股票1，买入股票2）
    Z-Score < -entry_threshold: 做多价差（买入股票1，卖出股票2）
    |Z-Score| < exit_threshold: 平仓
    """
    signals = pd.DataFrame(index=zscore.index)
    signals['zscore'] = zscore
    signals['position'] = 0
    
    # 简单信号逻辑
    signals['position'] = np.where(zscore > entry_threshold, -1, 
                            np.where(zscore < -entry_threshold, 1, 0))
    
    # 平仓逻辑（可选：只在Z-Score回归时平仓）
    # signals['position'] = signals['position'].replace(0, np.nan)
    # signals['position'] = signals['position'].fillna(method='ffill')
    # signals['position'] = np.where(abs(zscore) < exit_threshold, 0, 
    #                               signals['position'])
    
    return signals

# 完整示例
# spread, beta, _ = hedged_spread(stock1, stock2)
# zscore = calculate_zscore(spread)
# signals = generate_signals(zscore)
```

## 三、策略回测

### 3.1 回测框架

```python
class PairsTradingBacktest:
    def __init__(self, stock1_data, stock2_data, signals, 
                 initial_capital=100000, transaction_cost=0.001):
        """
        配对交易回测
        
        Parameters:
        -----------
        stock1_data : DataFrame, 股票1的价格数据
        stock2_data : DataFrame, 股票2的价格数据
        signals : DataFrame, 包含'position'列的交易信号
        initial_capital : float, 初始资金
        transaction_cost : float, 交易成本（单边）
        """
        self.stock1 = stock1_data
        self.stock2 = stock2_data
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.positions = pd.DataFrame(index=signals.index)
        self.portfolio = pd.DataFrame(index=signals.index)
        
    def backtest(self):
        """执行回测"""
        # 复制价格数据
        self.positions['stock1_price'] = self.stock1['Adj Close']
        self.positions['stock2_price'] = self.stock2['Adj Close']
        self.positions['position'] = self.signals['position']
        
        # 计算持仓数量（等金额配置）
        self.positions['stock1_shares'] = 0
        self.positions['stock2_shares'] = 0
        self.positions['cash'] = self.initial_capital
        self.positions['portfolio_value'] = self.initial_capital
        
        current_position = 0
        trade_count = 0
        
        for i in range(1, len(self.positions)):
            date = self.positions.index[i]
            prev_date = self.positions.index[i-1]
            
            # 复制前一日的持仓和现金
            self.positions.loc[date, 'stock1_shares'] = \
                self.positions.loc[prev_date, 'stock1_shares']
            self.positions.loc[date, 'stock2_shares'] = \
                self.positions.loc[prev_date, 'stock2_shares']
            self.positions.loc[date, 'cash'] = \
                self.positions.loc[prev_date, 'cash']
            
            # 检测仓位变化
            new_position = self.positions.loc[date, 'position']
            
            if new_position != current_position:
                # 平仓旧仓位
                if current_position != 0:
                    # 计算平仓价值
                    close_value = (
                        self.positions.loc[date, 'stock1_shares'] * 
                        self.positions.loc[date, 'stock1_price'] +
                        self.positions.loc[date, 'stock2_shares'] * 
                        self.positions.loc[date, 'stock2_price']
                    )
                    # 扣除交易成本
                    close_value -= abs(close_value) * self.transaction_cost
                    self.positions.loc[date, 'cash'] += close_value
                    self.positions.loc[date, 'stock1_shares'] = 0
                    self.positions.loc[date, 'stock2_shares'] = 0
                    trade_count += 1
                
                # 开仓新仓位
                if new_position != 0:
                    # 计算可投资金额（50%资金投入每只股票）
                    invest_capital = self.positions.loc[date, 'cash'] * 0.5
                    
                    # 买入/卖出股票1
                    stock1_shares = invest_capital / self.positions.loc[date, 'stock1_price']
                    self.positions.loc[date, 'stock1_shares'] = stock1_shares * new_position
                    self.positions.loc[date, 'cash'] -= (
                        stock1_shares * self.positions.loc[date, 'stock1_price'] * 
                        new_position
                    )
                    
                    # 买入/卖出股票2
                    stock2_shares = invest_capital / self.positions.loc[date, 'stock2_price']
                    self.positions.loc[date, 'stock2_shares'] = -stock2_shares * new_position
                    self.positions.loc[date, 'cash'] -= (
                        -stock2_shares * self.positions.loc[date, 'stock2_price'] * 
                        new_position
                    )
                    
                    # 扣除交易成本
                    trade_value = abs(stock1_shares * self.positions.loc[date, 'stock1_price']) + \
                                 abs(stock2_shares * self.positions.loc[date, 'stock2_price'])
                    self.positions.loc[date, 'cash'] -= trade_value * self.transaction_cost
                
                current_position = new_position
            
            # 计算当日组合价值
            portfolio_value = (
                self.positions.loc[date, 'cash'] +
                self.positions.loc[date, 'stock1_shares'] * 
                self.positions.loc[date, 'stock1_price'] +
                self.positions.loc[date, 'stock2_shares'] * 
                self.positions.loc[date, 'stock2_price']
            )
            self.positions.loc[date, 'portfolio_value'] = portfolio_value
        
        return self.positions, trade_count
    
    def calculate_metrics(self):
        """计算绩效指标"""
        portfolio_values = self.positions['portfolio_value']
        
        # 收益率
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        
        # 日收益率
        daily_returns = portfolio_values.pct_change().dropna()
        
        # 年化收益率
        trading_days = 252
        annual_return = (1 + total_return/100) ** (trading_days / len(daily_returns)) - 1
        annual_return *= 100
        
        # 夏普比率
        risk_free_rate = 0.02 / trading_days  # 假设无风险利率2%
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / daily_returns.std()
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
        
        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Annual Return (%)': round(annual_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Number of Trades': self.trade_count
        }
        
        return metrics

# 使用示例
# backtest = PairsTradingBacktest(stock1_data, stock2_data, signals)
# results, trade_count = backtest.backtest()
# metrics = backtest.calculate_metrics()
# print(metrics)
```

### 3.2 可视化分析

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pairs_trading_results(positions, stock1_name='Stock1', stock2_name='Stock2'):
    """可视化配对交易结果"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 图1：价差和Z-Score
    ax1 = axes[0]
    ax1.plot(positions.index, positions.get('spread', pd.Series(index=positions.index)), 
             label='Spread', color='blue', alpha=0.7)
    ax1.set_ylabel('Spread', color='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(positions.index, positions['zscore'], 
                  label='Z-Score', color='red', alpha=0.7)
    ax1_twin.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax1_twin.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax1_twin.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1_twin.set_ylabel('Z-Score', color='red')
    ax1_twin.legend(loc='upper right')
    
    # 图2：两只股票的归一化价格
    ax2 = axes[1]
    stock1_norm = positions['stock1_price'] / positions['stock1_price'].iloc[0]
    stock2_norm = positions['stock2_price'] / positions['stock2_price'].iloc[0]
    ax2.plot(positions.index, stock1_norm, label=stock1_name, color='blue')
    ax2.plot(positions.index, stock2_norm, label=stock2_name, color='orange')
    ax2.set_ylabel('Normalized Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3：组合价值
    ax3 = axes[2]
    ax3.plot(positions.index, positions['portfolio_value'], 
             label='Portfolio Value', color='green', linewidth=2)
    ax3.axhline(y=positions['portfolio_value'].iloc[0], 
                color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 使用示例
# plot_pairs_trading_results(results, 'AAPL', 'MSFT')
```

## 四、实战技巧与注意事项

### 4.1 标的筛选

**寻找协整对的技巧**：

1. **同行业股票**：业务相似，受相同宏观因素影响
2. **替代产品**：功能相似，长期价格趋势一致
3. **指数成分股**：流动性好，交易成本低

```python
def find_cointegrated_pairs(stocks_data, pvalue_threshold=0.05):
    """
    在股票列表中寻找协整对
    
    Parameters:
    -----------
    stocks_data : dict, {股票代码: 价格Series}
    pvalue_threshold : float, p-value阈值
    
    Returns:
    --------
    cointegrated_pairs : list, [(stock1, stock2, pvalue), ...]
    """
    cointegrated_pairs = []
    stocks = list(stocks_data.keys())
    
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            stock1 = stocks[i]
            stock2 = stocks[j]
            
            # 协整检验
            _, pvalue, _ = coint(stocks_data[stock1], stocks_data[stock2])
            
            if pvalue < pvalue_threshold:
                cointegrated_pairs.append((stock1, stock2, pvalue))
    
    # 按p-value排序（越小越好）
    cointegrated_pairs.sort(key=lambda x: x[2])
    
    return cointegrated_pairs

# 示例
# stocks = {'AAPL': aapl_data, 'MSFT': msft_data, 'GOOG': googl_data, ...}
# pairs = find_cointegrated_pairs(stocks)
# print(f"找到 {len(pairs)} 对协整股票")
# for pair in pairs[:5]:
#     print(f"{pair[0]} - {pair[1]}: p-value = {pair[2]:.4f}")
```

### 4.2 参数优化

**关键参数**：
- **Z-Score窗口**：常用20日、60日
- **入场阈值**：常用1.5、2.0、2.5
- **出场阈值**：常用0.5、0（回归均值）

```python
def optimize_parameters(stock1_data, stock2_data, 
                        zscore_windows=[20, 40, 60],
                        entry_thresholds=[1.5, 2.0, 2.5],
                        exit_thresholds=[0, 0.5, 1.0]):
    """
    参数优化（网格搜索）
    """
    best_sharpe = -np.inf
    best_params = None
    results = []
    
    for window in zscore_windows:
        for entry in entry_thresholds:
            for exit_t in exit_thresholds:
                # 计算价差和Z-Score
                spread, _, _ = hedged_spread(stock1_data['Adj Close'], 
                                             stock2_data['Adj Close'])
                zscore = calculate_zscore(spread, window)
                signals = generate_signals(zscore, entry, exit_t)
                
                # 回测
                backtest = PairsTradingBacktest(stock1_data, stock2_data, signals)
                positions, _ = backtest.backtest()
                metrics = backtest.calculate_metrics()
                
                sharpe = metrics['Sharpe Ratio']
                results.append({
                    'window': window,
                    'entry': entry,
                    'exit': exit_t,
                    'sharpe': sharpe,
                    'return': metrics['Total Return (%)'],
                    'drawdown': metrics['Max Drawdown (%)']
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (window, entry, exit_t)
    
    results_df = pd.DataFrame(results)
    print(f"最优参数: window={best_params[0]}, entry={best_params[1]}, exit={best_params[2]}")
    print(f"最优夏普比率: {best_sharpe:.4f}")
    
    return best_params, results_df

# 使用示例
# best_params, all_results = optimize_parameters(stock1_data, stock2_data)
```

### 4.3 风险管理

**关键风险点**：

1. **协整关系破裂**：历史关系不代表未来
   - **解决**：定期重新检验协整关系
   - **止损**：当价差持续扩大时强制平仓

2. **交易成本**：频繁交易会侵蚀利润
   - **解决**：提高入场阈值，减少交易次数
   - **优化**：选择低佣金券商

3. **市场环境变化**：趋势市中均值回归策略表现差
   - **解决**：结合市场环境判断（如用ADF检验判断平稳性）
   - **择时**：只在价差波动率较低时交易

```python
def dynamic_position_sizing(portfolio_value, volatility, max_risk_per_trade=0.02):
    """
    动态仓位管理：根据波动率调整仓位
    
    Parameters:
    -----------
    portfolio_value : float, 当前组合价值
    volatility : float, 价差波动率（如20日滚动标准差）
    max_risk_per_trade : float, 单笔交易最大风险敞口
    """
    # 风险预算
    risk_budget = portfolio_value * max_risk_per_trade
    
    # 根据波动率调整仓位（波动率越高，仓位越小）
    volatility_target = 0.02  # 目标波动率
    position_scaler = volatility_target / volatility if volatility > 0 else 1.0
    position_scaler = min(position_scaler, 2.0)  # 上限2倍
    position_scaler = max(position_scaler, 0.5)  # 下限0.5倍
    
    position_size = risk_budget * position_scaler
    
    return position_size

def stop_loss_check(spread, entry_spread, max_loss_threshold=3.0):
    """
    止损检查：当价差偏离超过阈值时止损
    
    Parameters:
    -----------
    spread : float, 当前价差
    entry_spread : float, 入场时的价差
    max_loss_threshold : float, 最大损失倍数（以标准差计）
    """
    spread_std = spread.rolling(window=20).std().iloc[-1]
    loss = abs(spread.iloc[-1] - entry_spread) / spread_std
    
    if loss > max_loss_threshold:
        print(f"触发止损！损失倍数: {loss:.2f}")
        return True
    return False
```

## 五、总结与展望

### 5.1 策略优缺点

**优点**：
- ✅ **市场中性**：多空对冲，不受大盘方向影响
- ✅ **统计基础**：有严谨的统计学理论支撑
- ✅ **低风险**：单次收益不高，但胜率较稳定

**缺点**：
- ❌ **配对难找**：真正协整的对子不多
- ❌ **关系破裂**：历史规律可能失效
- ❌ **低频交易**：交易机会有限，资金利用率低

### 5.2 进阶方向

1. **多因子配对**：不只依赖价格，加入基本面、动量等因子
2. **机器学习增强**：用随机森林、LSTM等预测价差回归时间
3. **高频统计套利**：利用分钟级、秒级数据捕捉短期偏离
4. **跨市场套利**：股票-ETF、股票-期货、跨交易所套利

### 5.3 实战建议

1. **严格回测**：在多个市场周期检验策略稳定性
2. **小额试错**：先用小资金实盘验证，再逐步加仓
3. **持续监控**：定期检查协整关系是否依然有效
4. **组合应用**：不要只依赖一对股票，构建多对组合分散风险

---

**免责声明**：本文仅供学术交流，不构成投资建议。统计套利虽然风险相对较低，但仍然存在亏损可能。实盘前请充分回测和模拟交易。

## 参考文献

1. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.
4. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.

---

**相关文章**：
- [配对交易与协整分析：市场中性策略的理论与实战](/blog/pairs-trading-cointegration/)
- [Python量化回测从零搭建：Backtrader实战与避坑指南](/blog/python-backtest-framework-build/)
- [因子衰减效应与因子择时：量化投资中的时间维度](/blog/factor-decay-timing/)
