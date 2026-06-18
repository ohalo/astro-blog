---
title: "配对交易与协整分析"
description: "深入探讨配对交易策略的核心原理，学习协整检验、对冲比率计算、交易信号构建的完整流程。包含美股AAPL-MSFT配对交易的Python实战代码。"
pubDate: 2026-06-18
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "Python"]
category: "量化策略"
featured: false
---

# 配对交易与协整分析

**配对交易**（Pairs Trading）是最经典的市场中性策略之一，通过对两只具有长期均衡关系的股票进行多空对冲，捕捉价格偏离后的均值回归机会。

本文将系统讲解配对交易的理论基础、协整检验方法、实战策略构建，并提供完整的Python实现框架。

## 配对交易的理论基础

### 什么是配对交易？

配对交易的核心思想：

1. **寻找配对**：找到两只价格具有长期协整关系的股票
2. **监控偏离**：当价格比（或价差）偏离历史均值时
3. **同时建仓**：做多低估股票、做空高估股票
4. **等待回归**：当价格比回归均值时平仓获利

### 为什么配对交易有效？

- **市场中性**：多空对冲，消除市场系统性风险
- **均值回归**：大多数股票对的价格偏离是暂时的
- **统计基础**：基于协整理论，具有严谨的数学基础

## 协整理论与检验方法

### 1. 平稳性检验（ADF检验）

在进行协整检验前，需先检验单个序列的平稳性：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import yfinance as yf
import matplotlib.pyplot as plt

def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller检验
    
    H0: 序列有单位根（非平稳）
    H1: 序列平稳
    
    Returns:
    --------
    is_stationary: bool, 是否平稳（5%显著性水平）
    p_value: float, p值
    """
    result = adfuller(series, autolag='AIC')
    
    print(f'\n{"="*60}')
    print(f'ADF Test: {title}')
    print(f'{"="*60}')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.4f}')
    
    is_stationary = result[1] < 0.05
    print(f'\nConclusion: {"Stationary" if is_stationary else "Non-Stationary"}')
    
    return is_stationary, result[1]

# 示例：检验股价序列的平稳性
def test_stock_stationarity(ticker, start='2020-01-01', end='2025-01-01'):
    """
    检验单只股票的平稳性
    """
    # 下载数据
    stock = yf.download(ticker, start=start, end=end, progress=False)
    price = stock['Adj Close'].dropna()
    
    # 检验原价格序列
    is_stationary_price, p_price = adf_test(price, f'{ticker} Price')
    
    # 检验对数收益率序列
    returns = np.log(price / price.shift(1)).dropna()
    is_stationary_ret, p_ret = adf_test(returns, f'{ticker} Log Returns')
    
    return {
        'price_stationary': is_stationary_price,
        'return_stationary': is_stationary_ret,
        'price_pvalue': p_price,
        'return_pvalue': p_ret
    }

# 检验AAPL的平稳性
result = test_stock_stationarity('AAPL')
```

**输出示例**：
```
============================================================
ADF Test: AAPL Price
============================================================
ADF Statistic: -1.2345
p-value: 0.6623
Critical Values:
  1%: -3.435
  5%: -2.863
  10%: -2.568

Conclusion: Non-Stationary

============================================================
ADF Test: AAPL Log Returns
============================================================
ADF Statistic: -15.6789
p-value: 0.0000
Critical Values:
  1%: -3.435
  5%: -2.863
  10%: -2.568

Conclusion: Stationary
```

### 2. 协整检验

协整关系：两个非平稳序列的线性组合是平稳的。

```python
def engle_granger_test(y, x, significance=0.05):
    """
    Engle-Granger两步法协整检验
    
    Parameters:
    -----------
    y: Series, 第一个序列
    x: Series, 第二个序列
    significance: float, 显著性水平
    
    Returns:
    --------
    is_cointegrated: bool, 是否存在协整关系
    p_value: float, p值
    hedge_ratio: float, 对冲比率（β）
    residuals: Series, 残差序列
    """
    # 第一步：OLS回归 y = α + βx + ε
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    beta = model.params.iloc[1]
    residuals = model.resid
    
    # 第二步：检验残差的平稳性（ADF检验）
    adf_stat, p_value, _, _, critical_values, _ = adfuller(
        residuals, autolag='AIC'
    )
    
    is_cointegrated = p_value < significance
    
    print(f'\n{"="*60}')
    print('Engle-Granger Cointegration Test')
    print(f'{"="*60}')
    print(f'Hedge Ratio (β): {beta:.4f}')
    print(f'ADF Statistic: {adf_stat:.4f}')
    print(f'p-value: {p_value:.4f}')
    print(f'5% Critical Value: {critical_values["5%"]:.4f}')
    print(f'\nConclusion: {"Cointegrated" if is_cointegrated else "Not Cointegrated"}')
    
    return is_cointegrated, p_value, beta, residuals

# 示例：检验AAPL和MSFT的协整关系
def test_pair_cointegration(ticker1, ticker2, start='2020-01-01', end='2025-01-01'):
    """
    检验一对股票的协整关系
    """
    # 下载数据
    stock1 = yf.download(ticker1, start=start, end=end, progress=False)
    stock2 = yf.download(ticker2, start=start, end=end, progress=False)
    
    price1 = stock1['Adj Close'].dropna()
    price2 = stock2['Adj Close'].dropna()
    
    # 对齐数据
    aligned = pd.concat([price1, price2], axis=1, join='inner')
    aligned.columns = [ticker1, ticker2]
    
    # 协整检验
    is_cointegrated, p_value, beta, residuals = engle_granger_test(
        aligned[ticker1], aligned[ticker2]
    )
    
    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 价格序列
    ax1 = axes[0]
    ax1.plot(aligned.index, aligned[ticker1], label=ticker1, linewidth=2)
    ax1.plot(aligned.index, aligned[ticker2] * beta, label=f'{ticker2} (β={beta:.2f})', 
             linewidth=2, linestyle='--')
    ax1.set_title(f'{ticker1} vs {ticker2} Price Series', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 价差（残差）
    ax2 = axes[1]
    ax2.plot(aligned.index, residuals, linewidth=1.5, color='blue')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=residuals.mean() + 2*residuals.std(), 
                color='g', linestyle=':', alpha=0.7, label='+2σ')
    ax2.axhline(y=residuals.mean() - 2*residuals.std(), 
                color='g', linestyle=':', alpha=0.7, label='-2σ')
    ax2.fill_between(aligned.index, 
                     residuals.mean() - 2*residuals.std(),
                     residuals.mean() + 2*residuals.std(),
                     alpha=0.2, color='green')
    ax2.set_title('Spread (Residuals)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差的ACF
    ax3 = axes[2]
    sm.graphics.tsa.plot_acf(residuals.dropna(), lags=40, ax=ax3)
    ax3.set_title('ACF of Residuals', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'public/images/pair-trading-cointegration/{ticker1}_{ticker2}_cointegration.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to public/images/pair-trading-cointegration/{ticker1}_{ticker2}_cointegration.png")
    
    return is_cointegrated, beta, residuals, aligned

# 执行检验
result = test_pair_cointegration('AAPL', 'MSFT')
```

### 3. Johansen协整检验（多变量扩展）

```python
def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多变量）
    
    Parameters:
    -----------
    data: DataFrame, 多变量时间序列
    det_order: int, 确定性项顺序
              0: 无常数项，无趋势
              1: 有常数项，无趋势
              2: 有常数项，有趋势
    k_ar_diff: int, 滞后阶数
    
    Returns:
    --------
    trace_stat: float, 迹统计量
    max_eig_stat: float, 最大特征值统计量
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    print(f'\n{"="*60}')
    print('Johansen Cointegration Test')
    print(f'{"="*60}')
    print(f'Trace Statistic: {result.lr1}')
    print(f'Max Eigenvalue Statistic: {result.lr2}')
    print(f'\nCritical Values (5%):')
    print(f'Trace: {result.cvt[:, 1]}')
    print(f'Max Eigenvalue: {result.cvm[:, 1]}')
    
    # 判断协整关系个数
    n_cointegration = sum(result.lr1 > result.cvt[:, 1])
    print(f'\nNumber of Cointegration Relations: {n_cointegration}')
    
    return result

# 多变量协整检验示例
def test_multiple_cointegration(tickers, start='2020-01-01', end='2025-01-01'):
    """
    检验多只股票的协整关系
    """
    # 下载数据
    prices = pd.DataFrame()
    for ticker in tickers:
        stock = yf.download(ticker, start=start, end=end, progress=False)
        prices[ticker] = stock['Adj Close']
    
    prices = prices.dropna()
    
    # Johansen检验
    result = johansen_test(prices[['AAPL', 'MSFT', 'GOOGL']])
    
    return result, prices

# 检验三只科技股
result, prices = test_multiple_cointegration(['AAPL', 'MSFT', 'GOOGL'])
```

## 配对交易策略构建

### 1. 计算交易信号

```python
class PairTradingStrategy:
    """
    配对交易策略类
    """
    def __init__(self, entry_z=2.0, exit_z=0.5, stop_loss_z=3.0):
        """
        Parameters:
        -----------
        entry_z: float, 入场z-score阈值
        exit_z: float, 出场z-score阈值
        stop_loss_z: float, 止损z-score阈值
        """
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        
    def calculate_spread(self, price1, price2, method='ols'):
        """
        计算价差（或比价）
        
        Methods:
        --------
        'ols': 基于OLS回归的价差
        'ratio': 价格比
        'log_ratio': 对数价格比
        """
        if method == 'ols':
            # OLS回归得到对冲比率
            X = sm.add_constant(price2)
            model = sm.OLS(price1, X).fit()
            beta = model.params.iloc[1]
            spread = price1 - beta * price2
            
        elif method == 'ratio':
            # 价格比
            spread = price1 / price2
            
        elif method == 'log_ratio':
            # 对数价格比
            spread = np.log(price1 / price2)
        
        return spread, beta if method == 'ols' else 1.0
    
    def calculate_z_score(self, spread, window=20):
        """
        计算价差的z-score
        """
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        z_score = (spread - mean) / std
        
        return z_score
    
    def generate_signals(self, z_score):
        """
        生成交易信号
        
        Returns:
        --------
        signals: DataFrame, 包含以下列：
          - position: 持仓方向（1: 多配对, -1: 空配对, 0: 无持仓）
          - long_entry: 做多信号
          - short_entry: 做空信号
          - exit: 平仓信号
        """
        signals = pd.DataFrame(index=z_score.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        
        # 入场信号
        signals['long_entry'] = (z_score < -self.entry_z)  # 价差偏低，做多配对
        signals['short_entry'] = (z_score > self.entry_z)   # 价差偏高，做空配对
        
        # 出场信号
        signals['exit'] = (np.abs(z_score) < self.exit_z)
        
        # 止损信号
        signals['stop_loss'] = (np.abs(z_score) > self.stop_loss_z)
        
        # 生成持仓序列
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i]['long_entry']:
                position = 1
            elif signals.iloc[i]['short_entry']:
                position = -1
            elif signals.iloc[i]['exit'] or signals.iloc[i]['stop_loss']:
                position = 0
            
            signals.iloc[i, signals.columns.get_loc('position')] = position
        
        return signals
    
    def backtest(self, price1, price2, signals, initial_capital=100000):
        """
        回测配对交易策略
        
        Returns:
        --------
        portfolio: DataFrame, 组合净值和收益
        performance: dict, 绩效指标
        """
        # 计算对冲比率
        spread, beta = self.calculate_spread(price1, price2, method='ols')
        
        # 初始化组合
        portfolio = pd.DataFrame(index=price1.index)
        portfolio['position'] = signals['position']
        portfolio['cash'] = initial_capital
        portfolio['stock1_shares'] = 0
        portfolio['stock2_shares'] = 0
        portfolio['portfolio_value'] = initial_capital
        
        # 回测循环
        for i in range(1, len(portfolio)):
            # 复制前一期的持仓
            portfolio.iloc[i, portfolio.columns.get_loc('stock1_shares')] = \
                portfolio.iloc[i-1]['stock1_shares']
            portfolio.iloc[i, portfolio.columns.get_loc('stock2_shares')] = \
                portfolio.iloc[i-1]['stock2_shares']
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] = \
                portfolio.iloc[i-1]['cash']
            
            # 交易信号
            if portfolio.iloc[i]['position'] != portfolio.iloc[i-1]['position']:
                # 平仓
                if portfolio.iloc[i-1]['position'] != 0:
                    portfolio.iloc[i, portfolio.columns.get_loc('cash')] += \
                        portfolio.iloc[i]['stock1_shares'] * price1.iloc[i] + \
                        portfolio.iloc[i]['stock2_shares'] * price2.iloc[i]
                    portfolio.iloc[i, portfolio.columns.get_loc('stock1_shares')] = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('stock2_shares')] = 0
                
                # 开仓
                if portfolio.iloc[i]['position'] == 1:  # 做多配对
                    # 买入stock1，卖出stock2
                    n_shares = int(portfolio.iloc[i]['cash'] / (price1.iloc[i] + beta * price2.iloc[i]))
                    portfolio.iloc[i, portfolio.columns.get_loc('stock1_shares')] = n_shares
                    portfolio.iloc[i, portfolio.columns.get_loc('stock2_shares')] = -int(n_shares * beta)
                    portfolio.iloc[i, portfolio.columns.get_loc('cash')] -= \
                        n_shares * price1.iloc[i] + int(n_shares * beta) * price2.iloc[i]
                    
                elif portfolio.iloc[i]['position'] == -1:  # 做空配对
                    # 卖出stock1，买入stock2
                    n_shares = int(portfolio.iloc[i]['cash'] / (price1.iloc[i] + beta * price2.iloc[i]))
                    portfolio.iloc[i, portfolio.columns.get_loc('stock1_shares')] = -n_shares
                    portfolio.iloc[i, portfolio.columns.get_loc('stock2_shares')] = int(n_shares * beta)
                    portfolio.iloc[i, portfolio.columns.get_loc('cash')] += \
                        n_shares * price1.iloc[i] + int(n_shares * beta) * price2.iloc[i]
            
            # 计算组合价值
            portfolio.iloc[i, portfolio.columns.get_loc('portfolio_value')] = \
                portfolio.iloc[i]['cash'] + \
                portfolio.iloc[i]['stock1_shares'] * price1.iloc[i] + \
                portfolio.iloc[i]['stock2_shares'] * price2.iloc[i]
        
        # 计算收益率
        portfolio['returns'] = portfolio['portfolio_value'].pct_change()
        
        # 计算绩效指标
        total_return = (portfolio['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        annual_return = (portfolio['portfolio_value'].iloc[-1] / initial_capital) ** \
                       (252 / len(portfolio)) - 1
        annual_vol = portfolio['returns'].std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cumulative = portfolio['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        performance = {
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': (portfolio['returns'] > 0).sum() / len(portfolio['returns'].dropna())
        }
        
        return portfolio, performance
```

### 2. 完整实战示例

```python
# 完整配对交易实战
def run_pair_trading_example(ticker1='AAPL', ticker2='MSFT', 
                            start='2020-01-01', end='2025-12-31'):
    """
    运行配对交易完整流程
    """
    # 1. 下载数据
    print(f"\n{'='*60}")
    print(f"Pair Trading Strategy: {ticker1} - {ticker2}")
    print(f"{'='*60}\n")
    
    stock1 = yf.download(ticker1, start=start, end=end, progress=False)
    stock2 = yf.download(ticker2, start=start, end=end, progress=False)
    
    price1 = stock1['Adj Close'].dropna()
    price2 = stock2['Adj Close'].dropna()
    
    # 对齐数据
    aligned = pd.concat([price1, price2], axis=1, join='inner')
    aligned.columns = [ticker1, ticker2]
    
    # 2. 协整检验
    is_cointegrated, p_value, beta, residuals = engle_granger_test(
        aligned[ticker1], aligned[ticker2]
    )
    
    if not is_cointegrated:
        print("\n⚠️ Warning: No cointegration relation found!")
        print("Continue anyway for demonstration purposes...\n")
    
    # 3. 计算价差和z-score
    strategy = PairTradingStrategy(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0)
    spread, beta = strategy.calculate_spread(aligned[ticker1], aligned[ticker2], method='ols')
    z_score = strategy.calculate_z_score(spread, window=20)
    
    # 4. 生成交易信号
    signals = strategy.generate_signals(z_score)
    
    # 5. 回测
    portfolio, performance = strategy.backtest(
        aligned[ticker1], aligned[ticker2], signals, initial_capital=100000
    )
    
    # 6. 打印绩效
    print(f"\n{'='*60}")
    print("Strategy Performance")
    print(f"{'='*60}")
    for key, value in performance.items():
        if key in ['total_return', 'annual_return', 'annual_volatility', 'max_drawdown']:
            print(f"{key}: {value:.2f}%")
        elif key == 'sharpe_ratio':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.2%}")
    
    # 7. 可视化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 价格序列
    ax1 = axes[0]
    ax1.plot(aligned.index, aligned[ticker1], label=ticker1, linewidth=2)
    ax1.plot(aligned.index, aligned[ticker2] * beta, 
             label=f'{ticker2} (β={beta:.2f})', linewidth=2, linestyle='--')
    ax1.set_title(f'{ticker1} vs {ticker2} Price Series', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-score
    ax2 = axes[1]
    ax2.plot(signals.index, signals['z_score'], linewidth=1.5, color='blue')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='Entry (+2σ)')
    ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.5, color='g', linestyle=':', alpha=0.7, label='Exit (±0.5σ)')
    ax2.axhline(y=-0.5, color='g', linestyle=':', alpha=0.7)
    ax2.fill_between(signals.index, -2, 2, alpha=0.1, color='gray')
    ax2.set_title('Z-Score of Spread', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 持仓
    ax3 = axes[2]
    ax3.plot(signals.index, signals['position'], linewidth=2, color='purple')
    ax3.set_title('Position', fontsize=14, fontweight='bold')
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Short', 'Neutral', 'Long'])
    ax3.grid(True, alpha=0.3)
    
    # 组合净值
    ax4 = axes[3]
    ax4.plot(portfolio.index, portfolio['portfolio_value'], 
             linewidth=2, color='green', label='Pair Trading')
    
    # 基准（等权买入持有）
    benchmark = (aligned[ticker1] + aligned[ticker2]) / 2
    benchmark = benchmark / benchmark.iloc[0] * 100000
    ax4.plot(benchmark.index, benchmark, linewidth=2, linestyle='--', 
             color='gray', label='Benchmark (50-50 B&H)')
    
    ax4.set_title('Portfolio Value', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/strategy_performance.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to public/images/pair-trading-cointegration/strategy_performance.png")
    
    return portfolio, performance, signals

# 执行策略
portfolio, performance, signals = run_pair_trading_example('AAPL', 'MSFT')
```

## 配对交易的关键风险

### 1. 协整关系破裂

```python
def monitor_cointegration_stability(price1, price2, window=252, significance=0.05):
    """
    监控协整关系的稳定性（滚动检验）
    
    Returns:
    --------
    stability: DataFrame, 滚动协整检验结果
    """
    dates = price1.index[window:]
    is_cointegrated_list = []
    p_values = []
    
    for i in range(window, len(price1)):
        window_price1 = price1.iloc[i-window:i]
        window_price2 = price2.iloc[i-window:i]
        
        # 滚动协整检验
        is_cointegrated, p_value, _, _ = engle_granger_test(
            window_price1, window_price2, significance
        )
        
        is_cointegrated_list.append(is_cointegrated)
        p_values.append(p_value)
    
    stability = pd.DataFrame({
        'is_cointegrated': is_cointegrated_list,
        'p_value': p_values
    }, index=dates)
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # p-value时序
    ax1 = axes[0]
    ax1.plot(stability.index, stability['p_value'], linewidth=1.5, color='blue')
    ax1.axhline(y=significance, color='r', linestyle='--', label=f'Significance ({significance})')
    ax1.set_title('Rolling Cointegration p-value', fontsize=14, fontweight='bold')
    ax1.set_ylabel('p-value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 协整关系是否存在
    ax2 = axes[1]
    ax2.plot(stability.index, stability['is_cointegrated'].astype(int), 
             linewidth=1.5, color='green')
    ax2.set_title('Cointegration Relation Existence', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Is Cointegrated (0/1)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/cointegration_stability.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to public/images/pair-trading-cointegration/cointegration_stability.png")
    
    # 统计协整关系破裂的频率
    n_breaks = (stability['is_cointegrated'].astype(int).diff() == -1).sum()
    print(f"\nCointegration breaks detected: {n_breaks}")
    print(f"Stability rate: {stability['is_cointegrated'].mean():.2%}")
    
    return stability

# 监控协整稳定性
stability = monitor_cointegration_stability(price1, price2, window=252)
```

### 2. 结构性断裂

```python
def detect_structural_breaks(spread, threshold=3):
    """
    检测价差的结构性断裂（CUSUM检验）
    """
    mean = spread.mean()
    std = spread.std()
    cusum = np.cumsum((spread - mean) / std)
    
    # 判断是否超出阈值
    breaks = np.abs(cusum) > threshold
    
    if breaks.any():
        print(f"\n⚠️ Structural breaks detected at:")
        print(spread.index[breaks])
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(spread.index, cusum, linewidth=2, color='blue', label='CUSUM')
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (±{threshold})')
    ax.axhline(y=-threshold, color='r', linestyle='--')
    ax.fill_between(spread.index, -threshold, threshold, alpha=0.1, color='gray')
    ax.set_title('CUSUM Test for Structural Breaks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('CUSUM Statistic')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/cusum_test.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to public/images/pair-trading-cointegration/cusum_test.png")
    
    return breaks
```

### 3. 交易成本影响

```python
def calculate_pair_trading_costs(portfolio, price1, price2, cost_rate=0.001):
    """
    计算配对交易的交易成本
    """
    # 计算换手率
    turnover = abs(portfolio['stock1_shares'].diff()) * price1 + \
               abs(portfolio['stock2_shares'].diff()) * price2
    turnover = turnover / portfolio['portfolio_value'].shift(1)
    
    # 计算交易成本
    costs = turnover * cost_rate
    
    print(f"\n{'='*60}")
    print("Transaction Cost Analysis")
    print(f"{'='*60}")
    print(f"Average Daily Turnover: {turnover.mean():.2%}")
    print(f"Average Daily Cost: {costs.mean():.4%}")
    print(f"Annualized Cost: {costs.sum() * 252 / len(costs):.2%}")
    
    # 扣除成本后的收益
    portfolio['returns_net'] = portfolio['returns'] - costs
    cumulative_gross = (1 + portfolio['returns']).cumprod()
    cumulative_net = (1 + portfolio['returns_net']).cumprod()
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cumulative_gross.index, cumulative_gross.values, 
            label='Gross Return', linewidth=2)
    ax.plot(cumulative_net.index, cumulative_net.values, 
            label='Net Return (After Costs)', linewidth=2, linestyle='--')
    ax.set_title('Gross vs Net Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/cost_impact.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to public/images/pair-trading-cointegration/cost_impact.png")
    
    return costs
```

## 配对交易的改进方向

### 1. 动态对冲比率

```python
def kalman_filter_hedge_ratio(price1, price2):
    """
    使用卡尔曼滤波动态估计对冲比率
    """
    # 观测矩阵
    H = price2.values.reshape(-1, 1)
    
    # 初始化
    x = np.array([[0]])  # 状态（对冲比率）
    P = np.array([[1]])  # 状态协方差
    Q = 0.001  # 过程噪声
    R = 1  # 观测噪声
    
    hedge_ratios = []
    
    for i in range(len(price1)):
        # 预测步骤
        x_pred = x
        P_pred = P + Q
        
        # 更新步骤
        y = price1.iloc[i]  # 观测值
        K = P_pred * H[i] / (H[i].T * P_pred * H[i] + R)  # 卡尔曼增益
        x = x_pred + K * (y - H[i].T * x_pred)
        P = (1 - K * H[i].T) * P_pred
        
        hedge_ratios.append(x[0, 0])
    
    hedge_ratios = pd.Series(hedge_ratios, index=price1.index)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hedge_ratios.index, hedge_ratios.values, 
            linewidth=2, color='blue', label='Dynamic Hedge Ratio (Kalman Filter)')
    ax.axhline(y=hedge_ratios.mean(), color='r', linestyle='--', 
               label=f'Static Mean ({hedge_ratios.mean():.2f})')
    ax.set_title('Dynamic Hedge Ratio Estimation', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Hedge Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/dynamic_hedge_ratio.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to public/images/pair-trading-cointegration/dynamic_hedge_ratio.png")
    
    return hedge_ratios
```

### 2. 机器学习优化信号

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def ml_pair_trading_signals(spread, lookback=20, n_estimators=100):
    """
    使用随机森林优化配对交易信号
    """
    # 构建特征
    features = pd.DataFrame(index=spread.index)
    features['z_score'] = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    features['momentum_5d'] = spread.diff(5)
    features['volatility_20d'] = spread.rolling(20).std()
    features['distance_from_mean'] = spread - spread.rolling(lookback).mean()
    
    # 构建标签（未来5日价差是否回归）
    labels = (spread.shift(-5) - spread) < 0  # 1 if spread decreases
    
    # 删除NaN
    features = features.dropna()
    labels = labels.loc[features.index]
    
    # 训练测试分割（时间序列交叉验证）
    split_idx = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    
    print(f"\n{'='*60}")
    print("Machine Learning Signal Optimization")
    print(f"{'='*60}")
    print(f"Training Accuracy: {model.score(X_train, y_train):.2%}")
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    return model, features, labels
```

## 总结

配对交易是一个理论严谨、实践丰富的量化策略。成功的关键在于：

1. **严格的协整检验**：确保配对具有长期均衡关系
2. **合理的信号构建**：z-score阈值需要根据市场状态动态调整
3. **风险控制**：协整关系破裂、结构性断裂是最大风险
4. **成本优化**：配对交易换手率高，需精细管理交易成本

**核心要点**：
- 协整关系是配对交易的基础，必须进行严格统计检验
- 动态对冲比率优于静态对冲比率
- 机器学习可以提升信号质量，但需警惕过度拟合
- 配对交易不是"印钞机"，需要持续监控和维护

配对交易作为市场中性策略的代表，在量化投资中占有重要地位。随着机器学习、高频数据的发展，配对交易策略也在不断进化。掌握其基本原理和实战技巧，是每位量化从业者的必修课。

---

**参考文献**：
1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.

**完整代码仓库**：[GitHub链接]（包含数据获取、协整检验、策略回测、风险管理的完整代码）
