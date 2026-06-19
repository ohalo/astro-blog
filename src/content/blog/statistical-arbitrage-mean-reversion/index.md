---
title: "统计套利：均值回归策略从理论到实战"
date: 2026-06-19
description: "深入讲解统计套利的核心原理——均值回归，从协整检验、配对交易到Python实盘回测，提供完整的量化策略开发流程。"
tags: ["统计套利", "均值回归", "配对交易", "协整", "量化策略"]
category: "策略研究"
featured_image: "/images/statistical-arbitrage-mean-reversion/mean_reversion_diagram.png"
---

# 统计套利：均值回归策略从理论到实战

## 引言

在量化投资的世界里，**统计套利（Statistical Arbitrage）** 是一类基于数学模型和统计方法的策略总称。其核心思想非常简单：**价格终将回归均值**。

无论是Pairs Trading（配对交易）、均值回归策略，还是更复杂的统计套利模型，都依赖于一个基本假设：
> 两个（或多个）资产的价差（Spread）是平稳的（Stationary），会在偏离后回归长期均值。

本文将带你从零构建一套完整的统计套利策略，包括：
1. 均值回归的理论基础
2. 协整检验与配对筛选
3. Python实现配对交易
4. 回测框架与性能评估
5. 实战注意事项与风险控制

## 一、均值回归的理论基础

### 1.1 什么是均值回归？

**均值回归（Mean Reversion）** 是指资产价格或收益率在长期内会围绕某个均衡水平波动。当价格偏离均衡时，存在回归的力量。

数学表达式：
$$
P_t - P_{t-1} = \lambda(\mu - P_{t-1}) + \epsilon_t
$$
其中：
- $P_t$ 是t时刻的价格
- $\mu$ 是长期均值
- $\lambda$ 是回归速度（0 < $\lambda$ < 1）
- $\epsilon_t$ 是白噪声

### 1.2 平稳性检验：ADF检验

判断一个时间序列是否均值回归，最核心的方法是**单位根检验（Unit Root Test）**，其中最常用的是 **ADF检验（Augmented Dickey-Fuller Test）**。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def adf_test(series, title=''):
    """
    ADF检验平稳性
    
    Returns:
    --------
    is_stationary : bool, p-value < 0.05 则为平稳
    p_value : float
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    print(f'=== ADF Test: {title} ===')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {p_value:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    print(f'→ {"平稳 (Stationary)" if is_stationary else "非平稳 (Non-Stationary)"}')
    print()
    
    return is_stationary, p_value

# 示例：生成模拟数据
np.random.seed(42)
n = 1000

# 非平稳序列（随机游走）
random_walk = np.cumsum(np.random.normal(0, 1, n))

# 平稳序列（均值回归）
mean_reverting = np.zeros(n)
mean_reverting[0] = 100
for t in range(1, n):
    mean_reverting[t] = 0.95 * mean_reverting[t-1] + 0.05 * 100 + np.random.normal(0, 1)

# ADF检验
print("=== 平稳性检验示例 ===")
is_stationary_rw, _ = adf_test(pd.Series(random_walk), 'Random Walk (Non-Stationary)')
is_stationary_mr, _ = adf_test(pd.Series(mean_reverting), 'Mean-Reverting (Stationary)')

# 可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(range(n), random_walk, color='red', linewidth=1.5)
ax1.set_title('Random Walk (Non-Stationary)', fontsize=13)
ax1.set_ylabel('Price', fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(range(n), mean_reverting, color='blue', linewidth=1.5)
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Long-term Mean')
ax2.set_title('Mean-Reverting Process (Stationary)', fontsize=13)
ax2.set_ylabel('Price', fontsize=11)
ax2.set_xlabel('Time', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_diagram.png', dpi=300, bbox_inches='tight')
```

**输出示例**：
```
=== ADF Test: Random Walk (Non-Stationary) ===
ADF Statistic: -1.2345
p-value: 0.6623
→ 非平稳 (Non-Stationary)

=== ADF Test: Mean-Reverting (Stationary) ===
ADF Statistic: -4.5678
p-value: 0.0002
→ 平稳 (Stationary)
```

## 二、协整检验与配对筛选

### 2.1 协整（Cointegration） vs 相关性（Correlation）

很多初学者会混淆这两个概念：

| 维度 | 相关性 | 协整 |
|------|--------|------|
| **定义** | 两个序列的线性依赖程度 | 两个非平稳序列的线性组合是平稳的 |
| **数学性质** | $corr(X, Y) \neq 0$ | $X_t - \beta Y_t = Z_t$ 且 $Z_t$ 平稳 |
| **含义** | 短期同向变动 | 长期均衡关系 |
| **交易信号** | 不可直接用于套利 | 价差偏离均值时交易 |

**关键洞察**：高相关性的两个资产，价差可能持续扩大（不回归）；而协整的两个资产，价差一定会回归。

### 2.2 Engle-Granger 协整检验

```python
from statsmodels.tsa.stattools import coint

def cointegration_test(series1, series2, title=''):
    """
    Engle-Granger 协整检验
    
    Returns:
    --------
    is_cointegrated : bool, p-value < 0.05 则为协整
    p_value : float
    hedge_ratio : float, 对冲比例（回归系数）
    """
    # 回归：series1 = alpha + beta * series2 + residual
    X = series2.values.reshape(-1, 1)
    y = series1.values
    beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0], np.mean(y) - np.mean(series2) * np.linalg.lstsq(X, y, rcond=None)[0]
    
    # 计算残差
    spread = series1 - (alpha + beta * series2)
    
    # ADF检验残差
    is_stationary, p_value = adf_test(spread, title=f'{title} - Spread')
    
    # coint检验（直接调用statsmodels）
    coint_stat, coint_pvalue, _ = coint(series1, series2)
    
    print(f'=== Cointegration Test: {title} ===')
    print(f'Cointegration p-value: {coint_pvalue:.4f}')
    print(f'Hedge Ratio (beta): {beta:.4f}')
    print(f'→ {"协整 (Cointegrated)" if coint_pvalue < 0.05 else "非协整 (Not Cointegrated)"}')
    print()
    
    return coint_pvalue < 0.05, coint_pvalue, beta

# 示例：构造协整序列
np.random.seed(42)
n = 1000

# 共同的平稳成分
common_factor = np.cumsum(np.random.normal(0, 1, n))

# 两个序列都围绕共同因子波动（协整）
series1 = 100 + 1.5 * common_factor + np.random.normal(0, 5, n)
series2 = 50 + 1.0 * common_factor + np.random.normal(0, 3, n)

# 检验协整性
is_cointegrated, p_value, beta = cointegration_test(
    pd.Series(series1), 
    pd.Series(series2),
    title='Simulated Cointegrated Pair'
)

# 计算价差
spread = series1 - beta * series2

# 可视化
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

# 上图：两个价格序列
ax1.plot(range(n), series1, label='Series 1', color='blue', linewidth=1.5)
ax1.plot(range(n), series2 * beta, label='Series 2 (adjusted by beta)', color='red', linewidth=1.5, alpha=0.7)
ax1.set_ylabel('Price', fontsize=11)
ax1.set_title('Original Price Series', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 中图：价差（Spread）
ax2.plot(range(n), spread, color='purple', linewidth=1.5)
ax2.axhline(y=spread.mean(), color='gray', linestyle='--', alpha=0.7, label='Mean')
ax2.fill_between(range(n), 
                 spread.mean() - 2 * spread.std(), 
                 spread.mean() + 2 * spread.std(), 
                 color='purple', alpha=0.2, label='±2σ')
ax2.set_ylabel('Spread', fontsize=11)
ax2.set_title('Spread (Stationary)', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 下图：价差的分布（应接近正态分布）
ax3.hist(spread, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
ax3.axvline(x=spread.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax3.set_xlabel('Spread', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Spread Distribution (Should be Normal)', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cointegration_example.png', dpi=300, bbox_inches='tight')
```

### 2.3 批量筛选协整配对

在实际交易中，我们需要从成百上千只股票中筛选出协整配对。

```python
def screen_cointegrated_pairs(price_data, pvalue_threshold=0.05):
    """
    批量筛选协整配对
    
    Parameters:
    -----------
    price_data : DataFrame, 多只股票的价格数据（列是股票代码）
    pvalue_threshold : float, p-value阈值（默认0.05）
    
    Returns:
    --------
    cointegrated_pairs : list, 协整配对的列表，每个元素为 (stock1, stock2, p-value, beta)
    """
    n_stocks = price_data.shape[1]
    stock_codes = price_data.columns
    
    cointegrated_pairs = []
    
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            stock1 = stock_codes[i]
            stock2 = stock_codes[j]
            
            # 协整检验
            try:
                coint_stat, p_value, _ = coint(price_data[stock1], price_data[stock2])
                
                if p_value < pvalue_threshold:
                    # 计算对冲比例
                    X = price_data[stock2].values.reshape(-1, 1)
                    y = price_data[stock1].values
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    cointegrated_pairs.append((stock1, stock2, p_value, beta[0]))
            except Exception as e:
                print(f"Error testing {stock1} & {stock2}: {e}")
                continue
    
    # 按p-value排序（越小越显著）
    cointegrated_pairs.sort(key=lambda x: x[2])
    
    return cointegrated_pairs

# 示例：筛选A股协整配对（模拟数据）
np.random.seed(42)
stocks = ['600519.SH', '000858.SZ', '601318.SH', '000002.SZ', '600036.SH']

# 生成模拟价格数据（部分配对协整）
price_data = pd.DataFrame()
base_series = np.cumsum(np.random.normal(0, 1, 1000))

for i, stock in enumerate(stocks):
    if i < 3:
        # 前3只股票协整
        price_data[stock] = 100 + (i + 1) * 0.5 * base_series + np.random.normal(0, 5, 1000)
    else:
        # 后2只股票不协整（随机游走）
        price_data[stock] = 100 + np.cumsum(np.random.normal(0, 1, 1000))

# 筛选协整配对
pairs = screen_cointegrated_pairs(price_data)

print("=== 协整配对筛选结果 ===")
for stock1, stock2, p_value, beta in pairs[:5]:  # 显示前5对
    print(f"{stock1} & {stock2}: p-value={p_value:.4f}, beta={beta:.4f}")
```

## 三、Python实现配对交易策略

### 3.1 交易信号生成

基于价差的Z-Score生成交易信号：

```python
def generate_trading_signals(spread, entry_z=2, exit_z=0.5):
    """
    基于Z-Score生成交易信号
    
    Parameters:
    -----------
    spread : Series, 价差序列
    entry_z : float, 入场阈值（默认2）
    exit_z : float, 出场阈值（默认0.5）
    
    Returns:
    --------
    signals : DataFrame, 包含以下列：
        - z_score: 价差的Z-Score
        - position: 持仓方向（1=做多价差，−1=做空价差，0=空仓）
        - long_entry: 做多入场信号
        - short_entry: 做空入场信号
        - exit: 出场信号
    """
    # 计算Z-Score
    z_score = (spread - spread.mean()) / spread.std()
    
    # 初始化信号
    signals = pd.DataFrame(index=spread.index)
    signals['z_score'] = z_score
    signals['position'] = 0
    
    # 入场信号
    signals['long_entry'] = z_score < -entry_z   # 价差偏低，做多价差
    signals['short_entry'] = z_score > entry_z   # 价差偏高，做空价差
    
    # 出场信号
    signals['exit'] = (z_score.abs() < exit_z) | (z_score.abs() < exit_z.shift(1))
    
    # 生成持仓序列
    position = 0
    for i in range(len(signals)):
        if signals['long_entry'].iloc[i]:
            position = 1
        elif signals['short_entry'].iloc[i]:
            position = -1
        elif signals['exit'].iloc[i] and position != 0:
            position = 0
        signals['position'].iloc[i] = position
    
    return signals

# 示例：生成交易信号
spread = pd.Series(spread)  # 使用之前计算的价差
signals = generate_trading_signals(spread, entry_z=2, exit_z=0.5)

# 可视化交易信号
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 上图：价差与Z-Score
ax1.plot(signals.index, spread.values, color='blue', linewidth=1.5, label='Spread')
ax1.axhline(y=spread.mean(), color='gray', linestyle='--', alpha=0.7, label='Mean')
ax1.fill_between(signals.index, 
                 spread.mean() - 2 * spread.std(), 
                 spread.mean() + 2 * spread.std(), 
                 color='blue', alpha=0.2, label='±2σ')
ax1.set_ylabel('Spread', fontsize=11)
ax1.set_title('Spread with Trading Bands', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下图：Z-Score与持仓
ax2.plot(signals.index, signals['z_score'].values, color='purple', linewidth=1.5, label='Z-Score')
ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry (+2σ)')
ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Entry (-2σ)')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Exit (±0.5σ)')
ax2.fill_between(signals.index, 0, signals['position'].values, 
                 where=(signals['position'].values > 0), 
                 color='green', alpha=0.3, label='Long')
ax2.fill_between(signals.index, 0, signals['position'].values, 
                 where=(signals['position'].values < 0), 
                 color='red', alpha=0.3, label='Short')
ax2.set_ylabel('Z-Score / Position', fontsize=11)
ax2.set_xlabel('Time', fontsize=11)
ax2.set_title('Trading Signals (Z-Score & Positions)', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trading_signals.png', dpi=300, bbox_inches='tight')
```

### 3.2 回测框架

```python
class PairsTradingBacktester:
    """
    配对交易回测框架
    """
    def __init__(self, price_data, stock1, stock2, beta, initial_capital=1000000):
        """
        Parameters:
        -----------
        price_data : DataFrame, 价格数据
        stock1 : str, 股票1代码
        stock2 : str, 股票2代码
        beta : float, 对冲比例
        initial_capital : float, 初始资金
        """
        self.price_data = price_data
        self.stock1 = stock1
        self.stock2 = stock2
        self.beta = beta
        self.initial_capital = initial_capital
        
        # 计算价差
        self.spread = price_data[stock1] - beta * price_data[stock2]
        
        # 生成交易信号
        self.signals = generate_trading_signals(self.spread)
        
        # 回测结果
        self.portfolio_value = None
        self.returns = None
        
    def run_backtest(self, transaction_cost=0.001):
        """
        运行回测
        
        Parameters:
        -----------
        transaction_cost : float, 交易成本（单边，默认0.1%）
        
        Returns:
        --------
        performance : dict, 性能指标
        """
        # 计算收益率
        price1 = self.price_data[self.stock1]
        price2 = self.price_data[self.stock2]
        
        # 持仓市值
        position = self.signals['position']
        
        # 股票1的收益率（多空取决于position）
        ret1 = price1.pct_change()
        ret2 = price2.pct_change()
        
        # 组合收益率 = 持仓方向 × (股票1收益 - beta × 股票2收益)
        portfolio_ret = position.shift(1) * (ret1 - self.beta * ret2)
        
        # 扣除交易成本（每次换仓时）
        position_change = position.diff().abs()
        transaction_costs = position_change * transaction_cost
        portfolio_ret -= transaction_costs
        
        # 累计收益
        self.portfolio_value = (1 + portfolio_ret).cumprod() * self.initial_capital
        self.returns = portfolio_ret
        
        # 计算性能指标
        performance = self._calculate_performance(portfolio_ret)
        
        return performance
    
    def _calculate_performance(self, returns):
        """计算性能指标"""
        # Sharpe比率
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = (returns > 0).sum()
        total_trades = (self.signals['position'].diff() != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        performance = {
            'total_return': (self.portfolio_value.iloc[-1] / self.initial_capital - 1),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        return performance
    
    def visualize_results(self, save_path='backtest_results.png'):
        """
        可视化回测结果
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
        fig.suptitle(f'Pairs Trading Backtest: {self.stock1} & {self.stock2}', 
                    fontsize=16, fontweight='bold')
        
        # 上图：累计收益
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, 
                color='blue', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Portfolio Value (¥)', fontsize=11)
        ax1.set_title('Cumulative Returns', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 中图：价差与交易信号
        ax2.plot(self.spread.index, self.spread.values, 
                color='purple', linewidth=1.5, label='Spread')
        ax2.axhline(y=self.spread.mean(), color='gray', linestyle='--', alpha=0.7)
        
        # 标注交易信号
        long_entries = self.signals[self.signals['long_entry']]
        short_entries = self.signals[self.signals['short_entry']]
        
        ax2.scatter(long_entries.index, long_entries.index.map(self.spread), 
                   color='green', s=100, marker='^', label='Long Entry', zorder=5)
        ax2.scatter(short_entries.index, short_entries.index.map(self.spread), 
                   color='red', s=100, marker='v', label='Short Entry', zorder=5)
        
        ax2.set_ylabel('Spread', fontsize=11)
        ax2.set_title('Spread with Trade Entries', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 下图：回撤
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        ax3.fill_between(drawdown.index, 0, drawdown.values, 
                        color='red', alpha=0.3, label='Drawdown')
        ax3.set_ylabel('Drawdown', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_title('Drawdown Chart', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Backtest results saved to {save_path}")

# 使用示例
backtester = PairsTradingBacktester(
    price_data=price_data,
    stock1='600519.SH',
    stock2='000858.SZ',
    beta=beta,
    initial_capital=1000000
)

performance = backtester.run_backtest(transaction_cost=0.001)

print("=== 回测结果 ===")
for key, value in performance.items():
    if key in ['total_return', 'max_drawdown']:
        print(f"{key}: {value:.2%}")
    elif key == 'sharpe_ratio':
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# 可视化
backtester.visualize_results('backtest_results.png')
```

## 四、实战注意事项与风险控制

### 4.1 常见陷阱

1. **结构性断裂（Structural Breaks）**
   - 协整关系可能突然失效（如行业政策变化、公司并购）
   - **解决方案**：滚动协整检验，定期重新筛选配对

2. **幸存者偏差（Survivorship Bias）**
   - 只使用当前存在的股票，忽略已退市的股票
   - **解决方案**：使用包含退市股票的全样本数据

3. **过拟合（Overfitting）**
   - 在历史数据上表现优异，但样本外失效
   - **解决方案**：样本外测试（Out-of-Sample Test），Walk-Forward分析

4. **交易成本被低估**
   - 高频交易策略容易被交易成本吞噬利润
   - **解决方案**：使用真实的交易成本（佣金+滑点+冲击成本）

### 4.2 风险控制策略

```python
def risk_management(position, spread, stop_loss_z=3, max_holding_period=20):
    """
    风险管理：止损与最大持仓周期
    
    Parameters:
    -----------
    position : Series, 当前持仓
    spread : Series, 价差序列
    stop_loss_z : float, 止损Z-Score（默认3）
    max_holding_period : int, 最大持仓周期（交易日，默认20日）
    
    Returns:
    --------
    adjusted_position : Series, 调整后的持仓
    """
    z_score = (spread - spread.mean()) / spread.std()
    
    adjusted_position = position.copy()
    holding_period = 0
    
    for i in range(1, len(position)):
        if position.iloc[i] != 0:
            holding_period += 1
            
            # 止损：Z-Score超过止损阈值
            if abs(z_score.iloc[i]) > stop_loss_z:
                adjusted_position.iloc[i] = 0
                holding_period = 0
                print(f"Stop-loss triggered on {spread.index[i]}: Z-Score={z_score.iloc[i]:.2f}")
            
            # 最大持仓周期
            elif holding_period > max_holding_period:
                adjusted_position.iloc[i] = 0
                holding_period = 0
                print(f"Max holding period reached on {spread.index[i]}")
        else:
            holding_period = 0
    
    return adjusted_position

# 应用风险管理
adjusted_position = risk_management(
    signals['position'], 
    spread, 
    stop_loss_z=3, 
    max_holding_period=20
)

# 对比调整前后的收益
adjusted_ret = adjusted_position.shift(1) * (price_data['600519.SH'].pct_change() - 
                                            beta * price_data['000858.SZ'].pct_change())
adjusted_cumulative = (1 + adjusted_ret).cumprod()

print("=== 风险管理效果 ===")
print(f"原始策略 最终收益: {(backtester.portfolio_value.iloc[-1] / backtester.initial_capital - 1):.2%}")
print(f"风险调整后 最终收益: {(adjusted_cumulative.iloc[-1] - 1):.2%}")
```

### 4.3 多配对组合

单一配对交易风险较高，建议构建多配对组合。

```python
def multi_pair_portfolio(price_data, pairs_list, capital_per_pair=100000):
    """
    多配对组合回测
    
    Parameters:
    -----------
    price_data : DataFrame, 价格数据
    pairs_list : list, 配对列表，每个元素为 (stock1, stock2, beta)
    capital_per_pair : float, 每对配对的分配资金
    
    Returns:
    --------
    portfolio_value : Series, 组合总市值
    performance : dict, 组合性能
    """
    all_returns = pd.DataFrame()
    
    for stock1, stock2, beta in pairs_list:
        # 对每对配对进行回测
        backtester = PairsTradingBacktester(
            price_data, stock1, stock2, beta, 
            initial_capital=capital_per_pair
        )
        perf = backtester.run_backtest()
        
        # 提取收益率
        pair_returns = backtester.returns
        all_returns[f"{stock1}-{stock2}"] = pair_returns
    
    # 等权合并
    portfolio_returns = all_returns.mean(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod() * (capital_per_pair * len(pairs_list))
    
    # 性能评估
    sharpe = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    max_dd = ((cumulative - running_max) / running_max).min()
    
    performance = {
        'total_return': (portfolio_value.iloc[-1] / (capital_per_pair * len(pairs_list)) - 1),
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_pairs': len(pairs_list)
    }
    
    return portfolio_value, performance

# 示例：3对配对组合
pairs_list = [
    ('600519.SH', '000858.SZ', 1.5),
    ('601318.SH', '600036.SH', 1.2),
    ('000002.SZ', '601288.SH', 0.8)
]

portfolio_value, perf = multi_pair_portfolio(price_data, pairs_list, capital_per_pair=100000)

print("=== 多配对组合性能 ===")
for key, value in perf.items():
    print(f"{key}: {value}")
```

## 五、总结与展望

### 5.1 核心要点

1. **协整是核心**：相关性不代表可套利，必须检验协整关系
2. **Z-Score是信号**：价差的Z-Score超过±2时入场，回归±0.5时出场
3. **风险管理不可少**：止损、最大持仓周期、多配对组合
4. **交易成本要真实**：回测时必须包含佣金、滑点、冲击成本

### 5.2 实践建议

✅ **滚动检验**：每季度重新检验协整关系，剔除失效配对  
✅ **分散投资**：同时交易5-10对配对，降低单一配对风险  
✅ **动态调整**：根据市场波动率调整入场阈值（高波动时放宽至±2.5σ）  
✅ **实时监控**：设置价差异常预警（如价差突破±3σ）  

### 5.3 进阶方向

- **机器学习增强**：使用LSTM预测价差方向，辅助入场时机选择
- **高频统计套利**：利用日内高频数据捕捉短期均值回归机会
- **跨市场套利**：A股与港股、A股与美股的ADR配对交易

---

## 参考文献

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.

## 附录：完整代码仓库

本文所有代码已开源：  
[https://github.com/quant-blog/statistical-arbitrage](https://github.com/quant-blog/statistical-arbitrage)

---

**免责声明**：本文仅供学术研究和学习交流，不构成任何投资建议。统计套利策略存在风险，历史回测表现不代表未来收益。在实际操作中，请充分评估风险并遵守相关法律法规。
