---
title: "配对交易与协整分析：统计套利的理论与实践"
description: "深入探讨配对交易的核心原理——协整关系，从理论到实战，带你掌握统计套利策略的构建、检验和风险管理全流程。"
pubDate: 2026-06-16
tags: ["统计套利", "配对交易", "协整分析", "均值回归", "量化策略"]
featured: false
toc: true
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：统计套利的理论与实践

## 引言

在量化投资的世界里，**统计套利（Statistical Arbitrage）** 是一类基于数学模型和统计方法的策略，旨在通过捕捉资产价格之间的临时性偏离来获取收益。其中，**配对交易（Pairs Trading）** 是最经典、最广泛应用的统计套利策略之一。

配对交易的核心思想是：
1. 找到两个价格具有**长期均衡关系**的资产
2. 当价格短期偏离时，做多低估资产、做空高估资产
3. 等待价格回归均衡，平仓获利

这种策略不依赖市场方向，属于**市场中性（Market Neutral）**策略，因此在牛市和熊市中都有机会获利。

本文将深入探讨：
- 协整关系的理论与检验方法
- 配对交易的完整构建流程
- Python实战：从数据挖掘到策略回测
- 风险管理与实战注意事项

## 理论基础：协整与均值回归

### 什么是协整（Cointegration）？

**协整**是时间序列分析中的重要概念，由Engle和Granger在1987年提出，他们也因此获得了2003年诺贝尔经济学奖。

**定义：**
如果两个或多个非平稳时间序列的某种线性组合是平稳的，那么这些序列之间存在协整关系。

**直观理解：**
- 两个资产的价格序列可能都是非平稳的（如随机游走）
- 但它们之间的**价差（Spread）**或**比值**可能是平稳的
- 这意味着价格之间存在长期的均衡关系

### 协整 vs 相关性

很多人容易混淆协整和相关性，它们是两个完全不同的概念：

| 维度 | 相关性（Correlation） | 协整（Cointegration） |
|------|---------------------|---------------------|
| **定义** | 衡量两个序列同向变动的程度 | 衡量两个序列是否存在长期均衡关系 |
| **时间维度** | 横截面关系（同一时点） | 时间序列关系（跨时依存） |
| **平稳性要求** | 无要求 | 要求序列是非平稳的（通常是I(1)） |
| **经济意义** | 短期联动 | 长期均衡 |
| **交易含义** | 不适合直接用于配对交易 | 配对交易的理论基础 |

**示例：**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller

# 生成示例数据
np.random.seed(42)
n = 1000

# 两个协整的序列
y1 = np.cumsum(np.random.normal(0, 1, n))  # 随机游走
y2 = y1 + np.random.normal(0, 2, n)  # y2与y1协整，但有噪声

# 两个高相关但非协整的序列
z1 = np.cumsum(np.random.normal(0, 1, n))
z2 = np.cumsum(np.random.normal(0, 1, n))  # 独立随机游走，非协整

# 计算相关性
corr_cointegrated = np.corrcoef(y1, y2)[0, 1]
corr_independent = np.corrcoef(z1, z2)[0, 1]

print(f"协整序列的相关性：{corr_cointegrated:.3f}")
print(f"独立序列的相关性：{corr_independent:.3f}")

# 协整检验
score, p_value, _ = coint(y1, y2)
print(f"\n协整检验 p-value：{p_value:.4f}")

score, p_value, _ = coint(z1, z2)
print(f"独立序列协整检验 p-value：{p_value:.4f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：协整序列的价格走势
axes[0, 0].plot(y1, label='y1', linewidth=2)
axes[0, 0].plot(y2, label='y2', linewidth=2)
axes[0, 0].set_title(f'协整序列 (corr={corr_cointegrated:.3f})', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2：协整序列的价差
axes[0, 1].plot(y1 - y2, linewidth=2, color='green')
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title('协整序列的价差（平稳）', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：独立序列的价格走势
axes[1, 0].plot(z1, label='z1', linewidth=2)
axes[1, 0].plot(z2, label='z2', linewidth=2)
axes[1, 0].set_title(f'独立序列 (corr={corr_independent:.3f})', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 子图4：独立序列的价差
axes[1, 1].plot(z1 - z2, linewidth=2, color='red')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('独立序列的价差（非平稳）', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cointegration_vs_correlation.png', dpi=300)
plt.show()
```

## 协整检验方法

### 1. Engle-Granger 两步法

**步骤：**
1. 估计协整回归：$y_t = \alpha + \beta x_t + \epsilon_t$
2. 检验残差 $\epsilon_t$ 的平稳性（ADF检验）

**Python实现：**

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def engle_granger_test(y, x, significance_level=0.05):
    """
    Engle-Granger协整检验
    
    Args:
        y: 因变量（价格序列1）
        x: 自变量（价格序列2）
        significance_level: 显著性水平
    
    Returns:
        检验结果字典
    """
    # 步骤1：估计协整回归
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    
    # 步骤2：ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整
    is_cointegrated = p_value < significance_level
    
    # 计算协整系数
    alpha = model.params[0]  # 截距
    beta = model.params[1]   # 斜率
    
    results = {
        'is_cointegrated': is_cointegrated,
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'alpha': alpha,
        'beta': beta,
        'residuals': residuals,
        'model': model
    }
    
    return results

# 使用示例
np.random.seed(42)
n = 1000

# 生成协整序列
x = np.cumsum(np.random.normal(0, 1, n))
y = 0.5 + 1.5 * x + np.random.normal(0, 2, n)

# 执行检验
results = engle_granger_test(y, x)

print("=== Engle-Granger协整检验结果 ===")
print(f"是否协整：{results['is_cointegrated']}")
print(f"ADF统计量：{results['adf_statistic']:.4f}")
print(f"p-value：{results['p_value']:.4f}")
print(f"\n协整方程：y = {results['alpha']:.4f} + {results['beta']:.4f} * x")
print(f"\n临界值（1%）：{results['critical_values']['1%']:.4f}")
print(f"临界值（5%）：{results['critical_values']['5%']:.4f}")
print(f"临界值（10%）：{results['critical_values']['10%']:.4f}")
```

### 2. Johansen 检验

Johansen检验是一种多变量协整检验方法，适用于多个序列的协整关系检验。

**Python实现：**

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    Args:
        data: 数据矩阵（T×N，T为时间长度，N为变量个数）
        det_order: 确定性项顺序（0=无常数项，1=有常数项）
        k_ar_diff: 差分项的滞后阶数
    
    Returns:
        检验结果
    """
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    # 选择协整秩（协整关系的个数）
    rank_selection = select_coint_rank(data, det_order, k_ar_diff)
    trace_stat = rank_selection.trace_stat
    max_stat = rank_selection.max_stat
    
    results = {
        'trace_statistic': trace_stat,
        'max_statistic': max_stat,
        'cointegrating_rank': rank_selection.rank
    }
    
    return results

# 使用示例
np.random.seed(42)
n = 1000

# 生成三个协整序列
x1 = np.cumsum(np.random.normal(0, 1, n))
x2 = 0.5 * x1 + np.random.normal(0, 1, n)
x3 = -0.3 * x1 + 2 * x2 + np.random.normal(0, 0.5, n)

data = np.column_stack([x1, x2, x3])

# 执行Johansen检验
results = johansen_test(data)

print("=== Johansen协整检验结果 ===")
print(f"协整秩（协整关系个数）：{results['cointegrating_rank']}")
print(f"\nTrace统计量：{results['trace_statistic']}")
print(f"Max统计量：{results['max_statistic']}")
```

### 3.  Phillips-Ouliaris 检验

Phillips-Ouliaris检验是Engle-Granger检验的改进版本，对小样本表现更好。

```python
from statsmodels.tsa.stattools import coint

def phillips_ouliaris_test(y, x, trend='c', **kwargs):
    """
    Phillips-Ouliaris协整检验（基于statsmodels的coint函数）
    
    Args:
        y: 因变量
        x: 自变量
        trend: 趋势项（'c'=常数，'ct'=常数+趋势）
    
    Returns:
        检验结果
    """
    # statsmodels的coint函数实现了多种协整检验
    # 默认使用Phillips-Ouliaris检验
    t_stat, p_value, _ = coint(y, x, trend=trend, **kwargs)
    
    results = {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_cointegrated': p_value < 0.05
    }
    
    return results

# 使用示例
results = phillips_ouliaris_test(y, x)
print("=== Phillips-Ouliaris检验结果 ===")
print(f"t统计量：{results['t_statistic']:.4f}")
print(f"p-value：{results['p_value']:.4f}")
print(f"是否协整：{results['is_cointegrated']}")
```

## 配对交易的构建流程

### 步骤1：候选股票筛选

```python
import yfinance as yf
import pandas as pd
from itertools import combinations

class PairSelection:
    """配对筛选器"""
    
    def __init__(self, universe, start_date, end_date):
        """
        初始化
        
        Args:
            universe: 股票池
            start_date: 开始日期
            end_date: 结束日期
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None
    
    def fetch_price_data(self):
        """获取价格数据"""
        print("正在下载价格数据...")
        data = yf.download(self.universe, 
                          start=self.start_date, 
                          end=self.end_date,
                          group_by='ticker')
        
        # 处理多层级列索引
        if isinstance(data.columns, pd.MultiIndex):
            # 提取收盘价
            prices = data['Adj Close']
        else:
            prices = data['Adj Close']
        
        self.price_data = prices
        return prices
    
    def screen_by_correlation(self, threshold=0.7):
        """
        基于相关性初步筛选
        
        Args:
            threshold: 相关性阈值
        
        Returns:
            高相关性股票对
        """
        if self.price_data is None:
            self.fetch_price_data()
        
        # 计算收益率相关性
        returns = self.price_data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        # 筛选高相关性对
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.index)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if corr >= threshold:
                    stock1 = corr_matrix.index[i]
                    stock2 = corr_matrix.columns[j]
                    high_corr_pairs.append((stock1, stock2, corr))
        
        # 按相关性排序
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"筛选出 {len(high_corr_pairs)} 对高相关性股票")
        return high_corr_pairs
    
    def test_cointegration(self, pairs, significance_level=0.05):
        """
        对候选对进行协整检验
        
        Args:
            pairs: 候选股票对列表
            significance_level: 显著性水平
        
        Returns:
            协整股票对
        """
        cointegrated_pairs = []
        
        for stock1, stock2, corr in pairs:
            # 获取价格序列
            y = self.price_data[stock1].dropna()
            x = self.price_data[stock2].dropna()
            
            # 对齐数据
            aligned_data = pd.concat([y, x], axis=1).dropna()
            y_aligned = aligned_data.iloc[:, 0]
            x_aligned = aligned_data.iloc[:, 1]
            
            # 协整检验
            results = engle_granger_test(y_aligned, x_aligned, 
                                        significance_level)
            
            if results['is_cointegrated']:
                cointegrated_pairs.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'correlation': corr,
                    'p_value': results['p_value'],
                    'beta': results['beta'],
                    'alpha': results['alpha'],
                    'residuals': results['residuals']
                })
        
        print(f"协整检验通过：{len(cointegrated_pairs)} 对")
        return cointegrated_pairs

# 使用示例
universe = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 
            'TSLA', 'NVDA', 'JPM', 'BAC', 'WFC']

selector = PairSelection(
    universe=universe,
    start_date='2020-01-01',
    end_date='2026-06-16'
)

# 步骤1：相关性筛选
high_corr_pairs = selector.screen_by_correlation(threshold=0.6)

# 步骤2：协整检验
cointegrated_pairs = selector.test_cointegration(high_corr_pairs)

# 显示结果
print("\n=== 协整股票对 ===")
for pair in cointegrated_pairs[:5]:
    print(f"{pair['stock1']} - {pair['stock2']}")
    print(f"  相关性：{pair['correlation']:.3f}")
    print(f"  p-value：{pair['p_value']:.4f}")
    print(f"  协整系数：{pair['beta']:.4f}\n")
```

### 步骤2：构建交易信号

```python
class PairTradingSignals:
    """配对交易信号生成器"""
    
    def __init__(self, stock1, stock2, lookback_window=63):
        """
        初始化
        
        Args:
            stock1: 股票1代码
            stock2: 股票2代码
            lookback_window: 回望窗口（用于计算价差均值和标准差）
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.lookback_window = lookback_window
        
    def calculate_spread(self, price1, price2, method='residual'):
        """
        计算价差
        
        Args:
            price1: 股票1价格
            price2: 股票2价格
            method: 价差计算方法（'difference', 'ratio', 'residual'）
        
        Returns:
            价差序列
        """
        if method == 'difference':
            # 简单价差
            spread = price1 - price2
        
        elif method == 'ratio':
            # 价格比
            spread = price1 / (price2 + 1e-8)
        
        else:  # residual
            # 残差（基于协整关系）
            # 估计协整系数
            X = sm.add_constant(price2)
            model = sm.OLS(price1, X).fit()
            beta = model.params[1]
            alpha = model.params[0]
            
            # 计算残差
            spread = price1 - (alpha + beta * price2)
        
        return spread
    
    def calculate_zscore(self, spread, window=None):
        """
        计算价差的Z-Score
        
        Args:
            spread: 价差序列
            window: 滚动窗口（如果为None，使用整个样本）
        
        Returns:
            Z-Score序列
        """
        if window is None:
            # 使用整个样本的均值和标准差
            mean = spread.mean()
            std = spread.std()
        else:
            # 使用滚动窗口
            mean = spread.rolling(window=window).mean()
            std = spread.rolling(window=window).std()
        
        z_score = (spread - mean) / (std + 1e-8)
        
        return z_score
    
    def generate_signals(self, price1, price2, 
                        entry_threshold=2.0, 
                        exit_threshold=0.5,
                        method='residual'):
        """
        生成交易信号
        
        Args:
            price1: 股票1价格
            price2: 股票2价格
            entry_threshold: 入场阈值（Z-Score绝对值）
            exit_threshold: 出场阈值（Z-Score绝对值）
            method: 价差计算方法
        
        Returns:
            交易信号DataFrame
        """
        # 计算价差
        spread = self.calculate_spread(price1, price2, method)
        
        # 计算Z-Score
        z_score = self.calculate_zscore(spread, window=self.lookback_window)
        
        # 初始化信号
        signals = pd.DataFrame(index=price1.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['position'] = 0  # 0=无仓位，1=多空仓，-1=空多仓
        
        # 生成交易信号
        position = 0
        
        for t in range(len(signals)):
            if position == 0:  # 无仓位
                if z_score.iloc[t] > entry_threshold:
                    # 价差过高，做空stock1，做多stock2
                    position = -1
                elif z_score.iloc[t] < -entry_threshold:
                    # 价差过低，做多stock1，做空stock2
                    position = 1
            
            elif position == 1:  # 持有多空仓
                if abs(z_score.iloc[t]) < exit_threshold:
                    # 价差回归，平仓
                    position = 0
            
            elif position == -1:  # 持有空多仓
                if abs(z_score.iloc[t]) < exit_threshold:
                    # 价差回归，平仓
                    position = 0
            
            signals.iloc[t, signals.columns.get_loc('position')] = position
        
        return signals

# 使用示例
# 获取价格数据（示例）
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2026-06-16', freq='D')

# 生成协整价格序列
x = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
y = 50 + 0.8 * x + np.random.normal(0, 5, len(dates))

price1 = pd.Series(y, index=dates)
price2 = pd.Series(x, index=dates)

# 生成信号
signal_generator = PairTradingSignals('stock1', 'stock2', lookback_window=63)
signals = signal_generator.generate_signals(
    price1, price2, 
    entry_threshold=2.0,
    exit_threshold=0.5,
    method='residual'
)

print("=== 交易信号示例 ===")
print(signals.head(10))
```

### 步骤3：回测框架

```python
class PairTradingBacktester:
    """配对交易回测器"""
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001):
        """
        初始化
        
        Args:
            initial_capital: 初始资金
            transaction_cost: 交易成本（单边）
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_value = []
        self.positions = []
        self.trades = []
    
    def backtest(self, price1, price2, signals, 
                position_size=10000):
        """
        回测
        
        Args:
            price1: 股票1价格
            price2: 股票2价格
            signals: 交易信号
            position_size: 每只股票头寸市值
        
        Returns:
            回测结果
        """
        # 初始化
        cash = self.initial_capital
        shares1 = 0
        shares2 = 0
        portfolio_values = []
        trade_log = []
        
        for t in range(1, len(signals)):
            current_position = signals['position'].iloc[t-1]
            new_position = signals['position'].iloc[t]
            
            # 计算当前组合价值
            portfolio_value = cash + shares1 * price1.iloc[t] + \
                             shares2 * price2.iloc[t]
            
            # 检测仓位变化
            if new_position != current_position:
                # 平仓（如果有）
                if current_position == 1:
                    # 平多空仓：卖出stock1，买入stock2
                    cash += shares1 * price1.iloc[t] * (1 - self.transaction_cost)
                    cash -= shares2 * price2.iloc[t] * (1 + self.transaction_cost)
                    shares1 = 0
                    shares2 = 0
                    
                    trade_log.append({
                        'date': signals.index[t],
                        'action': 'close_long_short',
                        'price1': price1.iloc[t],
                        'price2': price2.iloc[t]
                    })
                
                elif current_position == -1:
                    # 平空多仓：买入stock1，卖出stock2
                    cash -= shares1 * price1.iloc[t] * (1 + self.transaction_cost)
                    cash += shares2 * price2.iloc[t] * (1 - self.transaction_cost)
                    shares1 = 0
                    shares2 = 0
                    
                    trade_log.append({
                        'date': signals.index[t],
                        'action': 'close_short_long',
                        'price1': price1.iloc[t],
                        'price2': price2.iloc[t]
                    })
                
                # 开仓（如果需要）
                if new_position == 1:
                    # 开多空仓：买入stock1，做空stock2
                    shares1 = int(position_size / price1.iloc[t])
                    shares2 = int(position_size / price2.iloc[t])
                    
                    cash -= shares1 * price1.iloc[t] * (1 + self.transaction_cost)
                    cash += shares2 * price2.iloc[t] * (1 - self.transaction_cost)
                    
                    trade_log.append({
                        'date': signals.index[t],
                        'action': 'open_long_short',
                        'shares1': shares1,
                        'shares2': shares2,
                        'price1': price1.iloc[t],
                        'price2': price2.iloc[t]
                    })
                
                elif new_position == -1:
                    # 开空多仓：做空stock1，买入stock2
                    shares1 = -int(position_size / price1.iloc[t])
                    shares2 = -int(position_size / price2.iloc[t])
                    
                    cash += abs(shares1) * price1.iloc[t] * (1 - self.transaction_cost)
                    cash -= abs(shares2) * price2.iloc[t] * (1 + self.transaction_cost)
                    
                    trade_log.append({
                        'date': signals.index[t],
                        'action': 'open_short_long',
                        'shares1': shares1,
                        'shares2': shares2,
                        'price1': price1.iloc[t],
                        'price2': price2.iloc[t]
                    })
            
            portfolio_values.append({
                'date': signals.index[t],
                'portfolio_value': portfolio_value,
                'cash': cash,
                'shares1': shares1,
                'shares2': shares2,
                'position': new_position
            })
        
        # 转换为DataFrame
        results = pd.DataFrame(portfolio_values)
        results.set_index('date', inplace=True)
        
        # 计算收益
        results['returns'] = results['portfolio_value'].pct_change()
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        
        self.portfolio_value = results
        self.trades = trade_log
        
        return results
    
    def calculate_performance_metrics(self):
        """计算绩效指标"""
        if len(self.portfolio_value) == 0:
            raise ValueError("请先运行回测")
        
        returns = self.portfolio_value['returns'].dropna()
        
        # 总收益
        total_return = self.portfolio_value['cumulative_returns'].iloc[-1] - 1
        
        # 年化收益
        trading_days = len(returns)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 年化波动
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cumulative = self.portfolio_value['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        trade_returns = []
        for i in range(1, len(self.trades)):
            if self.trades[i]['action'].startswith('close'):
                # 简化计算：假设每笔交易收益为组合价值变化
                trade_return = returns.iloc[i] if i < len(returns) else 0
                trade_returns.append(trade_return)
        
        win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades)
        }
        
        return metrics
    
    def plot_results(self):
        """绘制回测结果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 子图1：组合价值
        axes[0].plot(self.portfolio_value.index,
                    self.portfolio_value['portfolio_value'],
                    linewidth=2, label='组合价值')
        axes[0].axhline(y=self.initial_capital, color='red', 
                       linestyle='--', label='初始资金')
        axes[0].set_ylabel('组合价值', fontsize=12)
        axes[0].set_title('配对交易回测：组合价值变化', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 子图2：累计收益
        axes[1].plot(self.portfolio_value.index,
                    self.portfolio_value['cumulative_returns'],
                    linewidth=2, color='green', label='累计收益')
        axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('累计收益', fontsize=12)
        axes[1].set_title('累计收益曲线', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 子图3：Z-Score与仓位
        ax3_twin = axes[2].twinx()
        
        axes[2].plot(self.portfolio_value.index,
                    self.portfolio_value['position'],
                    linewidth=2, color='blue', label='仓位')
        axes[2].set_ylabel('仓位', fontsize=12)
        axes[2].set_xlabel('日期', fontsize=12)
        axes[2].set_title('Z-Score与交易仓位', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        # 这里需要传入signals数据，为简化示例省略
        # ax3_twin.plot(..., color='red', label='Z-Score', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('pair_trading_backtest.png', dpi=300)
        plt.show()
        
        return fig

# 使用示例
backtester = PairTradingBacktester(
    initial_capital=1000000,
    transaction_cost=0.001
)

# 回测
results = backtester.backtest(price1, price2, signals, position_size=10000)

# 计算绩效
metrics = backtester.calculate_performance_metrics()

print("=== 回测绩效 ===")
for key, value in metrics.items():
    if key in ['total_return', 'annual_return', 'annual_volatility', 
               'max_drawdown', 'win_rate']:
        print(f"{key}: {value:.2%}")
    else:
        print(f"{key}: {value}")

# 可视化
backtester.plot_results()
```

## 实战案例：美股科技股配对交易

### 数据获取与预处理

```python
def real_world_example():
    """真实市场案例：美股科技股配对交易"""
    
    # 1. 选择股票对
    # 假设我们已经通过筛选找到了一对协整股票：AAPL 和 MSFT
    stock1 = 'AAPL'
    stock2 = 'MSFT'
    
    print(f"分析股票对：{stock1} - {stock2}\n")
    
    # 2. 获取真实数据
    print("正在下载数据...")
    start_date = '2020-01-01'
    end_date = '2026-06-16'
    
    data1 = yf.download(stock1, start=start_date, end=end_date)
    data2 = yf.download(stock2, start=start_date, end=end_date)
    
    price1 = data1['Adj Close']
    price2 = data2['Adj Close']
    
    # 3. 协整检验
    print("\n执行协整检验...")
    coint_results = engle_granger_test(price1, price2)
    
    print(f"ADF统计量：{coint_results['adf_statistic']:.4f}")
    print(f"p-value：{coint_results['p_value']:.4f}")
    print(f"是否协整：{coint_results['is_cointegrated']}")
    print(f"协整方程：{stock1} = {coint_results['alpha']:.2f} + "
          f"{coint_results['beta']:.4f} * {stock2}")
    
    if not coint_results['is_cointegrated']:
        print("\n⚠️ 警告：该股票对不存在协整关系，不适合配对交易！")
        return
    
    # 4. 生成交易信号
    print("\n生成交易信号...")
    signal_generator = PairTradingSignals(stock1, stock2, lookback_window=63)
    signals = signal_generator.generate_signals(
        price1, price2,
        entry_threshold=2.0,
        exit_threshold=0.5,
        method='residual'
    )
    
    # 5. 回测
    print("执行回测...")
    backtester = PairTradingBacktester(
        initial_capital=1000000,
        transaction_cost=0.001
    )
    
    results = backtester.backtest(price1, price2, signals, position_size=10000)
    
    # 6. 计算绩效
    metrics = backtester.calculate_performance_metrics()
    
    print("\n=== 回测结果 ===")
    for key, value in metrics.items():
        if key in ['total_return', 'annual_return', 'annual_volatility', 
                   'max_drawdown', 'win_rate']:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    
    # 7. 可视化
    print("\n生成图表...")
    backtester.plot_results()
    
    # 8. 绘制价差和Z-Score
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(signals.index, signals['spread'], linewidth=2)
    axes[0].set_ylabel('价差', fontsize=12)
    axes[0].set_title(f'{stock1} - {stock2} 价差时序', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(signals.index, signals['z_score'], linewidth=2, color='orange')
    axes[1].axhline(y=2, color='red', linestyle='--', label='入场阈值')
    axes[1].axhline(y=-2, color='red', linestyle='--')
    axes[1].axhline(y=0.5, color='green', linestyle='--', label='出场阈值')
    axes[1].axhline(y=-0.5, color='green', linestyle='--')
    axes[1].set_xlabel('日期', fontsize=12)
    axes[1].set_ylabel('Z-Score', fontsize=12)
    axes[1].set_title('价差Z-Score', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pair_trading_spread_analysis.png', dpi=300)
    plt.show()
    
    return results, metrics

# 执行真实案例
# 注意：由于需要下载真实数据，这里注释掉
# results, metrics = real_world_example()
```

## 风险管理与实战注意事项

### 1. 模型风险

**问题：**
- 协整关系可能随时间断裂（结构突变）
- 参数不稳定（协整系数变化）

**解决方案：**

```python
class RollingCointegrationTest:
    """滚动协整检验"""
    
    def __init__(self, window=252, significance_level=0.05):
        """
        初始化
        
        Args:
            window: 滚动窗口大小
            significance_level: 显著性水平
        """
        self.window = window
        self.significance_level = significance_level
    
    def rolling_test(self, price1, price2):
        """
        滚动协整检验
        
        Args:
            price1: 股票1价格
            price2: 股票2价格
        
        Returns:
            检验结果序列
        """
        results = []
        
        for t in range(self.window, len(price1)):
            # 提取滚动窗口数据
            y_window = price1.iloc[t-self.window:t]
            x_window = price2.iloc[t-self.window:t]
            
            # 协整检验
            test_result = engle_granger_test(y_window, x_window, 
                                            self.significance_level)
            
            results.append({
                'date': price1.index[t],
                'is_cointegrated': test_result['is_cointegrated'],
                'p_value': test_result['p_value'],
                'beta': test_result['beta']
            })
        
        return pd.DataFrame(results).set_index('date')
    
    def plot_rolling_results(self, results):
        """绘制滚动检验结果"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 子图1：p-value时序
        axes[0].plot(results.index, results['p_value'], 
                     linewidth=2, label='p-value')
        axes[0].axhline(y=self.significance_level, color='red', 
                       linestyle='--', label='显著性水平')
        axes[0].set_ylabel('p-value', fontsize=12)
        axes[0].set_title('滚动协整检验：p-value时序', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 子图2：协整系数时序
        axes[1].plot(results.index, results['beta'], 
                     linewidth=2, color='green', label='协整系数')
        axes[1].set_xlabel('日期', fontsize=12)
        axes[1].set_ylabel('协整系数（beta）', fontsize=12)
        axes[1].set_title('滚动协整检验：协整系数时序', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rolling_cointegration.png', dpi=300)
        plt.show()

# 使用示例
rolling_tester = RollingCointegrationTest(window=252)

# 模拟数据
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2026-06-16', freq='D')
x = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
y = 50 + 0.8 * x + np.random.normal(0, 5, len(dates))

price1 = pd.Series(y, index=dates)
price2 = pd.Series(x, index=dates)

# 滚动检验
rolling_results = rolling_tester.rolling_test(price1, price2)

# 可视化
rolling_tester.plot_rolling_results(rolling_results)
```

### 2. 执行风险

**问题：**
- 交易延迟导致滑点
- 流动性不足导致无法及时成交

**解决方案：**

```python
class ExecutionRiskManager:
    """执行风险管理器"""
    
    def __init__(self, max_slippage=0.01, min_liquidity=1000000):
        """
        初始化
        
        Args:
            max_slippage: 最大允许滑点
            min_liquidity: 最小流动性要求（成交量）
        """
        self.max_slippage = max_slippage
        self.min_liquidity = min_liquidity
    
    def check_liquidity(self, volume1, volume2):
        """
        检查流动性
        
        Args:
            volume1: 股票1成交量
            volume2: 股票2成交量
        
        Returns:
            是否通过流动性检查
        """
        if volume1 < self.min_liquidity or volume2 < self.min_liquidity:
            return False
        return True
    
    def estimate_slippage(self, order_size, volume, price):
        """
        估计滑点
        
        Args:
            order_size: 订单大小
            volume: 成交量
            price: 价格
        
        Returns:
            估计滑点
        """
        # 简化模型：滑点与订单大小/成交量成正比
        slippage = (order_size / (volume + 1e-8)) * 0.1
        return min(slippage, self.max_slippage)
    
    def adjust_order_size(self, original_size, volume, price):
        """
        根据流动性调整订单大小
        
        Args:
            original_size: 原始订单大小
            volume: 成交量
            price: 价格
        
        Returns:
            调整后订单大小
        """
        # 限制订单大小为成交量的一定比例（如5%）
        max_size = volume * 0.05 / (price + 1e-8)
        adjusted_size = min(original_size, max_size)
        
        return int(adjusted_size)
```

### 3. 持仓风险管理

```python
class PositionRiskManager:
    """持仓风险管理器"""
    
    def __init__(self, max_position_size=0.1, stop_loss_threshold=0.05):
        """
        初始化
        
        Args:
            max_position_size: 最大持仓比例
            stop_loss_threshold: 止损阈值
        """
        self.max_position_size = max_position_size
        self.stop_loss_threshold = stop_loss_threshold
    
    def check_position_limit(self, portfolio_value, position_value):
        """检查持仓限制"""
        position_ratio = position_value / portfolio_value
        
        if position_ratio > self.max_position_size:
            return False
        return True
    
    def check_stop_loss(self, entry_price, current_price, position_type):
        """
        检查止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 仓位类型（'long', 'short'）
        
        Returns:
            是否触发止损
        """
        if position_type == 'long':
            loss = (entry_price - current_price) / entry_price
        else:  # short
            loss = (current_price - entry_price) / entry_price
        
        if loss > self.stop_loss_threshold:
            return True
        return False
    
    def calculate_position_size(self, portfolio_value, price, volatility):
        """
        根据波动率调整仓位大小
        
        Args:
            portfolio_value: 组合价值
            price: 价格
            volatility: 波动率
        
        Returns:
            建议仓位大小
        """
        # 基于波动率的仓位调整：波动率高，仓位小
        base_size = portfolio_value * self.max_position_size
        volatility_adjustment = 1 / (volatility + 1e-8)
        
        adjusted_size = base_size * volatility_adjustment
        
        return int(adjusted_size / price)
```

## 总结

### 核心要点回顾

1. **协整是配对交易的理论基础**
   - 协整 ≠ 相关性
   - 必须通过统计检验验证协整关系

2. **完整的策略构建流程**
   - 候选股票筛选（相关性 → 协整检验）
   - 交易信号生成（价差Z-Score）
   - 回测与绩效评估

3. **风险管理至关重要**
   - 模型风险：协整关系断裂
   - 执行风险：滑点和流动性
   - 持仓风险：止损和仓位管理

4. **实战中的挑战**
   - 真实市场的摩擦成本
   - 高频的策略竞争
   - 监管风险

### 进阶方向

1. **多因子配对交易**
   - 不仅考虑价格协整，还加入基本面因子
   - 机器学习方法筛选配对

2. **高频配对交易**
   - 利用分钟级或秒级数据
   - 需要更低的延迟和更高的执行效率

3. **跨市场配对交易**
   - 不同交易所的同种资产
   - 不同国家的相似资产

### 实践建议

**对于初学者：**
- 从模拟交易开始，熟悉策略逻辑
- 使用真实数据回测，但注意过拟合
- 关注交易成本和执行细节

**对于进阶者：**
- 优化参数选择（入场/出场阈值、回望窗口）
- 引入机器学习方法改进信号生成
- 开发风险管理系统的自动化流程

**对于机构投资者：**
- 建立配对交易的监控体系
- 考虑策略容量限制
- 合规和风险管理优先

---

配对交易是量化投资中的经典策略，虽然原理简单，但在实战中需要深厚的技术和严谨的风险管理。希望本文能够帮助读者理解配对交易的核心原理，并掌握从理论到实践的完整流程。

**记住：在量化投资中，简单的策略往往最持久。配对交易的魅力就在于它的简洁与有效。**

## 参考资料

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule". *Review of Financial Studies*.
2. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing". *Econometrica*.
3. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
4. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.

## 代码资源

完整的Python代码和Jupyter Notebook可以在以下位置获取：
- GitHub: [链接]
- 数据文件: [链接]
- 在线演示: [链接]

*如果您对配对交易有任何疑问或想法，欢迎在评论区留言讨论！*
