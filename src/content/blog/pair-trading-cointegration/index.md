---
title: "配对交易与协整分析"
date: 2026-06-21
description: "深入探讨配对交易策略的理论基础与实践方法，学习如何使用协整分析识别配对机会，构建市场中性策略，获取稳定收益。"
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "量化策略"]
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

## 引言

在量化投资领域，**配对交易（Pairs Trading）** 是一种经典的市场中性策略，它通过寻找价格具有长期均衡关系的股票对，在价格偏离均衡时建立多空组合，等待价格回归均衡后平仓获利。

配对交易的核心思想是**均值回归**：如果两只股票的价格存在长期均衡关系，那么当价格暂时偏离时，最终会回归均衡。投资者可以在价格偏离时做多低估股票、做空高估股票，等待价格收敛后平仓，获取无风险利润。

本文将深入探讨配对交易的理论基础、协整分析方法以及Python实现，帮助读者理解并应用这一经典的统计套利策略。

## 什么是配对交易？

### 配对交易的基本原理

配对交易是一种**市场中性策略**，具有以下特点：

1. **市场中性**：通过同时建立多头和空头头寸，对冲市场风险
2. **均值回归**：基于价格偏离均衡后会回归的假设
3. **统计套利**：利用统计学方法识别交易机会
4. **低风险**：理论上风险低于单边投机

### 配对交易的历史

配对交易最早由**摩根士丹利**的数量部门在1980年代提出，随后被华尔街广泛采用。著名的**长期资本管理公司（LTCM）** 就将配对交易作为其核心策略之一。

尽管LTCM后来因过度杠杆而倒闭，但配对交易作为一种策略本身仍然有效，被众多对冲基金和量化投资者使用。

## 协整分析：配对交易的理论基础

### 平稳性与协整

在介绍协整之前，我们需要先理解**平稳性（Stationarity）**：

- **平稳序列**：均值、方差、自协方差不随时间变化
- **非平稳序列**：具有时间趋势或单位根

传统的回归分析要求序列平稳，否则会出现**伪回归（Spurious Regression）** 问题。

**协整（Cointegration）** 是指多个非平稳序列的线性组合是平稳的。换句话说，虽然单个序列不平稳，但它们之间存在长期均衡关系。

### 协整的直观理解

假设有两只股票A和B，它们的价格都是随机游走（非平稳）。但是，如果它们的价格差（或比率）是平稳的，那么我们就说A和B是协整的。

协整关系的经济学意义：
- 两只公司属于同一行业，面临相同的宏观环境
- 两只公司存在竞争或合作关系
- 两只公司的业务逻辑相似

### 协整检验方法

常用的协整检验方法包括：

1. **Engle-Granger两步法**：
   - 第一步：用OLS估计协整回归
   - 第二步：对残差进行单位根检验

2. **Johansen检验**：
   - 基于向量自回归（VAR）模型
   - 可以检验多个协整关系

3. **Phillips-Ouliaris检验**：
   - 对Engle-Granger方法的改进
   - 考虑了小样本偏差

## Python实战：构建配对交易策略

下面我们使用Python实现一个完整的配对交易策略。

### 数据准备

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取股票数据（示例）
# 实际中应使用的股票数据，如贵州茅台和五粮液
stocks = ['600519.SH', '000858.SZ']  # 贵州茅台和五粮液

# 假设我们已经获取了价格数据
# prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
# 这里我们使用模拟数据演示

dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
np.random.seed(42)

# 生成协整的价格序列
n = len(dates)
trend = np.linspace(0, 100, n)
noise1 = np.cumsum(np.random.normal(0, 1, n))
noise2 = np.cumsum(np.random.normal(0, 1, n))

price1 = 100 + trend + noise1
price2 = 50 + 0.5 * trend + 0.5 * noise1 + noise2 * 0.3

prices = pd.DataFrame({
    stocks[0]: price1,
    stocks[1]: price2
}, index=dates)

print("价格数据前5行:")
print(prices.head())
print(f"\n数据期间: {prices.index[0].date()} 至 {prices.index[-1].date()}")
print(f"交易日数: {len(prices)}")
```

### 协整检验

```python
def test_cointegration(price1, price2, name1, name2):
    """
    检验两个价格序列是否存在协整关系
    
    参数:
    - price1, price2: 价格序列
    - name1, name2: 股票名称
    
    返回:
    - coint_result: 协整检验结果
    """
    # Engle-Granger两步法
    # 第一步：OLS回归
    X = price1.values
    y = price2.values
    model = OLS(y, X).fit()
    residuals = model.resid
    
    # 第二步：残差的单位根检验
    adf_result = adfuller(residuals)
    
    # 使用statsmodels的协整检验
    coint_result = coint(price1, price2)
    
    print(f"\n=== {name1} 与 {name2} 的协整检验 ===")
    print(f"OLS回归斜率: {model.params[0]:.4f}")
    print(f"OLS回归R²: {model.rsquared:.4f}")
    print(f"\nADF检验统计量: {adf_result[0]:.4f}")
    print(f"p值: {adf_result[1]:.4f}")
    print(f"1%临界值: {adf_result[4]['1%']:.4f}")
    print(f"5%临界值: {adf_result[4]['5%']:.4f}")
    print(f"10%临界值: {adf_result[4]['10%']:.4f}")
    
    print(f"\nstatsmodels协整检验:")
    print(f"检验统计量: {coint_result[0]:.4f}")
    print(f"p值: {coint_result[1]:.4f}")
    
    # 判断是否存在协整关系
    if coint_result[1] < 0.05:
        print(f"\n✓ 存在协整关系（5%显著性水平）")
        return True
    else:
        print(f"\n✗ 不存在协整关系（5%显著性水平）")
        return False

# 执行协整检验
is_cointegrated = test_cointegration(
    prices[stocks[0]], 
    prices[stocks[1]], 
    stocks[0], 
    stocks[1]
)
```

### 计算价差和Z分数

```python
def calculate_spread_zscore(prices, stock1, stock2, window=20):
    """
    计算价差和Z分数
    
    参数:
    - prices: 价格DataFrame
    - stock1, stock2: 股票代码
    - window: 滚动窗口
    
    返回:
    - spread: 价差序列
    - zscore: Z分数序列
    """
    # 计算 hedge ratio（对冲比率）
    model = OLS(prices[stock2], prices[stock1]).fit()
    hedge_ratio = model.params[0]
    
    # 计算价差
    spread = prices[stock2] - hedge_ratio * prices[stock1]
    
    # 计算Z分数
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    return spread, zscore, hedge_ratio

# 计算价差和Z分数
spread, zscore, hedge_ratio = calculate_spread_zscore(prices, stocks[0], stocks[1])

print(f"\n对冲比率（hedge ratio）: {hedge_ratio:.4f}")
print(f"价差均值: {spread.mean():.4f}")
print(f"价差标准差: {spread.std():.4f}")
print(f"Z分数范围: [{zscore.min():.2f}, {zscore.max():.2f}]")
```

### 可视化分析

```python
def plot_pair_analysis(prices, spread, zscore, stocks):
    """
    可视化配对分析结果
    
    参数:
    - prices: 价格DataFrame
    - spread: 价差序列
    - zscore: Z分数序列
    - stocks: 股票列表
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1：价格序列
    ax1 = axes[0]
    ax1.plot(prices.index, prices[stocks[0]], label=stocks[0], linewidth=2)
    ax1.plot(prices.index, prices[stocks[1]], label=stocks[1], linewidth=2)
    ax1.set_title(f'{stocks[0]} 与 {stocks[1]} 的价格序列', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：价差序列
    ax2 = axes[1]
    ax2.plot(spread.index, spread, linewidth=2, color='orange')
    ax2.axhline(spread.mean(), color='red', linestyle='--', label='均值')
    ax2.axhline(spread.mean() + 2*spread.std(), color='green', linestyle='--', label='+2σ')
    ax2.axhline(spread.mean() - 2*spread.std(), color='green', linestyle='--', label='-2σ')
    ax2.set_title('价差序列', fontsize=14, fontweight='bold')
    ax2.set_ylabel('价差', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3：Z分数
    ax3 = axes[2]
    ax3.plot(zscore.index, zscore, linewidth=2, color='purple')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(2, color='red', linestyle='--', label='+2')
    ax3.axhline(-2, color='green', linestyle='--', label='-2')
    ax3.axhline(1, color='orange', linestyle=':', alpha=0.5)
    ax3.axhline(-1, color='orange', linestyle=':', alpha=0.5)
    ax3.set_title('Z分数', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Z分数', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pair_trading_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 可视化
plot_pair_analysis(prices, spread, zscore, stocks)
```

### 回测配对交易策略

```python
def backtest_pair_trading_strategy(prices, zscore, stocks, entry_threshold=2, exit_threshold=0, 
                                   stop_loss_threshold=3, transaction_cost=0.001):
    """
    回测配对交易策略
    
    参数:
    - prices: 价格DataFrame
    - zscore: Z分数序列
    - stocks: 股票列表
    - entry_threshold: 入场阈值
    - exit_threshold: 出场阈值
    - stop_loss_threshold: 止损阈值
    - transaction_cost: 交易成本
    
    返回:
    - performance: 策略表现
    """
    # 初始化变量
    position = 0  # 0: 无仓位, 1: 多空组合, -1: 空多组合
    portfolio_value = [1.0]  # 初始资金
    returns = []
    trades = []
    
    for i in range(1, len(zscore)):
        current_zscore = zscore.iloc[i]
        prev_zscore = zscore.iloc[i-1]
        
        # 入场信号
        if position == 0:
            if current_zscore < -entry_threshold:
                # Z分数低于-2，做多stock1，做空stock2
                position = 1
                entry_price1 = prices[stocks[0]].iloc[i]
                entry_price2 = prices[stocks[1]].iloc[i]
                trades.append({
                    'date': zscore.index[i],
                    'action': 'open_long_short',
                    'price1': entry_price1,
                    'price2': entry_price2,
                    'zscore': current_zscore
                })
            elif current_zscore > entry_threshold:
                # Z分数高于2，做空stock1，做多stock2
                position = -1
                entry_price1 = prices[stocks[0]].iloc[i]
                entry_price2 = prices[stocks[1]].iloc[i]
                trades.append({
                    'date': zscore.index[i],
                    'action': 'open_short_long',
                    'price1': entry_price1,
                    'price2': entry_price2,
                    'zscore': current_zscore
                })
        
        # 出场信号
        elif position == 1:
            if abs(current_zscore) < exit_threshold or abs(current_zscore) > stop_loss_threshold:
                # 平仓
                exit_price1 = prices[stocks[0]].iloc[i]
                exit_price2 = prices[stocks[1]].iloc[i]
                
                # 计算收益
                ret = (exit_price1 - entry_price1) / entry_price1 - (exit_price2 - entry_price2) / entry_price2
                ret -= transaction_cost * 2  # 双边交易成本
                
                returns.append(ret)
                position = 0
                
                trades.append({
                    'date': zscore.index[i],
                    'action': 'close',
                    'price1': exit_price1,
                    'price2': exit_price2,
                    'zscore': current_zscore,
                    'return': ret
                })
        
        elif position == -1:
            if abs(current_zscore) < exit_threshold or abs(current_zscore) > stop_loss_threshold:
                # 平仓
                exit_price1 = prices[stocks[0]].iloc[i]
                exit_price2 = prices[stocks[1]].iloc[i]
                
                # 计算收益
                ret = (entry_price1 - exit_price1) / entry_price1 - (exit_price2 - entry_price2) / entry_price2
                ret -= transaction_cost * 2  # 双边交易成本
                
                returns.append(ret)
                position = 0
                
                trades.append({
                    'date': zscore.index[i],
                    'action': 'close',
                    'price1': exit_price1,
                    'price2': exit_price2,
                    'zscore': current_zscore,
                    'return': ret
                })
        
        # 计算组合价值
        if position == 0:
            portfolio_value.append(portfolio_value[-1])
        else:
            # 简化：假设等权投资
            portfolio_value.append(portfolio_value[-1] * (1 + ret if returns else 0))
    
    # 转换为Series
    portfolio_value = pd.Series(portfolio_value, index=zscore.index[:len(portfolio_value)])
    
    # 计算绩效指标
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    num_trades = len([t for t in trades if t['action'] != 'close'])
    win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
    
    print(f"\n=== 配对交易策略回测结果 ===")
    print(f"总收益: {total_return:.2%}")
    print(f"交易次数: {num_trades}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均收益: {np.mean(returns):.2%}" if returns else "N/A")
    print(f"收益标准差: {np.std(returns):.2%}" if returns else "N/A")
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 子图1：组合价值
    axes[0].plot(portfolio_value.index, portfolio_value, linewidth=2)
    axes[0].set_title('配对交易策略：组合价值曲线', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('组合价值', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：Z分数与交易信号
    axes[1].plot(zscore.index, zscore, linewidth=2, label='Z分数')
    axes[1].axhline(entry_threshold, color='red', linestyle='--', label='入场阈值')
    axes[1].axhline(-entry_threshold, color='red', linestyle='--')
    axes[1].axhline(exit_threshold, color='green', linestyle='--', label='出场阈值')
    axes[1].axhline(-exit_threshold, color='green', linestyle='--')
    
    # 标记交易信号
    for trade in trades:
        if trade['action'] == 'open_long_short':
            axes[1].scatter(trade['date'], trade['zscore'], color='blue', marker='^', s=100, label='做多-做空' if trade == trades[0] else '')
        elif trade['action'] == 'open_short_long':
            axes[1].scatter(trade['date'], trade['zscore'], color='red', marker='v', s=100, label='做空-做多' if trade == trades[0] else '')
        elif trade['action'] == 'close':
            axes[1].scatter(trade['date'], trade['zscore'], color='black', marker='o', s=50, label='平仓' if trade == trades[2] else '')
    
    axes[1].set_title('Z分数与交易信号', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Z分数', fontsize=12)
    axes[1].set_xlabel('日期', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pair_trading_backtest.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'returns': returns,
        'trades': trades,
        'portfolio_value': portfolio_value
    }

# 运行回测
performance = backtest_pair_trading_strategy(prices, zscore, stocks)

# 保存配图
print("\n配图已保存: pair_trading_analysis.png, pair_trading_backtest.png")
```

## 实证研究：筛选配对

### 批量筛选协整配对

```python
def screen_cointegrated_pairs(stock_list, prices, significance=0.05):
    """
    批量筛选协整配对
    
    参数:
    - stock_list: 股票列表
    - prices: 价格DataFrame
    - significance: 显著性水平
    
    返回:
    - cointegrated_pairs: 协整配对列表
    """
    cointegrated_pairs = []
    
    for i in range(len(stock_list)):
        for j in range(i+1, len(stock_list)):
            stock1 = stock_list[i]
            stock2 = stock_list[j]
            
            # 协整检验
            try:
                result = coint(prices[stock1], prices[stock2])
                p_value = result[1]
                
                if p_value < significance:
                    cointegrated_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'p_value': p_value,
                        'test_statistic': result[0]
                    })
            except Exception as e:
                print(f"检验 {stock1} 与 {stock2} 时出错: {e}")
                continue
    
    # 按p值排序
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    print(f"\n找到 {len(cointegrated_pairs)} 个协整配对")
    if cointegrated_pairs:
        print("\n前5个最显著的配对:")
        for i, pair in enumerate(cointegrated_pairs[:5]):
            print(f"{i+1}. {pair['stock1']} - {pair['stock2']} (p值: {pair['p_value']:.4f})")
    
    return cointegrated_pairs

# 示例使用（假设有股票列表）
# stock_universe = ['600519.SH', '000858.SZ', '600036.SH', '601318.SH', ...]
# pairs = screen_cointegrated_pairs(stock_universe, prices)
```

## 风险提示与注意事项

### 1. 协整关系可能断裂

协整关系基于历史数据，可能在未来断裂：

- **原因**：公司基本面变化、行业格局改变、宏观经济环境变化
- **应对**：定期重新检验协整关系，及时剔除失效配对

### 2. 模型风险

配对交易依赖统计模型，存在模型风险：

- **参数敏感**：入场阈值、出场阈值的选取影响策略表现
- **过拟合**：在历史数据上优化参数可能导致过拟合
- **应对**：使用样本外测试验证策略稳健性

### 3. 执行风险

实际操作中存在执行风险：

- **滑点**：实际成交价格与信号价格存在偏差
- **流动性**：某些股票流动性不足，难以快速建仓/平仓
- **做空限制**：A股市场做空受限，需要融券成本
- **应对**：选择流动性好的股票，考虑交易成本

### 4. 市场风险

虽然配对交易是市场中性策略，但仍存在市场风险：

- **系统性风险**：极端市场环境下，所有股票相关性上升
- **流动性危机**：市场恐慌时，价差可能持续扩大而非收敛
- **应对**：设置止损，控制杠杆

## 结论

配对交易是一种经典的市场中性策略，通过协整分析识别具有长期均衡关系的股票对，在价格偏离时建立多空组合，等待价格回归后平仓获利。

**核心要点**：

1. **理论基础**：协整分析是配对交易的核心，用于识别具有长期均衡关系的股票对
2. **策略构建**：需要计算对冲比率、价差、Z分数，设定入场和出场阈值
3. **实证方法**：使用Python可以实现从数据获取到策略回测的完整流程
4. **风险管理**：需要注意协整关系断裂、模型风险、执行风险等

对于量化投资者而言，配对交易是一项值得掌握的技能。通过不断积累经验、优化模型，投资者可以构建稳定的市场中性策略，获取低风险收益。

---

**参考文献**：

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.

**免责声明**：本文仅为学术交流，不构成投资建议。配对交易存在风险，历史表现不代表未来收益。
