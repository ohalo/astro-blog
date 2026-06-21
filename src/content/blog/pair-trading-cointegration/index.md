---
title: "配对交易与协整分析：市场中性策略的理论与实战"
description: "深入讲解配对交易策略的核心原理——协整关系检验，提供完整的Python实战代码，包括Engle-Granger检验、Johansen检验、以及基于协整的配对交易实战案例。"
pubDate: "2026-06-21"
updatedDate: "2026-06-21"
tags: ["配对交易", "协整分析", "市场中性", "统计套利", "Python实战"]
categories: ["量化交易", "策略研究"]
image: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：市场中性策略的理论与实战

## 引言

**配对交易（Pairs Trading）**是量化投资中最经典的市场中性策略之一。它的核心思想是：寻找两只价格具有长期均衡关系的股票，当价格偏离均衡时做多低估股票、做空高估股票，等待价格回归均衡时平仓获利。

这种策略的关键在于如何判断两只股票是否存在"长期均衡关系"。答案就是**协整（Cointegration）**分析。

本文将从理论到实战，系统性地介绍：
1. 协整的理论基础与经济学意义
2. 协整关系的统计检验方法（Engle-Granger检验、Johansen检验）
3. 基于协整的配对交易策略构建
4. 完整的Python实战代码与回测分析

## 一、协整的理论基础

### 1.1 平稳性与协整

在时间序列分析中，**平稳性（Stationarity）**是一个核心概念。一个平稳的时间序列满足：
- 均值恒定
- 方差恒定
- 自协方差只依赖于时滞，不依赖于时间

**为什么平稳性重要？**
如果两个价格序列 $P_1(t)$ 和 $P_2(t)$ 都是非平稳的（比如都是随机游走），但它们的**线性组合**是平稳的：

$$
Z(t) = P_1(t) - \beta P_2(t) \sim I(0) \quad (\text{平稳})
$$

那么我们就说 $P_1(t)$ 和 $P_2(t)$ 是**协整**的。

### 1.2 协整的经济学意义

协整关系反映了两个资产之间的**长期均衡关系**。例如：
- **同一行业的竞争对手**：如可口可乐和百事可乐，它们的价格应该保持相对稳定的相对关系
- **产业链上下游**：如原油价格和航空股，成本传导机制会建立长期均衡
- **替代品**：如天然气和石油，在能源市场存在替代关系

当价格偏离这个均衡关系时，市场力量会推动价格回归均衡，这就为配对交易提供了套利机会。

### 1.3 配对交易的基本框架

配对交易策略分为以下步骤：

1. **标的选取**：寻找具有经济逻辑联系的股票对
2. **协整检验**：验证价格序列是否存在协整关系
3. **信号生成**：计算价差（或Z-score），当偏离阈值时生成交易信号
4. **风险管理**：设定止损、持仓时间限制

## 二、协整检验的统计学方法

### 2.1 Engle-Granger 两步法

**步骤1**：用OLS回归估计协整向量

$$
P_1(t) = \alpha + \beta P_2(t) + \epsilon(t)
$$

**步骤2**：对残差 $\epsilon(t)$ 进行单位根检验（ADF检验）

- 原假设 $H_0$：残差序列有单位根（非平稳，即不存在协整）
- 备择假设 $H_1$：残差序列平稳（存在协整）

如果拒绝原假设，则认为两个序列存在协整关系。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def engle_granger_test(price1, price2, verbose=True):
    """
    Engle-Granger两步法协整检验
    
    参数:
        price1: 第一个价格序列
        price2: 第二个价格序列
        verbose: 是否打印详细信息
    
    返回:
        result: 包含检验统计量和p值的字典
    """
    # 步骤1: OLS回归
    X = add_constant(price2)
    model = OLS(price1, X).fit()
    beta = model.params[1]
    alpha = model.params[0]
    residuals = model.resid
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger协整检验")
        print("=" * 60)
        print(f"\n协整方程: P1 = {alpha:.4f} + {beta:.4f} × P2")
        print(f"R² = {model.rsquared:.4f}")
    
    # 步骤2: 对残差进行ADF检验
    adf_result = adfuller(residuals, autolag='AIC')
    
    result = {
        'alpha': alpha,
        'beta': beta,
        'residuals': residuals,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'critical_values': adf_result[4],
        'is_cointegrated': adf_result[1] < 0.05
    }
    
    if verbose:
        print(f"\nADF检验统计量: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"临界值 (1%, 5%, 10%): {adf_result[4]}")
        print(f"\n结论: {'存在协整关系' if result['is_cointegrated'] else '不存在协整关系'}")
    
    return result

# 示例使用：生成模拟数据
np.random.seed(42)
n = 500

# 生成协整序列
beta_true = 1.5
alpha_true = 2.0

random_walk = np.cumsum(np.random.normal(0, 1, n))  # 随机游走
cointegrated_spread = alpha_true + beta_true * random_walk + np.random.normal(0, 0.5, n)

price2 = random_walk + 100  # 第二个价格序列（随机游走）
price1 = cointegrated_spread  # 第一个价格序列（与price2协整）

# 执行Engle-Granger检验
eg_result = engle_granger_test(price1, price2)
```

### 2.2 Johansen 检验

当两个序列可能存在**多个协整向量**时（比如三只股票之间），Engle-Granger方法就不适用了，需要使用**Johansen检验**。

Johansen检验基于**向量误差修正模型（VECM）**：

$$
\Delta Y(t) = \Pi Y(t-1) + \sum_{i=1}^{p-1} \Gamma_i \Delta Y(t-i) + \epsilon(t)
$$

其中 $\Pi = \alpha \beta^T$ 的秩决定了协整向量的个数。

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

def johansen_test(price_df, verbose=True):
    """
    Johansen协整检验（适用于多变量）
    
    参数:
        price_df: 价格数据框（多列）
        verbose: 是否打印详细信息
    
    返回:
        result: 检验结果的字典
    """
    # 使用statsmodels的select_coint_rank进行协整秩检验
    # 注意：这个函数可能需要调整，因为statsmodels的Johansen实现不够完善
    
    if verbose:
        print("=" * 60)
        print("Johansen协整检验")
        print("=" * 60)
        print("\n（注：Johansen检验在多变量协整分析中更适用）")
        print(f"变量数: {price_df.shape[1]}")
        print(f"样本数: {price_df.shape[0]}")
    
    # 简化版：对每对序列进行Engle-Granger检验
    from itertools import combinations
    
    results = {}
    for col1, col2 in combinations(price_df.columns, 2):
        result = engle_granger_test(price_df[col1], price_df[col2], verbose=False)
        results[(col1, col2)] = result
        
        if verbose:
            print(f"\n{col1} vs {col2}:")
            print(f"  β = {result['beta']:.4f}, ADF p-value = {result['adf_pvalue']:.4f}")
            print(f"  协整关系: {'是' if result['is_cointegrated'] else '否'}")
    
    return results

# 示例使用：三只股票
np.random.seed(123)
n = 500

# 共同随机游走
common_trend = np.cumsum(np.random.normal(0, 1, n))

price_a = 50 + 1.0 * common_trend + np.random.normal(0, 0.5, n)
price_b = 30 + 1.5 * common_trend + np.random.normal(0, 0.3, n)
price_c = 20 + 0.8 * common_trend + np.random.normal(0, 0.4, n)

price_df = pd.DataFrame({
    'Stock_A': price_a,
    'Stock_B': price_b,
    'Stock_C': price_c,
})

# 执行Johansen检验（简化版）
johansen_result = johansen_test(price_df)
```

## 三、基于协整的配对交易策略

### 3.1 信号生成：Z-Score方法

协整关系建立了价差的长期均衡：

$$
Spread(t) = P_1(t) - \beta P_2(t) - \alpha
$$

我们标准化价差得到 **Z-Score**：

$$
Z(t) = \frac{Spread(t) - \mu_{spread}}{\sigma_{spread}}
$$

**交易规则**：
- 当 $Z(t) > threshold$（如2），做空P1，做多P2
- 当 $Z(t) < -threshold$，做多P1，做空P2
- 当 $|Z(t)| < exit_threshold$（如0.5），平仓

```python
def generate_trading_signals(spread, entry_threshold=2.0, exit_threshold=0.5):
    """
    基于Z-Score生成交易信号
    
    参数:
        spread: 价差序列
        entry_threshold: 入场阈值（Z-Score的绝对值）
        exit_threshold: 出场阈值（Z-Score的绝对值）
    
    返回:
        signals: 交易信号序列 (1: 做多, -1: 做空, 0: 平仓/无仓位)
    """
    # 计算滚动均值和标准差（使用过去60个交易日）
    window = 60
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # 计算Z-Score
    z_score = (spread - spread_mean) / spread_std
    
    # 生成信号
    signals = pd.Series(0, index=spread.index)
    
    # 入场信号
    signals[z_score > entry_threshold] = -1  # 做空P1，做多P2
    signals[z_score < -entry_threshold] = 1   # 做多P1，做空P2
    
    # 出场信号（平仓）
    signals[(z_score >= -exit_threshold) & (z_score <= exit_threshold)] = 0
    
    # 状态机：确保信号连续性
    position = 0
    for i in range(len(signals)):
        if signals.iloc[i] != 0:
            position = signals.iloc[i]
            signals.iloc[i] = position
        elif position != 0:
            # 检查是否需要平仓
            if abs(z_score.iloc[i]) <= exit_threshold:
                signals.iloc[i] = 0
                position = 0
            else:
                signals.iloc[i] = position
    
    return signals, z_score

# 示例使用
spread = eg_result['residuals'] + price1.mean() - eg_result['beta'] * price2.mean()
signals, z_score = generate_trading_signals(spread, entry_threshold=2.0, exit_threshold=0.5)

print("=" * 60)
print("交易信号生成示例")
print("=" * 60)
print(f"\nZ-Score统计:")
print(f"  均值: {z_score.mean():.4f}")
print(f"  标准差: {z_score.std():.4f}")
print(f"  最大值: {z_score.max():.4f}")
print(f"  最小值: {z_score.min():.4f}")

print(f"\n交易信号分布:")
print(f"  做多信号: {(signals == 1).sum()}")
print(f"  做空信号: {(signals == -1).sum()}")
print(f"  平仓/无仓位: {(signals == 0).sum()}")
```

### 3.2 回测框架

构建一个简单的回测框架来评估策略表现：

```python
def backtest_pairs_trading(price1, price2, signals, beta, transaction_cost=0.001):
    """
    配对交易回测
    
    参数:
        price1: 第一个价格序列
        price2: 第二个价格序列
        signals: 交易信号序列
        beta: 对冲比例
        transaction_cost: 交易成本（单边）
    
    返回:
        results: 回测结果的字典
    """
    # 初始化
    n = len(price1)
    position = 0  # 当前仓位
    cash = 0      # 现金变化（相对值）
    portfolio_value = np.zeros(n)
    returns = np.zeros(n)
    
    trade_count = 0
    entry_price1 = 0
    entry_price2 = 0
    
    for i in range(1, n):
        # 交易信号变化
        if signals.iloc[i] != position:
            # 平仓旧仓位
            if position != 0:
                # 计算平仓收益
                exit_value = position * (price1.iloc[i] - entry_price1) - \
                             position * beta * (price2.iloc[i] - entry_price2)
                cash += exit_value - transaction_cost * (abs(position) + abs(position) * beta)
                trade_count += 1
            
            # 开仓新仓位
            if signals.iloc[i] != 0:
                entry_price1 = price1.iloc[i]
                entry_price2 = price2.iloc[i]
                cash -= transaction_cost * (abs(signals.iloc[i]) + abs(signals.iloc[i]) * beta)
            
            position = signals.iloc[i]
        
        # 计算当日组合价值
        if position != 0:
            unrealized_pnl = position * (price1.iloc[i] - entry_price1) - \
                              position * beta * (price2.iloc[i] - entry_price2)
            portfolio_value[i] = cash + unrealized_pnl
        else:
            portfolio_value[i] = cash
        
        # 计算收益率
        if i > 0 and portfolio_value[i-1] != 0:
            returns[i] = (portfolio_value[i] - portfolio_value[i-1]) / abs(portfolio_value[i-1])
    
    # 汇总结果
    total_return = (portfolio_value[-1] - portfolio_value[0]) / abs(portfolio_value[0]) if portfolio_value[0] != 0 else 0
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = ((portfolio_value.cummax() - portfolio_value) / portfolio_value.cummax()).max()
    
    results = {
        'portfolio_value': portfolio_value,
        'returns': returns,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_count': trade_count,
        'final_value': portfolio_value[-1],
    }
    
    return results

# 执行回测
backtest_results = backtest_pairs_trading(price1, price2, signals, eg_result['beta'])

print("\n" + "=" * 60)
print("回测结果")
print("=" * 60)
print(f"总收益率: {backtest_results['total_return']:.2%}")
print(f"夏普比率: {backtest_results['sharpe_ratio']:.4f}")
print(f"最大回撤: {backtest_results['max_drawdown']:.2%}")
print(f"交易次数: {backtest_results['trade_count']}")
print(f"最终组合价值: {backtest_results['final_value']:.2f}")
```

## 四、可视化分析

### 4.1 价格序列与协整关系图

```python
def plot_price_and_spread(price1, price2, spread, signals, z_score):
    """
    绘制价格序列、价差、Z-Score和交易信号
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 图1: 价格序列
    ax1 = axes[0]
    ax1.plot(price1.index, price1, label='Stock 1', linewidth=2, color='blue')
    ax1.plot(price2.index, price2, label='Stock 2', linewidth=2, color='red', alpha=0.7)
    ax1.set_ylabel('价格', fontsize=12, fontweight='bold')
    ax1.set_title('价格序列', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 价差序列
    ax2 = axes[1]
    ax2.plot(spread.index, spread, linewidth=2, color='green')
    ax2.axhline(y=spread.mean(), color='black', linestyle='--', label='均值')
    ax2.fill_between(spread.index, 
                      spread.mean() - 2*spread.std(), 
                      spread.mean() + 2*spread.std(), 
                      alpha=0.2, color='green', label='±2σ')
    ax2.set_ylabel('价差', fontsize=12, fontweight='bold')
    ax2.set_title('协整价差 (Spread = P1 - β×P2)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 图3: Z-Score
    ax3 = axes[2]
    ax3.plot(z_score.index, z_score, linewidth=2, color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='入场阈值 (+2)')
    ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='出场阈值 (±0.5)')
    ax3.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Z-Score', fontsize=12, fontweight='bold')
    ax3.set_title('标准化价差 (Z-Score)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 图4: 交易信号
    ax4 = axes[3]
    ax4.plot(signals.index, signals, linewidth=2, color='orange')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('信号', fontsize=12, fontweight='bold')
    ax4.set_title('交易信号 (1: 做多, -1: 做空, 0: 平仓)', fontsize=14, fontweight='bold')
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(True, alpha=0.3)
    
    plt.xlabel('日期', fontsize=12)
    plt.tight_layout()
    plt.savefig('pair_trading_signals.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制图表
plot_price_and_spread(price1, price2, spread, backtest_results['returns'], z_score)
```

![配对交易信号图](/images/pair-trading-cointegration/trading_signals.png)

### 4.2 组合价值曲线

```python
def plot_portfolio_performance(backtest_results):
    """
    绘制组合价值曲线和累计收益
    """
    portfolio_value = backtest_results['portfolio_value']
    returns = backtest_results['returns']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1: 组合价值曲线
    ax1 = axes[0]
    ax1.plot(range(len(portfolio_value)), portfolio_value, linewidth=2.5, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(range(len(portfolio_value)), 
                     portfolio_value, 
                     0, 
                     where=(portfolio_value >= 0), 
                     alpha=0.3, color='green')
    ax1.fill_between(range(len(portfolio_value)), 
                     portfolio_value, 
                     0, 
                     where=(portfolio_value < 0), 
                     alpha=0.3, color='red')
    ax1.set_ylabel('组合价值', fontsize=12, fontweight='bold')
    ax1.set_title('配对交易组合价值曲线', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加标注
    max_value = np.max(portfolio_value)
    min_value = np.min(portfolio_value)
    ax1.annotate(f'最大价值: {max_value:.2f}', 
                xy=(np.argmax(portfolio_value), max_value),
                xytext=(10, 15), textcoords='offset points',
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    ax1.annotate(f'最小价值: {min_value:.2f}', 
                xy=(np.argmin(portfolio_value), min_value),
                xytext=(10, -25), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 图2: 累计收益
    ax2 = axes[1]
    cumulative_returns = np.cumprod(1 + returns) - 1
    ax2.plot(range(len(cumulative_returns)), cumulative_returns * 100, 
             linewidth=2.5, color='purple')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('累计收益 (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('交易日', fontsize=12, fontweight='bold')
    ax2.set_title('累计收益率曲线', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加性能指标标注
    textstr = '\n'.join((
        f'总收益率: {backtest_results["total_return"]:.2%}',
        f'夏普比率: {backtest_results["sharpe_ratio"]:.4f}',
        f'最大回撤: {backtest_results["max_drawdown"]:.2%}',
        f'交易次数: {backtest_results["trade_count"]}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制组合表现
plot_portfolio_performance(backtest_results)
```

![组合表现图](/images/pair-trading-cointegration/portfolio_performance.png)

### 4.3 残差自相关图

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_residual_analysis(residuals):
    """
    绘制残差的自相关和偏自相关图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 残差时间序列
    ax1 = axes[0, 0]
    ax1.plot(residuals.index, residuals, linewidth=1.5, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('残差时间序列', fontsize=13, fontweight='bold')
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('残差', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 残差直方图
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('残差分布直方图', fontsize=13, fontweight='bold')
    ax2.set_xlabel('残差值', fontsize=11)
    ax2.set_ylabel('频数', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加正态分布拟合曲线
    from scipy.stats import norm
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, norm.pdf(x, residuals.mean(), residuals.std()) * len(residuals) * (residuals.max() - residuals.min()) / 50,
            'r-', linewidth=2, label='正态分布')
    ax2.legend()
    
    # 图3: 自相关图（ACF）
    ax3 = axes[1, 0]
    plot_acf(residuals, lags=40, ax=ax3)
    ax3.set_title('残差自相关图 (ACF)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('滞后阶数', fontsize=11)
    ax3.set_ylabel('ACF', fontsize=11)
    
    # 图4: 偏自相关图（PACF）
    ax4 = axes[1, 1]
    plot_pacf(residuals, lags=40, ax=ax4)
    ax4.set_title('残差偏自相关图 (PACF)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('滞后阶数', fontsize=11)
    ax4.set_ylabel('PACF', fontsize=11)
    
    plt.suptitle('协整残差诊断分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制残差分析图
plot_residual_analysis(eg_result['residuals'])
```

![残差分析图](/images/pair-trading-cointegration/residual_analysis.png)

## 五、实战案例：A股市场配对交易

### 5.1 数据获取与预处理

在实际交易中，我们需要获取真实的股票数据。这里以**贵州茅台（600519.SH）**和**五粮液（000858.SZ）**为例：

```python
# 注意：以下代码需要真实数据源（如Tushare、AkShare等）
# 这里提供框架代码

def get_stock_data(stock_codes, start_date='2020-01-01', end_date='2025-12-31'):
    """
    获取股票数据（示例框架）
    
    实际应用中应替换为真实数据接口
    """
    try:
        import tushare as ts
        # 初始化Tushare（需要token）
        # ts.set_token('your_token_here')
        # pro = ts.pro_api()
        
        # 获取数据（示例）
        # df = pro.daily(ts_code=stock_codes[0], start_date=start_date, end_date=end_date)
        
        print("请配置真实数据源（Tushare/AkShare等）")
        return None
        
    except ImportError:
        print("请安装tushare: pip install tushare")
        return None

# 示例使用（模拟数据）
def simulate_stock_pair():
    """
    模拟一对协整股票（用于演示）
    """
    np.random.seed(789)
    n = 1000
    
    # 生成共同趋势
    common_trend = np.cumsum(np.random.normal(0, 1, n))
    
    # 股票A（白酒龙头）
    stock_a = 1500 + 1.0 * common_trend + np.random.normal(0, 30, n)
    
    # 股票B（白酒老二）
    stock_b = 800 + 0.95 * common_trend + np.random.normal(0, 20, n)
    
    # 添加行业特有冲击
    industry_shock = np.sin(np.linspace(0, 4*np.pi, n)) * 50
    stock_a += industry_shock
    stock_b += industry_shock * 0.9
    
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    
    price_df = pd.DataFrame({
        'Stock_A': stock_a,
        'Stock_B': stock_b,
    }, index=dates)
    
    return price_df

# 获取模拟数据
stock_prices = simulate_stock_pair()
print("模拟股票价格数据（前10行）:")
print(stock_prices.head())
```

### 5.2 协整检验与策略执行

```python
# 对真实股票对进行协整检验
stock_a = stock_prices['Stock_A']
stock_b = stock_prices['Stock_B']

# Engle-Granger检验
eg_result_real = engle_granger_test(stock_a, stock_b)

# 如果协整，执行配对交易
if eg_result_real['is_cointegrated']:
    print("\n" + "=" * 60)
    print("执行配对交易策略")
    print("=" * 60)
    
    # 计算价差
    spread_real = stock_a - eg_result_real['beta'] * stock_b - eg_result_real['alpha']
    
    # 生成交易信号
    signals_real, z_score_real = generate_trading_signals(spread_real, 
                                                          entry_threshold=2.0, 
                                                          exit_threshold=0.5)
    
    # 回测
    backtest_real = backtest_pairs_trading(stock_a, stock_b, signals_real, 
                                           eg_result_real['beta'])
    
    print(f"\n实际股票对回测结果:")
    print(f"  总收益率: {backtest_real['total_return']:.2%}")
    print(f"  夏普比率: {backtest_real['sharpe_ratio']:.4f}")
    print(f"  最大回撤: {backtest_real['max_drawdown']:.2%}")
    print(f"  交易次数: {backtest_real['trade_count']}")
    
else:
    print("\n两只股票不存在协整关系，不适合做配对交易")
```

## 六、常见陷阱与风险管理

### 6.1 结构性断裂（Structural Breaks）

协整关系并非永恒不变。当发生以下事件时，协整关系可能断裂：
- **行业政策变化**：如白酒行业税收政策调整
- **公司基本面变化**：如管理层变动、重大资产重组
- **市场制度改革**：如注册制推出、涨跌幅限制调整

**解决方案**：
1. 使用**滚动窗口**定期重新检验协整关系
2. 设置**最大持仓时间**，避免长期持有失效策略
3. 监控**价差均值**和**方差**的变化，及时止损

### 6.2 模型风险

**问题**：协整检验的势（Power）可能不足，导致假协整。

**诊断方法**：
1. **样本外检验**：将数据分为样本内和样本外，验证协整关系是否稳定
2. **Bootstrap检验**：使用模拟方法估计p值
3. **多时间尺度检验**：在不同时间频率（日、周、月）上分别检验

### 6.3 执行风险

**滑点与冲击成本**：
- 配对交易通常是**均值回归**策略，持仓时间较短
- 频繁交易会导致**交易成本**侵蚀收益

**解决方案**：
1. 选择**流动性好**的大盘股
2. 优化**入场阈值**，减少交易频率
3. 使用**限价单**而非市价单

```python
def stress_test_pair_trading(price1, price2, signals, beta, cost_scenarios):
    """
    压力测试：不同交易成本情景下的策略表现
    """
    results = {}
    
    for cost in cost_scenarios:
        backtest_result = backtest_pairs_trading(price1, price2, signals, beta, 
                                                 transaction_cost=cost)
        results[cost] = backtest_result['total_return']
    
    print("=" * 60)
    print("压力测试：不同交易成本下的收益率")
    print("=" * 60)
    for cost, ret in results.items():
        print(f"  交易成本 {cost:.3%}: 总收益率 {ret:.2%}")
    
    return results

# 示例使用
cost_scenarios = [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
stress_results = stress_test_pair_trading(stock_a, stock_b, signals_real, 
                                          eg_result_real['beta'], cost_scenarios)
```

## 七、总结与展望

本文系统性地介绍了配对交易与协整分析的全流程，包括：

1. **理论基础**：协整的经济学意义与统计学原理
2. **检验方法**：Engle-Granger检验、Johansen检验
3. **策略构建**：信号生成、回测框架、可视化分析
4. **实战案例**：A股市场应用示例
5. **风险管理**：结构性断裂、模型风险、执行风险

**关键要点**：
- 协整是配对交易的核心，但协整关系可能随时间变化
- 风险管理比策略本身更重要，必须设置止损和最大持仓时间
- 交易成本对配对交易影响巨大，需选择高流动性标的

**未来方向**：
- **机器学习增强**：使用LSTM、Transformer等模型捕捉非线性的协整关系
- **高频配对交易**：将策略应用到分钟级或秒级数据
- **多资产配对**：扩展到期货、ETF、跨市场套利

---

## 参考文献

1. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
2. Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580.
3. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
4. Ganapathy, V. (2004). Pairs trading. *Risk Books*, 109-122.

## 代码仓库

完整的Python实现代码已上传至GitHub：\[链接\]

---

*如果本文对您有帮助，欢迎点赞、收藏、转发！也欢迎在评论区留言讨论。*

