---
title: "配对交易与协整分析：均值回归策略的数学基础与实践"
publishDate: '2026-06-16'
description: "深入讲解配对交易的理论基础、协整检验方法、实际操作流程和风险控制，附带完整的Python实现代码"
tags:
  - 量化交易
  - 统计套利
  - 配对交易
  - 协整分析
  - 均值回归
language: Chinese
---

# 配对交易与协整分析：均值回归策略的数学基础与实践

## 引言：从故事到数学

1980年代，摩根士丹利的量化团队发现了一个有趣的现象：可口可乐和百事可乐的股价走势高度相关，但偶尔会出现偏离。当偏离过大时，价格往往会回归到长期均衡关系。

这就是**配对交易（Pairs Trading）**的雏形。

配对交易的核心思想是：找到两个价格具有长期均衡关系的股票，当价格偏离时做多低估股票、做空高估股票，等待价格回归获利。

但问题来了：如何科学地判断两个股票价格是否存在"长期均衡关系"？

答案是：**协整（Cointegration）**。

## 协整理论：超越相关性

### 为什么相关性不够？

很多人误以为"两个股票价格走势像"就能做配对交易。错！

相关性衡量的是收益率的同步性，而协整衡量的是价格的长期均衡关系。

**例子**：
- 股票A和B的日收益率相关系数为0.9（高度相关）
- 但A的价格从100涨到200，B的价格从50涨到300
- 它们的价格差越来越大，不存在均值回归

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 模拟相关数据（无协整）
np.random.seed(42)
n = 500

# 两个股票的收益率高度相关
returns_a = np.random.normal(0.0005, 0.02, n)
returns_b = returns_a + np.random.normal(0, 0.005, n)  # 相关系数约0.9

# 累积得到价格
price_a = 100 * np.exp(np.cumsum(returns_a))
price_b = 50 * np.exp(np.cumsum(returns_b))

# 绘制价格走势
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(price_a, label='Stock A', linewidth=2)
axes[0].plot(price_b, label='Stock B', linewidth=2)
axes[0].set_title('Correlated but NOT Cointegrated (Price Diverges)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 计算价差
spread = price_a - price_b
axes[1].plot(spread, label='Price Spread', color='red', linewidth=2)
axes[1].axhline(y=spread.mean(), color='gray', linestyle='--', label='Mean')
axes[1].set_title('Spread (No Mean Reversion)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_not_cointegration.png', dpi=300, bbox_inches='tight')
```

### 协整的数学定义

两个时间序列 \(X_t\) 和 \(Y_t\) 是协整的，如果：

1. 它们自身是**非平稳**的（如随机游走）
2. 存在一组系数 \(\beta\)，使得线性组合 \(Z_t = Y_t - \beta X_t\) 是**平稳**的

这个线性组合 \(Z_t\) 就是**残差（Residual）**或**价差（Spread）**。

平稳性意味着：
- 均值恒定
- 方差恒定
- 自协方差只依赖于时滞（不依赖于时间）

用通俗的话说：**价差会围绕一个固定均值上下波动，并且不会偏离太远**。

## 协整检验方法

### 方法一：Engle-Granger两步法

这是最经典的协整检验方法。

**步骤**：
1. 用OLS回归：\(Y_t = \alpha + \beta X_t + \epsilon_t\)
2. 对残差 \(\epsilon_t\) 进行ADF检验（Augmented Dickey-Fuller Test）

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y : array-like
        第一个价格序列（被解释变量）
    x : array-like
        第二个价格序列（解释变量）
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    p_value : float
        ADF检验的p值（<0.05表示存在协整关系）
    beta : float
        协整系数
    residuals : array
        残差序列
    """
    # 步骤1: OLS回归
    x_with_const = sm.add_constant(x)
    model = OLS(y, x_with_const).fit()
    beta = model.params[1]
    residuals = model.resid
    
    # 步骤2: ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger Cointegration Test")
        print("=" * 60)
        print(f"ADF Statistic: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Critical Values:")
        for key, value in critical_values.items():
            print(f"  {key}: {value:.4f}")
        
        if p_value < 0.05:
            print("\n✓ 结论: 存在协整关系 (p < 0.05)")
        else:
            print("\n✗ 结论: 不存在协整关系 (p >= 0.05)")
        print("=" * 60)
    
    return p_value, beta, residuals

# 示例使用：生成协整数据
np.random.seed(42)
n = 1000
t = np.arange(n)

# 共同趋势
common_trend = 0.0002 * t + np.cumsum(np.random.normal(0, 0.01, n))

# 两个协整的股票
beta_true = 1.5
price_x = 100 + common_trend + np.random.normal(0, 0.5, n)
price_y = 50 + beta_true * common_trend + np.random.normal(0, 1, n)

# 检验协整
p_val, beta_est, residuals = engle_granger_test(price_y, price_x)
print(f"\n真实beta: {beta_true}, 估计beta: {beta_est:.4f}")
```

### 方法二：Johansen检验

当处理多个资产（不止一对）时，Johansen检验更合适。

```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多变量）
    
    Parameters:
    -----------
    data : DataFrame
        多个价格序列（每列一个资产）
    det_order : int
        确定性项的阶数（0=无常数项，1=有常数项）
    k_ar_diff : int
        VAR模型的最优滞后阶数
    
    Returns:
    --------
    result : Johansen结果对象
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    print("=" * 60)
    print("Johansen Cointegration Test")
    print("=" * 60)
    print("\nTrace Statistic (r<=0):", result.lr1[0])
    print("Max Statistic (r<=0):", result.lr2[0])
    print("\nCritical Values (90%, 95%, 99%):")
    print("Trace: ", result.cvt[0])
    print("Max: ", result.cvm[0])
    
    # 判断协整关系个数
    n_cointegrating = 0
    for i in range(len(result.lr1)):
        if result.lr1[i] > result.cvt[i, 1]:  # 95%临界值
            n_cointegrating += 1
    
    print(f"\n✓ 协整关系个数: {n_cointegrating}")
    print("=" * 60)
    
    return result

# 示例使用
data = pd.DataFrame({
    'Stock_A': price_x,
    'Stock_B': price_y,
    'Stock_C': price_x * 0.8 + np.random.normal(0, 0.3, n)  # 第三个协整资产
})

result = johansen_test(data)
```

## 配对交易实战流程

### 步骤1：寻找候选配对

不是所有股票都能做配对交易。我们需要：
- 同行业（业务相似）
- 市值相近
- 流动性好

```python
def find_cointegrated_pairs(stocks_data, p_value_threshold=0.05):
    """
    在多个股票中寻找协整配对
    
    Parameters:
    -----------
    stocks_data : DataFrame
        股票价格数据（每行一个时间点，每列一个股票）
    p_value_threshold : float
        p值阈值（默认0.05）
    
    Returns:
    --------
    cointegrated_pairs : list
        协整配对的列表，每个元素为(股票1, 股票2, p值, beta)
    """
    n_stocks = stocks_data.shape[1]
    cointegrated_pairs = []
    
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            stock1 = stocks_data.iloc[:, i]
            stock2 = stocks_data.iloc[:, j]
            
            # Engle-Granger检验
            p_val, beta, _ = engle_granger_test(
                stock1, stock2, verbose=False
            )
            
            if p_val < p_value_threshold:
                cointegrated_pairs.append({
                    'stock1': stocks_data.columns[i],
                    'stock2': stocks_data.columns[j],
                    'p_value': p_val,
                    'beta': beta
                })
    
    # 按p值排序（越小越好）
    cointegrated_pairs = sorted(
        cointegrated_pairs, 
        key=lambda x: x['p_value']
    )
    
    return cointegrated_pairs

# 示例使用：假设有10只银行股
# stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF']
# prices = yf.download(stocks, start='2020-01-01', end='2025-12-31')['Adj Close']
# pairs = find_cointegrated_pairs(prices)
```

### 步骤2：计算价差和交易信号

找到协整配对后，需要计算价差并设定交易阈值。

```python
def calculate_spread_and_signals(price1, price2, beta, entry_z=2.0, exit_z=0.5):
    """
    计算价差和生成交易信号
    
    Parameters:
    -----------
    price1, price2 : Series
        两个股票的价格序列
    beta : float
        协整系数
    entry_z : float
        入场阈值（标准差的倍数，默认2.0）
    exit_z : float
        出场阈值（默认0.5）
    
    Returns:
    --------
    signals : DataFrame
        包含价差、z-score和交易信号的DataFrame
    """
    # 计算价差
    spread = price1 - beta * price2
    
    # 计算z-score（使用滚动窗口标准化）
    window = 63  # 3个月
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 生成交易信号
    signals = pd.DataFrame(index=price1.index)
    signals['spread'] = spread
    signals['z_score'] = z_score
    signals['position'] = 0  # 1=做多价差, -1=做空价差, 0=空仓
    
    # 入场：z-score超过阈值
    signals.loc[z_score > entry_z, 'position'] = -1  # 做空价差（price1高估）
    signals.loc[z_score < -entry_z, 'position'] = 1   # 做多价差（price1低估）
    
    # 出场：z-score回归
    signals['position'] = signals['position'].replace(
        to_replace=[-1, 1], 
        method='ffill'
    )  # 持仓直到平仓
    signals.loc[z_score.abs() < exit_z, 'position'] = 0
    
    return signals

# 可视化信号
def plot_trading_signals(signals, price1, price2):
    """
    绘制交易信号图
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 图1: 价格走势
    ax1 = axes[0]
    ax1.plot(price1 / price1.iloc[0], label='Stock 1 (Normalized)', linewidth=2)
    ax1.plot(price2 / price2.iloc[0], label='Stock 2 (Normalized)', linewidth=2)
    ax1.set_title('Normalized Price Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 价差和z-score
    ax2 = axes[1]
    ax2.plot(signals['z_score'], label='Z-Score', color='purple', linewidth=2)
    ax2.axhline(y=2.0, color='red', linestyle='--', label='Entry (+2σ)')
    ax2.axhline(y=-2.0, color='green', linestyle='--', label='Entry (-2σ)')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Exit (+0.5σ)')
    ax2.axhline(y=-0.5, color='gray', linestyle=':', label='Exit (-0.5σ)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('Z-Score of Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 持仓信号
    ax3 = axes[2]
    colors = ['red', 'gray', 'green']
    labels = ['Short Spread', 'No Position', 'Long Spread']
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = signals['position'] == (1 - i)
        ax3.scatter(
            signals.index[mask], 
            signals['position'][mask],
            color=color, 
            label=label, 
            s=20, 
            alpha=0.6
        )
    ax3.set_title('Trading Positions')
    ax3.set_ylabel('Position')
    ax3.set_ylim(-1.5, 1.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pair_trading_signals.png', dpi=300, bbox_inches='tight')
```

### 步骤3：回测与绩效评估

```python
def backtest_pair_trading(price1, price2, signals, initial_capital=1000000):
    """
    回测配对交易策略
    
    Parameters:
    -----------
    price1, price2 : Series
        两个股票的价格序列
    signals : DataFrame
        交易信号（包含position列）
    initial_capital : float
        初始资金
    
    Returns:
    --------
    results : DataFrame
        回测结果（包含每日收益、累计收益等）
    """
    # 计算每日收益率
    ret1 = price1.pct_change()
    ret2 = price2.pct_change()
    
    # 策略收益率（持仓方向 × 价差收益率）
    # 注意：这里的逻辑是 position = 1 表示做多价差（买price1，卖price2）
    strategy_ret = signals['position'].shift(1) * (ret1 - signals.get('beta', 1) * ret2)
    
    # 计算累计收益
    cumulative_ret = (1 + strategy_ret).cumprod()
    
    # 计算绩效指标
    total_ret = cumulative_ret.iloc[-1] - 1
    n_days = len(strategy_ret)
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1
    
    volatility = strategy_ret.std() * np.sqrt(252)
    sharpe = annual_ret / volatility if volatility > 0 else 0
    
    # 最大回撤
    cumulative_max = cumulative_ret.cummax()
    drawdown = (cumulative_ret - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # 胜率
    winning_days = (strategy_ret > 0).sum()
    win_rate = winning_days / len(strategy_ret.dropna())
    
    # 汇总结果
    results = pd.DataFrame({
        'strategy_return': strategy_ret,
        'cumulative_return': cumulative_ret,
        'drawdown': drawdown
    })
    
    print("=" * 60)
    print("Pair Trading Backtest Results")
    print("=" * 60)
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_ret:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Number of Trades: {signals['position'].diff().abs().sum() // 2}")
    print("=" * 60)
    
    return results

# 绘制回测结果
def plot_backtest_results(results):
    """
    绘制回测绩效图
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: 累计收益
    axes[0].plot(results['cumulative_return'], linewidth=2.5, color='blue')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 回撤
    axes[1].fill_between(
        results.index, 
        0, 
        results['drawdown'], 
        color='red', 
        alpha=0.3
    )
    axes[1].plot(results['drawdown'], color='red', linewidth=1)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pair_trading_backtest.png', dpi=300, bbox_inches='tight')
```

## 实际案例分析：可口可乐 vs 百事可乐

让我们用一个真实案例来演示完整的配对交易流程。

```python
# 注：实际需要yfinance库下载数据
def case_study_coca_cola_vs_pepsi():
    """
    案例研究：可口可乐(KO) vs 百事可乐(PEP)
    """
    try:
        import yfinance as yf
        
        # 下载数据
        tickers = ['KO', 'PEP']
        data = yf.download(tickers, start='2020-01-01', end='2025-12-31')['Adj Close']
        
        # 检验协整
        p_val, beta, residuals = engle_granger_test(data['KO'], data['PEP'])
        
        if p_val < 0.05:
            print("\n✓ KO和PEP存在协整关系，可以进行配对交易")
            
            # 计算交易信号
            signals = calculate_spread_and_signals(
                data['KO'], data['PEP'], beta,
                entry_z=2.0, exit_z=0.5
            )
            
            # 回测
            results = backtest_pair_trading(
                data['KO'], data['PEP'], signals
            )
            
            # 绘图
            plot_trading_signals(signals, data['KO'], data['PEP'])
            plot_backtest_results(results)
            
            return results
        else:
            print("\n✗ KO和PEP不存在协整关系，不建议配对交易")
            return None
    
    except ImportError:
        print("需要安装yfinance库: pip install yfinance")
        return None

# 运行案例
# results = case_study_coca_cola_vs_pepsi()
```

## 风险控制与注意事项

### 风险1：协整关系断裂

协整关系不是永恒的。当公司基本面发生重大变化（并购、重组、行业变革）时，协整关系可能断裂。

**应对方法**：
- 定期重新检验协整关系（每月或每季度）
- 设置止损（当价差突破3倍标准差时强制平仓）

### 风险2：模型风险

OLS回归对异常值敏感，可能导致beta估计偏差。

**应对方法**：
- 使用稳健回归（Robust Regression）
- 考虑时变beta（滚动回归或使用卡尔曼滤波）

```python
from sklearn.linear_model import HuberRegressor

def robust_cointegration_test(y, x):
    """
    使用Huber回归的稳健协整检验
    """
    huber = HuberRegressor()
    huber.fit(x.values.reshape(-1, 1), y.values)
    beta_robust = huber.coef_[0]
    
    # 计算残差
    residuals = y - beta_robust * x
    
    # ADF检验
    adf_result = adfuller(residuals, autolag='AIC')
    p_value = adf_result[1]
    
    return p_value, beta_robust, residuals
```

### 风险3：交易成本

配对交易通常换手率较高，交易成本会侵蚀利润。

**应对方法**：
- 优化入场阈值（提高entry_z以减少交易次数）
- 考虑交易成本后再计算净收益

## 进阶话题

### 1. 多资产配对交易

不仅仅是两个资产，可以扩展到多个资产（使用Johansen检验）。

### 2. 机器学习增强

用机器学习方法改进配对选择：
- 用随机森林预测价差方向
- 用LSTM捕捉非线性的均值回归模式

### 3. 高频配对交易

在分钟级或秒级数据上实施配对交易（需要极低延迟的交易系统）。

## 结语

配对交易是一种经典的统计套利策略，它不依赖市场方向，而是利用资产间的长期均衡关系获利。

关键要点：
1. **协整是核心**：不要被高相关性迷惑
2. **风险管理至关重要**：协整关系会断裂，必须设置止损
3. **交易成本不可忽视**：高换手率会侵蚀利润

配对交易不是"印钞机"，但如果在严格的统计检验和风险管理框架下实施，它可以成为量化投资组合中有价值的阿尔法来源。

---

**参考文献**：
1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Ganapathy, V. (2004). "Statistical Arbitrage and Pairs Trading". MIT Sloan School of Management.
3. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing". Econometrica.

**完整代码仓库**：[GitHub链接](#)

*本文中的代码示例仅供参考，实际交易需谨慎评估风险。*
