---
title: "配对交易与协整分析：均值回归策略的理论与实战"
publishDate: 2026-06-21
description: "深入探讨配对交易的核心理论——协整关系，学习如何识别、测试和交易协整对，包含完整的Python实现和实战案例分析"
tags: ["配对交易", "协整分析", "均值回归", "统计套利", "Python"]
language: Chinese
---

# 配对交易与协整分析：均值回归策略的理论与实战

在量化投资的世界里，有一种策略既不依赖市场方向，也不依赖复杂的机器学习模型，却能在震荡市中稳健获利。这就是**配对交易（Pairs Trading）**——一种基于统计套利的经典策略。本文将深入探讨配对交易的核心理论——协整关系，并给出完整的Python实现。

## 什么是配对交易？

配对交易的想法非常简单：找到两只价格走势高度相关的股票，当它们的价格偏离历史均衡关系时，做多价格偏低的股票，做空价格偏高的股票，等待价格回归均衡后平仓获利。

**关键假设**：两只股票的价格虽然短期可能偏离，但长期会维持一个稳定的均衡关系。这种长期均衡关系，在计量经济学中被称为**协整（Cointegration）**。

## 理论基础：协整与误差修正模型

### 1. 平稳性 vs 协整性

在讨论协整之前，必须先理解**平稳性（Stationarity）**。

**定义**：一个时间序列 `{x_t}` 是平稳的，如果：
- 均值恒定：`E(x_t) = μ`（不随时间变化）
- 方差恒定：`Var(x_t) = σ²`
- 协方差只依赖于时差：`Cov(x_t, x_{t-k}) = γ_k`

**为什么重要？** 只有平稳的时间序列，我们才能使用标准的统计方法（如t检验、F检验）。非平稳序列会导致"伪回归"问题——即使两个序列完全独立，回归结果也可能显示显著相关。

### 2. 协整的定义

如果两个（或多个）非平稳序列的某种线性组合是平稳的，那么这些序列就是**协整**的。

数学表达：
```
y_{1,t} = β_0 + β_2 y_{2,t} + ... + β_N y_{N,t} + ε_t
```

如果：
- `y_{1,t}, y_{2,t}, ..., y_{N,t}` 都是非平稳的（通常是I(1)过程，即一阶差分平稳）
- 残差项 `ε_t` 是平稳的（I(0)过程）

那么 `y_{1,t}, y_{2,t}, ..., y_{N,t}` 之间存在协整关系。

### 3. 误差修正模型（ECM）

协整关系的动态表达是误差修正模型：

```
Δy_{1,t} = α_1 (y_{1,t-1} - β_0 - β_2 y_{2,t-1} - ... - β_N y_{N,t-1}) + Σ γ_i Δy_{1,t-i} + ε_{1,t}
```

其中：
- 括号中的项是**误差修正项**（ECT），表示对长期均衡的偏离
- `α_1` 是**调整速度**参数，负值表示价格会向均衡回归
- `Δ` 表示一阶差分

**直觉理解**：如果昨天价格偏离了均衡，今天的价格变化会"修正"这个偏离。

## 协整检验方法

### 方法1：Engle-Granger两步法

最简单的方法是Engle-Granger两步法：

**步骤1**：用OLS估计协整关系
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def engle_granger_test(y1, y2, trend='c'):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y1, y2 : Series
        两个价格序列
    trend : str
        回归中的趋势项：'c'（常数项）, 'ct'（常数项+时间趋势）, 'nc'（无常数项）
        
    Returns:
    --------
    result : dict
        检验结果，包含残差、ADF统计量、p-value等
    """
    # 步骤1：OLS回归
    if trend == 'c':
        X = sm.add_constant(y2)
    elif trend == 'ct':
        X = sm.add_constant(y2)
        X['trend'] = np.arange(len(y2))
    else:  # 'nc'
        X = y2.to_frame()
    
    model = sm.OLS(y1, X, missing='drop')
    result = model.fit()
    
    # 提取残差
    residuals = result.resid
    
    # 步骤2：对残差进行ADF检验
    adf_result = adfuller(residuals, regression='nc')  # 残差检验无常数项
    
    return {
        'cointegration_vector': result.params,
        'residuals': residuals,
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_cointegrated': adf_result[1] < 0.05  # 5%显著性水平
    }
```

**步骤2**：检验残差的平稳性（ADF检验）

```python
def adf_test(series, regression='c', max_lags=None):
    """
    增强Dickey-Fuller检验
    
    Parameters:
    -----------
    series : Series
        待检验的时间序列
    regression : str
        回归类型：'c'（常数项）, 'ct'（常数项+趋势）, 'ctt'（常数项+趋势+二次趋势）, 'nc'（无常数项）
    max_lags : int, optional
        最大滞后阶数
        
    Returns:
    --------
    result : dict
        检验结果
    """
    result = adfuller(series, regression=regression, maxlag=max_lags, autolag='AIC')
    
    print(f"ADF统计量: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("临界值:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    is_stationary = result[1] < 0.05
    print(f"\n结论: {'平稳' if is_stationary else '非平稳'}")
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': is_stationary
    }
```

### 方法2：Johansen检验（多变量协整）

当需要检验多个资产之间的协整关系时，Johansen检验更合适：

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    data : DataFrame
        多个价格序列（每列一个资产）
    det_order : int
        确定性项的阶数：
         -1: 无确定性项
          0: 常数项（无趋势）
          1: 线性趋势
    k_ar_diff : int
        VAR模型的差分滞后阶数
        
    Returns:
    --------
    result : dict
        检验结果，包含迹统计量和最大特征值统计量
    """
    # 使用VECM模型进行Johansen检验
    vecm_model = VECM(data, k_ar_diff=k_ar_diff, deterministic=det_order)
    vecm_result = vecm_model.fit()
    
    # 获取协整秩检验结果
    coint_rank = select_coint_rank(data, det_order=det_order, k_ar_diff=k_ar_diff)
    
    print(f"建议的协整秩（协整关系个数）: {coint_rank.rank}")
    print("\n迹统计量检验:")
    print(coint_rank.trace_stat)
    print("\n最大特征值检验:")
    print(coint_rank.max_eig_stat)
    
    return {
        'coint_rank': coint_rank.rank,
        'trace_stat': coint_rank.trace_stat,
        'max_eig_stat': coint_rank.max_eig_stat,
        'eigenvectors': vecm_result.beta  # 协整向量
    }
```

## 实战：构建配对交易策略

### 步骤1：寻找协整对

在A股市场，我们可以筛选出可能的协整对（通常是同行业、同产业链的公司）：

```python
def find_cointegrated_pairs(stocks_data, p_value_threshold=0.05):
    """
    在给定的股票池中寻找协整对
    
    Parameters:
    -----------
    stocks_data : DataFrame
        多个股票的价格数据（每列一个股票）
    p_value_threshold : float
        ADF检验的p-value阈值
        
    Returns:
    --------
    cointegrated_pairs : list
        协整对的列表，每个元素为 (stock1, stock2, p-value, hedge_ratio)
    """
    n = stocks_data.shape[1]
    cointegrated_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            stock1 = stocks_data.columns[i]
            stock2 = stocks_data.columns[j]
            
            # 进行Engle-Granger检验
            result = engle_granger_test(stocks_data[stock1], stocks_data[stock2])
            
            if result['is_cointegrated'] and result['p_value'] < p_value_threshold:
                hedge_ratio = result['cointegration_vector'][1]  # 协整向量中的对冲比例
                cointegrated_pairs.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'p_value': result['p_value'],
                    'adf_statistic': result['adf_statistic'],
                    'hedge_ratio': hedge_ratio,
                    'residuals': result['residuals']
                })
    
    # 按p-value排序（越小越好）
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    return cointegrated_pairs

# 使用示例
# stocks_data = pd.read_csv('stocks_prices.csv', index_col=0, parse_dates=True)
# pairs = find_cointegrated_pairs(stocks_data)
```

### 步骤2：计算价差和交易信号

找到协整对后，需要计算价差（spread）并设定交易信号：

```python
def calculate_spread(price1, price2, hedge_ratio, window=252):
    """
    计算价差（残差）
    
    Parameters:
    -----------
    price1, price2 : Series
        两个资产的价格序列
    hedge_ratio : float
        对冲比例（来自协整向量）
    window : int
        滚动计算均值的窗口
        
    Returns:
    --------
    spread : Series
        价差序列
    z_score : Series
        价差的z-score
    """
    # 计算价差：price1 - hedge_ratio * price2
    spread = price1 - hedge_ratio * price2
    
    # 计算z-score（标准化价差）
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std
    
    return spread, z_score

def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.0):
    """
    根据z-score生成交易信号
    
    Parameters:
    -----------
    z_score : Series
        价差的z-score
    entry_threshold : float
        入场阈值（绝对值）
    exit_threshold : float
        出场阈值（绝对值）
        
    Returns:
    --------
    signals : DataFrame
        交易信号，包含 'long' 和 'short' 两列
    """
    signals = pd.DataFrame(index=z_score.index)
    
    # 初始化信号
    signals['long'] = 0  # 做多价差（做多stock1，做空stock2）
    signals['short'] = 0  # 做空价差（做空stock1，做多stock2）
    
    # 生成信号
    signals['long'] = (z_score < -entry_threshold).astype(int)  # z-score过低，做多
    signals['short'] = (z_score > entry_threshold).astype(int)   # z-score过高，做空
    
    # 出场信号（当z-score回归时平仓）
    signals['long_exit'] = (z_score >= exit_threshold).astype(int)
    signals['short_exit'] = (z_score <= exit_threshold).astype(int)
    
    return signals
```

### 步骤3：回测配对交易策略

有了信号后，就可以进行回测：

```python
def backtest_pair_trading(price1, price2, signals, hedge_ratio, 
                          initial_capital=1000000, transaction_cost=0.001):
    """
    回测配对交易策略
    
    Parameters:
    -----------
    price1, price2 : Series
        两个资产的价格序列
    signals : DataFrame
        交易信号（来自generate_signals）
    hedge_ratio : float
        对冲比例
    initial_capital : float
        初始资金
    transaction_cost : float
        交易成本（单边）
        
    Returns:
    --------
    results : DataFrame
        回测结果，包含每日收益、累计收益、持仓等
    """
    # 初始化结果DataFrame
    results = pd.DataFrame(index=price1.index)
    results['price1'] = price1
    results['price2'] = price2
    results['long_signal'] = signals['long']
    results['short_signal'] = signals['short']
    
    # 初始化持仓和资金
    results['position1'] = 0  # stock1的持仓（股数）
    results['position2'] = 0  # stock2的持仓（股数）
    results['cash'] = initial_capital
    results['portfolio_value'] = initial_capital
    
    # 回测循环
    current_position = 0  # 0: 无持仓, 1: 做多价差, -1: 做空价差
    
    for i in range(1, len(results)):
        # 复制前一天的持仓和现金
        results.loc[results.index[i], 'position1'] = results['position1'].iloc[i-1]
        results.loc[results.index[i], 'position2'] = results['position2'].iloc[i-1]
        results.loc[results.index[i], 'cash'] = results['cash'].iloc[i-1]
        
        # 检查入场信号
        if current_position == 0:
            if results['long_signal'].iloc[i] == 1:
                # 做多价差：买入stock1，卖出stock2
                capital_per_stock = results['cash'].iloc[i] / 2
                shares1 = int(capital_per_stock / results['price1'].iloc[i])
                shares2 = int(capital_per_stock * hedge_ratio / results['price2'].iloc[i])
                
                # 执行交易（考虑交易成本）
                results.loc[results.index[i], 'position1'] = shares1
                results.loc[results.index[i], 'position2'] = -shares2  # 做空
                results.loc[results.index[i], 'cash'] -= (
                    shares1 * results['price1'].iloc[i] * (1 + transaction_cost) +
                    shares2 * results['price2'].iloc[i] * (1 + transaction_cost)
                )
                current_position = 1
                
            elif results['short_signal'].iloc[i] == 1:
                # 做空价差：卖出stock1，买入stock2
                capital_per_stock = results['cash'].iloc[i] / 2
                shares1 = int(capital_per_stock / results['price1'].iloc[i])
                shares2 = int(capital_per_stock * hedge_ratio / results['price2'].iloc[i])
                
                # 执行交易
                results.loc[results.index[i], 'position1'] = -shares1  # 做空
                results.loc[results.index[i], 'position2'] = shares2
                results.loc[results.index[i], 'cash'] += (
                    shares1 * results['price1'].iloc[i] * (1 - transaction_cost) +
                    shares2 * results['price2'].iloc[i] * (1 - transaction_cost)
                )
                current_position = -1
        
        # 检查出场信号
        elif current_position == 1 and results['long_exit'].iloc[i] == 1:
            # 平多仓
            results.loc[results.index[i], 'cash'] += (
                results['position1'].iloc[i] * results['price1'].iloc[i] * (1 - transaction_cost) +
                results['position2'].iloc[i] * results['price2'].iloc[i] * (1 - transaction_cost)
            )
            results.loc[results.index[i], 'position1'] = 0
            results.loc[results.index[i], 'position2'] = 0
            current_position = 0
            
        elif current_position == -1 and results['short_exit'].iloc[i] == 1:
            # 平空仓
            results.loc[results.index[i], 'cash'] -= (
                results['position1'].iloc[i] * results['price1'].iloc[i] * (1 + transaction_cost) +
                results['position2'].iloc[i] * results['price2'].iloc[i] * (1 + transaction_cost)
            )
            results.loc[results.index[i], 'position1'] = 0
            results.loc[results.index[i], 'position2'] = 0
            current_position = 0
        
        # 计算组合价值
        results.loc[results.index[i], 'portfolio_value'] = (
            results['cash'].iloc[i] +
            results['position1'].iloc[i] * results['price1'].iloc[i] +
            results['position2'].iloc[i] * results['price2'].iloc[i]
        )
    
    # 计算收益率
    results['returns'] = results['portfolio_value'].pct_change()
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results
```

### 步骤4：策略评估

```python
def evaluate_pair_trading(results):
    """
    评估配对交易策略的表现
    
    Parameters:
    -----------
    results : DataFrame
        回测结果（来自backtest_pair_trading）
        
    Returns:
    --------
    metrics : dict
        策略评估指标
    """
    returns = results['returns'].dropna()
    
    # 基本指标
    total_return = results['cumulative_returns'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    cumulative = results['cumulative_returns']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 交易统计
    num_trades = ((results['long_signal'] == 1) | (results['short_signal'] == 1)).sum()
    winning_trades = (results['returns'] > 0).sum()
    win_rate = winning_trades / len(returns) if len(returns) > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate
    }
    
    # 打印结果
    print("=== 配对交易策略评估 ===")
    print(f"总收益: {total_return:.2%}")
    print(f"年化收益: {annual_return:.2%}")
    print(f"年化波动率: {annual_vol:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"交易次数: {num_trades}")
    print(f"胜率: {win_rate:.2%}")
    
    return metrics
```

## 实战案例：A股配对交易

让我们用一个具体案例来演示完整流程。

### 案例：贵州茅台 vs 五粮液

这两只白酒龙头股历史上表现出较强的协整关系。

```python
# 步骤1：加载数据
import tushare as ts  # 假设使用tushare获取A股数据

# 设置token（需要先在tushare注册）
ts.set_token('your_token_here')
pro = ts.pro_api()

# 获取贵州茅台（600519.SH）和五粮液（000858.SZ）的日线数据
df1 = pro.daily(ts_code='600519.SH', start_date='20200101', end_date='20260621')
df2 = pro.daily(ts_code='000858.SZ', start_date='20200101', end_date='20260621')

# 处理数据
price1 = df1.set_index('trade_date')['close'].sort_index()
price2 = df2.set_index('trade_date')['close'].sort_index()

# 对齐数据
prices = pd.concat([price1, price2], axis=1, keys=['600519.SH', '000858.SZ']).dropna()

# 步骤2：协整检验
result = engle_granger_test(prices['600519.SH'], prices['000858.SZ'])
print(f"ADF统计量: {result['adf_statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"是否协整: {result['is_cointegrated']}")
hedge_ratio = result['cointegration_vector'][1]

# 步骤3：计算价差和信号
spread, z_score = calculate_spread(
    prices['600519.SH'], 
    prices['000858.SZ'], 
    hedge_ratio, 
    window=252
)
signals = generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5)

# 步骤4：回测
results = backtest_pair_trading(
    prices['600519.SH'], 
    prices['000858.SZ'], 
    signals, 
    hedge_ratio,
    initial_capital=1000000,
    transaction_cost=0.001
)

# 步骤5：评估
metrics = evaluate_pair_trading(results)

# 步骤6：可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1：价格序列
ax1 = axes[0]
ax1.plot(prices.index, prices['600519.SH'], label='贵州茅台')
ax1.plot(prices.index, prices['000858.SZ'] * hedge_ratio, label='五粮液（调整后）')
ax1.set_ylabel('价格')
ax1.legend()
ax1.set_title('价格走势')

# 子图2：价差和z-score
ax2 = axes[1]
ax2.plot(spread.index, spread, label='价差', alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel('价差')
ax2.set_title('价差（残差）')

ax2_twin = ax2.twinx()
ax2_twin.plot(z_score.index, z_score, label='Z-score', color='red', alpha=0.5)
ax2_twin.axhline(y=2, color='red', linestyle='--', alpha=0.5)
ax2_twin.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
ax2_twin.axhline(y=0, color='black', linewidth=0.5)
ax2_twin.set_ylabel('Z-score')

# 子图3：累计收益
ax3 = axes[2]
ax3.plot(results.index, results['cumulative_returns'], label='策略收益', linewidth=2)
ax3.set_ylabel('累计收益')
ax3.set_title('配对交易策略累计收益')
ax3.legend()

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/backtest_result.png', 
            dpi=300, bbox_inches='tight')

print("\n图表已保存！")
```

## 高级话题：动态对冲比例

传统配对交易使用固定的对冲比例（来自静态回归）。但在现实中，协整关系可能时变。我们可以使用**滚动窗口**或**卡尔曼滤波**来估计动态对冲比例。

### 方法：滚动窗口协整

```python
def dynamic_hedge_ratio(price1, price2, window=252):
    """
    使用滚动窗口估计动态对冲比例
    
    Parameters:
    -----------
    price1, price2 : Series
        两个资产的价格序列
    window : int
        滚动窗口长度
        
    Returns:
    --------
    hedge_ratios : Series
        动态的 hedge ratio 序列
    """
    hedge_ratios = pd.Series(index=price1.index)
    
    for end_date in price1.index[window:]:
        start_date = price1.index[price1.index < end_date][-window]
        
        # 截取窗口内数据
        y = price1.loc[start_date:end_date]
        X = price2.loc[start_date:end_date]
        
        # OLS回归
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const, missing='drop')
        result = model.fit()
        
        # 保存对冲比例
        hedge_ratios[end_date] = result.params[1]  # 不含常数项的系数
    
    return hedge_ratios.fillna(method='bfill')
```

### 方法：卡尔曼滤波

卡尔曼滤波可以更平滑地估计时变的协整向量：

```python
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(price1, price2):
    """
    使用卡尔曼滤波估计时变对冲比例
    
    Parameters:
    -----------
    price1, price2 : Series
        两个资产的价格序列
        
    Returns:
    --------
    state_means : array
        状态变量的估计值（包含截距和对冲比例）
    """
    # 观测方程：price1_t = intercept + hedge_ratio_t * price2_t + observation_noise
    # 状态方程： [intercept, hedge_ratio]_{t+1} = [intercept, hedge_ratio]_t + transition_noise
    
    # 准备数据
    observations = price1.values.reshape(-1, 1)
    X = price2.values.reshape(-1, 1)
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=np.concatenate([np.ones((len(X), 1)), X], axis=1).reshape(-1, 1, 2),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.01
    )
    
    # 滤波
    state_means, state_covariances = kf.filter(observations)
    
    return state_means
```

## 风险管理与实战建议

### 1. 风险来源

配对交易虽然理论上是市场风险中性的，但实践中仍有多种风险：

- **模型风险**：协整关系可能断裂（结构性变化）
- **执行风险**：做空可能受限（如A股的融券限制）
- **流动性风险**：小盘股的买卖价差可能很大
- **发散风险**：价差可能长期不回归，甚至持续扩大

### 2. 止损策略

必须设定合理的止损机制：

```python
def add_stop_loss(signals, z_score, stop_loss_threshold=3.0):
    """
    添加止损信号
    
    Parameters:
    -----------
    signals : DataFrame
        原始交易信号
    z_score : Series
        价差的z-score
    stop_loss_threshold : float
        止损阈值（z-score的绝对值）
        
    Returns:
    --------
    signals_with_stop : DataFrame
        添加止损信号后的交易信号
    """
    signals_with_stop = signals.copy()
    
    # 初始化止损信号
    signals_with_stop['stop_loss'] = 0
    
    # 当z-score超过止损阈值时，强制平仓
    signals_with_stop.loc[abs(z_score) > stop_loss_threshold, 'stop_loss'] = 1
    
    return signals_with_stop
```

### 3. 资金管理

即使是市场中性策略，也需要合理的资金管理：

- **仓位控制**：单一配对的最大仓位不超过总资金的10%
- **分散化**：同时交易多个不相关的配对
- **动态调整**：根据波动率调整仓位（波动率越高，仓位越低）

## 总结

配对交易是一种优雅的量化策略，它不依赖市场方向，只依赖统计规律。但要成功实施，需要注意：

1. **协整检验要严谨**：不能只看相关系数，必须用统计检验确认协整关系
2. **参数要稳健**：入场阈值、出场阈值、窗口长度都要经过充分回测
3. **风险要可控**：设定止损、控制仓位、分散投资
4. **执行要可行**：考虑交易成本、做空限制、流动性约束

### 延伸阅读

1. **Vidyamurthy (2004)**: *Pairs Trading: Quantitative Methods and Analysis* - 配对交易的经典教材
2. **Gatev et al. (2006)**: "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" - 学术论文，证明了配对交易的有效性
3. **Alexander & Dimitriu (2005)**: "Indexing and Statistical Arbitrage" - 讨论协整在统计套利中的应用

---

**完整代码仓库**: [GitHub链接]（包含数据获取、协整检验、回测、可视化等完整流程）

**免责声明**: 本文仅供学习交流，不构成投资建议。实盘交易需充分考虑风险。

*如果你对配对交易有任何疑问或想要讨论具体案例，欢迎在评论区留言！*
