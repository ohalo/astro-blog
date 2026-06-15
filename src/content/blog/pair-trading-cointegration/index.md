---
title: "配对交易与协整分析"
description: "从协整理论到实战策略，详解配对交易的统计基础、股票筛选方法和实际操作流程。"
pubDate: 2026-06-15
tags: ["统计套利", "协整分析", "配对交易"]
language: "zh"
tag: 量化交易
difficulty: 进阶
---

# 配对交易与协整分析

![配对交易示意图](/images/pair-trading-cointegration/cover.jpg)

## 引言

配对交易（Pairs Trading）是统计套利中最经典的策略之一。它不依赖市场方向，而是通过捕捉两只高度相关股票之间的暂时偏离来获利。当配对股票的价格关系出现偏离时，我们做多被低估的股票，做空被高估的股票，等待价格关系回归常态后平仓获利。

本文将系统介绍配对交易的理论基础——协整分析，并通过Python代码演示从股票筛选到策略执行的完整流程。

## 一、配对交易的理论基础

### 1.1 为什么要协整而非简单相关

很多初学者会混淆"相关性"和"协整性"。两只股票相关系数高，并不意味着它们适合配对交易。

**相关性**衡量的是收益率的同步程度，而**协整性**衡量的是价格序列之间的长期均衡关系。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller

# 示例：相关但不协整 vs 协整
np.random.seed(42)
n = 1000

# 情况1：相关但不协整（差分平稳）
x1 = np.cumsum(np.random.randn(n))  # 随机游走
y1 = x1 + np.random.randn(n) * 0.5   # 与x1相关，但无协整关系

# 情况2：协整
z = np.cumsum(np.random.randn(n))  # 共同的随机趋势
x2 = z + np.random.randn(n) * 0.1
y2 = z + np.random.randn(n) * 0.1  # x2和y2协整

# 计算相关性
corr1 = np.corrcoef(x1, y1)[0, 1]
corr2 = np.corrcoef(x2, y2)[0, 1]

print(f"情况1 - 相关系数: {corr1:.4f}, 协整检验p值: {coint(x1, y1)[1]:.4f}")
print(f"情况2 - 相关系数: {corr2:.4f}, 协整检验p值: {coint(x2, y2)[1]:.4f}")
```

**关键结论**：配对交易需要的是协整关系，而非简单的高相关性。

### 1.2 协整的数学定义

对于两个非平稳的时间序列 $X_t$ 和 $Y_t$，如果存在一个线性组合：

$$\beta_1 X_t + \beta_2 Y_t = Z_t$$

使得 $Z_t$ 是平稳序列，则称 $X_t$ 和 $Y_t$ 是协整的。

在配对交易中，我们通常设定：

$$Y_t = \alpha + \beta X_t + \epsilon_t$$

其中 $\epsilon_t$ 应该是平稳的（即均值回归的）。

## 二、协整检验的Python实现

### 2.1 Augmented Dickey-Fuller (ADF) 检验

在进行协整检验之前，需要先确认单个序列是否非平稳（即是否为I(1)过程）。

```python
def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller检验
    
    Parameters:
    -----------
    series: pd.Series, 时间序列
    title: str, 序列名称
    
    Returns:
    --------
    result: dict, 检验结果
    """
    print(f'ADF Test: {title}')
    result = adfuller(series, autolag='AIC')
    
    df_output = pd.Series(
        result[0:4],
        index=['Test Statistic', 'p-value', '#Lags Used', '#Observations']
    )
    
    for key, value in result[4].items():
        df_output[f'Critical Value ({key})'] = value
    
    print(df_output)
    print("结论:")
    if result[1] <= 0.05:
        print("拒绝原假设，序列是平稳的")
    else:
        print("不能拒绝原假设，序列是非平稳的")
    
    return {
        'statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] <= 0.05
    }

# 使用示例
# adf_result = adf_test(stock_price, 'Stock A Price')
```

### 2.2 Engle-Granger 协整检验

这是最常用的协整检验方法。

```python
def engle_granger_test(y, x, significance_level=0.05):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y: pd.Series, 第一个价格序列
    x: pd.Series, 第二个价格序列
    significance_level: float, 显著性水平
    
    Returns:
    --------
    result: dict, 检验结果和 hedge ratio
    """
    # 步骤1：OLS回归
    from sklearn.linear_model import LinearRegression
    
    X = x.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    
    beta = model.coef_[0]
    alpha = model.intercept_
    
    # 步骤2：计算残差
    spread = y - (alpha + beta * x)
    
    # 步骤3：对残差进行ADF检验
    adf_result = adfuller(spread, autolag='AIC')
    
    # 步骤4：判断协整关系
    is_cointegrated = adf_result[1] <= significance_level
    
    # 可选：使用statsmodels的coint函数（更严格）
    coint_stat, coint_pvalue, _ = coint(y, x)
    
    result = {
        'hedge_ratio': beta,
        'intercept': alpha,
        'spread': spread,
        'adf_statistic': adf_result[0],
        'adf_p_value': adf_result[1],
        'coint_p_value': coint_pvalue,
        'is_cointegrated': is_cointegrated,
        'is_cointegrated_strict': coint_pvalue <= significance_level
    }
    
    print(f"对冲比例 (beta): {beta:.4f}")
    print(f"截距 (alpha): {alpha:.4f}")
    print(f"ADF p-value: {adf_result[1]:.4f}")
    print(f"协整检验 p-value: {coint_pvalue:.4f}")
    print(f"是否协整: {is_cointegrated}")
    
    return result

# 使用示例
# coint_result = engle_granger_test(price_a, price_b)
```

### 2.3 Johansen 协整检验（多变量扩展）

当需要检验多个股票之间的协整关系时，Johansen检验更合适。

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多变量）
    
    Parameters:
    -----------
    data: pd.DataFrame, 多变量时间序列
    det_order: int, 确定性项的阶数
               0 - 无常数项，无趋势
               -1 - 无常数项，无趋势（无截距）
               1 - 有常数项，无趋势
    k_ar_diff: int, 滞后阶数
    
    Returns:
    --------
    result: dict, 检验结果
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    # 输出迹统计量结果
    print("Johansen协整检验结果:")
    print("="*50)
    for i in range(len(result.lr1)):
        print(f"协整秩 r<={i}: 迹统计量={result.lr1[i]:.4f}, "
              f"5%临界值={result.cvt[i, 1]:.4f}, "
              f"是否拒绝={result.lr1[i] > result.cvt[i, 1]}")
    
    return {
        'trace_statistic': result.lr1,
        'max_statistic': result.lr2,
        'critical_values': result.cvt,
        'eigenvectors': result.evec
    }
```

## 三、配对交易的实战流程

### 3.1 股票筛选：如何找到好的配对

筛选配对股票是策略成功的关键。我们通常从以下几个维度筛选：

1. **同行业筛选**：同行业公司更可能有协整关系
2. **市值相近**：避免流动性差异过大
3. **基本面相似**：业务模式、财务结构相似
4. **统计检验**：通过协整检验

```python
def screen_potential_pairs(stock_data, industry_map, min_corr=0.6):
    """
    筛选潜在的配对股票
    
    Parameters:
    -----------
    stock_data: dict, {stock: price_series}
    industry_map: dict, {stock: industry}
    min_corr: float, 最小相关系数
    
    Returns:
    --------
    potential_pairs: list, 潜在配对列表
    """
    import itertools
    
    potential_pairs = []
    stocks = list(stock_data.keys())
    
    # 生成所有可能的配对
    for stock1, stock2 in itertools.combinations(stocks, 2):
        # 条件1：同行业
        if industry_map.get(stock1) != industry_map.get(stock2):
            continue
        
        # 条件2：相关系数
        corr = stock_data[stock1].corr(stock_data[stock2])
        if corr < min_corr:
            continue
        
        # 条件3：协整检验
        coint_result = engle_granger_test(
            stock_data[stock1], 
            stock_data[stock2],
            significance_level=0.05
        )
        
        if coint_result['is_cointegrated']:
            potential_pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'correlation': corr,
                'hedge_ratio': coint_result['hedge_ratio'],
                'p_value': coint_result['coint_p_value']
            })
    
    # 按p-value排序（越小越好）
    potential_pairs.sort(key=lambda x: x['p_value'])
    
    return potential_pairs

# 使用示例
# pairs = screen_potential_pairs(price_data, industry_mapping)
# print(f"找到 {len(pairs)} 个潜在配对")
```

### 3.2 计算交易信号：Z-Score方法

确定配对后，核心是计算交易信号。最常用的方法是Z-Score。

```python
def calculate_z_score(spread, window=20):
    """
    计算价差的Z-Score
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    window: int, 滚动窗口
    
    Returns:
    --------
    z_score: pd.Series, Z-Score序列
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    
    z_score = (spread - mean) / std
    
    return z_score

def generate_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """
    根据Z-Score生成交易信号
    
    Parameters:
    -----------
    z_score: pd.Series, Z-Score序列
    entry_threshold: float, 入场阈值
    exit_threshold: float, 出场阈值
    
    Returns:
    --------
    signals: pd.DataFrame, 交易信号
    """
    signals = pd.DataFrame(index=z_score.index)
    signals['z_score'] = z_score
    signals['position'] = 0
    
    # 生成信号
    for i in range(1, len(signals)):
        if signals['z_score'].iloc[i] > entry_threshold:
            # Z-Score过高，做空价差（做空stock1，做多stock2）
            signals['position'].iloc[i] = -1
        elif signals['z_score'].iloc[i] < -entry_threshold:
            # Z-Score过低，做多价差（做多stock1，做空stock2）
            signals['position'].iloc[i] = 1
        elif abs(signals['z_score'].iloc[i]) < exit_threshold:
            # Z-Score回归，平仓
            signals['position'].iloc[i] = 0
        else:
            # 保持原有仓位
            signals['position'].iloc[i] = signals['position'].iloc[i-1]
    
    return signals

# 使用示例
# z_score = calculate_z_score(coint_result['spread'])
# signals = generate_trading_signals(z_score)
```

### 3.3 回测框架

有了信号后，需要进行回测验证策略有效性。

```python
def backtest_pairs_trading(stock1_prices, stock2_prices, signals, 
                          initial_capital=1000000, transaction_cost=0.001):
    """
    配对交易回测
    
    Parameters:
    -----------
    stock1_prices: pd.Series, 股票1价格
    stock2_prices: pd.Series, 股票2价格
    signals: pd.DataFrame, 交易信号
    initial_capital: float, 初始资金
    transaction_cost: float, 交易成本（单边）
    
    Returns:
    --------
    results: pd.DataFrame, 回测结果
    """
    results = pd.DataFrame(index=signals.index)
    results['stock1_price'] = stock1_prices
    results['stock2_price'] = stock2_prices
    results['position'] = signals['position']
    results['z_score'] = signals['z_score']
    
    # 计算持仓价值
    results['stock1_shares'] = 0
    results['stock2_shares'] = 0
    results['cash'] = initial_capital
    results['portfolio_value'] = initial_capital
    
    # 回测循环
    for i in range(1, len(results)):
        # 复制前一期的持仓
        results['stock1_shares'].iloc[i] = results['stock1_shares'].iloc[i-1]
        results['stock2_shares'].iloc[i] = results['stock2_shares'].iloc[i-1]
        results['cash'].iloc[i] = results['cash'].iloc[i-1]
        
        # 检测信号变化
        if results['position'].iloc[i] != results['position'].iloc[i-1]:
            # 先计算需要平仓的价值（按昨天收盘价）
            if results['position'].iloc[i-1] != 0:
                # 平仓
                stock1_value = results['stock1_shares'].iloc[i] * results['stock1_price'].iloc[i-1]
                stock2_value = results['stock2_shares'].iloc[i] * results['stock2_price'].iloc[i-1]
                
                # 平仓收入（考虑交易成本）
                if results['stock1_shares'].iloc[i] > 0:  # 卖出股票1
                    results['cash'].iloc[i] += stock1_value * (1 - transaction_cost)
                else:  # 买回股票1
                    results['cash'].iloc[i] -= abs(stock1_value) * (1 + transaction_cost)
                
                if results['stock2_shares'].iloc[i] > 0:  # 卖出股票2
                    results['cash'].iloc[i] += stock2_value * (1 - transaction_cost)
                else:  # 买回股票2
                    results['cash'].iloc[i] -= abs(stock2_value) * (1 + transaction_cost)
                
                results['stock1_shares'].iloc[i] = 0
                results['stock2_shares'].iloc[i] = 0
            
            # 开新仓（按今天开盘价，简化处理用今天价格）
            if results['position'].iloc[i] != 0:
                # 等价值投资
                total_value = results['cash'].iloc[i]
                half_value = total_value / 2
                
                if results['position'].iloc[i] == 1:
                    # 做多价差：做多stock1，做空stock2
                    results['stock1_shares'].iloc[i] = half_value / results['stock1_price'].iloc[i] * (1 - transaction_cost)
                    results['stock2_shares'].iloc[i] = -half_value / results['stock2_price'].iloc[i] * (1 + transaction_cost)
                else:
                    # 做空价差：做空stock1，做多stock2
                    results['stock1_shares'].iloc[i] = -half_value / results['stock1_price'].iloc[i] * (1 + transaction_cost)
                    results['stock2_shares'].iloc[i] = half_value / results['stock2_price'].iloc[i] * (1 - transaction_cost)
        
        # 计算当日组合价值
        portfolio_value = (results['stock1_shares'].iloc[i] * results['stock1_price'].iloc[i] +
                         results['stock2_shares'].iloc[i] * results['stock2_price'].iloc[i] +
                         results['cash'].iloc[i])
        results['portfolio_value'].iloc[i] = portfolio_value
    
    # 计算收益率
    results['returns'] = results['portfolio_value'].pct_change()
    
    # 计算累计收益
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results

# 计算绩效指标
def calculate_performance_metrics(results):
    """
    计算策略绩效指标
    """
    returns = results['returns'].dropna()
    
    # 总收益
    total_return = results['cumulative_returns'].iloc[-1] - 1
    
    # 年化收益
    trading_days = len(returns)
    years = trading_days / 252
    annual_return = (1 + total_return) ** (1/years) - 1
    
    # 夏普比率
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    
    # 最大回撤
    cumulative = results['cumulative_returns']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 胜率
    win_rate = (returns > 0).sum() / len(returns)
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    
    return metrics
```

## 四、策略优化与风险管理

### 4.1 动态对冲比例

传统的OLS方法使用固定的对冲比例（$\beta$），但在实际中，$\beta$可能会随时间变化。

```python
def rolling_hedge_ratio(y, x, window=60):
    """
    滚动计算对冲比例
    
    Parameters:
    -----------
    y: pd.Series, 因变量价格
    x: pd.Series, 自变量价格
    window: int, 滚动窗口
    
    Returns:
    --------
    hedge_ratios: pd.Series, 动态对冲比例
    """
    from sklearn.linear_model import LinearRegression
    
    hedge_ratios = pd.Series(index=y.index)
    
    for i in range(window, len(y)):
        y_window = y.iloc[i-window:i]
        x_window = x.iloc[i-window:i].values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x_window, y_window)
        
        hedge_ratios.iloc[i] = model.coef_[0]
    
    return hedge_ratios

# 使用动态对冲比例计算价差
def calculate_dynamic_spread(y, x, hedge_ratios):
    """
    使用动态对冲比例计算价差
    """
    spread = pd.Series(index=y.index)
    
    for i in range(len(y)):
        if pd.notna(hedge_ratios.iloc[i]):
            spread.iloc[i] = y.iloc[i] - hedge_ratios.iloc[i] * x.iloc[i]
    
    return spread
```

### 4.2  Kalman Filter方法

卡尔曼滤波是另一种估计时变对冲比例的方法，它更稳健且能处理噪声。

```python
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(y, x):
    """
    使用卡尔曼滤波估计时变对冲比例
    
    Parameters:
    -----------
    y: pd.Series, 股票1价格
    x: pd.Series, 股票2价格
    
    Returns:
    --------
    state_means: np.array, 估计的状态（包括截距和斜率）
    """
    # 准备观测矩阵
    observations = y.values.reshape(-1, 1)
    X = np.vstack([x.values, np.ones(len(x))]).T
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=X.reshape(-1, 1, 2)
    )
    
    # 拟合
    state_means, _ = kf.filter(observations)
    
    return state_means

# 使用示例
# states = kalman_filter_hedge_ratio(price_a, price_b)
# hedge_ratios_dynamic = states[:, 0]  # 第一列是beta（对冲比例）
```

### 4.3 风险管理要点

配对交易虽然理论上市场中性，但仍需注意以下风险：

1. **模型风险**：协整关系可能断裂
2. **执行风险**：卖空限制、流动性不足
3. **收敛风险**：价差可能长期不收敛
4. **黑天鹅风险**：突发事件导致配对关系永久破裂

**风险管理措施**：

```python
def risk_management_rules(z_score, position, max_holding_period=20, stop_loss_z=3.0):
    """
    风险管理规则
    
    Parameters:
    -----------
    z_score: pd.Series, Z-Score序列
    position: pd.Series, 持仓序列
    max_holding_period: int, 最大持仓周期
    stop_loss_z: float, 止损Z-Score阈值
    
    Returns:
    --------
    adjusted_position: pd.Series, 调整后的持仓
    """
    adjusted_position = position.copy()
    holding_period = 0
    
    for i in range(1, len(position)):
        if position.iloc[i] != 0:
            holding_period += 1
            
            # 规则1：止损（Z-Score超过止损线）
            if abs(z_score.iloc[i]) > stop_loss_z:
                adjusted_position.iloc[i] = 0
                holding_period = 0
                print(f"止损平仓: {z_score.index[i]}")
            
            # 规则2：最大持仓周期
            elif holding_period > max_holding_period:
                adjusted_position.iloc[i] = 0
                holding_period = 0
                print(f"到期平仓: {z_score.index[i]}")
        else:
            holding_period = 0
    
    return adjusted_position
```

## 五、实战案例：A股配对交易

### 5.1 数据获取与预处理

```python
import tushare as ts

def get_stock_data(stock_list, start_date, end_date):
    """
    获取A股股票数据（使用tushare）
    """
    ts.set_token('your_token')  # 需要申请tushare token
    pro = ts.pro_api()
    
    data = {}
    for stock in stock_list:
        df = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        data[stock] = df.set_index('trade_date')['close']
    
    return data

# 示例：获取银行股数据
# bank_stocks = ['600036.SH', '601166.SH', '601398.SH', '601988.SH']
# prices = get_stock_data(bank_stocks, '20200101', '20231231')
```

### 5.2 完整策略流程

```python
def complete_pairs_trading_pipeline(stock_data, trading_start_date):
    """
    完整的配对交易流程
    """
    # 步骤1：筛选配对
    print("步骤1：筛选配对...")
    potential_pairs = screen_potential_pairs(stock_data, industry_map)
    
    if len(potential_pairs) == 0:
        print("未找到合适的配对")
        return None
    
    best_pair = potential_pairs[0]
    print(f"最佳配对: {best_pair['stock1']} - {best_pair['stock2']}")
    
    # 步骤2：计算价差
    print("步骤2：计算价差...")
    stock1 = stock_data[best_pair['stock1']]
    stock2 = stock_data[best_pair['stock2']]
    
    spread = stock1 - best_pair['hedge_ratio'] * stock2
    
    # 步骤3：计算Z-Score
    print("步骤3：计算Z-Score...")
    z_score = calculate_z_score(spread)
    
    # 步骤4：生成信号
    print("步骤4：生成交易信号...")
    signals = generate_trading_signals(z_score)
    
    # 步骤5：回测
    print("步骤5：回测...")
    results = backtest_pairs_trading(stock1, stock2, signals)
    
    # 步骤6：计算绩效
    print("步骤6：计算绩效指标...")
    metrics = calculate_performance_metrics(results)
    
    print("\n========== 策略绩效 ==========")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return results, metrics

# 运行策略
# results, metrics = complete_pairs_trading_pipeline(prices, '2022-01-01')
```

### 5.3 参数优化

配对交易有多个参数需要优化：
- Z-Score入场阈值（通常1.5-2.5）
- Z-Score出场阈值（通常0-1）
- 滚动窗口（通常20-60天）
- 最大持仓周期

```python
def optimize_parameters(stock1_prices, stock2_prices, param_grid):
    """
    参数优化（网格搜索）
    """
    best_sharpe = -np.inf
    best_params = None
    
    for entry_thresh in param_grid['entry_threshold']:
        for exit_thresh in param_grid['exit_threshold']:
            for window in param_grid['window']:
                # 计算Z-Score
                coint_result = engle_granger_test(stock1_prices, stock2_prices)
                spread = coint_result['spread']
                z_score = calculate_z_score(spread, window)
                
                # 生成信号
                signals = generate_trading_signals(z_score, entry_thresh, exit_thresh)
                
                # 回测
                results = backtest_pairs_trading(stock1_prices, stock2_prices, signals)
                
                # 计算夏普比率
                metrics = calculate_performance_metrics(results)
                
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = {
                        'entry_threshold': entry_thresh,
                        'exit_threshold': exit_thresh,
                        'window': window
                    }
    
    print(f"最佳参数: {best_params}")
    print(f"最佳夏普比率: {best_sharpe:.4f}")
    
    return best_params
```

## 六、总结与展望

### 6.1 核心要点回顾

1. **协整是配对交易的基础**：不要被高相关性误导
2. **严格的统计检验**：ADF检验、协整检验必不可少
3. **合理的交易规则**：Z-Score阈值需要根据市场调整
4. **完善的风险管理**：止损、最大持仓周期等规则至关重要

### 6.2 策略局限性

1. **低频交易**：配对交易机会有限，资金利用率低
2. **模型假设**：现实中协整关系可能不稳定
3. **交易成本**：频繁交易会侵蚀利润
4. **市场风险**：极端行情下配对关系可能破裂

### 6.3 进阶方向

1. **机器学习增强**：使用PCA、聚类等方法改进配对筛选
2. **高频配对交易**：在分钟级或秒级数据上寻找机会
3. **多因子配对**：结合基本面、技术面因子
4. **跨市场配对**：ETF与成分股、期货与现货等

### 6.4 实践建议

1. **从简单开始**：先掌握经典方法，再尝试改进
2. **充分回测**：至少回测3-5年数据
3. **样本外测试**：保留最近数据进行样本外验证
4. **模拟盘先行**：实盘前务必进行模拟交易
5. **持续监控**：协整关系可能随时间衰减

---

配对交易是典型的"理论与实战结合"的量化策略。它既有坚实的统计学基础，又需要在实战中不断调整和优化。希望本文能帮助您建立起配对交易的完整知识体系，并在实践中取得成功。

*（本文约3200字，阅读时间约15分钟）*

![协整关系图](/images/pair-trading-cointegration/diagram1.jpg)

> **免责声明**：本文仅供学习交流，不构成投资建议。实盘交易需谨慎，建议充分回测和模拟验证。
