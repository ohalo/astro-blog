---
title: "配对交易与协整分析"
description: "深入探讨配对交易的理论基础、协整检验方法以及在实际交易中的应用，包含完整的Python实现代码"
pubDate: 2026-06-20
tags: ["配对交易", "协整分析", "统计套利", "均值回归", "Python"]
heroImage: "/images/pair-trading-cointegration/hero.jpg"
---

# 配对交易与协整分析

配对交易（Pairs Trading）是统计套利策略的经典代表，它通过寻找价格具有长期均衡关系的资产对，在价格偏离时建立对冲头寸，等待价格回归均衡后获利。本文将深入探讨配对交易的理论基础、协整检验方法以及在实际交易中的应用。

## 配对交易的核心思想

### 什么是配对交易？

配对交易基于一个简单的经济学原理：**相似资产的相对价格应该保持稳定**。当两个高度相关的资产价格出现暂时性偏离时，我们可以：

1. **做空价格相对较高的资产**
2. **做多价格相对较低的资产**
3. **等待价格差回归均值时平仓获利**

```python
# 配对交易的基本逻辑
# 假设股票A和B具有协整关系
# 当 spread = price_A - price_B 偏离均值时：
#   if spread > mean + threshold: sell A, buy B
#   if spread < mean - threshold: buy A, sell B
#   if abs(spread - mean) < exit_threshold: close position
```

### 为什么需要协整分析？

很多人误以为**相关性（Correlation）**就足以构建配对交易策略，这是一个常见误区。

**相关性 ≠ 协整性**

- **相关性**：衡量两个序列在同方向变动的程度
- **协整性**：衡量两个序列是否存在长期均衡关系

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 示例：相关但不协整的序列
np.random.seed(42)
n = 1000

# 两个独立的随机游走
y1 = np.cumsum(np.random.normal(0, 1, n))
y2 = np.cumsum(np.random.normal(0, 1, n))

# 计算相关性
correlation = np.corrcoef(y1, y2)[0, 1]
print(f"相关系数: {correlation:.3f}")  # 可能接近0，也可能不为0

# 但y1和y2都是非平稳的，不存在长期均衡关系
# 这就是"伪回归"问题
```

**协整关系的关键特征**：
1. 两个序列都是非平稳的（如随机游走）
2. 但它们的线性组合是平稳的
3. 这意味着两者之间存在长期均衡关系

## 协整检验方法

### 方法一：Engle-Granger两步法

这是最经典的协整检验方法。

```python
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def engle_granger_test(price1, price2):
    """
    Engle-Granger两步法协整检验
    
    Parameters:
    -----------
    price1 : Series
        第一个价格序列
    price2 : Series
        第二个价格序列
    
    Returns:
    --------
    result : dict
        包含协整检验结果的字典
    """
    # 第一步：估计协整关系
    # price1 = alpha + beta * price2 + residual
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    residuals = model.resid
    
    # 第二步：检验残差的平稳性（ADF检验）
    adf_result = adfuller(residuals, autolag='AIC')
    
    result = {
        'beta': model.params[1],  # 对冲比例
        'alpha': model.params[0],  # 截距项
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_cointegrated': adf_result[1] < 0.05  # p值小于0.05，拒绝原假设（存在协整）
    }
    
    return result

# 示例：检验两只股票是否协整
# stock1_prices = get_stock_price('600519.SH', start='2020-01-01')
# stock2_prices = get_stock_price('000858.SZ', start='2020-01-01')
# result = engle_granger_test(stock1_prices, stock2_prices)
# print(f"对冲比例: {result['beta']:.3f}")
# print(f"ADF统计量: {result['adf_statistic']:.3f}")
# print(f"P值: {result['p_value']:.3f}")
# print(f"是否协整: {result['is_cointegrated']}")
```

### 方法二：Johansen检验

Johansen检验可以同时检验多个变量之间的协整关系，适用于多资产配对。

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    price_matrix : DataFrame
        价格矩阵（列：不同资产，行：时间）
    det_order : int
        确定性项阶数（0：无常数项，1：有常数项）
    k_ar_diff : int
        滞后阶数
    
    Returns:
    --------
    result : dict
        协整检验结果
    """
    # 进行Johansen检验
    joh_result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 提取统计量和临界值
    trace_stat = joh_result.lr1  # 迹统计量
    max_stat = joh_result.lr2    # 最大特征值统计量
    crit_values = joh_result.cvt  # 临界值
    
    result = {
        'trace_statistic': trace_stat,
        'max_eigenvalue_statistic': max_stat,
        'critical_values': crit_values,
        'cointegrating_vectors': joh_result.evec  # 协整向量
    }
    
    return result

# 示例：多资产协整检验
# prices = pd.DataFrame({
#     'stock1': stock1_prices,
#     'stock2': stock2_prices,
#     'stock3': stock3_prices
# })
# joh_result = johansen_test(prices)
```

### 方法三：信息准则法（Hurst指数）

Hurst指数可以用来判断一个时间序列的长期记忆性。

```python
def calculate_hurst(series, max_lag=100):
    """
    计算Hurst指数
    
    Parameters:
    -----------
    series : array-like
        时间序列
    max_lag : int
        最大滞后阶数
    
    Returns:
    --------
    hurst : float
        Hurst指数
        H < 0.5: 均值回归（适合配对交易）
        H = 0.5: 随机游走
        H > 0.5: 趋势延续
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    
    # 拟合 log(lag) vs log(tau)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Hurst指数 = 多项式的斜率
    hurst = poly[0]
    
    return hurst

# 示例：计算价格差的Hurst指数
# spread = price1 - beta * price2
# hurst = calculate_hurst(spread, max_lag=50)
# print(f"Hurst指数: {hurst:.3f}")
# if hurst < 0.5:
#     print("序列具有均值回归特性，适合配对交易")
```

## 实战：构建配对交易策略

### 步骤一：寻找配对候选

我们需要筛选出可能具有协整关系的股票对。

```python
import itertools
from tqdm import tqdm

def find_cointegrated_pairs(stock_list, start_date, end_date, p_threshold=0.05):
    """
    寻找协整的股票对
    
    Parameters:
    -----------
    stock_list : list
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    p_threshold : float
        p值阈值
    
    Returns:
    --------
    cointegrated_pairs : list
        协整对的列表，每个元素为(股票1, 股票2, p值, 对冲比例)
    """
    cointegrated_pairs = []
    
    # 获取所有股票的价格数据
    print("正在下载股票数据...")
    price_data = {}
    for stock in tqdm(stock_list):
        try:
            price_data[stock] = get_stock_price(stock, start=start_date, end=end_date)
        except:
            continue
    
    # 两两组合检验协整性
    print("正在检验协整关系...")
    for stock1, stock2 in tqdm(itertools.combinations(price_data.keys(), 2)):
        price1 = price_data[stock1]
        price2 = price_data[stock2]
        
        # 确保两个序列长度一致
        common_idx = price1.index.intersection(price2.index)
        if len(common_idx) < 252:  # 至少需要1年数据
            continue
        
        price1_aligned = price1[common_idx]
        price2_aligned = price2[common_idx]
        
        # Engle-Granger检验
        result = engle_granger_test(price1_aligned, price2_aligned)
        
        if result['p_value'] < p_threshold:
            cointegrated_pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'p_value': result['p_value'],
                'beta': result['beta'],
                'alpha': result['alpha']
            })
    
    # 按p值排序（p值越小，协整关系越强）
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    return cointegrated_pairs

# 示例：在同行业股票中寻找配对
# industry_stocks = ['600519.SH', '000858.SZ', '603288.SH', '000568.SZ']  # 白酒行业
# pairs = find_cointegrated_pairs(industry_stocks, '2020-01-01', '2025-12-31')
# print(f"找到 {len(pairs)} 个协整对")
```

### 步骤二：计算交易信号

一旦确定了配对，就需要计算交易信号。

```python
def calculate_spread_signal(price1, price2, beta, window=20, entry_z=2.0, exit_z=0.5):
    """
    计算配对交易的价差信号
    
    Parameters:
    -----------
    price1 : Series
        第一个资产的价格
    price2 : Series
        第二个资产的价格
    beta : float
        对冲比例
    window : int
        计算均值的滚动窗口
    entry_z : float
        入场信号Z-score阈值
    exit_z : float
        出场信号Z-score阈值
    
    Returns:
    --------
    signals : DataFrame
        包含价差、Z-score和交易信号的DataFrame
    """
    # 计算价差
    spread = price1 - beta * price2
    
    # 计算价差的滚动均值和标准差
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # 计算Z-score
    z_score = (spread - spread_mean) / spread_std
    
    # 生成交易信号
    signals = pd.DataFrame(index=price1.index)
    signals['spread'] = spread
    signals['z_score'] = z_score
    signals['position'] = 0  # 0: 无仓位，1: 多价差（买1卖2），-1: 空价差（卖1买2）
    
    # 入场信号
    signals.loc[z_score > entry_z, 'position'] = -1  # 价差过高，做空价差
    signals.loc[z_score < -entry_z, 'position'] = 1   # 价差过低，做多价差
    
    # 出场信号（当Z-score回归到exit_z以内时平仓）
    for i in range(1, len(signals)):
        if signals['position'].iloc[i-1] != 0 and abs(signals['z_score'].iloc[i]) < exit_z:
            signals.loc[signals.index[i], 'position'] = 0
    
    return signals

# 示例：计算交易信号
# price1 = get_stock_price('600519.SH', start='2023-01-01')
# price2 = get_stock_price('000858.SZ', start='2023-01-01')
# signals = calculate_spread_signal(price1, price2, beta=1.5, window=20, entry_z=2.0, exit_z=0.5)
```

### 步骤三：回测策略

有了交易信号后，就可以进行回测。

```python
def backtest_pair_trading(signals, price1, price2, beta, transaction_cost=0.001):
    """
    回测配对交易策略
    
    Parameters:
    -----------
    signals : DataFrame
        交易信号DataFrame
    price1 : Series
        第一个资产的价格
    price2 : Series
        第二个资产的价格
    beta : float
        对冲比例
    transaction_cost : float
        单边交易费率
    
    Returns:
    --------
    results : DataFrame
        回测结果
    """
    results = signals.copy()
    
    # 计算收益率
    returns1 = price1.pct_change()
    returns2 = price2.pct_change()
    
    # 计算策略收益（考虑对冲比例）
    # 当 position = 1: 买入1，卖出2（按beta比例）
    # 当 position = -1: 卖出1，买入2（按beta比例）
    strategy_returns = results['position'].shift(1) * (returns1 - beta * returns2)
    
    # 计算交易成本
    position_change = results['position'].diff().abs()
    transaction_costs = position_change * transaction_cost * (1 + abs(beta))
    
    # 净收益
    net_returns = strategy_returns - transaction_costs
    
    # 计算累积收益
    cumulative_returns = (1 + net_returns).cumprod()
    
    results['strategy_return'] = net_returns
    results['cumulative_return'] = cumulative_returns
    results['transaction_costs'] = transaction_costs
    
    return results

# 示例：回测配对交易策略
# results = backtest_pair_trading(signals, price1, price2, beta=1.5, transaction_cost=0.001)
# 
# # 计算性能指标
# total_return = results['cumulative_return'].iloc[-1] - 1
# sharpe_ratio = results['strategy_return'].mean() / results['strategy_return'].std() * np.sqrt(252)
# max_drawdown = (results['cumulative_return'] / results['cumulative_return'].cummax() - 1).min()
# 
# print(f"总收益: {total_return:.2%}")
# print(f"夏普比率: {sharpe_ratio:.2f}")
# print(f"最大回撤: {max_drawdown:.2%}")
```

## 实战案例：A股白酒行业配对交易

让我们用实际的A股数据来构建一个配对交易策略。

### 数据准备

```python
# 选择白酒行业代表性股票
liquor_stocks = {
    '600519.SH': '贵州茅台',
    '000858.SZ': '五粮液',
    '603288.SH': '海天味业',  # 注意：这是调味品，仅作示例
    '000568.SZ': '泸州老窖'
}

# 下载2020-2025年日度数据
price_data = {}
for code, name in liquor_stocks.items():
    print(f"正在下载 {name} ({code}) 数据...")
    price_data[code] = get_stock_price(code, start='2020-01-01', end='2025-12-31')
    print(f"数据下载完成，共 {len(price_data[code])} 个交易日")
```

### 协整检验

```python
# 检验所有配对的协整性
pairs_results = []

for (code1, name1), (code2, name2) in itertools.combinations(liquor_stocks.items(), 2):
    price1 = price_data[code1]
    price2 = price_data[code2]
    
    # 对齐数据
    common_idx = price1.index.intersection(price2.index)
    price1_aligned = price1[common_idx]
    price2_aligned = price2[common_idx]
    
    # Engle-Granger检验
    result = engle_granger_test(price1_aligned, price2_aligned)
    
    if result['is_cointegrated']:
        pairs_results.append({
            'pair': f"{name1} - {name2}",
            'code1': code1,
            'code2': code2,
            'p_value': result['p_value'],
            'beta': result['beta'],
            'adf_statistic': result['adf_statistic']
        })

# 按p值排序
pairs_results.sort(key=lambda x: x['p_value'])

print("\n协整配对检验结果：")
for i, pair in enumerate(pairs_results[:5], 1):
    print(f"{i}. {pair['pair']}")
    print(f"   P值: {pair['p_value']:.4f}")
    print(f"   对冲比例(beta): {pair['beta']:.3f}")
    print(f"   ADF统计量: {pair['adf_statistic']:.3f}\n")
```

### 策略回测

选择p值最小的配对进行回测。

```python
# 选择最佳配对
best_pair = pairs_results[0]
code1 = best_pair['code1']
code2 = best_pair['code2']
beta = best_pair['beta']

print(f"最佳配对: {best_pair['pair']}")
print(f"对冲比例: {beta:.3f}")

# 获取价格数据
price1 = price_data[code1]
price2 = price_data[code2]

# 计算交易信号
signals = calculate_spread_signal(
    price1, price2, beta,
    window=20,       # 20日滚动窗口
    entry_z=2.0,     # Z-score > 2 或 < -2 时入场
    exit_z=0.5       # Z-score 回归到 ±0.5 时出场
)

# 回测
results = backtest_pair_trading(
    signals, price1, price2, beta,
    transaction_cost=0.001  # 单边0.1%交易成本
)

# 绘制结果
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1：价格序列
ax1 = axes[0]
ax1.plot(price1.index, price1.values, label=liquor_stocks[code1], alpha=0.7)
ax1.plot(price2.index, price2.values / beta, label=f"{liquor_stocks[code2]} (调整)", alpha=0.7)
ax1.set_ylabel('价格', fontsize=12)
ax1.set_title('配对股票价格走势', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：价差和Z-score
ax2 = axes[1]
ax2.plot(signals.index, signals['spread'].values, label='价差', alpha=0.7)
ax2.axhline(y=signals['spread'].mean(), color='r', linestyle='--', label='均值')
ax2.set_ylabel('价差', fontsize=12)
ax2.set_title('价差序列', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
ax2_twin.plot(signals.index, signals['z_score'].values, color='orange', label='Z-score', alpha=0.5)
ax2_twin.axhline(y=2.0, color='g', linestyle=':', label='入场阈值')
ax2_twin.axhline(y=-2.0, color='g', linestyle=':')
ax2_twin.axhline(y=0.5, color='r', linestyle=':', label='出场阈值')
ax2_twin.axhline(y=-0.5, color='r', linestyle=':')
ax2_twin.set_ylabel('Z-score', fontsize=12)

# 子图3：累积收益
ax3 = axes[2]
ax3.plot(results.index, results['cumulative_return'].values, label='策略累积收益', linewidth=2)
ax3.set_xlabel('日期', fontsize=12)
ax3.set_ylabel('累积收益', fontsize=12)
ax3.set_title('配对交易策略累积收益', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pair_trading_results.png', dpi=300, bbox_inches='tight')
print("回测结果图已保存为 pair_trading_results.png")
```

## 风险管理与策略优化

### 风险管理要点

1. **止损机制**：当价差持续扩大超过某个阈值时，强制平仓
2. **仓位管理**：根据价差的波动率动态调整仓位
3. **配对失效监控**：定期重新检验协整关系

```python
def add_risk_management(results, stop_loss_z=3.0, max_holding_days=20):
    """
    添加风险管理机制
    
    Parameters:
    -----------
    results : DataFrame
        回测结果DataFrame
    stop_loss_z : float
        止损Z-score阈值
    max_holding_days : int
        最大持仓天数
    
    Returns:
    --------
    results_with_rm : DataFrame
        添加风险管理后的结果
    """
    results_with_rm = results.copy()
    
    # 跟踪持仓天数
    holding_days = 0
    
    for i in range(1, len(results_with_rm)):
        current_position = results_with_rm['position'].iloc[i]
        previous_position = results_with_rm['position'].iloc[i-1]
        
        # 计算持仓天数
        if current_position != 0:
            if current_position == previous_position:
                holding_days += 1
            else:
                holding_days = 1  # 新仓位
        else:
            holding_days = 0
        
        # 止损：Z-score超过阈值
        if abs(results_with_rm['z_score'].iloc[i]) > stop_loss_z:
            results_with_rm.loc[results_with_rm.index[i], 'position'] = 0
            print(f"止损触发: {results_with_rm.index[i]}, Z-score: {results_with_rm['z_score'].iloc[i]:.2f}")
        
        # 强制平仓：持仓时间过长
        if holding_days > max_holding_days and current_position != 0:
            results_with_rm.loc[results_with_rm.index[i], 'position'] = 0
            print(f"强制平仓: {results_with_rm.index[i]}, 持仓天数: {holding_days}")
            holding_days = 0
    
    return results_with_rm

# 应用风险管理
# results_with_rm = add_risk_management(results, stop_loss_z=3.0, max_holding_days=20)
```

### 策略优化方向

1. **动态参数调整**：根据市场波动率动态调整Z-score阈值
2. **多配对组合**：同时交易多个配对，分散风险
3. **机器学习增强**：使用机器学习模型预测价差回归概率

```python
def dynamic_threshold_adjustment(volatility, base_entry_z=2.0, base_exit_z=0.5):
    """
    根据波动率动态调整入场出场阈值
    
    Parameters:
    -----------
    volatility : Series
        波动率序列
    base_entry_z : float
        基础入场Z-score
    base_exit_z : float
        基础出场Z-score
    
    Returns:
    --------
    dynamic_entry_z : Series
        动态入场阈值
    dynamic_exit_z : Series
        动态出场阈值
    """
    # 波动率分位数
    vol_quantile = volatility.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # 高波动率时期：提高阈值（减少交易频率，降低虚假信号）
    # 低波动率时期：降低阈值（增加交易机会）
    dynamic_entry_z = base_entry_z * (1 + 0.5 * (vol_quantile - 0.5))
    dynamic_exit_z = base_exit_z * (1 + 0.3 * (vol_quantile - 0.5))
    
    return dynamic_entry_z, dynamic_exit_z

# 示例：动态阈值调整
# market_volatility = calculate_market_volatility(market_returns, window=20)
# dynamic_entry_z, dynamic_exit_z = dynamic_threshold_adjustment(market_volatility)
```

## 实战中的挑战

### 挑战一：结构性断裂

协整关系可能随时间变化而断裂（如行业政策变化、公司重组等）。

**解决方案**：
- 使用滚动窗口定期重新检验协整关系
- 当p值持续升高时，停止交易该配对
- 设置"配对有效期"，到期后重新筛选

```python
def monitor_cointegration_break(price1, price2, beta, window=252, p_threshold=0.10):
    """
    监控协整关系是否断裂
    
    Parameters:
    -----------
    price1, price2 : Series
        价格序列
    beta : float
        对冲比例
    window : int
        滚动窗口大小
    p_threshold : float
        p值阈值（超过此值认为协整关系断裂）
    """
    break_dates = []
    
    for i in range(window, len(price1)):
        # 使用滚动窗口数据检验协整
        price1_window = price1.iloc[i-window:i]
        price2_window = price2.iloc[i-window:i]
        
        result = engle_granger_test(price1_window, price2_window)
        
        if result['p_value'] > p_threshold:
            break_dates.append(price1.index[i])
            print(f"警告：协整关系可能在 {price1.index[i]} 断裂，p值: {result['p_value']:.3f}")
    
    return break_dates

# 监控协整断裂
# break_dates = monitor_cointegration_break(price1, price2, beta, window=252, p_threshold=0.10)
```

### 挑战二：交易成本侵蚀

配对交易通常涉及频繁交易，交易成本会显著侵蚀收益。

**解决方案**：
- 优化Z-score阈值，减少虚假信号
- 使用限价单降低冲击成本
- 考虑配对的整体流动性，避免交易冷门股票

### 挑战三：模型风险

协整检验本身存在模型风险（如滞后阶数选择、确定性项选择等）。

**解决方案**：
- 使用多种协整检验方法交叉验证
- 进行样本外测试，验证策略稳健性
- 不盲目追求p值极小的结果（可能过拟合）

## 结论

配对交易是一种经典的统计套利策略，它通过捕捉价格偏离并等待均值回归来获利。协整分析是构建配对交易策略的核心工具，但实践中需要注意：

1. **相关性 ≠ 协整性**，必须进行严格的协整检验
2. **风险管理至关重要**，包括止损、仓位管理、配对失效监控
3. **交易成本不可忽视**，需要精细建模和优化
4. **市场环境变化可能导致协整关系断裂**，需要持续监控

随着机器学习、高频数据等技术的发展，配对交易策略也在不断演进。但无论技术如何进步，**严谨的实证分析**和**对模型风险的清醒认识**始终是成功的关键。

## 参考文献

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis"
4. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing"

---

**免责声明**：本文仅供学术交流使用，不构成投资建议。配对交易涉及模型风险、交易成本和市场风险，实盘应用需谨慎评估。
