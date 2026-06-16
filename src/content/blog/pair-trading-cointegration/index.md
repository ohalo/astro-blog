---
title: "配对交易与协整分析：市场中性套利策略实战指南"
description: "深入探讨配对交易的理论基础、协整检验方法、配对筛选技术和实盘执行策略，帮助投资者构建稳健的市场中性套利系统。"
date: "2026-06-16"
tags: ["配对交易", "协整分析", "市场中性", "统计套利"]
draft: false
---

# 配对交易与协整分析：市场中性套利策略实战指南

## 引言

在传统的主观投资和大部分的量化策略中，投资者都需要承担市场系统性风险（Beta）。然而，配对交易（Pairs Trading）作为一种经典的市场中性策略，通过对冲操作消除市场影响，仅捕捉相对价格偏离带来的收益机会。

配对交易的核心思想是：找到两个历史价格走势高度相关的资产，当它们的价格偏离正常关系时，做多被低估的资产、做空被高估的资产，等待价格回归正常后平仓获利。

本文将系统介绍配对交易的理论基础、协整分析方法、配对筛选技术以及实盘执行中的关键问题。

![配对交易原理](/images/pair-trading-cointegration/pair-trading-1.jpg)

## 配对交易的理论基础

### 什么是协整（Cointegration）？

协整是配对交易的统计学基础。简单来说，如果两个非平稳时间序列的线性组合是平稳的，那么这两个数
列就是协整的。

**直观理解：**
- 两只股票的价格序列都是非平稳的（有趋势、会漂移）
- 但它们的价差或比值序列是平稳的（围绕均值波动）
- 这意味着两者存在长期的均衡关系

**数学定义：**
如果两个序列 $X_t$ 和 $Y_t$ 都是一阶单整序列 I(1)，并且存在一个系数 $\beta$ 使得：
$$Z_t = Y_t - \beta X_t$$
是平稳序列 I(0)，则称 $X_t$ 和 $Y_t$ 是协整的。

### 平稳性的重要性

平稳序列具有以下特征：
1. **均值回归**：序列会围绕长期均值波动
2. **可预测性**：未来值可以通过历史信息预测
3. **统计性质稳定**：均值、方差不随时间变化

这正是配对交易能够盈利的核心理由：如果两只股票的价格关系（价差或比值）是平稳的，那么当偏离发生时，我们就有理由相信它会回归。

## 协整检验方法

![协整检验](/images/pair-trading-cointegration/pair-trading-2.jpg)

### 1. Engle-Granger 两步法

这是最经典的协整检验方法，分为两步：

**第一步：估计协整关系**
$$Y_t = \alpha + \beta X_t + \epsilon_t$$

**第二步：检验残差平稳性**
对残差 $\epsilon_t$ 进行 ADF (Augmented Dickey-Fuller) 检验。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger 协整检验
    
    Parameters:
    -----------
    y: Series, 第一个价格序列
    x: Series, 第二个价格序列
    verbose: bool, 是否打印详细信息
    
    Returns:
    --------
    dict: 包含协整系数、残差、ADF统计量、p值
    """
    # 第一步：OLS回归
    x_with_const = np.column_stack([np.ones(len(x)), x])
    model = OLS(y.values, x_with_const).fit()
    beta = model.params[1]
    alpha = model.params[0]
    residuals = model.resid
    
    # 第二步：ADF检验
    adf_result = adfuller(residuals, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整（5%显著性水平）
    is_cointegrated = p_value < 0.05
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger 协整检验结果")
        print("=" * 60)
        print(f"协整方程: Y = {alpha:.4f} + {beta:.4f} * X")
        print(f"\\nADF检验统计量: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"\\n临界值:")
        for key, value in critical_values.items():
            print(f"  {key}%: {value:.4f}")
        print(f"\\n结论: {'存在协整关系' if is_cointegrated else '不存在协整关系'}")
        print("=" * 60)
    
    return {
        'alpha': alpha,
        'beta': beta,
        'residuals': residuals,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'is_cointegrated': is_cointegrated,
        'critical_values': critical_values
    }

# 使用示例
# result = engle_granger_test(stock1_prices, stock2_prices)
# if result['is_cointegrated']:
#     print("可以进行配对交易！")
```

### 2. Johansen 检验

Johansen 检验是一种更强大的协整检验方法，可以同时检验多个变量之间的协整关系。

```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验
    
    Parameters:
    -----------
    data: DataFrame, 多列价格数据
    det_order: int, 确定性项的顺序
               0 - 无常数项，无趋势
               1 - 有常数项，无趋势
               2 - 有常数项，有趋势
    k_ar_diff: int, 滞后阶数
    
    Returns:
    --------
    dict: 检验结果
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    print("=" * 60)
    print("Johansen 协整检验")
    print("=" * 60)
    print(f"\\n特征值:")
    for i, val in enumerate(result.eig):
        print(f"  r={i}: {val:.4f}")
    
    print(f"\\n迹检验 (Trace Test):")
    for i in range(len(result.lr1)):
        print(f"  r<={i}: 统计量={result.lr1[i]:.4f}, "
              f"5%临界值={result.cvt[i, 1]:.4f}")
    
    print(f"\\n最大特征值检验 (Max Eigenvalue Test):")
    for i in range(len(result.lr2)):
        print(f"  r={i}: 统计量={result.lr2[i]:.4f}, "
              f"5%临界值={result.cvm[i, 1]:.4f}")
    
    return result

# 使用示例
# data = pd.DataFrame({'stock1': prices1, 'stock2': prices2, 'stock3': prices3})
# johansen_result = johansen_test(data)
```

### 3. 残差平稳性的其他检验

除了 ADF 检验，还可以使用：
- **PP检验** (Phillips-Perron)：对残差自相关更稳健
- **KPSS检验**：原假设是平稳的（与ADF相反）

```python
from statsmodels.tsa.stattools import ppoint, kpss

def comprehensive_stationarity_test(series):
    """
    综合平稳性检验
    """
    results = {}
    
    # ADF检验
    adf_result = adfuller(series, autolag='AIC')
    results['ADF'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'is_stationary': adf_result[1] < 0.05
    }
    
    # PP检验
    pp_result = ppoint(series)
    results['PP'] = {
        'statistic': pp_result[0],
        'p_value': pp_result[1],
        'is_stationary': pp_result[1] < 0.05
    }
    
    # KPSS检验（注意：原假设是平稳的）
    kpss_result = kpss(series, regression='c')
    results['KPSS'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'is_stationary': kpss_result[1] > 0.05  # 注意这里p值大于0.05才平稳
    }
    
    return results
```

## 配对筛选技术

### 1. 基于相关性的初筛

虽然高相关性不等于协整，但可以作为初筛手段。

```python
def correlation_screening(price_data, min_corr=0.7):
    """
    基于相关系数的配对初筛
    
    Parameters:
    -----------
    price_data: DataFrame, 股票价格为列，日期为行
    min_corr: float, 最小相关系数
    
    Returns:
    --------
    list: 候选配对列表
    """
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    candidate_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            stock1 = corr_matrix.columns[i]
            stock2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) >= min_corr:
                candidate_pairs.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'correlation': corr
                })
    
    # 按相关系数排序
    candidate_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return candidate_pairs

# 使用示例
# candidates = correlation_screening(price_data, min_corr=0.8)
# print(f"找到 {len(candidates)} 个候选配对")
```

### 2. 距离法（Distance Approach）

计算价格序列标准化后的欧氏距离，距离最小的作为候选配对。

```python
from scipy.spatial.distance import pdist, squareform

def distance_screening(price_data, top_n=50):
    """
    基于距离法的配对筛选
    
    Parameters:
    -----------
    price_data: DataFrame, 股票价格数据
    top_n: int, 返回距离最小的N个配对
    
    Returns:
    --------
    list: 候选配对列表
    """
    # 标准化价格（从1开始）
    normalized_prices = price_data / price_data.iloc[0]
    
    # 计算所有股票间的距离
    distances = pdist(normalized_prices.T, metric='euclidean')
    dist_matrix = squareform(distances)
    
    # 转换为DataFrame
    stocks = normalized_prices.columns
    dist_df = pd.DataFrame(dist_matrix, index=stocks, columns=stocks)
    
    # 找出距离最小的配对
    pairs = []
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            stock1 = stocks[i]
            stock2 = stocks[j]
            distance = dist_df.loc[stock1, stock2]
            pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'distance': distance
            })
    
    # 按距离排序
    pairs.sort(key=lambda x: x['distance'])
    
    return pairs[:top_n]

# 使用示例
# top_pairs = distance_screening(price_data, top_n=20)
# for pair in top_pairs[:5]:
#     print(f"{pair['stock1']} - {pair['stock2']}: 距离={pair['distance']:.4f}")
```

### 3. 协整筛选（完整流程）

结合上述方法，构建完整的配对筛选流程：

```python
class PairSelection:
    """配对筛选器"""
    
    def __init__(self, price_data, lookback_period=252):
        self.price_data = price_data
        self.lookback_period = lookback_period
        self.candidates = []
        
    def step1_correlation_filter(self, min_corr=0.7):
        """第一步：相关性过滤"""
        print("第一步：相关性过滤...")
        self.candidates = correlation_screening(self.price_data, min_corr)
        print(f"  剩余候选配对: {len(self.candidates)}")
        return self
        
    def step2_cointegration_test(self, significance_level=0.05):
        """第二步：协整检验"""
        print("第二步：协整检验...")
        cointegrated_pairs = []
        
        for pair in self.candidates:
            stock1 = pair['stock1']
            stock2 = pair['stock2']
            
            # 使用最近lookback_period天的数据
            y = self.price_data[stock1].tail(self.lookback_period)
            x = self.price_data[stock2].tail(self.lookback_period)
            
            # Engle-Granger检验
            result = engle_granger_test(y, x, verbose=False)
            
            if result['is_cointegrated']:
                pair['beta'] = result['beta']
                pair['alpha'] = result['alpha']
                pair['adf_statistic'] = result['adf_statistic']
                pair['p_value'] = result['p_value']
                cointegrated_pairs.append(pair)
        
        self.candidates = cointegrated_pairs
        print(f"  协整配对数量: {len(self.candidates)}")
        return self
        
    def step3_stationarity_check(self, z_score_window=63):
        """第三步：残差平稳性检验（滚动窗口）"""
        print("第三步：滚动平稳性检验...")
        stable_pairs = []
        
        for pair in self.candidates:
            stock1 = pair['stock1']
            stock2 = pair['stock2']
            beta = pair['beta']
            
            # 计算价差（残差）
            spread = self.price_data[stock1] - beta * self.price_data[stock2]
            
            # 滚动ADF检验
            is_stable = True
            for i in range(self.lookback_period, len(spread), 20):
                window_data = spread[i-self.lookback_period:i]
                adf_result = adfuller(window_data, autolag='AIC')
                if adf_result[1] >= 0.05:  # p值大于0.05，不平稳
                    is_stable = False
                    break
            
            if is_stable:
                stable_pairs.append(pair)
        
        self.candidates = stable_pairs
        print(f"  平稳配对数量: {len(self.candidates)}")
        return self
        
    def get_selected_pairs(self):
        """获取筛选后的配对"""
        return self.candidates

# 完整使用示例
# selector = PairSelection(price_data, lookback_period=252)
# selected_pairs = (selector
#                   .step1_correlation_filter(min_corr=0.7)
#                   .step2_cointegration_test(significance_level=0.05)
#                   .step3_stationarity_check(z_score_window=63)
#                   .get_selected_pairs())
# 
# print(f"\\n最终筛选出 {len(selected_pairs)} 个优质配对")
```

## 交易信号生成

### 1. Z-Score 信号

最常用的交易信号是基于价差的Z-Score。

```python
def calculate_z_score(spread, window=63):
    """
    计算价差的Z-Score
    
    Parameters:
    -----------
    spread: Series, 价差序列
    window: int, 滚动窗口
    
    Returns:
    --------
    Series: Z-Score序列
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    
    z_score = (spread - rolling_mean) / rolling_std
    
    return z_score

def generate_trading_signal(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """
    生成交易信号
    
    Parameters:
    -----------
    z_score: Series, Z-Score序列
    entry_threshold: float, 入场阈值
    exit_threshold: float, 出场阈值
    
    Returns:
    --------
    Series: 交易信号 (1: 做多价差, -1: 做空价差, 0: 平仓/观望)
    """
    signal = pd.Series(0, index=z_score.index)
    
    for i in range(1, len(z_score)):
        if z_score.iloc[i-1] == 0:  # 上一期无持仓
            if z_score.iloc[i] < -entry_threshold:
                signal.iloc[i] = 1  # 做多价差（做多stock1，做空stock2）
            elif z_score.iloc[i] > entry_threshold:
                signal.iloc[i] = -1  # 做空价差（做空stock1，做多stock2）
        else:  # 上一期有持仓
            if abs(z_score.iloc[i]) <= exit_threshold:
                signal.iloc[i] = 0  # 平仓
            else:
                signal.iloc[i] = z_score.iloc[i-1]  # 继续持有
    
    return signal

# 可视化信号
def plot_trading_signals(spread, z_score, signal, title="Pairs Trading Signals"):
    """绘制交易信号图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1：价差
    axes[0].plot(spread.index, spread.values, linewidth=1)
    axes[0].set_ylabel('Spread')
    axes[0].set_title(f'{title} - Spread')
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：Z-Score
    axes[1].plot(z_score.index, z_score.values, linewidth=1, label='Z-Score')
    axes[1].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
    axes[1].axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Entry (-2)')
    axes[1].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Exit (+0.5)')
    axes[1].axhline(y=-0.5, color='green', linestyle='--', alpha=0.5, label='Exit (-0.5)')
    axes[1].set_ylabel('Z-Score')
    axes[1].set_title('Z-Score with Thresholds')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：交易信号
    axes[2].plot(signal.index, signal.values, linewidth=2, marker='o', markersize=4)
    axes[2].set_ylabel('Signal')
    axes[2].set_title('Trading Signal (1: Long Spread, -1: Short Spread, 0: Flat)')
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 使用示例
# spread = stock1_prices - beta * stock2_prices
# z_score = calculate_z_score(spread, window=63)
# signal = generate_trading_signal(z_score, entry_threshold=2.0, exit_threshold=0.5)
# fig = plot_trading_signals(spread, z_score, signal)
```

### 2. 卡尔曼滤波动态对冲比

传统的OLS估计对冲比是静态的，卡尔曼滤波可以实现动态估计。

```python
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(y, x):
    """
    使用卡尔曼滤波动态估计对冲比
    
    Parameters:
    -----------
    y: Series, 第一个价格序列
    x: Series, 第二个价格序列
    
    Returns:
    --------
    tuple: (动态beta序列, 动态alpha序列)
    """
    # 准备观测矩阵
    X = np.column_stack([np.ones(len(x)), x])
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=X,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.01
    )
    
    # 滤波
    state_means, state_covs = kf.filter(y.values)
    
    # 提取alpha和beta
    alpha_dynamic = state_means[:, 0]
    beta_dynamic = state_means[:, 1]
    
    return beta_dynamic, alpha_dynamic

# 计算动态价差
# beta_dynamic, alpha_dynamic = kalman_filter_hedge_ratio(stock1_prices, stock2_prices)
# spread_dynamic = stock1_prices - (alpha_dynamic + beta_dynamic * stock2_prices)
```

## 风险控制与资金管理

### 1. 止损策略

配对交易虽然理论上是市场中性，但仍需要严格的风险控制。

```python
def implement_stop_loss(signal, spread, z_score, 
                       stop_loss_z=3.0, max_holding_days=20):
    """
    实施止损策略
    
    Parameters:
    -----------
    signal: Series, 原始交易信号
    spread: Series, 价差序列
    z_score: Series, Z-Score序列
    stop_loss_z: float, Z-Score止损阈值
    max_holding_days: int, 最大持有天数
    
    Returns:
    --------
    Series: 修正后的交易信号
    """
    new_signal = signal.copy()
    entry_date = None
    
    for i in range(len(signal)):
        if new_signal.iloc[i] != 0 and entry_date is None:
            # 新开仓
            entry_date = spread.index[i]
            
        elif new_signal.iloc[i] != 0 and entry_date is not None:
            # 已有持仓
            holding_days = (spread.index[i] - entry_date).days
            
            # 止损条件1：Z-Score超过止损线
            if abs(z_score.iloc[i]) >= stop_loss_z:
                print(f"⚠️ 止损触发 @ {spread.index[i]}: Z-Score={z_score.iloc[i]:.2f}")
                new_signal.iloc[i] = 0
                entry_date = None
                
            # 止损条件2：持有时间过长
            elif holding_days >= max_holding_days:
                print(f"⚠️ 时间止损触发 @ {spread.index[i]}: 持有{holding_days}天")
                new_signal.iloc[i] = 0
                entry_date = None
                
        else:
            entry_date = None
    
    return new_signal
```

### 2. 仓位管理

根据Z-Score的绝对值动态调整仓位大小。

```python
def dynamic_position_sizing(z_score, max_position=1.0, scaling='linear'):
    """
    动态仓位管理
    
    Parameters:
    -----------
    z_score: Series, Z-Score序列
    max_position: float, 最大仓位（1.0表示满仓）
    scaling: str, 缩放方式 ('linear', 'square', 'step')
    
    Returns:
    --------
    Series: 仓位大小序列
    """
    abs_z = abs(z_score)
    
    if scaling == 'linear':
        # 线性缩放：Z-Score越大，仓位越大
        position = abs_z / abs_z.max() * max_position
        
    elif scaling == 'square':
        # 平方缩放：加速加仓
        position = (abs_z ** 2) / (abs_z.max() ** 2) * max_position
        
    elif scaling == 'step':
        # 分段仓位
        position = pd.Series(0.3 * max_position, index=z_score.index)
        position[abs_z >= 1.5] = 0.6 * max_position
        position[abs_z >= 2.0] = 1.0 * max_position
        
    else:
        raise ValueError("scaling must be 'linear', 'square', or 'step'")
    
    return position

# 使用示例
# position_size = dynamic_position_sizing(z_score, max_position=1.0, scaling='linear')
```

## 实证案例分析

### 案例：可口可乐 vs 百事可乐

这两家公司业务高度相似，是配对交易的经典案例。

```python
# 假设我们有KO（可口可乐）和PEP（百事）的价格数据
# ko_prices, pep_prices

# 1. 协整检验
result = engle_granger_test(ko_prices, pep_prices)
print(f"协整关系: KO = {result['alpha']:.2f} + {result['beta']:.2f} * PEP")

# 2. 计算价差和Z-Score
spread = ko_prices - result['beta'] * pep_prices
z_score = calculate_z_score(spread, window=63)

# 3. 生成信号
signal = generate_trading_signal(z_score, entry_threshold=2.0, exit_threshold=0.5)

# 4. 计算收益
# 假设我们做多1美元价差时，买入1股KO，卖出beta股PEP
returns = pd.DataFrame({
    'ko_return': ko_prices.pct_change(),
    'pep_return': pep_prices.pct_change()
})

strategy_return = (signal.shift(1) * 
                   (returns['ko_return'] - result['beta'] * returns['pep_return']))

cumulative_return = (1 + strategy_return).cumprod()

print(f"策略累计收益: {cumulative_return.iloc[-1]:.2%}")
print(f"夏普比率: {strategy_return.mean() / strategy_return.std() * np.sqrt(252):.2f}")
```

## 实盘注意事项

### 1. 交易成本

配对交易通常换手率较高，必须考虑交易成本。

- **佣金**：选择低佣金券商
- **买卖价差**：避免流动性差的股票
- **滑点**：使用限价单，避免市价单

### 2. 执行延迟

- **实时数据**：使用高质量实时行情
- **订单执行**：使用算法交易或DMA（直接市场准入）
- **风险控制**：设置实时监控和自动止损

### 3. 配对失效

市场环境变化可能导致历史协整关系失效。

- **定期重新检验**：每月重新运行协整检验
- **多配对分散**：同时交易多个不相关的配对
- **动态调整**：使用卡尔曼滤波等动态模型

### 4. 监管要求

- **卖空限制**：某些市场不允许卖空，需要改造策略
- **持仓披露**：大额持仓需要披露
- **税务处理**：配对交易的税务处理可能较复杂

## 结论

配对交易是一种经典而有效的市场中性策略，通过对冲操作消除系统性风险，捕捉相对价格偏离带来的收益。成功的配对交易需要：

1. **严谨的统计学基础**：理解协整、平稳性等概念
2. **系统的筛选流程**：从相关性过滤到协整检验
3. **合理的信号处理**：Z-Score、卡尔曼滤波等方法
4. **严格的风险控制**：止损、仓位管理、分散投资
5. **高效的执行系统**：低延迟、低成本的交易执行

随着市场有效性的提升，简单的配对交易alpha在衰减。未来的发展方向包括：
- **高频配对交易**：利用分钟级或秒级数据
- **机器学习增强**：使用LSTM等模型预测价差
- **多因子配对**：结合基本面、技术面因子
- **跨市场配对**：在不同交易所或不同资产类别间寻找配对

配对交易之路任重道远，但只要坚持科学的方法和严格的风险管理，就能在市场的波澜中稳健前行。

---

**参考文献：**
1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies.
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." Wiley.
4. Chan, E. P. (2013). "Algorithmic Trading: Winning Strategies and Their Rationale." Wiley.
