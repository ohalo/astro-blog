---
title: "配对交易与协整分析：统计套利的核心技术"
description: "深入讲解配对交易的理论基础、协整检验方法、交易信号构建和风险管理，提供完整的Python实现代码，帮助读者掌握统计套利的核心技术。"
pubDate: 2026-06-16
tags: ["配对交易", "协整分析", "统计套利", "均值回归"]
category: "统计套利"
difficulty: "进阶"
featured: false
---

# 配对交易与协整分析：统计套利的核心技术

## 引言

在量化投资的世界里，有一种策略不需要预测市场的整体方向，也不需要判断宏观经济的走势，它只关注**相对价值**——这就是**配对交易（Pairs Trading）**。

配对交易是一种典型的**市场中性策略**，通过同时买入一只股票并卖出另一只高度相关的股票，从两者的价差回归中获利。这种策略的核心在于**协整分析（Cointegration Analysis）**——一种能够识别长期均衡关系的统计方法。

本文将深入讲解配对交易的理论基础、协整检验方法、交易信号构建和风险管理，并提供完整的Python实现代码。

## 配对交易的基本原理

### 什么是配对交易？

配对交易属于**统计套利（Statistical Arbitrage）**的一种。它的基本思想是：

1. 找到两只历史价格走势高度相关、但偶尔会出现价差偏离的股票
2. 当价差（Spread）偏离历史均值时，假设它会**均值回归（Mean Reversion）**
3. 做多价格偏低的股票，做空价格偏高的股票
4. 等待价差回归均值后平仓，赚取无风险利润

**举例说明：**
假设可口可乐（KO）和百事可乐（PEP）的历史价格比率为1.2（KO价格/PEP价格）。如果某天这个比率突然上升到1.5，我们认为价差过高，应该：
- 做空可口可乐（预期价格相对下跌）
- 做多百事可乐（预期价格相对上涨）
- 当比率回归1.2时平仓

### 配对交易的优势

1. **市场中性：** 无论大盘涨跌，只要价差回归就能盈利
2. **低风险：** 对冲了系统性风险（Beta）
3. **稳定收益：** 在市场震荡期表现优异
4. **可量化：** 完全基于统计模型，易于回测和优化

### 配对交易的挑战

1. **配对选择：** 如何找到真正具有协整关系的股票对？
2. **参数优化：** 何时开仓？何时平仓？止损设在哪里？
3. **模型失效：** 当股票的基本面发生变化时，历史关系可能破裂
4. **交易成本：** 频繁交易可能导致手续费侵蚀利润

## 协整分析理论基础

### 平稳性（Stationarity）

在介绍协整之前，必须先理解**平稳性**。

**定义：** 一个时间序列是平稳的，如果它满足：
1. 均值恒定
2. 方差恒定
3. 自协方差只依赖于时间差，不依赖于时间本身

**为什么重要？** 只有平稳的时间序列，我们才能使用标准的统计方法（如t检验、F检验）进行推断。非平稳序列会导致"伪回归（Spurious Regression）"问题。

**检验方法：ADF检验（Augmented Dickey-Fuller Test）**

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, max_lag=None, regression='c'):
    """
    ADF检验平稳性
    
    Parameters:
    -----------
    timeseries : Series
        时间序列数据
    max_lag : int
        最大滞后阶数（None表示自动选择）
    regression : str
        回归类型：'c'（常数）、'ct'（常数+趋势）、'ctt'（常数+趋势+二次趋势）
    
    Returns:
    --------
    result : dict
        ADF检验结果
    """
    result = adfuller(timeseries, maxlag=max_lag, regression=regression)
    
    output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'is_stationary': result[1] < 0.05  # p值小于0.05，拒绝原假设（非平稳）
    }
    
    return output

# 使用示例
# stock_price = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)['AAPL']
# result = test_stationarity(stock_price)
# print(f"ADF统计量: {result['ADF Statistic']:.4f}")
# print(f"p-value: {result['p-value']:.4f}")
# print(f"是否平稳: {result['is_stationary']}")
```

**解读：**
- **原假设（H0）：** 序列是非平稳的（存在单位根）
- **备择假设（H1）：** 序列是平稳的（不存在单位根）
- **判断标准：** 如果p-value < 0.05，拒绝原假设，认为序列平稳

**结论：** 大多数股票价格序列都是**非平稳的**（随机游走），但它们的**价差（线性组合）可能是平稳的**——这就是协整关系。

### 协整（Cointegration）

**定义：** 如果两个或多个非平稳时间序列的某种线性组合是平稳的，那么这些序列之间存在协整关系。

**数学表达：**
假设有两个非平稳序列 \(X_t\) 和 \(Y_t\)，如果存在一个系数 \(\beta\)，使得：

\[
Z_t = Y_t - \beta X_t
\]

是平稳的，那么 \(X_t\) 和 \(Y_t\) 是协整的。

**经济意义：** 协整关系意味着两个序列之间存在**长期均衡关系**。虽然短期内它们可能偏离，但长期来看，这种偏离会被修正（均值回归）。

### 协整检验方法

#### 1. Engle-Granger两步法

**步骤：**
1. 用OLS回归估计长期均衡关系：\(Y_t = \alpha + \beta X_t + \epsilon_t\)
2. 对残差 \(\epsilon_t\) 进行ADF检验

**Python实现：**
```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def engle_granger_test(y, x, max_lag=None):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y : Series
        因变量（被解释变量）
    x : Series
        自变量（解释变量）
    max_lag : int
        ADF检验的最大滞后阶数
    
    Returns:
    --------
    result : dict
        协整检验结果
    """
    # 步骤1：OLS回归
    X_with_const = add_constant(x)
    model = OLS(y, X_with_const).fit()
    beta = model.params[1]
    residuals = model.resid
    
    # 步骤2：对残差进行ADF检验
    adf_result = test_stationarity(residuals, max_lag=max_lag)
    
    # 计算协整得分（用于排序）
    cointegration_score = abs(adf_result['ADF Statistic'])
    
    result = {
        'beta': beta,
        'alpha': model.params[0],
        'residuals': residuals,
        'adf_statistic': adf_result['ADF Statistic'],
        'p_value': adf_result['p-value'],
        'is_cointegrated': adf_result['is_stationary'],
        'cointegration_score': cointegration_score
    }
    
    return result

# 使用示例
# stock1 = pd.read_csv('stock1.csv')['Close']
# stock2 = pd.read_csv('stock2.csv')['Close']
# result = engle_granger_test(stock1, stock2)
# if result['is_cointegrated']:
#     print(f"存在协整关系！beta={result['beta']:.4f}")
```

#### 2. Johansen检验

**优势：** Engle-Granger方法只能检验两个序列之间的协整关系，而Johansen检验可以处理**多个序列**的情况。

**Python实现：**
```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    data : DataFrame
        多个时间序列（每行是一个时间点，每列是一个序列）
    det_order : int
        确定性项的阶数：
        -1: 无确定性项
        0: 常数（截距）
        1: 常数 + 趋势
    k_ar_diff : int
        VAR模型的最优滞后阶数
    
    Returns:
    --------
    result : dict
        Johansen检验结果
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    # 提取特征值（Eigenvalues）
    eigenvalues = result.eig
    
    # 提取迹统计量（Trace Statistic）和最大特征值统计量
    trace_stat = result.lr1
    max_eig_stat = result.lr2
    
    # 临界值（5%显著性水平）
    critical_values = result.cvt  # 迹统计量的临界值
    critical_values_max = result.cvm  # 最大特征值统计量的临界值
    
    output = {
        'eigenvalues': eigenvalues,
        'trace_statistic': trace_stat,
        'max_eigenvalue_statistic': max_eig_stat,
        'critical_values': critical_values,
        'num_cointegrating_vectors': np.sum(trace_stat > critical_values[:, 1])  # 5%临界值
    }
    
    return output
```

## 配对选择的实战方法

### 方法1：基于行业分类的初筛

**逻辑：** 同一行业的股票更可能具有协整关系（因为它们面临相同的宏观经济因素）。

**步骤：**
```python
import yfinance as yf
from collections import defaultdict

def find_pairs_by_industry(industry, num_stocks=20):
    """
    根据行业分类寻找潜在的配对
    
    Parameters:
    -----------
    industry : str
        行业名称（如'Technology', 'Financial Services'）
    num_stocks : int
        每个行业选择的股票数量
    
    Returns:
    --------
    candidate_pairs : list
        候选配对列表
    """
    # 步骤1：获取行业内的股票列表
    # 这里可以使用yfinance、tushare等数据源
    industry_stocks = get_stocks_by_industry(industry)[:num_stocks]
    
    # 步骤2：下载价格数据
    price_data = download_price_data(industry_stocks, start='2020-01-01')
    
    # 步骤3：计算相关性矩阵
    returns = price_data.pct_change().dropna()
    correlation_matrix = returns.corr()
    
    # 步骤4：选择高相关性的配对（相关性 > 0.7）
    candidate_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if corr > 0.7:
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                candidate_pairs.append((stock1, stock2, corr))
    
    # 按相关性排序
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return candidate_pairs
```

### 方法2：距离法（Distance Method）

**逻辑：** 计算所有股票对之间的"距离"（如价格差的累积平方和），距离越小，越可能具有协整关系。

**公式：**
\[
\text{Distance}_{i,j} = \sum_{t=1}^{T} (P_{i,t} - P_{j,t})^2
\]

**Python实现：**
```python
def calculate_distance(stock_data):
    """
    计算所有股票对之间的距离
    
    Parameters:
    -----------
    stock_data : DataFrame
        股票价格数据（每行是一个时间点，每列是一只股票）
    
    Returns:
    --------
    distances : DataFrame
        股票对之间的距离矩阵
    """
    stocks = stock_data.columns
    n = len(stocks)
    
    distances = pd.DataFrame(np.zeros((n, n)), index=stocks, columns=stocks)
    
    for i in range(n):
        for j in range(i+1, n):
            # 标准化价格（除以初始价格）
            price1_norm = stock_data.iloc[:, i] / stock_data.iloc[0, i]
            price2_norm = stock_data.iloc[:, j] / stock_data.iloc[0, j]
            
            # 计算距离（SSD：Sum of Squared Differences）
            ssd = np.sum((price1_norm - price2_norm) ** 2)
            distances.iloc[i, j] = ssd
            distances.iloc[j, i] = ssd  # 对称矩阵
    
    return distances

# 选择距离最小的Top N配对
def select_top_pairs(distances, top_n=50):
    """
    选择距离最小的Top N配对
    """
    # 将距离矩阵转换为长格式
    pairs = []
    for i in range(len(distances.index)):
        for j in range(i+1, len(distances.columns)):
            stock1 = distances.index[i]
            stock2 = distances.columns[j]
            distance = distances.iloc[i, j]
            pairs.append((stock1, stock2, distance))
    
    # 按距离排序
    pairs.sort(key=lambda x: x[2])
    
    return pairs[:top_n]
```

### 方法3：协整得分排序

**逻辑：** 对所有候选配对进行协整检验，按协整得分（ADF统计量）排序。

**完整流程：**
```python
def screening_cointegrated_pairs(stock_universe, start_date, end_date, top_n=100):
    """
    筛选具有协整关系的股票对
    
    Parameters:
    -----------
    stock_universe : list
        股票代码列表
    start_date, end_date : str
        数据的起始和结束日期
    top_n : int
        返回的配对数量
    
    Returns:
    --------
    cointegrated_pairs : list
        协整配对列表，每个元素为(stock1, stock2, beta, adf_statistic)
    """
    # 步骤1：下载数据
    print("正在下载价格数据...")
    price_data = yf.download(stock_universe, start=start_date, end=end_date)['Adj Close']
    price_data = price_data.dropna(axis=1)  # 删除有缺失值的股票
    
    # 步骤2：遍历所有配对
    print("正在进行协整检验...")
    pairs_results = []
    
    stocks = price_data.columns
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            stock1 = stocks[i]
            stock2 = stocks[j]
            
            # Engle-Granger检验
            result = engle_granger_test(price_data[stock1], price_data[stock2])
            
            if result['is_cointegrated']:
                pairs_results.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'beta': result['beta'],
                    'alpha': result['alpha'],
                    'adf_statistic': result['adf_statistic'],
                    'p_value': result['p_value']
                })
    
    # 步骤3：按ADF统计量排序（越小越好）
    pairs_results.sort(key=lambda x: x['adf_statistic'])
    
    # 步骤4：返回Top N配对
    cointegrated_pairs = [
        (p['stock1'], p['stock2'], p['beta'], p['adf_statistic'])
        for p in pairs_results[:top_n]
    ]
    
    print(f"找到 {len(cointegrated_pairs)} 个协整配对")
    return cointegrated_pairs
```

## 交易信号的构建

### 1. 价差的计算

假设我们已经找到了协整配对（stock1, stock2），且协整方程为：

\[
\text{Spread}_t = P_{1,t} - \beta P_{2,t}
\]

**Python实现：**
```python
def calculate_spread(price1, price2, beta):
    """
    计算配对交易的价差
    
    Parameters:
    -----------
    price1, price2 : Series
        两只股票的价格序列
    beta : float
        协整系数
    
    Returns:
    --------
    spread : Series
        价差序列
    """
    spread = price1 - beta * price2
    return spread
```

### 2. 价差的平稳性验证

在构建交易信号之前，必须验证价差是平稳的（即存在均值回归特性）。

```python
def verify_mean_reversion(spread, significance_level=0.05):
    """
    验证价差是否具有均值回归特性
    
    Parameters:
    -----------
    spread : Series
        价差序列
    significance_level : float
        显著性水平
    
    Returns:
    --------
    is_mean_reverting : bool
        是否具有均值回归特性
    """
    # 方法1：ADF检验
    adf_result = test_stationarity(spread)
    
    # 方法2：Hurst指数
    hurst = calculate_hurst_exponent(spread)
    
    # 方法3：半衰期（Half-life）
    half_life = calculate_half_life(spread)
    
    print(f"ADF Statistic: {adf_result['ADF Statistic']:.4f}")
    print(f"p-value: {adf_result['p-value']:.4f}")
    print(f"Hurst Exponent: {hurst:.4f} (应该 < 0.5)")
    print(f"Half-life: {half_life:.2f} 天")
    
    # 判断标准
    is_mean_reverting = (
        adf_result['p-value'] < significance_level and  # ADF检验显著
        hurst < 0.5 and  # Hurst指数 < 0.5（均值回归）
        half_life > 0 and half_life < 252  # 半衰期在1天到1年之间
    )
    
    return is_mean_reverting

def calculate_hurst_exponent(timeseries, max_lag=100):
    """
    计算Hurst指数
    H < 0.5: 均值回归（反持久性）
    H = 0.5: 随机游走
    H > 0.5: 趋势延续（持久性）
    """
    lags = range(2, min(max_lag, len(timeseries)//2))
    tau = [np.std(np.subtract(timeseries[lag:], timeseries[:-lag])) for lag in lags]
    
    # 拟合 log(lag) vs log(tau)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Hurst指数 = 斜率
    hurst = poly[0]
    
    return hurst

def calculate_half_life(spread):
    """
    计算价差的半衰期（均值回归的速度）
    """
    # 使用OLS回归：Δspread_t = α + β * spread_{t-1} + ε_t
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # 对齐数据
    data = pd.concat([spread_lag, spread_diff], axis=1).dropna()
    X = add_constant(data.iloc[:, 0])
    y = data.iloc[:, 1]
    
    model = OLS(y, X).fit()
    beta = model.params[1]
    
    # 半衰期 = -log(2) / log(1 + beta)
    half_life = -np.log(2) / np.log(1 + beta)
    
    return half_life
```

### 3. 交易信号的生成

**基本思路：** 当价差偏离均值超过N个标准差时开仓，当价差回归均值时平仓。

```python
class PairsTradingStrategy:
    """
    配对交易策略类
    """
    def __init__(self, stock1, stock2, beta, entry_zscore=2.0, exit_zscore=0.0, 
                 stop_loss_zscore=3.0, lookback_period=252):
        """
        初始化策略参数
        
        Parameters:
        -----------
        stock1, stock2 : str
            股票代码
        beta : float
            协整系数
        entry_zscore : float
            开仓阈值（标准差的倍数）
        exit_zscore : float
            平仓阈值
        stop_loss_zscore : float
            止损阈值
        lookback_period : int
            计算均值和标准差的滚动窗口
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.beta = beta
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.lookback_period = lookback_period
        
        self.position = 0  # 持仓状态：0（空仓）、1（做多价差）、-1（做空价差）
        self.entry_spread = None  # 开仓时的价差
        
    def generate_signals(self, price_data):
        """
        生成交易信号
        
        Parameters:
        -----------
        price_data : DataFrame
            包含stock1和stock2价格的数据框
        
        Returns:
        --------
        signals : DataFrame
            交易信号（1：做多价差，-1：做空价差，0：平仓）
        """
        # 计算价差
        spread = price_data[self.stock1] - self.beta * price_data[self.stock2]
        
        # 计算价差的滚动均值和标准差
        spread_mean = spread.rolling(window=self.lookback_period).mean()
        spread_std = spread.rolling(window=self.lookback_period).std()
        
        # 计算z-score
        zscore = (spread - spread_mean) / spread_std
        
        # 初始化信号
        signals = pd.Series(0, index=price_data.index)
        
        for i in range(1, len(zscore)):
            today = zscore.index[i]
            yesterday = zscore.index[i-1]
            
            # 当前z-score
            z_today = zscore.loc[today]
            z_yesterday = zscore.loc[yesterday]
            
            # 空仓状态
            if self.position == 0:
                # 价差偏高 → 做空价差（做空stock1，做多stock2）
                if z_today > self.entry_zscore:
                    signals.loc[today] = -1
                    self.position = -1
                    self.entry_spread = spread.loc[today]
                
                # 价差偏低 → 做多价差（做多stock1，做空stock2）
                elif z_today < -self.entry_zscore:
                    signals.loc[today] = 1
                    self.position = 1
                    self.entry_spread = spread.loc[today]
            
            # 持多仓（做多价差）
            elif self.position == 1:
                # 价差回归 → 平仓
                if abs(z_today) < self.exit_zscore:
                    signals.loc[today] = 0  # 平仓信号
                    self.position = 0
                    self.entry_spread = None
                
                # 止损：价差继续扩大
                elif z_today < -self.stop_loss_zscore:
                    signals.loc[today] = 0  # 止损平仓
                    self.position = 0
                    self.entry_spread = None
            
            # 持空仓（做空价差）
            elif self.position == -1:
                # 价差回归 → 平仓
                if abs(z_today) < self.exit_zscore:
                    signals.loc[today] = 0  # 平仓信号
                    self.position = 0
                    self.entry_spread = None
                
                # 止损：价差继续扩大
                elif z_today > self.stop_loss_zscore:
                    signals.loc[today] = 0  # 止损平仓
                    self.position = 0
                    self.entry_spread = None
        
        return signals, zscore
```

### 4. 策略回测

```python
def backtest_pairs_strategy(price_data, signals, beta, initial_capital=100000):
    """
    回测配对交易策略
    
    Parameters:
    -----------
    price_data : DataFrame
        价格数据
    signals : Series
        交易信号
    beta : float
        协整系数
    initial_capital : float
        初始资金
    
    Returns:
    --------
    portfolio_value : Series
        组合价值的时间序列
    performance : dict
        绩效指标
    """
    # 初始化
    cash = initial_capital
    position = 0  # 持仓数量（单位为stock1的股数）
    portfolio_value = pd.Series(index=price_data.index, dtype=float)
    
    stock1_prices = price_data.iloc[:, 0]
    stock2_prices = price_data.iloc[:, 1]
    
    for i in range(len(signals)):
        date = signals.index[i]
        signal = signals.iloc[i]
        
        # 开仓
        if signal != 0 and position == 0:
            # 假设我们用一半资金买入stock1，一半资金买入stock2（根据beta调整）
            capital_per_stock = cash / 2
            
            if signal == 1:  # 做多价差
                # 买入stock1
                shares1 = capital_per_stock / stock1_prices.loc[date]
                # 卖出stock2（空头）
                shares2 = -capital_per_stock / stock2_prices.loc[date] * beta
                
                position = shares1  # 记录持仓
                cash -= (shares1 * stock1_prices.loc[date] + shares2 * stock2_prices.loc[date])
            
            elif signal == -1:  # 做空价差
                # 卖出stock1（空头）
                shares1 = -capital_per_stock / stock1_prices.loc[date]
                # 买入stock2
                shares2 = capital_per_stock / stock2_prices.loc[date] * beta
                
                position = shares1
                cash -= (shares1 * stock1_prices.loc[date] + shares2 * stock2_prices.loc[date])
        
        # 平仓
        elif signal == 0 and position != 0:
            # 平掉所有仓位
            cash += position * stock1_prices.loc[date]
            cash += (-position * beta) * stock2_prices.loc[date]
            position = 0
        
        # 计算当日组合价值
        portfolio_value.loc[date] = cash + position * stock1_prices.loc[date] + \
                                     (-position * beta) * stock2_prices.loc[date]
    
    # 计算绩效指标
    returns = portfolio_value.pct_change().dropna()
    
    performance = {
        'total_return': (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100,
        'annual_return': returns.mean() * 252 * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(portfolio_value),
        'win_rate': (returns > 0).sum() / len(returns)
    }
    
    return portfolio_value, performance

def calculate_max_drawdown(portfolio_value):
    """
    计算最大回撤
    """
    cumulative = portfolio_value / portfolio_value.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return max_drawdown
```

## 风险管理与实战技巧

### 1. 模型失效的检测

**问题：** 协整关系可能随时间破裂（结构断裂）。

**检测方法：**
```python
def detect_structural_break(spread, test_period=252):
    """
    检测协整关系的结构断裂
    
    Parameters:
    -----------
    spread : Series
        价差序列
    test_period : int
        滚动检验的窗口
    
    Returns:
    --------
    break_dates : list
        检测到结构断裂的日期
    """
    break_dates = []
    
    for i in range(test_period, len(sppread) - test_period):
        # 前半段
        spread_1 = spread.iloc[i-test_period:i]
        # 后半段
        spread_2 = spread.iloc[i:i+test_period]
        
        # 分别进行ADF检验
        result_1 = test_stationarity(spread_1)
        result_2 = test_stationarity(spread_2)
        
        # 如果前半段平稳，后半段不平稳 → 结构断裂
        if result_1['is_stationary'] and not result_2['is_stationary']:
            break_dates.append(spread.index[i])
    
    return break_dates
```

### 2. 动态对冲比率

**问题：** 协整系数β可能不是常数，会随时间变化。

**解决方案：** 使用滚动窗口或卡尔曼滤波动态估计β。

```python
from pykalman import KalmanFilter

def dynamic_beta_estimation(stock1_prices, stock2_prices):
    """
    使用卡尔曼滤波动态估计协整系数β
    
    Returns:
    --------
    beta_dynamic : Series
        动态β序列
    """
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=[1],  # 状态转移矩阵（假设β随机游走）
        observation_matrices=stock2_prices.values.reshape(-1, 1),
        initial_state_mean=1.0,
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=0.01  # 假设β变化缓慢
    )
    
    # 滤波
    state_means, _ = kf.filter(stock1_prices.values)
    
    beta_dynamic = pd.Series(state_means.flatten(), index=stock1_prices.index)
    
    return beta_dynamic
```

### 3. 交易成本考虑

**实战中必须考虑：**
- 佣金（Commission）
- 买卖价差（Bid-Ask Spread）
- 市场冲击（Market Impact）

**优化方法：**
```python
def calculate_transaction_cost(shares, price, commission_rate=0.001, bid_ask_spread=0.01):
    """
    计算交易成本
    
    Parameters:
    -----------
    shares : float
        交易股数（正数为买入，负数为卖出）
    price : float
        股票价格
    commission_rate : float
        佣金率
    bid_ask_spread : float
        买卖价差（以价格的比例表示）
    
    Returns:
    --------
    cost : float
        交易成本
    """
    # 佣金
    commission = abs(shares) * price * commission_rate
    
    # 买卖价差成本（买入时付高价，卖出时收低价）
    if shares > 0:  # 买入
        spread_cost = shares * price * bid_ask_spread / 2
    else:  # 卖出
        spread_cost = abs(shares) * price * bid_ask_spread / 2
    
    total_cost = commission + spread_cost
    
    return total_cost
```

## 完整实战案例

### 案例：可口可乐（KO） vs 百事可乐（PEP）

```python
# 步骤1：下载数据
tickers = ['KO', 'PEP']
start_date = '2018-01-01'
end_date = '2024-01-01'

price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 步骤2：协整检验
result = engle_granger_test(price_data['KO'], price_data['PEP'])
print(f"beta: {result['beta']:.4f}")
print(f"ADF Statistic: {result['adf_statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"是否协整: {result['is_cointegrated']}")

# 步骤3：计算价差
beta = result['beta']
spread = calculate_spread(price_data['KO'], price_data['PEP'], beta)

# 步骤4：验证均值回归
is_valid = verify_mean_reversion(spread)
print(f"是否均值回归: {is_valid}")

# 步骤5：生成交易信号
strategy = PairsTradingStrategy('KO', 'PEP', beta, entry_zscore=2.0, exit_zscore=0.5)
signals, zscore = strategy.generate_signals(price_data)

# 步骤6：回测
portfolio_value, performance = backtest_pairs_strategy(
    price_data, signals, beta, initial_capital=100000
)

print("\n=== 策略绩效 ===")
for key, value in performance.items():
    print(f"{key}: {value:.2f}")

# 步骤7：可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 子图1：价格走势
ax1 = axes[0]
ax1.plot(price_data.index, price_data['KO'], label='KO')
ax1.plot(price_data.index, price_data['PEP'], label='PEP')
ax1.set_title('Stock Prices')
ax1.legend()

# 子图2：价差和z-score
ax2 = axes[1]
ax2.plot(spread.index, spread, label='Spread')
ax2.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
ax2.set_title('Spread')
ax2.legend()

ax3 = axes[2]
ax3.plot(zscore.index, zscore, label='Z-score')
ax3.axhline(2, color='red', linestyle='--', label='Entry Threshold')
ax3.axhline(-2, color='red', linestyle='--')
ax3.axhline(0, color='green', linestyle='--', label='Exit Threshold')
ax3.set_title('Z-score')
ax3.legend()

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pairs_trading_example.png')
plt.show()
```

## 结论

配对交易是一种经典而有效的统计套利策略，其核心在于**协整分析**。通过本文的介绍，读者应该能够：

1. **理解理论基础：** 平稳性、协整关系、均值回归
2. **掌握检验方法：** ADF检验、Engle-Granger检验、Johansen检验
3. **实现配对选择：** 基于行业、距离法、协整得分
4. **构建交易信号：** z-score阈值、动态对冲、风险管理
5. **进行策略回测：** 计算绩效指标、优化参数

**关键要点：**
- 协整关系是配对交易的基石，必须经过严格的统计检验
- 交易信号的设计需要平衡收益和风险（入场、出场、止损）
- 实战中必须考虑交易成本、模型失效等现实因素
- 动态调整（如卡尔曼滤波）可以提升策略的稳健性

未来，随着机器学习技术的发展，配对交易也在不断进化。例如，使用深度学习预测价差的非线性动态、利用高频数据捕捉微观结构套利机会等。但这些先进方法的基础，仍然是本文介绍的协整分析框架。

---

**参考文献：**
1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
2. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies.
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." Wiley.

**免责声明：** 本文仅供参考，不构成投资建议。配对交易存在风险，历史表现不代表未来收益。在实际交易前，请进行充分的回测和风险评估。
