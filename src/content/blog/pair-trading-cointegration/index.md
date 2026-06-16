---
title: "配对交易与协整分析：市场中性策略的理论与实践"
description: "深入探讨配对交易策略的理论基础、协整检验方法、实际操作步骤和风险管理，帮助投资者构建稳健的市场中性策略。"
date: "2026-06-16"
tags: ["配对交易", "协整分析", "市场中性", "统计套利"]
categories: ["量化交易"]
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：市场中性策略的理论与实践

## 引言

在传统的主观交易和单纯的趋势跟踪之外，统计套利（Statistical Arbitrage）为量化投资者提供了另一种获取超额收益的思路。其中，**配对交易（Pairs Trading）**作为最经典的市场中性策略之一，通过对冲掉市场系统性风险，捕捉两个高度相关资产之间的暂时性价格偏离。

从1980年代摩根士丹利的量化团队首次系统化应用，到如今机器学习赋能的智能配对选择，这一策略已经演进了近半个世纪。本文将深入探讨：

- 配对交易的经济学逻辑与数学基础
- 协整关系检验的多种方法比较
- 从标的筛选到信号生成的完整流程
- 实战中的关键问题与解决方案
- Python实现与回测框架

## 一、配对交易的核心逻辑

### 1.1 基本思想

配对交易基于一个朴素而强大的假设：**历史上价格走势高度相关的两只股票，其价差（Spread）具有均值回归特性**。

策略流程：
1. **识别配对**：找到具有经济关联的两只股票（如同行业、同产业链、替代关系等）
2. **检验协整**：验证价格序列是否存在长期均衡关系
3. **计算价差**：构建线性组合，使残差平稳
4. **设定阈值**：根据历史波动设定开平仓信号
5. **执行交易**：做多低估资产、做空高估资产
6. **动态调仓**：定期重新估计参数，适应市场变化

### 1.2 为什么有效？

配对交易之所以能够持续获利，根源在于：

1. **经济联系**：同一行业的公司面临相似的宏观环境、监管政策、技术变革
2. **投资者行为**：过度反应、追涨杀跌导致短期价格偏离基本面
3. **套利机制**：市场参与者会逐步发现并纠正定价错误
4. **风险分散**：多空对冲抵消市场风险（Beta≈0）

### 1.3 策略分类

根据标的选择方法，配对交易可分为：

| 类型 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| 主观筛选 | 基于行业知识、产业链关系人工选择 | 逻辑清晰、可解释性强 | 覆盖面窄、主观性强 |
| 统计筛选 | 距离法、相关性分析、聚类算法 | 客观、可规模化 | 可能选出无经济逻辑的配对 |
| 机器学习 | 随机森林、神经网络辅助筛选 | 捕捉非线性关系 | 黑箱、过拟合风险 |
| 混合方法 | 统计筛选+基本面过滤 | 兼顾客观性与逻辑性 | 实现复杂度高 |

## 二、协整理论基础

### 2.1 平稳性检验

在深入协整之前，必须先理解**平稳性（Stationarity）**。

**定义**：一个时间序列$Y_t$是平稳的，如果：
- 均值常数：$\mathbb{E}[Y_t] = \mu$（与$t$无关）
- 方差常数：$\text{Var}(Y_t) = \sigma^2$（与$t$无关）
- 自协方差仅依赖于滞后阶数$k$：$\gamma_k = \text{Cov}(Y_t, Y_{t-k})$

**为什么重要？** 非平稳序列会导致"伪回归"（Spurious Regression）问题——即使两个完全独立的随机游走，回归得到的R²也可能很高。

#### Augmented Dickey-Fuller (ADF) 检验

最常用的平稳性检验方法：

$$
\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \varepsilon_t
$$

**原假设** $H_0: \gamma = 0$（序列有单位根，非平稳）  
**备择假设** $H_1: \gamma < 0$（序列平稳）

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

def test_stationarity(series, verbose=True):
    """
    ADF平稳性检验
    
    Parameters:
    -----------
    series : pd.Series
        待检验的时间序列
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    result : dict
        检验结果字典
    """
    result = adfuller(series, autolag='AIC')
    
    if verbose:
        print('ADF Statistic: %.4f' % result[0])
        print('p-value: %.4f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    
    is_stationary = result[1] <= 0.05
    
    if verbose:
        print("\n结论: %s" % ("平稳" if is_stationary else "非平稳"))
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': is_stationary
    }

# 示例：检验股价序列的平稳性
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2024-12-31', progress=False)

print(f"=== 检验 {ticker} 收盘价平稳性 ===")
result_price = test_stationarity(data['Close'].dropna())

print(f"\n=== 检验 {ticker} 收益率平稳性 ===")
returns = data['Close'].pct_change().dropna()
result_return = test_stationarity(returns)

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 原价格序列
axes[0, 0].plot(data.index, data['Close'])
axes[0, 0].set_title(f'{ticker} 收盘价', fontweight='bold')
axes[0, 0].set_ylabel('Price')
axes[0, 0].grid(True, alpha=0.3)

# 价格一阶差分
axes[0, 1].plot(data.index[1:], data['Close'].diff().dropna())
axes[0, 1].set_title(f'{ticker} 价格变化', fontweight='bold')
axes[0, 1].set_ylabel('Price Change')
axes[0, 1].grid(True, alpha=0.3)

# 收益率序列
axes[1, 0].plot(returns.index, returns)
axes[1, 0].set_title(f'{ticker} 收益率', fontweight='bold')
axes[1, 0].set_ylabel('Return')
axes[1, 0].grid(True, alpha=0.3)

# 收益率直方图
axes[1, 1].hist(returns, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'{ticker} 收益率分布', fontweight='bold')
axes[1, 1].set_xlabel('Return')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pair_trading_stationarity_test.png', dpi=300, bbox_inches='tight')
plt.show()
```

**结论**：股价序列通常是**非平稳**的（I(1)过程），而收益率序列通常是**平稳**的（I(0)过程）。

### 2.2 协整的定义

如果两个（或多个）非平稳时间序列的某种线性组合是平稳的，则称这些序列存在**协整关系**（Cointegration）。

**数学定义**：
对于 $N$ 个I(1)序列 $Y_{1,t}, Y_{2,t}, \ldots, Y_{N,t}$，如果存在系数向量 $\beta = [\beta_1, \beta_2, \ldots, \beta_N]'$，使得

$$
Z_t = \beta_1 Y_{1,t} + \beta_2 Y_{2,t} + \cdots + \beta_N Y_{N,t} \sim I(0)
$$

则称 $Y_{1,t}, Y_{2,t}, \ldots, Y_{N,t}$ 协整，$\beta$ 称为**协整向量**。

**直观理解**：协整关系意味着多个非平稳序列之间存在**长期均衡关系**，尽管短期可能偏离，但长期会回归均衡。

### 2.3 协整检验方法

#### 2.3.1 Engle-Granger 两步法

**步骤1**：用OLS估计协整回归

$$
Y_{1,t} = \alpha + \beta_2 Y_{2,t} + \cdots + \beta_N Y_{N,t} + \varepsilon_t
$$

**步骤2**：检验残差 $\hat{\varepsilon}_t$ 的平稳性（ADF检验）

```python
def engle_granger_test(y1, y2, verbose=True):
    """
    Engle-Granger两步法协整检验
    
    Parameters:
    -----------
    y1 : pd.Series
        第一个价格序列
    y2 : pd.Series
        第二个价格序列
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    result : dict
        检验结果
    """
    # 步骤1：OLS回归
    import statsmodels.api as sm
    X = sm.add_constant(y2)
    model = sm.OLS(y1, X).fit()
    residuals = model.resid
    
    # 步骤2：残差平稳性检验
    adf_result = adfuller(residuals, autolag='AIC')
    
    if verbose:
        print("=== Engle-Granger协整检验 ===")
        print("\n步骤1：OLS回归结果")
        print(f"  截距项 (α): {model.params[0]:.4f}")
        print(f"  系数 (β): {model.params[1]:.4f}")
        print(f"  R²: {model.rsquared:.4f}")
        
        print("\n步骤2：残差ADF检验")
        print(f"  ADF统计量: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        print(f"  临界值: {adf_result[4]}")
        
        is_cointegrated = adf_result[1] <= 0.05
        print(f"\n结论: {'存在协整关系' if is_cointegrated else '不存在协整关系'}")
    
    return {
        'alpha': model.params[0],
        'beta': model.params[1],
        'residuals': residuals,
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'is_cointegrated': adf_result[1] <= 0.05
    }

# 示例：检验两只银行股是否协整
tickers = ['JPM', 'BAC']
data1 = yf.download(tickers, start='2020-01-01', end='2024-12-31', progress=False)['Close']

# 确保数据对齐
prices1 = data1[tickers[0]].dropna()
prices2 = data1[tickers[1]].dropna()
common_idx = prices1.index.intersection(prices2.index)
prices1 = prices1.loc[common_idx]
prices2 = prices2.loc[common_idx]

print(f"=== 检验 {tickers[0]} 与 {tickers[1]} 的协整关系 ===")
eg_result = engle_granger_test(prices1, prices2)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 价格序列
axes[0, 0].plot(prices1.index, prices1 / prices1.iloc[0], 
                 label=tickers[0], linewidth=2)
axes[0, 0].plot(prices1.index, prices2 / prices2.iloc[0], 
                 label=tickers[1], linewidth=2, linestyle='--')
axes[0, 0].set_title('标准化价格对比', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 价差（残差）
axes[0, 1].plot(common_idx, eg_result['residuals'])
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[0, 1].set_title('协整残差（价差）', fontweight='bold')
axes[0, 1].set_ylabel('Spread')
axes[0, 1].grid(True, alpha=0.3)

# 残差分布
axes[1, 0].hist(eg_result['residuals'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('残差分布', fontweight='bold')
axes[1, 0].set_xlabel('Spread')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q图
from scipy import stats
stats.probplot(eg_result['residuals'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('残差Q-Q图', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pair_trading_cointegration_test.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### 2.3.2 Johansen 检验

对于多变量系统，Johansen检验更合适：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    data : pd.DataFrame
        多变量价格数据
    det_order : int
        确定性项设定（0：无常数项，1：有常数项，-1：无常数项和趋势）
    k_ar_diff : int
        滞后阶数
    
    Returns:
    --------
    result : dict
        检验结果
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    print("=== Johansen协整检验 ===")
    print("\n特征值（Eigenvalues）:")
    for i, val in enumerate(result.eig):
        print(f"  r={i}: {val:.4f}")
    
    print("\n迹检验（Trace Test）:")
    for i, (stat, crit) in enumerate(zip(result.lr1, result.cvt[:, 1])):  # 5%临界值
        print(f"  H0: r<={i} | 统计量: {stat:.2f}, 临界值(5%): {crit:.2f}, "
              f"结论: {'拒绝' if stat > crit else '不拒绝'}")
    
    print("\n最大特征值检验（Max-Eigen Test）:")
    for i, (stat, crit) in enumerate(zip(result.lr2, result.cvm[:, 1])):  # 5%临界值
        print(f"  H0: r={i} vs r={i+1} | 统计量: {stat:.2f}, 临界值(5%): {crit:.2f}, "
              f"结论: {'拒绝' if stat > crit else '不拒绝'}")
    
    return result

# 示例：多只银行股协整检验
tickers_multi = ['JPM', 'BAC', 'WFC', 'C']
data_multi = yf.download(tickers_multi, start='2020-01-01', end='2024-12-31', progress=False)['Close']

# 删除缺失值
data_multi = data_multi.dropna()

print(f"=== 检验 {tickers_multi} 的协整关系（Johansen检验）===")
johansen_result = johansen_test(data_multi)
```

### 2.4 配对选择的统计指标

除了协整检验，还可以用以下指标初筛：

#### 2.4.1 距离法（Distance Method）

计算价格序列标准化后的欧氏距离：

```python
def calculate_distance(y1, y2, method='ssd'):
    """
    计算两个价格序列的"距离"
    
    Parameters:
    -----------
    y1, y2 : pd.Series
        价格序列
    method : str
        距离度量方法（'ssd': 平方和, 'euclidean': 欧氏距离）
    
    Returns:
    --------
    distance : float
        距离值
    """
    # 标准化
    y1_norm = (y1 - y1.mean()) / y1.std()
    y2_norm = (y2 - y2.mean()) / y2.std()
    
    if method == 'ssd':
        distance = np.sum((y1_norm - y2_norm) ** 2)
    elif method == 'euclidean':
        distance = np.sqrt(np.sum((y1_norm - y2_norm) ** 2))
    else:
        raise ValueError("Method must be 'ssd' or 'euclidean'")
    
    return distance

# 示例
distance = calculate_distance(prices1, prices2, method='euclidean')
print(f"{tickers[0]} 与 {tickers[1]} 的距离: {distance:.4f}")
```

#### 2.4.2 相关性分析

```python
def calculate_correlation_metrics(y1, y2, window=252):
    """
    计算相关性指标
    
    Parameters:
    -----------
    y1, y2 : pd.Series
        价格序列
    window : int
        滚动窗口
    
    Returns:
    --------
    metrics : dict
        相关性指标
    """
    # 收益率
    ret1 = y1.pct_change().dropna()
    ret2 = y2.pct_change().dropna()
    
    # 整体相关系数
    overall_corr = ret1.corr(ret2)
    
    # 滚动相关系数
    rolling_corr = ret1.rolling(window).corr(ret2)
    
    # 协方差
    covariance = ret1.cov(ret2)
    
    return {
        'overall_correlation': overall_corr,
        'rolling_correlation': rolling_corr,
        'covariance': covariance
    }

# 示例
corr_metrics = calculate_correlation_metrics(prices1, prices2)
print(f"整体相关系数: {corr_metrics['overall_correlation']:.4f}")
```

## 三、实战策略构建

### 3.1 信号生成

基于协整残差（价差）的Z-Score生成交易信号：

```python
class PairsTradingStrategy:
    """
    配对交易策略类
    """
    
    def __init__(self, entry_z=2.0, exit_z=0.5, lookback=252):
        """
        初始化策略参数
        
        Parameters:
        -----------
        entry_z : float
            入场Z值阈值
        exit_z : float
            出场Z值阈值
        lookback : int
            参数估计窗口
        """
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        
    def estimate_parameters(self, y1, y2):
        """
        估计协整参数
        
        Parameters:
        -----------
        y1, y2 : pd.Series
            价格序列
        
        Returns:
        --------
        params : dict
            协整参数
        """
        # OLS回归
        import statsmodels.api as sm
        X = sm.add_constant(y2)
        model = sm.OLS(y1, X).fit()
        
        # 残差
        residuals = model.resid
        
        # 计算Z-Score的均值和标准差
        mean = residuals.mean()
        std = residuals.std()
        
        self.params = {
            'alpha': model.params[0],
            'beta': model.params[1],
            'residuals': residuals,
            'mean': mean,
            'std': std
        }
        
        return self.params
    
    def calculate_z_score(self, y1, y2, date):
        """
        计算当期Z值
        
        Parameters:
        -----------
        y1, y2 : pd.Series
            价格序列
        date : datetime
            当前日期
        
        Returns:
        --------
        z_score : float
            Z值
        """
        # 估计窗口数据
        est_start = date - pd.Timedelta(days=self.lookback)
        y1_est = y1.loc[est_start:date]
        y2_est = y2.loc[est_start:date]
        
        # 重新估计参数
        params = self.estimate_parameters(y1_est, y2_est)
        
        # 计算当期残差
        residual = y1.loc[date] - (params['alpha'] + params['beta'] * y2.loc[date])
        
        # 计算Z值
        z_score = (residual - params['mean']) / params['std']
        
        return z_score
    
    def generate_signals(self, y1, y2):
        """
        生成交易信号
        
        Parameters:
        -----------
        y1, y2 : pd.Series
            价格序列
        
        Returns:
        --------
        signals : pd.DataFrame
            交易信号
        """
        signals = pd.DataFrame(index=y1.index)
        signals['z_score'] = np.nan
        signals['position'] = 0  # 0: 无仓位, 1: 多y1空y2, -1: 空y1多y2
        
        # 滚动计算Z值
        for i, date in enumerate(y1.index[self.lookback:]):
            z_score = self.calculate_z_score(y1, y2, date)
            signals.loc[date, 'z_score'] = z_score
        
        # 生成仓位信号
        current_position = 0
        
        for date in signals.index[self.lookback:]:
            z_score = signals.loc[date, 'z_score']
            
            if pd.isna(z_score):
                continue
            
            # 入场信号
            if current_position == 0:
                if z_score > self.entry_z:
                    current_position = -1  # 空y1, 多y2
                elif z_score < -self.entry_z:
                    current_position = 1   # 多y1, 空y2
            
            # 出场信号
            elif current_position == 1:  # 当前持有多y1空y2
                if z_score >= -self.exit_z:
                    current_position = 0  # 平仓
            elif current_position == -1:  # 当前持有空y1多y2
                if z_score <= self.exit_z:
                    current_position = 0  # 平仓
            
            signals.loc[date, 'position'] = current_position
        
        return signals

# 示例：生成交易信号
strategy = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=252)
signals = strategy.generate_signals(prices1, prices2)

print("\n=== 交易信号统计 ===")
print(f"总交易日: {len(signals.dropna())}")
print(f"多头仓位天数: {(signals['position'] == 1).sum()}")
print(f"空头仓位天数: {(signals['position'] == -1).sum()}")
print(f"无仓位天数: {(signals['position'] == 0).sum()}")
```

### 3.2 回测框架

```python
class PairTradingBacktester:
    """
    配对交易回测框架
    """
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001):
        """
        初始化回测参数
        
        Parameters:
        -----------
        initial_capital : float
            初始资金
        transaction_cost : float
            交易成本（单边）
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def backtest(self, y1, y2, signals):
        """
        执行回测
        
        Parameters:
        -----------
        y1, y2 : pd.Series
            价格序列
        signals : pd.DataFrame
            交易信号
        
        Returns:
        --------
        results : pd.DataFrame
            回测结果
        """
        results = pd.DataFrame(index=signals.index)
        results['y1_price'] = y1
        results['y2_price'] = y2
        results['position'] = signals['position']
        results['z_score'] = signals['z_score']
        
        # 初始化账户
        results['capital'] = self.initial_capital
        results['y1_shares'] = 0
        results['y2_shares'] = 0
        results['y1_value'] = 0
        results['y2_value'] = 0
        results['total_value'] = self.initial_capital
        results['returns'] = 0.0
        
        prev_position = 0
        
        for i, date in enumerate(results.index[1:], 1):
            current_position = results.loc[date, 'position']
            prev_position = results.loc[results.index[i-1], 'position']
            
            # 如果仓位变化，执行交易
            if current_position != prev_position:
                # 平仓（如果有）
                if prev_position == 1:  # 平多y1空y2
                    results.loc[date, 'y1_shares'] = 0
                    results.loc[date, 'y2_shares'] = 0
                elif prev_position == -1:  # 平空y1多y2
                    results.loc[date, 'y1_shares'] = 0
                    results.loc[date, 'y2_shares'] = 0
                
                # 开仓（如果新仓位非0）
                if current_position == 1:  # 多y1空y2
                    # 等权重分配
                    allocation = results.loc[results.index[i-1], 'total_value'] / 2
                    
                    y1_shares = allocation / results.loc[date, 'y1_price']
                    y2_shares = allocation / results.loc[date, 'y2_price']
                    
                    results.loc[date, 'y1_shares'] = y1_shares
                    results.loc[date, 'y2_shares'] = -y2_shares  # 空头为负
                    
                elif current_position == -1:  # 空y1多y2
                    allocation = results.loc[results.index[i-1], 'total_value'] / 2
                    
                    y1_shares = allocation / results.loc[date, 'y1_price']
                    y2_shares = allocation / results.loc[date, 'y2_price']
                    
                    results.loc[date, 'y1_shares'] = -y1_shares  # 空头为负
                    results.loc[date, 'y2_shares'] = y2_shares
            
            else:
                # 仓位不变，继承上期
                results.loc[date, 'y1_shares'] = results.loc[results.index[i-1], 'y1_shares']
                results.loc[date, 'y2_shares'] = results.loc[results.index[i-1], 'y2_shares']
            
            # 计算当日市值
            y1_value = results.loc[date, 'y1_shares'] * results.loc[date, 'y1_price']
            y2_value = results.loc[date, 'y2_shares'] * results.loc[date, 'y2_price']
            
            results.loc[date, 'y1_value'] = y1_value
            results.loc[date, 'y2_value'] = y2_value
            results.loc[date, 'total_value'] = results.loc[results.index[i-1], 'capital'] + y1_value + y2_value
            
            # 计算收益
            if i > 1:
                results.loc[date, 'returns'] = (
                    results.loc[date, 'total_value'] / results.loc[results.index[i-1], 'total_value'] - 1
                )
        
        # 计算累积收益
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        
        return results
    
    def calculate_metrics(self, results):
        """
        计算绩效指标
        
        Parameters:
        -----------
        results : pd.DataFrame
            回测结果
        
        Returns:
        --------
        metrics : dict
            绩效指标
        """
        returns = results['returns'].dropna()
        
        # 总收益
        total_return = results['cumulative_returns'].iloc[-1] - 1
        
        # 年化收益
        trading_days = len(returns)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        
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
            'win_rate': win_rate,
            'num_trades': (results['position'].diff() != 0).sum()
        }
        
        return metrics

# 示例：回测
backtester = PairTradingBacktester(
    initial_capital=1000000,
    transaction_cost=0.001
)

results = backtester.backtest(prices1, prices2, signals)
metrics = backtester.calculate_metrics(results)

print("\n=== 回测结果 ===")
for key, value in metrics.items():
    if key in ['total_return', 'annual_return', 'max_drawdown']:
        print(f"{key}: {value*100:.2f}%")
    elif key == 'sharpe_ratio':
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 上图：价格与仓位
ax1 = axes[0]
ax1.plot(results.index, results['y1_price'] / results['y1_price'].iloc[0], 
         label=tickers[0], linewidth=2)
ax1.plot(results.index, results['y2_price'] / results['y2_price'].iloc[0], 
         label=tickers[1], linewidth=2, linestyle='--')
ax1.set_ylabel('Normalized Price', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 添加仓位背景色
for i in range(len(results)-1):
    date1 = results.index[i]
    date2 = results.index[i+1]
    position = results['position'].iloc[i]
    
    if position == 1:
        ax1.axvspan(date1, date2, alpha=0.3, color='green')
    elif position == -1:
        ax1.axvspan(date1, date2, alpha=0.3, color='red')

ax1.set_title('Price & Position', fontweight='bold')

# 中图：Z值
ax2 = axes[1]
ax2.plot(results.index, results['z_score'], linewidth=1.5, color='purple')
ax2.axhline(y=2, color='r', linestyle='--', label='Entry Threshold (+2)')
ax2.axhline(y=-2, color='r', linestyle='--', label='Entry Threshold (-2)')
ax2.axhline(y=0.5, color='g', linestyle=':', label='Exit Threshold (+0.5)')
ax2.axhline(y=-0.5, color='g', linestyle=':', label='Exit Threshold (-0.5)')
ax2.fill_between(results.index, -2, 2, alpha=0.2, color='gray')
ax2.set_ylabel('Z-Score', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Z-Score of Spread', fontweight='bold')

# 下图：累积收益
ax3 = axes[2]
ax3.plot(results.index, results['cumulative_returns'], 
         linewidth=2, color='blue', label='Pair Trading')
ax3.axhline(y=1, color='k', linestyle='--', linewidth=1)
ax3.set_ylabel('Cumulative Returns', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_title('Cumulative Returns', fontweight='bold')

plt.tight_layout()
plt.savefig('pair_trading_backtest_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 四、实战中的关键问题

### 4.1 标的筛选的规模化

手动筛选配对不可持续，需要系统化方法：

```python
def screen_pairs(universe, start_date, end_date, min_corr=0.7):
    """
    系统化筛选配对
    
    Parameters:
    -----------
    universe : list
        股票代码列表
    start_date, end_date : str
        日期范围
    min_corr : float
        最小相关系数阈值
    
    Returns:
    --------
    candidate_pairs : list
        候选配对列表
    """
    # 下载数据
    data = yf.download(universe, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna()
    
    candidate_pairs = []
    
    # 双重循环筛选
    for i in range(len(universe)):
        for j in range(i+1, len(universe)):
            stock1 = universe[i]
            stock2 = universe[j]
            
            # 计算相关系数
            ret1 = data[stock1].pct_change().dropna()
            ret2 = data[stock2].pct_change().dropna()
            correlation = ret1.corr(ret2)
            
            # 相关性过滤
            if correlation < min_corr:
                continue
            
            # 协整检验
            try:
                eg_result = engle_granger_test(
                    data[stock1], data[stock2], verbose=False
                )
                
                if eg_result['is_cointegrated']:
                    candidate_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'correlation': correlation,
                        'p_value': eg_result['p_value'],
                        'beta': eg_result['beta']
                    })
            except Exception as e:
                continue
    
    return candidate_pairs

# 示例：从银行股中筛选配对
bank_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BK', 'STT', 'USB']
print("=== 筛选银行股配对 ===")
candidate_pairs = screen_pairs(bank_stocks, '2020-01-01', '2024-12-31')

print(f"\n找到 {len(candidate_pairs)} 个候选配对:")
for pair in candidate_pairs:
    print(f"  {pair['stock1']} - {pair['stock2']} | "
          f"相关系数: {pair['correlation']:.3f} | "
          f"p-value: {pair['p_value']:.4f}")
```

### 4.2 参数稳定性

协整关系可能随时间退化，需要**滚动窗口**重新估计：

```python
def rolling_cointegration_test(y1, y2, window=252, step=20):
    """
    滚动协整检验
    
    Parameters:
    -----------
    y1, y2 : pd.Series
        价格序列
    window : int
        滚动窗口
    step : int
        滚动步长
    
    Returns:
    --------
    results : pd.DataFrame
        滚动检验结果
    """
    results = []
    
    for i in range(window, len(y1), step):
        # 截取子样本
        y1_sub = y1.iloc[i-window:i]
        y2_sub = y2.iloc[i-window:i]
        
        # 协整检验
        try:
            eg_result = engle_granger_test(y1_sub, y2_sub, verbose=False)
            
            results.append({
                'date': y1.index[i],
                'p_value': eg_result['p_value'],
                'is_cointegrated': eg_result['is_cointegrated'],
                'beta': eg_result['beta'],
                'alpha': eg_result['alpha']
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

# 示例
rolling_results = rolling_cointegration_test(prices1, prices2, window=252, step=20)

print("\n=== 滚动协整检验结果 ===")
print(f"总检验次数: {len(rolling_results)}")
print(f"协整比例: {rolling_results['is_cointegrated'].mean()*100:.2f}%")
print(f"beta均值: {rolling_results['beta'].mean():.4f}")
print(f"beta标准差: {rolling_results['beta'].std():.4f}")

# 可视化beta变化
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rolling_results['date'], rolling_results['beta'], 
        marker='o', linewidth=2, markersize=4)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Beta Coefficient', fontsize=12)
ax.set_title('Rolling Beta Coefficient', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pair_trading_rolling_beta.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.3 风险管理

配对交易虽为市场中性，但仍面临多种风险：

1. **配对解体风险**：协整关系永久性断裂
2. **流动性风险**：某只股票交易不活跃
3. **杠杆风险**：为放大收益过度加杠杆
4. **模型风险**：参数估计误差、结构性断点

```python
def risk_management_module(results, signals, max_holding_period=20, stop_loss_z=3.0):
    """
    风险管理模块
    
    Parameters:
    -----------
    results : pd.DataFrame
        回测结果
    signals : pd.DataFrame
        交易信号
    max_holding_period : int
        最大持仓周期（交易日）
    stop_loss_z : float
        止损Z值
    
    Returns:
    --------
    adjusted_signals : pd.DataFrame
        调整后的信号
    """
    adjusted_signals = signals.copy()
    
    # 1. 持仓时间限制
    position_start = None
    holding_days = 0
    
    for i, date in enumerate(adjusted_signals.index):
        position = adjusted_signals.loc[date, 'position']
        
        if position != 0:
            if position_start is None:
                position_start = date
                holding_days = 0
            else:
                holding_days = (date - position_start).days
                
                # 超过最大持仓时间，强制平仓
                if holding_days > max_holding_period:
                    adjusted_signals.loc[date, 'position'] = 0
                    position_start = None
        else:
            position_start = None
            holding_days = 0
    
    # 2. 止损机制
    for i, date in enumerate(adjusted_signals.index):
        z_score = adjusted_signals.loc[date, 'z_score']
        
        if pd.isna(z_score):
            continue
        
        # Z值超过止损线，强制平仓
        if abs(z_score) > stop_loss_z:
            adjusted_signals.loc[date, 'position'] = 0
    
    return adjusted_signals

# 示例：应用风险管理
adjusted_signals = risk_management_module(results, signals, 
                                         max_holding_period=20, 
                                         stop_loss_z=3.0)

print("\n=== 风险管理前后对比 ===")
print(f"原始信号交易次数: {(signals['position'].diff() != 0).sum()}")
print(f"调整后信号交易次数: {(adjusted_signals['position'].diff() != 0).sum()}")
```

## 五、进阶话题

### 5.1 多因子配对

单一配对容量有限，可扩展为**配对组合**：

```python
def multi_pair_portfolio(pairs, signals_dict, allocation_method='equal'):
    """
    多配对组合
    
    Parameters:
    -----------
    pairs : list
        配对列表
    signals_dict : dict
        每个配对的信号字典
    allocation_method : str
        资金分配方法（'equal': 等权, 'vol': 按波动率倒权）
    
    Returns:
    --------
    portfolio_returns : pd.Series
        组合收益率
    """
    # 合并所有信号的日期索引
    all_dates = sorted(set().union(*[signals.index for signals in signals_dict.values()]))
    
    portfolio_returns = pd.Series(0, index=all_dates)
    
    if allocation_method == 'equal':
        weights = {pair: 1.0 / len(pairs) for pair in pairs}
    elif allocation_method == 'vol':
        # 按波动率倒权（需要预先计算）
        pass
    
    # 汇总收益
    for pair in pairs:
        pair_returns = signals_dict[pair]['returns']
        weight = weights[pair]
        
        portfolio_returns = portfolio_returns.add(
            pair_returns.reindex(all_dates) * weight, fill_value=0
        )
    
    return portfolio_returns
```

### 5.2 机器学习增强

使用ML模型改进配对选择和时机判断：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def ml_enhanced_pair_selection(features, labels):
    """
    机器学习辅助配对筛选
    
    Parameters:
    -----------
    features : pd.DataFrame
        特征矩阵（相关性、距离、行业相似度等）
    labels : pd.Series
        标签（是否盈利）
    
    Returns:
    --------
    model : sklearn model
        训练好的模型
    """
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )
    
    # 训练随机森林
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    # 特征重要性
    importance = pd.Series(model.feature_importances_, index=features.columns)
    print("\n特征重要性:")
    print(importance.sort_values(ascending=False))
    
    return model

# 示例特征工程
def extract_pair_features(y1, y2, y1_fundamentals, y2_fundamentals):
    """
    提取配对特征
    
    Parameters:
    -----------
    y1, y2 : pd.Series
        价格序列
    y1_fundamentals, y2_fundamentals : dict
        基本面数据
    
    Returns:
    --------
    features : pd.DataFrame
        特征矩阵
    """
    features = pd.DataFrame()
    
    # 1. 价格相关性
    ret1 = y1.pct_change().dropna()
    ret2 = y2.pct_change().dropna()
    features['correlation'] = [ret1.corr(ret2)]
    
    # 2. 协整p-value
    eg_result = engle_granger_test(y1, y2, verbose=False)
    features['cointegration_pvalue'] = [eg_result['p_value']]
    
    # 3. 距离
    features['distance'] = [calculate_distance(y1, y2)]
    
    # 4. 基本面相似度
    # 行业相同=1，否则=0
    features['same_industry'] = [1 if y1_fundamentals['industry'] == y2_fundamentals['industry'] else 0]
    
    # 市值差异
    features['market_cap_diff'] = [
        abs(np.log(y1_fundamentals['market_cap']) - np.log(y2_fundamentals['market_cap']))
    ]
    
    return features
```

### 5.3 高频配对交易

将策略应用到分钟级或秒级数据：

```python
def high_frequency_pair_trading(y1_intraday, y2_intraday, window=60):
    """
    高频配对交易
    
    Parameters:
    -----------
    y1_intraday, y2_intraday : pd.DataFrame
        日内数据（包含open, high, low, close, volume）
    window : int
        滚动窗口（分钟）
    
    Returns:
    --------
    signals : pd.DataFrame
        高频交易信号
    """
    # 计算分钟级收益率
    ret1 = y1_intraday['close'].pct_change()
    ret2 = y2_intraday['close'].pct_change()
    
    # 滚动回归估计beta
    beta_rolling = []
    
    for i in range(window, len(ret1)):
        ret1_window = ret1.iloc[i-window:i]
        ret2_window = ret2.iloc[i-window:i]
        
        # 简单线性回归
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            ret2_window, ret1_window
        )
        
        beta_rolling.append({
            'datetime': ret1.index[i],
            'beta': slope,
            'alpha': intercept,
            'r_squared': r_value**2
        })
    
    beta_df = pd.DataFrame(beta_rolling).set_index('datetime')
    
    # 计算高频价差
    spread = ret1.loc[beta_df.index] - (
        beta_df['alpha'] + beta_df['beta'] * ret2.loc[beta_df.index]
    )
    
    # Z值
    z_score = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    # 生成信号（更频繁的入场出场）
    signals = pd.DataFrame(index=z_score.index)
    signals['z_score'] = z_score
    signals['position'] = 0
    
    # 高频阈值（更窄）
    entry_z = 1.5
    exit_z = 0.2
    
    current_position = 0
    for date in signals.index:
        z = signals.loc[date, 'z_score']
        
        if pd.isna(z):
            continue
        
        if current_position == 0:
            if z > entry_z:
                current_position = -1
            elif z < -entry_z:
                current_position = 1
        elif current_position == 1:
            if z >= -exit_z:
                current_position = 0
        elif current_position == -1:
            if z <= exit_z:
                current_position = 0
        
        signals.loc[date, 'position'] = current_position
    
    return signals
```

## 六、总结与展望

### 主要结论

1. **理论基础扎实**：协整分析为配对交易提供了严谨的统计学基础
2. **实践挑战多**：参数估计、配对解体、交易成本等实际问题需要仔细处理
3. **风险管理至关重要**：配对交易虽为市场中性，但并非无风险
4. **技术持续演进**：机器学习、高频数据等新技术不断拓展策略边界

### 最佳实践建议

1. **严格筛选配对**：结合统计指标与经济逻辑
2. **滚动估计参数**：适应市场结构变化
3. **多重风险控制**：持仓时间、止损、仓位限制
4. **充分考虑成本**：交易成本、滑点、卖空费用
5. **分散化配置**：多配对组合降低单一配对风险

### 未来方向

1. **非参数方法**：不依赖正态分布假设，更鲁棒
2. **图神经网络**：捕捉多个资产间的复杂关系网络
3. **加密货币配对**：24/7交易市场的特殊性
4. **ESG因子整合**：将环境、社会、治理因素纳入配对选择

## 参考资料

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
2. Ganapathy, V. (2004). *Statistical Arbitrage and Pairs Trading*
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*
4. Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction: Representation, estimation, and testing"
5. Johansen, S. (1991). "Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models"

---

**免责声明**：本文仅供学术研究和教育目的，不构成投资建议。配对交易涉及卖空操作，可能面临无限损失风险。在实际应用前，请充分了解相关风险并咨询专业投资顾问。
