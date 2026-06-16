---
title: "配对交易与协整分析：统计套利的理论与实践"
description: "配对交易是最经典的统计套利策略之一。本文深入探讨协整理论、配对筛选方法、交易信号构建和风险管理，并提供完整的Python实现框架。"
date: "2026-06-16"
tags: ["配对交易", "协整分析", "统计套利", "市场中性"]
topic: "quant"
difficulty: "进阶"
image: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：统计套利的理论与实践

## 引言

配对交易（Pair Trading）是一种经典的市场中性策略，由摩根士丹利在1980年代首次系统化应用。该策略基于一个简单而强大的思想：找到两只价格具有长期均衡关系的股票，当价格偏离时做多低估股票、做空高估股票，等待价格回归后平仓获利。

本文将系统介绍配对交易的理论基础（协整分析）、实战中的配对筛选方法、交易信号构建，以及完整的风险管理框架。

## 一、理论基础：协整与均值回归

### 1.1 平稳性与协整

#### 1.1.1 平稳性检验

时间序列的平稳性是协整分析的基础。一个平稳序列的均值、方差和自协方差不随时间变化。

**ADF检验（Augmented Dickey-Fuller Test）**：

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

def adf_test(series, verbose=True):
    """
    ADF单位根检验
    
    Parameters:
    -----------
    series: pd.Series, 时间序列
    verbose: bool, 是否打印详细信息
    
    Returns:
    --------
    result: dict, 检验结果
    """
    result = adfuller(series, autolag='AIC')
    
    if verbose:
        print('ADF Statistic: {:.4f}'.format(result[0]))
        print('p-value: {:.4f}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value:.4f}')
    
    is_stationary = result[1] < 0.05
    
    if verbose:
        print(f"\n结论: {'平稳' if is_stationary else '非平稳'}")
    
    return {
        'adf_stat': result[0],
        'p_value': result[1],
        'is_stationary': is_stationary,
        'critical_values': result[4]
    }

# 示例：检验股票价格序列
def example_adf_test():
    """ADF检验示例"""
    # 下载数据
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    
    for ticker in tickers:
        print(f"\n=== {ticker} 价格序列 ADF检验 ===")
        result_price = adf_test(data[ticker])
        
        print(f"\n=== {ticker} 收益率序列 ADF检验 ===")
        return_series = data[ticker].pct_change().dropna()
        result_return = adf_test(return_series)
    
    return data

# 运行示例
# price_data = example_adf_test()
```

**解读**：
- p-value < 0.05：拒绝原假设（存在单位根），序列平稳
- p-value ≥ 0.05：无法拒绝原假设，序列非平稳

**常见金融时间序列的平稳性**：
- 股票价格：非平稳（随机游走）
- 股票收益率：通常平稳
- 配对价差：若协整，则平稳

#### 1.1.2 协整检验

协整（Cointegration）指的是多个非平稳序列的线性组合是平稳的，表明它们之间存在长期均衡关系。

**Engle-Granger两步法**：

```python
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y: pd.Series, 第一个价格序列
    x: pd.Series, 第二个价格序列
    verbose: bool, 是否打印详细信息
    
    Returns:
    --------
    result: dict, 检验结果
    """
    # 步骤1：OLS回归
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    spread = model.resid
    
    # 步骤2：检验残差的平稳性（ADF检验）
    adf_result = adfuller(spread, autolag='AIC')
    
    # 计算协整统计量（简化版）
    # 完整的EG检验需要使用临界值表或coint函数
    coint_stat, p_value, _ = coint(y, x)
    
    if verbose:
        print("=== Engle-Granger协整检验 ===")
        print(f"协整统计量: {coint_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"残差ADF统计量: {adf_result[0]:.4f}")
        print(f"残差ADF p-value: {adf_result[1]:.4f}")
        print(f"\n结论: {'存在协整关系' if p_value < 0.05 else '不存在协整关系'}")
    
    # 计算对冲比例（hedge ratio）
    hedge_ratio = model.params[1]
    
    if verbose:
        print(f"\n对冲比例 (β): {hedge_ratio:.4f}")
        print(f"回归方程: y = {model.params[0]:.4f} + {hedge_ratio:.4f} * x")
    
    return {
        'coint_stat': coint_stat,
        'p_value': p_value,
        'is_cointegrated': p_value < 0.05,
        'hedge_ratio': hedge_ratio,
        'intercept': model.params[0],
        'spread': spread
    }

# 示例：检验两只股票是否协整
def example_cointegration_test():
    """协整检验示例"""
    # 下载数据
    tickers = ['KO', 'PEP']  # 可口可乐 vs 百事可乐
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    
    print("=== 价格序列平稳性检验 ===")
    for ticker in tickers:
        result = adf_test(data[ticker], verbose=False)
        print(f"{ticker}: {'平稳' if result['is_stationary'] else '非平稳'}")
    
    print("\n=== 协整检验 ===")
    result = engle_granger_test(data[tickers[0]], data[tickers[1]])
    
    # 可视化价差
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 子图1：价格序列
    ax1 = axes[0]
    ax1.plot(data.index, data[tickers[0]], label=tickers[0], linewidth=2)
    ax1.plot(data.index, data[tickers[1]], label=tickers[1], linewidth=2)
    ax1.set_title('Stock Prices', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：价差序列
    ax2 = axes[1]
    spread = result['spread']
    ax2.plot(spread.index, spread, label='Spread', color='green', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=spread.mean() + 2*spread.std(), color='red', 
                linestyle='--', alpha=0.5, label='±2σ')
    ax2.axhline(y=spread.mean() - 2*spread.std(), color='red', 
                linestyle='--', alpha=0.5)
    ax2.fill_between(spread.index, 
                     spread.mean() - 2*spread.std(),
                     spread.mean() + 2*spread.std(),
                     alpha=0.2, color='gray')
    ax2.set_title('Spread (Residuals from Cointegrating Regression)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cointegration_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: cointegration_example.png")
    
    return data, result

# 运行示例
# data, coint_result = example_cointegration_test()
```

**Johansen检验（多变量协整）**：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多只股票）
    
    Parameters:
    -----------
    data: pd.DataFrame, 多只股票的价格数据
    det_order: int, 确定性项的设定
               0: 无常数项，无趋势
               1: 有常数项，无趋势
               2: 有常数项，有趋势
    k_ar_diff: int, 滞后阶数
    
    Returns:
    --------
    result: dict, 检验结果
    """
    # 执行Johansen检验
    result = coint_johansen(data, det_order, k_ar_diff)
    
    # 提取统计量和临界值
    trace_stat = result.lr1  # 迹统计量
    max_stat = result.lr2  # 最大特征值统计量
    trace_crit = result.cvt  # 迹统计量临界值 (90%, 95%, 99%)
    max_crit = result.cvm  # 最大特征值临界值
    
    print("=== Johansen协整检验 ===")
    print("\n迹统计量检验 (Trace Test):")
    for i, stat in enumerate(trace_stat):
        print(f"  H0: 协整秩 ≤ {i}")
        print(f"  统计量: {stat:.4f}")
        print(f"  90%临界值: {trace_crit[i, 0]:.4f}")
        print(f"  95%临界值: {trace_crit[i, 1]:.4f}")
        print(f"  99%临界值: {trace_crit[i, 2]:.4f}")
        print()
    
    # 确定协整秩
    cointegrating_rank = 0
    for i, stat in enumerate(trace_stat):
        if stat > trace_crit[i, 1]:  # 使用95%临界值
            cointegrating_rank = i + 1
    
    print(f"结论: 协整秩 = {cointegrating_rank}")
    
    return {
        'trace_stat': trace_stat,
        'max_stat': max_stat,
        'trace_crit': trace_crit,
        'max_crit': max_crit,
        'cointegrating_rank': cointegrating_rank,
        'eigenvectors': result.evec  # 协整向量
    }

# 示例：多只股票协整检验
def example_johansen_test():
    """Johansen检验示例"""
    # 下载多只股票数据
    tickers = ['XOM', 'CVX', 'COP']  # 能源股
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    
    # 执行检验
    result = johansen_test(data[['XOM', 'CVX', 'COP']], det_order=1, k_ar_diff=1)
    
    # 提取协整向量
    if result['cointegrating_rank'] > 0:
        print("\n协整向量 (前{}个):".format(result['cointegrating_rank']))
        for i in range(result['cointegrating_rank']):
            print(f"\n协整向量 {i+1}:")
            for j, ticker in enumerate(tickers):
                print(f"  {ticker}: {result['eigenvectors'][j, i]:.4f}")
    
    return data, result

# 运行示例
# data_johansen, johansen_result = example_johansen_test()
```

### 1.2 配对交易的经济学逻辑

配对交易的有效性基于以下假设：

1. **长期均衡**：两只股票的基本面相关，价格存在长期均衡关系
2. **短期偏离**：临时冲击导致价格偏离均衡
3. **均值回归**：市场机制（套利、信息传播）推动价格回归均衡

**适合的配对类型**：
- **同行业龙头**：如KO-PEP（饮料）、XOM-CVX（能源）
- **产业链上下游**：如汽车制造商-零部件供应商
- **替代品**：如可口可乐-百事可乐
- **ETF与其成分股**：如QQQ与AAPL

## 二、配对筛选方法

### 2.1 距离法（Distance Method）

最简单的配对筛选方法：计算价格序列的标准化距离。

```python
def distance_method(prices, start_date, end_date, top_n=20):
    """
    距离法筛选配对
    
    Parameters:
    -----------
    prices: pd.DataFrame, 股票价格数据
    start_date, end_date: str, 训练期
    top_n: int, 返回top N个配对
    
    Returns:
    --------
    pairs: list, 筛选出的配对列表
    """
    # 截取训练期数据
    train_data = prices.loc[start_date:end_date]
    
    # 标准化价格（以第一个日期为基准100）
    normalized = train_data / train_data.iloc[0] * 100
    
    # 计算所有配对的欧氏距离
    distances = {}
    tickers = normalized.columns
    
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            pair_name = f"{tickers[i]}-{tickers[j]}"
            
            # 计算价差
            spread = normalized[tickers[i]] - normalized[tickers[j]]
            
            # 欧氏距离（实际上是SSD: Sum of Squared Differences）
            distance = np.sqrt((spread ** 2).sum())
            
            # 同时计算其他指标
            spread_std = spread.std()
            mean_reversion_speed = calculate_mean_reversion_speed(spread)
            
            distances[pair_name] = {
                'distance': distance,
                'spread_std': spread_std,
                'mean_reversion_speed': mean_reversion_speed,
                'ticker1': tickers[i],
                'ticker2': tickers[j]
            }
    
    # 按距离排序
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1]['distance'])
    
    # 返回top N
    top_pairs = []
    for pair_name, metrics in sorted_pairs[:top_n]:
        top_pairs.append({
            'pair': pair_name,
            'ticker1': metrics['ticker1'],
            'ticker2': metrics['ticker2'],
            'distance': metrics['distance'],
            'spread_std': metrics['spread_std'],
            'mean_reversion_speed': metrics['mean_reversion_speed']
        })
    
    return top_pairs

def calculate_mean_reversion_speed(spread, lag=1):
    """
    计算均值回归速度（半生命周期）
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    lag: int, 滞后阶数
    
    Returns:
    --------
    half_life: float, 半衰期（天数）
    """
    # 计算价差的变化
    spread_lag = spread.shift(lag)
    spread_diff = spread - spread_lag
    
    # OLS回归: Δspread = α + β * spread_lag + ε
    X = sm.add_constant(spread_lag.dropna())
    y = spread_diff.dropna()
    
    model = sm.OLS(y, X).fit()
    beta = model.params[1]
    
    # 半衰期 = ln(0.5) / ln(1 + β)
    if beta >= 0:
        return np.inf  # 不均值回归
    
    half_life = np.log(0.5) / np.log(1 + beta)
    
    return abs(half_life)

# 示例：使用距离法筛选配对
def example_distance_method():
    """距离法示例"""
    # 下载数据
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 
               'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    prices = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 筛选配对
    pairs = distance_method(prices, '2020-01-01', '2022-12-31', top_n=10)
    
    print("=== 距离法筛选结果 (Top 10) ===\n")
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. {pair['pair']}")
        print(f"   距离: {pair['distance']:.2f}")
        print(f"   价差波动率: {pair['spread_std']:.2f}")
        print(f"   半衰期: {pair['mean_reversion_speed']:.1f}天")
        print()
    
    return pairs

# 运行示例
# top_pairs = example_distance_method()
```

### 2.2 协整法（Cointegration Method）

基于严格的协整检验筛选配对。

```python
def cointegration_method(prices, start_date, end_date, 
                         p_threshold=0.05, top_n=20):
    """
    协整法筛选配对
    
    Parameters:
    -----------
    prices: pd.DataFrame, 股票价格数据
    start_date, end_date: str, 训练期
    p_threshold: float, p-value阈值
    top_n: int, 返回top N个配对
    
    Returns:
    --------
    pairs: list, 筛选出的配对列表
    """
    # 截取训练期数据
    train_data = prices.loc[start_date:end_date]
    
    # 检验所有配对
    cointegrated_pairs = []
    tickers = train_data.columns
    
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            # 协整检验
            try:
                result = engle_granger_test(
                    train_data[tickers[i]], 
                    train_data[tickers[j]],
                    verbose=False
                )
                
                if result['is_cointegrated']:
                    # 计算额外指标
                    spread = result['spread']
                    sharpe = calculate_spread_sharpe(spread)
                    half_life = calculate_mean_reversion_speed(spread)
                    
                    cointegrated_pairs.append({
                        'pair': f"{tickers[i]}-{tickers[j]}",
                        'ticker1': tickers[i],
                        'ticker2': tickers[j],
                        'p_value': result['p_value'],
                        'hedge_ratio': result['hedge_ratio'],
                        'intercept': result['intercept'],
                        'sharpe': sharpe,
                        'half_life': half_life,
                        'spread_std': spread.std()
                    })
            except Exception as e:
                # 跳过检验失败的配对
                continue
    
    # 按p-value排序（越小越好）
    sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x['p_value'])
    
    # 返回top N
    return sorted_pairs[:top_n]

def calculate_spread_sharpe(spread, window=252):
    """
    计算价差序列的Sharpe Ratio（用于评估配对质量）
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    window: int, 滚动窗口
    
    Returns:
    --------
    sharpe: float, Sharpe Ratio
    """
    # 价差的收益率（均值回归视角）
    spread_return = -spread.diff()  # 负的价格变化 = 均值回归收益
    
    # 年化Sharpe
    sharpe = spread_return.mean() / spread_return.std() * np.sqrt(window)
    
    return sharpe

# 示例：使用协整法筛选配对
def example_cointegration_method():
    """协整法示例"""
    # 下载数据
    tickers = ['KO', 'PEP', 'XOM', 'CVX', 'WMT', 'TGT', 
               'JPM', 'BAC', 'PG', 'UL']
    prices = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 筛选配对
    pairs = cointegration_method(prices, '2020-01-01', '2022-12-31', 
                                p_threshold=0.05, top_n=10)
    
    print("=== 协整法筛选结果 (Top 10) ===\n")
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. {pair['pair']}")
        print(f"   p-value: {pair['p_value']:.4f}")
        print(f"   对冲比例: {pair['hedge_ratio']:.4f}")
        print(f"   Sharpe: {pair['sharpe']:.2f}")
        print(f"   半衰期: {pair['half_life']:.1f}天")
        print()
    
    return pairs

# 运行示例
# cointegrated_pairs = example_cointegration_method()
```

### 2.3 相关性法（Correlation Method）

基于收益率相关性和协整双重筛选。

```python
def correlation_method(prices, start_date, end_date, 
                      corr_threshold=0.7, p_threshold=0.05, top_n=20):
    """
    相关性+协整双重筛选
    
    Parameters:
    -----------
    prices: pd.DataFrame, 股票价格数据
    start_date, end_date: str, 训练期
    corr_threshold: float, 相关性阈值
    p_threshold: float, p-value阈值
    top_n: int, 返回top N个配对
    
    Returns:
    --------
    pairs: list, 筛选出的配对列表
    """
    # 截取训练期数据
    train_data = prices.loc[start_date:end_date]
    
    # 计算收益率
    returns = train_data.pct_change().dropna()
    
    # 计算相关性矩阵
    corr_matrix = returns.corr()
    
    # 筛选高相关性配对
    high_corr_pairs = []
    tickers = returns.columns
    
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            corr = corr_matrix.loc[tickers[i], tickers[j]]
            
            if abs(corr) >= corr_threshold:
                # 进一步检验协整
                try:
                    result = engle_granger_test(
                        train_data[tickers[i]], 
                        train_data[tickers[j]],
                        verbose=False
                    )
                    
                    if result['is_cointegrated']:
                        spread = result['spread']
                        
                        high_corr_pairs.append({
                            'pair': f"{tickers[i]}-{tickers[j]}",
                            'ticker1': tickers[i],
                            'ticker2': tickers[j],
                            'correlation': corr,
                            'p_value': result['p_value'],
                            'hedge_ratio': result['hedge_ratio'],
                            'half_life': calculate_mean_reversion_speed(spread),
                            'spread_std': spread.std()
                        })
                except:
                    continue
    
    # 按相关性排序
    sorted_pairs = sorted(high_corr_pairs, 
                          key=lambda x: abs(x['correlation']), 
                          reverse=True)
    
    return sorted_pairs[:top_n]

# 示例
def example_correlation_method():
    """相关性法示例"""
    # 下载数据
    tickers = ['KO', 'PEP', 'XOM', 'CVX', 'WMT', 'TGT']
    prices = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 筛选配对
    pairs = correlation_method(prices, '2020-01-01', '2022-12-31',
                              corr_threshold=0.6, p_threshold=0.05, top_n=10)
    
    print("=== 相关性+协整法筛选结果 ===\n")
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. {pair['pair']}")
        print(f"   相关性: {pair['correlation']:.4f}")
        print(f"   p-value: {pair['p_value']:.4f}")
        print(f"   对冲比例: {pair['hedge_ratio']:.4f}")
        print(f"   半衰期: {pair['half_life']:.1f}天")
        print()
    
    return pairs

# 运行示例
# corr_pairs = example_correlation_method()
```

### 2.4 聚类法（Clustering Method）

使用机器学习聚类算法筛选潜在配对。

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def clustering_method(prices, start_date, end_date, 
                     n_clusters=5, method='hierarchical'):
    """
    聚类法筛选配对
    
    Parameters:
    -----------
    prices: pd.DataFrame, 股票价格数据
    start_date, end_date: str, 训练期
    n_clusters: int, 聚类数量
    method: str, 聚类方法 ('kmeans' or 'hierarchical')
    
    Returns:
    --------
    clusters: dict, 聚类结果
    pairs: list, 从聚类中提取的配对
    """
    # 截取训练期数据
    train_data = prices.loc[start_date:end_date]
    
    # 计算特征
    features = pd.DataFrame(index=train_data.columns)
    
    # 特征1：收益率均值
    returns = train_data.pct_change().dropna()
    features['return_mean'] = returns.mean()
    features['return_std'] = returns.std()
    
    # 特征2：Beta（相对于市场）
    market_return = returns.mean(axis=1)  # 简化：等权市场组合
    for ticker in features.index:
        beta = np.cov(returns[ticker], market_return)[0, 1] / np.var(market_return)
        features.loc[ticker, 'beta'] = beta
    
    # 特征3：市值（需要额外数据，这里用随机数替代）
    features['market_cap'] = np.random.lognormal(10, 1, len(features))
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 聚类
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:  # hierarchical
        model = AgglomerativeClustering(n_clusters=n_clusters)
    
    clusters = model.fit_predict(features_scaled)
    features['cluster'] = clusters
    
    # 从每个聚类内部筛选配对
    pairs = []
    for cluster_id in range(n_clusters):
        cluster_tickers = features[features['cluster'] == cluster_id].index
        
        if len(cluster_tickers) < 2:
            continue
        
        # 在聚类内部进行协整检验
        for i in range(len(cluster_tickers)):
            for j in range(i+1, len(cluster_tickers)):
                try:
                    result = engle_granger_test(
                        train_data[cluster_tickers[i]],
                        train_data[cluster_tickers[j]],
                        verbose=False
                    )
                    
                    if result['is_cointegrated']:
                        pairs.append({
                            'pair': f"{cluster_tickers[i]}-{cluster_tickers[j]}",
                            'ticker1': cluster_tickers[i],
                            'ticker2': cluster_tickers[j],
                            'cluster': cluster_id,
                            'p_value': result['p_value'],
                            'hedge_ratio': result['hedge_ratio']
                        })
                except:
                    continue
    
    print(f"=== 聚类法筛选结果 ({method}) ===")
    print(f"聚类数量: {n_clusters}")
    print(f"找到协整配对: {len(pairs)}个\n")
    
    for pair in pairs[:10]:  # 显示前10个
        print(f"  {pair['pair']} (Cluster {pair['cluster']})")
        print(f"    p-value: {pair['p_value']:.4f}")
        print(f"    对冲比例: {pair['hedge_ratio']:.4f}")
    
    return features, pairs

# 示例
def example_clustering_method():
    """聚类法示例"""
    # 下载数据
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN',
               'JPM', 'BAC', 'WFC', 'C', 'GS',
               'XOM', 'CVX', 'COP', 'SLB', 'HAL',
               'KO', 'PEP', 'WMT', 'TGT', 'PG']
    prices = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 聚类筛选
    features, pairs = clustering_method(
        prices, '2020-01-01', '2022-12-31',
        n_clusters=5, method='hierarchical'
    )
    
    return features, pairs

# 运行示例
# features, clustered_pairs = example_clustering_method()
```

## 三、交易信号构建

### 3.1 基于Z-Score的信号

最常用的交易信号：价差的Z-Score。

```python
def calculate_z_score_signal(spread, window=20, entry_threshold=2.0, 
                             exit_threshold=0.5):
    """
    基于Z-Score的交易信号
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    window: int, 滚动窗口
    entry_threshold: float, 入场阈值（Z-Score绝对值）
    exit_threshold: float, 出场阈值（Z-Score绝对值）
    
    Returns:
    --------
    signals: pd.DataFrame, 交易信号
    """
    # 计算滚动均值和标准差
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # 计算Z-Score
    z_score = (spread - spread_mean) / spread_std
    
    # 生成信号
    signals = pd.DataFrame(index=spread.index)
    signals['z_score'] = z_score
    signals['position'] = 0  # 0: 无仓位, 1: 多价差, -1: 空价差
    
    # 入场信号
    signals.loc[z_score > entry_threshold, 'position'] = -1  # 空价差（做空ticker1，做多ticker2）
    signals.loc[z_score < -entry_threshold, 'position'] = 1   # 多价差（做多ticker1，做空ticker2）
    
    # 出场信号（Z-Score回归）
    signals['position'] = signals['position'].replace(0, np.nan)
    signals['position'] = signals['position'].ffill()  # 向前填充持仓
    
    # 当Z-Score绝对值小于出场阈值时平仓
    exit_condition = (abs(z_score) < exit_threshold) & signals['position'].notna()
    signals.loc[exit_condition, 'position'] = 0
    
    # 再次向前填充（处理平仓后的NaN）
    signals['position'] = signals['position'].replace(0, np.nan)
    signals['position'] = signals['position'].ffill()
    signals['position'] = signals['position'].fillna(0)
    
    return signals

# 可视化信号
def visualize_signals(spread, signals):
    """
    可视化交易信号
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    signals: pd.DataFrame, 交易信号
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1：价差与Z-Score
    ax1 = axes[0]
    ax1.plot(signals.index, signals['z_score'], 
             label='Z-Score', color='blue', linewidth=2)
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, 
                label='Entry (+2σ)')
    ax1.axhline(y=-2.0, color='green', linestyle='--', alpha=0.5, 
                label='Entry (-2σ)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, 
                label='Exit (+0.5σ)')
    ax1.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5, 
                label='Exit (-0.5σ)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Z-Score of Spread', fontsize=14)
    ax1.set_ylabel('Z-Score', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：持仓信号
    ax2 = axes[1]
    ax2.plot(signals.index, signals['position'], 
             label='Position', color='purple', linewidth=2)
    ax2.fill_between(signals.index, 0, signals['position'], 
                     where=(signals['position'] > 0), 
                     alpha=0.3, color='green', label='Long Spread')
    ax2.fill_between(signals.index, 0, signals['position'], 
                     where=(signals['position'] < 0), 
                     alpha=0.3, color='red', label='Short Spread')
    ax2.set_title('Trading Position', fontsize=14)
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trading_signals.png', dpi=300, bbox_inches='tight')
    print("✓ 信号可视化已保存: trading_signals.png")

# 示例
def example_trading_signals():
    """交易信号示例"""
    # 使用前面协整检验的数据
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    
    # 计算价差
    result = engle_granger_test(data[tickers[0]], data[tickers[1]], verbose=False)
    spread = result['spread']
    
    # 生成信号
    signals = calculate_z_score_signal(
        spread, window=20, entry_threshold=2.0, exit_threshold=0.5
    )
    
    # 可视化
    visualize_signals(spread, signals)
    
    return signals

# 运行示例
# trading_signals = example_trading_signals()
```

### 3.2 基于Bollinger Bands的信号

布林带信号：价差突破±2倍标准差时入场。

```python
def calculate_bollinger_signals(spread, window=20, num_std=2.0, 
                                exit_threshold=0.5):
    """
    基于布林带的交易信号
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    window: int, 滚动窗口
    num_std: float, 标准差倍数
    exit_threshold: float, 出场阈值（占带宽度的比例）
    
    Returns:
    --------
    signals: pd.DataFrame, 交易信号
    """
    # 计算布林带
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    upper_band = spread_mean + num_std * spread_std
    lower_band = spread_mean - num_std * spread_std
    
    # 生成信号
    signals = pd.DataFrame(index=spread.index)
    signals['spread'] = spread
    signals['upper_band'] = upper_band
    signals['lower_band'] = lower_band
    signals['position'] = 0
    
    # 入场信号
    signals.loc[spread > upper_band, 'position'] = -1  # 空价差
    signals.loc[spread < lower_band, 'position'] = 1   # 多价差
    
    # 出场信号（价差回归到布林带中部）
    band_width = upper_band - lower_band
    middle_band = spread_mean
    
    exit_upper = middle_band + exit_threshold * (upper_band - middle_band)
    exit_lower = middle_band - exit_threshold * (middle_band - lower_band)
    
    exit_condition = ((signals['position'] == -1) & (spread <= exit_upper)) | \
                     ((signals['position'] == 1) & (spread >= exit_lower))
    
    signals.loc[exit_condition, 'position'] = 0
    
    # 向前填充
    signals['position'] = signals['position'].replace(0, np.nan)
    signals['position'] = signals['position'].ffill()
    signals['position'] = signals['position'].fillna(0)
    
    return signals

# 示例
def example_bollinger_signals():
    """布林带信号示例"""
    # 计算价差（使用前面的数据）
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    result = engle_granger_test(data[tickers[0]], data[tickers[1]], verbose=False)
    spread = result['spread']
    
    # 生成信号
    signals = calculate_bollinger_signals(
        spread, window=20, num_std=2.0, exit_threshold=0.5
    )
    
    # 可视化
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(signals.index, signals['spread'], 
            label='Spread', color='blue', linewidth=2)
    ax.plot(signals.index, signals['upper_band'], 
            label='Upper Band (+2σ)', color='red', linestyle='--', alpha=0.7)
    ax.plot(signals.index, signals['lower_band'], 
            label='Lower Band (-2σ)', color='green', linestyle='--', alpha=0.7)
    
    # 标注交易信号
    entry_long = signals[(signals['position'] == 1) & 
                         (signals['position'].shift(1) != 1)]
    entry_short = signals[(signals['position'] == -1) & 
                          (signals['position'].shift(1) != -1)]
    
    ax.scatter(entry_long.index, entry_long['spread'], 
               marker='^', color='green', s=100, label='Long Entry', zorder=5)
    ax.scatter(entry_short.index, entry_short['spread'], 
               marker='v', color='red', s=100, label='Short Entry', zorder=5)
    
    ax.fill_between(signals.index, 
                     signals['lower_band'], 
                     signals['upper_band'], 
                     alpha=0.1, color='gray')
    
    ax.set_title('Bollinger Bands Trading Signals', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Spread', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bollinger_signals.png', dpi=300, bbox_inches='tight')
    print("✓ 布林带信号已保存: bollinger_signals.png")
    
    return signals

# 运行示例
# bollinger_signals = example_bollinger_signals()
```

### 3.3 基于 Hurst 指数的信号

Hurst指数用于判断时间序列的记忆性，H < 0.5 表示均值回归。

```python
def calculate_hurst_exponent(series, max_lag=20):
    """
    计算Hurst指数
    
    Parameters:
    -----------
    series: pd.Series, 时间序列
    max_lag: int, 最大滞后阶数
    
    Returns:
    --------
    hurst: float, Hurst指数
    """
    lags = range(2, max_lag + 1)
    tau = []
    
    for lag in lags:
        # 计算滚动标准差
        std = series.rolling(window=lag).std().dropna()
        tau.append(np.std(std))
    
    # 拟合 log(lag) vs log(tau)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Hurst = 斜率
    hurst = poly[0]
    
    return hurst

def hurst_based_signal(spread, window=252, hurst_threshold=0.5):
    """
    基于Hurst指数的自适应信号
    
    Parameters:
    -----------
    spread: pd.Series, 价差序列
    window: int, 滚动窗口
    hurst_threshold: float, Hurst阈值（<0.5为均值回归）
    
    Returns:
    --------
    signals: pd.DataFrame, 交易信号
    """
    # 计算滚动Hurst指数
    hurst_series = pd.Series(index=spread.index[window:])
    for i in range(window, len(spread)):
        window_data = spread.iloc[i-window:i]
        hurst = calculate_hurst_exponent(window_data)
        hurst_series.iloc[i-window] = hurst
    
    # 只有当Hurst < threshold时才交易（均值回归）
    mean_reverting = hurst_series < hurst_threshold
    
    # 生成Z-Score信号（仅在均值回归时）
    signals = calculate_z_score_signal(spread[window:], window=20)
    signals['hurst'] = hurst_series
    signals['tradable'] = mean_reverting
    
    # 非均值回归时期禁用信号
    signals.loc[~signals['tradable'], 'position'] = 0
    
    return signals

# 示例
def example_hurst_signal():
    """Hurst指数信号示例"""
    # 计算价差
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    result = engle_granger_test(data[tickers[0]], data[tickers[1]], verbose=False)
    spread = result['spread']
    
    # 生成信号
    signals = hurst_based_signal(spread, window=252, hurst_threshold=0.5)
    
    print("=== Hurst指数分析 ===")
    print(f"平均Hurst指数: {signals['hurst'].mean():.4f}")
    print(f"均值回归时期占比: {(signals['tradable'].sum() / len(signals)):.2%}")
    
    return signals

# 运行示例
# hurst_signals = example_hurst_signal()
```

## 四、回测框架与绩效评估

### 4.1 回测引擎

完整的配对交易回测框架。

```python
class PairTradingBacktest:
    """配对交易回测引擎"""
    
    def __init__(self, prices, pair, hedge_ratio, initial_capital=1e6, 
                 transaction_cost=0.001):
        """
        初始化回测
        
        Parameters:
        -----------
        prices: pd.DataFrame, 价格数据
        pair: tuple, (ticker1, ticker2)
        hedge_ratio: float, 对冲比例
        initial_capital: float, 初始资金
        transaction_cost: float, 交易成本（单边）
        """
        self.prices = prices
        self.ticker1 = pair[0]
        self.ticker2 = pair[1]
        self.hedge_ratio = hedge_ratio
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # 计算价差
        self.spread = prices[self.ticker1] - hedge_ratio * prices[self.ticker2]
        
        # 回测结果
        self.portfolio = None
        
    def run_backtest(self, signals):
        """
        运行回测
        
        Parameters:
        -----------
        signals: pd.DataFrame, 交易信号（包含position列）
        
        Returns:
        --------
        portfolio: pd.DataFrame, 投资组合价值序列
        """
        # 初始化投资组合
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['cash'] = self.initial_capital
        portfolio['position'] = signals['position']
        portfolio['shares1'] = 0  # ticker1的持仓
        portfolio['shares2'] = 0  # ticker2的持仓
        portfolio['portfolio_value'] = self.initial_capital
        
        # 交易成本追踪
        portfolio['transaction_cost'] = 0.0
        
        # 遍历每个交易日
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i-1]
            
            # 获取当前价格和信号
            price1 = self.prices[self.ticker1].loc[date]
            price2 = self.prices[self.ticker2].loc[date]
            
            current_position = portfolio['position'].loc[date]
            prev_position = portfolio['position'].loc[prev_date]
            
            # 如果信号变化，调整仓位
            if current_position != prev_position:
                # 平仓旧仓位
                if prev_position != 0:
                    # 卖出ticker1
                    portfolio.loc[date, 'cash'] += \
                        portfolio.loc[prev_date, 'shares1'] * price1 * \
                        (1 - self.transaction_cost)
                    
                    # 买入/卖出ticker2（对冲）
                    portfolio.loc[date, 'cash'] += \
                        portfolio.loc[prev_date, 'shares2'] * price2 * \
                        (1 - self.transaction_cost)
                    
                    portfolio.loc[date, 'transaction_cost'] += \
                        abs(portfolio.loc[prev_date, 'shares1'] * price1 + \
                            portfolio.loc[prev_date, 'shares2'] * price2) * \
                        self.transaction_cost
                
                # 开仓新仓位
                if current_position != 0:
                    # 计算目标持仓（等价值）
                    portfolio_value = portfolio['portfolio_value'].loc[prev_date]
                    
                    if current_position == 1:  # 多价差：做多ticker1，做空ticker2
                        target_value1 = portfolio_value / 2
                        target_value2 = -portfolio_value / 2 * self.hedge_ratio
                    else:  # 空价差：做空ticker1，做多ticker2
                        target_value1 = -portfolio_value / 2
                        target_value2 = portfolio_value / 2 * self.hedge_ratio
                    
                    # 计算持仓股数
                    shares1 = target_value1 / price1
                    shares2 = target_value2 / price2
                    
                    portfolio.loc[date, 'shares1'] = shares1
                    portfolio.loc[date, 'shares2'] = shares2
                    
                    # 更新现金
                    portfolio.loc[date, 'cash'] -= \
                        (abs(shares1) * price1 + abs(shares2) * price2) * \
                        (1 + self.transaction_cost)
                    
                    portfolio.loc[date, 'transaction_cost'] += \
                        (abs(shares1) * price1 + abs(shares2) * price2) * \
                        self.transaction_cost
            else:
                # 保持仓位不变
                portfolio.loc[date, 'shares1'] = portfolio.loc[prev_date, 'shares1']
                portfolio.loc[date, 'shares2'] = portfolio.loc[prev_date, 'shares2']
                portfolio.loc[date, 'cash'] = portfolio.loc[prev_date, 'cash']
            
            # 计算当日投资组合价值
            portfolio_value = portfolio.loc[date, 'cash'] + \
                             portfolio.loc[date, 'shares1'] * price1 + \
                             portfolio.loc[date, 'shares2'] * price2
            
            portfolio.loc[date, 'portfolio_value'] = portfolio_value
        
        self.portfolio = portfolio
        
        return portfolio
    
    def calculate_performance(self):
        """
        计算绩效指标
        
        Returns:
        --------
        metrics: dict, 绩效指标字典
        """
        if self.portfolio is None:
            raise ValueError("请先运行回测！")
        
        # 计算收益率
        portfolio_value = self.portfolio['portfolio_value']
        returns = portfolio_value.pct_change().dropna()
        
        # 总收益
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        
        # 年化收益
        days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
        annual_return = (1 + total_return / 100) ** (365 / days) - 1
        annual_return *= 100
        
        # 夏普比率
        risk_free_rate = 0.02 / 252  # 假设无风险利率2%
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        winning_days = (returns > 0).sum()
        win_rate = winning_days / len(returns) * 100
        
        # 总交易成本
        total_cost = self.portfolio['transaction_cost'].sum()
        cost_ratio = total_cost / self.initial_capital * 100
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_cost': total_cost,
            'cost_ratio': cost_ratio,
            'num_trades': (self.portfolio['position'].diff() != 0).sum()
        }
        
        return metrics
    
    def plot_results(self):
        """可视化回测结果"""
        if self.portfolio is None:
            raise ValueError("请先运行回测！")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 子图1：投资组合价值
        ax1 = axes[0]
        ax1.plot(self.portfolio.index, self.portfolio['portfolio_value'], 
                linewidth=2, color='blue')
        ax1.set_title('Portfolio Value Over Time', fontsize=14)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：累计收益
        ax2 = axes[1]
        cumulative_returns = self.portfolio['portfolio_value'] / \
                            self.portfolio['portfolio_value'].iloc[0] - 1
        ax2.plot(cumulative_returns.index, cumulative_returns * 100, 
                linewidth=2, color='green')
        ax2.set_title('Cumulative Returns (%)', fontsize=14)
        ax2.set_ylabel('Returns (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 子图3：回撤
        ax3 = axes[2]
        cumulative = (1 + self.portfolio['portfolio_value'].pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        
        ax3.fill_between(drawdown.index, 0, drawdown, 
                        alpha=0.3, color='red', label='Drawdown')
        ax3.plot(drawdown.index, drawdown, 
                linewidth=1, color='darkred')
        ax3.set_title('Drawdown (%)', fontsize=14)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        print("✓ 回测结果已保存: backtest_results.png")

# 示例：完整回测流程
def example_backtest():
    """完整回测示例"""
    # 1. 下载数据
    tickers = ['KO', 'PEP']
    prices = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    
    # 2. 协整检验
    result = engle_granger_test(prices[tickers[0]], prices[tickers[1]])
    hedge_ratio = result['hedge_ratio']
    
    # 3. 生成交易信号
    spread = result['spread']
    signals = calculate_z_score_signal(spread, window=20, 
                                      entry_threshold=2.0, 
                                      exit_threshold=0.5)
    
    # 4. 初始化回测
    backtest = PairTradingBacktest(
        prices, 
        pair=(tickers[0], tickers[1]),
        hedge_ratio=hedge_ratio,
        initial_capital=1e6,
        transaction_cost=0.001
    )
    
    # 5. 运行回测
    portfolio = backtest.run_backtest(signals)
    
    # 6. 计算绩效
    metrics = backtest.calculate_performance()
    
    print("=== 回测绩效 ===")
    print(f"总收益率: {metrics['total_return']:.2f}%")
    print(f"年化收益率: {metrics['annual_return']:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"胜率: {metrics['win_rate']:.2f}%")
    print(f"交易次数: {metrics['num_trades']}")
    print(f"交易成本: ${metrics['total_cost']:,.2f} ({metrics['cost_ratio']:.2f}%)")
    
    # 7. 可视化
    backtest.plot_results()
    
    return backtest, metrics

# 运行示例
# backtest, performance = example_backtest()
```

## 五、风险管理

### 5.1 止损策略

配对交易中的止损机制。

```python
def add_stop_loss(signals, spread, stop_loss_threshold=3.0, 
                  stop_loss_type='z_score'):
    """
    添加止损机制
    
    Parameters:
    -----------
    signals: pd.DataFrame, 交易信号
    spread: pd.Series, 价差序列
    stop_loss_threshold: float, 止损阈值
    stop_loss_type: str, 止损类型 ('z_score', 'spread_value', 'time')
    
    Returns:
    --------
    signals_with_stoploss: pd.DataFrame, 更新后的信号
    """
    signals = signals.copy()
    
    if stop_loss_type == 'z_score':
        # 基于Z-Score的止损
        spread_mean = spread.rolling(window=20).mean()
        spread_std = spread.rolling(window=20).std()
        z_score = (spread - spread_mean) / spread_std
        
        # 当Z-Score超过止损阈值时强制平仓
        stop_condition = abs(z_score) > stop_loss_threshold
        
    elif stop_loss_type == 'spread_value':
        # 基于价差绝对值的止损
        spread_percentile = spread.rolling(window=252).apply(
            lambda x: np.percentile(x, 95)
        )
        stop_condition = abs(spread) > spread_percentile
        
    elif stop_loss_type == 'time':
        # 基于持仓时间的止损
        position_start = signals['position'].replace(0, np.nan).ffill()
        holding_period = position_start.notna().cumsum() - \
                        position_start.notna().cumsum().where(
                            signals['position'] == 0
                        ).ffill().fillna(0)
        stop_condition = holding_period > stop_loss_threshold
        
    else:
        raise ValueError("未知的止损类型")
    
    # 应用止损
    signals.loc[stop_condition, 'position'] = 0
    
    # 向前填充（保持平仓状态）
    signals['position'] = signals['position'].replace(0, np.nan)
    signals['position'] = signals['position'].ffill()
    signals['position'] = signals['position'].fillna(0)
    
    return signals

# 示例
def example_stop_loss():
    """止损示例"""
    # 生成信号
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    result = engle_granger_test(data[tickers[0]], data[tickers[1]], verbose=False)
    spread = result['spread']
    
    signals = calculate_z_score_signal(spread, window=20)
    
    # 添加止损
    signals_with_stoploss = add_stop_loss(
        signals, spread, 
        stop_loss_threshold=3.0, 
        stop_loss_type='z_score'
    )
    
    # 比较有无止损的表现
    print("=== 止损效果对比 ===")
    print(f"无止损交易次数: {(signals['position'].diff() != 0).sum()}")
    print(f"有止损交易次数: {(signals_with_stoploss['position'].diff() != 0).sum()}")
    
    return signals_with_stoploss

# 运行示例
# signals_stoploss = example_stop_loss()
```

### 5.2 仓位管理

动态调整仓位大小。

```python
def dynamic_position_sizing(signals, spread, volatility_window=20, 
                            max_position=1.0, vol_target=0.15):
    """
    动态仓位管理（基于波动率目标）
    
    Parameters:
    -----------
    signals: pd.DataFrame, 交易信号
    spread: pd.Series, 价差序列
    volatility_window: int, 波动率计算窗口
    max_position: float, 最大仓位
    vol_target: float, 目标波动率（年化）
    
    Returns:
    --------
    position_sizes: pd.Series, 动态仓位大小
    """
    # 计算价差收益率
    spread_return = spread.pct_change()
    
    # 计算滚动波动率（年化）
    volatility = spread_return.rolling(window=volatility_window).std() * np.sqrt(252)
    
    # 计算目标仓位（使波动率等于目标）
    target_position = vol_target / volatility
    
    # 限制仓位范围
    target_position = target_position.clip(upper=max_position)
    
    # 与信号结合
    position_sizes = target_position * signals['position'].abs()
    
    return position_sizes

# 集成到回测框架
class PairTradingBacktestWithRisk(PairTradingBacktest):
    """带风险管理的配对交易回测"""
    
    def run_backtest_with_risk_management(self, signals, 
                                           stop_loss_threshold=3.0,
                                           vol_target=0.15):
        """
        运行带风险管理的回测
        
        Parameters:
        -----------
        signals: pd.DataFrame, 交易信号
        stop_loss_threshold: float, 止损阈值
        vol_target: float, 目标波动率
        
        Returns:
        --------
        portfolio: pd.DataFrame, 投资组合价值序列
        """
        # 添加止损
        signals = add_stop_loss(
            signals, self.spread, 
            stop_loss_threshold=stop_loss_threshold,
            stop_loss_type='z_score'
        )
        
        # 动态仓位
        position_sizes = dynamic_position_sizing(
            signals, self.spread,
            volatility_window=20,
            max_position=1.0,
            vol_target=vol_target
        )
        
        # 运行回测（使用动态仓位）
        # ... (类似之前的run_backtest，但加入position_sizes)
        
        # 这里简化为调用父类方法
        portfolio = self.run_backtest(signals)
        
        return portfolio
```

## 六、实证研究与案例分析

### 6.1 经典配对：可口可乐 vs 百事可乐

```python
def case_study_ko_pep():
    """案例研究：KO-PEP配对交易"""
    
    print("="*60)
    print("案例研究：可口可乐 (KO) vs 百事可乐 (PEP)")
    print("="*60)
    
    # 1. 数据获取
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2015-01-01', end='2026-06-16')['Adj Close']
    
    # 2. 协整检验
    print("\n1. 协整检验")
    print("-" * 60)
    result = engle_granger_test(data[tickers[0]], data[tickers[1]])
    
    # 3. 配对筛选指标
    print("\n2. 配对质量指标")
    print("-" * 60)
    spread = result['spread']
    half_life = calculate_mean_reversion_speed(spread)
    sharpe = calculate_spread_sharpe(spread)
    
    print(f"半衰期: {half_life:.1f} 天")
    print(f"价差Sharpe比率: {sharpe:.2f}")
    print(f"价差波动率: {spread.std():.4f}")
    
    # 4. 交易信号生成
    print("\n3. 交易信号")
    print("-" * 60)
    signals = calculate_z_score_signal(
        spread, window=20, 
        entry_threshold=2.0, 
        exit_threshold=0.5
    )
    
    num_trades = (signals['position'].diff() != 0).sum() / 2
    print(f"交易次数: {num_trades:.0f}")
    print(f"平均持仓时间: {len(spread) / num_trades:.1f} 天")
    
    # 5. 回测
    print("\n4. 回测结果")
    print("-" * 60)
    backtest = PairTradingBacktest(
        data, pair=(tickers[0], tickers[1]),
        hedge_ratio=result['hedge_ratio'],
        initial_capital=1e6
    )
    
    portfolio = backtest.run_backtest(signals)
    metrics = backtest.calculate_performance()
    
    print(f"总收益率: {metrics['total_return']:.2f}%")
    print(f"年化收益率: {metrics['annual_return']:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"胜率: {metrics['win_rate']:.2f}%")
    
    # 6. 可视化
    print("\n5. 生成图表...")
    backtest.plot_results()
    
    return backtest, metrics

# 运行案例研究
# ko_pep_backtest, ko_pep_metrics = case_study_ko_pep()
```

### 6.2 配对交易在不同市场环境中的表现

```python
def performance_by_market_regime(prices, pair, hedge_ratio, signals):
    """
    分析不同市场环境下的表现
    
    Parameters:
    -----------
    prices: pd.DataFrame, 价格数据
    pair: tuple, (ticker1, ticker2)
    hedge_ratio: float, 对冲比例
    signals: pd.DataFrame, 交易信号
    """
    # 计算市场收益率（等权平均）
    market_return = prices.pct_change().mean(axis=1)
    
    # 定义市场状态
    market_vol = market_return.rolling(window=20).std() * np.sqrt(252)
    high_vol = market_vol > market_vol.median()
    
    bull_market = market_return.rolling(window=60).mean() > 0
    bear_market = ~bull_market
    
    # 运行回测
    backtest = PairTradingBacktest(
        prices, pair=pair, hedge_ratio=hedge_ratio
    )
    portfolio = backtest.run_backtest(signals)
    returns = portfolio['portfolio_value'].pct_change()
    
    # 分环境统计
    regimes = {
        'Bull-Low Vol': bull_market & ~high_vol,
        'Bull-High Vol': bull_market & high_vol,
        'Bear-Low Vol': bear_market & ~high_vol,
        'Bear-High Vol': bear_market & high_vol
    }
    
    print("=== 不同市场环境下的表现 ===\n")
    for regime_name, regime_mask in regimes.items():
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            annual_return = regime_returns.mean() * 252 * 100
            sharpe = np.sqrt(252) * regime_returns.mean() / regime_returns.std()
            
            print(f"{regime_name}:")
            print(f"  样本数: {len(regime_returns)}")
            print(f"  年化收益: {annual_return:.2f}%")
            print(f"  夏普比率: {sharpe:.2f}")
            print()
    
    return regimes

# 示例
def example_market_regime():
    """市场环境分析示例"""
    # 数据
    tickers = ['KO', 'PEP', 'SPY']  # 加入SPY作为市场代理
    prices = yf.download(tickers, start='2020-01-01', end='2026-06-16')['Adj Close']
    
    # 协整检验
    result = engle_granger_test(prices['KO'], prices['PEP'], verbose=False)
    
    # 信号
    signals = calculate_z_score_signal(result['spread'], window=20)
    
    # 分析
    regimes = performance_by_market_regime(
        prices[['KO', 'PEP']], 
        pair=('KO', 'PEP'),
        hedge_ratio=result['hedge_ratio'],
        signals=signals
    )
    
    return regimes

# 运行示例
# market_regimes = example_market_regime()
```

## 七、总结与未来方向

### 7.1 核心要点

1. **理论基础**：协整是配对交易的核心，确保价差的平稳性
2. **配对筛选**：距离法、协整法、相关性法、聚类法各有优劣
3. **信号构建**：Z-Score、布林带、Hurst指数是常用方法
4. **风险管理**：止损、仓位管理、市场环境适应至关重要
5. **实证表现**：配对交易在均值回归市场中表现优异，但在趋势市场中可能亏损

### 7.2 未来研究方向

1. **高频配对交易**：利用分钟级或秒级数据进行套利
2. **机器学习增强**：使用LSTM、Transformer预测价差方向
3. **多因子配对**：加入基本面、动量等因子优化配对筛选
4. **动态对冲比例**：时变对冲比例（Kalman Filter）
5. **跨市场配对**：不同交易所、不同国家的相似股票

### 7.3 实践建议

1. **严格筛选**：不要为了交易而交易，确保配对质量
2. **分散投资**：同时交易多个不相关配对，降低风险
3. **持续监控**：定期重新检验协整关系（结构突变）
4. **成本控制**：配对交易频繁交易，交易成本影响大
5. **风险优先**：始终设置止损，避免黑天鹅事件

---

**参考文献**：
1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
3. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). "Pairs trading"

**免责声明**：本文仅供学术交流和策略研究，不构成投资建议。配对交易虽为市场中性策略，但仍存在风险，包括模型风险、执行风险、流动性风险等。实盘交易前请充分测试并咨询专业人士。
