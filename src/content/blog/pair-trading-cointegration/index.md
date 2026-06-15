---
title: "配对交易与协整分析：均值回归策略的统计学基础"
description: "深入探讨配对交易的理论基础——协整关系，介绍Engle-Granger检验、Johansen检验等协整检验方法，并提供完整的Python实现代码。"
date: 2026-06-16
tags: ["配对交易", "协整分析", "均值回归", "统计套利", "量化策略"]
categories: ["量化交易"]
slug: "pair-trading-cointegration"
---

# 配对交易与协整分析：均值回归策略的统计学基础

## 引言

配对交易（Pairs Trading）是最经典的统计套利策略之一，其核心理念是"买入被低估的，卖出被高估的"。但如何科学地识别两个资产之间的长期均衡关系？答案是协整分析（Cointegration Analysis）。本文将深入探讨配对交易的理论基础，并提供完整的Python实现。

## 配对交易的基本原理

### 什么是配对交易？

配对交易寻找两个价格走势高度相关但暂时偏离均衡的资产，当价差扩大时做多低估资产、做空高估资产，等待价差回归时平仓获利。

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

# 下载示例数据：可口可乐 vs 百事可乐
def download_pair_data(ticker1, ticker2, start_date, end_date):
    """
    下载配对股票数据
    """
    stock1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)['Adj Close']
    stock2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)['Adj Close']
    
    # 合并数据
    pair_data = pd.DataFrame({
        ticker1: stock1,
        ticker2: stock2
    })
    
    # 计算对数价格
    pair_data['log_' + ticker1] = np.log(pair_data[ticker1])
    pair_data['log_' + ticker2] = np.log(pair_data[ticker2])
    
    # 计算价差
    pair_data['spread'] = pair_data['log_' + ticker1] - pair_data['log_' + ticker2]
    
    return pair_data

# 示例：KO（可口可乐） vs PEP（百事可乐）
pair_data = download_pair_data('KO', 'PEP', '2020-01-01', '2024-12-31')

print(f"数据形状: {pair_data.shape}")
print(f"\n价格相关性: {pair_data[['KO', 'PEP']].corr().iloc[0, 1]:.4f}")
print(f"对数价格相关性: {pair_data[['log_KO', 'log_PEP']].corr().iloc[0, 1]:.4f}")
```

### 可视化配对关系

```python
def plot_pair_relationship(pair_data, ticker1, ticker2):
    """
    可视化配对股票的关系
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 价格走势对比
    ax1 = axes[0]
    color1, color2 = 'tab:blue', 'tab:red'
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{ticker1} Price', color=color1)
    ax1.plot(pair_data.index, pair_data[ticker1], color=color1, linewidth=2, label=ticker1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax1b = ax1.twinx()
    ax1b.set_ylabel(f'{ticker2} Price', color=color2)
    ax1b.plot(pair_data.index, pair_data[ticker2], color=color2, linewidth=2, label=ticker2)
    ax1b.tick_params(axis='y', labelcolor=color2)
    
    # 2. 对数价格价差
    axes[1].plot(pair_data.index, pair_data['spread'], linewidth=2, color='purple')
    axes[1].axhline(y=pair_data['spread'].mean(), color='black', linestyle='--', 
                    label=f'Mean: {pair_data["spread"].mean():.4f}')
    axes[1].fill_between(pair_data.index, 
                         pair_data['spread'].mean() - 2*pair_data['spread'].std(),
                         pair_data['spread'].mean() + 2*pair_data['spread'].std(),
                         alpha=0.2, color='gray', label='±2 Std')
    axes[1].set_ylabel('Log Price Spread')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 价差分布直方图
    axes[2].hist(pair_data['spread'], bins=50, edgecolor='black', alpha=0.7, density=True)
    axes[2].axvline(x=pair_data['spread'].mean(), color='red', linestyle='--', 
                    linewidth=2, label='Mean')
    axes[2].axvline(x=pair_data['spread'].mean() + 2*pair_data['spread'].std(), 
                    color='orange', linestyle='--', linewidth=2, label='+2σ')
    axes[2].axvline(x=pair_data['spread'].mean() - 2*pair_data['spread'].std(), 
                    color='orange', linestyle='--', linewidth=2, label='-2σ')
    axes[2].set_xlabel('Spread')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Pairs Trading Analysis: {ticker1} vs {ticker2}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'pair_analysis_{ticker1}_{ticker2}.png', dpi=300, bbox_inches='tight')
    
    return fig

# 生成图表
fig = plot_pair_relationship(pair_data, 'KO', 'PEP')
```

## 协整理论深度解析

### 什么是协整？

协整是指两个或多个非平稳时间序列的线性组合是平稳的。用数学语言表达：

如果 $X_t$ 和 $Y_t$ 都是 I(1) 过程（一阶单整），但存在参数 $\alpha$ 使得：

$$Z_t = Y_t - \alpha X_t$$

是 I(0) 过程（平稳），则称 $X_t$ 和 $Y_t$ 是协整的。

### 为什么协整重要？

1. **避免伪回归**：普通回归可能得出虚假的相关性
2. **捕捉长期均衡**：协整关系反映了经济基本面的联系
3. **提供交易信号**：价差围绕均衡值波动，提供明确的买卖信号

## 协整检验方法

### 方法1：Engle-Granger 两步法

这是最经典的协整检验方法。

```python
def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y: Series, 因变量
    x: Series, 自变量
    verbose: bool, 是否打印详细信息
    
    Returns:
    --------
    results: dict, 检验结果
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    
    # 第一步：OLS回归
    X = add_constant(x)
    model = OLS(y, X).fit()
    residuals = model.resid
    
    if verbose:
        print("=" * 60)
        print("Step 1: OLS Regression")
        print("=" * 60)
        print(model.summary())
    
    # 第二步：ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: ADF Test on Residuals")
        print("=" * 60)
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"Critical Values:")
        for key, value in adf_result[4].items():
            print(f"  {key}: {value:.4f}")
    
    # 判断协整
    is_cointegrated = adf_result[1] < 0.05  # p-value < 0.05
    
    results = {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_cointegrated': is_cointegrated,
        'hedge_ratio': model.params[1],
        'residuals': residuals
    }
    
    if verbose:
        print(f"\n{'✓' if is_cointegrated else '✗'} Cointegration: {is_cointegrated}")
    
    return results

# 对KO和PEP进行Engle-Granger检验
eg_result = engle_granger_test(pair_data['log_KO'], pair_data['log_PEP'])
```

### 方法2：Johansen 检验

Johansen检验可以同时检验多个协整关系，更适合多资产配对。

```python
def johansen_test(data, det_order=0, k_ar_diff=1, verbose=True):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    data: DataFrame, 多变量时间序列
    det_order: int, 确定性项顺序 (0: no constant, 1: constant, 2: linear trend)
    k_ar_diff: int, 滞后阶数
    verbose: bool, 是否打印详细信息
    
    Returns:
    --------
    results: dict, 检验结果
    """
    # 进行Johansen检验
    joh_result = coint_johansen(data, det_order, k_ar_diff)
    
    if verbose:
        print("=" * 60)
        print("Johansen Cointegration Test")
        print("=" * 60)
        print("\nTrace Statistic:")
        for i in range(len(joh_result.lr1)):
            print(f"  r<={i}: {joh_result.lr1[i]:.4f}")
        
        print("\nMax Eigenvalue Statistic:")
        for i in range(len(joh_result.lr2)):
            print(f"  r={i}: {joh_result.lr2[i]:.4f}")
        
        print("\nCritical Values (95%):")
        print(f"  Trace: {joh_result.cvt[:, 1]}")
        print(f"  Max Eigen: {joh_result.cvm[:, 1]}")
    
    # 判断协整秩
    n_cointegrating = np.sum(joh_result.lr1 > joh_result.cvt[:, 1])
    
    results = {
        'trace_statistic': joh_result.lr1,
        'max_eigen_statistic': joh_result.lr2,
        'critical_values_trace': joh_result.cvt,
        'critical_values_max_eigen': joh_result.cvm,
        'n_cointegrating_relations': n_cointegrating,
        'eigenvectors': joh_result.evec
    }
    
    if verbose:
        print(f"\nNumber of cointegrating relations: {n_cointegrating}")
    
    return results

# 对KO和PEP进行Johansen检验
log_prices = pair_data[['log_KO', 'log_PEP']]
joh_result = johansen_test(log_prices)
```

### 方法3：Phillips-Ouliaris 检验

Phillips-Ouliaris检验是Engle-Granger检验的改进版本，对小样本更稳健。

```python
def phillips_ouliaris_test(y, x, verbose=True):
    """
    Phillips-Ouliaris协整检验
    
    注意：需要安装arch包
    pip install arch
    """
    try:
        from arch.unitroot import PhillipsOuliaris
    except ImportError:
        print("Please install arch package: pip install arch")
        return None
    
    # 合并数据
    data = pd.concat([y, x], axis=1).dropna()
    
    # 进行PO检验
    po_test = PhillipsOuliaris(data, trend='c')
    
    if verbose:
        print("=" * 60)
        print("Phillips-Ouliaris Cointegration Test")
        print("=" * 60)
        print(po_test.summary())
    
    results = {
        'statistic': po_test.stat,
        'p_value': po_test.pvalue,
        'is_cointegrated': po_test.pvalue < 0.05
    }
    
    return results

# 示例（需要安装arch包）
# po_result = phillips_ouliaris_test(pair_data['log_KO'], pair_data['log_PEP'])
```

## 配对选择的系统化方法

### 1. 距离法（Distance Approach）

计算价格序列的标准化距离，选择距离最小的配对。

```python
def calculate_ssd_distance(price1, price2):
    """
    计算平方和距离（Sum of Squared Differences）
    
    这是Gatev et al. (2006)提出的方法
    """
    # 标准化价格
    norm_price1 = price1 / price1.iloc[0]
    norm_price2 = price2 / price2.iloc[0]
    
    # 计算距离
    distance = np.sum((norm_price1 - norm_price2) ** 2)
    
    return distance

def find_pairs_by_distance(universe, start_date, end_date, top_n=50):
    """
    使用距离法寻找配对
    
    Parameters:
    -----------
    universe: list, 股票代码列表
    start_date, end_date: str, 日期范围
    top_n: int, 返回的配对数量
    
    Returns:
    --------
    pairs: list, 配对列表 [(ticker1, ticker2, distance), ...]
    """
    from itertools import combinations
    
    # 下载所有股票价格
    print("Downloading stock data...")
    prices = {}
    for ticker in universe:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, 
                              progress=False)['Adj Close']
            if len(data) > 0:
                prices[ticker] = data
        except:
            continue
    
    print(f"Downloaded {len(prices)} stocks")
    
    # 计算所有配对的距惠
    pairs = []
    for ticker1, ticker2 in combinations(prices.keys(), 2):
        # 对齐数据
        combined = pd.concat([prices[ticker1], prices[ticker2]], axis=1).dropna()
        
        if len(combined) < 252:  # 至少需要1年数据
            continue
        
        distance = calculate_ssd_distance(combined.iloc[:, 0], combined.iloc[:, 1])
        correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        
        # 只保留高相关的配对
        if correlation > 0.8:
            pairs.append((ticker1, ticker2, distance, correlation))
    
    # 按距离排序
    pairs.sort(key=lambda x: x[2])
    
    return pairs[:top_n]

# 示例：在S&P 500成分股中寻找配对
# universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT']
# pairs = find_pairs_by_distance(universe, '2020-01-01', '2024-12-31')
```

### 2. 协整评分法（Cointegration Ranking）

直接基于协整检验的p-value对配对排序。

```python
def rank_pairs_by_cointegration(universe, start_date, end_date, top_n=20):
    """
    使用协整检验p-value对配对排序
    """
    from itertools import combinations
    from joblib import Parallel, delayed
    
    def test_pair_cointegration(ticker1, ticker2):
        """测试单个配对的协整性"""
        try:
            # 下载数据
            data1 = yf.download(ticker1, start=start_date, end=end_date, 
                               progress=False)['Adj Close']
            data2 = yf.download(ticker2, start=start_date, end=end_date, 
                               progress=False)['Adj Close']
            
            # 对齐数据
            combined = pd.concat([data1, data2], axis=1).dropna()
            
            if len(combined) < 252:
                return None
            
            # Engle-Granger检验
            result = engle_granger_test(combined.iloc[:, 0], 
                                       combined.iloc[:, 1], 
                                       verbose=False)
            
            if result['is_cointegrated']:
                return (ticker1, ticker2, result['p_value'], result['hedge_ratio'])
            
        except:
            return None
        
        return None
    
    print("Testing cointegration for all pairs...")
    
    # 并行计算（加速）
    pairs_to_test = list(combinations(universe, 2))
    results = Parallel(n_jobs=-1)(
        delayed(test_pair_cointegration)(t1, t2) 
        for t1, t2 in pairs_to_test
    )
    
    # 过滤掉None结果
    cointegrated_pairs = [r for r in results if r is not None]
    
    # 按p-value排序（越小越好）
    cointegrated_pairs.sort(key=lambda x: x[2])
    
    print(f"Found {len(cointegrated_pairs)} cointegrated pairs")
    
    return cointegrated_pairs[:top_n]

# 示例
# top_pairs = rank_pairs_by_cointegration(universe, '2020-01-01', '2024-12-31')
```

### 3. 聚类法（Clustering Approach）

使用机器学习聚类算法寻找相似的股票。

```python
from sklearn.cluster import KMeans, HierarchicalClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

def find_pairs_by_clustering(features_df, n_clusters=10, pairs_per_cluster=5):
    """
    使用聚类方法寻找配对
    
    Parameters:
    -----------
    features_df: DataFrame, 股票特征矩阵（收益率、波动率、市值等）
    n_clusters: int, 聚类数量
    pairs_per_cluster: int, 每个聚类中选择多少对
    
    Returns:
    --------
    pairs: list, 配对列表
    """
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # 在每个聚类内寻找距离最近的配对
    pairs = []
    for cluster_id in range(n_clusters):
        # 获取该聚类内的股票
        cluster_stocks = features_df.index[clusters == cluster_id].tolist()
        
        if len(cluster_stocks) < 2:
            continue
        
        # 计算聚类内所有股票的距离矩阵
        cluster_features = scaled_features[clusters == cluster_id]
        distance_matrix = euclidean_distances(cluster_features)
        
        # 转换为DataFrame
        distance_df = pd.DataFrame(distance_matrix, 
                                  index=cluster_stocks, 
                                  columns=cluster_stocks)
        
        # 找到距离最小的配对
        for _ in range(min(pairs_per_cluster, len(cluster_stocks)//2)):
            # 找到最小距离（排除对角线）
            np.fill_diagonal(distance_df.values, np.inf)
            min_idx = np.unravel_index(distance_df.values.argmin(), 
                                      distance_df.shape)
            
            stock1 = distance_df.index[min_idx[0]]
            stock2 = distance_df.columns[min_idx[1]]
            
            pairs.append((stock1, stock2, distance_df.iloc[min_idx]))
            
            # 移除已配对的股票
            distance_df = distance_df.drop(index=[stock1, stock2], 
                                          columns=[stock1, stock2])
            
            if distance_df.empty:
                break
    
    return pairs

# 构建特征矩阵示例
def build_feature_matrix(universe, start_date, end_date):
    """
    构建股票特征矩阵
    """
    features = {}
    
    for ticker in universe:
        try:
            # 下载数据
            stock_data = yf.download(ticker, start=start_date, end=end_date, 
                                    progress=False)
            
            # 计算特征
            returns = stock_data['Adj Close'].pct_change().dropna()
            
            features[ticker] = {
                'mean_return': returns.mean() * 252,
                'volatility': returns.std() * np.sqrt(252),
                'sharpe': returns.mean() / returns.std() * np.sqrt(252),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'max_drawdown': ((returns + 1).cumprod() / 
                                (returns + 1).cumprod().expanding().max() - 1).min()
            }
        except:
            continue
    
    feature_df = pd.DataFrame(features).T
    
    return feature_df

# 示例
# features = build_feature_matrix(universe, '2020-01-01', '2024-12-31')
# pairs = find_pairs_by_clustering(features, n_clusters=10)
```

## 交易信号的生成

### 基于Z-Score的信号

```python
def generate_trading_signals(spread, entry_z=2.0, exit_z=0.5, 
                            stop_loss_z=3.0, lookback=252):
    """
    基于Z-Score生成交易信号
    
    Parameters:
    -----------
    spread: Series, 价差序列
    entry_z: float, 入场Z-Score阈值
    exit_z: float, 出场Z-Score阈值
    stop_loss_z: float, 止损Z-Score阈值
    lookback: int, 滚动窗口
    
    Returns:
    --------
    signals: DataFrame, 交易信号
    """
    # 计算滚动Z-Score
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 初始化信号
    signals = pd.DataFrame(index=spread.index)
    signals['z_score'] = z_score
    signals['position'] = 0  # 0: 空仓, 1: 做多价差, -1: 做空价差
    
    # 生成信号
    for i in range(1, len(signals)):
        if signals['position'].iloc[i-1] == 0:  # 当前空仓
            if z_score.iloc[i] > entry_z:
                # 价差过高，做空价差（做空stock1，做多stock2）
                signals.loc[signals.index[i], 'position'] = -1
            elif z_score.iloc[i] < -entry_z:
                # 价差过低，做多价差（做多stock1，做空stock2）
                signals.loc[signals.index[i], 'position'] = 1
        
        elif signals['position'].iloc[i-1] == 1:  # 当前做多价差
            if abs(z_score.iloc[i]) < exit_z:
                # 价差回归，平仓
                signals.loc[signals.index[i], 'position'] = 0
            elif z_score.iloc[i] > stop_loss_z:
                # 止损
                signals.loc[signals.index[i], 'position'] = 0
            else:
                # 继续持有
                signals.loc[signals.index[i], 'position'] = 1
        
        elif signals['position'].iloc[i-1] == -1:  # 当前做空价差
            if abs(z_score.iloc[i]) < exit_z:
                # 价差回归，平仓
                signals.loc[signals.index[i], 'position'] = 0
            elif z_score.iloc[i] < -stop_loss_z:
                # 止损
                signals.loc[signals.index[i], 'position'] = 0
            else:
                # 继续持有
                signals.loc[signals.index[i], 'position'] = -1
    
    return signals

# 为KO-PEP生成交易信号
signals = generate_trading_signals(pair_data['spread'])

# 可视化交易信号
def plot_trading_signals(pair_data, signals, ticker1, ticker2):
    """
    可视化交易信号
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：价差和Z-Score
    ax1 = axes[0]
    ax1.plot(signals.index, signals['z_score'], linewidth=2, label='Z-Score')
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Entry (+2σ)')
    ax1.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Exit (+0.5σ)')
    ax1.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between(signals.index, 0, signals['z_score'], 
                     where=(signals['position'] == 1), 
                     alpha=0.3, color='green', label='Long Spread')
    ax1.fill_between(signals.index, 0, signals['z_score'], 
                     where=(signals['position'] == -1), 
                     alpha=0.3, color='red', label='Short Spread')
    ax1.set_ylabel('Z-Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：累计收益
    # 计算策略收益
    returns1 = pair_data[ticker1].pct_change()
    returns2 = pair_data[ticker2].pct_change()
    
    # 对冲比例
    hedge_ratio = eg_result['hedge_ratio']
    
    # 策略收益
    strategy_returns = (signals['position'].shift(1) * 
                       (returns1 - hedge_ratio * returns2))
    strategy_cumret = (1 + strategy_returns).cumprod()
    
    axes[1].plot(strategy_cumret.index, strategy_cumret.values, 
                linewidth=2, label='Pairs Trading Strategy')
    axes[1].set_ylabel('Cumulative Return')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Trading Signals: {ticker1} vs {ticker2}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'trading_signals_{ticker1}_{ticker2}.png', dpi=300, bbox_inches='tight')
    
    return fig

# 生成信号图
fig = plot_trading_signals(pair_data, signals, 'KO', 'PEP')
```

### 卡尔曼滤波动态对冲比率

传统的OLS回归使用固定的对冲比率，但现实中这个比率可能是时变的。卡尔曼滤波可以提供动态的 hedge ratio。

```python
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(y, x):
    """
    使用卡尔曼滤波估计时变对冲比率
    
    Parameters:
    -----------
    y: Series, 因变量（被做空的股票）
    x: Series, 自变量（被做多的股票）
    
    Returns:
    --------
    state_means: array, 时变对冲比率
    """
    # 准备观测矩阵（每个时刻的x值）
    observation_matrix = np.column_stack([x.values, np.ones(len(x))])
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(2),  # 状态转移矩阵（假设对冲比率随机游走）
        observation_matrices=observation_matrix,
        initial_state_mean=np.array([1.0, 0.0]),  # 初始对冲比率=1, 截距=0
        initial_state_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0,  # 观测噪声
        transition_covariance=np.eye(2) * 0.01  # 状态转移噪声
    )
    
    # 使用EM算法估计参数
    kf = kf.em(y.values, n_iter=10)
    
    # 滤波（得到时变对冲比率）
    state_means, state_covariances = kf.filter(y.values)
    
    return state_means[:, 0]  # 返回对冲比率

# 对KO-PEP应用卡尔曼滤波
dynamic_hedge_ratio = kalman_filter_hedge_ratio(pair_data['log_KO'], 
                                                  pair_data['log_PEP'])

# 计算动态价差
pair_data['dynamic_spread'] = (pair_data['log_KO'] - 
                                dynamic_hedge_ratio * pair_data['log_PEP'])

# 可视化静态vs动态对冲比率
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(pair_data.index, [eg_result['hedge_ratio']] * len(pair_data), 
        label='Static Hedge Ratio (OLS)', linewidth=2)
ax.plot(pair_data.index, dynamic_hedge_ratio, 
        label='Dynamic Hedge Ratio (Kalman Filter)', linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Hedge Ratio')
ax.legend()
ax.grid(True, alpha=0.3)
plt.title('Static vs Dynamic Hedge Ratio')
plt.tight_layout()
plt.savefig('hedge_ratio_comparison.png', dpi=300, bbox_inches='tight')
```

## 风险管理和绩效评估

### 1. 最大回撤控制

```python
def calculate_max_drawdown(cumulative_returns):
    """
    计算最大回撤
    """
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return max_drawdown, drawdown

def plot_drawdown(cumulative_returns, title='Drawdown Chart'):
    """
    绘制回撤图
    """
    _, drawdown = calculate_max_drawdown(cumulative_returns)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(drawdown.index, 0, drawdown.values, 
                    color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, 
            color='darkred', linewidth=1)
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

# 计算策略回撤
strategy_cumret = (1 + strategy_returns).cumprod()
fig_dd = plot_drawdown(strategy_cumret, 'Pairs Trading Strategy Drawdown')
```

### 2. 夏普比率和信息比率

```python
def calculate_performance_metrics(returns, benchmark_returns=None, risk_free_rate=0.02/252):
    """
    计算策略绩效指标
    """
    # 年化收益
    annual_return = returns.mean() * 252
    
    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # 最大回撤
    cumret = (1 + returns).cumprod()
    max_dd, _ = calculate_max_drawdown(cumret)
    
    # 胜率
    win_rate = (returns > 0).sum() / len(returns)
    
    # 盈亏比
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    metrics = {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }
    
    # 如果有基准，计算信息比率
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
        information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        metrics['information_ratio'] = information_ratio
    
    return metrics

# 计算策略绩效
performance = calculate_performance_metrics(strategy_returns)

print("=" * 60)
print("Strategy Performance Metrics")
print("=" * 60)
for key, value in performance.items():
    if key in ['annual_return', 'annual_volatility', 'max_drawdown', 
               'win_rate', 'profit_loss_ratio']:
        print(f"{key.replace('_', ' ').title()}: {value:.2%}")
    else:
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
```

### 3. 压力测试和敏感性分析

```python
def stress_test_pairs_strategy(pair_data, signals, stress_scenarios):
    """
    对配对策略进行压力测试
    
    Parameters:
    -----------
    pair_data: DataFrame, 配对数据
    signals: DataFrame, 交易信号
    stress_scenarios: dict, 压力情景 {'name': {'shock1': value, 'shock2': value}}
    
    Returns:
    --------
    stress_results: dict, 各情景下的策略表现
    """
    stress_results = {}
    
    for scenario_name, shocks in stress_scenarios.items():
        # 复制原始数据
        stressed_data = pair_data.copy()
        
        # 应用冲击
        for stock, shock in shocks.items():
            if stock in stressed_data.columns:
                stressed_data[stock] = stressed_data[stock] * (1 + shock)
        
        # 重新计算价差和收益
        stressed_spread = (np.log(stressed_data['KO']) - 
                          np.log(stressed_data['PEP']))
        
        # 重新计算策略收益（使用原始信号）
        stressed_returns1 = stressed_data['KO'].pct_change()
        stressed_returns2 = stressed_data['PEP'].pct_change()
        
        hedge_ratio = eg_result['hedge_ratio']
        stressed_strategy_returns = (signals['position'].shift(1) * 
                                     (stressed_returns1 - 
                                      hedge_ratio * stressed_returns2))
        
        # 计算绩效
        stressed_performance = calculate_performance_metrics(stressed_strategy_returns)
        
        stress_results[scenario_name] = stressed_performance
    
    return stress_results

# 定义压力情景
stress_scenarios = {
    'market_crash': {'KO': -0.20, 'PEP': -0.18},  # 市场崩盘，KO跌更多
    'sector_rotation': {'KO': -0.10, 'PEP': 0.05},  # 行业轮动，PEP表现更好
    'liquidity_crisis': {'KO': -0.15, 'PEP': -0.15},  # 流动性危机，同跌
    'regulatory_shock': {'KO': -0.25, 'PEP': -0.05}  # 监管冲击，KO受影响更大
}

# 进行压力测试
stress_results = stress_test_pairs_strategy(pair_data, signals, stress_scenarios)

# 打印压力测试结果
print("=" * 60)
print("Stress Test Results")
print("=" * 60)
for scenario, metrics in stress_results.items():
    print(f"\nScenario: {scenario}")
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
```

## 实战案例：A股配对交易

### 案例1：招商银行 vs 平安银行

```python
# 下载A股数据（需要使用tushare或akshare）
import akshare as ak

def download_a_stock_data(ticker1, ticker2, start_date, end_date):
    """
    下载A股数据（使用akshare）
    """
    # 获取股票日线数据
    stock1 = ak.stock_zh_a_hist(symbol=ticker1, period="daily", 
                                start_date=start_date, end_date=end_date, adjust="qfq")
    stock2 = ak.stock_zh_a_hist(symbol=ticker2, period="daily", 
                                start_date=start_date, end_date=end_date, adjust="qfq")
    
    # 重命名列
    stock1 = stock1.rename(columns={'收盘': 'close', '日期': 'date'})
    stock2 = stock2.rename(columns={'收盘': 'close', '日期': 'date'})
    
    # 设置索引
    stock1['date'] = pd.to_datetime(stock1['date'])
    stock2['date'] = pd.to_datetime(stock2['date'])
    stock1 = stock1.set_index('date')
    stock2 = stock2.set_index('date')
    
    # 合并数据
    a_share_pair = pd.DataFrame({
        ticker1: stock1['close'],
        ticker2: stock2['close']
    })
    
    # 计算对数价格
    a_share_pair['log_' + ticker1] = np.log(a_share_pair[ticker1])
    a_share_pair['log_' + ticker2] = np.log(a_share_pair[ticker2])
    
    return a_share_pair

# 示例：600036（招商银行） vs 000001（平安银行）
# a_share_data = download_a_stock_data('600036', '000001', '20200101', '20241231')
# a_share_result = engle_granger_test(a_share_data['log_600036'], a_share_data['log_000001'])
```

### 案例2：贵州茅台 vs 五粮液

高端白酒行业的两大龙头，业务模式相似，适合配对交易。

```python
# 600519（贵州茅台） vs 000858（五粮液）
# baijiu_data = download_a_stock_data('600519', '000858', '20200101', '20241231')

def analyze_a_share_pair(ticker1, ticker2, start_date, end_date):
    """
    完整的A股配对分析流程
    """
    # 1. 下载数据
    print(f"Downloading data for {ticker1} and {ticker2}...")
    data = download_a_stock_data(ticker1, ticker2, start_date, end_date)
    
    # 2. 协整检验
    print("\nPerforming cointegration test...")
    result = engle_granger_test(data['log_' + ticker1], data['log_' + ticker2])
    
    if not result['is_cointegrated']:
        print("✗ Not cointegrated. Try other pairs.")
        return None
    
    print("✓ Cointegrated! Proceeding to signal generation...")
    
    # 3. 计算价差
    data['spread'] = result['residuals']
    
    # 4. 生成交易信号
    signals = generate_trading_signals(data['spread'])
    
    # 5. 计算策略收益
    returns1 = data[ticker1].pct_change()
    returns2 = data[ticker2].pct_change()
    strategy_returns = (signals['position'].shift(1) * 
                       (returns1 - result['hedge_ratio'] * returns2))
    
    # 6. 计算绩效
    performance = calculate_performance_metrics(strategy_returns)
    
    # 7. 可视化
    fig = plot_trading_signals(data, signals, ticker1, ticker2)
    
    return {
        'data': data,
        'signals': signals,
        'performance': performance,
        'fig': fig
    }

# 运行完整分析
# result = analyze_a_share_pair('600519', '000858', '20200101', '20241231')
```

## 配对交易的陷阱与应对

### 陷阱1：结构性断裂

协整关系可能因为基本面变化而断裂（如行业监管、公司并购等）。

**应对方法**：
- 使用滚动窗口定期重新检验协整关系
- 设置协整关系失效的止损条件
- 结合基本面分析，避免纯统计套利

```python
def monitor_cointegration_breakdown(spread, window=252, threshold=0.05):
    """
    监测协整关系的断裂
    
    通过滚动ADF检验，如果发现p-value持续上升超过阈值，
    则发出警告
    """
    breakdown_signal = pd.Series(index=spread.index, dtype=bool)
    
    for i in range(window, len(spread)):
        window_data = spread.iloc[i-window:i]
        adf_result = adfuller(window_data, autolag='AIC')
        
        # 如果p-value超过阈值，标记可能的断裂
        if adf_result[1] > threshold:
            breakdown_signal.iloc[i] = True
        else:
            breakdown_signal.iloc[i] = False
    
    # 计算连续断裂的天数
    breakdown_days = breakdown_signal.rolling(window=20).sum()
    
    return breakdown_signal, breakdown_days

# 应用断裂监测
breakdown_signal, breakdown_days = monitor_cointegration_breakdown(pair_data['spread'])

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(breakdown_signal.index, breakdown_signal.astype(int), 
            linewidth=2, label='Breakdown Signal')
axes[0].axhline(y=0.5, color='red', linestyle='--', label='Breakdown')
axes[0].set_ylabel('Breakdown Signal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(breakdown_days.index, breakdown_days.values, 
            linewidth=2, color='red', label='Consecutive Breakdown Days')
axes[1].axhline(y=20, color='darkred', linestyle='--', 
               label='Stop Trading Threshold')
axes[1].set_ylabel('Days')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Cointegration Breakdown Monitoring', fontsize=16)
plt.tight_layout()
```

### 陷阱2：流动性风险

小盘股的配对交易可能面临流动性不足的问题，导致滑点过大。

**应对方法**：
- 只交易日成交额超过阈值的股票（如>1000万元）
- 在信号生成时加入流动性过滤
- 使用限价单而非市价单

```python
def add_liquidity_filter(ticker_list, min_volume=1e7, min_turnover=0.01):
    """
    流动性过滤
    
    Parameters:
    -----------
    ticker_list: list, 股票列表
    min_volume: float, 最小日成交额（元）
    min_turnover: float, 最小换手率
    
    Returns:
    --------
    liquid_tickers: list, 流动性达标的股票
    """
    liquid_tickers = []
    
    for ticker in ticker_list:
        try:
            # 下载数据（包含成交量）
            data = ak.stock_zh_a_hist(symbol=ticker, period="daily",
                                     start_date='20240101', end_date='20241231', 
                                     adjust="qfq")
            
            # 计算平均成交额和换手率
            avg_volume = (data['收盘'] * data['成交量']).mean()
            avg_turnover = data['换手率'].mean()
            
            if avg_volume > min_volume and avg_turnover > min_turnover:
                liquid_tickers.append(ticker)
                print(f"✓ {ticker}: Volume={avg_volume/1e8:.2f}亿, Turnover={avg_turnover:.2%}")
            else:
                print(f"✗ {ticker}: Insufficient liquidity")
                
        except:
            print(f"✗ {ticker}: Data unavailable")
    
    return liquid_tickers
```

### 陷阱3：交易成本侵蚀收益

频繁交易会导致交易成本（佣金、印花税、滑点）侵蚀利润。

**应对方法**：
- 提高入场阈值（如Z-Score从2.0提高到2.5）
- 降低调仓频率（如使用周度调仓而非日度）
- 优化执行算法（如VWAP、TWAP）

```python
def calculate_transaction_costs(returns, position_changes, 
                                commission=0.0003, stamp_tax=0.001, 
                                slippage=0.001):
    """
    计算交易成本
    
    Parameters:
    -----------
    returns: Series, 毛收益
    position_changes: Series, 仓位变化（绝对值）
    commission: float, 佣金费率
    stamp_tax: float, 印花税率（卖出时收取）
    slippage: float, 滑点
    
    Returns:
    --------
    net_returns: Series, 扣除成本后的净收益
    """
    # 佣金（双向收取）
    commission_cost = position_changes * commission * 2
    
    # 印花税（只在卖出时收取）
    sell_signal = position_changes < 0
    stamp_tax_cost = abs(position_changes) * stamp_tax * sell_signal
    
    # 滑点（假设按成交量固定比例）
    slippage_cost = position_changes * slippage
    
    # 总成本
    total_cost = commission_cost + stamp_tax_cost + slippage_cost
    
    # 净收益
    net_returns = returns - total_cost
    
    print("=" * 60)
    print("Transaction Cost Analysis")
    print("=" * 60)
    print(f"Total Commission: {commission_cost.sum():.2%}")
    print(f"Total Stamp Tax: {stamp_tax_cost.sum():.2%}")
    print(f"Total Slippage: {slippage_cost.sum():.2%}")
    print(f"Total Cost: {total_cost.sum():.2%}")
    print(f"Gross Return: {returns.sum():.2%}")
    print(f"Net Return: {net_returns.sum():.2%}")
    print(f"Cost Drag: {(returns.sum() - net_returns.sum()):.2%}")
    
    return net_returns

# 计算KO-PEP策略的交易成本
position_changes = signals['position'].diff().abs()
net_returns = calculate_transaction_costs(strategy_returns, position_changes)
```

## 总结

配对交易是一种基于统计套利的量化策略，其核心在于：

1. **科学选择配对**：使用协整检验、距离法、聚类法等方法
2. **精确建模**：Engle-Granger检验、Johansen检验、卡尔曼滤波等
3. **严谨回测**：考虑交易成本、流动性、压力测试等现实约束
4. **风险管理**：监测协整断裂、控制最大回撤、设置止损

虽然配对交易看似简单，但要在实盘中稳定盈利，需要深厚的数理功底和丰富的实战经验。希望本文能为你提供坚实的理论基础和实用的代码框架。

## 参考文献

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies.
2. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." Econometrica.
3. Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." Econometrica.
4. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). "Pairs Trading." Quantitative Finance.
5. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.

---

**免责声明**：本文仅供学术研究和交流使用，不构成任何投资建议。配对交易存在风险，包括协整关系断裂风险、流动性风险、模型风险等。在实际应用前，请充分理解策略原理并进行充分的回测和模拟盘测试。
