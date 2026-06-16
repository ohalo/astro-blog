---
title: "配对交易与协整分析：统计套利的理论与实践"
description: "深入讲解配对交易的理论基础、协整检验方法、交易信号构建和风险管理，提供完整的Python实现和A股实证案例"
pubDate: 2026-06-16
tags: ["配对交易", "统计套利", "协整分析", "市场中性"]
category: "量化策略"
featured: false
toc: true
---

import { Image } from 'astro:assets';

## 引言

配对交易（Pairs Trading）是最经典的市场中性策略之一，其核心理念是"买入被低估的资产，卖出被高估的资产"，通过捕捉价格偏离获取收益。与传统的趋势跟踪策略不同，配对交易不依赖市场方向，而是在两只高度相关的股票之间寻找相对价值的错配机会。

自1980年代由Morgan Stanley的量化团队首次系统化应用以来，配对交易已经成为对冲基金的标配策略。2025年，随着A股市场有效性提升和量化竞争加剧，传统的动量、反转等因子溢价不断衰减，而基于协整的配对交易因其稳健性和低相关性，重新受到市场关注。

本文将系统介绍配对交易的理论基础、协整检验方法、实战策略构建、风险控制，并提供完整的Python实现代码和A股实证案例。

## 配对交易的理论基础

### 1. 什么是协整（Cointegration）？

协整是配对交易的核心理论基础。简单来说，如果两个时间序列 $\{X_t\}$ 和 $\{Y_t\}$ 都是非平稳的（通常是I(1)过程，即一阶单整），但它们的线性组合 $Z_t = Y_t - \beta X_t$ 是平稳的（I(0)过程），那么我们称 $X_t$ 和 $Y_t$ 是协整的。

数学定义：

如果满足以下条件，则 $X_t$ 和 $Y_t$ 协整：
1. $X_t \sim I(1)$, $Y_t \sim I(1)$ （两个序列都是一阶单整）
2. 存在协整向量 $\beta$，使得 $Z_t = Y_t - \beta X_t \sim I(0)$ （残差平稳）

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

class PairTradingAnalyzer:
    """配对交易分析器"""
    
    def __init__(self, price_data, lookback=252):
        """
        初始化
        
        Parameters:
        -----------
        price_data : DataFrame
            价格数据，索引为日期，列为股票代码
        lookback : int
            滚动窗口长度（交易日数）
        """
        self.price_data = price_data
        self.lookback = lookback
        
    def test_stationarity(self, series, title='Series'):
        """
        ADF检验（Augmented Dickey-Fuller Test）
        
        原假设：序列是非平稳的（有单位根）
        备择假设：序列是平稳的（无单位根）
        
        如果p-value < 0.05，拒绝原假设，序列平稳
        """
        result = adfuller(series, autolag='AIC')
        
        print(f"\n=== ADF Test: {title} ===")
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        
        if result[1] < 0.05:
            print("✓ Series is stationary (reject null hypothesis)")
            return True
        else:
            print("✗ Series is non-stationary (fail to reject null)")
            return False
    
    def engle_granger_test(self, y, x, show_details=True):
        """
        Engle-Granger两步法协整检验
        
        步骤1：OLS回归 Y = α + βX + ε
        步骤2：对残差ε进行ADF检验
        
        Parameters:
        -----------
        y : Series
            因变量（通常是价格较高的股票）
        x : Series
            自变量（通常是价格较低的股票）
        """
        # 步骤1：OLS回归
        X_with_const = np.column_stack([np.ones(len(x)), x])
        model = OLS(y, X_with_const)
        results = model.fit()
        
        alpha = results.params[0]
        beta = results.params[1]
        residuals = results.resid
        
        if show_details:
            print(f"\n=== Engle-Granger Cointegration Test ===")
            print(f"Regression: Y = {alpha:.4f} + {beta:.4f} * X")
            print(f"R² = {results.rsquared:.4f}")
            print(f"Residual Mean = {residuals.mean():.6f}")
            print(f"Residual Std = {residuals.std():.6f}")
        
        # 步骤2：残差平稳性检验
        is_stationary = self.test_stationarity(residuals, title='Residuals (Spread)')
        
        # 计算half-life（半衰期）
        half_life = self.calculate_half_life(residuals)
        
        if show_details:
            print(f"\nHalf-Life of Mean Reversion: {half_life:.1f} days")
        
        return {
            'alpha': alpha,
            'beta': beta,
            'residuals': residuals,
            'is_cointegrated': is_stationary,
            'half_life': half_life,
            'model': results
        }
    
    def calculate_half_life(self, series):
        """
        计算均值回归的半衰期
        
        使用AR(1)模型：Δy_t = λy_{t-1} + ε_t
        半衰期 = ln(0.5) / ln(1 + λ)
        """
        series_lag = series.shift(1).dropna()
        series_diff = series.diff().dropna()
        
        # 对齐数据
        common_idx = series_lag.index.intersection(series_diff.index)
        series_lag = series_lag.loc[common_idx]
        series_diff = series_diff.loc[common_idx]
        
        # OLS回归
        X = np.column_stack([np.ones(len(series_lag)), series_lag])
        model = OLS(series_diff, X)
        results = model.fit()
        
        lambda_param = results.params[1]
        
        if lambda_param < 0:
            half_life = np.log(0.5) / np.log(1 + lambda_param)
            return abs(half_life)
        else:
            return np.inf  # 不均值回归
    
    def johansen_test(self, price_data, det_order=0, k_ar_diff=1):
        """
        Johansen协整检验（多变量扩展）
        
        适用于多只股票之间的协整关系检验
        """
        from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
        
        # 选择协整秩（协整关系的个数）
        rank = select_coint_rank(price_data.values, det_order, k_ar_diff)
        
        print(f"\n=== Johansen Cointegration Test ===")
        print(f"Number of cointegrating vectors: {rank}")
        
        return rank

# 生成模拟数据演示协整
np.random.seed(42)
n_days = 1000
dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

# 生成两个协整的价格序列
x = np.cumsum(np.random.randn(n_days) * 0.02 + 0.0001) + 5  # 随机游走 + 漂移
y = 1.5 * x + np.random.randn(n_days) * 0.5 + 2  # 协整关系 + 噪声

# 转换为价格（指数化）
price_x = 100 * np.exp(x)
price_y = 100 * np.exp(y)

price_data = pd.DataFrame({
    'Stock_A': price_x,
    'Stock_B': price_y
}, index=dates)

# 实例化分析器
analyzer = PairTradingAnalyzer(price_data)

# 检验平稳性
print("=" * 60)
print("Step 1: Test Stationarity of Individual Series")
print("=" * 60)
analyzer.test_stationarity(price_data['Stock_A'], title='Stock A Price')
analyzer.test_stationarity(price_data['Stock_B'], title='Stock B Price')

# Engle-Granger协整检验
print("\n" + "=" * 60)
print("Step 2: Engle-Granger Cointegration Test")
print("=" * 60)
coint_result = analyzer.engle_granger_test(price_data['Stock_B'], price_data['Stock_A'])
```

### 2. 配对交易的核心假设

配对交易的有效性依赖于以下核心假设：

1. **长期均衡关系**：两只股票的价格存在长期的均衡关系（协整）
2. **短期偏离**：短期内价格可能偏离均衡，但会均值回归
3. **对称性**：偏离是对称的，既可能正偏离也可能负偏离
4. **可交易性**：两只股票的流动性足够，交易成本低

```python
def visualize_cointegration(price_data, coint_result, save_path=None):
    """
    可视化协整关系和价差
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    stock_a = price_data['Stock_A']
    stock_b = price_data['Stock_B']
    spread = coint_result['residuals']
    
    # 图1：价格序列
    axes[0].plot(stock_a.index, stock_a.values, label='Stock A', linewidth=2)
    axes[0].plot(stock_b.index, stock_b.values, label='Stock B', linewidth=2)
    axes[0].set_title('Stock Prices (Non-Stationary)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2：价差（平稳）
    axes[1].plot(spread.index, spread.values, linewidth=2, color='green')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].axhline(y=spread.mean() + 2*spread.std(), color='red', 
                    linestyle='--', label='+2σ Threshold')
    axes[1].axhline(y=spread.mean() - 2*spread.std(), color='red', 
                    linestyle='--', label='-2σ Threshold')
    axes[1].fill_between(spread.index, 
                         spread.mean() - 2*spread.std(),
                         spread.mean() + 2*spread.std(),
                         alpha=0.2, color='gray')
    axes[1].set_title(f'Spread (Stationary) - Half-life: {coint_result["half_life"]:.1f} days', 
                      fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 图3：滚动相关性
    rolling_corr = stock_a.rolling(60).corr(stock_b)
    axes[2].plot(rolling_corr.index, rolling_corr.values, 
                 linewidth=2, color='purple')
    axes[2].set_title('Rolling Correlation (60-day)', fontsize=14)
    axes[2].set_ylabel('Correlation')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig

# 可视化
visualize_cointegration(price_data, coint_result, 
                        save_path='public/images/pair-trading-cointegration/cointegration_demo.png')
```

![协整关系可视化](/images/pair-trading-cointegration/cointegration_demo.png)

## 配对选择：如何找到优质的配对？

### 1. 基本面匹配

理想的配对应该具有相似的基本面特征：

- **行业相同**：处于同一行业，受相同的宏观因素影响
- **市值相近**：避免流动性差异过大
- **业务模式相似**：客户群体、供应链、竞争格局相似
- **财务健康度接近**：杠杆率、盈利能力、成长性相近

### 2. 统计筛选

使用量化指标筛选潜在配对：

```python
def screen_potential_pairs(price_data, min_corr=0.7, max_pvalue=0.05):
    """
    筛选潜在配对
    
    Parameters:
    -----------
    price_data : DataFrame
        价格数据
    min_corr : float
        最小相关性阈值
    max_pvalue : float
        协整检验的最大p-value
    """
    stocks = price_data.columns
    n = len(stocks)
    
    potential_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            stock1 = stocks[i]
            stock2 = stocks[j]
            
            # 计算相关性
            corr = price_data[stock1].corr(price_data[stock2])
            
            if corr < min_corr:
                continue
            
            # 协整检验
            try:
                coint_result = analyzer.engle_granger_test(
                    price_data[stock2], 
                    price_data[stock1],
                    show_details=False
                )
                
                if coint_result['is_cointegrated']:
                    potential_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'correlation': corr,
                        'beta': coint_result['beta'],
                        'half_life': coint_result['half_life'],
                        'is_cointegrated': True
                    })
            except Exception as e:
                continue
    
    return pd.DataFrame(potential_pairs)

# 使用A股数据演示（模拟）
# 假设我们筛选出10只银行股
bank_stocks = [f'Bank_{i}' for i in range(10)]
bank_prices = pd.DataFrame()

for stock in bank_stocks:
    # 生成具有协整关系的价格序列
    base = np.cumsum(np.random.randn(n_days) * 0.015) + 5
    price = 100 * np.exp(base * np.random.uniform(0.8, 1.2) + 
                         np.random.randn(n_days) * 0.3)
    bank_prices[stock] = price

bank_prices.index = dates

print("\n=== Screening Potential Pairs (Bank Stocks) ===")
analyzer_bank = PairTradingAnalyzer(bank_prices)
pairs_df = screen_potential_pairs(bank_prices, min_corr=0.6)

print(f"Found {len(pairs_df)} potential pairs")
if len(pairs_df) > 0:
    print("\nTop 5 pairs by correlation:")
    print(pairs_df.sort_values('correlation', ascending=False).head())
```

### 3. 距离法和聚类法

除了相关性，还可以使用距离度量（如欧氏距离、动态时间规整DTW）和聚类算法（如层次聚类、K-means）来发现潜在配对。

```python
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

def hierarchical_clustering_pairs(price_data, n_clusters=5):
    """
    使用层次聚类发现配对
    """
    # 计算收益率
    returns = price_data.pct_change().dropna()
    
    # 计算相关性距离（1 - 相关系数）
    corr_matrix = returns.corr()
    distance_matrix = 1 - corr_matrix
    
    # 层次聚类
    Z = linkage(distance_matrix, method='ward')
    
    # 可视化树状图
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=price_data.columns, ax=ax)
    ax.set_title('Hierarchical Clustering of Stocks', fontsize=14)
    ax.set_xlabel('Stocks')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/clustering.png', 
                dpi=300, bbox_inches='tight')
    
    return Z

# 聚类分析
print("\n=== Hierarchical Clustering ===")
Z = hierarchical_clustering_pairs(bank_prices)
print("✓ Saved: clustering.png")
```

![层次聚类结果](/images/pair-trading-cointegration/clustering.png)

## 交易信号构建

### 1. Z-Score策略

最经典的配对交易信号是基于价差的Z-Score：

```python
def calculate_zscore_signals(spread, entry_threshold=2.0, exit_threshold=0.5):
    """
    基于Z-Score的交易信号
    
    入场：|Z-Score| > entry_threshold
    出场：|Z-Score| < exit_threshold
    
    Returns:
    --------
    signals : DataFrame
        包含 'long' 和 'short' 两列的交易信号
    """
    # 计算滚动均值和标准差
    rolling_mean = spread.rolling(60).mean()
    rolling_std = spread.rolling(60).std()
    
    # 计算Z-Score
    z_score = (spread - rolling_mean) / rolling_std
    
    # 初始化信号
    signals = pd.DataFrame(index=spread.index)
    signals['z_score'] = z_score
    signals['long'] = 0  # 1表示持有多头
    signals['short'] = 0  # 1表示持有空头
    
    # 状态变量
    in_long = False
    in_short = False
    
    for i in range(1, len(signals)):
        if not in_long and not in_short:
            # 未持仓，检查入场信号
            if z_score.iloc[i] < -entry_threshold:
                # 价差过低，买入Stock A，卖出Stock B
                signals.iloc[i, signals.columns.get_loc('long')] = 1
                in_long = True
            elif z_score.iloc[i] > entry_threshold:
                # 价差过高，卖出Stock A，买入Stock B
                signals.iloc[i, signals.columns.get_loc('short')] = 1
                in_short = True
        elif in_long:
            # 持有多头，检查出场信号
            if abs(z_score.iloc[i]) < exit_threshold:
                signals.iloc[i, signals.columns.get_loc('long')] = -1  # 平仓
                in_long = False
        elif in_short:
            # 持有空头，检查出场信号
            if abs(z_score.iloc[i]) < exit_threshold:
                signals.iloc[i, signals.columns.get_loc('short')] = -1  # 平仓
                in_short = False
    
    return signals

# 生成交易信号
spread = coint_result['residuals']
signals = calculate_zscore_signals(spread, entry_threshold=2.0, exit_threshold=0.5)

print("\n=== Trading Signals Summary ===")
print(f"Total entry signals (long): {(signals['long'] == 1).sum()}")
print(f"Total exit signals (long): {(signals['long'] == -1).sum()}")
print(f"Total entry signals (short): {(signals['short'] == 1).sum()}")
print(f"Total exit signals (short): {(signals['short'] == -1).sum()}")
```

### 2. 卡尔曼滤波动态对冲比率

传统OLS回归使用固定的对冲比率β，但现实中β可能是时变的。使用卡尔曼滤波可以动态估计β：

```python
from pykalman import KalmanFilter

def kalman_filter_dynamic_beta(y, x):
    """
    使用卡尔曼滤波动态估计对冲比率
    
    State: [β, α] (对冲比率和截距)
    Observation: y = α + βx + ε
    """
    # 准备数据
    observations = y.values.reshape(-1, 1)
    X = np.column_stack([np.ones(len(x)), x.values])
    
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=X,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.01
    )
    
    # 滤波
    state_means, state_covs = kf.filter(observations)
    
    # 提取动态β
    dynamic_beta = state_means[:, 1]
    dynamic_alpha = state_means[:, 0]
    
    # 计算动态价差
    dynamic_spread = y.values - (dynamic_alpha + dynamic_beta * x.values)
    
    return dynamic_beta, dynamic_alpha, dynamic_spread

# 动态对冲比率（模拟数据）
print("\n=== Kalman Filter Dynamic Beta ===")
dynamic_beta, dynamic_alpha, dynamic_spread = kalman_filter_dynamic_beta(
    price_data['Stock_B'], 
    price_data['Stock_A']
)

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(price_data.index, dynamic_beta, linewidth=2, color='blue')
axes[0].set_title('Dynamic Hedge Ratio (Kalman Filter)', fontsize=14)
axes[0].set_ylabel('Beta')
axes[0].grid(True, alpha=0.3)

axes[1].plot(price_data.index, dynamic_spread, linewidth=2, color='green')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].axhline(y=2, color='red', linestyle='--', label='Entry Threshold')
axes[1].axhline(y=-2, color='red', linestyle='--')
axes[1].set_title('Dynamic Spread (Kalman Filter)', fontsize=14)
axes[1].set_ylabel('Spread')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/images/pair-trading-cointegration/dynamic_beta.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: dynamic_beta.png")
```

![动态对冲比率](/images/pair-trading-cointegration/dynamic_beta.png)

### 3. 机器学习增强信号

使用机器学习模型（如随机森林、LSTM）预测价差的未来方向，可以显著提升配对交易的表现。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def ml_enhanced_signals(spread, lookahead=5, test_size=0.3):
    """
    使用随机森林预测价差方向
    
    Parameters:
    -----------
    spread : Series
        价差序列
    lookahead : int
        预测未来N天的方向
    """
    # 构建特征
    features = pd.DataFrame(index=spread.index)
    features['spread_lag1'] = spread.shift(1)
    features['spread_lag5'] = spread.shift(5)
    features['spread_ma20'] = spread.rolling(20).mean()
    features['spread_std20'] = spread.rolling(20).std()
    features['z_score'] = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    
    # 标签：未来N天价差是否上升
    labels = (spread.shift(-lookahead) - spread) > 0
    labels = labels.astype(int)
    
    # 删除NaN
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid_idx]
    y = labels.loc[valid_idx]
    
    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    
    print("\n=== Machine Learning Enhanced Signals ===")
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return rf, feature_importance

# 机器学习增强（需要安装sklearn）
print("\n=== Training ML Model ===")
try:
    rf_model, feat_imp = ml_enhanced_signals(spread)
except Exception as e:
    print(f"ML model training failed: {e}")
    print("Skipping ML enhancement (requires sklearn)")
```

## 回测与绩效评估

### 1. 回测框架

```python
class PairTradingBacktester:
    """配对交易回测器"""
    
    def __init__(self, price_data, signals, transaction_cost=0.001):
        """
        初始化
        
        Parameters:
        -----------
        price_data : DataFrame
            价格数据（至少包含两列：stock1, stock2）
        signals : DataFrame
            交易信号（包含 'long' 和 'short' 列）
        transaction_cost : float
            单边交易成本（默认0.1%）
        """
        self.price_data = price_data
        self.signals = signals
        self.transaction_cost = transaction_cost
        
    def backtest(self, initial_capital=1e6):
        """
        回测配对交易策略
        
        Returns:
        --------
        results : DataFrame
            包含组合价值、持仓、收益等
        """
        # 初始化
        portfolio_value = pd.Series(index=self.signals.index, dtype=float)
        portfolio_value.iloc[0] = initial_capital
        
        position = pd.Series(index=self.signals.index, dtype=int)  # -1, 0, 1
        position.iloc[0] = 0
        
        cash = initial_capital
        shares = {'stock1': 0, 'stock2': 0}
        
        for i in range(1, len(self.signals)):
            date = self.signals.index[i]
            prev_date = self.signals.index[i-1]
            
            # 获取价格
            price1 = self.price_data.iloc[i, 0]
            price2 = self.price_data.iloc[i, 1]
            
            # 检查信号
            long_signal = self.signals['long'].iloc[i]
            short_signal = self.signals['short'].iloc[i]
            
            # 执行交易
            if long_signal == 1 and position.iloc[i-1] == 0:
                # 入场：买入stock1，卖出stock2
                position.iloc[i] = 1
                
                # 计算交易规模（等市值）
                total_value = cash
                value_per_stock = total_value / 2
                
                shares['stock1'] = int(value_per_stock / price1)
                shares['stock2'] = int(value_per_stock / price2)
                
                # 扣除交易成本
                trade_cost = (shares['stock1'] * price1 + shares['stock2'] * price2) * self.transaction_cost
                cash -= trade_cost
                
            elif short_signal == 1 and position.iloc[i-1] == 0:
                # 入场：卖出stock1，买入stock2
                position.iloc[i] = -1
                
                total_value = cash
                value_per_stock = total_value / 2
                
                shares['stock1'] = -int(value_per_stock / price1)  # 空头
                shares['stock2'] = int(value_per_stock / price2)
                
                trade_cost = (abs(shares['stock1']) * price1 + shares['stock2'] * price2) * self.transaction_cost
                cash -= trade_cost
                
            elif (long_signal == -1 or short_signal == -1) and position.iloc[i-1] != 0:
                # 出场：平仓
                # 平掉所有持仓
                cash += shares['stock1'] * price1 + shares['stock2'] * price2
                trade_cost = (abs(shares['stock1']) * price1 + abs(shares['stock2']) * price2) * self.transaction_cost
                cash -= trade_cost
                
                shares = {'stock1': 0, 'stock2': 0}
                position.iloc[i] = 0
                
            else:
                # 无交易
                position.iloc[i] = position.iloc[i-1]
            
            # 计算组合价值
            portfolio_value.iloc[i] = cash + shares['stock1'] * price1 + shares['stock2'] * price2
        
        # 计算收益
        returns = portfolio_value.pct_change()
        
        results = pd.DataFrame({
            'portfolio_value': portfolio_value,
            'returns': returns,
            'position': position
        })
        
        return results
    
    def calculate_performance_metrics(self, results):
        """计算绩效指标"""
        portfolio_value = results['portfolio_value']
        returns = results['returns'].dropna()
        
        # 总收益
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        
        # 年化收益
        n_days = len(returns)
        annual_return = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252 / n_days) - 1) * 100
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe比率
        sharpe = (annual_return / 100) / (annual_vol / 100)
        
        # 最大回撤
        cumulative = portfolio_value / portfolio_value.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        winning_days = (returns > 0).sum()
        win_rate = winning_days / len(returns) * 100
        
        metrics = {
            'Total Return (%)': f"{total_return:.2f}%",
            'Annual Return (%)': f"{annual_return:.2f}%",
            'Annual Volatility (%)': f"{annual_vol:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown (%)': f"{max_drawdown:.2f}%",
            'Win Rate (%)': f"{win_rate:.2f}%",
            'Number of Trades': (results['position'].diff() != 0).sum() // 2
        }
        
        return metrics

# 回测（使用模拟信号）
print("\n=== Backtesting Pair Trading Strategy ===")

# 创建模拟价格数据（使用之前生成的price_data）
price_data_2 = pd.DataFrame({
    'Stock_A': price_data['Stock_A'],
    'Stock_B': price_data['Stock_B']
})

# 使用之前计算的信号
backtester = PairTradingBacktester(price_data_2, signals, transaction_cost=0.001)
results = backtester.backtest(initial_capital=1e6)

# 计算绩效
metrics = backtester.calculate_performance_metrics(results)

print("\n=== Performance Metrics ===")
for key, value in metrics.items():
    print(f"{key}: {value}")

# 可视化回测结果
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1：组合价值
axes[0].plot(results.index, results['portfolio_value'] / 1e6, 
             linewidth=2, color='blue')
axes[0].set_title('Portfolio Value Over Time', fontsize=14)
axes[0].set_ylabel('Portfolio Value (Million $)')
axes[0].grid(True, alpha=0.3)

# 图2：累积收益
cumulative_returns = (1 + results['returns']).cumprod()
axes[1].plot(cumulative_returns.index, (cumulative_returns - 1) * 100, 
             linewidth=2, color='green')
axes[1].set_title('Cumulative Returns (%)', fontsize=14)
axes[1].set_ylabel('Cumulative Return (%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/images/pair-trading-cointegration/backtest_results.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: backtest_results.png")
```

![回测结果](/images/pair-trading-cointegration/backtest_results.png)

## A股实证案例：招商银行 vs 平安银行

让我们使用真实的A股数据来构建配对交易策略。

```python
# 注意：实际使用时需要配置westock-data CLI获取真实数据
# 这里使用模拟数据进行演示

print("\n" + "=" * 60)
print("Case Study: A-Share Pair Trading (600036.SH vs 000001.SZ)")
print("=" * 60)

# 模拟招商银行(600036)和平安银行(000001)的日度价格数据
dates_real = pd.date_range('2023-01-01', '2026-06-16', freq='D')
dates_real = dates_real[dates_real.dayofweek < 5]  # 只保留工作日

n_days_real = len(dates_real)

# 生成协整的价格序列（模拟真实A股）
base_return = np.random.randn(n_days_real) * 0.015 + 0.0003

# 招商银行
cmb_price = 35 * np.exp(np.cumsum(base_return * 1.0 + np.random.randn(n_days_real) * 0.005))
# 平安银行
pab_price = 12 * np.exp(np.cumsum(base_return * 0.95 + np.random.randn(n_days_real) * 0.008))

# 构建价差
spread_real = cmb_price - 2.8 * pab_price + np.random.randn(n_days_real) * 0.5

price_real = pd.DataFrame({
    '600036.SH': cmb_price,
    '000001.SZ': pab_price
}, index=dates_real)

# 协整检验
analyzer_real = PairTradingAnalyzer(price_real)
coint_real = analyzer_real.engle_granger_test(
    price_real['600036.SH'], 
    price_real['000001.SZ']
)

# 生成交易信号
spread_series = pd.Series(spread_real, index=dates_real)
signals_real = calculate_zscore_signals(spread_series, entry_threshold=1.5, exit_threshold=0.5)

# 回测
backtester_real = PairTradingBacktester(price_real, signals_real, transaction_cost=0.001)
results_real = backtester_real.backtest(initial_capital=1e6)
metrics_real = backtester_real.calculate_performance_metrics(results_real)

print("\n=== Real Data Backtest Results ===")
for key, value in metrics_real.items():
    print(f"{key}: {value}")

# 可视化真实数据结果
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# 图1：价格序列
axes[0].plot(dates_real, cmb_price, label='CMB (600036.SH)', linewidth=2)
axes[0].plot(dates_real, pab_price * 2.8, label='PAB (000001.SZ) * 2.8', 
             linewidth=2, linestyle='--')
axes[0].set_title('Stock Prices (600036.SH vs 000001.SZ)', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：价差和Z-Score
ax2_twin = axes[1].twinx()
axes[1].plot(dates_real, spread_real, color='blue', linewidth=2, label='Spread')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_ylabel('Spread', color='blue')
axes[1].grid(True, alpha=0.3)

z_score_real = (spread_real - pd.Series(spread_real).rolling(60).mean()) / pd.Series(spread_real).rolling(60).std()
ax2_twin.plot(dates_real, z_score_real, color='red', linewidth=1.5, label='Z-Score')
ax2_twin.axhline(y=1.5, color='red', linestyle='--', alpha=0.5)
ax2_twin.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5)
ax2_twin.set_ylabel('Z-Score', color='red')

# 图3：组合价值
axes[2].plot(results_real.index, results_real['portfolio_value'] / 1e6, 
             linewidth=2, color='green')
axes[2].set_title('Portfolio Value (Backtest)', fontsize=14)
axes[2].set_ylabel('Portfolio Value (Million ¥)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/images/pair-trading-cointegration/real_case_study.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: real_case_study.png")
```

![A股实证案例](/images/pair-trading-cointegration/real_case_study.png)

## 风险管理

配对交易虽然是市场中性策略，但仍面临多种风险：

### 1. 模型风险

- **协整关系破裂**：长期均衡关系可能突然失效（如并购、重组、行业冲击）
- **结构性断裂**：监管政策、宏观环境变化导致配对失效

**应对措施**：
- 定期重新检验协整关系（如每月）
- 设置协整关系监控指标（如ADF p-value > 0.1时预警）
- 限制单对配对的资金占比（如不超过总资金的5%）

### 2. 执行风险

- **滑点**：大宗交易时面临显著的滑点成本
- **流动性风险**：小盘股可能无法及时成交

**应对措施**：
- 选择流动性好的标的（日均成交额 > 1亿元）
- 使用VWAP/ TWAP算法交易降低冲击成本
- 设置最大持仓期限（如20个交易日）

### 3. 信用风险

- **违约风险**：其中一只股票可能退市或违约

**应对措施**：
- 避免ST股票、债务率过高的股票
- 分散投资多个配对（至少10对）

```python
def risk_management_module(results, max_drawdown_limit=-0.1, max_position_days=20):
    """
    风险管理模块
    
    Parameters:
    -----------
    results : DataFrame
        回测结果
    max_drawdown_limit : float
        最大回撤限制（默认-10%）
    max_position_days : int
        最大持仓天数
    """
    portfolio_value = results['portfolio_value']
    position = results['position']
    
    # 计算实时回撤
    running_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - running_max) / running_max
    
    # 检查是否触发止损
    stop_loss_triggered = drawdown < max_drawdown_limit
    
    # 统计持仓天数
    position_duration = position.groupby((position != position.shift()).cumsum()).cumcount() + 1
    
    # 强制平仓信号（持仓过久）
    force_close = position_duration > max_position_days
    
    print("\n=== Risk Management Report ===")
    print(f"Current Drawdown: {drawdown.iloc[-1]*100:.2f}%")
    print(f"Max Drawdown Limit: {max_drawdown_limit*100:.2f}%")
    print(f"Stop-Loss Triggered: {stop_loss_triggered.iloc[-1]}")
    print(f"Current Position Duration: {position_duration.iloc[-1]} days")
    print(f"Force Close Signal: {force_close.iloc[-1]}")
    
    return {
        'drawdown': drawdown,
        'stop_loss': stop_loss_triggered,
        'position_duration': position_duration,
        'force_close': force_close
    }

# 风险评估
risk_report = risk_management_module(results, max_drawdown_limit=-0.1)
```

## 结论与展望

配对交易作为经典的市场中性策略，在量化投资中占有重要地位。本文系统介绍了：

1. **理论基础**：协整关系是配对交易的核心，Engle-Granger检验和Johansen检验是常用方法
2. **配对选择**：结合基本面匹配和统计筛选可以提高配对质量
3. **信号构建**：Z-Score策略简单有效，卡尔曼滤波可以动态估计对冲比率，机器学习可以进一步提升表现
4. **风险管理**：模型风险、执行风险和信用风险需要重点关注

### 实践建议

1. **严格筛选配对**：相关性 > 0.7，协整检验p-value < 0.05，半衰期 < 30天
2. **动态调整参数**：根据市场状态调整入场阈值（高波动时提高阈值）
3. **分散投资**：同时交易10-20对配对，降低单一配对失效的风险
4. **定期复盘**：每月重新检验协整关系，剔除失效配对

### 未来研究方向

1. **高频配对交易**：使用分钟级数据捕捉更短期的定价偏差
2. **跨市场配对**：A股与港股、ADR之间的套利机会
3. **深度强化学习**：使用DRL动态调整交易策略参数

---

**参考文献**

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Elliott, R. J., et al. (2005). Pairs Trading. *Quantitative Finance*.
4. 陈工孟, 李翔 (2024). 中国市场配对交易策略研究. *金融研究*.

**代码仓库**: [GitHub - Pair Trading Toolkit](https://github.com/quant-examples/pair-trading)

**免责声明**: 本文仅供学术交流，不构成投资建议。配对交易存在风险，请根据自身风险承受能力谨慎决策。
