---
title: "统计套利：均值回归策略"
date: "2026-06-19"
description: "详细介绍统计套利的核心思想、配对交易方法和均值回归策略的实现技巧。"
slug: "statistical-arbitrage-mean-reversion"
tags: ["统计套利", "配对交易", "均值回归"]
cover: "/images/statistical-arbitrage-mean-reversion/cover.jpg"
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是量化投资中最经典也最经久不衰的策略类别之一。其核心思想简单而优雅：**价格偏离统计关系后倾向于回归**。然而，从理论到实盘之间有着巨大的鸿沟——理解协整检验的数学公式很容易，但构建一个能在交易成本扣除后仍获利的均值回归策略，需要处理数据偏差、协整关系断裂、执行延迟等一系列现实问题。

本文将从零开始，完整实现几种常见的均值回归策略，包括配对交易、篮子交易、PCA套利等，并重点讨论从回测到实盘的关键注意事项。

## 均值回归的统计学基础

### 平稳性与单位根

均值回归策略的前提是**价格序列的某种线性组合是平稳的**。如果一个时间序列的统计特性（均值、方差、自协方差）不随时间变化，我们称之为平稳序列。

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(series, significance=0.05):
    """
    ADF检验判断时间序列平稳性
    
    参数：
    - series: 时间序列
    - significance: 显著性水平
    
    返回：
    - is_stationary: 是否平稳
    - p_value: p值
    - adf_stat: ADF统计量
    """
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value < significance
    
    print(f"ADF统计量: {adf_stat:.4f}")
    print(f"p值: {p_value:.6f}")
    print(f"{'✅ 平稳' if is_stationary else '❌ 非平稳'} (α={significance})")
    print(f"使用滞后阶数: {result[2]}")
    print(f"观测数: {result[3]}")
    
    return is_stationary, p_value, adf_stat

# 示例1：检验股价是否平稳（通常不是）
np.random.seed(42)
prices = 100 + np.cumsum(np.random.normal(0.1, 2, 500))

print("=== 股价序列平稳性检验 ===")
is_stationary, _, _ = check_stationarity(pd.Series(prices))
print(f"结论: 股价序列通常是非平稳的（随机游走）\n")

# 示例2：对收益率检验（通常平稳）
returns = np.diff(np.log(prices))
print("=== 收益率序列平稳性检验 ===")
is_stationary, _, _ = check_stationarity(pd.Series(returns))
print(f"结论: 收益率序列通常是平稳的\n")

# 示例3：检验两个价格序列的线性组合（如果协整，则平稳）
prices_a = 100 + np.cumsum(np.random.normal(0.05, 1, 500))
prices_b = 98 + 0.95 * (prices_a - 100) + np.cumsum(np.random.normal(0, 0.5, 500))

print("=== 价格A与价格B的线性组合平稳性检验 ===")
hedge_ratio = OLS(prices_a, prices_b).fit().params[0]
spread = prices_a - hedge_ratio * prices_b
is_stationary, _, _ = check_stationarity(pd.Series(spread))
print(f"结论: {'两只股票价格存在协整关系，价差平稳' if is_stationary else '不存在协整关系'}")
```

### 协整关系：配对交易的理论基石

两只非平稳的股票，如果存在某种长期均衡关系，它们的价格差（或线性组合）可能平稳。这就是**协整（Cointegration）**。

```python
def find_cointegrated_pairs(data, significance=0.05):
    """
    在多只股票中寻找协整对
    
    参数：
    - data: DataFrame, columns=股票名, index=日期, values=收盘价
    - significance: 显著性水平
    
    返回：
    - pairs: DataFrame, 包含协整对、p值、对冲比率等信息
    """
    n = data.shape[1]
    keys = data.columns
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            stock_a = data[keys[i]].dropna()
            stock_b = data[keys[j]].dropna()
            
            # 对齐日期
            common_idx = stock_a.index.intersection(stock_b.index)
            stock_a = stock_a.loc[common_idx]
            stock_b = stock_b.loc[common_idx]
            
            # Engle-Granger协整检验
            score, pvalue, _ = coint(stock_a, stock_b)
            
            if pvalue < significance:
                # 计算对冲比率（用OLS回归）
                model = OLS(stock_a, stock_b).fit()
                hedge_ratio = model.params[0]
                
                # 计算价差
                spread = stock_a - hedge_ratio * stock_b
                
                # 检验价差平稳性
                is_spread_stationary, spread_pvalue, _ = check_stationarity(spread)
                
                pairs.append({
                    'stock_a': keys[i],
                    'stock_b': keys[j],
                    'p_value': pvalue,
                    'hedge_ratio': hedge_ratio,
                    'spread_mean': spread.mean(),
                    'spread_std': spread.std(),
                    'is_spread_stationary': is_spread_stationary,
                    'spread_pvalue': spread_pvalue
                })
    
    return pd.DataFrame(pairs).sort_values('p_value')

# 示例：在5只股票中寻找协整对
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# 生成模拟价格数据（其中AAPL和MSFT存在协整关系）
prices_aapl = 100 + np.cumsum(np.random.normal(0.05, 1, 500))
prices_msft = 98 + 0.95 * (prices_aapl - 100) + np.cumsum(np.random.normal(0, 0.5, 500))
prices_googl = 150 + np.cumsum(np.random.normal(0.06, 1.2, 500))
prices_amzn = 200 + np.cumsum(np.random.normal(0.07, 1.5, 500))
prices_meta = 180 + np.cumsum(np.random.normal(0.04, 1.1, 500))

price_data = pd.DataFrame({
    'AAPL': prices_aapl,
    'MSFT': prices_msft,
    'GOOGL': prices_googl,
    'AMZN': prices_amzn,
    'META': prices_meta
}, index=dates)

print("=== 寻找协整对 ===")
pairs = find_cointegrated_pairs(price_data)
print(f"\n找到 {len(pairs)} 个协整对:\n")
print(pairs[['stock_a', 'stock_b', 'p_value', 'hedge_ratio']].head())
```

## 配对交易策略实现

配对交易（Pairs Trading）是统计套利中最经典的策略。其基本思想是：找到一对协整股票，当价差偏离均值时开仓，等待价差回归时平仓获利。

### 基本配对交易框架

```python
class PairsTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, stock_a, stock_b, entry_threshold=2.0, exit_threshold=0.5):
        """
        初始化配对交易策略
        
        参数：
        - stock_a: 股票A的价格序列
        - stock_b: 股票B的价格序列
        - entry_threshold: 入场阈值（Z-Score绝对值）
        - exit_threshold: 出场阈值（Z-Score绝对值）
        """
        self.stock_a = stock_a
        self.stock_b = stock_b
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # 计算对冲比率和价差
        self.calculate_hedge_ratio()
        self.calculate_spread()
        
    def calculate_hedge_ratio(self):
        """计算对冲比率（滚动窗口）"""
        # 使用滚动60日窗口计算对冲比率
        window = 60
        hedge_ratios = []
        
        for i in range(window, len(self.stock_a)):
            y = self.stock_a.iloc[i-window:i]
            x = self.stock_b.iloc[i-window:i]
            model = OLS(y, x).fit()
            hedge_ratios.append(model.params[0])
        
        # 填充前期数据
        hedge_ratios = [hedge_ratios[0]] * window + hedge_ratios
        self.hedge_ratio = pd.Series(hedge_ratios, index=self.stock_a.index)
    
    def calculate_spread(self):
        """计算价差和Z-Score"""
        self.spread = self.stock_a - self.hedge_ratio * self.stock_b
        
        # 计算价差的滚动均值和标准差
        self.spread_mean = self.spread.rolling(window=60).mean()
        self.spread_std = self.spread.rolling(window=60).std()
        
        # 计算Z-Score
        self.zscore = (self.spread - self.spread_mean) / self.spread_std
    
    def generate_signals(self):
        """生成交易信号"""
        self.signals = pd.DataFrame(index=self.stock_a.index)
        self.signals['zscore'] = self.zscore
        
        # 初始化持仓信号
        self.signals['position'] = 0
        
        # 入场信号：Z-Score超过阈值
        self.signals.loc[self.zscore > self.entry_threshold, 'position'] = -1  # 做空价差
        self.signals.loc[self.zscore < -self.entry_threshold, 'position'] = 1   # 做多价差
        
        # 出场信号：Z-Score回归
        self.signals.loc[abs(self.zscore) < self.exit_threshold, 'position'] = 0
        
        # 填充持仓（保持持仓直到出场信号）
        self.signals['position'] = self.signals['position'].replace(0, np.nan)
        self.signals['position'] = self.signals['position'].fillna(method='ffill').fillna(0)
        
        return self.signals
    
    def backtest(self, transaction_cost=0.001):
        """
        回测配对交易策略
        
        参数：
        - transaction_cost: 交易成本（单边）
        
        返回：
        - results: 回测结果DataFrame
        """
        # 生成信号
        signals = self.generate_signals()
        
        # 计算收益
        results = pd.DataFrame(index=self.stock_a.index)
        results['spread'] = self.spread
        results['zscore'] = self.zscore
        results['position'] = signals['position']
        
        # 计算价差收益（假设我们交易价差）
        results['spread_return'] = self.spread.pct_change()
        
        # 计算策略收益（考虑持仓和交易成本）
        results['strategy_return'] = results['position'].shift(1) * results['spread_return']
        
        # 计算交易成本（持仓变化时）
        position_change = results['position'].diff().abs()
        results['transaction_cost'] = position_change * transaction_cost
        results['net_return'] = results['strategy_return'] - results['transaction_cost']
        
        # 计算累积收益
        results['cumulative_return'] = (1 + results['net_return']).cumprod()
        
        # 计算绩效指标
        total_return = results['cumulative_return'].iloc[-1] - 1
        sharpe_ratio = results['net_return'].mean() / results['net_return'].std() * np.sqrt(252)
        max_drawdown = (results['cumulative_return'] / results['cumulative_return'].cummax() - 1).min()
        
        print(f"=== 配对交易回测结果 ===")
        print(f"总收益: {total_return:.2%}")
        print(f"年化收益: {((1 + total_return) ** (252 / len(results)) - 1):.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"交易次数: {position_change.sum():.0f}")
        
        return results
    
    def visualize(self, results):
        """可视化回测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 子图1：价格和价差
        ax1 = axes[0]
        ax1.plot(self.stock_a.index, self.stock_a.values, label='Stock A', alpha=0.7)
        ax1.plot(self.stock_b.index, self.stock_b.values, label='Stock B', alpha=0.7)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.spread.index, self.spread.values, color='gray', alpha=0.5, label='Spread')
        ax1_twin.set_ylabel('Spread')
        
        # 子图2：Z-Score和交易信号
        ax2 = axes[1]
        ax2.plot(self.zscore.index, self.zscore.values, label='Z-Score', color='blue')
        ax2.axhline(y=self.entry_threshold, color='red', linestyle='--', label='Entry Threshold')
        ax2.axhline(y=-self.entry_threshold, color='red', linestyle='--')
        ax2.axhline(y=self.exit_threshold, color='green', linestyle='--', label='Exit Threshold')
        ax2.axhline(y=-self.exit_threshold, color='green', linestyle='--')
        ax2.fill_between(self.zscore.index, 0, self.zscore.values, 
                         where=(results['position'] == 1), alpha=0.3, color='green', label='Long')
        ax2.fill_between(self.zscore.index, 0, self.zscore.values, 
                         where=(results['position'] == -1), alpha=0.3, color='red', label='Short')
        ax2.set_ylabel('Z-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：累积收益
        ax3 = axes[2]
        ax3.plot(results.index, results['cumulative_return'].values, 
                label='Strategy Return', color='purple', linewidth=2)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Cumulative Return')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('public/images/statistical-arbitrage-mean-reversion/backtest.png', 
                   dpi=150, bbox_inches='tight')
        print("✅ 回测结果图表已保存")

# 示例使用配对交易策略
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')

# 生成协整股票价格
prices_a = 100 + np.cumsum(np.random.normal(0.05, 1, 500))
prices_b = 98 + 0.95 * (prices_a - 100) + np.cumsum(np.random.normal(0, 0.5, 500))

stock_a = pd.Series(prices_a, index=dates)
stock_b = pd.Series(prices_b, index=dates)

# 创建策略实例
strategy = PairsTradingStrategy(stock_a, stock_b, entry_threshold=2.0, exit_threshold=0.5)

# 回测
results = strategy.backtest(transaction_cost=0.001)

# 可视化
strategy.visualize(results)
```

### 改进一：动态对冲比率

固定对冲比率在协整关系变化时会导致价差漂移。使用卡尔曼滤波（Kalman Filter）可以动态估计时变对冲比率。

```python
from pykalman import KalmanFilter

def dynamic_hedge_ratio_kalman(stock_a, stock_b):
    """
    使用卡尔曼滤波动态估计对冲比率
    
    参数：
    - stock_a: 股票A价格
    - stock_b: 股票B价格
    
    返回：
    - dynamic_hedge_ratio: 动态对冲比率序列
    - spread: 动态价差序列
    """
    # 准备观测数据
    Y = stock_a.values.reshape(-1, 1)
    X = stock_b.values.reshape(-1, 1)
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=X.reshape(-1, 1, 1),
        initial_state_mean=1.0,
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=0.01
    )
    
    # 滤波
    state_means, state_covariances = kf.filter(Y)
    
    # 提取动态对冲比率
    dynamic_hedge_ratio = pd.Series(state_means.flatten(), index=stock_a.index)
    
    # 计算动态价差
    spread = stock_a - dynamic_hedge_ratio * stock_b
    
    return dynamic_hedge_ratio, spread

# 示例：比较固定对冲比率和动态对冲比率
print("=== 固定对冲比率 vs 动态对冲比率 ===")

# 使用前面生成的股票价格
fixed_hedge_ratio = OLS(stock_a, stock_b).fit().params[0]
fixed_spread = stock_a - fixed_hedge_ratio * stock_b

dynamic_hedge_ratio, dynamic_spread = dynamic_hedge_ratio_kalman(stock_a, stock_b)

print(f"固定对冲比率: {fixed_hedge_ratio:.4f}")
print(f"动态对冲比率范围: [{dynamic_hedge_ratio.min():.4f}, {dynamic_hedge_ratio.max():.4f}]")
print(f"固定价差标准差: {fixed_spread.std():.4f}")
print(f"动态价差标准差: {dynamic_spread.std():.4f}")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(dynamic_hedge_ratio.index, dynamic_hedge_ratio.values, 
             label='Dynamic Hedge Ratio', linewidth=2)
axes[0].axhline(y=fixed_hedge_ratio, color='red', linestyle='--', 
                label=f'Fixed Hedge Ratio ({fixed_hedge_ratio:.4f})')
axes[0].set_ylabel('Hedge Ratio')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(fixed_spread.index, fixed_spread.values, 
             label='Fixed Spread', alpha=0.7)
axes[1].plot(dynamic_spread.index, dynamic_spread.values, 
             label='Dynamic Spread', alpha=0.7)
axes[1].set_ylabel('Spread')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/images/statistical-arbitrage-mean-reversion/dynamic_hedge_ratio.png', 
           dpi=150, bbox_inches='tight')
print("✅ 动态对冲比率比较图表已保存")
```

### 改进二：机器学习增强信号

传统的Z-Score阈值法过于简单，可以使用机器学习模型预测价差的未来走势，生成更精准的交易信号。

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MLEnhancedPairsTrading:
    """机器学习增强的配对交易"""
    
    def __init__(self, stock_a, stock_b, lookback=20, prediction_horizon=5):
        """
        初始化
        
        参数：
        - stock_a: 股票A价格
        - stock_b: 股票B价格
        - lookback: 特征回溯期
        - prediction_horizon: 预测周期
        """
        self.stock_a = stock_a
        self.stock_b = stock_b
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        
        # 计算价差
        self.calculate_spread()
        
        # 准备机器学习特征
        self.prepare_features()
        
    def calculate_spread(self):
        """计算价差"""
        hedge_ratio = OLS(self.stock_a, self.stock_b).fit().params[0]
        self.spread = self.stock_a - hedge_ratio * self.stock_b
        
    def prepare_features(self):
        """准备机器学习特征"""
        features = pd.DataFrame(index=self.spread.index)
        
        # 特征1：Z-Score
        features['zscore'] = (self.spread - self.spread.rolling(60).mean()) / self.spread.rolling(60).std()
        
        # 特征2：价差的移动平均
        for lag in [5, 10, 20]:
            features[f'spread_ma_{lag}'] = self.spread.rolling(lag).mean()
        
        # 特征3：价差的波动率
        for lag in [10, 20, 60]:
            features[f'spread_vol_{lag}'] = self.spread.rolling(lag).std()
        
        # 特征4：成交量的变化（如果有）
        if hasattr(self, 'volume_a') and hasattr(self, 'volume_b'):
            features['volume_ratio'] = self.volume_a / self.volume_b
            features['volume_ratio_change'] = features['volume_ratio'].pct_change(5)
        
        # 特征5：技术指标
        features['rsi'] = self.calculate_rsi(self.spread, period=14)
        features['bollinger_upper'], features['bollinger_lower'] = self.calculate_bollinger_bands(self.spread, period=20)
        
        # 目标变量：未来n期价差收益
        self.target = self.spread.pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # 移除NaN
        valid_idx = features.dropna().index.intersection(self.target.dropna().index)
        self.features = features.loc[valid_idx]
        self.target = self.target.loc[valid_idx]
        
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    def train_model(self, model_type='random_forest'):
        """
        训练机器学习模型
        
        参数：
        - model_type: 模型类型 ('random_forest', 'gradient_boosting')
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.3, random_state=42
        )
        
        # 选择模型
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"=== {model_type} 模型评估 ===")
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.4f}")
        
        # 特征重要性
        feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.features.columns
        ).sort_values(ascending=False)
        
        print(f"\n特征重要性排序:")
        print(feature_importance)
        
    def generate_ml_signals(self, threshold=0.001):
        """
        基于机器学习预测生成交易信号
        
        参数：
        - threshold: 预测收益阈值
        
        返回：
        - signals: 交易信号
        """
        # 预测未来收益
        predicted_return = self.model.predict(self.features)
        
        # 生成信号
        signals = pd.Series(index=self.features.index, dtype=int)
        signals[predicted_return > threshold] = 1   # 预测上涨，做多价差
        signals[predicted_return < -threshold] = -1  # 预测下跌，做空价差
        signals[abs(predicted_return) <= threshold] = 0  # 预测变化不大，不持仓
        
        return signals

# 示例：使用机器学习增强配对交易
print("\n=== 机器学习增强配对交易 ===")

# 使用前面生成的股票价格
ml_strategy = MLEnhancedPairsTrading(stock_a, stock_b, lookback=20, prediction_horizon=5)

# 训练模型
ml_strategy.train_model(model_type='random_forest')

# 生成信号
ml_signals = ml_strategy.generate_ml_signals(threshold=0.001)

print(f"\n交易信号统计:")
print(f"  做多信号: {(ml_signals == 1).sum()}")
print(f"  做空信号: {(ml_signals == -1).sum()}")
print(f"  不持仓: {(ml_signals == 0).sum()}")
```

## 篮子交易策略

除了配对交易，我们还可以构建包含多只股票的**篮子交易（Basket Trading）**策略。其核心思想是：找到一个股票组合，使得该组合的收益率可以表示为若干个共同因子的线性组合，然后利用残差进行均值回归交易。

### 基于PCA的篮子交易

主成分分析（PCA）可以帮助我们识别股票组合中的主要共同因子，然后利用这些因子构建均值回归策略。

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCABasketTrading:
    """基于PCA的篮子交易"""
    
    def __init__(self, price_data, n_components=3):
        """
        初始化
        
        参数：
        - price_data: 价格数据（DataFrame, columns=股票名, index=日期）
        - n_components: PCA主成分数量
        """
        self.price_data = price_data
        self.n_components = n_components
        
        # 计算收益率
        self.returns_data = price_data.pct_change().dropna()
        
        # 标准化
        self.scaler = StandardScaler()
        self.returns_scaled = self.scaler.fit_transform(self.returns_data)
        
        # PCA分解
        self.apply_pca()
        
    def apply_pca(self):
        """应用PCA分解"""
        self.pca = PCA(n_components=self.n_components)
        self.pca_components = self.pca.fit_transform(self.returns_scaled)
        
        # 重建收益率
        self.returns_reconstructed = self.pca.inverse_transform(self.pca_components)
        self.returns_reconstructed = self.scaler.inverse_transform(self.returns_reconstructed)
        
        # 计算残差（原收益率 - 重建收益率）
        self.residuals = self.returns_scaled - self.pca.inverse_transform(self.pca_components)
        
        print(f"=== PCA分解结果 ===")
        print(f"解释方差比例: {self.pca.explained_variance_ratio_}")
        print(f"累计解释方差: {self.pca.explained_variance_ratio_.cumsum()}")
        
    def construct_mean_reversion_portfolio(self, component_idx=0):
        """
        构建均值回归组合
        
        参数：
        - component_idx: 使用哪个主成分（0=第一主成分）
        
        返回：
        - portfolio_weights: 组合权重
        """
        # 使用主成分载荷（loadings）作为组合权重
        loadings = self.pca.components_[component_idx]
        
        # 标准化权重
        portfolio_weights = loadings / np.abs(loadings).sum()
        
        print(f"\n=== 主成分 {component_idx+1} 载荷（组合权重）===")
        for i, stock in enumerate(self.returns_data.columns):
            print(f"  {stock}: {portfolio_weights[i]:.4f}")
        
        return portfolio_weights
    
    def backtest_residual_mean_reversion(self, entry_zscore=2.0, exit_zscore=0.5):
        """
        回残差均值回归策略
        
        参数：
        - entry_zscore: 入场Z-Score阈值
        - exit_zscore: 出场Z-Score阈值
        
        返回：
        - results: 回测结果
        """
        # 计算组合收益（使用所有股票的等权组合）
        portfolio_return = self.returns_data.mean(axis=1)
        
        # 计算PCA拟合值（共同因子部分）
        fitted_return = pd.DataFrame(
            self.returns_reconstructed,
            index=self.returns_data.index,
            columns=self.returns_data.columns
        ).mean(axis=1)
        
        # 计算残差
        residual = portfolio_return - fitted_return
        
        # 计算残差的Z-Score
        residual_mean = residual.rolling(60).mean()
        residual_std = residual.rolling(60).std()
        residual_zscore = (residual - residual_mean) / residual_std
        
        # 生成交易信号
        signals = pd.Series(index=residual.index, dtype=int)
        signals[residual_zscore > entry_zscore] = -1  # 残差过高，做空
        signals[residual_zscore < -entry_zscore] = 1   # 残差过低，做多
        signals[abs(residual_zscore) < exit_zscore] = 0  # 残差回归，平仓
        
        # 填充持仓
        signals = signals.replace(0, np.nan)
        signals = signals.fillna(method='ffill').fillna(0)
        
        # 计算策略收益
        strategy_return = signals.shift(1) * residual
        
        # 计算累积收益
        cumulative_return = (1 + strategy_return).cumprod()
        
        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        axes[0].plot(residual.index, residual.values, label='Residual', alpha=0.7)
        axes[0].plot(residual_zscore.index, residual_zscore.values * residual.std(), 
                    label='Z-Score (scaled)', alpha=0.7)
        axes[0].axhline(y=entry_zscore * residual.std(), color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=-entry_zscore * residual.std(), color='red', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Residual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(residual_zscore.index, residual_zscore.values, label='Residual Z-Score')
        axes[1].axhline(y=entry_zscore, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-entry_zscore, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=exit_zscore, color='green', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-exit_zscore, color='green', linestyle='--', alpha=0.5)
        axes[1].fill_between(residual_zscore.index, 0, residual_zscore.values,
                           where=(signals == 1), alpha=0.3, color='green')
        axes[1].fill_between(residual_zscore.index, 0, residual_zscore.values,
                           where=(signals == -1), alpha=0.3, color='red')
        axes[1].set_ylabel('Z-Score')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(cumulative_return.index, cumulative_return.values, 
                    label='Cumulative Return', color='purple', linewidth=2)
        axes[2].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Cumulative Return')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('public/images/statistical-arbitrage-mean-reversion/pca_basket.png', 
                   dpi=150, bbox_inches='tight')
        print("✅ PCA篮子交易回测图表已保存")
        
        return cumulative_return

# 示例：使用PCA篮子交易
print("\n=== PCA篮子交易策略 ===")

# 生成模拟价格数据（10只股票）
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
n_stocks = 10

# 生成相关股票收益率
base_return = np.random.normal(0.0005, 0.01, 500)
stock_returns = np.zeros((500, n_stocks))

for i in range(n_stocks):
    stock_returns[:, i] = base_return + np.random.normal(0, 0.005, 500)

# 转换为价格
stock_prices = 100 * np.exp(np.cumsum(stock_returns, axis=0))
price_data = pd.DataFrame(
    stock_prices,
    index=dates,
    columns=[f'STOCK_{i+1}' for i in range(n_stocks)]
)

# 创建PCA篮子交易策略
pca_strategy = PCABasketTrading(price_data, n_components=3)

# 构建均值回归组合
weights = pca_strategy.construct_mean_reversion_portfolio(component_idx=0)

# 回测残差均值回归
cumulative_return = pca_strategy.backtest_residual_mean_reversion(
    entry_zscore=2.0, 
    exit_zscore=0.5
)

print(f"\n总收益: {cumulative_return.iloc[-1] - 1:.2%}")
```

## 从回测到实盘：关键注意事项

统计套利策略从回测到实盘之间存在巨大的鸿沟。以下是几个关键注意事项：

### 1. 数据窥探偏差（Data Snooping Bias）

过度优化参数会导致回测结果虚高。解决方法：
- 使用样本外数据验证
- 进行Walk-Forward分析
- 使用多个不同的股票池测试

```python
def walk_forward_analysis(strategy_class, price_data, train_window=252, test_window=63):
    """
    Walk-Forward分析
    
    参数：
    - strategy_class: 策略类
    - price_data: 价格数据
    - train_window: 训练窗口（交易日数）
    - test_window: 测试窗口（交易日数）
    
    返回：
    - wf_results: Walk-Forward结果
    """
    n_total = len(price_data)
    wf_results = []
    
    for start in range(0, n_total - train_window - test_window, test_window):
        # 训练期
        train_start = start
        train_end = start + train_window
        
        # 测试期
        test_start = train_end
        test_end = min(train_end + test_window, n_total)
        
        # 训练策略
        train_data = price_data.iloc[train_start:train_end]
        strategy = strategy_class(train_data)
        optimal_params = strategy.optimize_parameters()
        
        # 测试策略
        test_data = price_data.iloc[test_start:test_end]
        test_strategy = strategy_class(test_data, **optimal_params)
        test_return = test_strategy.backtest()['cumulative_return'].iloc[-1]
        
        wf_results.append({
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'optimal_params': optimal_params,
            'test_return': test_return
        })
    
    return wf_results
```

### 2. 交易成本考量

统计套利策略通常交易频繁，交易成本对收益有重大影响。

```python
def calculate_transaction_cost_impact(trades, price_per_trade=0.001, min_commission=5):
    """
    计算交易成本影响
    
    参数：
    - trades: 交易记录（DataFrame, 包含'shares'和'price'列）
    - price_per_trade: 交易费率（0.001 = 0.1%）
    - min_commission: 最低佣金
    
    返回：
    - cost_summary: 成本汇总
    """
    # 计算交易金额
    trades['trade_value'] = abs(trades['shares'] * trades['price'])
    
    # 计算佣金
    trades['commission'] = trades['trade_value'] * price_per_trade
    trades['commission'] = trades['commission'].apply(lambda x: max(x, min_commission))
    
    # 汇总
    total_commission = trades['commission'].sum()
    avg_commission_rate = (trades['commission'] / trades['trade_value']).mean()
    n_trades = len(trades)
    
    cost_summary = {
        'total_commission': total_commission,
        'avg_commission_rate': avg_commission_rate,
        'n_trades': n_trades,
        'avg_trade_value': trades['trade_value'].mean()
    }
    
    print(f"=== 交易成本分析 ===")
    print(f"总佣金: ${total_commission:.2f}")
    print(f"平均佣金率: {avg_commission_rate:.4%}")
    print(f"交易次数: {n_trades}")
    print(f"平均交易金额: ${cost_summary['avg_trade_value']:.2f}")
    
    return cost_summary
```

### 3. 协整关系断裂

协整关系可能随时间断裂，导致策略失效。需要实时监测协整关系。

```python
def monitor_cointegration_break(stock_a, stock_b, window=60, significance=0.05):
    """
    监测协整关系是否断裂
    
    参数：
    - stock_a: 股票A价格
    - stock_b: 股票B价格
    - window: 滚动窗口
    - significance: 显著性水平
    
    返回：
    - breakpoints: 协整关系断裂点
    """
    n = len(stock_a)
    p_values = []
    breakpoints = []
    
    for i in range(window, n):
        # 使用滚动窗口数据检验协整
        y = stock_a.iloc[i-window:i]
        x = stock_b.iloc[i-window:i]
        
        _, p_value, _ = coint(y, x)
        p_values.append(p_value)
        
        # 判断是否断裂
        if p_value >= significance:
            breakpoints.append(stock_a.index[i])
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(stock_a.index[window:], stock_a.iloc[window:], label='Stock A', alpha=0.7)
    axes[0].plot(stock_b.index[window:], stock_b.iloc[window:], label='Stock B', alpha=0.7)
    for bp in breakpoints:
        axes[0].axvline(x=bp, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(stock_a.index[window:], p_values, label='p-value', color='blue')
    axes[1].axhline(y=significance, color='red', linestyle='--', label='Significance Level')
    axes[1].set_ylabel('p-value')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/statistical-arbitrage-mean-reversion/cointegration_break.png', 
               dpi=150, bbox_inches='tight')
    print(f"✅ 协整关系监测图表已保存")
    print(f"检测到 {len(breakpoints)} 个可能的协整断裂点")
    
    return breakpoints
```

### 4. 执行延迟和滑点

回测中通常假设以收盘价成交，但实盘中会存在执行延迟和滑点。

```python
def simulate_execution_delay(signals, execution_delay=1, slippage_rate=0.001):
    """
    模拟执行延迟和滑点
    
    参数：
    - signals: 交易信号
    - execution_delay: 执行延迟（交易日数）
    - slippage_rate: 滑点率
    
    返回：
    - adjusted_returns: 调整后的收益
    """
    # 延迟执行信号
    delayed_signals = signals.shift(execution_delay)
    
    # 计算理论收益
    theoretical_returns = delayed_signals * signals.pct_change()
    
    # 计算滑点成本
    slippage_cost = abs(delayed_signals.diff()) * slippage_rate
    
    # 调整后收益
    adjusted_returns = theoretical_returns - slippage_cost
    
    print(f"=== 执行延迟和滑点模拟 ===")
    print(f"执行延迟: {execution_delay} 个交易日")
    print(f"滑点率: {slippage_rate:.4%}")
    print(f"滑点成本占总收益比例: {slippage_cost.sum() / theoretical_returns.sum():.2%}")
    
    return adjusted_returns
```

### 5. 风险管理

统计套利策略也需要严格的风险管理。

```python
class StatisticalArbitrageRiskManager:
    """统计套利风险管理器"""
    
    def __init__(self, max_position_size=0.1, max_leverage=2.0, stop_loss=0.05):
        """
        初始化
        
        参数：
        - max_position_size: 最大单一头寸占比
        - max_leverage: 最大杠杆
        - stop_loss: 止损线
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.stop_loss = stop_loss
        
    def calculate_position_size(self, signal_strength, volatility, account_value):
        """
        计算头寸规模
        
        参数：
        - signal_strength: 信号强度（Z-Score）
        - volatility: 波动率
        - account_value: 账户价值
        
        返回：
        - position_size: 头寸规模
        """
        # 基于信号强度调整头寸
        signal_scaled = min(abs(signal_strength) / 3, 1.0)  # 归一化到[0,1]
        
        # 基于波动率调整头寸（波动率越高，头寸越小）
        vol_scaled = 1 / (1 + volatility * 100)
        
        # 计算头寸规模
        position_size = account_value * self.max_position_size * signal_scaled * vol_scaled
        
        return position_size
    
    def check_stop_loss(self, current_pnl, entry_value):
        """
        检查是否触发止损
        
        参数：
        - current_pnl: 当前盈亏
        - entry_value: 入场价值
        
        返回：
        - stop_triggered: 是否触发止损
        """
        pnl_ratio = current_pnl / entry_value
        
        if pnl_ratio <= -self.stop_loss:
            print(f"⚠️ 触发止损！亏损比例: {pnl_ratio:.2%}")
            return True
        
        return False
    
    def monitor_correlation(self, portfolio_returns, benchmark_returns, window=60):
        """
        监测组合与基准的相关性
        
        参数：
        - portfolio_returns: 组合收益
        - benchmark_returns: 基准收益
        - window: 滚动窗口
        
        返回：
        - correlation_warning: 相关性预警
        """
        rolling_corr = portfolio_returns.rolling(window).corr(benchmark_returns)
        
        # 如果相关性过高，说明失去市场中性
        high_corr_periods = rolling_corr[abs(rolling_corr) > 0.7]
        
        if len(high_corr_periods) > 0:
            print(f"⚠️ 警告：组合与基准相关性过高！")
            print(f"  高相关性时段: {len(high_corr_periods)} 个交易日")
            print(f"  最大相关性: {rolling_corr.max():.4f}")
            return True
        
        return False
```

## 实证分析：A股市场统计套利

让我们通过一个A股市场的真实案例，展示统计套利策略的完整实现流程。

### 案例：银行股配对交易

选择A股市场中估值相近、业务模式相似的银行股进行配对交易。

```python
# 模拟A股银行股数据
dates = pd.date_range('2020-01-01', periods=756, freq='D')

# 生成股价数据（假设工商银行和农业银行存在协整关系）
icbc_price = 5 + np.cumsum(np.random.normal(0.001, 0.02, 756))
abc_price = 4.8 + 0.92 * (icbc_price - 5) + np.cumsum(np.random.normal(0, 0.015, 756))

icbc = pd.Series(icbc_price, index=dates)
abc = pd.Series(abc_price, index=dates)

print("=== A股银行股配对交易案例 ===")

# 检验协整关系
_, p_value, _ = coint(icbc, abc)
print(f"协整检验p值: {p_value:.6f}")
print(f"结论: {'存在协整关系' if p_value < 0.05 else '不存在协整关系'}")

# 创建配对交易策略
bank_strategy = PairsTradingStrategy(icbc, abc, entry_threshold=2.0, exit_threshold=0.5)

# 回测
results = bank_strategy.backtest(transaction_cost=0.001)

# 可视化
bank_strategy.visualize(results)

# 风险管理
risk_manager = StatisticalArbitrageRiskManager(max_position_size=0.15, stop_loss=0.03)

# 计算头寸规模（假设账户价值100万）
position_size = risk_manager.calculate_position_size(
    signal_strength=2.5,
    volatility=results['net_return'].std() * np.sqrt(252),
    account_value=1000000
)

print(f"\n建议头寸规模: ¥{position_size:,.0f}")
```

## 总结

统计套利是一类优雅而强大的量化策略，但其成功实施需要严谨的方法和严格的风险管理。本文系统介绍了：

1. **统计学基础**：平稳性检验、协整关系、ADF检验
2. **配对交易策略**：基本框架、动态对冲比率、机器学习增强
3. **篮子交易策略**：基于PCA的均值回归
4. **实盘注意事项**：数据窥探偏差、交易成本、协整断裂、执行延迟、风险管理
5. **A股实证案例**：银行股配对交易

统计套利的核心在于**均值回归**，但真正的挑战在于：
- 如何识别真正的均值回归机会（而非价值陷阱）
- 如何管理交易成本（统计套利通常高频交易）
- 如何应对协整关系断裂（黑天鹅事件）
- 如何实现真正的市场中性（控制风险暴露）

只有充分理解这些挑战，并在策略设计中加以考虑，才能构建出在实盘中稳定盈利的统计套利系统。

---

**参考文献**

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule". Review of Financial Studies.
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis". Wiley.
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis". Wiley.
4. 李尉, 等. (2019). 《统计套利：理论与实证》. 金融研究.
5. 张然. (2020). 《配对交易策略在中国A股市场的应用研究》. 投资研究.

**免责声明**

本文仅供学术研究和交流使用，不构成任何投资建议。统计套利策略在实际应用中需要考虑交易成本、市场冲击、流动性风险、模型风险等多种现实因素。历史回测结果不代表未来收益，投资者应在充分理解策略逻辑和风险的前提下谨慎决策。
