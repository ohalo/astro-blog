---
title: "统计套利：均值回归策略"
publishDate: '2026-06-17'
description: "统计套利：均值回归策略 - halo的技术博客"
tags:
 - AI观察
language: Chinese
image: "/images/statistical-arbitrage-mean-reversion/cover.jpg"
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是量化投资中的重要策略类别，它利用数学模型识别资产价格之间的暂时偏离，通过均值回归特性获取收益。本文将深入探讨统计套利的理论基础、方法论框架以及实战应用。

## 统计套利的理论基础

### 有效市场假说与定价偏差

有效市场假说（EMH）认为资产价格反映了所有可用信息。然而，现实市场中存在着各类摩擦和约束，导致价格偏离其合理价值：

1. **交易约束**：卖空限制、交易成本、流动性约束
2. **行为偏差**：投资者情绪、认知偏差、羊群效应
3. **信息不对称**：信息传递的时滞、差异化解读
4. **结构性断裂**：制度变化、突发事件、技术性冲击

统计套利正是利用这些暂时性的定价偏差，通过数理模型识别并捕捉均值回归的机会。

### 均值回归的理论依据

均值回归（Mean Reversion）是指资产价格或收益率在长期内倾向于回归其历史平均水平的现象。其理论依据包括：

- **均值回归过程**：Ornstein-Uhlenbeck (OU) 过程
- **协整关系**：非平稳序列的线性组合可能平稳
- **方差比率检验**：价格序列的方差随时间以非线性速度增长

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import het_arch

def test_mean_reversion(price_series, max_lag=20):
    """
    检验价格序列的均值回归特性
    
    参数:
        price_series: Series, 价格序列
        max_lag: int, 最大滞后阶数
    
    返回:
        dict: 包含各项检验结果
    """
    results = {}
    
    # 1. Augmented Dickey-Fuller检验
    adf_result = adfuller(price_series, maxlag=max_lag, autolag='AIC')
    results['ADF_Statistic'] = adf_result[0]
    results['ADF_pvalue'] = adf_result[1]
    results['ADF_Stationary'] = results['ADF_pvalue'] < 0.05
    
    # 2. 方差比率检验
    def variance_ratio(returns, hold_period):
        """计算方差比率"""
        n = len(returns)
        # 计算hold_period期收益率的方差
        multi_period_returns = np.log(price_series / price_series.shift(hold_period)).dropna()
        var_multi = np.var(multi_period_returns) / hold_period
        
        # 计算1期收益率的方差
        single_period_returns = np.log(price_series / price_series.shift(1)).dropna()
        var_single = np.var(single_period_returns)
        
        return var_multi / var_single
    
    returns = np.diff(np.log(price_series))
    vr_2 = variance_ratio(returns, 2)
    vr_4 = variance_ratio(returns, 4)
    vr_8 = variance_ratio(returns, 8)
    
    results['Variance_Ratio_2'] = vr_2
    results['Variance_Ratio_4'] = vr_4
    results['Variance_Ratio_8'] = vr_8
    results['Mean_Reversion_Score'] = (vr_2 + vr_4 + vr_8) / 3
    
    # 3. Hurst指数
    def hurst_exponent(price_series):
        """计算Hurst指数"""
        lags = range(2, min(50, len(price_series)//2))
        tau = [np.std(np.subtract(price_series[lag:], price_series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2
    
    results['Hurst_Exponent'] = hurst_exponent(price_series.values)
    results['Is_Mean_Reverting'] = results['Hurst_Exponent'] < 0.5
    
    return results

# 使用示例
price_data = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
test_results = test_mean_reversion(price_data['AAPL'])

print(f"ADF统计量: {test_results['ADF_Statistic']:.4f}")
print(f"ADF p值: {test_results['ADF_pvalue']:.4f}")
print(f"是否平稳: {test_results['ADF_Stationary']}")
print(f"Hurst指数: {test_results['Hurst_Exponent']:.4f}")
print(f"是否均值回归: {test_results['Is_Mean_Reverting']}")
```

## 配对交易的核心框架

### 配对选择：协整vs相关性

配对交易（Pairs Trading）是统计套利中最经典的策略。选择合适的配对是成功的关键：

```python
def select_pairs_cointegration(price_data, significance=0.05):
    """
    基于协整关系选择交易配对
    
    参数:
        price_data: DataFrame, 多资产价格数据
        significance: float, 协整检验的显著性水平
    
    返回:
        list: 协整配对的列表
    """
    n_assets = price_data.shape[1]
    pairs = []
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            asset1 = price_data.columns[i]
            asset2 = price_data.columns[j]
            
            # 协整检验
            coint_result = coint(price_data[asset1], price_data[asset2])
            coint_pvalue = coint_result[1]
            
            if coint_pvalue < significance:
                # 计算对冲比例（通过OLS回归）
                from sklearn.linear_model import LinearRegression
                X = price_data[asset2].values.reshape(-1, 1)
                y = price_data[asset1].values
                model = LinearRegression()
                model.fit(X, y)
                hedge_ratio = model.coef_[0]
                
                pairs.append({
                    'Asset1': asset1,
                    'Asset2': asset2,
                    'Hedge_Ratio': hedge_ratio,
                    'Coint_pvalue': coint_pvalue,
                    'Spread_Mean': np.mean(y - hedge_ratio * X.flatten()),
                    'Spread_Std': np.std(y - hedge_ratio * X.flatten())
                })
    
    return pairs

# 对比：基于相关性的配对选择
def select_pairs_correlation(returns_data, corr_threshold=0.7):
    """
    基于相关性选择交易配对（不推荐，仅用于对比）
    """
    corr_matrix = returns_data.corr()
    pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                pairs.append({
                    'Asset1': corr_matrix.columns[i],
                    'Asset2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    return pairs
```

### 交易信号的构建

构建有效的交易信号需要确定：
1. **入场阈值**：价差偏离多大时入场
2. **出场阈值**：何时平仓
3. **止损机制**：何时认亏出场

```python
class PairsTradingStrategy:
    """
    配对交易策略框架
    """
    
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5, stop_zscore=3.0):
        """
        初始化策略参数
        
        参数:
            entry_zscore: float, 入场Z-score阈值
            exit_zscore: float, 出场Z-score阈值
            stop_zscore: float, 止损Z-score阈值
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        
    def calculate_spread(self, price1, price2, method='OLS', window=60):
        """
        计算价格价差
        
        参数:
            price1, price2: Series, 两只股票的价格
            method: str, 计算方法（'OLS'或'Ratio'）
            window: int, 滚动窗口
        
        返回:
            Series: 价差序列
        """
        if method == 'OLS':
            # 滚动OLS回归计算对冲比例
            hedge_ratios = pd.Series(index=price1.index[window:])
            for i in range(window, len(price1)):
                X = price2.iloc[i-window:i].values.reshape(-1, 1)
                y = price1.iloc[i-window:i].values
                model = LinearRegression()
                model.fit(X, y)
                hedge_ratios.iloc[i-window] = model.coef_[0]
            
            # 对齐索引
            hedge_ratios = hedge_ratios.reindex(price1.index, method='ffill')
            spread = price1 - hedge_ratios * price2
            
        elif method == 'Ratio':
            # 简单价格比
            spread = price1 / price2
        
        return spread
    
    def generate_signals(self, spread):
        """
        生成交易信号
        
        参数:
            spread: Series, 价差序列
        
        返回:
            DataFrame: 包含价差值、Z-score和交易信号的DataFrame
        """
        # 计算Z-score（滚动均值和标准差）
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()
        z_score = (spread - spread_mean) / spread_std
        
        # 初始化信号
        signals = pd.DataFrame(index=spread.index)
        signals['Spread'] = spread
        signals['Z_Score'] = z_score
        signals['Position'] = 0
        
        # 生成交易信号
        position = 0
        for i in range(1, len(signals)):
            if position == 0:  # 空仓
                if z_score.iloc[i] > self.entry_zscore:
                    position = -1  # 做空价差
                elif z_score.iloc[i] < -self.entry_zscore:
                    position = 1   # 做多价差
            elif position == 1:  # 持多仓
                if z_score.iloc[i] <= self.exit_zscore:
                    position = 0  # 平仓
                elif z_score.iloc[i] > self.stop_zscore:
                    position = 0  # 止损
            elif position == -1:  # 持空仓
                if z_score.iloc[i] >= -self.exit_zscore:
                    position = 0  # 平仓
                elif z_score.iloc[i] < -self.stop_zscore:
                    position = 0  # 止损
            
            signals['Position'].iloc[i] = position
        
        return signals
    
    def backtest(self, price1, price2, signals):
        """
        回测策略
        
        参数:
            price1, price2: Series, 两只股票的价格
            signals: DataFrame, 交易信号
        
        返回:
            DataFrame: 回测结果
        """
        # 计算每日收益
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        # 策略收益（假设等权投资）
        strategy_returns = signals['Position'].shift(1) * (returns1 - returns2)
        
        # 计算累积收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算绩效指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_dd = self.calculate_max_drawdown(cumulative_returns)
        
        results = {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_dd,
            'Cumulative_Returns': cumulative_returns,
            'Strategy_Returns': strategy_returns
        }
        
        return results
    
    def calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

# 使用示例
strategy = PairsTradingStrategy(entry_zscore=2.0, exit_zscore=0.5, stop_zscore=3.0)

# 假设有两组价格数据
price1 = price_data['AAPL']
price2 = price_data['MSFT']

# 计算价差
spread = strategy.calculate_spread(price1, price2, method='OLS', window=60)

# 生成信号
signals = strategy.generate_signals(spread)

# 回测
results = strategy.backtest(price1, price2, signals)

print(f"年化收益: {results['Annual_Return']:.2%}")
print(f"夏普比率: {results['Sharpe_Ratio']:.2f}")
print(f"最大回撤: {results['Max_Drawdown']:.2%}")
```

## 多因子统计套利模型

### 主成分分析（PCA）方法

单一配对交易受限于可交易的配对数量。多因子模型可以同时处理大量资产：

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_statistical_arbitrage(price_data, n_components=5, lookback=60):
    """
    基于PCA的统计套利模型
    
    参数:
        price_data: DataFrame, 价格数据
        n_components: int, 主成分数量
        lookback: int, 滚动窗口
    
    返回:
        DataFrame: 残差收益和交易信号
    """
    # 转换为收益率
    returns = price_data.pct_change().dropna()
    
    # 标准化
    scaler = StandardScaler()
    normalized_returns = pd.DataFrame(
        scaler.fit_transform(returns),
        index=returns.index,
        columns=returns.columns
    )
    
    # 滚动PCA
    residuals = pd.DataFrame(index=returns.index[lookback:], columns=returns.columns)
    
    for i in range(lookback, len(returns)):
        # 提取训练数据
        train_returns = normalized_returns.iloc[i-lookback:i]
        
        # PCA分解
        pca = PCA(n_components=n_components)
        pca.fit(train_returns)
        
        # 重构
        reconstructed = pca.inverse_transform(pca.transform(train_returns))
        
        # 计算残差
        residual = train_returns - reconstructed
        
        # 保存最新的残差
        residuals.iloc[i-lookback] = residual.iloc[-1]
    
    # 生成交易信号（基于残差的Z-score）
    signals = pd.DataFrame(index=residuals.index, columns=residuals.columns)
    for col in residuals.columns:
        z_score = (residuals[col] - residuals[col].rolling(lookback).mean()) / residuals[col].rolling(lookback).std()
        signals[col] = -z_score  # 负向押注均值回归
    
    return residuals, signals

# 使用示例
residuals, signals = pca_statistical_arbitrage(price_data, n_components=5, lookback=60)

# 计算策略收益
strategy_returns = (signals.shift(1) * returns.loc[signals.index]).sum(axis=1)
```

### 预期收益最大化框架

将统计套利形式化为预期收益最大化问题：

```python
from scipy.optimize import minimize

def expected_return_stat_arb(price_data, factor_model, risk_model, lambda_reg=0.1):
    """
    预期收益最大化的统计套利框架
    
    参数:
        price_data: DataFrame, 价格数据
        factor_model: dict, 因子模型参数
        risk_model: dict, 风险模型参数
        lambda_reg: float, 正则化参数
    
    返回:
        array: 最优投资组合权重
    """
    returns = price_data.pct_change().dropna()
    n_assets = returns.shape[1]
    
    # 估计预期收益（基于均值回归）
    expected_returns = returns.rolling(60).mean().iloc[-1]
    
    # 估计协方差矩阵
    cov_matrix = returns.rolling(60).cov().iloc[-n_assets:]
    
    # 市场中性约束：权重和为0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w)},
                   {'type': 'eq', 'fun': lambda w: factor_model['beta'].dot(w)},  # beta中性
                   {'type': 'eq', 'fun': lambda w: factor_model['industry'].dot(w)})  # 行业中性
    
    # 边界约束：限制单个资产权重
    bounds = tuple((-0.1, 0.1) for _ in range(n_assets))
    
    # 目标函数：最大化预期收益 - 风险惩罚 - 正则化
    def objective(w):
        expected_ret = -expected_returns.dot(w)  # 负号因为scipy是最小化
        risk_penalty = w.dot(cov_matrix).dot(w)
        regularization = lambda_reg * np.sum(np.abs(w))
        return expected_ret + risk_penalty + regularization
    
    # 优化
    initial_weights = np.zeros(n_assets)
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x

# 使用示例
factor_model = {
    'beta': np.random.randn(n_assets),
    'industry': np.random.randn(n_assets)
}
risk_model = {}

optimal_weights = expected_return_stat_arb(price_data, factor_model, risk_model, lambda_reg=0.1)
```

## 风险控制与实务要点

### 1. 集中度风险控制

统计套利策略容易在少数资产上过度集中：

```python
def concentration_risk_control(weights, max_weight=0.05, max_sector_weight=0.20):
    """
    集中度风险控制
    
    参数:
        weights: array, 投资组合权重
        max_weight: float, 单个资产最大权重
        max_sector_weight: float, 单个行业最大权重
    
    返回:
        array: 调整后的权重
    """
    # 限制单个资产权重
    adjusted_weights = np.clip(weights, -max_weight, max_weight)
    
    # 重新归一化（保持市场中性）
    adjusted_weights = adjusted_weights - np.mean(adjusted_weights)
    
    return adjusted_weights

# 行业集中度检查
def check_sector_concentration(weights, sector_mapping):
    """
    检查行业集中度
    
    参数:
        weights: array, 投资组合权重
        sector_mapping: dict, 资产到行业的映射
    
    返回:
        dict: 各行业权重
    """
    sector_weights = {}
    for asset, weight in zip(price_data.columns, weights):
        sector = sector_mapping.get(asset, 'Other')
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    return sector_weights
```

### 2. 交易成本与滑点

高频调仓的统计套利策略对交易成本敏感：

```python
def transaction_cost_analysis(weights, previous_weights, transaction_cost=0.001):
    """
    交易成本分析
    
    参数:
        weights: array, 当前权重
        previous_weights: array, 上期权重
        transaction_cost: float, 单边交易成本
    
    返回:
        dict: 包含换手率和交易成本的结果
    """
    # 计算换手率
    turnover = np.sum(np.abs(weights - previous_weights))
    
    # 计算交易成本
    cost = turnover * transaction_cost
    
    # 成本调整后收益
    gross_return = weights.dot(expected_returns)
    net_return = gross_return - cost
    
    return {
        'Turnover': turnover,
        'Transaction_Cost': cost,
        'Gross_Return': gross_return,
        'Net_Return': net_return,
        'Cost_Drag': cost / gross_return if gross_return != 0 else np.inf
    }
```

### 3. 模型衰减与适应性

统计套利策略面临模型衰减（Model Decay）问题：

```python
def model_decay_monitoring(actual_returns, predicted_returns, window=20):
    """
    模型衰减监控
    
    参数:
        actual_returns: Series, 实际收益
        predicted_returns: Series, 预测收益
        window: int, 滚动窗口
    
    返回:
        DataFrame: 包含模型表现指标的DataFrame
    """
    # 计算滚动IC
    ic_series = pd.Series(index=actual_returns.index[window:])
    for i in range(window, len(actual_returns)):
        ic = spearmanr(predicted_returns.iloc[i-window:i], 
                      actual_returns.iloc[i-window:i])[0]
        ic_series.iloc[i-window] = ic
    
    # 计算滚动R²
    r2_series = pd.Series(index=actual_returns.index[window:])
    for i in range(window, len(actual_returns)):
        from sklearn.metrics import r2_score
        r2 = r2_score(actual_returns.iloc[i-window:i], 
                     predicted_returns.iloc[i-window:i])
        r2_series.iloc[i-window] = r2
    
    # 检测衰减（IC或R²的持续下降）
    ic_trend = np.polyfit(range(len(ic_series)), ic_series.values, 1)[0]
    r2_trend = np.polyfit(range(len(r2_series)), r2_series.values, 1)[0]
    
    decay_signal = (ic_trend < -0.01) or (r2_trend < -0.01)
    
    return {
        'IC_Series': ic_series,
        'R2_Series': r2_series,
        'IC_Trend': ic_trend,
        'R2_Trend': r2_trend,
        'Decay_Detected': decay_signal
    }
```

## 总结

统计套利与均值回归策略为量化投资提供了系统化、规则化的收益来源。成功的统计套利需要：

1. **严谨的数学基础**：理解协整、均值回归等概念
2. **精细的模型构建**：选择合适的配对或因子模型
3. **严格的风险管理**：控制集中度、交易成本、模型风险
4. **持续的监控优化**：跟踪模型表现，及时适应市场变化

随着市场效率的提升，简单的统计套利策略逐渐失效。未来的方向包括：
- 结合机器学习提升预测能力
- 拓展到更丰富的资产类别和市场
- 融合基本面信息的混合策略
- 考虑市场摩擦的现实约束优化

统计套利不是圣杯，但在严谨的框架下，它仍然是量化投资工具箱中的重要组成部分。

---

**参考文献**：

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.
4. Hogan, S., Jarrow, R., Teo, M., & Warachka, M. (2004). "Testing Market Efficiency using Statistical Arbitrage." *Mathematical Finance*.
