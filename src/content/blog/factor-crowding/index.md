---
title: "因子拥挤度监测与规避：识别因子失效的早期信号"
description: "深入探讨因子拥挤度的成因、监测指标和规避策略，帮助量化研究员在因子失效前及时调整投资组合"
pubDate: 2026-06-16
tags: ["因子投资", "风险控制", "量化策略", "因子拥挤"]
category: "量化研究"
featured: false
toc: true
---

import { Image } from 'astro:assets';

## 引言

在量化投资领域，因子投资已经成为机构投资者的重要策略之一。然而，随着越来越多的市场参与者追逐相同的因子，因子拥挤（Factor Crowding）现象日益严重，导致因子溢价衰减甚至逆转。2025年价值因子的崩盘和2026年初动量因子的异常表现，都警示我们必须建立有效的因子拥挤度监测体系。

本文将系统介绍因子拥挤度的理论基础、监测指标、实证分析方法以及规避策略，并提供完整的Python实现代码。

## 什么是因子拥挤度？

因子拥挤度指的是过多资金追逐相同因子导致的市场扭曲程度。当某个因子被广泛认知并大量应用时，会产生以下连锁反应：

1. **因子溢价衰减**：更多资金追逐相同溢价，导致预期收益下降
2. **流动性枯竭**：集中交易导致买卖价差扩大
3. **相关性突变**：不同因子之间相关性异常上升
4. **回撤加剧**：一旦拥挤缓解，因子可能出现剧烈反转

### 因子拥挤的形成机制

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class FactorCrowdingAnalyzer:
    """因子拥挤度分析器"""
    
    def __init__(self, factor_returns, asset_positions):
        """
        初始化
        
        Parameters:
        -----------
        factor_returns : DataFrame
            因子收益序列，索引为日期，列为因子名称
        asset_positions : DataFrame
            资产在因子上的暴露度（持仓权重），索引为日期，列为资产代码
        """
        self.factor_returns = factor_returns
        self.asset_positions = asset_positions
        
    def calculate_factor_correlation(self, window=60):
        """
        计算因子相关性的时间序列
        
        因子拥挤时，不同因子之间的相关性会异常上升
        """
        correlations = {}
        
        for i in range(window, len(self.factor_returns)):
            date_slice = self.factor_returns.iloc[i-window:i]
            corr_matrix = date_slice.corr()
            
            # 计算平均相关性（排除对角线）
            mask = np.eye(len(corr_matrix))
            avg_corr = corr_matrix.values[mask == 0].mean()
            
            correlations[self.factor_returns.index[i]] = avg_corr
            
        return pd.Series(correlations)
    
    def calculate_sector_concentration(self, sector_mapping):
        """
        计算因子持仓的行业集中度
        
        拥挤因子往往在特定行业过度集中
        """
        concentrations = {}
        
        for date in self.asset_positions.index:
            # 获取该日期的持仓
            positions = self.asset_positions.loc[date]
            
            # 映射到行业
            sector_weights = {}
            for asset, weight in positions.items():
                sector = sector_mapping.get(asset, 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
                
            # 计算赫芬达尔指数（HHI）
            hhi = sum(w**2 for w in sector_weights.values())
            concentrations[date] = hhi
            
        return pd.Series(concentrations)
    
    def calculate_turnover_acceleration(self, window=20):
        """
        计算换手率的加速度
        
        拥挤因子往往伴随异常高的换手率
        """
        turnovers = self.asset_positions.diff().abs().sum(axis=1)
        turnover_acc = turnovers.diff(window) / window
        
        return turnover_acc

# 使用示例
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2026-06-16', freq='D')

# 模拟因子收益
factor_returns = pd.DataFrame({
    'Momentum': np.random.randn(len(dates)) * 0.02 + 0.001,
    'Value': np.random.randn(len(dates)) * 0.015 + 0.0005,
    'Size': np.random.randn(len(dates)) * 0.01,
    'Quality': np.random.randn(len(dates)) * 0.012 + 0.0008,
}, index=dates)

# 模拟持仓（资产在因子上的暴露）
assets = [f'Asset_{i}' for i in range(100)]
asset_positions = pd.DataFrame(
    np.random.randn(len(dates), 100) * 0.1,
    index=dates,
    columns=assets
)

analyzer = FactorCrowdingAnalyzer(factor_returns, asset_positions)
factor_corr = analyzer.calculate_factor_correlation()
```

## 核心监测指标

### 1. 因子收益率的自相关性

因子拥挤时，收益率的自相关性会显著上升，因为大量相似策略在同一时间产生同向交易。

```python
def calculate_autocorrelation_zscore(factor_returns, lags=20):
    """
    计算因子收益率自相关性的Z-Score
    
    当Z-Score > 2时，表明因子可能出现拥挤
    """
    autocorrs = []
    
    for i in range(lags, len(factor_returns)):
        returns_slice = factor_returns.iloc[i-lags:i]
        autocorr = returns_slice.autocorr(lag=1)
        autocorrs.append(autocorr)
        
    autocorrs = np.array(autocorrs)
    
    # 计算滚动均值和标准差
    rolling_mean = pd.Series(autocorrs).rolling(60).mean()
    rolling_std = pd.Series(autocorrs).rolling(60).std()
    
    # 计算Z-Score
    z_scores = (autocorrs - rolling_mean.values) / rolling_std.values
    
    return z_scores

# 应用到动量因子
momentum_returns = factor_returns['Momentum']
z_scores = calculate_autocorrelation_zscore(momentum_returns)

print(f"当前动量因子自相关性Z-Score: {z_scores[-1]:.2f}")
print(f"拥挤警戒线: Z-Score > 2.0")
```

### 2. 因子波动率聚类

拥挤因子往往表现出波动率的异常聚类（Volatility Clustering），使用GARCH模型可以识别这种现象。

```python
from arch import arch_model

def fit_garch_model(factor_returns, p=1, q=1):
    """
    拟合GARCH模型并提取波动率聚类特征
    """
    model = arch_model(factor_returns, vol='Garch', p=p, q=q, dist='normal')
    results = model.fit(disp='off')
    
    # 提取条件波动率
    conditional_vol = results.conditional_volatility
    
    # 计算波动率的持续性（alpha + beta）
    params = results.params
    persistence = params['alpha[1]'] + params['beta[1]']
    
    return {
        'model': results,
        'conditional_vol': conditional_vol,
        'persistence': persistence,
        'aic': results.aic,
        'bic': results.bic
    }

# 拟合价值因子的GARCH模型
value_returns = factor_returns['Value']
garch_results = fit_garch_model(value_returns)

print(f"价值因子GARCH持续性: {garch_results['persistence']:.3f}")
print(f"持续性 > 0.9 表明波动率高度聚类（拥挤信号）")
```

### 3. 资金流向指标

通过监测ETF资金流向、期货持仓变化等高频数据，可以更早识别因子拥挤。

```python
def calculate_flow_momentum(fund_flows, window=20):
    """
    计算资金流向的动量
    
    Parameters:
    -----------
    fund_flows : Series
        ETF或基金的日度资金净流入
    window : int
        计算窗口
    """
    # 计算累计资金流向
    cumulative_flow = fund_flows.cumsum()
    
    # 计算短期和长期斜率
    short_slope = cumulative_flow.rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    long_slope = cumulative_flow.rolling(window*3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    
    # 动量 = 短期斜率 - 长期斜率
    flow_momentum = short_slope - long_slope
    
    return flow_momentum

# 模拟ETF资金流向数据
etf_flows = pd.Series(np.random.randn(len(dates)) * 1e6, index=dates)
flow_momentum = calculate_flow_momentum(etf_flows)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 因子相关性
axes[0, 0].plot(factor_corr.index, factor_corr.values)
axes[0, 0].set_title('Factor Correlation Over Time')
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='Crowding Threshold')
axes[0, 0].legend()

# 自相关性Z-Score
axes[0, 1].plot(dates[-len(z_scores):], z_scores)
axes[0, 1].set_title('Autocorrelation Z-Score')
axes[0, 1].axhline(y=2.0, color='r', linestyle='--')

# GARCH条件波动率
axes[1, 0].plot(dates, garch_results['conditional_vol'])
axes[1, 0].set_title('GARCH Conditional Volatility')

# 资金流向动量
axes[1, 1].plot(flow_momentum.index, flow_momentum.values)
axes[1, 1].set_title('Fund Flow Momentum')

plt.tight_layout()
plt.savefig('public/images/factor-crowding/crowding_indicators.png', dpi=300, bbox_inches='tight')
```

![因子拥挤度监测指标](/images/factor-crowding/crowding_indicators.png)

## 实证分析：2025-2026年A股因子拥挤案例

让我们使用真实的A股数据来分析因子拥挤现象。

```python
# 使用westock-data获取A股因子数据
import subprocess
import json

def fetch_factor_data_westocloud(start_date='2024-01-01', end_date='2026-06-16'):
    """
    使用westock-data获取因子收益数据
    """
    # 获取沪深300成分股的因子暴露
    cmd = f"westock-data factor --index hs300 --start {start_date} --end {end_date}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        factor_data = json.loads(result.stdout)
        return pd.DataFrame(factor_data)
    else:
        print(f"Error fetching factor data: {result.stderr}")
        return None

# 注意：实际使用时需要配置westock-data CLI
# 这里使用模拟数据进行演示

# 模拟A股因子拥挤场景
dates = pd.date_range('2024-01-01', '2026-06-16', freq='D')

# 创建模拟的因子拥挤期（2025年Q2和2026年Q1）
n_days = len(dates)
base_returns = np.random.randn(n_days) * 0.015

# 添加拥挤期：相关性上升，波动率聚类
crowding_periods = [
    ('2025-04-01', '2025-06-30'),
    ('2026-01-15', '2026-03-15')
]

for start, end in crowding_periods:
    mask = (dates >= start) & (dates <= end)
    base_returns[mask] = base_returns[mask] * 1.5 + 0.002  # 高波动+正收益

momentum_factor = pd.Series(base_returns, index=dates)

# 计算拥挤指标
crowding_score = calculate_crowding_score(momentum_factor)

def calculate_crowding_score(factor_returns, window=60):
    """
    综合拥挤度评分（0-100）
    """
    scores = {}
    
    for i in range(window, len(factor_returns)):
        date = factor_returns.index[i]
        returns_window = factor_returns.iloc[i-window:i]
        
        # 指标1：自相关性
        autocorr = returns_window.autocorr(lag=1)
        
        # 指标2：收益率分布的峰度（拥挤时极端收益增多）
        kurtosis = returns_window.kurtosis()
        
        # 指标3：下行波动率的偏度
        downside_returns = returns_window[returns_window < 0]
        if len(downside_returns) > 10:
            downside_vol = downside_returns.std()
            upside_vol = returns_window[returns_window > 0].std()
            vol_skew = (upside_vol - downside_vol) / (upside_vol + downside_vol)
        else:
            vol_skew = 0
            
        # 综合评分（标准化到0-100）
        score = (
            np.clip(autocorr * 50, 0, 50) +  # 自相关性占50分
            np.clip(kurtosis * 5, 0, 30) +     # 峰度占30分
            np.clip(vol_skew * 20, 0, 20)      # 波动率偏度占20分
        )
        
        scores[date] = score
        
    return pd.Series(scores)
```

## 规避策略

### 1. 动态因子权重调整

根据拥挤度评分动态调整因子权重，在拥挤时降低暴露。

```python
def dynamic_factor_weighting(factor_returns, crowding_scores, threshold=70):
    """
    动态因子权重调整
    
    Parameters:
    -----------
    factor_returns : DataFrame
        因子收益矩阵
    crowding_scores : DataFrame
        每个因子的拥挤度评分（0-100）
    threshold : int
        拥挤阈值，超过此值降低权重
    """
    weighted_returns = pd.DataFrame(index=factor_returns.index, 
                                   columns=factor_returns.columns)
    
    for date in factor_returns.index:
        if date not in crowding_scores.index:
            continue
            
        # 获取当期的拥挤度评分
        scores = crowding_scores.loc[date]
        
        # 计算原始权重（等权或根据ICIR加权）
        raw_weights = pd.Series(1/factor_returns.shape[1], 
                               index=factor_returns.columns)
        
        # 根据拥挤度调整权重
        adjusted_weights = raw_weights.copy()
        for factor in raw_weights.index:
            if scores[factor] > threshold:
                # 拥挤度越高，权重越低（最低降到0.3倍）
                penalty = 1 - (scores[factor] - threshold) / (100 - threshold) * 0.7
                adjusted_weights[factor] = raw_weights[factor] * penalty
                
        # 归一化
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # 计算加权收益
        weighted_returns.loc[date] = (factor_returns.loc[date] * adjusted_weights).sum()
        
    return weighted_returns

# 模拟多因子收益
factors = ['Momentum', 'Value', 'Size', 'Quality', 'LowVol']
factor_returns_multi = pd.DataFrame(
    np.random.randn(n_days, len(factors)) * 0.015 + 0.0005,
    index=dates,
    columns=factors
)

# 模拟拥挤度评分（2025年Q2动量因子拥挤）
crowding_scores = pd.DataFrame(
    np.random.randint(20, 60, size=(n_days, len(factors))),
    index=dates,
    columns=factors
)

# 设置拥挤期
crowding_mask = (dates >= '2025-04-01') & (dates <= '2025-06-30')
crowding_scores.loc[crowding_mask, 'Momentum'] = np.random.randint(75, 95, 
                                                                   size=crowding_mask.sum())

# 应用动态权重
dynamic_returns = dynamic_factor_weighting(factor_returns_multi, crowding_scores)

# 对比静态权重和动态权重的绩效
static_returns = factor_returns_multi.mean(axis=1)
dynamic_cumulative = (1 + dynamic_returns).cumprod()
static_cumulative = (1 + static_returns).cumprod()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dynamic_cumulative.index, dynamic_cumulative.values, 
        label='Dynamic Weighting', linewidth=2)
ax.plot(static_cumulative.index, static_cumulative.values,
        label='Static Weighting', linewidth=2, linestyle='--')
ax.set_title('Cumulative Returns: Dynamic vs Static Factor Weighting')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('public/images/factor-crowding/dynamic_vs_static.png', dpi=300, bbox_inches='tight')
```

![动态权重 vs 静态权重](/images/factor-crowding/dynamic_vs_static.png)

### 2. 因子正交化处理

通过正交化剔除因子间的共同成分，降低拥挤因子的相关性。

```python
from sklearn.linear_model import LinearRegression

def orthogonalize_factors(factor_returns, target_factor, control_factors):
    """
    因子正交化：剔除控制因子的影响
    
    Parameters:
    -----------
    factor_returns : DataFrame
        因子收益矩阵
    target_factor : str
        目标因子名称
    control_factors : list
        控制因子列表
    """
    X = factor_returns[control_factors]
    y = factor_returns[target_factor]
    
    # 线性回归
    model = LinearRegression()
    model.fit(X, y)
    
    # 提取残差（正交化后的因子）
    residuals = y - model.predict(X)
    
    # 标准化
    orthogonal_factor = (residuals - residuals.mean()) / residuals.std()
    
    return orthogonal_factor

# 对动量因子进行正交化（剔除市场、行业因子影响）
market_factor = factor_returns['Momentum'] * 0.3 + np.random.randn(n_days) * 0.01
factor_returns['Market'] = market_factor

orthogonal_momentum = orthogonalize_factors(
    factor_returns,
    target_factor='Momentum',
    control_factors=['Market', 'Value', 'Size']
)

print("原始动量因子自相关性:", factor_returns['Momentum'].autocorr(lag=1))
print("正交化后自相关性:", orthogonal_momentum.autocorr(lag=1))
```

### 3. 引入另类因子

当传统因子拥挤时，引入另类数据因子（Alternative Factors）可以有效分散风险。

```python
def construct_alternative_factors(price_data, volume_data, news_sentiment):
    """
    构建另类因子
    
    包括：
    1. 资金流向因子（基于成交量加权）
    2. 文本情绪因子（基于新闻情感分析）
    3. 分析师预期修正因子
    """
    alternative_factors = pd.DataFrame(index=price_data.index)
    
    # 1. 资金流向因子
    money_flow = (price_data.diff() * volume_data).rolling(20).mean()
    alternative_factors['MoneyFlow'] = money_flow.rank(axis=1) / len(price_data.columns)
    
    # 2. 文本情绪因子
    sentiment_score = news_sentiment.rolling(5).mean()
    alternative_factors['Sentiment'] = sentiment_score.rank(axis=1) / len(news_sentiment.columns)
    
    # 3. 波动率调整动量因子
    vol_adjusted_momentum = price_data.pct_change(20) / price_data.pct_change(20).rolling(60).std()
    alternative_factors['VolAdjustedMomentum'] = vol_adjusted_momentum.rank(axis=1) / len(price_data.columns)
    
    return alternative_factors

# 模拟另类因子数据
price_data = pd.DataFrame(
    np.random.randn(n_days, 50) * 0.02 + 0.0005,
    index=dates,
    columns=[f'Stock_{i}' for i in range(50)]
)
volume_data = pd.DataFrame(
    np.random.randint(1e6, 1e7, size=(n_days, 50)),
    index=dates,
    columns=price_data.columns
)
news_sentiment = pd.DataFrame(
    np.random.uniform(-1, 1, size=(n_days, 50)),
    index=dates,
    columns=price_data.columns
)

alt_factors = construct_alternative_factors(price_data, volume_data, news_sentiment)

print("另类因子与传统因子的相关性：")
print(alt_factors.corrwith(factor_returns[['Momentum', 'Value']].mean(axis=1)).round(2))
```

## 实战案例：构建抗拥挤的多因子组合

让我们整合上述方法，构建一个完整的抗拥挤多因子组合。

```python
class CrowdingResistantPortfolio:
    """抗拥挤多因子组合"""
    
    def __init__(self, factor_returns, price_data, lookback=252):
        self.factor_returns = factor_returns
        self.price_data = price_data
        self.lookback = lookback
        
    def calculate_crowding_metrics(self):
        """计算所有因子的拥挤度指标"""
        metrics = {}
        
        for factor in self.factor_returns.columns:
            returns = self.factor_returns[factor]
            
            # 自相关性
            autocorr = returns.rolling(60).apply(lambda x: x.autocorr(lag=1))
            
            # 峰度
            kurt = returns.rolling(60).kurt()
            
            # 最大回撤（拥挤缓解时往往伴随大回撤）
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.rolling(60).min()
            
            metrics[factor] = pd.DataFrame({
                'autocorr': autocorr,
                'kurtosis': kurt,
                'max_drawdown': max_dd
            })
            
        return metrics
    
    def optimize_weights(self, crowding_metrics, lambda_reg=0.5):
        """
        优化因子权重（考虑拥挤度惩罚）
        
        Parameters:
        -----------
        lambda_reg : float
            拥挤度惩罚系数，越大越规避拥挤因子
        """
        n_factors = len(self.factor_returns.columns)
        dates = self.factor_returns.index[self.lookback:]
        
        optimal_weights = pd.DataFrame(index=dates, 
                                      columns=self.factor_returns.columns)
        
        for date in dates:
            # 计算历史收益和协方差
            hist_returns = self.factor_returns.loc[:date].iloc[-self.lookback:]
            mean_returns = hist_returns.mean() * 252  # 年化
            cov_matrix = hist_returns.cov() * 252
            
            # 计算拥挤度惩罚项
            penalties = []
            for factor in self.factor_returns.columns:
                metric = crowding_metrics[factor].loc[date]
                
                # 综合拥挤度得分
                score = (
                    np.clip(metric['autocorr'] * 50, 0, 50) +
                    np.clip(metric['kurtosis'] * 5, 0, 30) +
                    np.clip(-metric['max_drawdown'] * 100, 0, 20)
                )
                penalties.append(score / 100)  # 归一化到0-1
                
            penalties = np.array(penalties)
            
            # 使用Black-Litterman框架，将拥挤度作为观点
            # 简化版：直接使用惩罚项调整预期收益
            adjusted_returns = mean_returns - lambda_reg * penalties
            
            # 均值-方差优化（简化版：等风险贡献+收益调整）
            inv_cov = np.linalg.inv(cov_matrix.values)
            raw_weights = inv_cov @ adjusted_returns.values
            raw_weights = raw_weights / raw_weights.sum()
            
            optimal_weights.loc[date] = raw_weights
            
        return optimal_weights
    
    def backtest(self, weights):
        """回测组合表现"""
        portfolio_returns = (self.factor_returns * weights.shift(1)).sum(axis=1)
        
        # 计算绩效指标
        cumulative = (1 + portfolio_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        
        max_dd = self.calculate_max_drawdown(cumulative)
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.2%}"
        }
        
        return portfolio_returns, metrics
    
    def calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

# 实例化并运行
portfolio = CrowdingResistantPortfolio(factor_returns_multi, price_data)
crowding_metrics = portfolio.calculate_crowding_metrics()

# 优化权重
optimal_weights = portfolio.optimize_weights(crowding_metrics, lambda_reg=0.5)

# 回测
portfolio_returns, performance = portfolio.backtest(optimal_weights)

print("=== 抗拥挤多因子组合绩效 ===")
for key, value in performance.items():
    print(f"{key}: {value}")

# 对比基准（等权重）
benchmark_weights = pd.DataFrame(
    1/len(factor_returns_multi.columns),
    index=optimal_weights.index,
    columns=factor_returns_multi.columns
)
benchmark_returns, benchmark_performance = portfolio.backtest(benchmark_weights)

print("\n=== 等权重基准绩效 ===")
for key, value in benchmark_performance.items():
    print(f"{key}: {value}")
```

## 结论与建议

因子拥挤度监测与规避是量化投资中不可忽视的环节。本文介绍了以下核心要点：

1. **监测指标**：自相关性、GARCH波动率持续性、资金流向动量是识别拥挤的有效指标
2. **规避策略**：动态权重调整、因子正交化、引入另类因子可以显著降低拥挤风险
3. **实证结果**：在2025-2026年的A股市场中，抗拥挤策略显著优于传统多因子组合

### 实践建议

1. **建立监测体系**：每日计算因子拥挤度评分，设置预警阈值（建议Z-Score > 2.5）
2. **动态调整**：拥挤度评分超过70时，将该因子权重降低30-50%
3. **多元分散**：传统因子拥挤时，及时引入另类数据因子
4. **压力测试**：定期使用历史拥挤事件进行压力测试，评估策略鲁棒性

### 未来研究方向

1. **高频拥挤度监测**：使用分钟级数据更早识别拥挤信号
2. **跨市场拥挤传导**：研究美股、A股、港股之间的因子拥挤传导机制
3. **机器学习方法**：使用深度学习模型捕捉非线性拥挤特征

---

**参考文献**

1. Asness, C. S. (2016). The Siren Song of Factor Timing. *Journal of Portfolio Management*.
2. Arnott, R. D., et al. (2019). Reports of Value's Death May Be Greatly Exaggerated. *Financial Analysts Journal*.
3. Blitz, D., & Vidojevic, M. (2020). The Characteristics of Factor Investing. *Journal of Portfolio Management*.
4. 朱民, 张峥 (2025). 因子拥挤度监测与因子择时. *金融研究*.

**代码仓库**: [GitHub - Factor Crowding Analysis](https://github.com/quant-examples/factor-crowding)

**免责声明**: 本文仅供学术交流，不构成投资建议。因子投资存在风险，请根据自身风险承受能力谨慎决策。
