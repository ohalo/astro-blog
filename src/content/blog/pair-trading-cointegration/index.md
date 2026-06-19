---
title: "配对交易与协整分析"
description: "深入探讨配对交易的理论基础与实践方法，学习协整检验、套利策略构建和风险管理，掌握市场中性策略的核心技术。"
date: 2026-06-19
tags: ["配对交易", "协整分析", "统计套利", "市场中性"]
categories: ["量化交易"]
image: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

配对交易（Pairs Trading）是统计套利中最经典的策略之一。它通过寻找价格具有长期均衡关系的资产对，在价格偏离时进行反向操作，等待均值回归获利。协整分析是识别这种长期均衡关系的核心工具。本文将深入探讨配对交易的理论基础、协整检验方法、策略构建和实战风险管理。

## 配对交易的理论基础

### 为什么配对交易有效？

配对交易的有效性建立在三个经济学原理之上：

1. **均值回归**：相关资产的相对价格围绕长期均衡波动，偏离后会回归
2. **市场中性**：多空对冲消除市场系统性风险，获取纯alpha
3. **统计套利**：利用统计规律而非基本面分析，适合高频执行

### 配对交易的核心假设

成功的配对交易需要满足：

- **协整关系**：资产价格存在长期均衡
- **均值回复**：短期偏离会回归长期趋势
- **流动性充足**：能够低成本建仓平仓
- **低交易成本**：频繁交易要求低成本

## 协整理论与检验方法

### 什么是协整？

协整（Cointegration）描述的是两个或多个非平稳时间序列的线性组合是平稳的。

**数学定义**：
若时间序列 $X_t$ 和 $Y_t$ 都是 I(1) 过程（一阶单整），存在参数 $\beta$ 使得：
$$Z_t = Y_t - \beta X_t$$
是平稳过程（I(0)），则称 $X_t$ 和 $Y_t$ 协整。

### Engle-Granger 两步法

最经典的协整检验方法：

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

class CointegrationAnalysis:
    """协整分析工具类"""
    
    def __init__(self, price_data):
        """
        参数：
        price_data: DataFrame, 包含多个资产的价格数据
        """
        self.prices = price_data
        self.hedge_ratios = {}
        self.spreads = {}
        
    def engle_granger_test(self, asset1, asset2, significance=0.05):
        """
        Engle-Granger两步法协整检验
        
        返回：
        - coint_stat: 协整统计量
        - p_value: p值
        - is_cointegrated: 是否协整
        - hedge_ratio: 对冲比率
        """
        # 第一步：协整回归
        Y = self.prices[asset1]
        X = self.prices[asset2]
        X = sm.add_constant(X)
        
        model = OLS(Y, X).fit()
        hedge_ratio = model.params[asset2]
        spread = Y - hedge_ratio * self.prices[asset2]
        
        # 第二步：检验残差平稳性
        adf_stat, p_value, _ = adfuller(spread, autolag='AIC')
        
        # 使用专门的协整检验
        coint_stat, p_value, _ = coint(Y, self.prices[asset2])
        
        is_cointegrated = p_value < significance
        
        result = {
            'coint_statistic': coint_stat,
            'p_value': p_value,
            'is_cointegrated': is_cointegrated,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread.mean(),
            'spread_std': spread.std()
        }
        
        return result
    
    def johansen_test(self, assets, det_order=0, k_ar_diff=1):
        """
        Johansen协整检验（适用于多资产）
        
        参数：
        det_order: 确定性项顺序 (-1: no const, 0: const, 1: const+trend)
        k_ar_diff: 滞后阶数
        """
        from statsmodels.tsa.johansen import coint_johansen
        
        data = self.prices[assets].values
        result = coint_johansen(data, det_order, k_ar_diff)
        
        # 提取统计量
        trace_stats = result.lr1
        max_eig_stats = result.lr2
        
        # 临界值 (90%, 95%, 99%)
        trace_crit = result.cvt
        eig_crit = result.cvm
        
        return {
            'trace_statistics': trace_stats,
            'max_eigenvalue_statistics': max_eig_stats,
            'trace_critical_values': trace_crit,
            'eigenvalue_critical_values': eig_crit,
            'num_cointegrating_vectors': self._determine_cointegration_rank(
                trace_stats, trace_crit
            )
        }
    
    def _determine_cointegration_rank(self, trace_stats, crit_values):
        """确定协整向量个数"""
        rank = 0
        for i, stat in enumerate(trace_stats):
            if stat > crit_values[i, 1]:  # 95% 临界值
                rank += 1
        return rank
    
    def calculate_half_life(self, spread):
        """计算均值回复的半衰期"""
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        
        # 回归：spread_ret = alpha + beta * spread_lag + error
        X = sm.add_constant(spread_lag.dropna())
        y = spread_ret.dropna()
        
        model = OLS(y, X).fit()
        beta = model.params[1]
        
        half_life = -np.log(2) / beta if beta < 0 else np.inf
        
        return half_life

# 使用示例
import statsmodels.api as sm

# 生成模拟数据
np.random.seed(42)
n_obs = 1000

# 生成协整序列
x = np.cumsum(np.random.normal(0, 1, n_obs))
y = 1.5 * x + np.random.normal(0, 10, n_obs)

prices = pd.DataFrame({
    'Stock_A': y,
    'Stock_B': x
}, index=pd.date_range('2020-01-01', periods=n_obs))

# 协整分析
coint_analysis = CointegrationAnalysis(prices)
result = coint_analysis.engle_granger_test('Stock_A', 'Stock_B')

print("Engle-Granger协整检验结果：")
print(f"协整统计量: {result['coint_statistic']:.4f}")
print(f"p值: {result['p_value']:.4f}")
print(f"是否协整: {result['is_cointegrated']}")
print(f"对冲比率: {result['hedge_ratio']:.4f}")
```

### 其他协整检验方法

除了Engle-Granger方法，还有：

1. **Johansen检验**：适用于多变量系统
2. **Phillips-Ouliaris检验**：对Engle-Granger的改进
3. **Gregory-Hansen检验**：允许结构断点

```python
def phillips_ouliaris_test(y, x, trend='c'):
    """Phillips-Ouliaris协整检验"""
    from statsmodels.tsa.stattools import coint
    
    # 多种方法的p值
    methods = ['pegasus', 'aegis']
    results = {}
    
    for method in methods:
        try:
            coint_stat, p_value, _ = coint(y, x, method=method)
            results[method] = {
                'statistic': coint_stat,
                'p_value': p_value
            }
        except:
            continue
            
    return results
```

## 配对选择策略

### 1. 基本面相似性筛选

最直观的配对选择方法：

```python
class FundamentalSimilarity:
    """基于基本面的配对筛选"""
    
    def __init__(self, stock_data, fundamental_data):
        """
        参数：
        stock_data: Dict, 股票价格数据
        fundamental_data: DataFrame, 基本面数据（行业、市值、财务指标等）
        """
        self.stocks = stock_data
        self.fundamentals = fundamental_data
        
    def filter_by_industry(self, industry):
        """按行业筛选股票"""
        industry_stocks = self.fundamentals[
            self.fundamentals['industry'] == industry
        ].index.tolist()
        return industry_stocks
    
    def calculate_similarity_score(self, stock1, stock2):
        """计算基本面相似度得分"""
        fundamentals1 = self.fundamentals.loc[stock1]
        fundamentals2 = self.fundamentals.loc[stock2]
        
        # 相似度指标
        market_cap_diff = abs(
            np.log(fundamentals1['market_cap']) - 
            np.log(fundamentals2['market_cap'])
        )
        
        pe_diff = abs(fundamentals1['pe_ratio'] - fundamentals2['pe_ratio'])
        pb_diff = abs(fundamentals1['pb_ratio'] - fundamentals2['pb_ratio'])
        
        # 综合相似度得分（越小越相似）
        similarity_score = (
            0.3 * market_cap_diff +
            0.35 * pe_diff / 10 +
            0.35 * pb_diff / 2
        )
        
        return similarity_score
    
    def find_similar_pairs(self, min_similarity=0.5, top_n=50):
        """寻找相似股票对"""
        stocks = self.fundamentals.index.tolist()
        pairs = []
        
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                score = self.calculate_similarity_score(stocks[i], stocks[j])
                if score < (1 - min_similarity) * 10:  # 转换相似度阈值
                    pairs.append((stocks[i], stocks[j], score))
                    
        # 按相似度排序
        pairs.sort(key=lambda x: x[2])
        
        return pairs[:top_n]
```

### 2. 距离方法

基于价格历史距离的配对选择：

```python
class DistanceMethod:
    """距离方法选择配对"""
    
    def __init__(self, price_data, lookback=252):
        """
        参数：
        price_data: DataFrame, 价格数据
        lookback: int, 回看期
        """
        self.prices = price_data
        self.lookback = lookback
        
    def calculate_ssd(self, stock1, stock2, start_date, end_date):
        """
        计算平方和距离（Sum of Squared Differences）
        
        SSD = sum((price1_t - price2_t)^2)
        """
        p1 = self.prices.loc[start_date:end_date, stock1]
        p2 = self.prices.loc[start_date:end_date, stock2]
        
        # 标准化价格
        p1_norm = p1 / p1.iloc[0]
        p2_norm = p2 / p2.iloc[0]
        
        ssd = ((p1_norm - p2_norm) ** 2).sum()
        
        return ssd
    
    def calculate_correlation_distance(self, stock1, stock2):
        """基于相关系数的距离"""
        returns1 = self.prices[stock1].pct_change().dropna()
        returns2 = self.prices[stock2].pct_change().dropna()
        
        correlation = returns1.corr(returns2)
        distance = 1 - correlation  # 距离越大，相关性越低
        
        return distance
    
    def select_pairs_by_distance(self, num_pairs=30):
        """基于距离选择配对"""
        stocks = self.prices.columns.tolist()
        distances = []
        
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                # 计算多种距离度量
                ssd = self.calculate_ssd(
                    stocks[i], stocks[j],
                    self.prices.index[0],
                    self.prices.index[-self.lookback]
                )
                
                corr_dist = self.calculate_correlation_distance(
                    stocks[i], stocks[j]
                )
                
                # 综合距离得分
                combined_score = 0.5 * ssd / 100 + 0.5 * corr_dist
                
                distances.append({
                    'stock1': stocks[i],
                    'stock2': stocks[j],
                    'ssd': ssd,
                    'correlation_distance': corr_dist,
                    'combined_score': combined_score
                })
                
        # 按综合得分排序（越小越好）
        distances_df = pd.DataFrame(distances)
        distances_df = distances_df.sort_values('combined_score')
        
        return distances_df.head(num_pairs)
```

### 3. 协整得分方法

直接基于协整统计量的配对选择：

```python
class CointegrationScoreMethod:
    """基于协整得分的配对选择"""
    
    def __init__(self, price_data, significance=0.05):
        self.prices = price_data
        self.significance = significance
        self.coint_analysis = CointegrationAnalysis(price_data)
        
    def calculate_cointegration_score(self, stock1, stock2):
        """计算协整得分"""
        result = self.coint_analysis.engle_granger_test(stock1, stock2)
        
        if not result['is_cointegrated']:
            return None
            
        # 协整得分由多个指标组成
        p_value_score = -np.log(result['p_value'])  # p值越小越好
        half_life = self.coint_analysis.calculate_half_life(
            self._get_spread(stock1, stock2, result['hedge_ratio'])
        )
        
        # 半衰期在10-100天之间较优
        if half_life < 10:
            half_life_score = 0.5  # 太快，可能噪音
        elif half_life > 100:
            half_life_score = 0.5  # 太慢，资金占用久
        else:
            half_life_score = 1.0  # 合适
            
        # 综合得分
        score = 0.6 * p_value_score + 0.4 * half_life_score
        
        return {
            'stock1': stock1,
            'stock2': stock2,
            'cointegration_score': score,
            'p_value': result['p_value'],
            'hedge_ratio': result['hedge_ratio'],
            'half_life': half_life
        }
    
    def _get_spread(self, stock1, stock2, hedge_ratio):
        """计算价格差（残差）"""
        return self.prices[stock1] - hedge_ratio * self.prices[stock2]
    
    def select_pairs_by_cointegration(self, top_n=30):
        """基于协整得分选择配对"""
        stocks = self.prices.columns.tolist()
        coint_scores = []
        
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                score = self.calculate_cointegration_score(stocks[i], stocks[j])
                if score is not None:
                    coint_scores.append(score)
                    
        # 按协整得分排序
        scores_df = pd.DataFrame(coint_scores)
        scores_df = scores_df.sort_values('cointegration_score', ascending=False)
        
        return scores_df.head(top_n)
```

## 交易信号生成

### Z-Score 信号

最经典的配对交易信号：

```python
class PairTradingSignals:
    """配对交易信号生成"""
    
    def __init__(self, price_data, pairs, lookback=63):
        """
        参数：
        price_data: DataFrame, 价格数据
        pairs: List, 配对列表 [(stock1, stock2, hedge_ratio), ...]
        lookback: int, 计算均值的回看期
        """
        self.prices = price_data
        self.pairs = pairs
        self.lookback = lookback
        self.signals = {}
        
    def calculate_spread_zscore(self, stock1, stock2, hedge_ratio):
        """计算价差的Z-Score"""
        spread = self.prices[stock1] - hedge_ratio * self.prices[stock2]
        
        # 滚动均值和标准差
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()
        
        # Z-Score
        z_score = (spread - rolling_mean) / rolling_std
        
        return z_score, spread
    
    def generate_entry_signals(self, z_threshold=2.0, closing_z=0.5):
        """
        生成进出场信号
        
        z_threshold: 入场Z值阈值
        closing_z: 平仓Z值阈值
        """
        signals = {}
        
        for pair in self.pairs:
            stock1, stock2, hedge_ratio = pair[:3]
            
            z_score, spread = self.calculate_spread_zscore(
                stock1, stock2, hedge_ratio
            )
            
            # 信号矩阵
            signal = pd.DataFrame(index=z_score.index)
            signal['z_score'] = z_score
            signal['spread'] = spread
            
            # 多空信号
            # Z > threshold: 价差偏高，做空价差（卖stock1，买stock2）
            # Z < -threshold: 价差偏低，做多价差（买stock1，卖stock2）
            signal['position'] = 0
            signal.loc[z_score > z_threshold, 'position'] = -1  # 做空价差
            signal.loc[z_score < -z_threshold, 'position'] = 1   # 做多价差
            
            # 平仓信号：Z值回归到一定范围
            signal['exit_signal'] = (
                (signal['position'] != 0) & 
                (abs(z_score) < closing_z)
            )
            
            # 更新持仓
            for i in range(1, len(signal)):
                if signal.iloc[i]['exit_signal']:
                    signal.iloc[i:, signal.columns.get_loc('position')] = 0
                elif signal.iloc[i-1]['position'] != 0:
                    signal.iloc[i, signal.columns.get_loc('position')] = \
                        signal.iloc[i-1]['position']
                    
            signals[(stock1, stock2)] = signal
            
        self.signals = signals
        return signals
    
    def calculate_position_sizes(self, capital=1000000, risk_per_trade=0.02):
        """计算仓位大小"""
        positions = {}
        
        for (stock1, stock2), signal in self.signals.items():
            # 基于波动率调整仓位
            spread_vol = signal['spread'].rolling(63).std().iloc[-1]
            position_value = capital * risk_per_trade / spread_vol
            
            # 分配至每只股票
            hedge_ratio = self.pairs[
                [(s1, s2) for s1, s2 in self.pairs if (s1, s2) == (stock1, stock2)][0]
            ][2]
            
            positions[(stock1, stock2)] = {
                'stock1_weight': position_value,
                'stock2_weight': position_value * hedge_ratio,
                'total_capital': capital
            }
            
        return positions
```

### 卡尔曼滤波动态对冲

使用卡尔曼滤波动态调整对冲比率：

```python
class KalmanFilterPairTrading:
    """基于卡尔曼滤波的配对交易"""
    
    def __init__(self, observation_cov=1e-4, state_cov=1e-6):
        """
        参数：
        observation_cov: float, 观测噪声协方差
        state_cov: float, 状态噪声协方差
        """
        self.obs_cov = observation_cov
        self.state_cov = state_cov
        
    def kalman_filter(self, y, x):
        """
        卡尔曼滤波估计时变对冲比率
        
        y: array, 资产1价格
        x: array, 资产2价格
        """
        n = len(y)
        
        # 状态：对冲比率 beta
        # 观测：y_t = alpha + beta_t * x_t + error
        
        beta_est = np.zeros(n)
        beta_est[0] = np.cov(y, x)[0, 1] / np.var(x)
        
        P = np.ones(n) * 1.0  # 估计误差协方差
        
        for t in range(1, n):
            # 预测步骤
            beta_pred = beta_est[t-1]
            P_pred = P[t-1] + self.state_cov
            
            # 更新步骤
            K = P_pred * x[t] / (x[t] * P_pred * x[t] + self.obs_cov)  # 卡尔曼增益
            beta_est[t] = beta_pred + K * (y[t] - beta_pred * x[t])
            P[t] = (1 - K * x[t]) * P_pred
            
        return beta_est
    
    def dynamic_spread(self, y, x):
        """计算动态对冲比率下的价差"""
        beta_dynamic = self.kalman_filter(y, x)
        spread = y - beta_dynamic * x
        return spread, beta_dynamic
    
    def backtest_kalman_pair(self, price1, price2, z_threshold=2.0):
        """回测卡尔曼滤波配对策略"""
        spread, beta_dynamic = self.dynamic_spread(price1.values, price2.values)
        
        # 计算Z-Score
        spread_series = pd.Series(spread, index=price1.index)
        z_score = (spread_series - spread_series.rolling(63).mean()) / \
                  spread_series.rolling(63).std()
        
        # 生成信号
        signals = pd.DataFrame(index=price1.index)
        signals['z_score'] = z_score
        signals['beta'] = beta_dynamic
        signals['spread'] = spread
        
        signals['position'] = 0
        signals.loc[z_score > z_threshold, 'position'] = -1
        signals.loc[z_score < -z_threshold, 'position'] = 1
        signals['position'] = signals['position'].replace(0, np.nan).fillna(method='ffill')
        
        # 计算收益
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        strategy_returns = (
            signals['position'].shift(1) * 
            (returns1 - signals['beta'].shift(1) * returns2)
        )
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        return strategy_returns, cumulative_returns, signals
```

## 风险管理与绩效评估

### 关键风险指标

```python
class PairTradingRiskManagement:
    """配对交易风险管理"""
    
    def __init__(self, pair_returns, benchmark_returns=None):
        self.returns = pair_returns
        self.benchmark = benchmark_returns
        
    def calculate_max_drawdown(self):
        """计算最大回撤"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        return max_dd, drawdown
    
    def calculate_pair_stability(self, window=63):
        """计算配对稳定性"""
        # 滚动相关性
        rolling_corr = self.returns.rolling(window).corr()
        
        # 滚动协整检验（简化：使用残差平稳性）
        is_stable = []
        for i in range(window, len(self.returns)):
            window_data = self.returns[i-window:i]
            # 简化：用自相关性检验平稳性
            acf = acf(window_data, nlags=10)
            is_stable.append(acf[1] < 0)  # 一阶自相关为负说明均值回复
            
        stability_ratio = np.mean(is_stable)
        return stability_ratio, rolling_corr
    
    def calculate_terror_ratio(self, signals, threshold=2.0):
        """计算偏离阈值的时间比例"""
        if 'z_score' not in signals.columns:
            raise ValueError("Signals must contain 'z_score' column")
            
        time_beyond_threshold = (abs(signals['z_score']) > threshold).mean()
        return time_beyond_threshold
    
    def stop_loss_check(self, cum_returns, max_loss=0.05):
        """止损检查"""
        drawdown = (cum_returns - cum_returns.expanding().max()) / \
                   cum_returns.expanding().max()
        
        stop_loss_triggered = (drawdown < -max_loss).any()
        return stop_loss_triggered, drawdown

# 使用示例
pair_returns = pd.Series(np.random.normal(0.0005, 0.01, 1000))
risk_mgmt = PairTradingRiskManagement(pair_returns)

max_dd, dd_series = risk_mgmt.calculate_max_drawdown()
print(f"最大回撤: {max_dd:.2%}")
```

### 绩效归因

```python
def performance_attribution(strategy_returns, factor_returns):
    """绩效归因：分解alpha和beta"""
    from sklearn.linear_model import LinearRegression
    
    # 多元回归：策略收益 ~ 因子收益
    X = factor_returns.values
    y = strategy_returns.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 拟合值（因子解释部分）
    fitted_values = model.predict(X)
    residuals = y - fitted_values  # 残差（alpha）
    
    attribution = {
        'alpha': residuals.mean() * 252,  # 年化alpha
        'beta': model.coef_,
        'r_squared': model.score(X, y),
        'residual_vol': residuals.std() * np.sqrt(252)
    }
    
    return attribution
```

## 实战案例：A股配对交易

完整的A股配对交易策略示例：

```python
# 数据获取（使用akshare）
import akshare as ak

def get_a_share_data(stock_list, start_date, end_date):
    """获取A股数据"""
    prices = pd.DataFrame()
    
    for stock in stock_list:
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            prices[stock] = df['收盘']
        except:
            print(f"获取 {stock} 数据失败")
            continue
            
    return prices

# 选择配对
stock_list = ['600036', '601398', '600016', '601988', '601318']  # 示例股票
prices = get_a_share_data(stock_list, '20200101', '20251231')

# 协整检验
coint_analysis = CointegrationAnalysis(prices)
pairs = coint_analysis.select_pairs_by_cointegration(top_n=5)

# 回测
for _, row in pairs.iterrows():
    stock1, stock2 = row['stock1'], row['stock2']
    hedge_ratio = row['hedge_ratio']
    
    signals = PairTradingSignals(
        prices, 
        [(stock1, stock2, hedge_ratio)]
    )
    signals.generate_entry_signals(z_threshold=2.0)
    
    # 计算收益
    # ... (省略具体回测代码)
    
    print(f"配对 {stock1}-{stock2} 回测完成")
```

## 结论

配对交易是一个理论扎实、实践可行的量化策略。成功的关键在于：

1. **严谨的配对选择**：协整分析是基础
2. **合理的信号设计**：Z-Score、卡尔曼滤波等
3. **严格的风险管理**：止损、仓位控制
4. **低交易成本**：选择流动性好的标的

随着机器学习的发展，现代配对交易正在融合：
- **高维协整分析**：处理数百只股票的配对选择
- **深度学习信号**：捕捉非线性均值回复模式
- **高频执行**：利用微观结构获利

配对交易永远不会过时，因为均值回归是金融市场的本质特征之一。

---

**参考文献**：
1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis."
2. Gatev, E., et al. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
3. Elliott, R. J., et al. (2005). "Pairs Trading." Quantitative Finance.
4. Liu, J. (2016). "Economic Explanation of the Pairs Trading Strategy."
