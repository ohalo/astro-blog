---
title: "配对交易与协整分析：统计套利的理论与实践"
publishDate: '2026-06-16'
description: "配对交易与协整分析：统计套利的理论与实践 - 量化策略专家的博客"
tags:
  - 量化交易
  - 统计套利
  - 配对交易
language: Chinese
---

# 配对交易与协整分析：统计套利的理论与实践

## 引言：从华尔街到散户的统计套利

1980年代，摩根士丹利的量化团队发现了一个有趣的现象：可口可乐（KO）和百事可乐（PEP）的股价走势高度相关，但偶尔会出现偏离。当两者价差扩大到历史极值时，买入低估的、卖出高估的，等待价差回归，就能获得几乎无风险的收益。

这就是**配对交易（Pairs Trading）**的雏形。三十多年后的今天，配对交易依然是统计套利的核心策略之一，被对冲基金、做市商和量化团队广泛使用。

本文将深入探讨：
- 配对交易的理论基础：协整与均值回归
- 如何科学地选择配对股票
- 基于Python的完整策略实现
- 实战中的陷阱与应对策略

## 一、配对交易的理论基础

### 1.1 为什么配对交易有效？

配对交易的核心理念是**均值回归（Mean Reversion）**。在有效市场中，相关股票的价差应该围绕某个均衡值波动。当价差偏离均衡时，市场力量会推动其回归。

**数学表达：**

假设两只股票的价格序列为 $P_t^A$ 和 $P_t^B$，我们构建一个价差序列：

$$
S_t = P_t^A - \beta \cdot P_t^B
$$

其中 $\beta$ 是对冲比率（Hedge Ratio）。

如果 $S_t$ 是平稳序列（Stationary），那么当 $S_t$ 偏离其均值 $\mu$ 时，我们会预期它在未来回归到 $\mu$。

### 1.2 协整：平稳性的统计学基础

**问题：** 如何判断两个价格序列的线性组合是平稳的？

**答案：** 协整检验（Cointegration Test）。

**定义：** 如果两个非平稳时间序列 $X_t$ 和 $Y_t$ 的线性组合是平稳的，那么我们说它们是协整的。

**Engle-Granger两步法：**

1. **第一步：** 用OLS回归估计长期均衡关系
   $$
   Y_t = \alpha + \beta X_t + \epsilon_t
   $$

2. **第二步：** 对残差 $\epsilon_t$ 进行平稳性检验（如ADF检验）

如果残差是平稳的，则 $X_t$ 和 $Y_t$ 协整。

### 1.3 协整 vs 相关性

**重要区别：**

- **相关性**衡量的是收益率的同向变动程度
- **协整**衡量的是价格水平的长期均衡关系

两个股票可以高度相关但不协整（比如两只科技股同涨同跌，但价差持续扩大）；也可以协整但相关性不高（比如可口可乐和百事可乐，短期走势可能不同，但长期价差稳定）。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf

# 示例：检验可口可乐和百事可乐的协整关系
def check_cointegration_example():
    """
    示例：检验KO和PEP的协整关系
    """
    # 下载数据
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 提取价格序列
    ko_prices = data['KO'].dropna()
    pep_prices = data['PEP'].dropna()
    
    # 1. 计算相关性
    correlation = ko_prices.corr(pep_prices)
    print(f"KO和PEP的价格相关性: {correlation:.4f}")
    
    # 2. 协整检验（Engle-Granger）
    coint_stat, p_value, _ = coint(ko_prices, pep_prices)
    print(f"\n协整检验统计量: {coint_stat:.4f}")
    print(f"P值: {p_value:.4f}")
    print(f"结论: {'协整（配对交易可行）' if p_value < 0.05 else '不协整（配对交易不可行）'}")
    
    # 3. 可视化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 子图1：价格走势
    axes[0].plot(ko_prices.index, ko_prices.values, label='KO (可口可乐)', linewidth=2)
    axes[0].plot(pep_prices.index, pep_prices.values, label='PEP (百事可乐)', linewidth=2)
    axes[0].set_title('KO vs PEP 价格走势', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：价差
    spread = ko_prices - pep_prices
    axes[1].plot(spread.index, spread.values, linewidth=2, color='purple')
    axes[1].axhline(y=spread.mean(), color='red', linestyle='--', linewidth=2, label='均值')
    axes[1].fill_between(spread.index, 
                         spread.mean() - 2*spread.std(),
                         spread.mean() + 2*spread.std(),
                         alpha=0.2, color='purple', label='±2倍标准差')
    axes[1].set_title('价格价差 (KO - PEP)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：残差（对冲比率优化后）
    # 用OLS估计对冲比率
    import statsmodels.api as sm
    X = sm.add_constant(pep_prices)
    model = sm.OLS(ko_prices, X).fit()
    beta = model.params[1]
    
    residuals = ko_prices - beta * pep_prices
    axes[2].plot(residuals.index, residuals.values, linewidth=2, color='green')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[2].set_title(f'残差序列 (对冲比率 β={beta:.4f})', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # ADF检验残差
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(residuals)
    axes[2].set_xlabel(f'ADF检验 P值: {adf_pvalue:.4f} ' + 
                        f"({'平稳' if adf_pvalue < 0.05 else '非平稳'})",
                        fontsize=11)
    
    plt.tight_layout()
    plt.savefig('ko_pep_cointegration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, residuals

# 运行示例
if __name__ == "__main__":
    model, residuals = check_cointegration_example()
```

## 二、配对选择的科学方法

选择合适的配对是策略成功的关键。以下是三种主流的配对选择方法：

### 2.1 距离法（Distance Approach）

**原理：** 计算所有股票对的价格距离（如欧氏距离、马氏距离），选择距离最小的对。

**优点：** 简单直观，计算速度快  
**缺点：** 不考虑协整关系，可能产生假信号

```python
def distance_method(stock_data, top_n=50):
    """
    距离法选择配对
    
    参数:
    stock_data: DataFrame, 股票价格数据 (date x stocks)
    top_n: int, 返回前N个配对
    
    返回:
    pairs: list, [(stock1, stock2, distance), ...]
    """
    import itertools
    from scipy.spatial.distance import pdist, squareform
    
    stocks = stock_data.columns
    n = len(stocks)
    
    # 标准化价格（消除量纲影响）
    normalized_data = (stock_data - stock_data.mean()) / stock_data.std()
    
    # 计算所有股票对的距离
    distances = pdist(normalized_data.T, metric='euclidean')  # 转置，使每列是一只股票
    distance_matrix = squareform(distances)
    
    # 找出距离最小的配对
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((stocks[i], stocks[j], distance_matrix[i, j]))
    
    # 按距离排序
    pairs.sort(key=lambda x: x[2])
    
    return pairs[:top_n]

# 使用示例
# pairs = distance_method(stock_prices, top_n=20)
# for stock1, stock2, dist in pairs[:5]:
#     print(f"{stock1} - {stock2}: 距离 = {dist:.4f}")
```

### 2.2 协整法（Cointegration Approach）

**原理：** 对所有股票对进行协整检验，选择P值最小的配对。

**优点：** 理论基础扎实，配对质量高  
**缺点：** 计算量大，需要多重检验校正

```python
def cointegration_method(stock_data, significance_level=0.05, top_n=50):
    """
    协整法选择配对
    
    参数:
    stock_data: DataFrame, 股票价格数据 (date x stocks)
    significance_level: float, 显著性水平
    top_n: int, 返回前N个配对
    
    返回:
    pairs: list, [(stock1, stock2, p_value, beta), ...]
    """
    import itertools
    from statsmodels.tsa.stattools import coint
    import statsmodels.api as sm
    
    stocks = stock_data.columns
    pairs = []
    
    print(f"开始协整检验：共 {len(stocks)} 只股票，需要检验 {len(stocks)*(len(stocks)-1)//2} 对...")
    
    for i, stock1 in enumerate(stocks):
        for stock2 in stocks[i+1:]:
            # 提取价格序列
            price1 = stock_data[stock1].dropna()
            price2 = stock_data[stock2].dropna()
            
            # 对齐数据
            aligned_data = pd.concat([price1, price2], axis=1).dropna()
            if len(aligned_data) < 252:  # 至少需要1年数据
                continue
            
            # 协整检验
            try:
                _, p_value, _ = coint(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                
                if p_value < significance_level:
                    # 估计对冲比率
                    X = sm.add_constant(aligned_data.iloc[:, 1])
                    model = sm.OLS(aligned_data.iloc[:, 0], X).fit()
                    beta = model.params[1]
                    
                    pairs.append((stock1, stock2, p_value, beta))
            except:
                continue
    
    # 按P值排序（越小越好）
    pairs.sort(key=lambda x: x[2])
    
    print(f"找到 {len(pairs)} 个协整配对")
    
    return pairs[:top_n]

# 并行化加速（适用于大规模股票池）
from multiprocessing import Pool
import numpy as np

def parallel_cointegration_method(stock_data, significance_level=0.05, top_n=50, n_jobs=4):
    """
    并行化协整检验
    """
    stocks = stock_data.columns
    stock_pairs = list(itertools.combinations(stocks, 2))
    
    def test_pair(pair):
        stock1, stock2 = pair
        price1 = stock_data[stock1].dropna()
        price2 = stock_data[stock2].dropna()
        
        aligned_data = pd.concat([price1, price2], axis=1).dropna()
        if len(aligned_data) < 252:
            return None
        
        try:
            _, p_value, _ = coint(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
            if p_value < significance_level:
                X = sm.add_constant(aligned_data.iloc[:, 1])
                model = sm.OLS(aligned_data.iloc[:, 0], X).fit()
                beta = model.params[1]
                return (stock1, stock2, p_value, beta)
        except:
            return None
        
        return None
    
    # 并行计算
    with Pool(n_jobs) as pool:
        results = pool.map(test_pair, stock_pairs)
    
    # 过滤掉None结果
    pairs = [r for r in results if r is not None]
    pairs.sort(key=lambda x: x[2])
    
    return pairs[:top_n]
```

### 2.3 相关性 + 协整混合法

**原理：** 先用水相关性筛选候选配对，再进行协整检验。

**优点：** 平衡计算效率和配对质量  
**缺点：** 需要手动调整相关性阈值

```python
def hybrid_method(stock_data, correlation_threshold=0.7, significance_level=0.05, top_n=50):
    """
    相关性 + 协整混合法
    
    参数:
    stock_data: DataFrame, 股票价格数据
    correlation_threshold: float, 相关性阈值
    significance_level: float, 协整检验显著性水平
    top_n: int, 返回前N个配对
    
    返回:
    pairs: list, 筛选后的配对列表
    """
    # 第一步：计算收益率相关性
    returns_data = stock_data.pct_change().dropna()
    correlation_matrix = returns_data.corr()
    
    # 第二步：筛选高相关性配对
    candidate_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if corr > correlation_threshold:
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                candidate_pairs.append((stock1, stock2, corr))
    
    print(f"高相关性配对数量: {len(candidate_pairs)}")
    
    # 第三步：对候选配对进行协整检验
    pairs = []
    for stock1, stock2, corr in candidate_pairs:
        price1 = stock_data[stock1].dropna()
        price2 = stock_data[stock2].dropna()
        
        aligned_data = pd.concat([price1, price2], axis=1).dropna()
        
        try:
            _, p_value, _ = coint(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
            if p_value < significance_level:
                X = sm.add_constant(aligned_data.iloc[:, 1])
                model = sm.OLS(aligned_data.iloc[:, 0], X).fit()
                beta = model.params[1]
                pairs.append((stock1, stock2, p_value, beta, corr))
        except:
            continue
    
    # 按协整P值排序
    pairs.sort(key=lambda x: x[2])
    
    print(f"最终协整配对数量: {len(pairs)}")
    
    return pairs[:top_n]

# 完整流程示例
if __name__ == "__main__":
    # 假设我们有股票价格数据
    # stock_prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
    
    # 使用混合法选择配对
    # pairs = hybrid_method(stock_prices, correlation_threshold=0.6, top_n=20)
    
    # 打印前5个配对
    # for stock1, stock2, p_value, beta, corr in pairs[:5]:
    #     print(f"{stock1} - {stock2}: 相关性={corr:.4f}, 协整P值={p_value:.4f}, β={beta:.4f}")
```

## 三、交易信号与仓位管理

选好配对后，如何构建交易信号？以下是三种主流方法：

### 3.1 Z-Score信号法

**原理：** 计算价差的Z-Score，当Z-Score超过阈值时触发交易。

```python
class ZScoreSignalGenerator:
    """
    Z-Score交易信号生成器
    """
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, lookback=252):
        """
        参数:
        entry_threshold: float, 入场阈值（Z-Score绝对值）
        exit_threshold: float, 出场阈值（Z-Score绝对值）
        lookback: int, 滚动窗口长度
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback
    
    def calculate_z_score(self, spread):
        """
        计算滚动Z-Score
        
        参数:
        spread: Series, 价差序列
        
        返回:
        z_score: Series, Z-Score序列
        """
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()
        
        z_score = (spread - rolling_mean) / rolling_std
        
        return z_score
    
    def generate_signals(self, spread):
        """
        生成交易信号
        
        返回:
        signals: DataFrame, 包含以下列：
            - z_score: Z-Score值
            - position: 仓位 (-1: 做空价差, 0: 空仓, 1: 做多价差)
            - long_stock: 做多的股票
            - short_stock: 做空的股票
        """
        z_score = self.calculate_z_score(spread)
        
        signals = pd.DataFrame(index=spread.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        
        # 生成信号
        for i in range(1, len(signals)):
            if signals['z_score'].iloc[i-1] == 0:  # 第一个有效值
                continue
            
            # 入场信号
            if signals['position'].iloc[i-1] == 0:  # 当前空仓
                if signals['z_score'].iloc[i] < -self.entry_threshold:
                    # Z-Score < -2，做多价差（买入stock1，卖出stock2）
                    signals.loc[signals.index[i], 'position'] = 1
                elif signals['z_score'].iloc[i] > self.entry_threshold:
                    # Z-Score > 2，做空价差（卖出stock1，买入stock2）
                    signals.loc[signals.index[i], 'position'] = -1
            
            # 出场信号
            elif signals['position'].iloc[i-1] != 0:  # 当前有仓位
                if abs(signals['z_score'].iloc[i]) < self.exit_threshold:
                    # Z-Score回归到阈值内，平仓
                    signals.loc[signals.index[i], 'position'] = 0
                else:
                    # 保持仓位
                    signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]
        
        return signals

# 使用示例
# generator = ZScoreSignalGenerator(entry_threshold=2.0, exit_threshold=0.5)
# signals = generator.generate_signals(spread)
```

### 3.2 布林带信号法

**原理：** 用布林带（Bollinger Bands）代替Z-Score，更直观地识别极值。

```python
class BollingerBandSignalGenerator:
    """
    布林带交易信号生成器
    """
    def __init__(self, entry_std=2.0, exit_std=0.5, lookback=252):
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.lookback = lookback
    
    def generate_signals(self, spread):
        """
        生成布林带信号
        """
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()
        
        upper_band = rolling_mean + self.entry_std * rolling_std
        lower_band = rolling_mean - self.entry_std * rolling_std
        middle_band = rolling_mean
        
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        signals['middle_band'] = middle_band
        signals['position'] = 0
        
        # 生成信号（逻辑与Z-Score类似）
        for i in range(1, len(signals)):
            if signals['position'].iloc[i-1] == 0:  # 空仓
                if spread.iloc[i] < lower_band.iloc[i]:
                    signals.loc[signals.index[i], 'position'] = 1  # 做多
                elif spread.iloc[i] > upper_band.iloc[i]:
                    signals.loc[signals.index[i], 'position'] = -1  # 做空
            elif signals['position'].iloc[i-1] != 0:  # 有仓位
                if abs(spread.iloc[i] - middle_band.iloc[i]) < self.exit_std * rolling_std.iloc[i]:
                    signals.loc[signals.index[i], 'position'] = 0  # 平仓
                else:
                    signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]
        
        return signals
```

### 3.3 卡尔曼滤波信号法

**原理：** 用卡尔曼滤波（Kalman Filter）动态估计对冲比率和价差，适应市场结构变化。

```python
from pykalman import KalmanFilter

class KalmanFilterSignalGenerator:
    """
    卡尔曼滤波交易信号生成器
    """
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def dynamic_hedge_ratio(self, price1, price2):
        """
        用卡尔曼滤波动态估计对冲比率
        
        返回:
        state_means: array, 动态对冲比率序列
        """
        # 状态转移矩阵（假设对冲比率随机游走）
        transition_matrix = [[1, 0], [0, 1]]
        
        # 观测矩阵（观测到的是price1）
        observation_matrix = [[1, price2.iloc[0]]]
        
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=[0, 1],  # [alpha, beta]
            initial_state_covariance=np.eye(2) * 0.01,
            observation_covariance=1.0,
            transition_covariance=np.eye(2) * 0.01
        )
        
        # 逐点更新
        state_means = []
        current_state = [0, 1]
        
        for t in range(len(price1)):
            # 更新观测矩阵
            kf.observation_matrices = [[1, price2.iloc[t]]]
            
            # 卡尔曼滤波更新
            current_state = kf.filter(price1.iloc[t:t+1])[0][-1]
            state_means.append(current_state)
        
        return np.array(state_means)
    
    def generate_signals(self, price1, price2):
        """
        生成动态对冲比率信号
        """
        # 动态估计对冲比率
        state_means = self.dynamic_hedge_ratio(price1, price2)
        beta_dynamic = state_means[:, 1]
        
        # 计算动态价差
        spread_dynamic = price1.values - beta_dynamic * price2.values
        spread_series = pd.Series(spread_dynamic, index=price1.index)
        
        # 计算Z-Score
        z_score = (spread_series - spread_series.rolling(252).mean()) / spread_series.rolling(252).std()
        
        # 生成信号（与Z-Score方法相同）
        signals = pd.DataFrame(index=price1.index)
        signals['spread'] = spread_series
        signals['beta_dynamic'] = beta_dynamic
        signals['z_score'] = z_score
        signals['position'] = 0
        
        for i in range(1, len(signals)):
            if signals['position'].iloc[i-1] == 0:
                if signals['z_score'].iloc[i] < -self.entry_threshold:
                    signals.loc[signals.index[i], 'position'] = 1
                elif signals['z_score'].iloc[i] > self.entry_threshold:
                    signals.loc[signals.index[i], 'position'] = -1
            elif signals['position'].iloc[i-1] != 0:
                if abs(signals['z_score'].iloc[i]) < self.exit_threshold:
                    signals.loc[signals.index[i], 'position'] = 0
                else:
                    signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]
        
        return signals
```

## 四、完整的配对交易策略实现

让我们将上述组件整合，构建一个完整的配对交易策略回测框架。

```python
class PairsTradingStrategy:
    """
    完整的配对交易策略
    """
    def __init__(self, stock1, stock2, data, signal_generator, 
                 initial_capital=1000000, transaction_cost=0.001):
        """
        参数:
        stock1, stock2: str, 配对股票代码
        data: DataFrame, 价格数据
        signal_generator: object, 信号生成器
        initial_capital: float, 初始资金
        transaction_cost: float, 交易成本（单边）
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.data = data
        self.signal_generator = signal_generator
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # 计算结果
        self.signals = None
        self.returns = None
        self.portfolio_value = None
    
    def backtest(self):
        """
        回测策略
        """
        # 提取价格数据
        price1 = self.data[self.stock1].dropna()
        price2 = self.data[self.stock2].dropna()
        
        # 对齐数据
        aligned_prices = pd.concat([price1, price2], axis=1).dropna()
        price1_aligned = aligned_prices.iloc[:, 0]
        price2_aligned = aligned_prices.iloc[:, 1]
        
        # 计算价差
        beta = self._estimate_hedge_ratio(price1_aligned, price2_aligned)
        spread = price1_aligned - beta * price2_aligned
        
        # 生成交易信号
        self.signals = self.signal_generator.generate_signals(spread)
        
        # 计算策略收益
        self._calculate_returns(price1_aligned, price2_aligned, beta)
        
        # 计算绩效指标
        performance = self._calculate_performance()
        
        return self.signals, performance
    
    def _estimate_hedge_ratio(self, price1, price2):
        """
        估计对冲比率
        """
        import statsmodels.api as sm
        X = sm.add_constant(price2)
        model = sm.OLS(price1, X).fit()
        beta = model.params[1]
        return beta
    
    def _calculate_returns(self, price1, price2, beta):
        """
        计算策略收益率
        """
        # 初始化
        n = len(self.signals)
        strategy_returns = np.zeros(n)
        portfolio_values = np.ones(n) * self.initial_capital
        positions = np.zeros(n)  # 持仓数量
        cash = self.initial_capital
        
        for i in range(1, n):
            if self.signals['position'].iloc[i] != self.signals['position'].iloc[i-1]:
                # 仓位变化，执行交易
                old_position = self.signals['position'].iloc[i-1]
                new_position = self.signals['position'].iloc[i]
                
                # 平仓旧仓位
                if old_position == 1:
                    # 平多价差：卖出stock1，买入stock2
                    cash += positions[i-1] * price1.iloc[i]  # 卖出stock1
                    cash -= positions[i-1] * beta * price2.iloc[i] * (1 + self.transaction_cost)  # 买入stock2（含成本）
                elif old_position == -1:
                    # 平空价差：买入stock1，卖出stock2
                    cash -= positions[i-1] * price1.iloc[i] * (1 + self.transaction_cost)  # 买入stock1（含成本）
                    cash += positions[i-1] * beta * price2.iloc[i]  # 卖出stock2
                
                # 建立新仓位
                if new_position == 1:
                    # 做多价差：买入stock1，卖出stock2
                    position_size = cash / (price1.iloc[i] + beta * price2.iloc[i])
                    positions[i] = position_size
                    cash -= position_size * price1.iloc[i] * (1 + self.transaction_cost)  # 买入stock1
                    cash += position_size * beta * price2.iloc[i] * (1 - self.transaction_cost)  # 卖出stock2（收入）
                elif new_position == -1:
                    # 做空价差：卖出stock1，买入stock2
                    position_size = cash / (price1.iloc[i] + beta * price2.iloc[i])
                    positions[i] = -position_size
                    cash += position_size * price1.iloc[i] * (1 - self.transaction_cost)  # 卖出stock1（收入）
                    cash -= position_size * beta * price2.iloc[i] * (1 + self.transaction_cost)  # 买入stock2
                else:
                    positions[i] = 0
            else:
                # 仓位不变
                positions[i] = positions[i-1]
            
            # 计算当日组合价值
            if positions[i] > 0:
                portfolio_values[i] = positions[i] * price1.iloc[i] - positions[i] * beta * price2.iloc[i]
            elif positions[i] < 0:
                portfolio_values[i] = -positions[i] * beta * price2.iloc[i] - (-positions[i]) * price1.iloc[i]
            else:
                portfolio_values[i] = cash
            
            # 计算收益率
            strategy_returns[i] = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        
        self.returns = pd.Series(strategy_returns, index=self.signals.index)
        self.portfolio_value = pd.Series(portfolio_values, index=self.signals.index)
    
    def _calculate_performance(self):
        """
        计算策略绩效指标
        """
        # 总收益
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益
        trading_days = len(self.returns)
        years = trading_days / 252
        annualized_return = (1 + total_return/100) ** (1/years) - 1
        
        # 夏普比率
        risk_free_rate = 0.02 / 252  # 假设无风险利率2%
        excess_returns = self.returns - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # 最大回撤
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        winning_trades = (self.returns > 0).sum()
        total_trades = (self.returns != 0).sum()
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        return performance
    
    def plot_results(self):
        """
        可视化回测结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        
        # 子图1：价差与信号
        axes[0].plot(self.signals.index, self.signals['spread'], 
                    linewidth=2, color='blue', label='价差')
        axes[0].axhline(y=self.signals['spread'].mean(), 
                       color='red', linestyle='--', linewidth=2, label='均值')
        
        # 标记交易信号
        entry_long = self.signals[self.signals['position'] == 1]
        entry_short = self.signals[self.signals['position'] == -1]
        exit_signals = self.signals[(self.signals['position'] == 0) & 
                                    (self.signals['position'].shift(1) != 0)]
        
        axes[0].scatter(entry_long.index, entry_long['spread'], 
                       color='green', s=50, marker='^', label='做多入场', zorder=5)
        axes[0].scatter(entry_short.index, entry_short['spread'], 
                       color='red', s=50, marker='v', label='做空入场', zorder=5)
        axes[0].scatter(exit_signals.index, exit_signals['spread'], 
                       color='gray', s=50, marker='o', label='平仓', zorder=5)
        
        axes[0].set_title(f'配对交易信号: {self.stock1} - {self.stock2}', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 子图2：组合价值曲线
        axes[1].plot(self.portfolio_value.index, self.portfolio_value.values, 
                    linewidth=2, color='purple', label='组合价值')
        axes[1].axhline(y=self.initial_capital, color='gray', 
                       linestyle='--', linewidth=2, label='初始资金')
        axes[1].set_title('组合价值曲线', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('价值 ($)', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 子图3：累积收益
        cumulative_returns = (1 + self.returns).cumprod()
        axes[2].plot(cumulative_returns.index, cumulative_returns.values, 
                    linewidth=2, color='green', label='策略收益')
        
        # 基准收益（等权重买入并持有stock1和stock2）
        benchmark_returns = 0.5 * self.data[self.stock1].pct_change() + \
                           0.5 * self.data[self.stock2].pct_change()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        axes[2].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    linewidth=2, color='gray', alpha=0.7, label='基准（等权重）')
        
        axes[2].set_title('累积收益对比', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('日期', fontsize=12)
        axes[2].set_ylabel('累积收益', fontsize=12)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'pairs_trading_results_{self.stock1}_{self.stock2}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# 完整使用示例
if __name__ == "__main__":
    # 1. 下载数据
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 2. 初始化信号生成器
    signal_gen = ZScoreSignalGenerator(entry_threshold=2.0, exit_threshold=0.5)
    
    # 3. 初始化策略
    strategy = PairsTradingStrategy('KO', 'PEP', data, signal_gen, 
                                   initial_capital=1000000, transaction_cost=0.001)
    
    # 4. 回测
    signals, performance = strategy.backtest()
    
    # 5. 打印绩效指标
    print("=" * 60)
    print("配对交易策略回测结果")
    print("=" * 60)
    print(f"股票配对: {strategy.stock1} - {strategy.stock2}")
    print(f"总收益率: {performance['total_return']:.2f}%")
    print(f"年化收益率: {performance['annualized_return']:.2f}%")
    print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
    print(f"最大回撤: {performance['max_drawdown']:.2f}%")
    print(f"胜率: {performance['win_rate']:.2f}%")
    print(f"总交易次数: {performance['total_trades']}")
    print("=" * 60)
    
    # 6. 可视化
    strategy.plot_results()
```

## 五、实战中的陷阱与应对

配对交易看似简单，实则有许多陷阱。以下是常见的五个问题及解决方案：

### 5.1 陷阱1：结构性断裂

**问题：** 配对的协整关系可能因为公司并购、行业变革等原因突然断裂。

**案例：** 2020年新冠疫情爆发，许多曾经的协整配对（如航空股）突然失效，因为行业基本面发生了永久性变化。

**解决方案：**
- 定期重新检验协整关系（建议每月或每季度）
- 设置"关系断裂"止损：如果连续N天价差不回归，强制平仓
- 分散投资多个配对，降低单一配对失效的风险

```python
def structural_break_stop_loss(spread, window=63, threshold=0.05):
    """
    结构性断裂止损
    
    参数:
    spread: Series, 价差序列
    window: int, 滚动窗口
    threshold: float, ADF检验P值阈值
    
    返回:
    break_signal: bool, 是否检测到结构性断裂
    """
    from statsmodels.tsa.stattools import adfuller
    
    # 滚动检验价差平稳性
    for i in range(window, len(spread)):
        window_data = spread[i-window:i]
        adf_stat, p_value, _, _, _, _ = adfuller(window_data)
        
        if p_value > threshold:  # P值过大，说明不平稳
            return True
    
    return False
```

### 5.2 陷阱2：交易成本侵蚀收益

**问题：** 配对交易通常交易频繁，交易成本会显著侵蚀收益。

**解决方案：**
- 优化入场阈值：提高入场Z-Score阈值（如从2.0提高到2.5），减少交易次数
- 考虑交易成本调整信号：只有预期收益覆盖交易成本时才入场
- 使用低成本的交易渠道（如期权、期货）

```python
def transaction_cost_adjusted_signal(spread, transaction_cost, expected_mean_reversion=0.5):
    """
    考虑交易成本的信号调整
    
    参数:
    spread: Series, 价差序列
    transaction_cost: float, 单边交易成本
    expected_mean_reversion: float, 预期均值回归幅度
    
    返回:
    adjusted_signal: Series, 调整后的信号
    """
    # 计算预期收益
    z_score = (spread - spread.rolling(252).mean()) / spread.rolling(252).std()
    expected_profit = z_score * expected_mean_reversion  # 简化模型
    
    # 只有当预期收益覆盖交易成本时才入场
    adjusted_signal = z_score.copy()
    adjusted_signal[expected_profit < 2 * transaction_cost] = 0  # 需要覆盖双边成本
    
    return adjusted_signal
```

### 5.3 陷阱3：流动性风险

**问题：** 某些股票可能流动性不足，导致滑点（Slippage）巨大。

**解决方案：**
- 选择高流动性的股票（如大盘股、ETF）
- 限制单笔交易规模（如不超过日均成交量的1%）
- 使用限价单而非市价单

### 5.4 陷阱4：模型过拟合

**问题：** 通过历史数据优化得到的参数（如Z-Score阈值）可能过拟合。

**解决方案：**
- 使用样本外测试（Out-of-Sample Testing）
- 简化模型：不要过度优化参数
- 使用交叉验证（Rolling Window Cross-Validation）

```python
def rolling_window_validation(data, strategy_class, signal_generator, window_size=252, step=63):
    """
    滚动窗口交叉验证
    
    参数:
    data: DataFrame, 价格数据
    strategy_class: class, 策略类
    signal_generator: object, 信号生成器
    window_size: int, 训练窗口长度
    step: int, 滚动步长
    
    返回:
    performance_list: list, 每个窗口的绩效指标
    """
    performance_list = []
    
    for start in range(0, len(data) - window_size, step):
        # 训练窗口
        train_data = data.iloc[start:start+window_size]
        
        # 测试窗口
        test_start = start + window_size
        test_end = min(test_start + step, len(data))
        test_data = data.iloc[test_start:test_end]
        
        # 在训练窗口上优化参数（如果有）
        # ...
        
        # 在测试窗口上测试策略
        strategy = strategy_class(train_data.columns[0], train_data.columns[1], 
                                test_data, signal_generator)
        _, performance = strategy.backtest()
        
        performance_list.append(performance)
    
    return performance_list
```

### 5.5 陷阱5：风险集中

**问题：** 如果多个配对高度相关，风险会集中。

**解决方案：**
- 计算配对间的相关性，避免选择高度相关的配对
- 限制单个行业或板块的配对数量
- 使用风险平价（Risk Parity）方法分配资金

## 六、总结与展望

配对交易是统计套利的经典策略，核心在于**识别均值回归的机会**。本文介绍了一套完整的配对交易框架：

### 核心要点

1. **理论基础：** 协整关系是配对交易的基石，比简单的相关性更可靠
2. **科学选对：** 使用协整检验、混合法或距离法科学地选择配对
3. **信号生成：** Z-Score、布林带、卡尔曼滤波都是有效方法
4. **风险管理：** 警惕结构性断裂、交易成本、流动性风险等陷阱

### 实践建议

- **从小做起：** 先用模拟盘测试，再投入实盘
- **持续优化：** 市场环境在变化，策略也需要不断迭代
- **保持谦逊：** 统计套利不是"印钞机"，也有失效的时候

### 未来方向

1. **机器学习：** 用随机森林、LSTM等方法预测价差回归概率
2. **高频配对交易：** 利用tick级数据进行高频统计套利
3. **跨市场配对：** 在不同市场（如股票与期货、不同国家的股票）间寻找配对机会

---

**参考文献：**

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies.
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
3. Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques." Wiley.

**免责声明：** 本文仅供学术交流，不构成投资建议。配对交易有风险，入市需谨慎。
