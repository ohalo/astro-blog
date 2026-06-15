---
title: "配对交易与协整分析：统计套利的核心技术"
description: "深入探讨配对交易的理论基础、协整检验方法、交易信号构建及风险管理，为量化交易者提供完整的统计套利实战框架。"
date: "2026-06-16"
tags: ["配对交易", "协整分析", "统计套利", "均值回归", "量化策略"]
categories: ["量化交易"]
slug: "pair-trading-cointegration"
draft: false
---

# 配对交易与协整分析：统计套利的核心技术

## 引言

配对交易（Pair Trading）作为统计套利（Statistical Arbitrage）的经典策略，凭借其市场中性、风险可控的特点，在量化投资领域占据重要地位。本文将系统介绍配对交易的理论基础——协整分析，从数学原理到实战应用，为读者构建一个完整的配对交易框架。

## 配对交易的基本原理

配对交易的核心思想是寻找两个价格具有长期均衡关系的资产，当它们的价格偏离均衡时建立对冲头寸，等待价格回归时获利。

### 为什么要使用协整分析？

传统的相关系数分析只能衡量短期同步性，而协整分析能够：
1. **识别长期均衡关系**：发现价格序列之间的稳定关系
2. **提供套利依据**：偏离均衡意味着均值回归机会
3. **构建市场中性策略**：多空对冲消除市场风险

## 协整关系的数学基础

### 平稳性与协整的定义

**平稳性（Stationarity）**：一个时间序列 $\{X_t\}$ 是平稳的，如果其统计特性（均值、方差、自协方差）不随时间变化。

**协整（Cointegration）**：如果两个或多个非平稳时间序列的线性组合是平稳的，则这些序列是协整的。

数学表达：
对于两个非平稳序列 $\{X_t\}$ 和 $\{Y_t\}$，如果存在参数 $\beta$ 使得：
$$Z_t = Y_t - \beta X_t$$
是平稳序列，则称 $X_t$ 和 $Y_t$ 是协整的。

### Engle-Granger 两步法

检验协整关系最经典的方法是 Engle-Granger 两步法：

**第一步**：估计协整回归
$$Y_t = \alpha + \beta X_t + \epsilon_t$$

**第二步**：检验残差 $\hat{\epsilon}_t$ 的平稳性

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger 协整检验
    
    参数:
        y: 因变量序列
        x: 自变量序列
        verbose: 是否打印详细信息
    
    返回:
        adf_statistic: ADF检验统计量
        p_value: p值
        is_cointegrated: 是否协整（5%显著性水平）
    """
    # 第一步：OLS回归
    x_with_const = np.column_stack([np.ones(len(x)), x])
    model = OLS(y, x_with_const).fit()
    residuals = model.resid
    
    # 第二步：ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整（5%显著性水平）
    is_cointegrated = p_value < 0.05
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger 协整检验结果")
        print("=" * 60)
        print(f"\n协整回归结果:")
        print(f"  截距 (α): {model.params[0]:.4f}")
        print(f"  斜率 (β): {model.params[1]:.4f}")
        print(f"  R²: {model.rsquared:.4f}")
        
        print(f"\n残差平稳性检验 (ADF Test):")
        print(f"  ADF统计量: {adf_statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  临界值 (1%, 5%, 10%): {critical_values}")
        print(f"\n  结论: {'存在协整关系' if is_cointegrated else '不存在协整关系'}")
        print("=" * 60)
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'is_cointegrated': is_cointegrated,
        'alpha': model.params[0],
        'beta': model.params[1],
        'residuals': residuals
    }

# 示例：生成协整序列并检验
np.random.seed(42)
n = 1000

# 生成随机游走
x = np.cumsum(np.random.randn(n) * 0.01 + 0.0001)

# 生成协整序列 y = alpha + beta*x + error
alpha_true = 0.5
beta_true = 1.2
y = alpha_true + beta_true * x + np.random.randn(n) * 0.02

# 执行协整检验
result = engle_granger_test(y, x)
```

## 配对选择的实战方法

### 1. 距离法（Distance Approach）

基于价格差异的归一化距离选择配对：

```python
def calculate_distance_measure(price1, price2, lookback=252):
    """
    计算配对的距离度量（归一化价格差异）
    
    参数:
        price1, price2: 两个资产的价格序列
        lookback: 回看窗口
    
    返回:
        distance: 距离得分（越小越好）
        spread: 价差序列
    """
    # 归一化价格
    norm_price1 = price1 / price1.iloc[0]
    norm_price2 = price2 / price2.iloc[0]
    
    # 计算价差
    spread = norm_price1 - norm_price2
    
    # 计算距离（价差的滚动标准差）
    distance = spread.rolling(window=lookback).std()
    
    return distance, spread

# 示例：在多个股票中寻找最佳配对
def find_best_pairs_universe(stock_data, top_n=10):
    """
    在股票池中寻找最佳配对
    
    参数:
        stock_data: 股票价格数据框（各列为不同股票）
        top_n: 返回前N个最佳配对
    
    返回:
        best_pairs: 最佳配对列表
    """
    stocks = stock_data.columns
    n = len(stocks)
    
    pair_scores = []
    
    for i in range(n):
        for j in range(i+1, n):
            stock1 = stocks[i]
            stock2 = stocks[j]
            
            # 计算距离
            distance, spread = calculate_distance_measure(
                stock_data[stock1],
                stock_data[stock2]
            )
            
            # 计算最终得分（使用最近的距离值）
            final_score = distance.iloc[-1]
            
            # 检验协整
            try:
                result = engle_granger_test(
                    stock_data[stock2].values,
                    stock_data[stock1].values,
                    verbose=False
                )
                
                if result['is_cointegrated']:
                    pair_scores.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'distance': final_score,
                        'adf_statistic': result['adf_statistic'],
                        'p_value': result['p_value'],
                        'beta': result['beta']
                    })
            except:
                continue
    
    # 按距离排序
    best_pairs = sorted(pair_scores, key=lambda x: x['distance'])[:top_n]
    
    return best_pairs

# 模拟数据示例
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'GS', 'BAC']

# 生成模拟价格数据
price_data = pd.DataFrame(index=dates)
for stock in stocks:
    # 生成具有不同协整关系的价格序列
    base = np.cumsum(np.random.randn(1000) * 0.01)
    if stock in ['JPM', 'GS', 'BAC']:  # 金融股组
        price = 100 + base * 2 + np.random.randn(1000) * 5
    elif stock in ['AAPL', 'MSFT', 'GOOGL']:  # 科技股组
        price = 200 + base * 1.5 + np.random.randn(1000) * 8
    else:
        price = 150 + base + np.random.randn(1000) * 10
    
    price_data[stock] = price

# 寻找最佳配对
best_pairs = find_best_pairs_universe(price_data, top_n=5)

print("\n=== 最佳配对排名 ===")
for i, pair in enumerate(best_pairs, 1):
    print(f"{i}. {pair['stock1']} - {pair['stock2']}")
    print(f"   距离得分: {pair['distance']:.4f}")
    print(f"   ADF统计量: {pair['adf_statistic']:.4f}")
    print(f"   p-value: {pair['p_value']:.4f}")
    print(f"   对冲比例 (β): {pair['beta']:.4f}\n")
```

### 2. 协整秩法（Cointegration Ranking）

基于Johansen检验的多变量协整分析：

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

def johansen_cointegration_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验（适用于多变量）
    
    参数:
        data: 数据矩阵（T×N，T为样本数，N为变量数）
        det_order: 确定性项顺序（0=无常数，1=有常数，2=有常数和趋势）
        k_ar_diff: 差分项的滞后阶数
    
    返回:
        trace_stat: Trace统计量
        max_stat: Maximum特征值统计量
        coint_rank: 协整秩（协整关系个数）
    """
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    # 选择协整秩
    rank_selection = select_coint_rank(data, det_order, k_ar_diff)
    
    # 获取统计量
    trace_stat = rank_selection.trace_stat
    max_stat = rank_selection.max_stat
    coint_rank = rank_selection.rank
    
    print("=" * 60)
    print("Johansen 协整检验")
    print("=" * 60)
    print(f"\n协整秩 (协整关系个数): {coint_rank}")
    print(f"\nTrace 统计量:")
    for i, stat in enumerate(trace_stat):
        print(f"  r<={i}: {stat:.4f}")
    print(f"\nMaximum 特征值统计量:")
    for i, stat in enumerate(max_stat):
        print(f"  r={i}: {stat:.4f}")
    print("=" * 60)
    
    return {
        'trace_stat': trace_stat,
        'max_stat': max_stat,
        'coint_rank': coint_rank
    }

# 示例：三变量协整检验
np.random.seed(42)
n = 1000

# 生成两个独立的随机游走
rw1 = np.cumsum(np.random.randn(n) * 0.01)
rw2 = np.cumsum(np.random.randn(n) * 0.01)

# 生成第三个序列，与前两个协整
y = 0.5 * rw1 + 0.3 * rw2 + np.random.randn(n) * 0.02

# 构建数据矩阵
data_matrix = np.column_stack([rw1, rw2, y])

# 执行Johansen检验
johansen_result = johansen_cointegration_test(data_matrix)
```

## 交易信号的构建

### 基于Z-Score的信号生成

```python
class PairTradingStrategy:
    """配对交易策略类"""
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, lookback=252):
        """
        初始化策略参数
        
        参数:
            entry_threshold: 入场阈值（Z-Score绝对值）
            exit_threshold: 出场阈值（Z-Score绝对值）
            lookback: 计算均值的回看窗口
        """
        self.entry_thresh = entry_threshold
        self.exit_thresh = exit_threshold
        self.lookback = lookback
        
        self.spread = None
        self.zscore = None
        self.position = 0  # 0=无仓位, 1=多空仓, -1=空多仓
        
    def calculate_spread(self, price1, price2, method='ols'):
        """
        计算价差（对冲后）
        
        参数:
            price1, price2: 两个资产的价格序列
            method: 计算方法 ('ols' 或 'ratio')
        
        返回:
            spread: 价差序列
        """
        if method == 'ols':
            # 使用OLS回归计算对冲比例
            x = np.column_stack([np.ones(len(price1)), price1])
            model = OLS(price2, x).fit()
            beta = model.params[1]
            spread = price2 - beta * price1
            
        elif method == 'ratio':
            # 使用价格比
            spread = np.log(price2 / price1)
            
        self.spread = spread
        return spread
    
    def calculate_zscore(self, spread=None):
        """计算Z-Score"""
        if spread is None:
            spread = self.spread
        
        # 计算滚动均值和标准差
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()
        
        # 计算Z-Score
        zscore = (spread - rolling_mean) / rolling_std
        self.zscore = zscore
        
        return zscore
    
    def generate_signals(self, zscore=None):
        """
        生成交易信号
        
        返回:
            signals: 交易信号序列 (1=做多价差, -1=做空价差, 0=平仓)
        """
        if zscore is None:
            zscore = self.zscore
        
        signals = pd.Series(index=zscore.index, data=0)
        
        # 生成信号
        for i in range(1, len(zscore)):
            if self.position == 0:  # 无仓位
                if zscore.iloc[i] < -self.entry_thresh:
                    # Z-Score过低，做多价差（买入资产2，卖出资产1）
                    signals.iloc[i] = 1
                    self.position = 1
                elif zscore.iloc[i] > self.entry_thresh:
                    # Z-Score过高，做空价差（卖出资产2，买入资产1）
                    signals.iloc[i] = -1
                    self.position = -1
                    
            elif self.position == 1:  # 持有做多价差仓位
                if abs(zscore.iloc[i]) < self.exit_thresh:
                    # Z-Score回归，平仓
                    signals.iloc[i] = 0
                    self.position = 0
                    
            elif self.position == -1:  # 持有做空价差仓位
                if abs(zscore.iloc[i]) < self.exit_thresh:
                    # Z-Score回归，平仓
                    signals.iloc[i] = 0
                    self.position = 0
        
        return signals
    
    def backtest(self, price1, price2, signals=None):
        """
        回测策略
        
        参数:
            price1, price2: 价格序列
            signals: 交易信号（如果不提供则自动生成）
        
        返回:
            results: 回测结果字典
        """
        # 计算价差和信号
        if self.spread is None:
            self.calculate_spread(price1, price2)
        if self.zscore is None:
            self.calculate_zscore()
        if signals is None:
            signals = self.generate_signals()
        
        # 计算对冲比例
        x = np.column_stack([np.ones(len(price1)), price1])
        model = OLS(price2, x).fit()
        beta = model.params[1]
        
        # 计算策略收益
        strategy_returns = pd.Series(index=price1.index, data=0.0)
        
        for i in range(1, len(signals)):
            if signals.iloc[i] != 0:
                # 入场或维持仓位
                # 做多价差：买入资产2，卖出资产1
                ret2 = (price2.iloc[i] - price2.iloc[i-1]) / price2.iloc[i-1]
                ret1 = (price1.iloc[i] - price1.iloc[i-1]) / price1.iloc[i-1]
                
                if signals.iloc[i] == 1:  # 做多价差
                    strategy_returns.iloc[i] = ret2 - beta * ret1
                elif signals.iloc[i] == -1:  # 做空价差
                    strategy_returns.iloc[i] = -ret2 + beta * ret1
        
        # 计算累积收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算性能指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        results = {
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'signals': signals
        }
        
        return results

# 示例使用策略
strategy = PairTradingStrategy(entry_threshold=2.0, exit_threshold=0.5)

# 使用之前生成的协整序列
price1 = pd.Series(x, index=dates)
price2 = pd.Series(y, index=dates)

# 回测策略
backtest_results = strategy.backtest(price1, price2)

print("\n=== 配对交易策略回测结果 ===")
print(f"总收益: {backtest_results['total_return']*100:.2f}%")
print(f"年化收益: {backtest_results['annual_return']*100:.2f}%")
print(f"夏普比率: {backtest_results['sharpe_ratio']:.4f}")
print(f"最大回撤: {backtest_results['max_drawdown']*100:.2f}%")
```

## 风险管理和实战要点

### 1. 止损策略

```python
def add_stop_loss(signals, zscore, stop_loss_threshold=3.0):
    """
    添加止损机制
    
    参数:
        signals: 原始交易信号
        zscore: Z-Score序列
        stop_loss_threshold: 止损阈值
    
    返回:
        signals_with_sl: 带止损的信号
    """
    signals_with_sl = signals.copy()
    position = 0
    
    for i in range(1, len(signals)):
        if position != 0:  # 有仓位
            # 检查是否触发止损
            if abs(zscore.iloc[i]) > stop_loss_threshold:
                signals_with_sl.iloc[i] = 0  # 平仓
                position = 0
                print(f"止损触发 @ {zscore.index[i]}: Z-Score = {zscore.iloc[i]:.2f}")
        else:
            position = signals.iloc[i]
    
    return signals_with_sl
```

### 2. 仓位管理

```python
def kelly_criterion_optimization(returns, lookback=252):
    """
    使用凯利公式计算最优仓位
    
    参数:
        returns: 策略收益率序列
        lookback: 回看窗口
    
    返回:
        kelly_fraction: 凯利分数（最优仓位比例）
    """
    # 计算滚动胜率和盈亏比
    rolling_returns = returns.rolling(window=lookback)
    
    win_rate = (rolling_returns.apply(lambda x: (x > 0).sum() / len(x))).fillna(0.5)
    avg_win = rolling_returns.apply(lambda x: x[x > 0].mean()).fillna(0)
    avg_loss = abs(rolling_returns.apply(lambda x: x[x < 0].mean())).fillna(0)
    
    # 计算盈亏比
    win_loss_ratio = avg_win / avg_loss.replace(0, np.inf)
    
    # 凯利公式: f = p - q/b
    # p = 胜率, q = 1-p, b = 盈亏比
    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
    
    # 限制仓位（不超过50%）
    kelly_fraction = kelly_fraction.clip(0, 0.5)
    
    return kelly_fraction

# 示例：计算动态仓位
returns = backtest_results['returns']
kelly_fractions = kelly_criterion_optimization(returns)

print(f"\n最新凯利仓位建议: {kelly_fractions.iloc[-1]*100:.1f}%")
```

### 3. 配对衰变监测

```python
def monitor_pair_degradation(spread, window=252, degradation_threshold=0.05):
    """
    监测配对关系的衰变
    
    参数:
        spread: 价差序列
        window: 滚动窗口
        degradation_threshold: 衰变阈值（ADF p-value）
    
    返回:
        degradation_alert: 衰变警报序列
    """
    degradation_alert = pd.Series(index=spread.index, data=False)
    
    for i in range(window, len(spread)):
        # 使用滚动窗口检验协整
        sub_sample = spread[i-window:i]
        
        try:
            adf_result = adfuller(sub_sample, autolag='AIC')
            p_value = adf_result[1]
            
            # 如果p-value过高，说明协整关系可能衰变
            if p_value > degradation_threshold:
                degradation_alert.iloc[i] = True
        except:
            continue
    
    return degradation_alert

# 监测配对衰变
degradation = monitor_pair_degradation(strategy.spread)

if degradation.any():
    print(f"\n⚠️ 检测到 {degradation.sum()} 个配对衰变点")
    print("建议：重新检验配对关系或暂停交易")
```

## 实战案例分析

让我们通过一个完整的实战案例来总结配对交易的流程：

```python
def complete_pair_trading_pipeline(stock1, stock2, price_data):
    """
    完整的配对交易流程
    
    参数:
        stock1, stock2: 股票代码
        price_data: 价格数据
    
    返回:
        report: 完整分析报告
    """
    print("\n" + "=" * 80)
    print(f"配对交易完整分析报告: {stock1} vs {stock2}")
    print("=" * 80)
    
    # 步骤1: 协整检验
    print("\n[步骤1] 协整关系检验...")
    result = engle_granger_test(
        price_data[stock2].values,
        price_data[stock1].values
    )
    
    if not result['is_cointegrated']:
        print("❌ 未通过协整检验，不建议配对交易")
        return None
    
    print("✅ 通过协整检验")
    
    # 步骤2: 计算价差和Z-Score
    print("\n[步骤2] 计算价差和交易信号...")
    strategy = PairTradingStrategy(entry_threshold=2.0, exit_threshold=0.5)
    strategy.calculate_spread(price_data[stock1], price_data[stock2])
    strategy.calculate_zscore()
    signals = strategy.generate_signals()
    
    # 步骤3: 回测
    print("\n[步骤3] 策略回测...")
    backtest = strategy.backtest(price_data[stock1], price_data[stock2], signals)
    
    # 步骤4: 风险指标
    print("\n[步骤4] 风险分析...")
    degradation = monitor_pair_degradation(strategy.spread)
    
    # 生成报告
    report = {
        'pair': f"{stock1}-{stock2}",
        'cointegration': result,
        'backtest': backtest,
        'degradation_points': degradation.sum(),
        'recommendation': 'PROCEED' if backtest['sharpe_ratio'] > 1.0 else 'CAUTION'
    }
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("报告摘要")
    print("=" * 80)
    print(f"配对: {report['pair']}")
    print(f"协整关系: {'存在' if result['is_cointegrated'] else '不存在'}")
    print(f"年化收益: {backtest['annual_return']*100:.2f}%")
    print(f"夏普比率: {backtest['sharpe_ratio']:.4f}")
    print(f"最大回撤: {backtest['max_drawdown']*100:.2f}%")
    print(f"配对衰变点: {report['degradation_points']}")
    print(f"\n最终建议: {'✅ 推荐交易' if report['recommendation'] == 'PROCEED' else '⚠️ 谨慎交易'}")
    print("=" * 80)
    
    return report

# 运行完整流程（使用之前找到的最佳配对）
if best_pairs:
    best = best_pairs[0]
    report = complete_pair_trading_pipeline(
        best['stock1'],
        best['stock2'],
        price_data
    )
```

## 结论

配对交易是一门艺术与科学的结合。通过本文的系统介绍，我们涵盖了：

1. **理论基础**：协整分析的数学原理
2. **配对选择**：距离法、协整秩法等实用方法
3. **信号构建**：基于Z-Score的交易系统
4. **风险管理**：止损、仓位管理、配对监测

### 实践建议

- **持续监测**：定期检查配对的协整关系是否依然稳定
- **分散投资**：同时交易多个不相关的配对，降低单一配对风险
- **成本控制**：考虑交易成本和滑点，确保策略净收益为正
- **市场适应性**：在不同市场环境下测试策略的稳健性

配对交易不是"印钞机"，但凭借其科学的框架和严格的风险管理，它可以为量化投资组合提供稳定的收益来源。

---

*本文所有代码示例仅供参考学习，实际交易需要结合实时数据和具体市场环境。量化交易有风险，实盘需谨慎。*
