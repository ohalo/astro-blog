---
title: "配对交易与协整分析：统计套利实战指南"
description: "深入讲解配对交易策略的原理、协整检验方法、交易信号构建和实战回测，提供完整的Python实现代码和风险控制方案"
publishDate: '2026-06-17'
language: Chinese
category: "量化交易"
tags: ["配对交易", "协整分析", "统计套利", "均值回归", "Python量化"]
---

# 配对交易与协整分析：统计套利实战指南

## 引言

配对交易（Pairs Trading）是最经典的统计套利策略之一，其核心理念是"买入被低估的资产，做空被高估的资产，等待价差回归"。本文将系统讲解配对交易的理论基础、协整检验方法、实战策略构建，并提供完整的Python实现。

## 一、配对交易的理论基础

### 1.1 什么是协整（Cointegration）？

协整是配对交易的核心数学基础。简单说，两个非平稳时间序列如果它们的线性组合是平稳的，那么这两个序列就是协整的。

**数学定义**：
对于两个I(1)过程（一阶单整，即一阶差分后平稳）$X_t$ 和 $Y_t$，如果存在参数 $\beta$ 使得：

$$
Z_t = Y_t - \beta X_t
$$

是平稳过程（I(0)），则称 $X_t$ 和 $Y_t$ 是协整的，$\beta$ 称为协整系数。

**经济学含义**：
协整关系意味着两个资产的价格之间存在长期均衡关系，尽管短期可能偏离，但偏离是暂时的，最终会回归均衡。

### 1.2 为什么配对交易有效？

1. **均值回归特性**：价差（spread）具有均值回归特性，这是统计套利的利润来源。
2. **市场中性**：同时做多和做空，对冲市场风险，获取纯alpha。
3. **低风险**：不依赖市场方向，适合震荡市和趋势不明显的市场。

## 二、协整检验方法

### 2.1 Engle-Granger两步法

**步骤1**：估计协整回归

$$
Y_t = \alpha + \beta X_t + \epsilon_t
$$

**步骤2**：检验残差 $\hat{\epsilon}_t$ 的平稳性（使用ADF检验）

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger协整检验
    
    参数：
    y: array-like, 因变量
    x: array-like, 自变量
    verbose: bool, 是否打印结果
    
    返回：
    dict: 包含协整系数、残差ADF统计量、p值等
    """
    # 步骤1：OLS回归
    X = sm.add_constant(x)
    model = OLS(y, X).fit()
    beta = model.params[1]
    alpha = model.params[0]
    residuals = model.resid
    
    # 步骤2：ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整（5%显著性水平）
    is_cointegrated = adf_stat < critical_values['5%']
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger协整检验结果")
        print("=" * 60)
        print(f"协整方程: Y = {alpha:.4f} + {beta:.4f} * X")
        print(f"\n残差ADF检验统计量: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"临界值 (5%): {critical_values['5%']:.4f}")
        print(f"是否协整: {'是' if is_cointegrated else '否'}")
        print("=" * 60)
    
    return {
        'alpha': alpha,
        'beta': beta,
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_cointegrated': is_cointegrated,
        'residuals': residuals
    }

# 示例
np.random.seed(42)
T = 1000
x = np.cumsum(np.random.normal(0, 1, T))  # 随机游走
y = 0.5 * x + np.random.normal(0, 0.5, T)  # 协整关系

result = engle_granger_test(y, x)
```

### 2.2 Johansen检验（多变量协整）

当需要检验多个变量之间的协整关系时，使用Johansen检验：

```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    参数：
    data: DataFrame, 多变量时间序列
    det_order: int, 确定性项顺序（0=无常数项，1=有常数项）
    k_ar_diff: int, 滞后阶数
    
    返回：
    result: Johansen检验结果
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    print("=" * 60)
    print("Johansen协整检验结果")
    print("=" * 60)
    print(f"特征值 (Eigenvalues): {result.eig}")
    print(f"\n迹统计量 (Trace Statistic):")
    for i, (stat, crit) in enumerate(zip(result.lr1, result.cvt[:, 1])):  # 5%临界值
        print(f"  r<={i}: {stat:.4f} (临界值: {crit:.4f})")
    print("=" * 60)
    
    return result

# 示例：三个资产的协整检验
data = pd.DataFrame({
    'Asset1': np.cumsum(np.random.normal(0, 1, T)),
    'Asset2': 0.5 * np.cumsum(np.random.normal(0, 1, T)) + np.random.normal(0, 0.5, T),
    'Asset3': -0.3 * np.cumsum(np.random.normal(0, 1, T)) + np.random.normal(0, 0.3, T)
})

result = johansen_test(data)
```

## 三、实战：构建配对交易策略

### 3.1 数据准备与配对筛选

首先，我们需要筛选出适合配对交易的股票对。常用方法：

1. **相关性筛选**：相关系数 > 0.7
2. **协整检验**：ADF p-value < 0.05
3. **行业分类**：同行业股票更可能存在协整关系

```python
class PairSelector:
    """
    配对筛选器：从股票池中筛选协整配对
    """
    
    def __init__(self, price_data, min_corr=0.7, max_pvalue=0.05):
        """
        初始化
        
        参数：
        price_data: DataFrame, 股票价格数据（已经过处理）
        min_corr: float, 最小相关系数
        max_pvalue: float, 最大ADF p值
        """
        self.price_data = price_data
        self.min_corr = min_corr
        self.max_pvalue = max_pvalue
        self.returns_data = price_data.pct_change().dropna()
        
    def compute_correlation_matrix(self):
        """计算相关系数矩阵"""
        corr_matrix = self.returns_data.corr()
        return corr_matrix
    
    def screen_by_correlation(self):
        """第一步：相关性筛选"""
        corr_matrix = self.compute_correlation_matrix()
        
        pairs = []
        stocks = corr_matrix.columns
        
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= self.min_corr:
                    pairs.append((stocks[i], stocks[j], corr))
        
        # 按相关系数排序
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"通过相关性筛选的配对数: {len(pairs)}")
        return pairs
    
    def test_cointegration(self, pairs):
        """
        第二步：协整检验
        
        参数：
        pairs: list, (stock1, stock2, corr) 元组列表
        
        返回：
        cointegrated_pairs: list, 通过协整检验的配对
        """
        cointegrated_pairs = []
        
        for stock1, stock2, corr in pairs:
            # 获取价格序列
            y = self.price_data[stock1].values
            x = self.price_data[stock2].values
            
            # Engle-Granger检验
            result = engle_granger_test(y, x, verbose=False)
            
            if result['is_cointegrated'] and result['p_value'] < self.max_pvalue:
                cointegrated_pairs.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'correlation': corr,
                    'beta': result['beta'],
                    'alpha': result['alpha'],
                    'adf_stat': result['adf_stat'],
                    'p_value': result['p_value'],
                    'residuals': result['residuals']
                })
        
        print(f"通过协整检验的配对数: {len(cointegrated_pairs)}")
        return cointegrated_pairs
    
    def select_pairs(self, top_n=10):
        """
        完整筛选流程
        
        返回：
        top_pairs: list, 前N个最优配对
        """
        # 第一步：相关性筛选
        pairs = self.screen_by_correlation()
        
        # 第二步：协整检验
        cointegrated_pairs = self.test_cointegration(pairs)
        
        # 选择前N个
        top_pairs = cointegrated_pairs[:top_n]
        
        print("\n=== Top 配对 ===")
        for i, pair in enumerate(top_pairs, 1):
            print(f"{i}. {pair['stock1']} - {pair['stock2']}")
            print(f"   相关系数: {pair['correlation']:.4f}")
            print(f"   Beta: {pair['beta']:.4f}")
            print(f"   ADF p-value: {pair['p_value']:.4f}")
        
        return top_pairs
```

### 3.2 交易信号构建

配对交易的核心是利用价差的均值回归特性。常用方法：

**方法1：Z-Score策略**

```python
def compute_zscore(spread, window=20):
    """
    计算价差的Z-Score
    
    参数：
    spread: array-like, 价差序列
    window: int, 滚动窗口
    
    返回：
    zscore: array, Z-Score序列
    """
    mean = pd.Series(spread).rolling(window=window).mean()
    std = pd.Series(spread).rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore.values

def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    生成交易信号
    
    参数：
    zscore: array, Z-Score序列
    entry_threshold: float, 入场阈值
    exit_threshold: float, 出场阈值
    
    返回：
    signals: DataFrame, 包含交易信号
    """
    signals = pd.DataFrame(index=range(len(zscore)))
    signals['zscore'] = zscore
    signals['position'] = 0  # 0=空仓, 1=做多价差, -1=做空价差
    
    # 简单阈值策略
    for i in range(1, len(signals)):
        if signals.loc[i-1, 'position'] == 0:  # 当前空仓
            if signals.loc[i, 'zscore'] < -entry_threshold:
                signals.loc[i, 'position'] = 1  # 做多价差（买stock1，卖stock2）
            elif signals.loc[i, 'zscore'] > entry_threshold:
                signals.loc[i, 'position'] = -1  # 做空价差（卖stock1，买stock2）
        
        elif signals.loc[i-1, 'position'] == 1:  # 当前做多价差
            if signals.loc[i, 'zscore'] >= -exit_threshold:
                signals.loc[i, 'position'] = 0  # 平仓
            else:
                signals.loc[i, 'position'] = 1  # 继续持有
        
        elif signals.loc[i-1, 'position'] == -1:  # 当前做空价差
            if signals.loc[i, 'zscore'] <= exit_threshold:
                signals.loc[i, 'position'] = 0  # 平仓
            else:
                signals.loc[i, 'position'] = -1  # 继续持有
    
    return signals
```

**方法2：布林带策略**

```python
def bollinger_band_strategy(spread, window=20, num_std=2.0):
    """
    布林带策略
    
    参数：
    spread: array-like, 价差序列
    window: int, 移动窗口
    num_std: float, 标准差倍数
    
    返回：
    signals: DataFrame, 交易信号
    """
    spread_series = pd.Series(spread)
    mean = spread_series.rolling(window=window).mean()
    std = spread_series.rolling(window=window).std()
    
    upper_band = mean + num_std * std
    lower_band = mean - num_std * std
    
    signals = pd.DataFrame(index=spread_series.index)
    signals['spread'] = spread
    signals['mean'] = mean
    signals['upper'] = upper_band
    signals['lower'] = lower_band
    signals['position'] = 0
    
    # 生成信号
    for i in range(1, len(signals)):
        if signals.loc[i-1, 'position'] == 0:
            if signals.loc[i, 'spread'] < signals.loc[i, 'lower']:
                signals.loc[i, 'position'] = 1  # 做多价差
            elif signals.loc[i, 'spread'] > signals.loc[i, 'upper']:
                signals.loc[i, 'position'] = -1  # 做空价差
        else:
            # 平仓条件：价差回归均值
            if abs(signals.loc[i, 'spread'] - signals.loc[i, 'mean']) < std.iloc[i] * 0.5:
                signals.loc[i, 'position'] = 0
            else:
                signals.loc[i, 'position'] = signals.loc[i-1, 'position']
    
    return signals
```

### 3.3 回测框架

```python
class PairTradingBacktester:
    """
    配对交易回测器
    """
    
    def __init__(self, price_data, initial_capital=1000000, transaction_cost=0.001):
        """
        初始化
        
        参数：
        price_data: DataFrame, 包含stock1和stock2的价格
        initial_capital: float, 初始资金
        transaction_cost: float, 交易成本（单边）
        """
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def backtest(self, signals, pair_info, plot=True):
        """
        回测
        
        参数：
        signals: DataFrame, 交易信号
        pair_info: dict, 配对信息（beta等）
        plot: bool, 是否绘图
        
        返回：
        results: dict, 回测结果
        """
        stock1 = pair_info['stock1']
        stock2 = pair_info['stock2']
        beta = pair_info['beta']
        
        # 初始化
        capital = self.initial_capital
        position = 0  # 持仓方向
        shares1 = 0
        shares2 = 0
        
        portfolio_value = []
        trades = []
        
        for i in range(len(signals)):
            date = signals.index[i]
            signal = signals['position'].iloc[i]
            
            price1 = self.price_data[stock1].iloc[i]
            price2 = self.price_data[stock2].iloc[i]
            
            # 交易逻辑
            if signal != position:  # 信号变化
                if position == 0:  # 开仓
                    # 计算仓位（等市值）
                    notional = capital * 0.5  # 每个股票分配50%资金
                    shares1 = int(notional / price1 / 100) * 100  # A股100股整数倍
                    shares2 = int(shares1 * beta / 100) * 100
                    
                    # 交易成本
                    cost = (shares1 * price1 + shares2 * price2) * self.transaction_cost
                    capital -= cost
                    
                    position = signal
                    trades.append({
                        'date': date,
                        'action': 'OPEN',
                        'signal': signal,
                        'price1': price1,
                        'price2': price2,
                        'shares1': shares1,
                        'shares2': shares2,
                        'cost': cost
                    })
                
                elif signal == 0:  # 平仓
                    # 卖出持仓
                    proceeds = shares1 * price1 + shares2 * price2
                    cost = proceeds * self.transaction_cost
                    capital += proceeds - cost
                    
                    position = 0
                    shares1 = 0
                    shares2 = 0
                    
                    trades.append({
                        'date': date,
                        'action': 'CLOSE',
                        'price1': price1,
                        'price2': price2,
                        'proceeds': proceeds,
                        'cost': cost
                    })
            
            # 计算组合价值
            portfolio = capital + shares1 * price1 + shares2 * price2
            portfolio_value.append({
                'date': date,
                'value': portfolio
            })
        
        # 计算回测指标
        results = self.compute_performance(portfolio_value, trades)
        
        if plot:
            self.plot_results(portfolio_value, signals, pair_info)
        
        return results
    
    def compute_performance(self, portfolio_value, trades):
        """计算回测性能指标"""
        values = [pv['value'] for pv in portfolio_value]
        
        total_return = (values[-1] - values[0]) / values[0] * 100
        annual_return = (values[-1] / values[0]) ** (252 / len(values)) - 1
        
        # 计算最大回撤
        peak = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio（简化版）
        returns = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'portfolio_value': portfolio_value
        }
        
        return results
    
    def plot_results(self, portfolio_value, signals, pair_info):
        """绘制回测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 图1：组合价值曲线
        dates = [pv['date'] for pv in portfolio_value]
        values = [pv['value'] for pv in portfolio_value]
        
        axes[0].plot(dates, values, linewidth=2, color='#5470c6')
        axes[0].set_title('Portfolio Value', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 图2：价差与Z-Score
        spread = pair_info['residuals']
        zscore = compute_zscore(spread)
        
        ax2_twin = axes[1].twinx()
        
        axes[1].plot(dates, spread, linewidth=2, color='#91cc75', label='Spread')
        ax2_twin.plot(dates, zscore, linewidth=2, color='#ee6666', label='Z-Score')
        
        axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        ax2_twin.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        
        axes[1].set_title('Spread and Z-Score', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Spread', fontsize=12)
        ax2_twin.set_ylabel('Z-Score', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 图3：持仓
        axes[2].plot(dates, signals['position'], linewidth=2, color='#fac858')
        axes[2].set_title('Position', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Position', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pair_trading_backtest.png', dpi=300, bbox_inches='tight')
        plt.show()
```

## 四、实战案例：A股配对交易

```python
# 完整实战流程示例

# 1. 获取数据（使用westock-data）
# 假设我们获取了以下几只银行股的数据
stocks = ['600036.SH', '601398.SH', '601939.SH', '601288.SH']  # 招商银行、工商银行、建设银行、农业银行

# 2. 读取价格数据
# price_data = pd.DataFrame({stock: fetch_price(stock) for stock in stocks})

# 3. 配对筛选
# selector = PairSelector(price_data)
# top_pairs = selector.select_pairs(top_n=5)

# 4. 选择最优配对进行回测
# best_pair = top_pairs[0]
# signals = generate_signals(best_pair['residuals'])
# backtester = PairTradingBacktester(price_data)
# results = backtester.backtest(signals, best_pair)

# 5. 输出结果
# print(f"总收益率: {results['total_return']:.2f}%")
# print(f"年化收益率: {results['annual_return']*100:.2f}%")
# print(f"Sharpe比率: {results['sharpe_ratio']:.2f}")
# print(f"最大回撤: {results['max_drawdown']:.2f}%")
```

## 五、风险控制与改进

### 5.1 常见风险

1. **协整关系失效**：市场结构变化导致长期关系破裂
2. **价差不回归**：黑天鹅事件导致价差持续扩大
3. **交易成本**：频繁交易会侵蚀利润

### 5.2 改进方法

✅ **动态对冲比例**：使用滚动窗口估计时变Beta

```python
def rolling_beta(y, x, window=252):
    """滚动估计Beta"""
    beta_series = pd.Series(index=y.index)
    
    for i in range(window, len(y)):
        y_window = y.iloc[i-window:i]
        x_window = x.iloc[i-window:i]
        
        X = sm.add_constant(x_window)
        model = OLS(y_window, X).fit()
        beta_series.iloc[i] = model.params[1]
    
    return beta_series
```

✅ **止损机制**：设定最大亏损阈值

```python
def add_stop_loss(signals, spread, stop_loss_threshold=3.0):
    """添加止损"""
    zscore = compute_zscore(spread)
    
    for i in range(len(signals)):
        if signals.loc[i, 'position'] != 0:
            if abs(zscore[i]) > stop_loss_threshold:
                signals.loc[i, 'position'] = 0  # 强制平仓
    
    return signals
```

✅ **配对组合**：同时交易多个配对，分散风险

## 六、总结

配对交易是一种经典的统计套利策略，核心在于：

1. **协整检验**：使用Engle-Granger或Johansen检验确认长期均衡关系
2. **信号构建**：基于Z-Score或布林带捕捉均值回归机会
3. **风险控制**：动态对冲、止损、多配对分散

**实战建议**：
- 优先选择同行业、业务模式相似的公司
- 定期重新检验协整关系（建议每季度）
- 控制单次交易规模，避免过度杠杆
- 结合基本面分析，避免价值陷阱

---

**附录：完整代码仓库**

本文完整代码已上传至GitHub：[链接]（发布时添加）

**参考文献**：
1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
2. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*
