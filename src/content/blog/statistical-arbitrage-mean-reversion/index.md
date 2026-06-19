---
title: "统计套利：均值回归策略的理论与实战"
date: 2026-06-19
description: "深入讲解统计套利的核心原理——均值回归，从协整检验到配对交易实战，提供完整的Python实现代码和风险管理框架。"
tags:
  - 统计套利
  - 均值回归
  - 配对交易
  - 协整分析
cover: /images/statistical-arbitrage-mean-reversion/cover.jpg
---

# 统计套利：均值回归策略的理论与实战

## 引言

**统计套利（Statistical Arbitrage）** 是量化交易中最经典的策略之一，其核心思想是：**价格偏离均值后终将回归**。通过数学模型识别价格偏离，在偏离达到一定程度时建立头寸，等待价格回归获利。

本文将系统讲解：
- 均值回归的理论基础
- 协整检验与配对交易
- Python实战：从数据获取到策略回测
- 风险管理与仓位控制
- 实际案例分析

## 一、均值回归的理论基础

### 1.1 随机游走 vs 均值回归

**随机游走假设（Random Walk）：**
```
P(t) = P(t-1) + ε(t)
```
其中 ε(t) 为白噪声，价格序列无记忆性，未来不可预测。

**均值回归过程（Mean Reversion）：**
```
P(t) - P(t-1) = -γ(P(t-1) - μ) + ε(t)
```
其中：
- μ 为长期均值
- γ 为回归速度（0 < γ < 1）
- 价格偏离均值后，会以指数速度回归

### 1.2 平稳性检验

**为什么需要平稳性？**
- 非平稳序列的统计特征随时间变化
- 回归模型会产生"伪回归"问题
- 只有平稳序列才能进行可靠的统计套利

**常用检验方法：**
1. **ADF检验（Augmented Dickey-Fuller Test）**
2. **KPSS检验（Kwiatkowski-Phillips-Schmidt-Shin）**
3. **PP检验（Phillips-Perron）**

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import yfinance as yf

def test_stationarity(price_series, verbose=True):
    """
    综合平稳性检验（ADF + KPSS）
    
    参数：
    - price_series: 价格序列（pandas Series）
    - verbose: 是否打印详细信息
    
    返回：
    - dict: 包含ADF和KPSS检验结果的字典
    """
    results = {}
    
    # 1. ADF检验
    # H0: 序列有单位根（非平稳）
    # H1: 序列平稳
    adf_result = adfuller(price_series, autolag='AIC')
    
    results['ADF'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05  # p值 < 0.05 拒绝原假设
    }
    
    # 2. KPSS检验
    # H0: 序列平稳
    # H1: 序列有单位根（非平稳）
    kpss_result = kpss(price_series, regression='c', nlags='auto')
    
    results['KPSS'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] > 0.05  # p值 > 0.05 不能拒绝原假设
    }
    
    if verbose:
        print("=" * 60)
        print("平稳性检验结果")
        print("=" * 60)
        print(f"\nADF检验：")
        print(f"  检验统计量: {results['ADF']['statistic']:.4f}")
        print(f"  p-value: {results['ADF']['p_value']:.4f}")
        print(f"  临界值: {results['ADF']['critical_values']}")
        print(f"  结论: {'平稳' if results['ADF']['is_stationary'] else '非平稳'}")
        
        print(f"\nKPSS检验：")
        print(f"  检验统计量: {results['KPSS']['statistic']:.4f}")
        print(f"  p-value: {results['KPSS']['p_value']:.4f}")
        print(f"  临界值: {results['KPSS']['critical_values']}")
        print(f"  结论: {'平稳' if results['KPSS']['is_stationary'] else '非平稳'}")
        
        # 综合判断
        if results['ADF']['is_stationary'] and results['KPSS']['is_stationary']:
            print(f"\n✅ 综合结论: 序列平稳")
        elif not results['ADF']['is_stationary'] and not results['KPSS']['is_stationary']:
            print(f"\n❌ 综合结论: 序列非平稳（有单位根）")
        else:
            print(f"\n⚠️  综合结论: 结果不确定（可能为趋势平稳）")
    
    return results

# 使用示例
# 下载数据
# data = yf.download('AAPL', start='2020-01-01', end='2026-06-19')['Adj Close']
# results = test_stationarity(data['AAPL'])
```

### 1.3 均值回归速度

**半衰期（Half-life）：**
```
HL = ln(2) / |ln(1 - γ)|
```

```python
def calculate_half_life(price_series):
    """
    计算均值回归的半衰期
    
    参数：
    - price_series: 价格序列
    
    返回：
    - half_life: 半衰期（天数）
    - gamma: 回归速度
    """
    # 计算价格变化
    price_diff = price_series.diff().dropna()
    price_lag = price_series.shift(1).dropna()
    
    # 回归: ΔP(t) = α + β * P(t-1) + ε(t)
    # 其中 β = -(1 - e^(-λΔt)) ≈ -(1 - γ)
    import statsmodels.api as sm
    X = sm.add_constant(price_lag.loc[price_diff.index])
    model = sm.OLS(price_diff, X).fit()
    
    beta = model.params.iloc[1]
    gamma = 1 + beta  # γ = 1 + β
    
    # 计算半衰期
    if gamma < 1 and gamma > 0:
        half_life = np.log(2) / abs(np.log(gamma))
    else:
        half_life = np.inf  # 非均值回归过程
    
    return half_life, gamma

# 可视化均值回归过程
def plot_mean_reversion_simulation():
    """模拟并可视化均值回归过程"""
    np.random.seed(42)
    T = 252  # 1年交易日
    mu = 100  # 长期均值
    gamma = 0.05  # 日度回归速度
    sigma = 2  # 波动率
    
    prices = [mu]
    for t in range(1, T):
        drift = -gamma * (prices[-1] - mu)
        shock = np.random.normal(0, sigma)
        new_price = prices[-1] + drift + shock
        prices.append(new_price)
    
    # 绘图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(range(T), prices, 'b-', linewidth=1.5, label='Price Path')
    ax.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Long-term Mean ({mu})')
    
    # 标记偏离和回归
    ax.fill_between(range(T), mu, prices, 
                    where=np.array(prices) > mu, 
                    alpha=0.3, color='red', label='Above Mean')
    ax.fill_between(range(T), mu, prices, 
                    where=np.array(prices) < mu, 
                    alpha=0.3, color='green', label='Below Mean')
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Mean Reversion Process Simulation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mean_reversion_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return prices

# 运行模拟
# prices = plot_mean_reversion_simulation()
# half_life, gamma = calculate_half_life(pd.Series(prices))
# print(f"回归速度 γ: {gamma:.4f}")
# print(f"半衰期: {half_life:.2f} 天")
```

## 二、协整检验与配对交易

### 2.1 协整（Cointegration）原理

**定义：** 两个或多个非平稳序列的线性组合是平稳的，则称这些序列协整。

**经济学意义：** 即使单个股票价格随机游走，它们之间的相对价格（价差）可能平稳，提供套利机会。

**数学表达：**
```
P₁(t) 和 P₂(t) 均为 I(1) 序列（非平稳）
如果存在 β 使得：
  Spread(t) = P₁(t) - β * P₂(t) 为 I(0) 序列（平稳）
则 P₁(t) 和 P₂(t) 协整。
```

### 2.2 Engle-Granger 协整检验

**检验步骤：**
1. 用OLS估计协整关系：`P₁(t) = α + β * P₂(t) + ε(t)`
2. 对残差 ε(t) 进行ADF检验
3. 若残差平稳，则两序列协整

```python
def engle_granger_test(price1, price2, verbose=True):
    """
    Engle-Granger协整检验
    
    参数：
    - price1, price2: 两个价格序列
    - verbose: 是否打印详细信息
    
    返回：
    - dict: 检验结果
    """
    # 步骤1: OLS回归
    import statsmodels.api as sm
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    
    alpha = model.params['const']
    beta = model.params[price2.name if hasattr(price2, 'name') else price2.columns[0]]
    spread = model.resid
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger协整检验")
        print("=" * 60)
        print(f"\n协整方程: P₁ = {alpha:.4f} + {beta:.4f} * P₂")
        print(f"R² = {model.rsquared:.4f}")
    
    # 步骤2: 对残差进行ADF检验
    adf_result = adfuller(spread, autolag='AIC')
    
    results = {
        'alpha': alpha,
        'beta': beta,
        'spread': spread,
        'adf_statistic': adf_result[0],
        'adf_p_value': adf_result[1],
        'adf_critical_values': adf_result[4],
        'is_cointegrated': adf_result[1] < 0.05
    }
    
    if verbose:
        print(f"\n残差ADF检验：")
        print(f"  检验统计量: {results['adf_statistic']:.4f}")
        print(f"  p-value: {results['adf_p_value']:.4f}")
        print(f"  临界值: {results['adf_critical_values']}")
        print(f"\n  结论: {'协整' if results['is_cointegrated'] else '不协整'}")
    
    return results

# 使用示例
# 下载两只相关股票的数据
# stock1 = yf.download('AAPL', start='2020-01-01', end='2026-06-19')['Adj Close']
# stock2 = yf.download('MSFT', start='2020-01-01', end='2026-06-19')['Adj Close']
# result = engle_granger_test(stock1, stock2)
```

### 2.3 配对交易策略设计

**核心逻辑：**
1. 计算价差（Spread）：`S(t) = P₁(t) - β * P₂(t)`
2. 计算价差的Z-Score：`Z(t) = (S(t) - μ) / σ`
3. 交易信号：
   - `Z(t) > 2`：做空价差（卖P₁，买P₂）
   - `Z(t) < -2`：做多价差（买P₁，卖P₂）
   - `|Z(t)| < 0.5`：平仓

```python
class PairsTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, stock1, stock2, lookback=63, entry_z=2.0, exit_z=0.5):
        """
        初始化配对交易策略
        
        参数：
        - stock1, stock2: 两只股票的价格序列
        - lookback: 滚动窗口（用于计算均值和标准差）
        - entry_z: 入场Z-Score阈值
        - exit_z: 出场Z-Score阈值
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        
        # 协整检验
        self.cointegration_result = engle_granger_test(stock1, stock2, verbose=False)
        self.spread = self.cointegration_result['spread']
        
    def calculate_z_score(self):
        """计算滚动Z-Score"""
        spread_mean = self.spread.rolling(window=self.lookback).mean()
        spread_std = self.spread.rolling(window=self.lookback).std()
        
        z_score = (self.spread - spread_mean) / spread_std
        
        return z_score
    
    def generate_signals(self):
        """生成交易信号"""
        z_score = self.calculate_z_score()
        
        # 初始化信号
        signals = pd.DataFrame(index=z_score.index)
        signals['z_score'] = z_score
        signals['position'] = 0  # 0: 空仓, 1: 做多价差, -1: 做空价差
        
        # 生成信号
        signals['position'] = np.where(
            z_score < -self.entry_z, 1, 
            np.where(z_score > self.entry_z, -1, 0)
        )
        
        # 平仓信号（当Z-Score回归时）
        for i in range(1, len(signals)):
            if signals['position'].iloc[i-1] != 0 and abs(z_score.iloc[i]) < self.exit_z:
                signals.loc[signals.index[i], 'position'] = 0
            else:
                if i < len(signals):
                    signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]
        
        return signals
    
    def backtest(self, initial_capital=1000000, transaction_cost=0.001):
        """
        回测配对交易策略
        
        参数：
        - initial_capital: 初始资金
        - transaction_cost: 交易成本（双边）
        
        返回：
        - results: 回测结果DataFrame
        """
        signals = self.generate_signals()
        
        # 计算收益率
        ret1 = self.stock1.pct_change()
        ret2 = self.stock2.pct_change()
        
        # 计算策略收益
        # 做多价差：买stock1，卖stock2
        # 做空价差：卖stock1，买stock2
        strategy_ret = signals['position'].shift(1) * (ret1 - self.cointegration_result['beta'] * ret2)
        
        # 扣除交易成本
        trades = signals['position'].diff().fillna(0).abs()
        cost = trades * transaction_cost
        strategy_ret_net = strategy_ret - cost
        
        # 计算累积收益
        results = pd.DataFrame(index=strategy_ret.index)
        results['strategy_returns'] = strategy_ret_net
        results['cumulative_returns'] = (1 + strategy_ret_net).cumprod()
        results['buy_hold_returns'] = (1 + ret1).cumprod()  # 基准：买入持有stock1
        results['spread'] = self.spread
        results['z_score'] = signals['z_score']
        results['position'] = signals['position']
        
        # 计算绩效指标
        total_return = results['cumulative_returns'].iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * strategy_ret_net.mean() / strategy_ret_net.std()
        max_drawdown = (results['cumulative_returns'] / results['cumulative_returns'].cummax() - 1).min()
        
        print("=" * 60)
        print("配对交易策略回测结果")
        print("=" * 60)
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益率: {((1+total_return)**(252/len(results))-1)*100:.2f}%")
        print(f"夏普比率: {sharpe_ratio:.4f}")
        print(f"最大回撤: {max_drawdown*100:.2f}%")
        print(f"交易次数: {int(trades.sum()/2)}")
        
        return results

# 可视化策略表现
def plot_pairs_trading_results(results, stock1_name='Stock1', stock2_name='Stock2'):
    """绘制配对交易策略结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 子图1：累积收益对比
    axes[0].plot(results.index, results['cumulative_returns'], 
                'b-', linewidth=2, label='Pairs Trading Strategy')
    axes[0].plot(results.index, results['buy_hold_returns'], 
                'r--', linewidth=2, label=f'Buy & Hold ({stock1_name})')
    axes[0].set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Returns', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：价差与Z-Score
    ax2_twin = axes[1].twinx()
    
    axes[1].plot(results.index, results['spread'], 
                'g-', linewidth=1.5, label='Spread')
    axes[1].axhline(y=results['spread'].mean(), color='black', 
                    linestyle='--', alpha=0.5, label='Mean')
    
    z_score = results['z_score']
    ax2_twin.plot(results.index, z_score, 'b-', linewidth=1, alpha=0.7, label='Z-Score')
    ax2_twin.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax2_twin.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
    
    axes[1].set_title('Spread and Z-Score', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Spread', fontsize=12)
    ax2_twin.set_ylabel('Z-Score', fontsize=12)
    axes[1].legend(loc='upper left', fontsize=9)
    ax2_twin.legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：仓位变化
    axes[2].plot(results.index, results['position'], 
                'k-', linewidth=2, label='Position')
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[2].fill_between(results.index, 0, results['position'], 
                        where=results['position'] > 0, 
                        alpha=0.3, color='green', label='Long Spread')
    axes[2].fill_between(results.index, 0, results['position'], 
                        where=results['position'] < 0, 
                        alpha=0.3, color='red', label='Short Spread')
    axes[2].set_title('Position Over Time', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Position', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pairs_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 完整使用示例
# 1. 下载数据
# stock1 = yf.download('AAPL', start='2020-01-01', end='2026-06-19')['Adj Close']
# stock2 = yf.download('MSFT', start='2020-01-01', end='2026-06-19')['Adj Close']

# 2. 初始化策略
# strategy = PairsTradingStrategy(stock1, stock2, lookback=63, entry_z=2.0, exit_z=0.5)

# 3. 回测
# results = strategy.backtest(initial_capital=1000000, transaction_cost=0.001)

# 4. 可视化
# plot_pairs_trading_results(results, 'AAPL', 'MSFT')
```

## 三、多因子均值回归模型

### 3.1 扩展协整关系

**多因子模型：**
```
P₁(t) = α + β₁*P₂(t) + β₂*P₃(t) + ... + βₙ*Pₙ₊₁(t) + ε(t)
```

```python
def multi_factor_cointegration(target_stock, factor_stocks, lookback=252):
    """
    多因子协整分析
    
    参数：
    - target_stock: 目标股票价格序列
    - factor_stocks: 因子股票列表（多个）
    - lookback: 滚动窗口
    
    返回：
    - results: 包含回归结果和残差检验结果
    """
    import statsmodels.api as sm
    
    # 合并数据
    all_data = pd.concat([target_stock] + factor_stocks, axis=1).dropna()
    
    # OLS回归
    X = all_data[factor_stocks.columns]
    X = sm.add_constant(X)
    y = all_data[target_stock.name]
    
    model = sm.OLS(y, X).fit()
    spread = model.resid
    
    # ADF检验残差
    adf_result = adfuller(spread, autolag='AIC')
    
    results = {
        'model': model,
        'spread': spread,
        'adf_statistic': adf_result[0],
        'adf_p_value': adf_result[1],
        'is_cointegrated': adf_result[1] < 0.05,
        'r_squared': model.rsquared
    }
    
    print("=" * 60)
    print("多因子协整检验结果")
    print("=" * 60)
    print(model.summary())
    print(f"\n残差ADF检验 p-value: {results['adf_p_value']:.4f}")
    print(f"结论: {'协整' if results['is_cointegrated'] else '不协整'}")
    
    return results
```

### 3.2 卡尔曼滤波动态对冲比率

**优势：** 传统OLS使用固定对冲比率，卡尔曼滤波可以动态调整。

```python
from pykalman import KalmanFilter

def kalman_filter_dynamic_hedge(stock1, stock2):
    """
    使用卡尔曼滤波估计动态对冲比率
    
    参数：
    - stock1, stock2: 两个价格序列
    
    返回：
    - dynamic_beta: 动态对冲比率序列
    - spread: 动态价差序列
    """
    # 准备观测矩阵（stock2的价格）
    observations = stock2.values.reshape(-1, 1)
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=[1],  # 状态转移矩阵（假设beta随机游走）
        observation_matrices=observations,  # 观测矩阵
    )
    
    # 状态均值和协方差
    state_means, state_covariances = kf.filter(stock1.values)
    
    # 提取动态beta
    dynamic_beta = state_means.flatten()
    
    # 计算动态价差
    spread = stock1.values - dynamic_beta * stock2.values
    
    return dynamic_beta, spread

# 可视化动态对冲比率
def plot_dynamic_beta(stock1, stock2, dynamic_beta):
    """对比固定beta和动态beta"""
    import matplotlib.pyplot as plt
    
    # 固定beta（OLS）
    import statsmodels.api as sm
    X = sm.add_constant(stock2)
    model = sm.OLS(stock1, X).fit()
    fixed_beta = model.params[stock2.name]
    
    # 计算两种价差
    fixed_spread = stock1 - fixed_beta * stock2
    dynamic_spread = stock1 - pd.Series(dynamic_beta, index=stock1.index) * stock2
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1：动态beta vs 固定beta
    axes[0].plot(dynamic_beta, 'b-', linewidth=2, label='Dynamic Beta (Kalman Filter)')
    axes[0].axhline(y=fixed_beta, color='red', linestyle='--', 
                    linewidth=2, label=f'Fixed Beta (OLS): {fixed_beta:.4f}')
    axes[0].fill_between(range(len(dynamic_beta)), 
                        fixed_beta - 0.1, fixed_beta + 0.1, 
                        alpha=0.2, color='gray', label='±0.1 Range')
    axes[0].set_title('Dynamic vs Fixed Hedge Ratio', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Beta', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：价差对比
    axes[1].plot(fixed_spread, 'r-', linewidth=1.5, alpha=0.7, label='Fixed Beta Spread')
    axes[1].plot(dynamic_spread, 'b-', linewidth=1.5, alpha=0.7, label='Dynamic Beta Spread')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_title('Spread Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Spread', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fixed_spread, dynamic_spread
```

## 四、风险管理与仓位控制

### 4.1 止损策略

**三种止损方式：**

1. **时间止损：** 持仓超过N天强制平仓
2. **价格止损：** 价差突破历史极值的X倍标准差
3. **资金止损：** 亏损达到账户资金的Y%

```python
def implement_stop_loss(results, max_holding_days=20, max_loss_std=3, max_capital_loss=0.02):
    """
    实现多维度止损
    
    参数：
    - results: 回测结果DataFrame
    - max_holding_days: 最大持仓天数
    - max_loss_std: 最大亏损标准差倍数
    - max_capital_loss: 最大资金亏损比例
    
    返回：
    - results_with_stoploss: 加入止损信号的结果
    """
    results = results.copy()
    
    # 计算持仓天数
    position = results['position']
    holding_days = position.diff().ne(0).cumsum()
    
    # 计算价差止损阈值
    spread_mean = results['spread'].rolling(window=252).mean()
    spread_std = results['spread'].rolling(window=252).std()
    upper_bound = spread_mean + max_loss_std * spread_std
    lower_bound = spread_mean - max_loss_std * spread_std
    
    # 初始化止损信号
    results['stop_loss'] = 0
    
    for i in range(1, len(results)):
        # 条件1：时间止损
        if holding_days.iloc[i] > max_holding_days and position.iloc[i] != 0:
            results.loc[results.index[i], 'stop_loss'] = 1
        
        # 条件2：价格止损
        if position.iloc[i] > 0 and results['spread'].iloc[i] < lower_bound.iloc[i]:
            results.loc[results.index[i], 'stop_loss'] = 1
        elif position.iloc[i] < 0 and results['spread'].iloc[i] > upper_bound.iloc[i]:
            results.loc[results.index[i], 'stop_loss'] = 1
        
        # 条件3：资金止损（需要累积收益计算）
        cumulative_ret = results['cumulative_returns'].iloc[i]
        if cumulative_ret < 1 - max_capital_loss:
            results.loc[results.index[i], 'stop_loss'] = 1
    
    # 应用止损（强制平仓）
    results['position_with_stoploss'] = results['position'].copy()
    results.loc[results['stop_loss'] == 1, 'position_with_stoploss'] = 0
    
    return results
```

### 4.2 仓位管理

**Kelly公式计算最优仓位：**

```
f* = (p * b - q) / b
```
其中：
- f*: 最优仓位比例
- p: 胜率
- q: 败率（1-p）
- b: 盈亏比（平均盈利/平均亏损）

```python
def kelly_position_sizing(win_rate, win_loss_ratio, max_position=0.25):
    """
    计算Kelly仓位
    
    参数：
    - win_rate: 胜率
    - win_loss_ratio: 盈亏比
    - max_position: 最大仓位限制（防止过度杠杆）
    
    返回：
    - kelly_fraction: Kelly仓位比例
    """
    loss_rate = 1 - win_rate
    
    # 完整Kelly公式
    kelly_full = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
    
    # 半Kelly（更保守）
    kelly_half = kelly_full / 2
    
    # 限制最大仓位
    kelly_final = min(kelly_half, max_position)
    
    print("=" * 60)
    print("Kelly仓位计算")
    print("=" * 60)
    print(f"胜率: {win_rate*100:.2f}%")
    print(f"盈亏比: {win_loss_ratio:.2f}")
    print(f"完整Kelly: {kelly_full*100:.2f}%")
    print(f"半Kelly: {kelly_half*100:.2f}%")
    print(f"最终仓位（含限制）: {kelly_final*100:.2f}%")
    
    return kelly_final

# 根据历史交易记录计算胜率和盈亏比
def calculate_trading_statistics(trade_returns):
    """
    计算交易统计数据
    
    参数：
    - trade_returns: 每笔交易的收益率列表
    
    返回：
    - stats: 包含胜率、盈亏比等的字典
    """
    import numpy as np
    
    trades = np.array(trade_returns)
    
    # 胜率
    win_rate = len(trades[trades > 0]) / len(trades)
    
    # 平均盈利和亏损
    avg_win = trades[trades > 0].mean() if len(trades[trades > 0]) > 0 else 0
    avg_loss = abs(trades[trades < 0].mean()) if len(trades[trades < 0]) > 0 else 0
    
    # 盈亏比
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
    
    stats = {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'expectancy': win_rate * avg_win - (1 - win_rate) * avg_loss
    }
    
    return stats
```

## 五、实战案例：A股配对交易

### 5.1 标的选取

**选取原则：**
1. 同行业（相关性高）
2. 市值相近（避免流动性差异）
3. 业务模式相似（基本面驱动因素一致）

**案例：中国平安 vs 中国人寿**

```python
# 假设数据已下载
# 使用tushare或akshare获取A股数据
import akshare as ak

def get_ashare_data(stock_code, start_date, end_date):
    """
    获取A股日线数据
    """
    df = ak.stock_zh_a_hist(symbol=stock_code, 
                             start_date=start_date, 
                             end_date=end_date, 
                             adjust="qfq")  # 前复权
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)
    
    return df['收盘']

# 获取中国平安和中国人寿数据
# pa = get_ashare_data('601318', '20200101', '20260619')
# rs = get_ashare_data('601628', '20200101', '20260619')

# 协整检验
# result = engle_granger_test(pa, rs)
```

### 5.2 回测结果分析

**关键指标：**
- 年化收益率
- 夏普比率
- 最大回撤
- 胜率
- 盈亏比

**优化方向：**
1. 调整入场/出场阈值
2. 引入动态对冲比率
3. 加入止损机制
4. 多品种组合

## 六、总结与展望

### 6.1 核心要点

1. **均值回归是统计套利的基石**：理解平稳性和协整是成功的关键
2. **风险管理至关重要**：止损和仓位管理决定了策略的生存能力
3. **动态调整优于静态模型**：卡尔曼滤波等工具可以提升策略表现
4. **交易成本不可忽视**：高频交易容易被交易成本侵蚀利润

### 6.2 实施建议

**对于初学者：**
- 从最经典的配对交易开始（2只股票）
- 使用日线数据，避免高频数据的噪声
- 充分回测后再考虑实盘

**对于进阶者：**
- 扩展到多因子模型
- 引入机器学习预测价差方向
- 构建多策略组合降低风险

### 6.3 未来方向

**高频统计套利：**
- 利用分钟级或秒级数据
- 需要更强大的计算基础设施

**机器学习应用：**
- 使用LSTM预测价差走势
- 强化学习优化交易执行

**跨市场套利：**
- A股 vs 港股
- 股票 vs 期货

---

## 参考文献

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Gatev, E., et al. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis"
4. Chan, E. (2013). "Algorithmic Trading: Winning Strategies and Their Rationale"

## 代码示例仓库

完整代码已上传至GitHub：  
[https://github.com/quant-blog/statistical-arbitrage](https://github.com/quant-blog/statistical-arbitrage)

包含：
- 协整检验模块
- 配对交易回测框架
- 动态对冲比率估计
- 风险管理系统

---

**免责声明：** 本文仅供参考，不构成投资建议。统计套利有风险，入市需谨慎。
