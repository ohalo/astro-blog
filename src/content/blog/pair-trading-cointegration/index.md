---
title: "配对交易与协整分析：市场中性策略的理论与实践"
description: "深入探讨配对交易的核心原理——协整关系，从Engle-Granger检验到Johansen方法，结合Python实战案例，构建稳健的市场中性策略"
pubDate: 2026-06-17
category: "量化策略"
tags: ["配对交易", "协整分析", "市场中性", "统计套利", "量化策略"]
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：市场中性策略的理论与实践

## 引言

在量化投资领域，**配对交易（Pairs Trading）**是最经典的市场中性策略之一。它不依赖市场方向，而是通过捕捉两个高度相关资产的暂时偏离来获取收益。

配对交易的核心在于识别**协整关系（Cointegration）**——两个非平稳时间序列的线性组合是平稳的。本文将深入探讨：

- 协整关系的数学原理
- Engle-Granger检验与Johansen检验
- Python实战：从数据获取到策略回测
- 实战案例分析：A股配对交易
- 风险管理与策略优化

## 协整关系：配对交易的理论基石

### 什么是协整？

两个时间序列 \(X_t\) 和 \(Y_t\) 被称为协整的，如果存在向量 \(\beta\)，使得：

\[
Z_t = Y_t - \beta X_t
\]

是平稳过程（Stationary Process）。

**直观理解**：
- 两只股票的价格都是非平稳的（有趋势、有单位根）
- 但它们的**价差（Spread）**是平稳的
- 价差会在均值附近波动，不会无限扩大

### 为什么要协整，而不是简单相关？

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# 生成示例数据
np.random.seed(42)
n = 1000

# 情况1：简单相关但非协整
x1 = np.cumsum(np.random.randn(n))  # 随机游走
y1 = np.cumsum(np.random.randn(n))  # 另一个随机游走（不相关）

# 情况2：协整关系
x2 = np.cumsum(np.random.randn(n))  # 随机游走
y2 = 1.5 * x2 + np.random.randn(n) * 2  # 与x2协整

# 绘制对比图
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Correlation vs Cointegration', fontsize=16, fontweight='bold')

# 情况1：非协整
axes[0, 0].plot(x1, label='X (Random Walk)', linewidth=2)
axes[0, 0].plot(y1, label='Y (Random Walk)', linewidth=2)
axes[0, 0].set_title(f'Case 1: No Cointegration\nCorrelation: {np.corrcoef(x1, y1)[0,1]:.3f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 情况1的价差
spread1 = y1 - x1
axes[0, 1].plot(spread1, color='red', linewidth=2)
axes[0, 1].axhline(y=spread1.mean(), color='black', linestyle='--', linewidth=1.5)
axes[0, 1].set_title('Spread (Not Mean-Reverting)')
axes[0, 1].grid(True, alpha=0.3)

# 情况1的ADF检验
adf_stat1, p_val1, _, _, critical_values1, _ = adfuller(spread1)
axes[0, 2].text(0.1, 0.5, f'ADF p-value: {p_val1:.3f}\n(Non-Stationary)', 
                  fontsize=12, transform=axes[0, 2].transAxes)
axes[0, 2].axis('off')

# 情况2：协整
axes[1, 0].plot(x2, label='X (Random Walk)', linewidth=2)
axes[1, 0].plot(y2, label='Y (Cointegrated)', linewidth=2)
axes[1, 0].set_title(f'Case 2: Cointegration Exists\nCorrelation: {np.corrcoef(x2, y2)[0,1]:.3f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 情况2的价差
spread2 = y2 - 1.5 * x2
axes[1, 1].plot(spread2, color='green', linewidth=2)
axes[1, 1].axhline(y=spread2.mean(), color='black', linestyle='--', linewidth=1.5)
axes[1, 1].set_title('Spread (Mean-Reverting)')
axes[1, 1].grid(True, alpha=0.3)

# 情况2的ADF检验
adf_stat2, p_val2, _, _, critical_values2, _ = adfuller(spread2)
axes[1, 2].text(0.1, 0.5, f'ADF p-value: {p_val2:.3f}\n(Stationary)', 
                  fontsize=12, transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('cointegration_demo.png', dpi=300, bbox_inches='tight')
```

**关键结论**：
- 高相关性 ≠ 协整关系
- 协整关系的价差是平稳的（均值回归）
- 非协整的价差可能漂移（不回归均值）

## 协整检验方法

### 1. Engle-Granger 两步法

**步骤**：
1. 用OLS估计协整向量：\(\hat{\beta} = (X'X)^{-1}X'Y\)
2. 对残差 \(\hat{Z}_t = Y_t - \hat{\beta}X_t\) 进行ADF检验

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger协整检验
    
    参数:
    - y: 因变量（被解释变量）
    - x: 自变量（解释变量）
    
    返回:
    - coint_vector: 协整向量（截距, 系数）
    - p_value: 协整检验的p值
    - spread: 价差序列
    """
    # 步骤1：OLS回归
    X = add_constant(x)
    model = OLS(y, X).fit()
    beta = model.params[1]  # 斜率
    alpha = model.params[0]  # 截距
    
    # 步骤2：计算价差（残差）
    spread = y - (alpha + beta * x)
    
    # 步骤3：ADF检验残差
    adf_stat, p_value, _, _, critical_values, _ = adfuller(spread, autolag='AIC')
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger Cointegration Test")
        print("=" * 60)
        print(f"Cointegrating Vector: Y = {alpha:.4f} + {beta:.4f} * X")
        print(f"\nADF Test Statistic: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Critical Values:")
        for key, val in critical_values.items():
            print(f"  {key}%: {val:.4f}")
        
        if p_value < 0.05:
            print("\n✅ Conclusion: Cointegration exists (Reject H0)")
        else:
            print("\n❌ Conclusion: No cointegration (Fail to reject H0)")
    
    return {'alpha': alpha, 'beta': beta}, p_value, spread

# 使用示例
# result = engle_granger_test(y_series, x_series)
```

### 2. Johansen 检验（多变量协整）

当有三个或更多变量时，可能存在**多个协整关系**。Johansen检验可以：
- 确定协整关系的个数（r）
- 估计所有协整向量

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

def johansen_test(data, det_order=0, k_ar_diff=1, verbose=True):
    """
    Johansen协整检验（适用于多变量）
    
    参数:
    - data: DataFrame, 多变量时间序列
    - det_order: 确定性项的顺序
                0: 无常数项
                1: 有常数项，无趋势
                -1: 无常数项，无趋势
    - k_ar_diff: VAR模型的滞后阶数
    
    返回:
    - trace_stat: 迹统计量
    - max_eig_stat: 最大特征值统计量
    - coint_vectors: 协整向量矩阵
    """
    # 进行Johansen检验
    result = coint_johansen(data, det_order, k_ar_diff)
    
    if verbose:
        print("=" * 60)
        print("Johansen Cointegration Test")
        print("=" * 60)
        print(f"\nNumber of variables: {data.shape[1]}")
        print(f"Number of observations: {data.shape[0]}")
        
        print("\nTrace Statistic (Null: r <= k):")
        for i in range(len(result.lr1)):
            print(f"  r <= {i}: {result.lr1[i]:.4f}")
        
        print("\nMaximum Eigenvalue Statistic (Null: r = k):")
        for i in range(len(result.lr2)):
            print(f"  r = {i}: {result.lr2[i]:.4f}")
        
        print("\nCritical Values (90%, 95%, 99%):")
        print("Trace Statistic Critical Values:")
        for i in range(len(result.cvt)):
            print(f"  r <= {i}: {result.cvt[i, 0]:.2f}, {result.cvt[i, 1]:.2f}, {result.cvt[i, 2]:.2f}")
        
        # 确定协整关系个数
        coint_rank = 0
        for i in range(len(result.lr1)):
            if result.lr1[i] > result.cvt[i, 1]:  # 95%临界值
                coint_rank += 1
        
        print(f"\n✅ Number of cointegrating relations: {coint_rank}")
    
    return result

# 使用示例
# data = pd.DataFrame({'Stock_A': price_a, 'Stock_B': price_b, 'Stock_C': price_c})
# johansen_result = johansen_test(data)
```

### 3. 实战：选择配对股票

```python
import yfinance as yf
from itertools import combinations

def find_cointegrated_pairs(tickers, start_date, end_date, p_threshold=0.05):
    """
    在股票列表中寻找协整配对的股票
    
    参数:
    - tickers: 股票代码列表
    - start_date, end_date: 日期范围
    - p_threshold: p值阈值（默认0.05）
    
    返回:
    - cointegrated_pairs: 协整配对的列表
    """
    # 下载数据
    print(f"Downloading data for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    cointegrated_pairs = []
    n = len(tickers)
    
    # 两两组合进行检验
    for i in range(n):
        for j in range(i + 1, n):
            stock1 = tickers[i]
            stock2 = tickers[j]
            
            # 去除缺失值
            y = data[stock1].dropna()
            x = data[stock2].dropna()
            
            # 对齐数据
            aligned = pd.concat([y, x], axis=1).dropna()
            if len(aligned) < 252:  # 至少需要1年数据
                continue
            
            y_aligned = aligned[stock1]
            x_aligned = aligned[stock2]
            
            # Engle-Granger检验
            try:
                _, p_value, spread = engle_granger_test(y_aligned, x_aligned, verbose=False)
                
                if p_value < p_threshold:
                    # 计算价差的均值回归速度（Hurst指数）
                    hurst = calculate_hurst_exponent(spread)
                    
                    cointegrated_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'p_value': p_value,
                        'hurst_exponent': hurst,
                        'spread_mean': spread.mean(),
                        'spread_std': spread.std()
                    })
            except Exception as e:
                continue
    
    # 按p值排序（越小越好）
    cointegrated_pairs = sorted(cointegrated_pairs, key=lambda x: x['p_value'])
    
    print(f"\n✅ Found {len(cointegrated_pairs)} cointegrated pairs")
    
    return cointegrated_pairs, data

def calculate_hurst_exponent(series, max_lag=100):
    """
    计算Hurst指数（判断均值回归/趋势/随机游走）
    
    Hurst < 0.5: 均值回归（Mean-Reverting）
    Hurst = 0.5: 随机游走（Random Walk）
    Hurst > 0.5: 趋势（Trending）
    """
    series = series.values if isinstance(series, pd.Series) else series
    series = series - series.mean()  # 去均值
    
    lags = range(2, min(max_lag, len(series) // 2))
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    
    # 拟合 log(lag) vs log(tau)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    hurst = poly[0]
    
    return hurst

# 使用示例
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA']
# pairs, data = find_cointegrated_pairs(tickers, '2020-01-01', '2024-12-31')
```

## 配对交易策略设计

### 1. 信号生成：Z-Score方法

```python
def generate_trading_signals(spread, entry_z=2.0, exit_z=0.5, lookback=252):
    """
    基于Z-Score生成交易信号
    
    参数:
    - spread: 价差序列
    - entry_z: 入场Z值阈值（默认2.0）
    - exit_z: 出场Z值阈值（默认0.5）
    - lookback: 滚动窗口（用于计算均值和标准差）
    
    返回:
    - signals: 交易信号序列
               1: 做多价差（买入低估的，卖出高估的）
               -1: 做空价差
               0: 平仓/不持仓
    """
    signals = pd.Series(0, index=spread.index)
    
    # 滚动计算Z-Score
    spread_mean = spread.rolling(window=lookback, min_periods=lookback//2).mean()
    spread_std = spread.rolling(window=lookback, min_periods=lookback//2).std()
    
    z_score = (spread - spread_mean) / spread_std
    
    # 生成信号
    position = 0  # 当前持仓
    
    for t in range(len(z_score)):
        if pd.isna(z_score.iloc[t]):
            signals.iloc[t] = 0
            continue
        
        if position == 0:  # 无持仓
            if z_score.iloc[t] > entry_z:
                # 价差过高，做空价差
                signals.iloc[t] = -1
                position = -1
            elif z_score.iloc[t] < -entry_z:
                # 价差过低，做多价差
                signals.iloc[t] = 1
                position = 1
        
        elif position == 1:  # 持多仓
            if z_score.iloc[t] <= exit_z:
                # 价差回归，平仓
                signals.iloc[t] = 0
                position = 0
            else:
                # 继续持有
                signals.iloc[t] = 1
        
        elif position == -1:  # 持空仓
            if z_score.iloc[t] >= -exit_z:
                # 价差回归，平仓
                signals.iloc[t] = 0
                position = 0
            else:
                # 继续持有
                signals.iloc[t] = -1
    
    return signals, z_score

# 可视化信号
def plot_trading_signals(spread, signals, z_score, entry_z=2.0, exit_z=0.5):
    """
    绘制交易信号图
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 上图：价差与Z-Score
    ax1 = axes[0]
    ax1.plot(spread.index, spread.values, color='blue', linewidth=1.5, label='Spread')
    ax1.axhline(y=spread.mean(), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between(spread.index, 
                      spread.mean() + entry_z * spread.std(),
                      spread.mean() - entry_z * spread.std(),
                      alpha=0.2, color='gray', label='Entry Zone')
    ax1.set_ylabel('Spread', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 标记交易信号
    long_signals = signals == 1
    short_signals = signals == -1
    
    ax1.scatter(spread.index[long_signals], spread[long_signals], 
                color='green', marker='^', s=100, label='Long', zorder=5)
    ax1.scatter(spread.index[short_signals], spread[short_signals], 
                color='red', marker='v', s=100, label='Short', zorder=5)
    
    # 下图：Z-Score
    ax2 = axes[1]
    ax2.plot(z_score.index, z_score.values, color='purple', linewidth=1.5, label='Z-Score')
    ax2.axhline(y=entry_z, color='red', linestyle='--', linewidth=1.5, label='Entry Threshold')
    ax2.axhline(y=-entry_z, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(y=exit_z, color='green', linestyle='--', linewidth=1.5, label='Exit Threshold')
    ax2.axhline(y=-exit_z, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
```

### 2. 回测框架

```python
class PairsTradingBacktest:
    """
    配对交易回测框架
    """
    
    def __init__(self, price1, price2, coint_vector, transaction_cost=0.001):
        """
        初始化
        
        参数:
        - price1, price2: 两只股票的价格序列
        - coint_vector: 协整向量 {'alpha': alpha, 'beta': beta}
        - transaction_cost: 交易成本（单边，默认0.1%）
        """
        self.price1 = price1
        self.price2 = price2
        self.alpha = coint_vector['alpha']
        self.beta = coint_vector['beta']
        self.tc = transaction_cost
        
        # 计算价差
        self.spread = price2 - (self.alpha + self.beta * price1)
        
    def run_backtest(self, entry_z=2.0, exit_z=0.5, lookback=252):
        """
        运行回测
        
        返回:
        - results: 回测结果字典
        """
        # 生成信号
        signals, z_score = generate_trading_signals(
            self.spread, entry_z, exit_z, lookback
        )
        
        # 初始化投资组合
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['signal'] = signals
        portfolio['z_score'] = z_score
        portfolio['spread'] = self.spread
        
        # 持仓（单位：股）
        portfolio['position1'] = 0  # 股票1的持仓
        portfolio['position2'] = 0  # 股票2的持仓
        
        # 现金和总资产
        portfolio['cash'] = 0.0
        portfolio['total_value'] = 0.0
        
        # 交易成本记录
        portfolio['transaction_cost'] = 0.0
        
        # 回测循环
        initial_capital = 1000000  # 100万
        cash = initial_capital
        position1 = 0
        position2 = 0
        
        for t in range(1, len(portfolio)):
            date = portfolio.index[t]
            prev_date = portfolio.index[t-1]
            
            signal = signals.iloc[t]
            prev_signal = signals.iloc[t-1]
            
            price1_t = self.price1.iloc[t]
            price2_t = self.price2.iloc[t]
            
            # 检测信号变化（交易）
            if signal != prev_signal:
                # 计算交易股数（等市值）
                trade_value = cash * 0.5  # 用一半现金交易
                shares1 = int(trade_value / (self.beta * price1_t))  # 对冲比例
                shares2 = int(trade_value / price2_t)
                
                if signal == 1:  # 做多价差：买入股票2，卖出股票1
                    # 卖出股票1（做空）
                    cash += shares1 * price1_t * (1 - self.tc)
                    position1 -= shares1
                    
                    # 买入股票2
                    cash -= shares2 * price2_t * (1 + self.tc)
                    position2 += shares2
                    
                    portfolio.loc[date, 'transaction_cost'] = (
                        abs(shares1 * price1_t + shares2 * price2_t) * self.tc
                    )
                
                elif signal == -1:  # 做空价差：卖出股票2，买入股票1
                    # 买入股票1
                    cash -= shares1 * price1_t * (1 + self.tc)
                    position1 += shares1
                    
                    # 卖出股票2（做空）
                    cash += shares2 * price2_t * (1 - self.tc)
                    position2 -= shares2
                    
                    portfolio.loc[date, 'transaction_cost'] = (
                        abs(shares1 * price1_t + shares2 * price2_t) * self.tc
                    )
                
                elif signal == 0:  # 平仓
                    # 平掉所有持仓
                    cash += position1 * price1_t * (1 - self.tc) if position1 > 0 else position1 * price1_t * (1 + self.tc)
                    cash += position2 * price2_t * (1 - self.tc) if position2 > 0 else position2 * price2_t * (1 + self.tc)
                    
                    portfolio.loc[date, 'transaction_cost'] = (
                        abs(position1 * price1_t + position2 * price2_t) * self.tc
                    )
                    
                    position1 = 0
                    position2 = 0
            
            # 更新持仓
            portfolio.loc[date, 'position1'] = position1
            portfolio.loc[date, 'position2'] = position2
            portfolio.loc[date, 'cash'] = cash
            
            # 计算总资产
            total = cash
            if position1 != 0:
                total += position1 * price1_t
            if position2 != 0:
                total += position2 * price2_t
            
            portfolio.loc[date, 'total_value'] = total
        
        # 计算收益
        portfolio['returns'] = portfolio['total_value'].pct_change()
        
        # 性能指标
        results = self._calculate_performance(portfolio, initial_capital)
        
        return portfolio, results
    
    def _calculate_performance(self, portfolio, initial_capital):
        """
        计算策略性能指标
        """
        total_value = portfolio['total_value']
        
        # 累计收益
        cumulative_return = (total_value.iloc[-1] / initial_capital) - 1
        
        # 年化收益
        trading_days = len(total_value)
        years = trading_days / 252
        annual_return = (1 + cumulative_return) ** (1 / years) - 1
        
        # 夏普比率
        returns = portfolio['returns'].dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'trading_days': trading_days
        }
        
        return results
    
    def plot_results(self, portfolio):
        """
        绘制回测结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 图1：总资产曲线
        axes[0].plot(portfolio.index, portfolio['total_value'], 
                     color='blue', linewidth=2, label='Portfolio Value')
        axes[0].axhline(y=portfolio['total_value'].iloc[0], 
                         color='red', linestyle='--', linewidth=1.5, label='Initial Capital')
        axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
        axes[0].set_title('Pairs Trading Strategy - Portfolio Value', fontsize=14)
        axes[0].legend()
        axes[0]..grid(True, alpha=0.3)
        
        # 图2：累积收益
        cumulative_returns = (1 + portfolio['returns']).cumprod()
        axes[1].plot(cumulative_returns.index, cumulative_returns.values,
                     color='green', linewidth=2, label='Strategy')
        
        # 基准：买入持有股票1和股票2的平均收益
        benchmark_return = 0.5 * (self.price1 / self.price1.iloc[0] + self.price2 / self.price2.iloc[0])
        axes[1].plot(benchmark_return.index, benchmark_return.values,
                     color='gray', linewidth=1.5, linestyle='--', label='Benchmark (50/50)')
        
        axes[1].set_ylabel('Cumulative Return', fontsize=12)
        axes[1].set_title('Cumulative Returns', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 图3：回撤
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[2].fill_between(drawdown.index, 0, drawdown.values, 
                              color='red', alpha=0.3, label='Drawdown')
        axes[2].plot(drawdown.index, drawdown.values, 
                      color='darkred', linewidth=1, label='Drawdown')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[2].set_ylabel('Drawdown', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_title('Drawdown Chart', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig

# 完整使用示例
# backtest = PairsTradingBacktest(price_a, price_b, coint_vector)
# portfolio, results = backtest.run_backtest(entry_z=2.0, exit_z=0.5)
# backtest.plot_results(portfolio)
```

## 实战案例：A股配对交易

### 案例背景

选择**招商银行（600036.SH）**和**平安银行（000001.SZ）**作为配对标的。

**选择理由**：
1. 同属银行板块，业务模式相似
2. 市值相近，流动性好
3. 历史价格走势高度相关

### 数据获取与协整检验

```python
# 注意：实际使用时需要替换为有效的数据源
# 这里使用模拟数据演示

def demo_pairs_trading():
    """
    A股配对交易完整案例
    """
    # 1. 生成模拟数据（实际应从Tushare、Wind等获取）
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')  # 交易日
    
    n = len(dates)
    
    # 生成协整的价格序列
    x = 10 + np.cumsum(np.random.randn(n) * 0.02)  # 平安银行（假想价格）
    y = 1.2 * x + np.random.randn(n) * 0.5 + 3  # 招商银行（协整）
    
    price1 = pd.Series(y, index=dates, name='600036.SH')  # 招商银行
    price2 = pd.Series(x, index=dates, name='000001.SZ')  # 平安银行
    
    # 2. 协整检验
    print("\n" + "="*60)
    print("Step 1: Cointegration Test")
    print("="*60)
    
    coint_vector, p_value, spread = engle_granger_test(price1, price2)
    
    # 3. 生成交易信号
    print("\n" + "="*60)
    print("Step 2: Generate Trading Signals")
    print("="*60)
    
    signals, z_score = generate_trading_signals(spread, entry_z=2.0, exit_z=0.5)
    
    # 统计信号
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_flat = (signals == 0).sum()
    
    print(f"Long signals: {n_long}")
    print(f"Short signals: {n_short}")
    print(f"Flat signals: {n_flat}")
    
    # 4. 回测
    print("\n" + "="*60)
    print("Step 3: Backtest")
    print("="*60)
    
    backtest = PairsTradingBacktest(price1, price2, coint_vector, transaction_cost=0.001)
    portfolio, results = backtest.run_backtest(entry_z=2.0, exit_z=0.5)
    
    # 5. 输出结果
    print("\n" + "="*60)
    print("Backtest Results")
    print("="*60)
    print(f"Cumulative Return: {results['cumulative_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    
    # 6. 可视化
    fig1 = plot_trading_signals(spread, signals, z_score)
    fig1.savefig('pair_trading_signals.png', dpi=300, bbox_inches='tight')
    
    fig2 = backtest.plot_results(portfolio)
    fig2.savefig('pair_trading_backtest.png', dpi=300, bbox_inches='tight')
    
    print("\n✅ Charts saved: pair_trading_signals.png, pair_trading_backtest.png")
    
    return portfolio, results

# 运行案例
# portfolio, results = demo_pairs_trading()
```

### 实证结果分析

假设回测结果为：

```
============================================================
Backtest Results (2020-2024)
============================================================
Cumulative Return: 45.23%
Annual Return: 9.76%
Sharpe Ratio: 1.85
Max Drawdown: -8.34%
Win Rate: 58.7%
Total Trades: 126
Average Holding Period: 12 days
```

**结果解读**：
1. **年化收益9.76%**：跑赢沪深300（同期约5%）
2. **夏普比率1.85**：风险调整后收益优秀
3. **最大回撤-8.34%**：市场中性策略，回撤较小
4. **胜率58.7%**：略高于随机，说明信号有效

## 策略优化与风险管理

### 1. 动态参数优化

```python
def optimize_parameters(price1, price2, coint_vector, param_grid):
    """
    参数优化：寻找最优的entry_z和exit_z
    
    参数:
    - param_grid: 参数网格 {'entry_z': [1.5, 2.0, 2.5], 'exit_z': [0.5, 1.0]}
    """
    best_sharpe = -np.inf
    best_params = None
    results_list = []
    
    for entry_z in param_grid['entry_z']:
        for exit_z in param_grid['exit_z']:
            # 运行回测
            backtest = PairsTradingBacktest(price1, price2, coint_vector)
            _, results = backtest.run_backtest(entry_z, exit_z)
            
            sharpe = results['sharpe_ratio']
            results_list.append({
                'entry_z': entry_z,
                'exit_z': exit_z,
                'sharpe': sharpe,
                'annual_return': results['annual_return'],
                'max_dd': results['max_drawdown']
            })
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'entry_z': entry_z, 'exit_z': exit_z}
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results_list)
    
    print("\nParameter Optimization Results:")
    print(results_df.sort_values('sharpe', ascending=False).head(10))
    print(f"\n✅ Best Parameters: {best_params}, Sharpe: {best_sharpe:.2f}")
    
    return best_params, results_df
```

### 2. 风险管理

```python
def risk_management_module(portfolio, max_position_size=0.5, stop_loss=-0.05):
    """
    风险管理模块
    
    参数:
    - portfolio: 投资组合DataFrame
    - max_position_size: 最大持仓比例（默认50%）
    - stop_loss: 止损线（默认-5%）
    """
    # 1. 仓位管理
    portfolio['position_value'] = (
        portfolio['position1'] * portfolio.get('price1', 0) +
        portfolio['position2'] * portfolio.get('price2', 0)
    ).abs()
    
    portfolio['position_ratio'] = portfolio['position_value'] / portfolio['total_value']
    
    # 标记超限仓位
    overweight = portfolio['position_ratio'] > max_position_size
    
    # 2. 止损
    cumulative_returns = (1 + portfolio['returns']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    trigger_stop_loss = drawdown < stop_loss
    
    # 3. 生成风控信号
    portfolio['risk_signal'] = 0
    portfolio.loc[overweight, 'risk_signal'] = -1  # 减仓
    portfolio.loc[trigger_stop_loss, 'risk_signal'] = -2  # 止损
    
    # 统计
    n_overweight = overweight.sum()
    n_stop_loss = trigger_stop_loss.sum()
    
    print(f"\nRisk Management Report:")
    print(f"  Overweight alerts: {n_overweight}")
    print(f"  Stop-loss triggers: {n_stop_loss}")
    
    return portfolio
```

### 3. 多配对组合

```python
def multi_pair_portfolio(pairs_list, capital=1000000):
    """
    多配对组合：分散风险
    
    参数:
    - pairs_list: 配对列表 [{'price1': ..., 'price2': ..., 'coint_vector': ...}, ...]
    - capital: 总资金
    """
    n_pairs = len(pairs_list)
    capital_per_pair = capital / n_pairs
    
    all_portfolios = []
    
    for i, pair in enumerate(pairs_list):
        print(f"\nBacktesting Pair {i+1}/{n_pairs}...")
        
        backtest = PairsTradingBacktest(
            pair['price1'], 
            pair['price2'], 
            pair['coint_vector']
        )
        
        portfolio, results = backtest.run_backtest()
        
        # 按资金比例缩放
        portfolio['total_value'] = portfolio['total_value'] * (capital_per_pair / portfolio['total_value'].iloc[0])
        
        all_portfolios.append(portfolio)
    
    # 合并投资组合
    combined_value = sum([p['total_value'] for p in all_portfolios])
    
    # 计算组合性能
    combined_returns = combined_value.pct_change()
    combined_sharpe = np.sqrt(252) * combined_returns.mean() / combined_returns.std()
    
    print(f"\n✅ Multi-Pair Portfolio Sharpe Ratio: {combined_sharpe:.2f}")
    
    return combined_value, all_portfolios
```

## 结论与展望

配对交易是一种经典而有效的市场中性策略。通过本文的探讨，我们了解到：

### 核心要点

1. **协整关系是关键**：不是所有高相关的股票都适合配对交易，必须通过统计检验确认协整关系
2. **参数敏感**：入场/出场阈值对策略性能影响显著，需要动态优化
3. **风险管理至关重要**：仓位管理、止损、多配对分散都是必要的
4. **交易成本影响大**：配对交易频繁交易，低成本券商或ETF更适合

### 未来改进方向

1. **机器学习优化**：
   - 用LSTM预测价差方向
   - 用强化学习动态调整参数

2. **高频配对交易**：
   - 基于分钟级或tick级数据
   - 捕捉更短期的均值回归机会

3. **多因子配对**：
   - 不仅基于价格，还加入基本面因子（PE、PB等）
   - 构建多维度配对评分系统

4. **跨市场配对**：
   - A股与港股配对（如A+H股）
   - 股票与期货配对（期现套利）

---

**实战建议**：
- 先用模拟盘验证策略
- 从小资金开始，逐步放大
- 持续监控协整关系的稳定性（结构突变检验）
- 关注市场制度变化（如注册制改革）对配对关系的影响

**参考资料**：
1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Ganapathy, V. (2004). "Statistical Arbitrage and Pairs Trading"
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis"

*Disclaimer: 本文仅供参考，不构成投资建议。配对交易虽为市场中性策略，但仍存在风险，包括模型风险、执行风险、流动性风险等。*
