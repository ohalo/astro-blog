---
title: "配对交易与协整分析：市场中性策略的统计基石"
description: "深入讲解配对交易的理论基础、协整检验方法（Engle-Granger检验、Johansen检验）和实际交易策略构建。包含完整的Python代码示例，从数据获取到实盘执行的完整流程。"
pubDate: 2026-06-16
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "均值回归"]
tag: "量化交易"
difficulty: "进阶"
featured: false
---

# 配对交易与协整分析：市场中性策略的统计基石

## 引言

配对交易（Pairs Trading）是最经典的市场中性策略之一，其核心理念是：寻找两个价格具有长期均衡关系的资产，当价格偏离时做多低估资产、做空高估资产，等待价格回归均衡后获利。本文将深入探讨配对交易的理论基础——协整分析，以及如何构建实际的配对交易策略。

## 配对交易的核心逻辑

### 为什么需要协整？

配对交易的关键在于识别"长期的均衡关系"。简单的相关性分析是不够的：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller

# 生成示例数据
np.random.seed(42)
n = 1000

# 情况1: 协整序列（有长期均衡关系）
x = np.cumsum(np.random.normal(0, 1, n))
y_cointegrated = 0.5 * x + np.random.normal(0, 1, n)

# 情况2: 非协整但高相关序列
z = np.random.normal(0, 1, n)
y_correlated = 0.5 * z + np.random.normal(0, 0.1, n)

# 计算相关性
corr_cointegrated = np.corrcoef(x[100:], y_cointegrated[100:])[0, 1]
corr_noncointegrated = np.corrcoef(z, y_correlated)[0, 1]

print(f"协整序列相关性: {corr_cointegrated:.4f}")
print(f"非协整序列相关性: {corr_noncointegrated:.4f}")

# 协整检验
score, pvalue, _ = coint(x[100:], y_cointegrated[100:])
print(f"\n协整检验 p-value: {pvalue:.4f}")
print(f"结论: {'存在协整关系' if pvalue < 0.05 else '不存在协整关系'}")
```

**关键发现**：高相关性不等于协整！协整要求残差序列平稳（Stationary）。

## 协整检验方法

### 方法一：Engle-Granger 两步法

最经典的协整检验方法：

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def engle_granger_test(y, x, significance=0.05):
    """
    Engle-Granger协整检验
    
    步骤:
    1. 回归: y = α + βx + ε
    2. ADF检验残差ε的平稳性
    """
    # 步骤1: OLS回归
    X = add_constant(x)
    model = OLS(y, X).fit()
    residuals = model.resid
    
    # 步骤2: ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整
    is_cointegrated = p_value < significance
    
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_cointegrated': is_cointegrated,
        'hedge_ratio': model.params[1],
        'intercept': model.params[0],
        'residuals': residuals
    }

# 使用示例
result = engle_granger_test(y_cointegrated[100:], x[100:])
print(f"ADF统计量: {result['adf_statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"对冲比例(β): {result['hedge_ratio']:.4f}")
print(f"是否存在协整: {'是' if result['is_cointegrated'] else '否'}")
```

### 方法二：Johansen 检验

适用于多变量协整关系检验：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    参数:
    - data: DataFrame, 多变量时间序列
    - det_order: 确定性项 (0: 无常数, 1: 有常数, 2: 常数+趋势)
    - k_ar_diff: 滞后阶数
    """
    # 进行Johansen检验
    joh_result = coint_johansen(data, det_order, k_ar_diff)
    
    # 提取结果
    trace_stat = joh_result.lr1  # 迹统计量
    max_stat = joh_result.lr2    # 最大特征值统计量
    trace_crit = joh_result.cvt  # 迹检验临界值
    max_crit = joh_result.cvm    # 最大特征值检验临界值
    
    # 判断协整关系个数
    n_cointegrating = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_crit[i, 1]:  # 5%临界值
            n_cointegrating += 1
    
    return {
        'trace_statistic': trace_stat,
        'max_eigenvalue': max_stat,
        'trace_critical': trace_crit,
        'max_critical': max_crit,
        'n_cointegrating': n_cointegrating,
        'eigenvectors': joh_result.evec
    }

# 多变量示例
data = pd.DataFrame({
    'Asset1': x[100:],
    'Asset2': y_cointegrated[100:],
    'Asset3': 0.3*x[100:] + np.random.normal(0, 1, 900)
})

result = johansen_test(data)
print(f"协整关系个数: {result['n_cointegrating']}")
```

### 方法三：Phillips-Ouliaris 检验

更稳健的协整检验方法：

```python
from statsmodels.tsa.stattools import coint

def phillips_ouliaris_test(y, x, trend='c', method='aeg'):
    """
    Phillips-Ouliaris协整检验
    
    参数:
    - trend: 'c'(常数), 'ct'(常数+趋势), 'ctt'(常数+趋势+二次趋势)
    - method: 'aeg'(ADF类型), 'b'(Banerjee类型)
    """
    # 使用statsmodels的coint函数（实现了P-O检验）
    score, pvalue, _ = coint(y, x, trend=trend, method=method, maxlag=None)
    
    return {
        'test_statistic': score,
        'p_value': pvalue,
        'is_cointegrated': pvalue < 0.05
    }

# 比较不同检验方法
methods = ['aeg', 'b']
for method in methods:
    result = phillips_ouliaris_test(
        y_cointegrated[100:],
        x[100:],
        method=method
    )
    print(f"\nMethod: {method}")
    print(f"Test Statistic: {result['test_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
```

## 实战：构建配对交易策略

### 步骤1：寻找配对

使用统计方法筛选合适的股票对：

```python
import yfinance as yf
from itertools import combinations

def find_cointegrated_pairs(tickers, start='2020-01-01', end='2024-01-01'):
    """
    寻找协整的股票对
    """
    # 下载数据
    print("正在下载数据...")
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    
    # 计算收益率
    returns = data.pct_change().dropna()
    
    # 存储结果
    cointegrated_pairs = []
    
    # 遍历所有组合
    for ticker1, ticker2 in combinations(tickers, 2):
        # Engle-Granger检验
        result = engle_granger_test(
            data[ticker1].values,
            data[ticker2].values
        )
        
        if result['is_cointegrated']:
            # 计算相关性
            correlation = data[ticker1].corr(data[ticker2])
            
            cointegrated_pairs.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'p_value': result['p_value'],
                'hedge_ratio': result['hedge_ratio'],
                'correlation': correlation,
                'residuals': result['residuals']
            })
    
    # 按p-value排序
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    return cointegrated_pairs, data, returns

# 使用示例（需要yfinance库）
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA']
# pairs, price_data, returns_data = find_cointegrated_pairs(tickers)
```

### 步骤2：计算交易信号

基于残差（偏离度）构建交易信号：

```python
def calculate_trading_signals(residuals, entry_z=2.0, exit_z=0.5):
    """
    计算配对交易信号
    
    策略逻辑:
    - 残差 > entry_z: 做空残差（做空y，做多x）
    - 残差 < -entry_z: 做多残差（做多y，做空x）
    - 残差回归到 exit_z 以内: 平仓
    """
    # 标准化残差（计算Z-score）
    z_score = (residuals - residuals.mean()) / residuals.std()
    
    # 初始化信号
    signals = pd.Series(0, index=residuals.index)
    position = 0  # 当前持仓：1为做多残差，-1为做空残差，0为空仓
    
    for i in range(1, len(z_score)):
        if position == 0:  # 空仓
            if z_score.iloc[i] < -entry_z:
                position = 1  # 做多残差
                signals.iloc[i] = 1
            elif z_score.iloc[i] > entry_z:
                position = -1  # 做空残差
                signals.iloc[i] = -1
        else:  # 有持仓
            if position == 1 and abs(z_score.iloc[i]) < exit_z:
                position = 0  # 平仓
                signals.iloc[i] = 0
            elif position == -1 and abs(z_score.iloc[i]) < exit_z:
                position = 0  # 平仓
                signals.iloc[i] = 0
            else:
                signals.iloc[i] = position  # 维持持仓
    
    return signals, z_score

# 可视化信号
def plot_trading_signals(residuals, signals, z_score, entry_z=2.0, exit_z=0.5):
    """可视化交易信号"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1: 残差序列
    axes[0].plot(residuals.index, residuals.values, linewidth=1)
    axes[0].axhline(y=residuals.mean() + entry_z*residuals.std(), 
                    color='r', linestyle='--', label=f'+{entry_z}σ')
    axes[0].axhline(y=residuals.mean() - entry_z*residuals.std(), 
                    color='r', linestyle='--', label=f'-{entry_z}σ')
    axes[0].axhline(y=residuals.mean() + exit_z*residuals.std(), 
                    color='g', linestyle=':', label=f'+{exit_z}σ')
    axes[0].axhline(y=residuals.mean() - exit_z*residuals.std(), 
                    color='g', linestyle=':', label=f'-{exit_z}σ')
    axes[0].set_title('Residuals (Spread)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: Z-score
    axes[1].plot(z_score.index, z_score.values, linewidth=1, label='Z-score')
    axes[1].axhline(y=entry_z, color='r', linestyle='--')
    axes[1].axhline(y=-entry_z, color='r', linestyle='--')
    axes[1].axhline(y=exit_z, color='g', linestyle=':')
    axes[1].axhline(y=-exit_z, color='g', linestyle=':')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1].set_title('Z-Score')
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: 交易信号
    axes[2].plot(signals.index, signals.values, linewidth=2, 
                 marker='o', markersize=3, label='Position')
    axes[2].fill_between(signals.index, 0, signals.values, alpha=0.3)
    axes[2].set_title('Trading Signals (1=Long Spread, -1=Short Spread)')
    axes[2].set_ylim([-1.5, 1.5])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 示例
signals, z_score = calculate_trading_signals(result['residuals'])
plot_trading_signals(result['residuals'], signals, z_score)
```

### 步骤3：回测策略

计算策略收益和评估指标：

```python
def backtest_pairs_strategy(prices1, prices2, signals, hedge_ratio):
    """
    回测配对交易策略
    
    参数:
    - prices1, prices2: 两个资产的价格序列
    - signals: 交易信号（1: 做多残差, -1: 做空残差）
    - hedge_ratio: 对冲比例（β）
    """
    # 计算组合价值
    # 当 signals=1: 做多y，做空β*x
    # 当 signals=-1: 做空y，做多β*x
    portfolio_value = pd.Series(0, index=prices1.index)
    
    for i in range(1, len(signals)):
        if signals.iloc[i] != 0:
            # 计算收益率
            ret1 = prices1.iloc[i] / prices1.iloc[i-1] - 1
            ret2 = prices2.iloc[i] / prices2.iloc[i-1] - 1
            
            # 组合收益
            if signals.iloc[i] == 1:  # 做多残差
                portfolio_ret = ret1 - hedge_ratio * ret2
            else:  # 做空残差
                portfolio_ret = -ret1 + hedge_ratio * ret2
            
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_ret)
        else:
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
    
    # 计算累计收益
    cumulative_returns = (portfolio_value / portfolio_value.iloc[0] - 1)
    
    # 计算评估指标
    total_return = cumulative_returns.iloc[-1]
    n_trades = (signals != signals.shift(1)).sum() // 2
    
    # 年化收益率
    days = (prices1.index[-1] - prices1.index[0]).days
    annual_return = (1 + total_return) ** (365 / days) - 1
    
    # 夏普比率（假设无风险利率为0）
    daily_returns = portfolio_value.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    
    # 最大回撤
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades
    }

# 使用示例
prices1 = pd.Series(x[100:], index=pd.date_range('2020-01-01', periods=900, freq='B'))
prices2 = pd.Series(y_cointegrated[100:], index=prices1.index)

backtest_result = backtest_pairs_strategy(
    prices1, prices2, signals, result['hedge_ratio']
)

print(f"总收益率: {backtest_result['total_return']:.2%}")
print(f"年化收益率: {backtest_result['annual_return']:.2%}")
print(f"夏普比率: {backtest_result['sharpe_ratio']:.2f}")
print(f"最大回撤: {backtest_result['max_drawdown']:.2%}")
print(f"交易次数: {backtest_result['n_trades']}")
```

## 实战案例：A股配对交易

### 数据获取与预处理

```python
import tushare as ts

# 设置tushare token（需要注册获取）
# ts.set_token('your_token_here')

def get_a_share_data(stocks, start='20200101', end='20240101'):
    """
    获取A股数据
    """
    pro = ts.pro_api()
    
    all_data = {}
    for stock in stocks:
        try:
            df = pro.daily(ts_code=stock, start_date=start, end_date=end)
            df = df.sort_values('trade_date')
            df.set_index('trade_date', inplace=True)
            all_data[stock] = df['close']
        except Exception as e:
            print(f"获取 {stock} 数据失败: {e}")
    
    return pd.DataFrame(all_data)

# 示例：金融板块股票
financial_stocks = ['600036.SH', '601398.SH', '601939.SH', '601988.SH']
# data = get_a_share_data(financial_stocks)
```

### 动态对冲比例

实际交易中，对冲比例会随时间变化：

```python
def rolling_hedge_ratio(y, x, window=60):
    """
    滚动计算对冲比例（时变β）
    """
    hedge_ratios = pd.Series(index=y.index[window:], dtype=float)
    
    for i in range(window, len(y)):
        y_window = y.iloc[i-window:i]
        x_window = x.iloc[i-window:i]
        
        X = add_constant(x_window)
        model = OLS(y_window, X).fit()
        hedge_ratios.iloc[i-window] = model.params[1]
    
    return hedge_ratios

# 计算时变对冲比例
rolling_beta = rolling_hedge_ratio(
    data['600036.SH'],
    data['601398.SH'],
    window=60
)

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(rolling_beta.index, rolling_beta.values, linewidth=2)
plt.title('Rolling Hedge Ratio (β) Over Time')
plt.xlabel('Date')
plt.ylabel('Hedge Ratio')
plt.grid(True, alpha=0.3)
plt.show()
```

## 风险管理

### 1. 止损策略

```python
def pairs_stop_loss(residuals, signals, stop_loss_z=3.0):
    """
    配对交易止损策略
    
    当残差超过stop_loss_z时强制平仓
    """
    z_score = (residuals - residuals.mean()) / residuals.std()
    
    adjusted_signals = signals.copy()
    for i in range(len(z_score)):
        if abs(z_score.iloc[i]) > stop_loss_z:
            adjusted_signals.iloc[i] = 0  # 强制平仓
    
    return adjusted_signals

# 应用止损
signals_with_stop = pairs_stop_loss(result['residuals'], signals, stop_loss_z=3.0)
```

### 2. 仓位管理

```python
def kelly_criterion(win_rate, win_loss_ratio):
    """
    凯利公式计算最优仓位
    """
    kelly_f = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0, min(0.25, kelly_f))  # 限制最大仓位25%

# 计算胜率和盈亏比
def calculate_win_rate_and_ratio(returns):
    """计算胜率和盈亏比"""
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    return win_rate, win_loss_ratio

# 示例
daily_returns = backtest_result['cumulative_returns'].pct_change().dropna()
win_rate, wl_ratio = calculate_win_rate_and_ratio(daily_returns)
optimal_f = kelly_criterion(win_rate, wl_ratio)
print(f"凯利最优仓位: {optimal_f:.2%}")
```

## 总结

配对交易是一种经典的市场中性策略，其核心在于：

1. **协整分析**：使用Engle-Granger、Johansen等方法识别长期均衡关系
2. **信号构建**：基于残差的Z-score构建交易信号
3. **风险管理**：设置止损、动态对冲、仓位管理
4. **回测验证**：严格回测并评估策略表现

**注意事项**：
- 协整关系可能随时间断裂（结构断裂）
- 交易成本对策略收益影响显著
- 需要持续监控配对的稳定性

## 参考资料

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Elliott, R. J., et al. (2005). "Pairs trading"
3. Gatev, E., et al. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"

---

**免责声明**：本文仅供学术交流，不构成投资建议。配对交易存在风险，历史表现不代表未来收益。
