---
title: "统计套利：均值回归策略"
description: "深入讲解统计套利和均值回归策略的理论基础、协整检验方法与实践应用。包含配对交易、协整分析、Hurst指数等完整Python代码实现。"
pubDate: 2026-06-23
tags: ["统计套利", "均值回归", "配对交易", "协整分析", "量化策略"]
category: "量化策略"
featured: false
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是量化投资中的重要策略类型，其核心思想是利用资产价格之间的统计关系进行套利。本文将系统介绍均值回归策略的理论基础、协整分析方法，并提供完整的Python实现代码。

## 什么是统计套利？

统计套利是一类基于量化模型的投资策略，通过识别资产价格之间的统计关系（如协整关系、相关性等），在价格偏离常态时建立对冲头寸，待价格回归时获利。

### 主要特征

1. **市场中性**：通常构建多空组合，对冲市场风险
2. **统计驱动**：基于统计模型而非基本面分析
3. **高频交易**：许多统计套利策略属于高频策略
4. **风险管理**：严格的风险控制是成功关键

### 常见类型

- **配对交易（Pairs Trading）**：寻找价格具有协整关系的两只股票
- **多因子统计套利**：基于因子模型的统计套利
- **均值回归策略**：利用价格偏离均值的反转特性
- **机器学习套利**：利用AI模型识别复杂模式

## 理论基础：均值回归与协整

### 1. 均值回归（Mean Reversion）

均值回归是指资产价格在长期内倾向于回归其均衡水平的现象。数学上，可以用**平稳过程**描述：

$$y_t = \mu + \phi(y_{t-1} - \mu) + \epsilon_t$$

其中 $|\phi| < 1$，价格会向均值 $\mu$ 回归。

### 2. 协整（Cointegration）

协整是比相关性更强的统计关系。两个非平稳时间序列 $X_t$ 和 $Y_t$ 如果存在线性组合：

$$Z_t = Y_t - \beta X_t$$

使得 $Z_t$ 成为平稳序列，则称 $X_t$ 和 $Y_t$ 协整。

协整关系是配对交易的理论基础。

### 3. Hurst指数

Hurst指数用于判断时间序列的记忆性：

- $H < 0.5$：均值回归（反持久性）
- $H = 0.5$：随机游走
- $H > 0.5$：趋势持续（持久性）

## Python实战：配对交易策略

下面我们用Python实现一个完整的配对交易策略。

### 数据获取与预处理

```python
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 下载股票数据（示例：选取同行业两只股票）
def download_stock_data(tickers, start_date, end_date):
    """
    下载股票数据
    
    参数:
        tickers: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    
    返回:
        prices: 价格DataFrame
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    return data.dropna()

# 示例：中国平安 vs 中国人寿（同行业配对）
tickers = ['601318.SS', '601628.SS']  # 需要替换为实际可下载的代码
# 或者使用美国股票作为示例
tickers = ['AAPL', 'MSFT']
start_date = '2020-01-01'
end_date = '2024-12-31'

print(f"下载 {tickers} 的数据...")
prices = download_stock_data(tickers, start_date, end_date)
print(f"数据维度: {prices.shape}")
print(f"时间范围: {prices.index[0]} 至 {prices.index[-1]}")
```

### 协整检验

```python
def test_cointegration(price1, price2, verbose=True):
    """
    检验两个价格序列的协整关系
    
    参数:
        price1: 第一个价格序列
        price2: 第二个价格序列
        verbose: 是否打印详细信息
    
    返回:
        coint_result: 协整检验结果
        hedge_ratio: 对冲比率（回归系数）
    """
    # 1. 单位根检验（ADF Test）
    adf1 = adfuller(price1, autolag='AIC')
    adf2 = adfuller(price2, autolag='AIC')
    
    if verbose:
        print("\n=== ADF单位根检验 ===")
        print(f"序列1 ADF统计量: {adf1[0]:.4f}, p-value: {adf1[1]:.4f}")
        print(f"序列2 ADF统计量: {adf2[0]:.4f}, p-value: {adf2[1]:.4f}")
    
    # 2. 协整检验
    coint_result = coint(price1, price2)
    
    if verbose:
        print("\n=== 协整检验 ===")
        print(f"协整统计量: {coint_result[0]:.4f}")
        print(f"p-value: {coint_result[1]:.4f}")
        print(f"临界值 (1%, 5%, 10%): {coint_result[2]}")
    
    # 3. 计算对冲比率（用OLS回归）
    model = OLS(price2.values, price1.values).fit()
    hedge_ratio = model.params[0]
    
    if verbose:
        print(f"\n=== 对冲比率 ===")
        print(f"hedge_ratio (beta): {hedge_ratio:.4f}")
    
    return coint_result, hedge_ratio

# 检验协整关系
coint_result, hedge_ratio = test_cointegration(
    prices[tickers[0]], 
    prices[tickers[1]],
    verbose=True
)
```

### 构建配对交易信号

```python
def construct_pair_trading_signals(prices, ticker1, ticker2, hedge_ratio, window=20, entry_z=2.0, exit_z=0.5):
    """
    构建配对交易信号
    
    参数:
        prices: 价格DataFrame
        ticker1: 第一个股票代码
        ticker2: 第二个股票代码
        hedge_ratio: 对冲比率
        window: 滚动窗口（用于计算Z-score）
        entry_z: 入场Z-score阈值
        exit_z: 出场Z-score阈值
    
    返回:
        signals: 包含价格、价差、Z-score、仓位的DataFrame
    """
    df = pd.DataFrame(index=prices.index)
    df['price1'] = prices[ticker1]
    df['price2'] = prices[ticker2]
    
    # 计算价差（spread）
    df['spread'] = df['price2'] - hedge_ratio * df['price1']
    
    # 计算价差的滚动均值和标准差
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()
    
    # 计算Z-score
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # 生成交易信号
    df['position'] = 0
    
    # 入场信号：Z-score超过阈值
    df.loc[df['z_score'] > entry_z, 'position'] = -1  # 价差高估，做空价差
    df.loc[df['z_score'] < -entry_z, 'position'] = 1   # 价差低估，做多价差
    
    # 出场信号：Z-score回归
    df['position'] = df['position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    df.loc[df['z_score'].abs() < exit_z, 'position'] = 0
    
    # 填充NaN
    df['position'] = df['position'].fillna(0)
    
    return df

# 构建信号
signals = construct_pair_trading_signals(
    prices, 
    tickers[0], 
    tickers[1], 
    hedge_ratio,
    window=20,
    entry_z=2.0,
    exit_z=0.5
)

print("\n=== 交易信号统计 ===")
print(f"总交易次数: {(signals['position'] != signals['position'].shift(1)).sum() // 2}")
print(f"多头持仓期数: {(signals['position'] == 1).sum()}")
print(f"空头持仓期数: {(signals['position'] == -1).sum()}")
```

### 回测配对交易策略

```python
def backtest_pair_trading(signals, initial_capital=100000):
    """
    回测配对交易策略
    
    参数:
        signals: 包含信号的DataFrame
        initial_capital: 初始资金
    
    返回:
        results: 回测结果DataFrame
    """
    results = signals.copy()
    
    # 计算每日收益
    results['return1'] = results['price1'].pct_change()
    results['return2'] = results['price2'].pct_change()
    
    # 计算策略收益（假设等权投资两只股票，根据position调整方向）
    # position = 1: 做多价差（做多price2，做空hedge_ratio*price1）
    # position = -1: 做空价差（做空price2，做多hedge_ratio*price1）
    results['strategy_return'] = 0
    
    # 计算对冲组合的收益
    results['hedge_portfolio_return'] = results['return2'] - hedge_ratio * results['return1']
    
    # 根据仓位计算策略收益
    results['strategy_return'] = results['position'].shift(1) * results['hedge_portfolio_return']
    
    # 计算累积收益
    results['cumulative_return'] = (1 + results['strategy_return']).cumprod()
    
    # 计算回撤
    results['cummax'] = results['cumulative_return'].cummax()
    results['drawdown'] = (results['cummax'] - results['cumulative_return']) / results['cummax']
    
    # 计算性能指标
    total_return = results['cumulative_return'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(results)) - 1
    sharpe_ratio = results['strategy_return'].mean() / results['strategy_return'].std() * np.sqrt(252)
    max_drawdown = results['drawdown'].max()
    
    print("\n=== 回测结果 ===")
    print(f"总收益: {total_return:.2%}")
    print(f"年化收益: {annual_return:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")
    
    return results

# 回测
results = backtest_pair_trading(signals)
```

### 可视化结果

```python
def plot_pair_trading_results(results, ticker1, ticker2):
    """绘制配对交易结果"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. 价格序列
    ax1 = axes[0]
    ax1.plot(results.index, results['price1'], label=ticker1, linewidth=2)
    ax1.plot(results.index, results['price2'], label=ticker2, linewidth=2, alpha=0.7)
    ax1.set_title(f'{ticker1} vs {ticker2} 价格序列', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 价差和Z-score
    ax2 = axes[1]
    ax2.plot(results.index, results['spread'], label='Spread', color='blue', linewidth=2)
    ax2.axhline(y=results['spread_mean'].mean(), color='red', linestyle='--', label='Mean')
    ax2.fill_between(results.index, 
                     results['spread_mean'] + 2*results['spread_std'],
                     results['spread_mean'] - 2*results['spread_std'],
                     alpha=0.2, color='gray', label='±2 STD')
    ax2.set_title('价差序列', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-score和交易信号
    ax3 = axes[2]
    ax3.plot(results.index, results['z_score'], label='Z-score', color='purple', linewidth=2)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
    ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Entry (-2)')
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Exit (+0.5)')
    ax3.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5, label='Exit (-0.5)')
    
    # 标记交易信号
    long_signals = results[results['position'] == 1].index
    short_signals = results[results['position'] == -1].index
    ax3.scatter(long_signals, results.loc[long_signals, 'z_score'], 
               color='green', marker='^', s=100, label='Long', zorder=5)
    ax3.scatter(short_signals, results.loc[short_signals, 'z_score'], 
               color='red', marker='v', s=100, label='Short', zorder=5)
    
    ax3.set_title('Z-score与交易信号', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Z-score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积收益和回撤
    ax4 = axes[3]
    ax4.plot(results.index, results['cumulative_return'], label='累积收益', 
             color='blue', linewidth=2.5)
    ax4.set_title('策略累积收益', fontsize=14, fontweight='bold')
    ax4.set_xlabel('日期')
    ax4.set_ylabel('累积收益')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加回撤子图
    ax4_twin = ax4.twinx()
    ax4_twin.fill_between(results.index, 0, results['drawdown'], 
                          alpha=0.3, color='red', label='回撤')
    ax4_twin.set_ylabel('回撤', color='red')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('pair_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制结果
plot_pair_trading_results(results, tickers[0], tickers[1])
```

## Hurst指数：判断均值回归特性

```python
def calculate_hurst_exponent(price_series, max_lag=100):
    """
    计算Hurst指数
    
    参数:
        price_series: 价格序列
        max_lag: 最大滞后阶数
    
    返回:
        hurst: Hurst指数
    """
    # 转换为对数价格
    log_price = np.log(price_series)
    
    # 计算不同时间尺度下的波动范围
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        # 计算价格变化的累计和
        y = log_price[lag:] - log_price[:-lag]
        
        # 计算波动范围（标准差）
        tau.append(np.std(y))
    
    # 拟合log(lag) vs log(tau)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Hurst指数 = 斜率
    hurst = poly[0]
    
    return hurst

# 计算价差的Hurst指数
hurst = calculate_hurst_exponent(results['spread'].dropna())
print(f"\n=== Hurst指数 ===")
print(f"Spread Hurst指数: {hurst:.4f}")
if hurst < 0.5:
    print("结论: 均值回归特性（反持久性）")
elif hurst > 0.5:
    print("结论: 趋势持续性")
else:
    print("结论: 随机游走")

# 可视化Hurst指数
fig, ax = plt.subplots(figsize=(10, 6))
lags = range(2, 100)
tau = []
log_price = np.log(results['spread'].dropna())

for lag in lags:
    y = log_price[lag:] - log_price[:-lag]
    tau.append(np.std(y))

ax.scatter(np.log(lags), np.log(tau), color='blue', alpha=0.6, s=50)
poly = np.polyfit(np.log(lags), np.log(tau), 1)
ax.plot(np.log(lags), np.polyval(poly, np.log(lags)), 
        color='red', linewidth=2, label=f'Hurst = {poly[0]:.4f}')
ax.set_xlabel('log(Lag)', fontsize=12)
ax.set_ylabel('log(Fluctuation)', fontsize=12)
ax.set_title('Hurst Exponent Estimation', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hurst_exponent.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 实战中的关键问题

### 1. 配对选择

选择合适的配对是成功的关键：

- **行业匹配**：同行业股票更可能具有协整关系
- **市值相近**：避免流动性差异过大
- **基本面相似**：业务模式、财务结构相似
- **相关性筛选**：先筛选高相关性的股票对

### 2. 参数优化

关键参数需要谨慎优化：

- **滚动窗口**：20-60个交易日
- **入场阈值**：Z-score 1.5-2.5
- **出场阈值**：Z-score 0-1.0

**陷阱**：过度优化会导致过拟合，务必进行样本外测试。

### 3. 风险控制

统计套利并非无风险：

- **配对失效**：协整关系可能断裂
- **流动性风险**：价差扩大时可能难以平仓
- **执行风险**：交易成本、滑点

**应对方法**：
- 设置止损（如Z-score超过3.5强制平仓）
- 限制单个配对的最大敞口
- 定期重新检验协整关系

### 4. 多维统计套利

单一配对交易容量有限，可以扩展为多维策略：

- **多配对组合**：同时交易多个配对
- **主成分分析（PCA）**：提取行业主因子
- **机器学习**：用随机森林、LSTM预测价差

## 进阶主题

### 1. 时变协整

传统协整假设关系恒定，但现实中协整关系可能时变：

- **状态切换协整**：马尔可夫切换模型
- **滚动协整检验**：定期重新检验

### 2. 高频统计套利

将统计套利应用到更高频率：

- **分钟级配对交易**
- **盘口套利**：利用订单簿不平衡

### 3. 跨资产统计套利

扩展到不同资产类别：

- **股票-ETF套利**
- **期货-现货套利**
- **跨市场套利**

## 总结

统计套利和均值回归策略为量化投资提供了丰富的工具箱。成功的统计套利需要：

1. **扎实的统计学基础**：理解协整、平稳性等概念
2. **严谨的回测**：避免前视偏差、过拟合
3. **有效的风险控制**：配对失效、流动性风险
4. **持续的监控**：市场结构变化会导致策略失效

随着市场有效性提升，简单的统计套利策略alpha逐渐衰减。未来，结合机器学习和高频数据的统计套利将成为主流。

## 参考资料

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Ganapathy, V. (2004). *Statistical Arbitrage and High-Frequency Data*. 
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.
4. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*.

---

**示例代码仓库**: [GitHub链接](#)

**免责声明**: 本文仅供学术交流，不构成投资建议。统计套利涉及复杂的风险管理，实盘应用需谨慎。
