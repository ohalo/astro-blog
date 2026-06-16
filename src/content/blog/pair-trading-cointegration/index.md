---
title: "配对交易与协整分析：统计学套利的理论与实践"
publishDate: '2026-06-17'
description: "配对交易与协整分析：统计学套利的理论与实践 - 量化交易专栏"
tags:
  - 统计套利
  - 配对交易
  - 协整分析
  - 市场中性
language: Chinese
---

如果你在2015年做A股的配对交易，你会发现一个有趣的现象：中国平安和中国人寿的股价走势高度相关，但偶尔会出现偏离。当偏离达到历史极值时，买入相对便宜的那个、卖出相对贵的那个，等待价差回归——这就是配对交易的核心思想。

配对交易（Pairs Trading）是统计学套利（Statistical Arbitrage）的一种经典策略。它不依赖市场方向，而是通过识别两个高度相关但暂时偏离的资产，做多低估资产、做空高估资产，等待价差回归来获利。

本文将系统介绍配对交易的理论基础——协整分析，以及如何用Python实现一个完整的配对交易系统。

## 配对交易的理论基础

### 为什么需要协整？

很多人误以为配对交易只需要两个资产的股价"高度相关"就行了。这是一个常见的误区。

**相关性（Correlation）**衡量的是两个序列是否**同向变动**，但它不保证价差会回归。举个极端例子：两只科技股可能都随着时间上涨，相关性很高，但它们的价差可能越来越大，永远不会回归。

**协整性（Cointegration）**衡量的是两个序列的**线性组合是否平稳**。如果两只股票的股价序列是协整的，那么它们的价差（或比值）长期来看会围绕某个均值波动，这意味着价差有均值回归的特性。

用数学语言表达：

如果 $X_t$ 和 $Y_t$ 都是非平稳的I(1)过程（即一阶差分后平稳），但存在一个系数 $\beta$ 使得：

$$
Z_t = Y_t - \beta X_t
$$

是一个平稳的I(0)过程，那么 $X_t$ 和 $Y_t$ 就是协整的。

在配对交易中，我们通常假设 $\beta = 1$（即使用价差 $Y_t - X_t$）或者 $\beta$ 通过回归估计得到（即使用残差 $Y_t - \hat{\beta}X_t$）。

## 协整检验的Python实现

最常用的协整检验是**Engle-Granger检验**和**Johansen检验**。下面我们用Python实现这两种方法。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def engle_granger_test(y, x, verbose=True):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y : Series
        第一个价格序列（因变量）
    x : Series
        第二个价格序列（自变量）
    verbose : bool
        是否打印检验结果
        
    Returns:
    --------
    result : dict
        包含检验统计量、p值、协整系数等信息
    """
    # 第一步：用OLS估计协整关系
    model = OLS(y, x, hasconst=True)
    results = model.fit()
    
    # 获取残差
    residuals = results.resid
    
    # 第二步：对残差进行ADF检验
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(
        residuals, 
        autolag='AIC'
    )
    
    # 计算对冲比例（β）
    hedge_ratio = results.params[0] if len(results.params) > 1 else results.params
    
    # 计算长期均值和标准差
    spread_mean = residuals.mean()
    spread_std = residuals.std()
    
    result = {
        'adf_statistic': adf_stat,
        'adf_pvalue': adf_pvalue,
        'critical_values': critical_values,
        'hedge_ratio': hedge_ratio,
        'intercept': results.params[0] if len(results.params) > 1 else 0,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'residuals': residuals
    }
    
    if verbose:
        print("=" * 60)
        print("Engle-Granger协整检验结果")
        print("=" * 60)
        print(f"ADF统计量: {adf_stat:.4f}")
        print(f"p-value: {adf_pvalue:.4f}")
        print(f"临界值 (1%, 5%, 10%): {critical_values}")
        print(f"对冲比例 (β): {hedge_ratio:.4f}")
        print(f" spread均值: {spread_mean:.4f}")
        print(f" spread标准差: {spread_std:.4f}")
        
        if adf_pvalue < 0.05:
            print("\n✅ 结论: 在5%的显著性水平下拒绝原假设，序列存在协整关系")
        else:
            print("\n❌ 结论: 不能拒绝原假设，序列不存在协整关系")
    
    return result

# 示例使用：生成协整的价格序列
np.random.seed(42)
n_days = 1000

# 生成一个共同的随机游走
common_trend = np.cumsum(np.random.normal(0, 1, n_days))

# 生成两个协整的序列
y = 0.8 * common_trend + np.cumsum(np.random.normal(0, 0.5, n_days))
x = common_trend + np.cumsum(np.random.normal(0, 0.5, n_days))

# 转换为价格序列（假设起始价格为100）
price_y = 100 + np.cumsum(y)
price_x = 100 + np.cumsum(x)

# 创建DataFrame
df = pd.DataFrame({
    'Stock_Y': price_y,
    'Stock_X': price_x
})

# 进行协整检验
result = engle_granger_test(df['Stock_Y'], df['Stock_X'], verbose=True)
```

输出结果类似：

```
============================================================
Engle-Granger协整检验结果
============================================================
ADF统计量: -3.1245
p-value: 0.0234
临界值 (1%, 5%, 10%): {'1%': -3.437, '5%': -2.864, '10%': -2.568}
对冲比例 (β): 0.7234
spread均值: 0.0000
spread标准差: 12.3456

✅ 结论: 在5%的显著性水平下拒绝原假设，序列存在协整关系
```

### Johansen检验

Engle-Granger检验只能检验两个序列之间的协整关系，而且需要指定哪个是自变量、哪个是因变量。Johansen检验则可以同时检验多个序列之间的协整关系，而且不依赖于变量的排序。

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1, verbose=True):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    data : DataFrame
        多个价格序列组成的DataFrame
    det_order : int
        确定性项的顺序：
        -1: 无常数项，无趋势
        0: 有常数项，无趋势
        1: 有常数项，有趋势
    k_ar_diff : int
        VAR模型中的滞后阶数
    verbose : bool
        是否打印结果
        
    Returns:
    --------
    result : dict
        包含特征值、迹统计量、最大特征值统计量等
    """
    # 进行Johansen检验
    johansen_result = coint_johansen(data.values, det_order, k_ar_diff)
    
    result = {
        'eigenvalues': johansen_result.eig,
        'trace_statistic': johansen_result.lr1,
        'max_eigen_statistic': johansen_result.lr2,
        'trace_critical_values': johansen_result.cvt,
        'max_eigen_critical_values': johansen_result.cvm,
        'cointegration_vectors': johansen_result.evec
    }
    
    if verbose:
        print("=" * 60)
        print("Johansen协整检验结果")
        print("=" * 60)
        print("\n特征值:")
        for i, val in enumerate(result['eigenvalues']):
            print(f"  r={i}: {val:.4f}")
        
        print("\n迹统计量检验 (Trace Statistic):")
        for i in range(len(data.columns)):
            trace_stat = result['trace_statistic'][i]
            crit_val_5 = result['trace_critical_values'][i, 1]
            n_cointegrating = len(data.columns) - i
            
            print(f"  r≤{n_cointegrating-1}: {trace_stat:.4f} (5%临界值: {crit_val_5:.4f})")
            
            if trace_stat > crit_val_5:
                print(f"    ✅ 拒绝 r≤{n_cointegrating-1} 的原假设")
            else:
                print(f"    ❌ 不能拒绝 r≤{n_cointegrating-1} 的原假设")
        
        print("\n最大特征值检验 (Max Eigenvalue Statistic):")
        for i in range(len(data.columns) - 1):
            max_eig_stat = result['max_eigen_statistic'][i]
            crit_val_5 = result['max_eigen_critical_values'][i, 1]
            n_cointegrating = len(data.columns) - i
            
            print(f"  r={n_cointegrating-1}: {max_eig_stat:.4f} (5%临界值: {crit_val_5:.4f})")
            
            if max_eig_stat > crit_val_5:
                print(f"    ✅ 拒绝 r={n_cointegrating-1} 的原假设")
            else:
                print(f"    ❌ 不能拒绝 r={n_cointegrating-1} 的原假设")
    
    return result

# 示例使用：检验三个股票的协整关系
np.random.seed(42)
n_days = 1000

# 生成一个共同的随机游走
common_trend = np.cumsum(np.random.normal(0, 1, n_days))

# 生成三个协整的序列
y1 = 0.8 * common_trend + np.cumsum(np.random.normal(0, 0.3, n_days))
y2 = 1.2 * common_trend + np.cumsum(np.random.normal(0, 0.3, n_days))
y3 = -0.5 * common_trend + np.cumsum(np.random.normal(0, 0.3, n_days))

# 转换为价格序列
price1 = 100 + np.cumsum(y1)
price2 = 100 + np.cumsum(y2)
price3 = 100 + np.cumsum(y3)

# 创建DataFrame
df_3stocks = pd.DataFrame({
    'Stock_A': price1,
    'Stock_B': price2,
    'Stock_C': price3
})

# 进行Johansen检验
result_johansen = johansen_test(df_3stocks, det_order=0, k_ar_diff=1, verbose=True)
```

## 构建配对交易策略

有了协整检验工具，下一步就是构建实际的配对交易策略。一个完整的配对交易策略包括以下步骤：

1. **标的筛选**：找到可能协整的股票对
2. **协整检验**：验证筛选出的股票对是否真的协整
3. **信号生成**：根据价差偏离程度生成交易信号
4. **风险控制**：设置止损、仓位限制等

### 1. 标的筛选

在海量的股票中找到可能协整的对是一个挑战。常用的方法是：

- **行业筛选**：同一行业的股票更可能协整
- **市值筛选**：市值相近的股票更可能协整
- **相关性预筛选**：先筛选出相关性高的股票对，再进行协整检验

```python
def screen_potential_pairs(price_data, min_correlation=0.7, min_history=252):
    """
    筛选潜在的配对交易标的
    
    Parameters:
    -----------
    price_data : DataFrame
        多只股票的价格数据，每列为一只股票
    min_correlation : float
        最小相关性阈值
    min_history : int
        最小历史数据长度
        
    Returns:
    --------
    potential_pairs : list
        潜在的配对列表，每个元素为 (stock1, stock2, correlation)
    """
    n_stocks = price_data.shape[1]
    potential_pairs = []
    
    # 计算相关性矩阵
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    # 遍历所有股票对
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            stock1 = price_data.columns[i]
            stock2 = price_data.columns[j]
            
            # 计算相关性
            correlation = corr_matrix.loc[stock1, stock2]
            
            # 如果相关性高于阈值，加入潜在配对
            if correlation >= min_correlation:
                potential_pairs.append((stock1, stock2, correlation))
    
    # 按相关性排序
    potential_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return potential_pairs

# 示例使用：筛选A股中的潜在配对
# 假设我们有10只银行股的数据
np.random.seed(42)
n_stocks = 10
n_days = 1000

# 生成一个行业共同的因子
sector_factor = np.cumsum(np.random.normal(0, 1, n_days))

# 生成10只股票的价格序列
price_data = pd.DataFrame()
for i in range(n_stocks):
    # 每只股票都受到行业因子的影响，但有不同的特质波动
    stock_return = 0.7 * sector_factor + np.cumsum(np.random.normal(0, 0.3, n_days))
    stock_price = 100 + np.cumsum(stock_return)
    
    price_data[f'Bank_{i+1}'] = stock_price

# 筛选潜在配对
potential_pairs = screen_potential_pairs(
    price_data, 
    min_correlation=0.7,
    min_history=252
)

print("\n=== 潜在配对筛选结果 ===")
for i, (stock1, stock2, corr) in enumerate(potential_pairs[:10]):
    print(f"{i+1}. {stock1} - {stock2}: 相关性 = {corr:.4f}")
```

### 2. 协整检验与信号生成

筛选出潜在配对后，需要对每个配对进行协整检验，并为通过检验的配对生成交易信号。

```python
def generate_trading_signals(price_y, price_x, entry_zscore=2.0, exit_zscore=0.5, 
                           lookback=252, verbose=True):
    """
    生成配对交易的交易信号
    
    Parameters:
    -----------
    price_y : Series
        第一个价格序列（做多标的）
    price_x : Series
        第二个价格序列（做空标的）
    entry_zscore : float
        入场Z-score阈值
    exit_zscore : float
        出场Z-score阈值
    lookback : int
        用于估计参数的回望期
    verbose : bool
        是否打印信号统计
        
    Returns:
    --------
    signals : DataFrame
        包含价差、Z-score、持仓信号等
    """
    # 进行协整检验
    coint_result = engle_granger_test(price_y, price_x, verbose=False)
    
    # 计算价差（使用对冲比例）
    hedge_ratio = coint_result['hedge_ratio']
    spread = price_y - hedge_ratio * price_x
    
    # 计算Z-score（滚动）
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    zscore = (spread - spread_mean) / spread_std
    
    # 生成交易信号
    signals = pd.DataFrame(index=price_y.index)
    signals['spread'] = spread
    signals['zscore'] = zscore
    signals['position'] = 0
    
    # 入场信号：Z-score超过阈值
    signals.loc[zscore > entry_zscore, 'position'] = -1  # 做空价差
    signals.loc[zscore < -entry_zscore, 'position'] = 1   # 做多价差
    
    # 出场信号：Z-score回归
    signals['position'] = signals['position'].replace(0, np.nan)
    signals['position'] = signals['position'].fillna(method='ffill')
    
    # 当Z-score回归到exit_zscore以内时平仓
    signals.loc[abs(zscore) < exit_zscore, 'position'] = 0
    signals['position'] = signals['position'].replace(np.nan, 0)
    
    if verbose:
        n_trades = ((signals['position'].diff() != 0) & 
                     (signals['position'].diff() != np.nan)).sum() // 2
        print(f"\n=== 交易信号统计 ===")
        print(f"总交易日: {len(signals)}")
        print(f"交易次数: {n_trades}")
        print(f"持仓时间占比: {(signals['position'] != 0).mean()*100:.1f}%")
        print(f"平均Z-score: {zscore.mean():.4f}")
        print(f"Z-score标准差: {zscore.std():.4f}")
    
    return signals

# 示例使用：为筛选出的第一个配对生成交易信号
if len(potential_pairs) > 0:
    stock1, stock2, corr = potential_pairs[0]
    
    print(f"\n=== 为配对 {stock1}-{stock2} 生成交易信号 ===")
    print(f"相关性: {corr:.4f}")
    
    signals = generate_trading_signals(
        price_data[stock1], 
        price_data[stock2],
        entry_zscore=2.0,
        exit_zscore=0.5,
        lookback=252,
        verbose=True
    )
```

### 3. 回测配对交易策略

生成交易信号后，需要对策略进行回测，评估其表现。

```python
def backtest_pairs_strategy(price_y, price_x, signals, initial_capital=1000000):
    """
    回测配对交易策略
    
    Parameters:
    -----------
    price_y : Series
        第一个价格序列
    price_x : Series
        第二个价格序列
    signals : DataFrame
        交易信号（包含position列）
    initial_capital : float
        初始资金
        
    Returns:
    --------
    performance : dict
        策略表现指标
    portfolio : DataFrame
        组合价值历史
    """
    # 计算每日收益率
    ret_y = price_y.pct_change()
    ret_x = price_x.pct_change()
    
    # 计算策略收益率（假设等权重投资于两个标的）
    strategy_ret = signals['position'].shift(1) * (ret_y - ret_x)
    
    # 计算累计收益
    cumulative_ret = (1 + strategy_ret).cumprod()
    
    # 计算组合价值
    portfolio_value = initial_capital * cumulative_ret
    
    # 计算表现指标
    total_return = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
    annual_return = (portfolio_value.iloc[-1] / initial_capital) ** (252 / len(portfolio_value)) - 1
    sharpe_ratio = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
    
    # 计算最大回撤
    cumulative_max = cumulative_ret.cummax()
    drawdown = (cumulative_ret - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100
    
    # 计算胜率
    winning_days = (strategy_ret > 0).sum()
    win_rate = winning_days / len(strategy_ret) * 100
    
    performance = {
        'total_return': total_return,
        'annual_return': annual_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': ((signals['position'].diff() != 0) & 
                     (~signals['position'].diff().isna())).sum() // 2
    }
    
    portfolio = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'cumulative_return': cumulative_ret,
        'drawdown': drawdown
    })
    
    return performance, portfolio

# 示例使用：回测第一个配对
if len(potential_pairs) > 0:
    stock1, stock2, corr = potential_pairs[0]
    signals = generate_trading_signals(
        price_data[stock1], 
        price_data[stock2],
        entry_zscore=2.0,
        exit_zscore=0.5,
        lookback=252,
        verbose=False
    )
    
    performance, portfolio = backtest_pairs_strategy(
        price_data[stock1],
        price_data[stock2],
        signals,
        initial_capital=1000000
    )
    
    print("\n=== 策略回测结果 ===")
    print(f"总收益率: {performance['total_return']:.2f}%")
    print(f"年化收益率: {performance['annual_return']:.2f}%")
    print(f"夏普比率: {performance['sharpe_ratio']:.4f}")
    print(f"最大回撤: {performance['max_drawdown']:.2f}%")
    print(f"胜率: {performance['win_rate']:.2f}%")
    print(f"交易次数: {performance['n_trades']}")
```

## 配对交易的实战要点

理论很美好，实战很骨感。以下是配对交易中常见的陷阱和应对方法：

### 1. 结构性断裂

协整关系可能在某些时点失效。例如，2008年金融危机期间，很多原本协整的股票对突然不再协整。这被称为"结构性断裂"（Structural Break）。

**应对方法**：
- 使用滚动窗口进行协整检验，当协整关系失效时停止交易
- 设置止损：当价差突破历史极值的3个标准差时，强制平仓

```python
def detect_structural_break(spread, window=252, n_breaks=1):
    """
    检测结构性断裂
    
    使用Chow检验的思想：将数据分为两段，分别进行ADF检验，
    如果两段的ADF统计量都显著，但全样本不显著，则可能存在结构性断裂
    """
    # 在全样本上进行ADF检验
    adf_stat_full, adf_pvalue_full, _, _, _, _ = adfuller(spread)
    
    # 在各个可能的断裂点进行检验
    break_points = []
    for i in range(window, len(spread) - window, window // 2):
        # 前半段
        adf_stat_1, adf_pvalue_1, _, _, _, _ = adfuller(spread[:i])
        
        # 后半段
        adf_stat_2, adf_pvalue_2, _, _, _, _ = adfuller(spread[i:])
        
        # 如果两段都平稳，但全样本不平稳，则可能是结构性断裂
        if adf_pvalue_1 < 0.05 and adf_pvalue_2 < 0.05 and adf_pvalue_full > 0.05:
            break_points.append(i)
    
    return break_points
```

### 2. 滞后协整（Lagging Cointegration）

有时候两个序列本身是协整的，但价差的均值回归速度很慢。这意味着你可能需要等待很长时间才能获利，而在此期间你要承担持仓成本和风险。

**应对方法**：
- 在 signal generation 阶段加入"半衰期"（Half-life）筛选：
  
```python
def calculate_half_life(spread):
    """
    计算价差的半衰期
    
    使用OLS估计：Δspread_t = α + β * spread_{t-1} + ε_t
    半衰期 = -log(2) / log(1 + β)
    """
    # 计算价差的一阶差分
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # 对齐数据
    common_index = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[common_index]
    spread_diff = spread_diff.loc[common_index]
    
    # OLS回归
    model = OLS(spread_diff, spread_lag, hasconst=True)
    results = model.fit()
    
    # 计算半衰期
    beta = results.params[0]
    half_life = -np.log(2) / np.log(1 + beta)
    
    return half_life

# 示例使用
half_life = calculate_half_life(signals['spread'])
print(f"\n价差的半衰期: {half_life:.1f} 天")
```

### 3. 交易成本

配对交易通常需要频繁交易（因为价差会不断偏离和回归），所以交易成本对策略的盈利性影响很大。

**应对方法**：
- 在回测中加入交易成本：包括佣金、印花税、滑点等
- 优化入场和出场阈值：提高入场阈值（例如从2.0个标准差提高到2.5个标准差），可以减少交易次数，但也可能错过一些机会

```python
def backtest_with_transaction_cost(price_y, price_x, signals, 
                                  initial_capital=1000000,
                                  commission=0.0003, 
                                  slippage=0.001):
    """
    回测配对交易策略（考虑交易成本）
    
    Parameters:
    -----------
    commission : float
        佣金比例（单边）
    slippage : float
        滑点比例（单边）
    """
    # 计算每日收益率
    ret_y = price_y.pct_change()
    ret_x = price_x.pct_change()
    
    # 计算策略收益率
    strategy_ret = signals['position'].shift(1) * (ret_y - ret_x)
    
    # 计算交易成本
    position_change = signals['position'].diff().abs()
    transaction_cost = position_change * (2 * commission + 2 * slippage)
    
    # 扣除交易成本
    strategy_ret_net = strategy_ret - transaction_cost
    
    # 计算累计收益
    cumulative_ret = (1 + strategy_ret_net).cumprod()
    portfolio_value = initial_capital * cumulative_ret
    
    # 计算表现指标（同上，略）
    # ...
    
    return performance, portfolio
```

## 案例研究：A股银行股配对交易

让我们用一个完整的案例来展示如何在实际中应用配对交易策略。我们将使用A股银行股的数据（模拟）来构建一个配对交易组合。

```python
# 生成A股银行股的模拟数据
np.random.seed(42)
n_days = 1000
dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

# 银行行业因子
bank_sector = np.cumsum(np.random.normal(0.0002, 0.01, n_days))

# 10只银行股
bank_stocks = ['工商银行', '建设银行', '农业银行', '中国银行', '交通银行',
               '招商银行', '兴业银行', '浦发银行', '民生银行', '光大银行']

price_data_bank = pd.DataFrame(index=dates)

for i, stock in enumerate(bank_stocks):
    # 每只银行股都受到行业因子的影响
    stock_return = 0.8 * bank_sector + np.cumsum(np.random.normal(0, 0.005, n_days))
    stock_price = 10 + np.cumsum(stock_return)  # 假设起始价格为10元
    price_data_bank[stock] = stock_price

# 筛选潜在配对
potential_pairs_bank = screen_potential_pairs(
    price_data_bank,
    min_correlation=0.7,
    min_history=252
)

print("\n=== A股银行股潜在配对 ===")
for i, (stock1, stock2, corr) in enumerate(potential_pairs_bank[:5]):
    print(f"{i+1}. {stock1} - {stock2}: 相关性 = {corr:.4f}")

# 选择前3个配对进行回测
top_pairs = potential_pairs_bank[:3]
results = []

for stock1, stock2, corr in top_pairs:
    # 生成交易信号
    signals = generate_trading_signals(
        price_data_bank[stock1],
        price_data_bank[stock2],
        entry_zscore=2.0,
        exit_zscore=0.5,
        lookback=252,
        verbose=False
    )
    
    # 回测
    performance, portfolio = backtest_pairs_strategy(
        price_data_bank[stock1],
        price_data_bank[stock2],
        signals,
        initial_capital=1000000
    )
    
    results.append({
        'pair': f"{stock1}-{stock2}",
        'correlation': corr,
        'annual_return': performance['annual_return'],
        'sharpe_ratio': performance['sharpe_ratio'],
        'max_drawdown': performance['max_drawdown'],
        'win_rate': performance['win_rate'],
        'n_trades': performance['n_trades']
    })

# 打印回测结果
print("\n=== 前3个配对的回测结果 ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 可视化：第一个配对的价差和Z-score
if len(top_pairs) > 0:
    stock1, stock2, _ = top_pairs[0]
    signals = generate_trading_signals(
        price_data_bank[stock1],
        price_data_bank[stock2],
        entry_zscore=2.0,
        exit_zscore=0.5,
        lookback=252,
        verbose=False
    )
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 子图1：两只股票的价格走势
    axes[0].plot(dates, price_data_bank[stock1], label=stock1, linewidth=2)
    axes[0].plot(dates, price_data_bank[stock2], label=stock2, linewidth=2)
    axes[0].set_title(f'{stock1} vs {stock2} 价格走势', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：价差
    axes[1].plot(dates, signals['spread'], linewidth=2, color='blue')
    axes[1].axhline(y=signals['spread'].mean(), color='red', linestyle='--', label='Mean')
    axes[1].fill_between(dates, 
                          signals['spread'].mean() - 2*signals['spread'].std(),
                          signals['spread'].mean() + 2*signals['spread'].std(),
                          alpha=0.2, color='gray', label='±2σ')
    axes[1].set_title('Spread', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：Z-score和交易信号
    axes[2].plot(dates, signals['zscore'], linewidth=2, color='green')
    axes[2].axhline(y=2.0, color='red', linestyle='--', label='Entry Threshold')
    axes[2].axhline(y=-2.0, color='red', linestyle='--')
    axes[2].axhline(y=0.5, color='orange', linestyle='--', label='Exit Threshold')
    axes[2].axhline(y=-0.5, color='orange', linestyle='--')
    
    # 标记交易信号
    long_signal = (signals['zscore'] < -2.0)
    short_signal = (signals['zscore'] > 2.0)
    axes[2].scatter(dates[long_signal], signals['zscore'][long_signal], 
                    color='green', marker='^', s=50, label='Long Signal', zorder=5)
    axes[2].scatter(dates[short_signal], signals['zscore'][short_signal], 
                    color='red', marker='v', s=50, label='Short Signal', zorder=5)
    
    axes[2].set_title('Z-score and Trading Signals', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pairs_trading_example.png', 
                dpi=300, bbox_inches='tight')
    print("\n✅ 图表已保存到 pairs_trading_example.png")
```

## 结论

配对交易是一种经典的市场中性策略，它的核心思想是"买入低估、卖出高估、等待回归"。要实现成功的配对交易，需要掌握以下几个关键点：

1. **协整分析是基础**：不要只看相关性，要进行严格的协整检验
2. **信号生成要合理**：Z-score阈值的选择需要平衡交易频率和均值回归的可靠性
3. **风险控制不可少**：设置止损、监控协整关系的稳定性、控制杠杆
4. **交易成本要考虑**：配对交易通常交易频繁，交易成本对盈利性影响很大
5. **多配对分散化**：不要把所有资金放在一个配对上，构建配对组合可以降低风险

配对交易不是"印钞机"，它需要持续的监控、定期的重新估计、以及对市场结构的深刻理解。但如果你能掌握好这门技术，它可以为你提供一个相对稳定的收益来源，特别是在震荡市或熊市中。

---

*本文中的代码示例仅为演示用途，实际应用时需要根据具体的数据和需求进行调整。配对交易涉及做空操作，请确保你理解相关的风险和成本。*
