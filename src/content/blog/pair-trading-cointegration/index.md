---
title: '配对交易与协整分析'
description: '深入探讨配对交易策略的核心原理——协整关系，学习如何用统计方法识别配对标的，构建市场中性策略，并附完整Python实现代码。'
pubDate: 2026-06-16
tags: ['量化交易', '统计套利', '协整']
difficulty: '进阶'
---

# 配对交易与协整分析：构建市场中性策略的统计基础

## 引言

配对交易（Pairs Trading）是经典的**市场中性**策略，属于统计套利的一种。核心思想：找到两个价格长期均衡的资产，当价差偏离时做多低估资产、做空高估资产，等待均值回归获利。

**关键问题**：如何科学选择配对标的？答案：**协整分析**（Cointegration Analysis）。

---

## 一、配对交易的核心逻辑

### 1.1 基本思想

配对交易的本质是**均值回归**（Mean Reversion）。假设有两只股票A和B，它们的价格走势长期保持相对稳定的关系。例如：
- **可口可乐（KO）和百事可乐（PEP）**：同属饮料行业，面临相似的原材料成本、消费需求和市场环境
- **沃尔玛（WMT）和塔吉特（TGT）**：同属零售行业，受消费周期影响相似
- **中石油和中石化**：同属能源行业，油价波动对两者影响一致

这些公司处于同一行业，面临相似的市场环境，因此它们的股价往往同向变动。但短期内，由于市场情绪、突发新闻、财报差异等因素，它们的相对价格可能出现偏离。

**交易信号生成机制**：
- 当价差 **高于** 历史均值 + N倍标准差 → 说明A相对B高估，做空A，做多B
- 当价差 **低于** 历史均值 - N倍标准差 → 说明A相对B低估，做多A，做空B
- 当价差 **回归** 均值附近 → 平仓获利，完成一次套利

这种策略的优势在于**市场中性**：无论大盘涨跌，只要配对标的的相对价格回归，就能盈利。它不依赖于市场方向，而是依赖于**相对价格的关系**。

### 1.2 为什么需要协整？

初学者常犯的一个致命错误是：**用相关系数选择配对标的**。

这里有一个重要的概念区分：
- **相关系数**衡量的是**收益率**的相关性（两个资产每天涨跌幅的相关性）
- **协整关系**衡量的是**价格**的长期均衡关系（两个资产价格序列的线性组合是否平稳）

两者有本质区别：

**反例说明**：
假设股票A和B的收益率相关系数高达0.9（高度相关），但它们的价格走势是这样的：
- A: 100 → 110 → 121 → 133 (每天涨10%)
- B: 100 → 109 → 119 → 130 (每天涨9%)

收益率高度相关（同涨同跌），但价格比值是 1.0 → 1.009 → 1.017 → 1.023，**持续扩大**，不存在均值回归。如果进行配对交易，会持续亏损。

**协整关系的直观理解**：
如果两个资产价格存在协整关系，那么它们之间的**价差（或比值）是平稳的**，即：
- 价差的均值不随时间变化
- 价差的波动围绕均值上下摆动
- 出现大的偏离后，会回归到均值附近

这才是配对交易能够盈利的统计基础。

---

## 二、协整理论的数学基础

### 2.1 平稳性（Stationarity）

在理解协整之前，必须先理解**平稳性**这个概念。

**定义**：一个时间序列 {Y_t} 是平稳的，如果满足以下三个条件：
1. **均值恒定**：E(Y_t) = μ（常数，不随时间变化）
2. **方差恒定**：Var(Y_t) = σ²（常数，不随时间变化）
3. **自协方差只依赖于时滞**：Cov(Y_t, Y_{t-k}) = γ_k（只与间隔k有关，与时间点t无关）

**直观理解**：
平稳序列的统计特性不随时间变化，它有**向均值回归**的趋势。这是配对交易能够盈利的核心假设。

**非平稳序列的问题**：
- 股价序列通常是**非平稳的**（有趋势、有单位根、均值和方差随时间变化）
- 对非平稳序列进行回归，可能产生**伪回归**（Spurious Regression）：即使两个序列毫无经济关系，回归结果也会显示"显著相关"
- 伪回归会导致错误的交易信号，造成实盘亏损

### 2.2 协整的定义

**正式定义**：如果两个或多个非平稳时间序列的某个线性组合是平稳的，那么这些序列之间存在**协整关系**。

数学表达：
对于两个非平稳序列 {X_t} 和 {Y_t}（它们都是I(1)过程，即一阶单整），如果存在参数 α 和 β，使得：

```
Z_t = Y_t - (α + βX_t)  ~ I(0)  （平稳）
```

则称 X_t 和 Y_t 是协整的。其中：
- α 是截距项
- β 是**对冲比率**（Hedge Ratio），表示Y对X的敏感度
- Z_t 是**残差序列**（即价差）

**直观理解**：
- X_t 和 Y_t 各自是随机游走（非平稳），长期趋势不确定
- 但它们的**线性组合**（即价差）是平稳的
- 这意味着长期来看，Y_t 和 X_t 之间存在**均衡关系**，不会无限偏离

### 2.3 Engle-Granger 两步法

检验协整关系的经典方法是 **Engle-Granger 两步法**（1987年诺贝尔经济学奖成果）：

**第一步**：用普通最小二乘法（OLS）估计长期均衡关系
```
Y_t = α + βX_t + ε_t
```
其中 ε_t 是残差项。

**第二步**：检验残差项 ε_t 是否平稳
- 使用 **ADF检验**（Augmented Dickey-Fuller Test）
- 原假设 H₀：残差序列有单位根（非平稳）
- 备择假设 H₁：残差序列平稳
- 如果拒绝原假设（p-value < 0.05），则X和Y协整

**Python实现要点**：
- 使用 `statsmodels.tsa.stattools.adfuller` 进行ADF检验
- 注意选择适当的滞后阶数（可使用AIC或BIC准则）
- 临界值使用MacKinnon (1994) 的近似临界值

---

## 三、Python实战：协整检验与配对选择

理论讲完了，现在进入实战环节。我们将用Python：
1. 获取股票数据
2. 进行协整检验
3. 可视化配对关系
4. 构建交易信号

### 3.1 环境准备与数据获取

```python
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 设置中文字体（Mac系统）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 下载股票数据函数
def get_stock_data(tickers, start_date, end_date):
    """
    获取多只股票的历史数据
    
    参数:
    - tickers: 股票代码列表，如 ['KO', 'PEP']
    - start_date: 开始日期，如 '2020-01-01'
    - end_date: 结束日期，如 '2024-12-31'
    
    返回:
    - DataFrame: 收盘价矩阵
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna()

# 示例：获取可口可乐和百事可乐的数据
tickers = ['KO', 'PEP']
start_date = '2020-01-01'
end_date = '2024-12-31'

prices = get_stock_data(tickers, start_date, end_date)
print(f"数据形状: {prices.shape}")
print(f"时间范围: {prices.index[0]} 到 {prices.index[-1]}")
print(prices.head())
```

**关键要点**：
- 使用**调整后收盘价**（Adjusted Close），它考虑了分红、拆股等因素
- 数据频率选择：日线数据足够，小时线或分钟线会产生过多噪声
- 数据长度：建议至少2-3年，以捕捉完整的市场周期

### 3.2 协整检验完整函数

```python
def test_cointegration(y, x, significance_level=0.05):
    """
    使用Engle-Granger两步法进行协整检验
    
    参数:
    - y: 因变量（数组或Series）
    - x: 自变量（数组或Series）
    - significance_level: 显著性水平（默认0.05）
    
    返回:
    - result: 包含检验结果的字典
    - residuals: 残差序列（即价差）
    """
    # 第一步：OLS回归估计长期均衡关系
    x_with_const = sm.add_constant(x)  # 添加常数项
    model = sm.OLS(y, x_with_const).fit()
    residuals = model.resid
    
    # 第二步：ADF检验残差是否平稳
    adf_result = adfuller(residuals, autolag='AIC')
    
    # 整理结果
    result = {
        'hedge_ratio': model.params[1],  # β（对冲比率）
        'intercept': model.params[0],    # α（截距）
        'residual_mean': residuals.mean(),
        'residual_std': residuals.std(),
        'adf_statistic': adf_result[0],  # ADF统计量
        'p_value': adf_result[1],        # p-value
        'critical_values': adf_result[4], # 临界值（1%, 5%, 10%）
        'is_cointegrated': adf_result[1] < significance_level
    }
    
    return result, residuals

# 使用示例
ko_prices = prices['KO'].values
pep_prices = prices['PEP'].values

result, residuals = test_cointegration(ko_prices, pep_prices)

print("=" * 60)
print("协整检验结果：可口可乐 vs 百事可乐")
print("=" * 60)
print(f"对冲比率 (β): {result['hedge_ratio']:.4f}")
print(f"截距 (α): {result['intercept']:.4f}")
print(f"残差均值: {result['residual_mean']:.4f}")
print(f"残差标准差: {result['residual_std']:.4f}")
print(f"ADF统计量: {result['adf_statistic']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"临界值 (1%): {result['critical_values']['1%']:.4f}")
print(f"临界值 (5%): {result['critical_values']['5%']:.4f}")
print(f"临界值 (10%): {result['critical_values']['10%']:.4f}")
print(f"是否协整: {'是' if result['is_cointegrated'] else '否'}")
```

**结果解读**：
- 如果 p-value < 0.05，拒绝原假设，认为残差平稳，两资产协整
- 对冲比率 β 表示：X变动1单位，Y平均变动β单位
- 交易时，每做多1股Y，应做空β股X，实现**市场中性**

### 3.3 可视化配对关系

数据可视化能够帮助我们直观理解配对关系：

```python
def plot_pair_relationship(prices, ticker1, ticker2, result, residuals):
    """
    可视化配对标的的价格走势和价差特征
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1：原始价格走势
    ax1 = axes[0]
    ax1.plot(prices.index, prices[ticker1], label=ticker1, linewidth=2.5, color='#2E86AB')
    ax1.plot(prices.index, prices[ticker2], label=ticker2, linewidth=2.5, color='#A23B72')
    ax1.set_title(f'{ticker1} vs {ticker2} - 价格走势对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格 ($)')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 图2：价差（残差）序列
    ax2 = axes[1]
    ax2.plot(prices.index, residuals, color='#F18F01', linewidth=2)
    ax2.axhline(y=result['residual_mean'], color='gray', linestyle='--', 
                linewidth=1.5, label=f"均值 ({result['residual_mean']:.2f})")
    ax2.axhline(y=result['residual_mean'] + 2*result['residual_std'], 
                color='red', linestyle='--', linewidth=1.5, label='+2σ')
    ax2.axhline(y=result['residual_mean'] - 2*result['residual_std'], 
                color='green', linestyle='--', linewidth=1.5, label='-2σ')
    ax2.fill_between(prices.index, 
                     result['residual_mean'] - 2*result['residual_std'],
                     result['residual_mean'] + 2*result['residual_std'],
                     alpha=0.15, color='gray', label='交易区间')
    ax2.set_title('价差（残差）序列 - 均值回归特性', fontsize=14, fontweight='bold')
    ax2.set_ylabel('价差')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pair_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 配对分析图已保存")
    plt.show()

# 绘制图表
plot_pair_relationship(prices, 'KO', 'PEP', result, residuals)
```

**图表解读**：
- **上图**：两只股票价格走势高度同步，说明基本面相关
- **中图**：价差围绕均值摆动，且大部分时间落在±2σ区间内，符合均值回归特征

---

## 四、构建配对交易策略

有了协整关系，我们就可以构建具体的交易策略了。策略的核心是将统计关系转化为交易信号。

### 4.1 交易信号生成

最常用的信号生成方法是基于**Z-Score**（标准差倍数）：

```python
def generate_trading_signals(residuals, entry_z=2.0, exit_z=0.5):
    """
    基于价差的z-score生成交易信号
    
    参数:
    - residuals: 残差序列（即价差）
    - entry_z: 入场z-score阈值（默认2.0，表示±2倍标准差）
    - exit_z: 出场z-score阈值（默认0.5，表示回归到±0.5倍标准差时平仓）
    
    返回:
    - signals: 交易信号序列 
              1  = 做多价差（做多Y，做空X）
              -1 = 做空价差（做空Y，做多X）
              0  = 平仓/无持仓
    - z_score: 价差的z-score序列
    """
    # 计算z-score
    z_score = (residuals - residuals.mean()) / residuals.std()
    
    # 初始化信号序列和持仓状态
    signals = pd.Series(0, index=residuals.index)
    position = 0  # 当前持仓：0=无持仓，1=做多价差，-1=做空价差
    
    for i in range(len(z_score)):
        if position == 0:  # 无持仓状态
            if z_score.iloc[i] < -entry_z:
                # 价差过低（Y相对X低估），做多价差
                signals.iloc[i] = 1
                position = 1
            elif z_score.iloc[i] > entry_z:
                # 价差过高（Y相对X高估），做空价差
                signals.iloc[i] = -1
                position = -1
        else:  # 有持仓状态
            if abs(z_score.iloc[i]) < exit_z:
                # 价差回归到均值附近，平仓
                signals.iloc[i] = 0
                position = 0
            else:
                # 继续持有
                signals.iloc[i] = position
    
    return signals, z_score

# 生成信号
signals, z_score = generate_trading_signals(residuals, entry_z=2.0, exit_z=0.5)

print("交易信号统计：")
print(f"做多信号次数: {(signals == 1).sum()}")
print(f"做空信号次数: {(signals == -1).sum()}")
```

**参数选择建议**：
- `entry_z = 2.0`：平衡交易频率和胜率。太小会频繁交易（成本高），太大会错过机会
- `exit_z = 0.5`：不要等到完全回归均值才平仓，可以提前止盈
- 可以根据历史数据优化这些参数（但要小心过拟合）

### 4.2 策略回测

```python
def backtest_strategy(data, signals, hedge_ratio):
    """
    回测配对交易策略
    
    参数:
    - data: 价格数据DataFrame
    - signals: 交易信号序列
    - hedge_ratio: 对冲比率
    
    返回:
    - returns: 策略每日收益率
    - cumulative: 累计收益
    """
    # 计算每日收益率
    returns_1 = data['KO'].pct_change()
    returns_2 = data['PEP'].pct_change()
    
    # 策略收益 = 信号 * (做多资产收益 - 对冲比率 * 做空资产收益)
    strategy_returns = pd.Series(0.0, index=data.index)
    
    for i in range(1, len(signals)):
        if signals.iloc[i-1] != 0:  # 有持仓
            if signals.iloc[i-1] == 1:  # 做多KO，做空PEP
                strategy_returns.iloc[i] = (returns_1.iloc[i] - 
                                           hedge_ratio * returns_2.iloc[i])
            else:  # 做空KO，做多PEP
                strategy_returns.iloc[i] = -(returns_1.iloc[i] - 
                                            hedge_ratio * returns_2.iloc[i])
    
    # 累计收益
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    return strategy_returns, cumulative_returns

# 执行回测
strategy_returns, cumulative_returns = backtest_strategy(
    prices, signals, result['hedge_ratio']
)

# 计算绩效指标
def calculate_metrics(returns):
    """计算策略绩效指标"""
    # 去除NaN
    returns = returns.dropna()
    
    # 累计收益
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    # 夏普比率（假设无风险利率为0）
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    return {
        '总收益率 (%)': f'{total_return:.2f}%',
        '夏普比率': f'{sharpe_ratio:.2f}',
        '最大回撤 (%)': f'{max_drawdown:.2f}%'
    }

metrics = calculate_metrics(strategy_returns)
print("=" * 60)
print("策略绩效评估")
print("=" * 60)
for key, value in metrics.items():
    print(f"{key}: {value}")
```

---

## 五、实战中的注意事项

### 5.1 协整关系的稳定性

协整关系不是永恒的，可能因行业变化、公司基本面变化、宏观环境变化而失效。

**解决方案**：
1. **滚动窗口检验**：定期重新检验协整关系
2. **设置止损**：价差持续扩大及时止损
3. **多元化配对**：分散风险

### 5.2 交易成本与滑点

配对交易通常交易频繁（均值回归策略会多次进出场），交易成本对收益影响显著。

**需要考虑的成本**：
- **佣金**（Commission）：按交易金额或笔数收取
- **买卖价差**（Bid-Ask Spread）：特别是小盘股，价差可能很大
- **滑点**（Slippage）：大单交易时，实际成交价偏离下单价
- **做空成本**：借券费用、股息补偿（做空需要支付分红给借出方）

**优化建议**：
- 选择**流动性好**的标的（大盘股、ETF）
- 设置**合理的入场阈值**（如entry_z=2.5而非2.0），避免频繁交易
- 使用**限价单**（Limit Order）而非市价单（Market Order）
- 对于小资金，优先考虑**ETF配对**（如SPY vs IVV，都是标普500 ETF）

### 5.3 风险管理

#### 5.3.1 仓位管理

```python
def calculate_position_size(account_value, risk_per_trade, entry_price, stop_loss_price):
    """
    基于风险敞口计算仓位大小（固定风险模型）
    
    参数:
    - account_value: 账户总价值
    - risk_per_trade: 单笔交易风险敞口（如0.02表示2%）
    - entry_price: 入场价格
    - stop_loss_price: 止损价格
    
    返回:
    - int: 仓位大小（股数）
    """
    risk_amount = account_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    position_size = risk_amount / price_risk
    
    return int(position_size)

# 示例
account = 100000  # 10万美元账户
risk = 0.02  # 单笔风险2%
entry = 50.0  # 入场价50美元
stop = 48.0  # 止损价48美元

position = calculate_position_size(account, risk, entry, stop)
print(f"建议仓位: {position} 股")
print(f"实际风险金额: ${position * (entry - stop):.2f}")
```

#### 5.3.2 止损策略

配对交易常见的止损方法：

1. **时间止损**：如果持仓超过N天（如20天）价差仍未回归，强制平仓
2. **价格止损**：如果价差继续扩大超过入场时的M倍标准差（如3σ），止损
3. **协整关系失效止损**：定期（如每季度）重新检验协整关系，如果p-value显著上升（>0.1），止损

---

## 六、批量筛选配对标的

在实际应用中，我们往往需要从数百只股票中筛选出协整对。以下是一个批量筛选的框架：

```python
def screen_cointegrated_pairs(tickers, start_date, end_date, p_threshold=0.05):
    """
    批量筛选协整配对
    
    参数:
    - tickers: 股票代码列表
    - start_date: 开始日期
    - end_date: 结束日期
    - p_threshold: p-value阈值（默认0.05）
    
    返回:
    - cointegrated_pairs: 协整配对列表
    """
    # 获取数据
    print("正在下载数据...")
    prices = get_stock_data(tickers, start_date, end_date)
    
    # 存储结果
    cointegrated_pairs = []
    
    # 双重循环检验所有组合
    n = len(tickers)
    total_pairs = n * (n - 1) // 2
    current_pair = 0
    
    for i in range(n):
        for j in range(i+1, n):
            current_pair += 1
            print(f"进度: {current_pair}/{total_pairs} - 检验 {tickers[i]} vs {tickers[j]}", end='\r')
            
            # 协整检验
            y = prices[tickers[i]].values
            x = prices[tickers[j]].values
            
            try:
                t_stat, p_value, crit_values = coint(y, x)
                
                if p_value < p_threshold:
                    # 计算对冲比率
                    x_with_const = sm.add_constant(x)
                    model = sm.OLS(y, x_with_const).fit()
                    
                    cointegrated_pairs.append({
                        'ticker1': tickers[i],
                        'ticker2': tickers[j],
                        'p_value': p_value,
                        't_stat': t_stat,
                        'hedge_ratio': model.params[1],
                        'intercept': model.params[0]
                    })
            except Exception as e:
                print(f"\n检验 {tickers[i]} vs {tickers[j]} 失败: {e}")
    
    print(f"\n筛选完成！找到 {len(cointegrated_pairs)} 个协整配对")
    
    # 按p-value排序（越小越好）
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    return cointegrated_pairs

# 使用示例（以S&P 500部分成分股为例）
# universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
#             'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
# 
# pairs = screen_cointegrated_pairs(universe, '2020-01-01', '2024-12-31')
# 
# # 显示前10个配对
# print("\n前10个协整配对：")
# for i, pair in enumerate(pairs[:10]):
#     print(f"{i+1}. {pair['ticker1']} - {pair['ticker2']} "
#           f"(p-value: {pair['p_value']:.4f}, 对冲比率: {pair['hedge_ratio']:.4f})")
```

**优化建议**：
- 先按行业分类，只在同行业内筛选配对（提高协整概率）
- 使用多线程/多进程加速计算
- 考虑使用距离法（Distance Method）作为协整检验的补充

---

## 七、总结与展望

配对交易是一个经典而优雅的量化策略，它的核心优势在于：
1. **市场中性**：无论大盘涨跌，都有可能盈利
2. **统计基础扎实**：协整理论提供了严谨的数学框架
3. **可扩展性强**：从简单的一对一到复杂的多资产组合

但它也面临挑战：
1. **协整关系可能失效**：需要持续监控和重新检验
2. **交易成本侵蚀收益**：需要优化执行和选择合适的标的
3. **需要有严格的纪律**：不能因为短期亏损就放弃策略

**未来方向**：
- 结合**另类数据**（社交媒体情绪、卫星图像、信用卡数据）改进配对选择
- 用**高频数据**捕捉日内套利机会（适合高频交易团队）
- 将配对交易与**期权策略**结合（如做多价差 + 卖出跨式组合，赚取时间价值）
- 使用**机器学习**优化参数（动态对冲比率、自适应入场阈值）

---

## 参考文献

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.
3. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
4. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797-827.

---

**免责声明**：本文仅供学习交流，不构成投资建议。量化交易有风险，入市需谨慎。历史回测结果不代表未来收益，实际应用前请充分测试和评估风险。
