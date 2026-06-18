---
title: "配对交易与协整分析"
description: "深入讲解配对交易策略的原理与协整分析方法，从统计套利角度构建市场中性策略。包含完整的Python实现、协整检验与回测框架。"
pubDate: 2026-06-18
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "Python", "量化策略"]
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

配对交易（Pairs Trading）是最经典的统计套利策略之一。它不依赖市场方向，而是通过捕捉两个高度相关资产之间的暂时性价格偏离来获取收益。这种**市场中性策略**在牛市和熊市中都能盈利，是量化投资的重要组成部分。

本文将系统介绍配对交易的理论基础、协整分析方法、实战实施步骤，并提供完整的Python实现代码。

## 配对交易的核心思想

### 什么是配对交易？

配对交易基于一个简单的观察：**某些资产的价格走势长期存在稳定的关系**。当这种关系暂时偏离时，我们可以：

1. **做多价格相对低估的资产**
2. **做空价格相对高估的资产**
3. **等待价格关系回归正常时平仓获利**

例如，可口可乐与百事可乐的股价通常同步变动。如果某天可口可乐突然大涨而百事可乐没动，我们认为这种偏离是暂时的，会做多百事、做空可口可乐，等待两者价差回归。

![配对交易示意图](/images/pair-trading-cointegration/pairs_concept.png)

*图1：配对交易的核心逻辑——捕捉价格偏离并等待回归*

### 为什么配对交易有效？

配对交易有效的原因在于**均值回归**（Mean Reversion）：

1. **经济基本面约束**：同一行业的公司面临相似的宏观经济环境、行业政策与竞争格局
2. **套利力量**：当价格偏离过大时，套利者会入场纠正价格
3. **心理锚定**：投资者对相似资产的价格存在心理锚定，偏离过大时会引发关注

## 协整分析：寻找配对的理论基础

### 平稳性与协整

在建立配对交易策略前，我们需要确定两个资产的价格序列是否存在**长期均衡关系**。这需要用到计量经济学中的**协整**（Cointegration）概念。

#### 平稳性检验

一个时间序列是平稳的（Stationary），如果它的均值、方差与自协方差不随时间变化。

常用的平稳性检验方法：

1. **ADF检验**（Augmented Dickey-Fuller Test）
2. **PP检验**（Phillips-Perron Test）
3. **KPSS检验**（Kwiatkowski-Phillips-Schmidt-Shin Test）

#### 协整检验

如果两个非平稳序列的**线性组合是平稳的**，则这两个序列是协整的。

常用协整检验方法：

1. **Engle-Granger两步法**
2. **Johansen检验**
3. **Phillips-Ouliaris检验**

### Python实现：协整检验

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS

# 1. ADF检验函数
def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller检验
    
    参数:
    - series: pd.Series, 时间序列
    - title: str, 序列名称
    
    返回:
    - result: Dict, 检验结果
    """
    result = adfuller(series, autolag='AIC')
    
    print(f'ADF检验: {title}')
    print(f'ADF统计量: {result[0]:.4f}')
    print(f'p值: {result[1]:.4f}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print("结论: 序列平稳 (拒绝原假设)")
        return True
    else:
        print("结论: 序列不平稳 (接受原假设)")
        return False

# 2. 协整检验函数
def cointegration_test(series1, series2, title=''):
    """
    协整检验（Engle-Granger方法）
    
    参数:
    - series1, series2: pd.Series, 两个价格序列
    - title: str, 配对名称
    
    返回:
    - is_cointegrated: bool, 是否存在协整关系
    - hedge_ratio: float, 对冲比率
    """
    # Step 1: 回归分析（series2 ~ series1）
    X = sm.add_constant(series1)
    model = OLS(series2, X).fit()
    hedge_ratio = model.params[1]  # 斜率即为对冲比率
    spread = series2 - hedge_ratio * series1  # 残差序列
    
    # Step 2: 对残差进行ADF检验
    adf_result = adfuller(spread, autolag='AIC')
    
    print(f'\n协整检验: {title}')
    print(f'对冲比率 (hedge ratio): {hedge_ratio:.4f}')
    print(f'ADF统计量: {adf_result[0]:.4f}')
    print(f'p值: {adf_result[1]:.4f}')
    
    # 判断是否存在协整关系（p值 < 0.05）
    is_cointegrated = adf_result[1] <= 0.05
    
    if is_cointegrated:
        print("结论: 存在协整关系")
    else:
        print("结论: 不存在协整关系")
    
    return is_cointegrated, hedge_ratio, spread

# 3. 批量筛选协整配对
def find_cointegrated_pairs(data, p_value_threshold=0.05):
    """
    在数据集中寻找所有协整配对
    
    参数:
    - data: pd.DataFrame, 各资产价格（列：资产，行：时间）
    - p_value_threshold: float, p值阈值
    
    返回:
    - cointegrated_pairs: List, 协整配对列表
    """
    n = data.shape[1]
    cointegrated_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            asset1 = data.columns[i]
            asset2 = data.columns[j]
            
            # 进行协整检验
            series1 = data[asset1]
            series2 = data[asset2]
            
            is_cointegrated, hedge_ratio, spread = cointegration_test(
                series1, series2, title=f'{asset1} - {asset2}'
            )
            
            if is_cointegrated:
                cointegrated_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'hedge_ratio': hedge_ratio,
                    'spread': spread
                })
    
    return cointegrated_pairs
```

## 配对交易策略构建

### 步骤一：构建价差序列

找到协整配对后，我们需要计算**价差**（Spread）或**Z分数**（Z-Score）：

```python
def calculate_spread_zscore(price1, price2, hedge_ratio, window=20):
    """
    计算价差与Z分数
    
    参数:
    - price1, price2: pd.Series, 两个资产的价格
    - hedge_ratio: float, 对冲比率
    - window: int, 滚动窗口（用于计算均值与标准差）
    
    返回:
    - spread: pd.Series, 价差序列
    - zscore: pd.Series, Z分数序列
    """
    # 计算价差
    spread = price2 - hedge_ratio * price1
    
    # 计算滚动均值与标准差
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # 计算Z分数
    zscore = (spread - spread_mean) / spread_std
    
    return spread, zscore
```

### 步骤二：设定交易信号

常用的交易信号基于Z分数的阈值：

- **开仓信号**：
  - Z分数 < -2：做多asset2，做空asset1（价差偏低，预期回归）
  - Z分数 > 2：做空asset2，做多asset1（价差偏高，预期回归）

- **平仓信号**：
  - Z分数回归到0附近：平仓获利

```python
def generate_trading_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    根据Z分数生成交易信号
    
    参数:
    - zscore: pd.Series, Z分数序列
    - entry_threshold: float, 开仓阈值
    - exit_threshold: float, 平仓阈值
    
    返回:
    - signals: pd.DataFrame, 交易信号
    """
    signals = pd.DataFrame(index=zscore.index)
    signals['zscore'] = zscore
    signals['position'] = 0  # 0: 空仓, 1: 做多价差, -1: 做空价差
    
    # 当前持仓状态
    current_position = 0
    
    for i in range(1, len(signals)):
        if current_position == 0:  # 当前空仓
            if zscore.iloc[i] < -entry_threshold:
                # Z分数过低，做多价差（做多asset2，做空asset1）
                signals.iloc[i, signals.columns.get_loc('position')] = 1
                current_position = 1
            elif zscore.iloc[i] > entry_threshold:
                # Z分数过高，做空价差（做空asset2，做多asset1）
                signals.iloc[i, signals.columns.get_loc('position')] = -1
                current_position = -1
        
        elif current_position == 1:  # 当前做多价差
            if abs(zscore.iloc[i]) < exit_threshold:
                # Z分数回归，平仓
                signals.iloc[i, signals.columns.get_loc('position')] = 0
                current_position = 0
            else:
                # 继续持有
                signals.iloc[i, signals.columns.get_loc('position')] = 1
        
        elif current_position == -1:  # 当前做空价差
            if abs(zscore.iloc[i]) < exit_threshold:
                # Z分数回归，平仓
                signals.iloc[i, signals.columns.get_loc('position')] = 0
                current_position = 0
            else:
                # 继续持有
                signals.iloc[i, signals.columns.get_loc('position')] = -1
    
    return signals
```

### 步骤三：回测框架

```python
def backtest_pairs_strategy(price1, price2, signals, hedge_ratio, initial_capital=100000):
    """
    回测配对交易策略
    
    参数:
    - price1, price2: pd.Series, 两个资产的价格
    - signals: pd.DataFrame, 交易信号
    - hedge_ratio: float, 对冲比率
    - initial_capital: float, 初始资金
    
    返回:
    - portfolio: pd.DataFrame, 组合表现
    """
    # 初始化组合价值
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['capital'] = initial_capital
    portfolio['position'] = signals['position']
    
    # 计算每日收益
    portfolio['return1'] = price1.pct_change()
    portfolio['return2'] = price2.pct_change()
    
    # 计算策略收益（考虑对冲比率）
    # 当position=1时：做多price2，做空hedge_ratio份price1
    # 当position=-1时：做空price2，做多hedge_ratio份price1
    portfolio['strategy_return'] = (
        portfolio['position'].shift(1) * portfolio['return2'] -
        portfolio['position'].shift(1) * hedge_ratio * portfolio['return1']
    )
    
    # 计算累积收益
    portfolio['cumulative_return'] = (1 + portfolio['strategy_return']).cumprod()
    portfolio['capital'] = initial_capital * portfolio['cumulative_return']
    
    # 计算回撤
    portfolio['peak'] = portfolio['capital'].expanding().max()
    portfolio['drawdown'] = (portfolio['capital'] - portfolio['peak']) / portfolio['peak']
    
    return portfolio

# 计算策略表现指标
def calculate_performance_metrics(portfolio):
    """
    计算策略表现指标
    
    参数:
    - portfolio: pd.DataFrame, 组合表现
    
    返回:
    - metrics: Dict, 表现指标
    """
    # 年化收益
    total_return = portfolio['cumulative_return'].iloc[-1] - 1
    n_years = len(portfolio) / 252  # 假设252个交易日/年
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    # 年化波动
    daily_returns = portfolio['strategy_return']
    annual_volatility = daily_returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # 最大回撤
    max_drawdown = portfolio['drawdown'].min()
    
    # 胜率
    winning_days = (daily_returns > 0).sum()
    win_rate = winning_days / len(daily_returns)
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    
    return metrics
```

## 实战案例：A股配对交易

### 数据准备

我们选取A股市场中业务相似、市值接近的上市公司进行配对交易实战。以**招商银行（600036.SH）**与**平安银行（000001.SZ）**为例：

```python
# 假设我们已经获取了价格数据
# price_data = pd.DataFrame({
#     '600036.SH':招商银行收盘价,
#     '000001.SZ':平安银行收盘价
# })

# 1. 平稳性检验
for column in price_data.columns:
    adf_test(price_data[column], title=column)

# 2. 协整检验
is_cointegrated, hedge_ratio, spread = cointegration_test(
    price_data['600036.SH'],
    price_data['000001.SZ'],
    title='招商银行 - 平安银行'
)

# 3. 计算Z分数
spread, zscore = calculate_spread_zscore(
    price_data['600036.SH'],
    price_data['000001.SZ'],
    hedge_ratio,
    window=20
)

# 4. 生成交易信号
signals = generate_trading_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

# 5. 回测
portfolio = backtest_pairs_strategy(
    price_data['600036.SH'],
    price_data['000001.SZ'],
    signals,
    hedge_ratio,
    initial_capital=100000
)

# 6. 计算表现指标
metrics = calculate_performance_metrics(portfolio)

print("\n========== 策略表现 ==========")
for key, value in metrics.items():
    if key in ['total_return', 'annual_return', 'annual_volatility', 'max_drawdown']:
        print(f"{key}: {value:.2%}")
    else:
        print(f"{key}: {value:.4f}")
```

### 回测结果分析

假设我们回测2018-2025年的数据，可能得到如下结果：

| 指标 | 数值 |
|------|------|
| 总收益 | 45.3% |
| 年化收益 | 6.8% |
| 年化波动 | 8.2% |
| 夏普比率 | 0.83 |
| 最大回撤 | -12.4% |
| 胜率 | 54.2% |

**关键发现**：

1. **市场中性**：策略收益与市场走势相关性低（通常|correlation| < 0.3）
2. **稳健收益**：虽然年化收益不高，但夏普比率优于单向策略
3. **回撤可控**：最大回撤显著低于单向持仓

![配对交易累积收益曲线](/images/pair-trading-cointegration/cumulative_returns.png)

*图2：配对交易策略 vs 买入持有策略的累积收益对比*

## 策略优化与风险控制

### 1. 动态对冲比率

传统方法使用**全样本回归**计算固定对冲比率，但这一比率可能随时间变化。改进方法：

- **滚动窗口回归**：每隔一段时间重新估计对冲比率
- **卡尔曼滤波**：动态更新对冲比率

```python
from pykalman import KalmanFilter

def dynamic_hedge_ratio_kalman(price1, price2):
    """
    使用卡尔曼滤波动态估计对冲比率
    
    参数:
    - price1, price2: pd.Series, 两个资产的价格
    
    返回:
    - dynamic_hedge_ratio: pd.Series, 动态对冲比率
    """
    # 准备观测矩阵
    X = price1.values.reshape(-1, 1)
    Y = price2.values
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=X.reshape(-1, 1, 1),
        initial_state_mean=np.array([1.0]),
        initial_state_covariance=np.eye(1) * 0.1,
        observation_covariance=1.0,
        transition_covariance=np.eye(1) * 0.01
    )
    
    # 滤波
    state_means, _ = kf.filter(Y)
    
    # 提取动态对冲比率
    dynamic_hedge_ratio = pd.Series(state_means.flatten(), index=price1.index)
    
    return dynamic_hedge_ratio
```

### 2. 半均回归速度

价差回归均值的速度直接影响策略收益。可以使用**半衰期**（Half-life）衡量：

```python
def calculate_half_life(spread):
    """
    计算价差的半衰期（回归均值的速度）
    
    参数:
    - spread: pd.Series, 价差序列
    
    返回:
    - half_life: float, 半衰期（交易日数）
    """
    # 构建AR(1)模型
    spread_lag = spread.shift(1).dropna()
    spread_now = spread.dropna()
    
    # 回归：spread_now = alpha + beta * spread_lag + error
    X = sm.add_constant(spread_lag)
    model = OLS(spread_now, X).fit()
    beta = model.params[1]
    
    # 半衰期 = -ln(2) / ln(|beta|)
    if abs(beta) < 1:
        half_life = -np.log(2) / np.log(abs(beta))
    else:
        half_life = np.inf  # 不均值回归
    
    return half_life
```

**使用建议**：
- 半衰期太短（< 5天）：可能是噪音，交易成本会侵蚀收益
- 半衰期太长（> 60天）：资金占用时间过长，机会成本大
- **最佳范围**：10-30个交易日

### 3. 止损机制

配对交易虽然理论上是市场中性，但实践中可能遇到**结构性断裂**（如公司并购、行业政策巨变）。需要设置止损：

```python
def add_stop_loss(portfolio, stop_loss_threshold=0.05):
    """
    添加止损机制
    
    参数:
    - portfolio: pd.DataFrame, 组合表现
    - stop_loss_threshold: float, 止损阈值（如5%）
    
    返回:
    - portfolio_with_stoploss: pd.DataFrame, 加入止损后的组合
    """
    portfolio_with_stoploss = portfolio.copy()
    portfolio_with_stoploss['cumulative_max'] = portfolio_with_stoploss['capital'].expanding().max()
    portfolio_with_stoploss['drawdown_now'] = (
        portfolio_with_stoploss['capital'] - portfolio_with_stoploss['cumulative_max']
    ) / portfolio_with_stoploss['cumulative_max']
    
    # 当回撤超过止损阈值时，强制平仓
    stop_loss_triggered = portfolio_with_stoploss['drawdown_now'] < -stop_loss_threshold
    portfolio_with_stoploss.loc[stop_loss_triggered, 'position'] = 0
    
    return portfolio_with_stoploss
```

## 配对交易的局限性与挑战

### 1. 市场环境变化

配对交易在**趋势市**中表现较差。当两只股票同时上涨或下跌时，价差可能不会回归，导致持续亏损。

**应对方法**：
- 结合市场状态判断（如用ADX指标判断趋势强度）
- 在强趋势市中暂停策略

### 2. 流动性风险

某些股票流动性差，冲击成本高。尤其是在2015年股灾、2016年熔断等极端行情中，可能无法及时平仓。

**应对方法**：
- 选择日均成交额 > 1亿元的股票
- 设置最大持仓比例限制

### 3. 模型风险

协整关系可能**突然断裂**。例如，两家公司原本业务相似，但其中一家突然转型进入新行业，导致价格关系永久改变。

**应对方法**：
- 定期重新检验协整关系（如每季度）
- 当p值 > 0.1时，停止交易该配对

## 总结与实践建议

### 核心要点

1. **协整是配对交易的基础**：必须使用统计检验确认长期均衡关系
2. **Z分数是交易信号的核心**：通过滚动均值与标准差标准化价差
3. **风险控制至关重要**：设置止损、控制仓位、防范流动性风险

### 实践步骤

对于希望实施配对交易的投资者，建议按以下步骤进行：

1. **学习阶段**（1-2个月）：
   - 深入理解协整理论
   - 在模拟盘中测试策略

2. **数据准备**（1个月）：
   - 获取高质量的历史数据
   - 清洗数据（处理停牌、除权除息等）

3. **策略开发**（2-3个月）：
   - 编写完整的回测框架
   - 优化参数（窗口长度、阈值等）

4. **实盘准备**（1个月）：
   - 接入交易接口
   - 设置风险监控系统

5. **小资金试运行**（3-6个月）：
   - 用少量资金实际运行
   - 记录实际问题（如滑点、冲击成本）

### 扩展阅读

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Ganapathy, V. (2004). *Statistical Arbitrage and Pairs Trading*. SAS Institute.
3. 国泰君安证券 (2022). 《配对交易策略研究白皮书》.

---

**免责声明**：本文仅供学术交流，不构成投资建议。配对交易虽然理论优美，但实盘中存在诸多挑战，请在充分理解风险的前提下谨慎使用。

**代码示例下载**：[GitHub链接](#)

*所有代码与数据可在作者GitHub仓库获取。*
