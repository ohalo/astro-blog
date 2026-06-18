---
title: "配对交易与协整分析"
date: 2026-06-19
description: "深入讲解配对交易策略的核心原理——协整关系，学习如何寻找协整股票对、构建交易信号、进行风险管理的完整流程。包含实际Python代码和回测案例。"
tags: ["配对交易", "协整分析", "统计套利", "均值回归", "Python"]
image: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

## 引言

在量化交易的世界里，**配对交易（Pairs Trading）** 是一种经典的市场中性策略。它不依赖市场方向，而是通过捕捉两个高度相关资产之间的暂时性价格偏离来获取收益。

配对交易的核心思想是：**找到两只具有长期均衡关系的股票，当它们的价格偏离历史常态时做空高估的、做多低估的，等待价格回归均衡后平仓**。

这种策略的魅力在于：
- ✅ **市场中性**：多空对冲，不受大盘涨跌影响
- ✅ **均值回归**：基于统计学原理，有理论支撑
- ✅ **风险可控**：止损清晰，持仓时间有限

然而，配对交易并非简单的"相关性交易"。真正有效的配对交易必须建立在**协整关系（Cointegration）** 之上。本文将深入探讨协整分析的理论与实践，帮助读者构建稳健的配对交易系统。

## 1. 相关性 ≠ 协整性

### 1.1 相关性的陷阱

很多初学者会误以为：只要两只股票相关系数高，就可以做配对交易。这是一个危险的误区。

**相关系数**衡量的是**收益率**的线性相关性，而**协整性**衡量的是**价格**的长期均衡关系。

**举个例子**：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 模拟两只股票
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')

# 股票A：随机游走
stock_a = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))

# 股票B：与A高度相关，但长期趋势不同
stock_b = 50 + 0.5 * (stock_a - 100) + np.cumsum(np.random.normal(0, 0.5, len(dates)))

# 计算相关系数
corr = np.corrcoef(stock_a, stock_b)[0, 1]
print(f"相关系数: {corr:.4f}")  # 可能很高，比如0.8+

# 绘制价格走势
plt.figure(figsize=(12, 5))
plt.plot(dates, stock_a, label='股票A', linewidth=2)
plt.plot(dates, stock_b * 2, label='股票B (放大2倍)', linewidth=2)  # 放大以便观察
plt.xlabel('日期')
plt.ylabel('价格')
plt.title(f'高度相关但非协整（相关系数={corr:.2f}）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/correlation_not_cointegration.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ 图表已保存: correlation_not_cointegration.png")
```

**问题在哪里？**
- 虽然短期相关性高，但两只股票的价格序列**没有共同的随机趋势**
- 价差的方差会随时间发散（不平稳）
- 均值回归策略会失效

### 1.2 协整关系的定义

**协整关系**的严格定义：

> 如果两个（或多个）非平稳时间序列的某种线性组合是平稳的，那么这些序列之间存在协整关系。

**直观理解**：
- 单只股票价格通常是**非平稳**的（有单位根，随机游走）
- 但如果两只股票的**价差（或对数价差）是平稳的**，它们就是协整的
- 平稳意味着：价差会围绕均值波动，且方差不随时间增加

**数学表达**：

如果 $P_t^A$ 和 $P_t^B$ 都是I(1)过程（一阶单整），且存在参数 $\beta$ 使得：

$$
\varepsilon_t = P_t^A - \beta \cdot P_t^B \sim I(0) \quad (\text{平稳})
$$

则称 $P_t^A$ 和 $P_t^B$ 是协整的。

## 2. 协整检验方法

### 2.1 Engle-Granger 两步法

**步骤1**：用OLS估计协整方程

$$
P_t^A = \alpha + \beta \cdot P_t^B + \varepsilon_t
$$

**步骤2**：对残差 $\varepsilon_t$ 进行单位根检验（ADF检验）

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def engle_granger_test(price_a, price_b, significance_level=0.05):
    """
    Engle-Granger协整检验
    
    参数:
    - price_a: 股票A价格序列
    - price_b: 股票B价格序列
    - significance_level: 显著性水平（默认5%）
    
    返回:
    - is_cointegrated: 是否协整
    - p_value: ADF检验p值
    - beta: 对冲比例
    - intercept: 截距
    - residuals: 残差序列
    """
    # 步骤1: OLS回归
    X = sm.add_constant(price_b)
    model = OLS(price_a, X).fit()
    beta = model.params[1]
    intercept = model.params[0]
    residuals = model.resid
    
    # 步骤2: ADF检验残差
    adf_result = adfuller(residuals, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整（p值小于显著性水平）
    is_cointegrated = p_value < significance_level
    
    return {
        'is_cointegrated': is_cointegrated,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'beta': beta,
        'intercept': intercept,
        'residuals': residuals
    }

# 示例：生成协整价格序列
np.random.seed(456)
n = 1000
dates = pd.date_range('2022-01-01', periods=n, freq='D')

# 共同趋势
common_trend = np.cumsum(np.random.normal(0, 1, n))

# 股票A和B共享趋势，但有各自idiosyncratic冲击
stock_a = 100 + common_trend + np.cumsum(np.random.normal(0, 0.5, n))
stock_b = 50 + 0.6 * common_trend + np.cumsum(np.random.normal(0, 0.3, n))

# 协整检验
result = engle_granger_test(stock_a, stock_b)

print("========== Engle-Granger协整检验 ==========")
print(f"ADF统计量: {result['adf_statistic']:.4f}")
print(f"p值: {result['p_value']:.4f}")
print(f"临界值(5%): {result['critical_values']['5%']:.4f}")
print(f"是否协整: {'是' if result['is_cointegrated'] else '否'}")
print(f"对冲比例(beta): {result['beta']:.4f}")
print(f"截距: {result['intercept']:.4f}")
```

### 2.2 Johansen 检验（多变量协整）

当有**多于两只股票**时（如统计套利组合），需要使用Johansen检验。

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多资产）
    
    参数:
    - price_matrix: DataFrame，每列为一个价格序列
    - det_order: 确定性项顺序（0=无常数项, 1=有常数项）
    - k_ar_diff: 滞后阶数
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 输出结果
    print("========== Johansen协整检验 ==========")
    print(f"迹统计量 (Trace Statistic):")
    for i, stat in enumerate(result.lr1):
        print(f"  r<={i}: {stat:.4f}")
    
    print(f"\n最大特征值统计量 (Max Eigen Statistic):")
    for i, stat in enumerate(result.lr2):
        print(f"  r={i}: {stat:.4f}")
    
    print(f"\n临界值 (5%):")
    print(f"  迹检验: {result.cvt[:, 1]}")
    print(f"  最大特征值检验: {result.cvm[:, 1]}")
    
    return result

# 示例：3只协整股票
stock_c = 30 + 0.4 * common_trend + np.cumsum(np.random.normal(0, 0.4, n))
price_df = pd.DataFrame({'StockA': stock_a, 'StockB': stock_b, 'StockC': stock_c}, index=dates)

# Johansen检验
johansen_result = johansen_test(price_df)
```

## 3. 如何寻找协整股票对？

### 3.1 基本面筛选

协整关系往往源于**相似的商业模式**或**共同的行业驱动因素**：

✅ **高概率协整对**：
- 同一行业的龙头企业（如可口可乐 vs 百事可乐）
- 产业链上下游（如钢铁厂 vs 铁矿石开采商）
- 替代品（如微软 vs 谷歌）
- ETF及其成分股（如SPY vs 其重仓股）

❌ **低概率协整对**：
- 不同行业、不同商业模式
- 一个是价值股，一个是成长股
- 一个是周期股，一个是防御股

### 3.2 数据驱动的搜索

```python
def search_cointegrated_pairs(stock_data, p_value_threshold=0.05):
    """
    在全市场搜索协整股票对
    
    参数:
    - stock_data: DataFrame，每列为一只股票的价格
    - p_value_threshold: p值阈值
    
    返回:
    - cointegrated_pairs: 协整对列表
    """
    cointegrated_pairs = []
    n_stocks = stock_data.shape[1]
    
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            stock_a = stock_data.iloc[:, i]
            stock_b = stock_data.iloc[:, j]
            
            # 协整检验
            result = engle_granger_test(stock_a, stock_b, significance_level=p_value_threshold)
            
            if result['is_cointegrated']:
                cointegrated_pairs.append({
                    'stock_a': stock_data.columns[i],
                    'stock_b': stock_data.columns[j],
                    'p_value': result['p_value'],
                    'beta': result['beta'],
                    'adf_statistic': result['adf_statistic']
                })
    
    # 按p值排序（p值越小，协整关系越强）
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    return cointegrated_pairs

# 示例：模拟10只股票的数据
np.random.seed(789)
n_days = 1000
stocks = {}

# 生成5个共同趋势
common_trends = [np.cumsum(np.random.normal(0, 1, n_days)) for _ in range(5)]

for i in range(10):
    # 每只股票随机选择一个共同趋势
    trend_idx = i % 5
    stocks[f'Stock_{i}'] = (100 + 
                            2 * common_trends[trend_idx] + 
                            np.cumsum(np.random.normal(0, 0.5, n_days)))

stock_df = pd.DataFrame(stocks, index=dates)

# 搜索协整对
pairs = search_cointegrated_pairs(stock_df, p_value_threshold=0.05)

print(f"\n找到 {len(pairs)} 个协整对:")
for pair in pairs[:5]:  # 显示前5个
    print(f"  {pair['stock_a']} - {pair['stock_b']}: p值={pair['p_value']:.4f}, beta={pair['beta']:.4f}")
```

### 3.3 半寿命（Half-life）筛选

即使两只股票协整，如果均值回归太慢（半寿命过长），也不是好的交易标的。

**半寿命**是指：价差从偏离状态回归到均值一半所需的时间。

```python
def calculate_half_life(spread):
    """
    计算价差的半寿命
    
    参数:
    - spread: 价差序列
    
    返回:
    - half_life: 半寿命（交易日数）
    """
    # 构建回归模型: Δspread_t = α + β * spread_{t-1} + ε_t
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # 对齐数据
    aligned = pd.concat([spread_lag, spread_diff], axis=1).dropna()
    X = sm.add_constant(aligned.iloc[:, 0])
    y = aligned.iloc[:, 1]
    
    model = OLS(y, X).fit()
    beta = model.params[1]
    
    # 半寿命公式: ln(2) / |β|
    half_life = np.log(2) / abs(beta)
    
    return half_life

# 计算示例价差的一半寿命
spread = result['residuals']
half_life = calculate_half_life(spread)
print(f"\n价差半寿命: {half_life:.1f} 个交易日 ({half_life/252:.2f} 年)")
```

**经验法则**：
- 半寿命 < 30天：过于频繁交易，成本高
- 半寿命 30-90天：**最佳区间**
- 半寿命 > 180天：资金占用太久，机会成本高

## 4. 构建交易信号

### 4.1 Z-Score 信号

最常用的交易信号是**标准化价差（Z-Score）**：

$$
z_t = \frac{\varepsilon_t - \mu_{\varepsilon}}{\sigma_{\varepsilon}}
$$

其中：
- $\varepsilon_t$ 是当前价差
- $\mu_{\varepsilon}$ 是价差的滚动均值
- $\sigma_{\varepsilon}$ 是价差的滚动标准差

```python
def generate_trading_signals(spread, window=63, entry_z=2.0, exit_z=0.5):
    """
    生成配对交易信号
    
    参数:
    - spread: 价差序列
    - window: 滚动窗口（默认63个交易日=3个月）
    - entry_z: 入场Z值（默认2.0）
    - exit_z: 出场Z值（默认0.5）
    
    返回:
    - signals: DataFrame，包含z_score和position
    """
    # 计算滚动Z-Score
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 生成仓位信号
    position = pd.Series(0, index=spread.index)
    
    for i in range(1, len(z_score)):
        if pd.isna(z_score.iloc[i]):
            position.iloc[i] = position.iloc[i-1]
        else:
            # 入场：|z_score| > entry_z
            if z_score.iloc[i] > entry_z and position.iloc[i-1] == 0:
                position.iloc[i] = -1  # 做空价差（A高估，B低估）
            elif z_score.iloc[i] < -entry_z and position.iloc[i-1] == 0:
                position.iloc[i] = 1   # 做多价差（A低估，B高估）
            
            # 出场：|z_score| < exit_z
            elif abs(z_score.iloc[i]) < exit_z and position.iloc[i-1] != 0:
                position.iloc[i] = 0   # 平仓
            
            # 保持现有仓位
            else:
                position.iloc[i] = position.iloc[i-1]
    
    signals = pd.DataFrame({
        'spread': spread,
        'z_score': z_score,
        'position': position
    })
    
    return signals

# 生成交易信号
signals = generate_trading_signals(spread, window=63, entry_z=2.0, exit_z=0.5)

print("\n========== 交易信号统计 ==========")
print(f"总交易次数: {(signals['position'].diff() != 0).sum() / 2}")
print(f"当前仓位: {signals['position'].iloc[-1]}")
print(f"平均Z-Score: {signals['z_score'].mean():.4f}")
print(f"Z-Score标准差: {signals['z_score'].std():.4f}")
```

### 4.2 信号可视化

```python
# 绘制价差和Z-Score
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# 上图：价差
ax1.plot(signals.index, signals['spread'], label='价差', linewidth=1.5, color='blue')
ax1.axhline(y=signals['spread'].mean(), color='red', linestyle='--', label='均值')
ax1.fill_between(signals.index, 
                  signals['spread'].mean() + 2*signals['spread'].std(),
                  signals['spread'].mean() - 2*signals['spread'].std(),
                  alpha=0.2, color='gray', label='±2σ')
ax1.set_ylabel('价差', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('配对交易：价差序列', fontsize=13, fontweight='bold')

# 下图：Z-Score和仓位
ax2.plot(signals.index, signals['z_score'], label='Z-Score', linewidth=1.5, color='purple')
ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='入场阈值(+2)')
ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='出场阈值(±0.5)')
ax2.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7)
ax2.set_ylabel('Z-Score', fontsize=11)
ax2.set_xlabel('日期', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 标记交易信号
for i in range(1, len(signals)):
    if signals['position'].iloc[i] != signals['position'].iloc[i-1]:
        if signals['position'].iloc[i] == 1:
            ax2.scatter(signals.index[i], signals['z_score'].iloc[i], 
                       color='green', s=50, marker='^', zorder=5, label='做多' if i==1 else '')
        elif signals['position'].iloc[i] == -1:
            ax2.scatter(signals.index[i], signals['z_score'].iloc[i], 
                       color='red', s=50, marker='v', zorder=5, label='做空' if i==1 else '')
        elif signals['position'].iloc[i] == 0 and signals['position'].iloc[i-1] != 0:
            ax2.scatter(signals.index[i], signals['z_score'].iloc[i], 
                       color='gray', s=50, marker='o', zorder=5, label='平仓' if i==1 else '')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/trading_signals.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ 交易信号图已保存: trading_signals.png")
```

## 5. 回测配对交易策略

### 5.1 策略逻辑

```python
def backtest_pair_trading(stock_a_price, stock_b_price, signals, beta, initial_capital=1000000):
    """
    回测配对交易策略
    
    参数:
    - stock_a_price: 股票A价格
    - stock_b_price: 股票B价格
    - signals: 交易信号DataFrame
    - beta: 对冲比例
    - initial_capital: 初始资金
    
    返回:
    - portfolio_value: 组合净值
    - trade_log: 交易记录
    """
    # 对齐数据
    data = pd.concat([stock_a_price, stock_b_price, signals], axis=1, join='inner')
    data.columns = ['price_a', 'price_b', 'spread', 'z_score', 'position']
    
    # 初始化
    cash = initial_capital
    shares_a = 0
    shares_b = 0
    portfolio_value = []
    trade_log = []
    
    for i in range(1, len(data)):
        date = data.index[i]
        price_a = data['price_a'].iloc[i]
        price_b = data['price_b'].iloc[i]
        position = data['position'].iloc[i]
        prev_position = data['position'].iloc[i-1]
        
        # 仓位变化
        if position != prev_position:
            if position == 1:  # 做多价差（做多A，做空B）
                # 计算持仓数量（等市值）
                total_value = cash
                shares_a = int(total_value / (price_a + beta * price_b) * 1)  # 做多1单位A
                shares_b = int(shares_a * beta)  # 做空beta单位B
                
                # 执行交易
                cash -= shares_a * price_a  # 买入A
                cash += shares_b * price_b  # 卖空B
                
                trade_log.append({
                    'date': date,
                    'action': 'OPEN_LONG',
                    'shares_a': shares_a,
                    'shares_b': -shares_b,
                    'price_a': price_a,
                    'price_b': price_b,
                    'cash': cash
                })
                
            elif position == -1:  # 做空价差（做空A，做多B）
                total_value = cash
                shares_a = int(total_value / (price_a + beta * price_b) * 1)
                shares_b = int(shares_a * beta)
                
                cash += shares_a * price_a  # 卖空A
                cash -= shares_b * price_b  # 买入B
                
                trade_log.append({
                    'date': date,
                    'action': 'OPEN_SHORT',
                    'shares_a': -shares_a,
                    'shares_b': shares_b,
                    'price_a': price_a,
                    'price_b': price_b,
                    'cash': cash
                })
                
            elif position == 0 and prev_position != 0:  # 平仓
                # 平掉所有仓位
                if prev_position == 1:
                    cash += shares_a * price_a  # 卖出A
                    cash -= shares_b * price_b  # 买回B（平仓空仓）
                elif prev_position == -1:
                    cash -= shares_a * price_a  # 买回A（平仓空仓）
                    cash += shares_b * price_b  # 卖出B
                
                shares_a = 0
                shares_b = 0
                
                trade_log.append({
                    'date': date,
                    'action': 'CLOSE',
                    'price_a': price_a,
                    'price_b': price_b,
                    'cash': cash
                })
        
        # 计算当日净值
        current_value = cash + shares_a * price_a + shares_b * price_b
        portfolio_value.append({'date': date, 'value': current_value})
    
    portfolio_df = pd.DataFrame(portfolio_value).set_index('date')
    
    return portfolio_df, trade_log

# 回测
portfolio_value, trade_log = backtest_pair_trading(
    pd.Series(stock_a, index=dates), 
    pd.Series(stock_b, index=dates), 
    signals, 
    result['beta']
)

print(f"\n========== 回测结果 ==========")
print(f"交易次数: {len(trade_log)}")
print(f"最终净值: {portfolio_value['value'].iloc[-1]:.2f}")
print(f"累计收益: {(portfolio_value['value'].iloc[-1] / 1000000 - 1):.2%}")
```

### 5.2 绩效分析

```python
def analyze_pair_trading_performance(portfolio_value):
    """分析配对交易策略绩效"""
    returns = portfolio_value['value'].pct_change().dropna()
    
    # 基础指标
    total_ret = (portfolio_value['value'].iloc[-1] / portfolio_value['value'].iloc[0]) - 1
    annual_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_ret / volatility if volatility > 0 else 0
    
    # 最大回撤
    cumret = portfolio_value['value'] / portfolio_value['value'].iloc[0]
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    max_dd = drawdown.min()
    
    # 胜率
    winning_days = (returns > 0).sum()
    win_rate = winning_days / len(returns)
    
    print("\n========== 策略绩效 ==========")
    print(f"累计收益: {total_ret:.2%}")
    print(f"年化收益: {annual_ret:.2%}")
    print(f"年化波动率: {volatility:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"日胜率: {win_rate:.2%}")
    
    return {
        'total_return': total_ret,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate
    }

performance = analyze_pair_trading_performance(portfolio_value)
```

## 6. 风险管理与实战要点

### 6.1 关键风险

⚠️ **协整关系破裂**：
- 公司基本面发生重大变化（并购、重组、行业转型）
- 宏观经济环境结构性改变
- **应对**：定期重新检验协整关系（如每季度）

⚠️ **价差发散**：
- 入场后价差继续扩大，未如期均值回归
- **应对**：设置止损（如Z-Score超过±4）

⚠️ **交易成本**：
- 配对交易频繁调仓，成本侵蚀收益
- **应对**：优化阈值（提高entry_z），减少虚假信号

### 6.2 实战建议

✅ **配对选择**：
- 优先选择同行业、相似市值的股票
- 协整检验p值 < 0.05
- 半寿命在30-90天之间

✅ **参数优化**：
- 滚动窗口：63-126个交易日
- 入场Z值：1.5-2.5
- 出场Z值：0.5-1.0

✅ **仓位管理**：
- 单对最大仓位：总资金的10-20%
- 同时持有多个不相关对，分散风险

✅ **监控指标**：
- 协整检验p值（定期重检）
- 价差的滚动均值和方差（检测 regime change）
- 累计未平仓亏损

## 7. 总结

配对交易是一种**理论严谨、实践可行**的量化策略。其核心在于：

1. **协整分析**是寻找有效配对的基础（而非简单相关性）
2. **Z-Score信号**提供了清晰的入场出场规则
3. **风险管理**至关重要（协整破裂、止损、成本控制）

**未来扩展方向**：
- **机器学习增强**：用随机森林预测价差方向
- **高频配对交易**：利用日内微小定价偏差
- **跨资产配对**：股票-ETF、期货-现货

---

**参考文献**：
1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Ganapathy, V. (2004). "Statistical Arbitrage and Pairs Trading"
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis"

**免责声明**：本文仅供学习交流，不构成投资建议。配对交易有风险，实盘需谨慎。
