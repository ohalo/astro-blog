---
title: "配对交易与协整分析"
description: "深入探讨配对交易的理论基础与协整分析方法，学习如何识别协整关系、构建配对交易策略，并提供完整的Python实现代码与实战案例。"
publishDate: 2026-06-19
category: "量化策略"
tags:
  - 配对交易
  - 协整分析
  - 统计套利
  - 市场中性
  - Python实战
image: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

## 引言

**配对交易（Pairs Trading）** 是一种经典的市场中性策略，起源于1980年代摩根士丹利的数量部门。该策略的核心思想是：找到两个价格具有长期均衡关系的股票，当它们的价格偏离均衡时，做多价格偏低的股票、做空价格偏高的股票，等待价格回归均衡后平仓获利。

配对交易的优势在于：
- **市场中性**：多空对冲，不受大盘方向影响
- **低风险**：基于均值回归，风险相对可控
- **适用广泛**：适用于股票、期货、ETF等多个市场

本文将深入探讨配对交易的理论基础——**协整分析**，并提供完整的Python实现代码。

## 一、配对交易的理论基础

### 1.1 平稳性与协整

在介绍协整之前，需要先理解**平稳性（Stationarity）**的概念。

**平稳时间序列**满足：
1. 均值恒定
2. 方差恒定
3. 自协方差仅依赖于时间差

金融价格序列通常是**非平稳**的（存在单位根），但它们的**线性组合**可能是平稳的，这种关系就是**协整（Cointegration）**。

**数学定义**：
若两个非平稳序列 $X_t$ 和 $Y_t$ 满足：
$$Y_t = \alpha + \beta X_t + \epsilon_t$$
其中 $\epsilon_t$ 是平稳序列，则称 $X_t$ 和 $Y_t$ 是协整的。

### 1.2 配对交易的逻辑

如果两个股票的价格协整，那么：
1. 它们的价差（或比率）会围绕均值波动
2. 当价差偏离均值时，未来会回归均值
3. 我们可以在价差偏低时做多价差（做多$Y$，做空$\beta$份$X$），在价差偏高时做空价差

## 二、协整检验方法

### 2.1 Engle-Granger 两步法

**步骤1**：用OLS回归估计协整关系
$$Y_t = \alpha + \beta X_t + \epsilon_t$$

**步骤2**：对残差 $\epsilon_t$ 进行**单位根检验**（如ADF检验）
- 若残差是平稳的（p值 < 0.05），则 $X_t$ 和 $Y_t$ 协整
- 若残差非平稳，则不存在协整关系

### 2.2 Johansen 检验

Engle-Granger方法只能检验两个变量之间的协整关系，而**Johansen检验**可以检验多个变量之间的协整关系。

**优点**：
- 适用于多变量系统
- 可以检验多个协整向量

**缺点**：
- 计算复杂
- 对小样本不太稳健

### 2.3 Phillips-Ouliaris 检验

这是ADF检验的改进版本，考虑了估计误差，在大样本中表现更好。

## 三、Python实战：构建配对交易策略

下面我们用Python实现一个完整的配对交易策略，包括：
1. 寻找协整股票对
2. 构建交易信号
3. 回测策略性能

### 3.1 数据准备

我们使用A股市场的银行股作为示例，因为同行业股票更容易存在协整关系。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import tushare as ts

# 设置tushare token
ts.set_token('your_tushare_token')
pro = ts.pro_api()

def get_stock_pairs(start_date, end_date):
    """
    获取股票数据，筛选同行业股票对
    """
    # 获取股票基本信息
    stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    
    # 选择银行股
    bank_stocks = stocks[stocks['industry'] == '银行']['ts_code'].tolist()[:10]
    
    # 获取日线数据
    all_data = []
    for code in bank_stocks:
        try:
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            df = df[['ts_code', 'trade_date', 'close']]
            all_data.append(df)
        except:
            continue
    
    # 合并数据
    data = pd.concat(all_data, ignore_index=True)
    data = data.pivot(index='trade_date', columns='ts_code', values='close')
    
    return data

# 获取数据
start_date = '20200101'
end_date = '20241231'
price_data = get_stock_pairs(start_date, end_date)

print(f"获取到 {price_data.shape[1]} 只股票，{price_data.shape[0]} 个交易日的数据")
```

### 3.2 寻找协整对

```python
def find_cointegrated_pairs(data, p_value_threshold=0.05):
    """
    寻找协整的股票对
    """
    n = data.shape[1]
    pairs = []
    p_values = []
    
    for i in range(n):
        for j in range(i+1, n):
            stock1 = data.iloc[:, i]
            stock2 = data.iloc[:, j]
            
            # 去除缺失值
            combined = pd.concat([stock1, stock2], axis=1).dropna()
            s1 = combined.iloc[:, 0]
            s2 = combined.iloc[:, 1]
            
            # 进行协整检验
            try:
                score, p_value, _ = coint(s1, s2)
                if p_value < p_value_threshold:
                    pairs.append((data.columns[i], data.columns[j]))
                    p_values.append(p_value)
            except:
                continue
    
    return pairs, p_values

# 寻找协整对
pairs, p_values = find_cointegrated_pairs(price_data)

print(f"\n找到 {len(pairs)} 个协整对：")
for i, (stock1, stock2) in enumerate(pairs):
    print(f"{i+1}. {stock1} - {stock2} (p-value: {p_values[i]:.4f})")
```

### 3.3 构建交易信号

找到协整对后，我们需要构建交易信号。常用的方法是**Z-Score**方法：

1. 计算残差：$\epsilon_t = Y_t - (\alpha + \beta X_t)$
2. 计算Z-Score：$z_t = \frac{\epsilon_t - \mu_{\epsilon}}{\sigma_{\epsilon}}$
3. 交易规则：
   - 当 $z_t < -2$ 时，做多价差（做多$Y$，做空$\beta$份$X$）
   - 当 $z_t > 2$ 时，做空价差
   - 当 $z_t$ 回归到 $[-0.5, 0.5]$ 时，平仓

```python
def calculate_z_score(data, stock1, stock2, window=60):
    """
    计算价差的Z-Score
    """
    # 获取价格序列
    y = data[stock2]
    x = data[stock1]
    
    # OLS回归
    model = OLS(y, sm.add_constant(x)).fit()
    beta = model.params[1]
    
    # 计算残差
    residual = y - (model.params[0] + beta * x)
    
    # 计算滚动均值和标准差
    rolling_mean = residual.rolling(window=window).mean()
    rolling_std = residual.rolling(window=window).std()
    
    # 计算Z-Score
    z_score = (residual - rolling_mean) / rolling_std
    
    return z_score, residual, beta

def generate_signals(z_score, entry_threshold=2, exit_threshold=0.5):
    """
    生成交易信号
    """
    signals = pd.Series(0, index=z_score.index)
    
    # 1: 做多价差（做多Y，做空X）
    # -1: 做空价差（做空Y，做多X）
    
    position = 0
    for i in range(1, len(z_score)):
        if position == 0:
            if z_score.iloc[i] < -entry_threshold:
                position = 1
                signals.iloc[i] = 1
            elif z_score.iloc[i] > entry_threshold:
                position = -1
                signals.iloc[i] = -1
        elif position == 1:
            if abs(z_score.iloc[i]) < exit_threshold:
                position = 0
                signals.iloc[i] = 0
        elif position == -1:
            if abs(z_score.iloc[i]) < exit_threshold:
                position = 0
                signals.iloc[i] = 0
    
    return signals

# 选择第一个协整对进行回测
if len(pairs) > 0:
    stock1, stock2 = pairs[0]
    print(f"\n回测协整对：{stock1} - {stock2}")
    
    # 计算Z-Score
    z_score, residual, beta = calculate_z_score(price_data, stock1, stock2)
    
    # 生成交易信号
    signals = generate_signals(z_score)
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 子图1：价格序列
    axes[0].plot(price_data[stock1], label=stock1)
    axes[0].plot(price_data[stock2], label=stock2)
    axes[0].set_title('价格序列')
    axes[0].legend()
    
    # 子图2：Z-Score
    axes[1].plot(z_score, label='Z-Score')
    axes[1].axhline(y=2, color='r', linestyle='--', label='入场阈值')
    axes[1].axhline(y=-2, color='r', linestyle='--')
    axes[1].axhline(y=0.5, color='g', linestyle='--', label='出场阈值')
    axes[1].axhline(y=-0.5, color='g', linestyle='--')
    axes[1].set_title('Z-Score')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('pair_trading_signals.png', dpi=150, bbox_inches='tight')
    print("✓ 生成信号图: pair_trading_signals.png")
```

### 3.4 回测策略

```python
def backtest_pair_trading(data, stock1, stock2, signals, beta):
    """
    回测配对交易策略
    """
    # 获取价格数据
    price1 = data[stock1]
    price2 = data[stock2]
    
    # 计算收益率
    ret1 = price1.pct_change()
    ret2 = price2.pct_change()
    
    # 计算策略收益
    strategy_ret = pd.Series(0, index=signals.index)
    
    for i in range(1, len(signals)):
        if signals.iloc[i-1] == 1:  # 做多价差
            strategy_ret.iloc[i] = ret2.iloc[i] - beta * ret1.iloc[i]
        elif signals.iloc[i-1] == -1:  # 做空价差
            strategy_ret.iloc[i] = -ret2.iloc[i] + beta * ret1.iloc[i]
    
    # 计算累计收益
    cumulative_ret = (1 + strategy_ret).cumprod()
    
    # 计算绩效指标
    total_return = cumulative_ret.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(strategy_ret)) - 1
    sharpe_ratio = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)
    max_drawdown = (cumulative_ret / cumulative_ret.cummax() - 1).min()
    
    return {
        'cumulative_returns': cumulative_ret,
        'strategy_returns': strategy_ret,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

# 回测
if len(pairs) > 0:
    results = backtest_pair_trading(price_data, stock1, stock2, signals, beta)
    
    print("\n=== 回测结果 ===")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    
    # 可视化累计收益
    plt.figure(figsize=(10, 6))
    plt.plot(results['cumulative_returns'], label='配对交易策略', linewidth=2)
    plt.title('配对交易策略累计收益')
    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pair_trading_returns.png', dpi=150, bbox_inches='tight')
    print("✓ 生成收益图: pair_trading_returns.png")
```

## 四、配对交易的关键问题

### 4.1 协整关系断裂

协整关系可能不是永久的。当股票的商业模式、行业环境发生变化时，协整关系可能断裂，导致策略失效。

**解决方法**：
- 使用**滚动窗口**定期重新检验协整关系
- 设置**止损机制**：当残差持续扩大时及时止损

### 4.2 模型风险

OLS回归假设残差独立同分布，但实际应用中残差可能存在**自相关性**或**异方差**。

**解决方法**：
- 使用**稳健标准误**
- 考虑使用**VECM（向量误差修正模型）**

### 4.3 交易成本

配对交易通常需要频繁调仓，交易成本会显著影响策略收益。

**解决方法**：
- 设置**交易成本约束**：只有当预期收益大于交易成本时才交易
- 使用**低频调仓**：每天或每周调仓一次

### 4.4 选对难度

在A股市场，找到稳定的协整对并不容易。建议：
- 优先选择**同行业、同板块**的股票
- 使用**聚类分析**预先筛选相似股票
- 考虑**ETF配对**（如沪深300ETF与中证500ETF）

## 五、实战建议

### 5.1 多元配对

不要只依赖一个配对，应该构建**配对组合**，分散风险。

### 5.2 动态调参

Z-Score的入场阈值（如2）和出场阈值（如0.5）需要根据市场状态动态调整。

### 5.3 结合基本面

在协整分析的基础上，结合**基本面分析**（如估值、盈利），可以提升策略稳健性。

### 5.4 风险控制

- 设置**单笔最大亏损**
- 设置**最大持仓时间**
- 监控**配对相关性**：当相关性显著下降时，考虑平仓

## 六、总结

配对交易是一种经典的市场中性策略，核心在于识别协整关系。本文介绍了协整检验的理论与方法，并提供了完整的Python实现代码。

**关键要点**：
1. 协整是配对交易的理论基础
2. Engle-Granger两步法是常用的协整检验方法
3. Z-Score是构建交易信号的常用工具
4. 实际应用中需要注意协整关系断裂、模型风险和交易成本

希望本文能帮助你理解配对交易与协整分析，并在实践中构建稳健的量化策略。

## 参考资料

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Elliott, R. J., et al. (2005). Pairs trading. *Quantitative Finance*.
3. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*.

---

**免责声明**：本文仅供参考，不构成投资建议。量化投资有风险，入市需谨慎。
