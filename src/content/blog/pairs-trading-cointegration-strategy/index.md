---
title: "配对交易实战：基于协整关系的均值回归策略"
publishDate: '2026-06-12'
description: "配对交易实战：基于协整关系的均值回归策略 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言

配对交易（Pairs Trading）是统计套利中最经典的策略之一，其核心理念是寻找两个价格具有长期均衡关系的股票，当价格偏离时做多低估股票、做空高估股票，等待价格回归均衡后获利。这种市场中性策略不依赖市场方向，而是通过捕捉相对价格的均值回归特性来获取收益。

## 配对交易的理论基础

### 1. 协整关系（Cointegration）

协整是配对交易的理论基石。简单来说，如果两个时间序列都是非平稳的（如股价），但它们的某个线性组合是平稳的，那么这两个序列就存在协整关系。

数学表达：
```
如果 Y_t = α + βX_t + ε_t
其中 ε_t 是平稳序列（均值回归），则 Y 和 X 存在协整关系
```

### 2. 为什么要协整而非简单相关？

- **相关性**只衡量同涨同跌的程度，可能是短期现象
- **协整性**保证长期均衡关系，价格偏离是暂时的
- 协整关系意味着"价格会回归"，这是配对交易的盈利来源

## 实战步骤：从选股到交易

### 步骤1：候选股票筛选

以A股市场为例，我们可以：

1. **同行业选股**：同一行业（如银行、地产）的股票更可能有协整关系
2. **市值匹配**：避免大小盘差异过大导致的结构性偏离
3. **流动性筛选**：确保两只股票都有足够的成交量

**代码示例：获取候选股票**

```python
import tushare as ts
import pandas as pd

# 获取银行股列表
banks = ['601398.SH', '601939.SH', '601288.SH', '600036.SH', '601166.SH']

# 下载近一年的收盘价
def get_price_data(stocks, start='20250101', end='20260612'):
    df = pd.DataFrame()
    for code in stocks:
        df[code] = ts.get_k_data(code, start=start, end=end)['close']
    return df

price_data = get_price_data(banks)
```

### 步骤2：协整检验

常用方法：**Engle-Granger 两步法** 或 **Johansen 检验**

这里展示简化的 Engle-Granger 检验：

```python
from statsmodels.tsa.stattools import coint
import numpy as np

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    
    keys = data.keys()
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            
            if pvalue < 0.05:  # p值小于0.05，认为存在协整关系
                pairs.append((keys[i], keys[j], score, pvalue))
    
    return pairs, score_matrix, pvalue_matrix

# 找出所有协整配对的股票
pairs, scores, pvalues = find_cointegrated_pairs(price_data)
```

### 步骤3：计算对冲比率（Hedge Ratio）

使用**OLS回归**计算持有比例：

```python
from sklearn.linear_model import LinearRegression

def calculate_hedge_ratio(Y, X):
    """
    Y: 被解释变量（要做多的股票）
    X: 解释变量（要做空的股票）
    返回：对冲比率（β）
    """
    model = LinearRegression()
    model.fit(X.values.reshape(-1, 1), Y.values)
    return model.coef_[0]

# 示例：计算工商银行(601398.SH)与建设银行(601939.SH)的对冲比率
beta = calculate_hedge_ratio(price_data['601398.SH'], price_data['601939.SH'])
print(f"对冲比率 β = {beta:.4f}")
```

### 步骤4：构建交易信号

核心指标：**Z-Score（标准化价差）**

```python
def calculate_z_score(spread, window=20):
    """
    计算价差的Z分数
    spread: 两只股票的价差（或残差）
    window: 滚动窗口
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std
    return z_score

# 构建交易信号
def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """
    入场：|Z-score| > entry_threshold
    出场：|Z-score| < exit_threshold
    """
    signal = pd.Series(0, index=z_score.index)
    signal[z_score > entry_threshold] = -1  # 做空价差（卖出Y，买入X）
    signal[z_score < -entry_threshold] = 1   # 做多价差（买入Y，卖出X）
    
    # 平仓信号
    signal[(signal == 1) & (z_score > -exit_threshold)] = 0
    signal[(signal == -1) & (z_score < exit_threshold)] = 0
    
    return signal

# 完整流程
spread = price_data['601398.SH'] - beta * price_data['601939.SH']
z_score = calculate_z_score(sppread)
signals = generate_signals(z_score)
```

## 实战案例：招商银行 vs 平安银行

让我们用真实数据测试这个策略：

```python
# 回测框架（简化版）
def backtest_pairs_strategy(price1, price2, signals, initial_capital=1000000):
    """
    简单的配对交易回测
    """
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['positions'] = signals
    portfolio['price1'] = price1
    portfolio['price2'] = price2
    
    # 计算持仓价值（假设每次等权重投资）
    portfolio['holdings1'] = portfolio['positions'] * portfolio['price1']
    portfolio['holdings2'] = -portfolio['positions'] * portfolio['price2'] * beta
    
    # 计算收益
    portfolio['strategy_returns'] = portfolio['holdings1'].pct_change() + portfolio['holdings2'].pct_change()
    
    # 累计收益
    portfolio['cumulative_returns'] = (1 + portfolio['strategy_returns']).cumprod()
    
    return portfolio

# 执行回测
results = backtest_pairs_strategy(
    price_data['600036.SH'],  # 招商银行
    price_data['000001.SZ'],  # 平安银行
    signals
)

# 绘制权益曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['cumulative_returns'])
plt.title('配对交易策略累计收益')
plt.xlabel('日期')
plt.ylabel('累计收益')
plt.grid(True)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pairs-trading-cointegration-strategy/equity_curve.png')
plt.close()
```

![配对交易权益曲线](/images/pairs-trading-cointegration-strategy/equity_curve.png)

## 风险控制与实战要点

### 1. 止损机制

配对交易虽然理论上风险有限，但实战中必须设置止损：

- **时间止损**：如果价差在N天内未回归，强制平仓（避免"死 pair"）
- **幅度止损**：如果Z-score继续扩大到3.0以上，说明协整关系可能失效
- **基本面止损**：如果其中一只股票出现重大基本面变化（如重组、退市风险）

```python
def add_stop_loss(signals, z_score, max_holding_days=20, max_z_score=3.0):
    """
    添加止损逻辑
    """
    position_start = None
    new_signals = signals.copy()
    
    for i in range(len(signals)):
        if signals.iloc[i] != 0:
            if position_start is None:
                position_start = i
            else:
                # 检查持仓天数
                holding_days = i - position_start
                if holding_days > max_holding_days:
                    new_signals.iloc[i] = 0  # 时间止损
                    position_start = None
                
                # 检查Z-score是否继续扩大
                if abs(z_score.iloc[i]) > max_z_score:
                    new_signals.iloc[i] = 0  # 幅度止损
                    position_start = None
        else:
            position_start = None
    
    return new_signals
```

### 2. 交易成本考虑

A股做空成本较高（融券利率约8-10%），需考虑：

- **融券费用**：按日计息，长期持仓成本巨大
- **冲击成本**：小盘股冲击成本可能吃掉全部利润
- **换手率**：频繁交易会导致手续费累积

**优化建议**：
1. 优先选择易融券的大盘蓝筹
2. 适当放宽入场阈值（如从2.0调整到2.5），减少交易次数
3. 使用IB（Interactive Brokers）等低成本券商的融券服务

### 3. 配对衰变（Pair Decay）

协整关系不是永恒的！以下情况可能导致配对失效：

- 行业政策变化（如地产调控导致地产股分化）
- 公司基本面分化（如一家业绩暴雷）
- 市场结构变化（如注册制改革改变估值体系）

**监控方法**：
- 每月重新运行协整检验
- 跟踪对冲比率β的变化（如果β显著变化，说明关系弱化）
- 设置"配对健康度"指标（如R²下降则预警）

## 进阶话题：多因子配对交易

传统配对交易只考虑价格关系，可以加入基本面因子提升效果：

```python
# 多因子配对交易框架
def multi_factor_pairs_selection(stock1, stock2, fundamental_data):
    """
    除了价格协整，还要求基本面相似
    fundamental_data: PB、PE、ROE、市值等
    """
    # 计算基本面距离
    distance = np.sqrt(
        (fundamental_data[stock1]['PB'] - fundamental_data[stock2]['PB'])**2 +
        (fundamental_data[stock1]['PE'] - fundamental_data[stock2]['PE'])**2 +
        ...
    )
    
    # 只有基本面相似 + 价格协整的配对才入选
    if distance < threshold and is_cointegrated(stock1, stock2):
        return True
    return False
```

## 总结

配对交易是一种逻辑清晰、风险可控的量化策略，但也面临以下挑战：

✅ **优点**：
- 市场中性，不需要预测大盘方向
- 胜率较高（均值回归是金融市场的普遍现象）
- 策略容量较大（相比于高频策略）

❌ **缺点**：
- A股做空成本高，限制策略收益
- 协整关系可能突然失效
- 交易机会有限（大部分时间价差是平稳的）

**实战建议**：
1. 从同行业大盘股开始练习（如银行、保险）
2. 严格控制止损，不要扛单
3. 结合基本面分析，避免"伪回归"
4. 关注融券成本，确保预期收益覆盖费用

---

**参考文献**：
- Ganapathy Vidyamurthy, *Pairs Trading: Quantitative Methods and Analysis*
- Ernest Chan, *Algorithmic Trading: Winning Strategies and Their Rationale*
- 索津欣, 《统计套利：在A股市场的应用实证研究》

![配对交易原理示意图](/images/pairs-trading-cointegration-strategy/pairs_trading_diagram.png)
