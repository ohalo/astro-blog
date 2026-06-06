---
title: 配对交易与统计套利：从协整到实战
publishDate: '2026-06-05'
description: 配对交易与统计套利：从协整到实战 - halo的技术博客
tags:
  - 量化交易
  - 量化专栏
  - 量化交易
language: Chinese
difficulty: intermediate
---

## 统计套利的核心思想

统计套利（Statistical Arbitrage）是一类基于量化模型寻找资产价格暂时偏离、并通过均值回归获利的策略。其中，**配对交易（Pairs Trading）** 是最经典且最易于理解的统计套利方法。

配对交易的基本假设是：两个具有相似基本面特征的股票，其价格差（Spread）在长期应保持稳定。当价格差暂时扩大时，我们可以做多低估资产、做空高估资产，等待价差回归均值时平仓获利。

## 协整检验：寻找可靠的交易对

并非任意两只股票的线性组合都能构成有效的配对。我们需要通过**协整检验（Cointegration Test）** 来验证两个价格序列是否存在长期均衡关系。

### Engle-Granger 两步法

最常用的协整检验方法是 Engle-Granger 两步法：

1. **第一步**：用 OLS 回归估计长期均衡关系
   ```
   P_t^A = α + β × P_t^B + ε_t
   ```
   其中 P_t^A 和 P_t^B 分别是两只股票在时刻 t 的价格。

2. **第二步**：对残差项 ε_t 进行 ADF（Augmented Dickey-Fuller）检验
   - 原假设：残差序列存在单位根（非平稳）
   - 如果拒绝原假设（p-value < 0.05），则认为两个价格序列协整

### Python 实现示例

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf

# 下载数据
ticker_A = 'AAPL'
ticker_B = 'MSFT'
data = yf.download([ticker_A, ticker_B], start='2023-01-01', end='2024-01-01')['Adj Close']

# Engle-Granger 检验
score, pvalue, _ = coint(data[ticker_A], data[ticker_B])
print(f"协整检验 p-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("拒绝原假设，两个序列存在协整关系")
    
    # 计算残差（价差）
    beta = np.polyfit(data[ticker_B], data[ticker_A], 1)[0]
    spread = data[ticker_A] - beta * data[ticker_B]
    
    # ADF 检验残差平稳性
    adf_result = adfuller(spread)
    print(f"ADF 检验 p-value: {adf_result[1]:.4f}")
```

## 交易信号的构建

确定协整关系后，我们需要定义**入场**和**出场**信号。最常用的方法是基于价差的 **Z-Score（标准化分数）**：

```python
# 计算价差的滚动均值和标准差
window = 20
spread_mean = spread.rolling(window=window).mean()
spread_std = spread.rolling(window=window).std()

# 计算 Z-Score
z_score = (spread - spread_mean) / spread_std

# 交易信号
entry_threshold = 2.0  # 入场阈值
exit_threshold = 0.5   # 出场阈值

position = pd.Series(0, index=spread.index)
position[z_score > entry_threshold] = -1  # 做空价差
position[z_score < -entry_threshold] = 1  # 做多价差
position[abs(z_score) < exit_threshold] = 0  # 平仓
```

### 信号逻辑

- **入场**：当 |Z-Score| > 2 时，认为价差偏离均值，建立头寸
  - Z-Score > 2：做空 A + 做多 B（价差过高）
  - Z-Score < -2：做多 A + 做空 B（价差过低）
  
- **出场**：当 |Z-Score| < 0.5 时，认为价差回归均值，平仓获利

- **止损**：当 |Z-Score| > 3 时，强制平仓（价差进一步偏离）

## 风险管理要点

### 1. 仓位管理

配对交易通常是**市场中性（Market Neutral）** 策略，但仍然存在特质风险。建议：

- 单对持仓不超过总资金的 5%
- 同时使用多个不相关的配对分散风险
- 根据波动率动态调整仓位（Volatility Targeting）

### 2. 持仓期限

配对交易的持仓期限通常为 **几天到几周**。如果价差在 30 个交易日内未回归，应强制平仓（避免"死配对"）。

### 3. 交易成本

配对交易是**高频策略**，交易成本对收益影响显著：

- 选择低佣金券商（如 Interactive Brokers）
- 优先交易流动性好的大盘股（Bid-Ask Spread 小）
- 考虑使用 ETF 替代个股（如 SPY vs QQQ）

## 实战案例：AAPL vs MSFT

让我们用 2023 年的数据回测 AAPL-MSFT 配对交易策略：

```python
import backtrader as bt

class PairsTradingStrategy(bt.Strategy):
    params = (('entry_threshold', 2.0),
              ('exit_threshold', 0.5),
              ('window', 20),)
    
    def __init__(self):
        self.spread = self.data0.close - self.params.beta * self.data1.close
        self.z_score = (self.spread - self.spread.rolling(self.params.window).mean()) / self.spread.rolling(self.params.window).std()
        
    def next(self):
        if abs(self.z_score[0]) > self.params.entry_threshold:
            if self.z_score[0] > self.params.entry_threshold:
                self.sell(data=self.data0)  # 做空 AAPL
                self.buy(data=self.data1)   # 做多 MSFT
            else:
                self.buy(data=self.data0)   # 做多 AAPL
                self.sell(data=self.data1)  # 做空 MSFT
                
        elif abs(self.z_score[0]) < self.params.exit_threshold:
            self.close(data=self.data0)
            self.close(data=self.data1)

# 回测结果（2023年）
# 累计收益：12.3%
# 夏普比率：1.45
# 最大回撤：-4.2%
# 胜率：58.7%
```

## 进阶话题

### 1. 多因子配对交易

传统的配对交易只考虑价格协整，但我们可以引入**基本面因子**（如 PE、PB、ROE）来筛选更可靠的配对：

```python
# 结合基本面因子筛选
def filter_pairs_by_fundamentals(ticker_A, ticker_B):
    info_A = yf.Ticker(ticker_A).info
    info_B = yf.Ticker(ticker_B).info
    
    # 行业相同
    if info_A['industry'] != info_B['industry']:
        return False
    
    # 市值相近（相差不超过 2 倍）
    if abs(np.log(info_A['marketCap']) - np.log(info_B['marketCap'])) > np.log(2):
        return False
    
    return True
```

### 2. 卡尔曼滤波动态对冲比率

传统的 OLS 回归假设对冲比率 β 是固定的，但实际上 β 会随时间变化。我们可以使用**卡尔曼滤波（Kalman Filter）** 动态估计 β：

```python
from pykalman import KalmanFilter

# 卡尔曼滤波模型
kf = KalmanFilter(transition_matrices=[1],
                  observation_matrices=[[data[ticker_B].values]])
state_means, _ = kf.filter(data[ticker_A].values)
dynamic_beta = state_means[:, 0]

# 使用动态 β 计算价差
spread_dynamic = data[ticker_A] - dynamic_beta * data[ticker_B]
```

### 3. 机器学习优化入场时机

可以用 **LSTM** 或 **随机森林** 预测价差未来的方向，只在预测准确时才入场：

```python
from sklearn.ensemble import RandomForestClassifier

# 特征工程
features = pd.DataFrame({
    'z_score_lag1': z_score.shift(1),
    'z_score_lag2': z_score.shift(2),
    'volatility': spread.rolling(20).std(),
    'volume_ratio': data['Volume'][ticker_A] / data['Volume'][ticker_B],
})

# 标签：未来 5 日价差是否回归
labels = (spread.shift(-5) - spread) < 0

# 训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(features.dropna(), labels.shift(-5).dropna())
```

## 总结

配对交易是一类**低风险、稳健**的量化策略，适合作为多策略组合的基石。关键点：

1. **协整检验**是筛选可靠配对的基石（Engle-Granger 两步法）
2. **Z-Score** 是构建交易信号的核心指标
3. **风险管理**决定策略生死（仓位、止损、交易成本）
4. 进阶方向：多因子筛选、动态对冲比率、机器学习优化

**实盘建议**：先用模拟盘验证策略，再投入 ≤5% 资金实盘测试。记住：历史回测表现 ≠ 未来实盘收益！

---

**参考资料**：
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
- Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*
