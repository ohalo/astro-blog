---
title: "配对交易与协整分析：统计套利的实战指南"
publishDate: '2026-06-10'
description: "配对交易与协整分析：统计套利的实战指南 - halo的技术博客"
tags:
  - 量化交易
language: Chinese
---

## 什么是配对交易？

配对交易（Pairs Trading）是一种典型的**市场中性策略**，通过寻找两个高度相关的资产，当价格偏离历史均衡时做多低估资产、做空高估资产，等待价格回归获利。

核心思想：**均值回归** + **对冲市场系统性风险**

## 协整理论：配对交易的数学基础

### 平稳性检验

在进行配对交易前，必须验证两个价格序列是否存在**协整关系**：

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

# Augmented Dickey-Fuller检验（平稳性）
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }

# 协整检验
def cointegration_test(series1, series2):
    result = coint(series1, series2)
    return {
        'coint_stat': result[0],
        'p-value': result[1],
        'critical_values': result[2]
    }
```

### 协整 vs 相关性

| 维度 | 相关性 | 协整 |
|------|--------|------|
| 衡量 | 短期联动 | 长期均衡 |
| 平稳性要求 | 无 | 需要 |
| 套利意义 | 弱 | 强 |
| 实际案例 | 同行业股票 | 股指期货vs现货 |

## 实战步骤：构建配对交易策略

### Step 1: 候选股票筛选

```python
# 示例：在同一行业内寻找潜在配对
import yfinance as yf
from itertools import combinations

def find_potential_pairs(tickers, start='2020-01-01'):
    """寻找协整配对的候选股票"""
    data = yf.download(tickers, start=start)['Adj Close']
    
    pairs = []
    for ticker1, ticker2 in combinations(tickers, 2):
        # 协整检验
        score, pvalue, _ = coint(data[ticker1], data[ticker2])
        if pvalue < 0.05:  # 显著协整
            pairs.append({
                'pair': (ticker1, ticker2),
                'p-value': pvalue,
                'coint_score': score
            })
    
    return sorted(pairs, key=lambda x: x['p-value'])
```

### Step 2: 计算价差和Z分数

```python
def calculate_spread_zscore(price1, price2, window=20):
    """计算价差和Z分数"""
    # 对冲比例（用OLS回归）
    import statsmodels.api as sm
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    beta = model.params[1]
    
    # 计算价差
    spread = price1 - beta * price2
    
    # 计算Z分数
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    return spread, zscore, beta
```

![配对交易价差可视化](/images/2026-06-10-pairs-trading-cointegration/spread_visualization.jpg)

### Step 3: 交易信号生成

```python
class PairsTradingStrategy:
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        
    def generate_signals(self, zscore):
        """生成交易信号：1=做多价差，-1=做空价差，0=平仓"""
        signals = pd.Series(0, index=zscore.index)
        
        # 进场：Z分数超过阈值
        signals[zscore > self.entry_zscore] = -1  # 做空价差
        signals[zscore < -self.entry_zscore] = 1   # 做多价差
        
        # 出场：Z分数回归
        signals[(signals != 0) & (abs(zscore) < self.exit_zscore)] = 0
        
        return signals.fillna(0)
```

## 风险管理要点

### 1. 止损规则

```python
def stop_loss(zscore, max_zscore=4.0):
    """Z分数超过历史极值时止损"""
    return abs(zscore) > max_zscore
```

### 2. 持仓时间限制

配对交易应有**最大持仓周期**（如20个交易日），避免价差长期不回归。

### 3. 资金管理

- 单对配对：总资金的2-5%
- 同时持仓配对数量：≤5对
- 动态调整：根据波动率调整仓位

## 实证案例：A股银行股配对

以**招商银行(600036)** 和 **平安银行(000001)** 为例：

```python
# 回测设置
pair = ('600036.SS', '000001.SZ')
start_date = '2020-01-01'
end_date = '2025-12-31'

# 回测结果
results = backtest_pairs_strategy(
    pair, 
    start_date, 
    end_date,
    initial_capital=1000000,
    commission=0.0003
)

print(f"年化收益率: {results['annual_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

![回测净值曲线](/images/2026-06-10-pairs-trading-cointegration/backtest_equity.jpg)

## 常见陷阱与应对

### 陷阱1：伪协整

**现象**：样本内协整显著，样本外失效

**应对**：
- 使用滚动窗口检验协整稳定性
- 样本外测试至少6个月
- 考虑结构断裂检验

### 陷阱2：交易成本侵蚀利润

**现象**：高频来回交易，手续费吃掉收益

**应对**：
- 设置合理的Z分数阈值（≥2.0）
- 考虑交易成本优化阈值
- 使用限价单减少滑点

### 陷阱3：模型退化

**现象**：市场结构变化导致配对关系失效

**应对**：
- 定期重新检验协整关系
- 设置配对失效监控指标
- 建立配对轮换机制

## 进阶话题

### 多资产配对交易

扩展到**一篮子股票** vs **指数ETF**：
- 更稳定的协整关系
- 更好的流动性
- 更高的对冲效率

### 机器学习增强

使用**LSTM**预测价差回归时间：
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 构建LSTM模型预测价差方向
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, n_features)),
    LSTM(50),
    Dense(1)
])
```

## 总结

配对交易是统计套利的核心策略，成功的关键在于：

1. **严格的协整检验**：确保长期均衡关系存在
2. **合理的风险管理**：止损、仓位、持仓时间
3. **持续的监控与优化**：市场变化时需要调整参数
4. **低交易成本执行**：对手续费和滑点敏感

**实战建议**：从流动性好的大盘股开始，使用分钟级数据回测，逐步建立自己的配对交易系统。

---

**参考资源**：
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
- Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*
