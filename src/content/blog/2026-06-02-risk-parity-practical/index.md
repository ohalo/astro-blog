---
title: 风险平价模型实战：超越马科维茨的配置革命
publishDate: '2026-06-02'
description: 风险平价模型实战 - halo的技术博客
tags:
  - 量化交易
language: Chinese
difficulty: intermediate
---

## 引言：传统资产配置的困境

现代投资组合理论（MPT）由哈里·马科维茨在1952年提出，其核心是通过资产配置实现风险与收益的平衡。然而，传统MPT存在几个关键问题：

1. **收益率预测不可靠**：未来收益难以准确预测
2. **参数敏感性高**：预期收益的小变化会导致最优权重的大幅变动
3. **集中度风险**：优化结果往往集中在少数资产上

**风险平价（Risk Parity）模型**正是为了解决这些问题而诞生的。它不再依赖收益预测，而是专注于**风险贡献的均衡分配**。

![投资组合配置示意图](/images/2026-06-02-risk-parity-practical/portfolio-allocation.jpg)

## 风险平价的核心思想

### 1. 什么是风险平价？

风险平价的核心原则是：**投资组合中每种资产对总风险的贡献应该相等**。

用数学语言表达：
```
资产i的风险贡献 = 资产i的权重 × 资产i与组合的协方差 / 组合总风险
```

在风险平价模型中，我们追求：
```
风险贡献₁ = 风险贡献₂ = ... = 风险贡献ₙ
```

### 2. 与传统模型的对比

| 特性 | 传统均值方差优化 | 风险平价模型 |
|------|----------------|-------------|
| **输入参数** | 预期收益 + 协方差矩阵 | 仅需协方差矩阵 |
| **优化目标** | 最大化夏普比率 | 风险贡献均衡 |
| **对收益预测的依赖** | 高（结果高度敏感） | 低（不依赖收益预测） |
| **集中度** | 容易集中在少数资产 | 自然分散化 |
| **杠杆使用** | 通常无杠杆 | 常使用杠杆达到目标波动率 |

### 3. 风险平价的直觉理解

想象一个天平，每种资产都是天平上的一个砝码。传统模型试图预测每个砝码的"价值"（预期收益），而风险平价只关心每个砝码的"重量"（风险贡献）是否相等。

## Python实战：构建风险平价组合

让我们用Python实现一个简单的风险平价模型：

```python
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. 下载数据
tickers = ['SPY', 'TLT', 'GLD', 'QQQ']  # 股票、债券、黄金、科技股
data = yf.download(tickers, start="2020-01-01", end="2025-12-31")['Adj Close']

# 2. 计算收益率
returns = data.pct_change().dropna()

# 3. 计算协方差矩阵（年化）
cov_matrix = returns.cov() * 252

# 4. 风险平价优化函数
def risk_parity_objective(weights, cov_matrix):
    """风险平价的目标函数：最小化风险贡献的差异"""
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    risk_contributions = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
    target_risk = np.mean(risk_contributions)
    return np.sum((risk_contributions - target_risk) ** 2)

# 5. 约束条件：权重之和为1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.01, 1) for _ in range(len(tickers)))  # 最小权重1%

# 6. 优化
initial_weights = np.array([1/len(tickers)] * len(tickers))
result = minimize(risk_parity_objective, 
                  initial_weights, 
                  args=(cov_matrix,),
                  method='SLSQP',
                  bounds=bounds,
                  constraints=constraints)

risk_parity_weights = result.x

# 7. 输出结果
print("风险平价权重分配:")
for ticker, weight in zip(tickers, risk_parity_weights):
    print(f"  {ticker}: {weight:.2%}")

# 8. 计算风险贡献
portfolio_vol = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix, risk_parity_weights)))
risk_contributions = risk_parity_weights * (np.dot(cov_matrix, risk_parity_weights)) / portfolio_vol

print("\n风险贡献:")
for ticker, rc in zip(tickers, risk_contributions):
    print(f"  {ticker}: {rc:.2%}")
```

## 风险平价的实际应用

### 1. 经典案例：桥水全天候基金

雷·达里奥的桥水基金推出的"全天候（All Weather）"基金是风险平价的最著名应用。其核心配置为：

- **股票**：30%（增长资产）
- **长期国债**：40%（通胀对冲）
- **中期国债**：15%（ deflation 对冲）
- **黄金**：7.5%（通胀对冲）
- **大宗商品**：7.5%（通胀对冲）

这个配置的目标是：在四种经济环境（增长高于预期、增长低于预期、通胀高于预期、通胀低于预期）中，都有资产能够表现良好。

### 2. 杠杆化的风险平价

由于风险平价通常给债券等低波动资产分配较高权重，其预期收益可能较低。为了解决这个问题，许多基金使用**杠杆**来提高组合的波动率，从而达到更高的预期收益。

```python
# 杠杆化风险平价组合
TARGET_VOL = 0.15  # 目标波动率15%
current_vol = portfolio_vol  # 当前组合波动率

leverage = TARGET_VOL / current_vol
leveraged_weights = risk_parity_weights * leverage

print(f"当前组合波动率: {current_vol:.2%}")
print(f"目标波动率: {TARGET_VOL:.2%}")
print(f"所需杠杆: {leverage:.2f}倍")
```

### 3. 带约束的风险平价

在实际应用中，我们可能需要加入各种约束：
- **最小/最大权重约束**：避免过于集中或过于分散
- **交易成本约束**：限制换手率
- **流动性约束**：确保资产具有足够流动性

```python
# 带最大权重约束的风险平价
max_weight = 0.5  # 单一资产最大权重50%
bounds_with_constraint = tuple((0.01, max_weight) for _ in range(len(tickers)))

result_constrained = minimize(risk_parity_objective, 
                            initial_weights, 
                            args=(cov_matrix,),
                            method='SLSQP',
                            bounds=bounds_with_constraint,
                            constraints=constraints)

constrained_weights = result_constrained.x
```

## 风险平价的优缺点

### 优点

1. **不依赖收益预测**：避免了传统模型中收益预测不准确的问题
2. **自然分散化**：避免集中在少数资产上
3. **稳健性好**：在不同市场环境下表现相对稳定
4. **透明度高**：逻辑清晰，易于理解和解释

### 缺点

1. **杠杆风险**：为了达到目标波动率，可能需要使用高杠杆
2. **利率风险**：债券权重通常较高，利率上升时表现不佳
3. **相关性变化**：危机时期资产相关性趋于1，分散化效果降低
4. **模型风险**：依赖波动率预测的准确性

## 实战中的改进方向

### 1. 引入层次风险平价（Hierarchical Risk Parity, HRP）

传统的HRP通过资产的相关性构建层次聚类，然后在层次结构上分配风险，避免了协方差矩阵求逆的不稳定性。

```python
# HRP的简化示例（实际需要更复杂的聚类算法）
def hierarchical_risk_parity(returns):
    """层次风险平价"""
    # 1. 计算相关性矩阵
    corr = returns.corr()
    
    # 2. 层次聚类（这里简化为按相关性排序）
    # 实际中应使用如scipy的层次聚类
    sorted_assets = corr.mean().sort_values().index
    
    # 3. 递归分配权重
    def recursive_weights(assets):
        if len(assets) == 1:
            return {assets[0]: 1.0}
        
        # 将资产分为两组
        mid = len(assets) // 2
        group1 = assets[:mid]
        group2 = assets[mid:]
        
        # 计算两组的风险贡献
        # ...（简化，实际需要计算协方差）
        w1 = 0.5  # 简化：等权
        w2 = 0.5
        
        weights = {}
        for asset, w in recursive_weights(group1).items():
            weights[asset] = w * w1
        for asset, w in recursive_weights(group2).items():
            weights[asset] = w * w2
            
        return weights
    
    hrp_weights = recursive_weights(list(sorted_assets))
    return hrp_weights
```

### 2. 加入宏观状态切换

不同的宏观环境下，资产的风险特征不同。可以结合马尔可夫状态切换模型，在不同状态下应用不同的风险平价配置。

### 3. 动态风险预算

不一定非要完全相等的风险贡献，可以根据资产的风险调整后收益（如夏普比率）动态调整风险预算。

![风险平衡示意图](/images/2026-06-02-risk-parity-practical/risk-balance.jpg)

## 结论

风险平价模型是对传统均值方差优化的重要改进，它：
1. **避免了对收益预测的依赖**
2. **实现了风险的均衡分配**
3. **提供了更稳健的资产配置方案**

然而，它并非没有缺陷。投资者需要：
- 理解杠杆使用的风险
- 关注利率环境的变化
- 意识到危机时期相关性集中的问题

**完美模型不存在，但风险平价为我们提供了一个更稳健的资产配置框架。**

---

*希望这篇文章能帮助你理解风险平价模型的核心原理和实际应用。如果你有任何问题或想法，欢迎在评论区讨论！*
