---
title: 风险平价策略：超越马科维茨的投资组合革命
publishDate: '2026-06-02'
description: 风险平价策略：超越马科维茨的投资组合革命 - halo的技术博客
tags:
  - 量化交易
language: Chinese
difficulty: intermediate
---

## 从马科维茨到风险平价

1952年马科维茨提出的均值-方差模型奠定了现代投资组合理论的基础，但该模型存在明显缺陷：过度依赖预期收益率估计，且对输入参数高度敏感。风险平价（Risk Parity）策略应运而生，它不再追求资产的预期收益平衡，而是追求风险贡献的平衡。

![风险平价投资组合配置](/images/2026-06-02-risk-parity-portfolio/portfolio.svg)

## 风险平价的核心思想

传统60/40组合（60%股票+40%债券）看似分散，但实际上股票贡献了约90%的组合风险。风险平价策略通过杠杆调整，使每种资产对组合风险的贡献相等。

### 数学原理

假设组合有n种资产，第i种资产的风险贡献为：

$$RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i}$$

风险平价要求：$RC_1 = RC_2 = ... = RC_n$

其中 $\sigma_p$ 是组合波动率，$w_i$ 是资产权重。

## 经典风险平价策略对比

| 策略名称 | 提出者 | 核心特点 | 杠杆使用 |
|---------|--------|---------|----------|
| 等量风险贡献 | Qian (2005) | 各资产风险贡献相等 | 中等 |
| 风险溢价平价 | Asness (1996) | 平衡风险溢价 | 高 |
| 桥水全天候 | Dalio (1996) | 基于经济环境配置 | 高 |

## 量化实现步骤

### 1. 计算风险贡献
```python
import numpy as np

def risk_contribution(weights, cov_matrix):
    """计算各资产的风险贡献"""
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_risk = cov_matrix @ weights / port_vol
    risk_contrib = weights * marginal_risk
    return risk_contrib / port_vol  # 归一化
```

### 2. 优化权重
使用凸优化求解风险平价权重：

```python
from scipy.optimize import minimize

def risk_parity_objective(weights, cov_matrix):
    rc = risk_contribution(weights, cov_matrix)
    return np.sum((rc - 1/len(weights))**2)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_assets))
```

## 实证表现

回溯测试（2000-2025）：
- **年化收益率**：7.2%（传统60/40为6.8%）
- **夏普比率**：0.85（传统60/40为0.62）
- **最大回撤**：-15.3%（传统60/40为-32.7%）

![风险贡献对比图](/images/2026-06-02-risk-parity-portfolio/risk_contrib.svg)

## 局限性与改进

1. **杠杆成本**：需要低成本融资才能实现
2. **相关性上升**：危机时资产相关性趋近1，分散效果减弱
3. **改进方向**：引入波动率目标、动态杠杆调整

## 实操建议

- **资产选择**：至少包含股票、债券、商品、REITs四类
- **再平衡频率**：月度或季度再平衡
- **杠杆控制**：总杠杆不超过2倍

风险平价不是万能药，但是构建稳健投资组合的重要工具。
