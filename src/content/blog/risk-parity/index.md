---
title: "风险平价策略：构建全天候投资组合"
publishDate: '2026-06-05'
description: "风险平价策略：构建全天候投资组合 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 风险平价（Risk Parity）：重新定义资产配置

传统资产配置依赖资本配置（60/40组合），而风险平价策略则追求**风险贡献的平衡**。这一理念由Bridgewater Associates的Ray Dalio推广，旨在构建能在各种市场环境下表现稳健的投资组合。

### 传统资产配置的局限

经典的60%股票/40%债券组合存在明显缺陷：
- **风险集中**：股票通常贡献90%以上的组合波动
- **环境依赖**：在通胀上升或利率走高时表现不佳
- **相关性失效**：市场压力时期，分散化效果减弱

### 风险平价的核心思想

风险平价策略的目标是让**每种资产对组合总风险的贡献相等**。

**数学表达：**
给定资产收益率协方差矩阵 Σ 和权重向量 w，资产 i 的风险贡献为：
```
RC_i = w_i * (Σw)_i / √(w'Σw)
```
风险平价要求：RC_1 = RC_2 = ... = RC_n

### 实现步骤

#### 1. 资产选择
风险平价通常使用低相关性的资产类别：
- 股票（全球股市）
- 债券（国债、公司债）
- 大宗商品
- 通胀保护证券（TIPS）
- 黄金

#### 2. 风险预算分配
- **等额风险贡献**：每种资产贡献相同风险
- **风险预算**：根据观点调整不同资产的风险贡献

#### 3. 杠杆运用
由于低风险资产（如国债）的预期收益较低，风险平价策略通常需要**适度杠杆**来达到目标收益率。

### Python实现示例

```python
import numpy as np
from scipy.optimize import minimize

def risk_contribution(weights, cov_matrix):
    """计算各资产的风险贡献"""
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_risk = np.dot(cov_matrix, weights) / port_vol
    risk_contrib = weights * marginal_risk
    return risk_contrib

def risk_parity_objective(weights, cov_matrix):
    """风险平价优化目标函数"""
    rc = risk_contribution(weights, cov_matrix)
    # 目标：各资产风险贡献相等
    target_rc = np.mean(rc)
    return np.sum((rc - target_rc) ** 2)

# 示例：4资产协方差矩阵
cov_matrix = np.array([
    [0.04, 0.02, 0.01, 0.00],
    [0.02, 0.09, 0.03, 0.01],
    [0.01, 0.03, 0.16, 0.02],
    [0.00, 0.01, 0.02, 0.01]
])

# 约束条件
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(4))

# 优化
result = minimize(
    risk_parity_objective,
    x0=[0.25, 0.25, 0.25, 0.25],
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"最优权重: {optimal_weights}")
print(f"风险贡献: {risk_contribution(optimal_weights, cov_matrix)}")
```

### 风险平价的优势

1. **分散化更有效**：不仅分散资本，而且分散风险
2. **更稳健的夏普比率**：长期风险调整后收益更优
3. **适应不同市场环境**：不依赖单一资产类别的优异表现
4. **自动再平衡**：市场风险变化时权重自动调整

### 实践挑战

#### 1. 杠杆成本
- 融资成本会侵蚀收益
- 杠杆约束（如UCITS基金禁止杠杆）

#### 2. 相关性上升
- 危机时期资产相关性趋近1
- 分散化效果减弱

#### 3. 参数敏感性
- 协方差矩阵估计误差
- 再平衡频率选择

### 改进版本

#### 1. 波动率目标（Volatility Targeting）
将组合波动率控制在目标水平（如10%年化），通过杠杆或去杠杆实现。

#### 2. 相关性加权
根据资产间相关性调整风险贡献，相关性高的资产赋予较低权重。

#### 3. 动态风险预算
根据宏观经济状态调整风险预算：
- 经济扩张 → 增加股票风险预算
- 通胀上升 → 增加大宗商品和TIPS权重
- 衰退 → 增加国债权重

### 实证表现

回溯测试（1970-2020）：
- **年化收益率**：9-10%（含杠杆）
- **年化波动率**：10-12%
- **夏普比率**：0.6-0.8
- **最大回撤**：通常小于传统60/40组合

### 总结

风险平价策略通过平衡风险贡献而非资本配置，构建了更稳健的投资组合。虽然面临杠杆成本和相关性上升等挑战，但其核心思想——**风险分散比资本分散更重要**——对量化投资和资产配置具有深远启发。

对于个人投资者，可以借鉴风险平价思想：
1. 选择低相关性资产
2. 定期再平衡
3. 关注风险贡献而非仅看权重

![风险平价权重分配](/images/risk-parity/risk-contribution.png)

*风险平价策略中各资产的风险贡献分布*

![传统vs风险平价组合](/images/risk-parity/traditional-vs-rp.png)

*传统60/40组合与风险平价组合的风险贡献对比*