---
title: Black-Litterman 模型：把主观观点融入量化配置
publishDate: '2026-06-03'
description: Black-Litterman 模型：把主观观点融入量化配置 - halo的技术博客
tags:
  - 量化交易
language: Chinese
difficulty: intermediate
---

## 现代投资组合理论的困境

马科维茨均值-方差优化（MVO）在实践中常常失效：
- **输入敏感性极高**：预期收益率微小变化 → 权重剧烈调整
- **集中化倾向**：输出权重集中在少数资产，缺乏分散性
- **估计误差放大**：历史收益率的噪声被优化器放大

Black-Litterman（BL）模型正是为解决这些问题而生。

## Black-Litterman 的核心思想

BL 模型由 Fischer Black 和 Robert Litterman 在 1992 年提出，核心创新是**把主观观点与市场均衡收益相结合**。

### 两大约束
1. **均衡约束**：在无观点时，组合权重应等于市场权重（CAPM 隐含收益）
2. **观点约束**：允许投资者输入"相对观点"或"绝对观点"，并赋予置信度

## 数学框架（简化版）

### 第1步：计算隐含均衡收益
\[
E[R_{eq}] = \lambda \cdot \Sigma \cdot w_{mkt}
\]
其中：
- \(\lambda\) 为风险厌恶系数（通常取 2.5-3.0）
- \(\Sigma\) 为资产协方差矩阵
- \(w_{mkt}\) 为市场权重

### 第2步：融合主观观点
\[
E[R] = [( \tau \Sigma )^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} E[R_{eq}] + P^T \Omega^{-1} Q]
\]
其中：
- \(P\) 为观点矩阵（k × n，k 个观点，n 个资产）
- \(Q\) 为观点收益向量
- \(\Omega\) 为观点不确定性的协方差矩阵

![Black-Litterman 模型结构示意图](/images/2026-06-03-black-litterman/bl-model.jpg)

## 实战案例：A股行业配置

### 场景设定
假设我们在 2025 年初配置 A 股 6 个行业 ETF：
- 消费、医药、科技、金融、周期、公用事业

### 输入市场权重
| 行业 | 市场权重 |
|------|---------|
| 消费 | 25% |
| 医药 | 18% |
| 科技 | 22% |
| 金融 | 20% |
| 周期 | 10% |
| 公用 | 5% |

### 输入主观观点
1. **相对观点**（置信度 65%）：科技未来半年跑赢消费 5%
2. **绝对观点**（置信度 80%）：医药年化收益将达到 12%

### BL 输出权重
| 行业 | MVO 权重 | BL 权重 |
|------|---------|---------|
| 消费 | 8% | 22% |
| 医药 | 35% | 21% |
| 科技 | 42% | 28% |
| 金融 | 5% | 18% |
| 周期 | 10% | 7% |
| 公用 | 0% | 4% |

BL 权重明显更分散、更稳健。

![BL 模型 vs MVO 权重对比](/images/2026-06-03-black-litterman/bl-vs-mvo.jpg)

## Python 实现要点

```python
import numpy as np
from scipy.optimize import minimize

def black_litterman(mu_eq, Sigma, P, Q, Omega, tau=0.05):
    """
    mu_eq: 均衡预期收益 (n,)
    Sigma: 协方差矩阵 (n, n)
    P: 观点矩阵 (k, n)
    Q: 观点收益 (k,)
    Omega: 观点不确定性 (k, k)
    tau: 缩放因子（通常 0.02-0.05）
    """
    tau_Sigma = tau * Sigma
    M_inv = np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(Omega) @ P
    M = np.linalg.inv(M_inv)
    mu_bl = M @ (np.linalg.inv(tau_Sigma) @ mu_eq + P.T @ np.linalg.inv(Omega) @ Q)
    Sigma_bl = M @ tau_Sigma  # 后验协方差（可选）
    return mu_bl, Sigma_bl
```

## 实践中的注意事项

1. **观点置信度不要设太高**：建议单个观点置信度 ≤ 80%
2. **协方差矩阵需要收缩**：Ledoit-Wolf 收缩能显著改善稳定性
3. **观点要可回溯测试**：记录每次观点的准确性，迭代优化
4. **定期重新校准**：市场结构变化后需重新估计均衡收益

## 延伸阅读

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*
- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios*

---

*系列回顾：本文是量化配置系列第3篇，前篇参见《因子拥挤与拥挤崩塌》《均值回归验证》。*
