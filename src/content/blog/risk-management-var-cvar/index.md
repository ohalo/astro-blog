---
title: "风险管理实战：VaR与CVaR的计算与应用"
publishDate: '2026-06-05'
description: "风险管理实战：VaR与CVaR的计算与应用 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 为什么风险管理比策略更重要？

量化交易圈有一句老话：

> **"策略决定你能赚多少，风控决定你能活多久。"**

无数明星基金经理（如长期资本管理公司 LTCM）不是因为策略失效倒闭，而是因为**风控失误**导致爆仓。

本文将深入讲解量化交易中最核心的两个风险指标：**VaR（在险价值）**和 **CVaR（条件在险价值）**，以及它们的计算方法和实战应用。

## VaR（Value at Risk）：最广泛使用的风险指标

### 定义

**VaR 回答的问题是**：

> 在给定置信水平 $\alpha$ 下，未来 $T$ 个交易日内，投资组合的最大损失不会超过多少？

数学定义：

$$
P(L_T > \text{VaR}_\alpha) = 1 - \alpha
$$

其中 $L_T$ 是未来 $T$ 期的损失。

**举例**：95% VaR = 100 万元，意味着有 95% 把握，未来 1 天损失不超过 100 万元（5% 概率损失超过 100 万）。

### VaR 的三种计算方法

#### 方法 1：历史模拟法（Historical Simulation）

**逻辑**：用过去 $N$ 天的收益率分布，直接取分位数。

**步骤**：
1. 收集过去 500-1000 天的投资组合价值序列
2. 计算每日收益率 $r_t = \frac{V_t - V_{t-1}}{V_{t-1}}$
3. 对收益率排序，取第 $(1-\alpha) \times N$ 分位数

**优点**：
- ✅ 不假设收益率服从正态分布
- ✅ 能捕捉肥尾特征

**缺点**：
- ❌ 假设"过去能代表未来"
- ❌ 对极端事件估计不足（2008 年金融危机）

**Python 实现**：

```python
import numpy as np
import pandas as pd

def var_historical(returns, confidence=0.95, holding_period=1):
    """
    历史模拟法计算 VaR
    
    Parameters:
    - returns: 收益率序列 (pd.Series)
    - confidence: 置信水平 (default 0.95)
    - holding_period: 持有期 (default 1 天)
    """
    # 将日收益率转换为 holding_period 期收益率
    if holding_period > 1:
        returns = (1 + returns).rolling(holding_period).apply(np.prod) - 1
    
    # 取分位数
    var = np.percentile(returns, (1 - confidence) * 100)
    return abs(var)

# 示例
portfolio_returns = pd.Series(...)  # 你的组合收益率数据
var_95 = var_historical(portfolio_returns, confidence=0.95)
print(f"95% VaR = {var_95:.2%}")```

#### 方法 2：方差-协方差法（Parametric / Delta-Normal）

**逻辑**：假设收益率服从正态分布，用均值和方差直接计算。

**公式**（正态分布假设下）：

$$
\text{VaR}_\alpha = -(\mu + \sigma \cdot \Phi^{-1}(\alpha)) \cdot V_0
$$

其中：
- $\mu$ 和 $\sigma$ 是收益率的均值和标准差
- $\Phi^{-1}(\alpha)$ 是标准正态分布的 $\alpha$ 分位数
- $V_0$ 是当前投资组合价值

对 95% VaR：$\Phi^{-1}(0.05) \approx -1.645$

**优点**：
- ✅ 计算速度快
- ✅ 解析解，易于理解

**缺点**：
- ❌ 严重低估肥尾风险（收益率通常非正态）
- ❌ 对期权等非线性衍生品不适用

**Python 实现**：

```python
from scipy.stats import norm

def var_parametric(returns, confidence=0.95, holding_period=1):
    """
    方差-协方差法计算 VaR（正态分布假设）
    """
    mu = returns.mean() * holding_period
    sigma = returns.std() * np.sqrt(holding_period)
    z_score = norm.ppf(1 - confidence)
    var = -(mu + sigma * z_score)
    return var

var_95_param = var_parametric(portfolio_returns, confidence=0.95)
print(f"95% VaR (正态假设) = {var_95_param:.2%}")```

#### 方法 3：蒙特卡洛模拟（Monte Carlo Simulation）

**逻辑**：用随机过程模拟未来收益率路径，生成大量场景后取分位数。

**常用模型**：
- **几何布朗运动（GBM）**：$dS_t = \mu S_t dt + \sigma S_t dW_t$
- **跳跃扩散模型（Merton Jump-Diffusion）**：加入泊松跳跃项
- **GARCH 模型**：波动率聚类效应

**优点**：
- ✅ 最灵活，可模拟复杂衍生品
- ✅ 能捕捉非线性风险

**缺点**：
- ❌ 计算成本高
- ❌ 依赖模型假设（模型风险）

**Python 实现（GBM）**：

```python
def var_monte_carlo(returns, confidence=0.95, holding_period=1, n_sims=10000):
    """
    蒙特卡洛模拟计算 VaR（几何布朗运动）
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # 模拟 holding_period 期的收益率
    np.random.seed(42)
    z = np.random.normal(0, 1, n_sims)
    simulated_returns = mu * holding_period + sigma * np.sqrt(holding_period) * z
    
    # 取分位数
    var = np.percentile(simulated_returns, (1 - confidence) * 100)
    return abs(var)

var_95_mc = var_monte_carlo(portfolio_returns, confidence=0.95, n_sims=50000)
print(f"95% VaR (蒙特卡洛) = {var_95_mc:.2%}")```

### 三种方法对比

| 方法 | 计算速度 | 准确性 | 假设条件 | 适用场景 |
|------|---------|--------|---------|---------|
| 历史模拟 | ⭐⭐⭐ | ⭐⭐ | 无（非参数） | 常规股票组合 |
| 方差-协方差 | ⭐⭐⭐⭐⭐ | ⭐ | 正态分布 | 快速估算、线性资产 |
| 蒙特卡洛 | ⭐ | ⭐⭐⭐⭐ | 随机过程模型 | 期权、复杂衍生品 |

## CVaR（Conditional VaR）：更稳健的风险指标

### VaR 的致命缺陷

VaR 有一个严重问题：**不关心超过 VaR 的部分**。

> 如果两个投资组合的 95% VaR 都是 100 万，但组合 A 在 99% 情况下损失 200 万，组合 B 在 99% 情况下损失 1000 万，VaR 无法区分！

### CVaR 的定义

**CVaR（又称 Expected Shortfall）** 回答：

> 当损失超过 VaR 时，**平均损失是多少**？

数学定义：

$$
\text{CVaR}_\alpha = E[L_T | L_T > \text{VaR}_\alpha]
$$

**性质**：
- ✅ **次可加性（Subadditivity）**：满足风险度量的公理
- ✅ **更注重尾部风险**：对极端损失更敏感
- ✅ **一致性风险度量**（Artzner et al., 1999）

### CVaR 的计算

#### 历史模拟法

```python
def cvar_historical(returns, confidence=0.95):
    """
    历史模拟法计算 CVaR
    """
    var = var_historical(returns, confidence)
    # 取超过 VaR 的损失的平均值
    tail_losses = returns[returns < -var]
    cvar = tail_losses.mean()
    return abs(cvar)

cvar_95 = cvar_historical(portfolio_returns, confidence=0.95)
print(f"95% CVaR = {cvar_95:.2%}")```

#### 蒙特卡洛法

```python
def cvar_monte_carlo(returns, confidence=0.95, n_sims=50000):
    """
    蒙特卡洛模拟计算 CVaR
    """
    var = var_monte_carlo(returns, confidence, n_sims=n_sims)
    
    # 生成模拟收益率
    mu = returns.mean()
    sigma = returns.std()
    np.random.seed(42)
    z = np.random.normal(0, 1, n_sims)
    simulated_returns = mu + sigma * z
    
    # 取超过 VaR 的部分计算平均
    tail_losses = simulated_returns[simulated_returns < -var]
    cvar = tail_losses.mean()
    return abs(cvar)
```

![VaR与CVaR分布可视化](/images/risk-management-var-cvar/var_cvar_distribution.png)

## 实战应用：风险管理框架

### 1. 仓位管理：基于 VaR 的头寸调整

**目标**：控制整个组合的 VaR 不超过总资金的某个比例（如 2%）。

**公式**：

$$
w_i = \frac{\text{VaR}_{\text{target}}}{\text{VaR}_i}
$$

其中 $w_i$ 是资产 $i$ 的目标权重，$\text{VaR}_i$ 是资产 $i$ 的 VaR。

**Python 实现**：

```python
def optimize_position_size(asset_returns, target_var=0.02, confidence=0.95):
    """
    基于 VaR 的仓位优化
    target_var: 目标 VaR（占总资金比例）
    """
    asset_var = var_historical(asset_returns, confidence)
    position_size = target_var / asset_var
    return min(position_size, 1.0)  # 不超过 100% 仓位

# 示例：单资产仓位建议
position = optimize_position_size(stock_returns, target_var=0.02)
print(f"建议仓位: {position:.1%}")```

### 2. 压力测试（Stress Testing）

**逻辑**：用历史极端事件（如 2008 年金融危机、2015 年 A 股股灾）或假设的极端场景，测试组合表现。

**常见场景**：
- 2008 年雷曼兄弟破产（-20% 单日跌幅）
- 2020 年疫情爆发（VIX 飙升至 85）
- 利率突然上升 200 bp
- 人民币贬值 10%

**Python 实现**：

```python
def stress_test(portfolio_value, scenario_shocks):
    """
    压力测试
    scenario_shocks: dict, {资产: 冲击幅度}
    """
    losses = {}
    for asset, shock in scenario_shocks.items():
        losses[asset] = portfolio_value[asset] * shock
    
    total_loss = sum(losses.values())
    return total_loss, losses

# 示例：2008 年金融危机场景
scenario_2008 = {
    '股票': -0.40,   # 沪深 300 下跌 40%
    '债券': -0.05,   # 债券下跌 5%
    '商品': -0.30    # 工业品下跌 30%
}
total_loss, details = stress_test(portfolio_value, scenario_2008)
print(f"总损失: {total_loss:.2f} 万, 占比: {total_loss/sum(portfolio_value.values()):.1%}")```

### 3. 风险预算配置（Risk Parity）

**逻辑**：不按资金权重配置，而是按**风险贡献**配置。

每个资产的风险贡献：

$$
RC_i = w_i \cdot \frac{\partial \text{VaR}}{\partial w_i}
$$

目标：让所有资产的 $RC_i$ 相等。

## A股实战中的注意事项

### 1. 涨跌停限制

A 股有 ±10%（主板）或 ±20%（创业板/科创板）涨跌停限制，导致：

- **无法及时止损**：极端行情下卖不出
- **收益率截断**：历史模拟法低估真实风险

**应对**：在 VaR 计算中加入**流动性调整因子**。

### 2. 换手率限制

A 股 T+1 交易制度，当天买入无法卖出。

**应对**：用 **VaR 预测多期风险**（至少 2 天）。

### 3. 因子暴露

A 股存在明显的**风格轮动**（如 2021 年成长 → 价值），组合可能暴露在某种风格因子下。

**应对**：计算 **因子 VaR**（将组合收益分解为市场、风格、行业因子）。

## 总结

| 指标 | 优点 | 缺点 | 推荐使用场景 |
|------|------|------|-------------|
| **VaR** | 直观易懂、监管认可 | 不满足次可加性、忽略尾部 | 日常风险监控、监管报告 |
| **CVaR** | 关注极端风险、一致性度量 | 计算复杂、需要更多数据 | 尾部风险管理、压力测试 |

**实战建议**：

1. ✅ **日常监控用 VaR**：计算简单，便于沟通
2. ✅ **尾部风险评估用 CVaR**：捕捉极端风险
3. ✅ **结合压力测试**：不要只依赖历史数据
4. ✅ **动态调整**：市场波动率变化时（如 VIX 飙升），及时更新 VaR 模型参数
5. ❌ **不要盲目相信 99% VaR**：黑天鹅事件会打破所有模型假设

![不同置信水平下VaR与CVaR对比](/images/risk-management-var-cvar/var_cvar_comparison.png)

---

**下期预告**：我们将讲解**如何用机器学习预测波动率**（GARCH-LSTM 混合模型），以及**动态对冲策略**的实现细节。

**参考资料**：
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*
- Artzner, P., et al. (1999). "Coherent Measures of Risk", *Mathematical Finance*