---
title: "VaR与CVaR风险度量方法实战"
publishDate: '2026-06-13'
description: "VaR与CVaR风险度量 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 风险度量的重要性

在量化交易中，**风险管理**是区分业余玩家与专业机构的关键分水岭。无论策略回测表现多么优异，如果缺乏严格的风险度量与控制框架，实盘随时可能遭遇毁灭性打击。

### 为什么需要风险度量？

- **资本保全**：避免单次交易或组合回撤过大导致爆仓
- **仓位管理**：根据风险水平动态调整持仓规模
- **监管合规**：金融机构必须满足监管层的风险资本要求
- **投资者信心**：透明的风险报告增强资金方信任

## VaR（Value at Risk）详解

### VaR 的定义

**VaR（风险价值）** 表示在给定置信水平下，特定时间窗口内，投资组合可能遭受的**最大损失**。

**数学定义**：
```
P(Loss > VaR_α) = 1 - α
```
其中 α 为置信水平（通常取 95% 或 99%）。

**示例**：日度 VaR_95% = 10万元，意味着在 95% 的置信水平下，未来一天的损失不会超过 10万元；但有 5% 的概率损失会超过 10万元。

### VaR 的三种计算方法

#### 1. 历史模拟法（Historical Simulation）

**原理**：直接使用历史收益率数据，按分位数计算 VaR。

**优点**：
- 不需要假设收益率分布（非参数方法）
- 能捕捉厚尾和偏度特征

**缺点**：
- 假设历史会重演（黑天鹅事件可能未被历史数据覆盖）
- 对数据窗口长度敏感

**Python 实现**：
```python
import numpy as np

def historical_var(returns, confidence=0.95):
    """历史模拟法计算 VaR"""
    return np.percentile(returns, (1 - confidence) * 100)

# 示例
returns = np.random.normal(0, 0.05, 1000)  # 模拟日收益率
var_95 = historical_var(returns, 0.95)
print(f"日度 VaR_95%: {var_95:.4f}")
```

#### 2. 参数法（Parametric / Variance-Covariance）

**原理**：假设收益率服从特定分布（如正态分布），利用均值和方差计算 VaR。

**优点**：
- 计算速度快
- 解析解易于理解

**缺点**：
- 正态分布假设低估肥尾风险
- 无法捕捉极端事件

**公式**：
```
VaR_α = μ - z_α * σ
```
其中 z_α 为标准正态分布的 α 分位数。

#### 3. 蒙特卡洛模拟法（Monte Carlo Simulation）

**原理**：通过随机模拟未来收益率路径，生成大量情景计算 VaR。

**优点**：
- 可模拟复杂的非线性衍生品组合
- 能捕捉尾部依赖和相关性变化

**缺点**：
- 计算成本高
- 依赖模型假设（如随机过程参数）

## CVaR（Conditional VaR）的优势

### VaR 的局限性

1. **尾部信息缺失**：VaR 只告诉你在置信水平内的损失，不关心超过 VaR 的部分
2. **非次可加性**：违反风险度量的公理（多个组合的总风险可能大于各组合风险之和）
3. **鼓励赌博**：交易者可能通过卖出深度虚值期权来"隐藏"风险（因为 VaR 不捕捉极端损失）

### CVaR 的定义

**CVaR（条件风险价值）**，也称 **Expected Shortfall（预期缺口）**，表示在损失超过 VaR 的条件下，平均损失是多少。

**数学定义**：
```
CVaR_α = E[Loss | Loss > VaR_α]
```

**示例**：如果 VaR_95% = 10万元，CVaR_95% = 15万元，意味着在 5% 的极端情况下，平均损失为 15万元。

### 为什么 CVaR 更好？

1. **捕捉尾部风险**：提供超过 VaR 阈值的期望损失
2. **次可加性**：满足风险度量的数学公理
3. **监管认可**：巴塞尔协议 III 推荐使用 Expected Shortfall 替代 VaR

## Python 实战：计算并比较 VaR 与 CVaR

```python
import numpy as np
import scipy.stats as stats

def calculate_var_cvar(returns, confidence=0.95):
    """计算 VaR 和 CVaR（历史模拟法）"""
    # VaR
    var = np.percentile(returns, (1-confidence) * 100)
    
    # CVaR: 平均超过 VaR 的损失
    cvar = returns[returns <= var].mean()
    
    return var, cvar

def parametric_var(returns, confidence=0.95):
    """参数法 VaR（假设正态分布）"""
    mu = returns.mean()
    sigma = returns.std()
    z_score = stats.norm.ppf(confidence)
    return mu - z_score * sigma

# 示例：对比历史模拟法与参数法
np.random.seed(42)
# 使用 t 分布模拟肥尾数据
returns = np.random.standard_t(df=4, size=1000) * 0.05

var_hist, cvar_hist = calculate_var_cvar(returns, 0.95)
var_param = parametric_var(returns, 0.95)

print(f"历史模拟法 VaR_95%: {var_hist:.4f}")
print(f"历史模拟法 CVaR_95%: {cvar_hist:.4f}")
print(f"参数法 VaR_95%: {var_param:.4f}")
print(f"（参数法低估了风险！）")
```

## 实战应用：风险管理框架

### 1. 组合层面风险度量

对于多资产组合，需要考虑资产间的相关性：

```python
import pandas as pd

def portfolio_var(returns_df, weights, confidence=0.95):
    """计算投资组合 VaR（参数法）"""
    # 组合收益率
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # VaR
    var = np.percentile(portfolio_returns, (1-confidence) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    return var, cvar

# 示例：3 资产组合
returns_df = pd.DataFrame({
    'Stock_A': np.random.normal(0.0005, 0.02, 1000),
    'Stock_B': np.random.normal(0.0003, 0.015, 1000),
    'Stock_C': np.random.normal(0.0004, 0.018, 1000)
})
weights = np.array([0.4, 0.3, 0.3])

var, cvar = portfolio_var(returns_df, weights)
print(f"组合 VaR_95%: {var:.4f}")
print(f"组合 CVaR_95%: {cvar:.4f}")
```

### 2. 回测 VaR 模型

**失败率检验**（Kupiec 检验）：
- 如果 95% VaR 模型准确，应该有约 5% 的时间损失超过 VaR
- 如果失败率显著偏离 5%，说明 VaR 模型校准不当

```python
def backtest_var(actual_returns, var_estimate, confidence=0.95):
    """回测 VaR 模型"""
    failures = actual_returns < var_estimate
    failure_rate = failures.mean()
    
    expected_rate = 1 - confidence
    print(f"期望失败率: {expected_rate:.2%}")
    print(f"实际失败率: {failure_rate:.2%}")
    print(f"总交易日: {len(actual_returns)}")
    print(f"失败天数: {failures.sum()}")
    
    return failure_rate

# 示例
actual_returns = returns_df.sum(axis=1)  # 等权重组合收益率
var_estimate = np.percentile(actual_returns, 5)
backtest_var(actual_returns, var_estimate)
```

### 3. 动态仓位管理

根据 VaR/CVaR 调整仓位大小：

```python
def position_sizing(var, max_risk_amount, current_price):
    """根据 VaR 计算最大持仓量"""
    # 单笔交易最大损失不超过账户资金的 2%
    max_loss_per_share = abs(var * current_price)
    max_shares = max_risk_amount / max_loss_per_share
    return int(max_shares)

# 示例
account_value = 1000000  # 100万账户
max_risk_amount = account_value * 0.02  # 最大风险 2%
current_price = 100
var_daily = 0.02  # 日度 VaR 2%

max_shares = position_sizing(var_daily, max_risk_amount, current_price)
print(f"最大持仓量: {max_shares} 股")
print(f"持仓市值: {max_shares * current_price:.0f} 元")
```

## 风险提示

1. **模型风险**：VaR/CVaR 依赖历史数据和分布假设，黑天鹅事件可能击穿模型
2. **流动性风险**：市场恐慌时，平仓成本可能远超预期
3. **相关性突变**：危机期间资产相关性趋近于 1，分散化失效
4. **补充措施**：结合压力测试（Stress Testing）和情景分析（Scenario Analysis）

## 总结

- **VaR** 提供置信水平内的最大损失估计，但忽略尾部风险
- **CVaR** 捕捉极端损失的期望值，是更稳健的风险度量工具
- 实战中应**同时使用多种方法**（历史模拟 + 参数法 + 蒙特卡洛）
- 风险管理不是一次性工作，需要**每日监控、定期回测、动态调整**

对于量化团队，建议搭建自动化的风险度量系统，实时计算组合 VaR/CVaR，并设置多级预警机制（如 VaR 超过账户 5% 时触发警报）。

![VaR与CVaR风险度量示意图](/images/2026-06-13-var-cvar-risk-management/var-cvar-comparison.jpg)
