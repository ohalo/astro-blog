---
title: "Black-Litterman模型实战：融合市场均衡与投资者观点的资产配置"
publishDate: '2026-06-15'
description: "Black-Litterman模型实战：融合市场均衡与投资者观点的资产配置 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 为什么需要Black-Litterman模型？

传统均值方差优化（Markowitz, 1952）存在严重的问题：

1. **输入敏感性**：微小的预期收益率变化会导致资产配置剧烈变动
2. **集中化**：优化结果往往集中在少数资产上
3. **估计误差**：历史收益率是未来收益率的有偏估计

Black-Litterman模型（1992）通过**将市场均衡组合与投资者主观观点相结合**，优雅地解决了这些问题。

![Black-Litterman模型框架](/images/2026-06-15-black-litterman-model/bl_framework.jpg)

## Black-Litterman模型的核心思想

### 1. 逆向优化（Reverse Optimization）

首先，从市场权重推导出**隐含的预期收益率**：

```
π = λΣw_mkt
```

其中：
- π：隐含预期收益率向量
- λ：风险厌恶系数
- Σ：收益率协方差矩阵
- w_mkt：市场权重向量

**直觉**：如果市场权重是最优的，那么背后的预期收益率应该是什么？

### 2. 观点融合（Opinion Fusion）

将投资者观点与隐含收益率结合：

```
E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
```

其中：
- τ：尺度因子（通常取0.05-0.1）
- P：观点矩阵（K×N）
- Q：观点收益向量（K×1）
- Ω：观点不确定性矩阵（K×K）

## Python实现：完整流程

### Step 1: 数据准备

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

# 下载资产数据
tickers = ['SPY', 'EFA', 'EEM', 'AGG', 'GSG', 'VNQ']  # 美股、发达市场、新兴市场、债券、大宗商品、REITs
data = yf.download(tickers, start='2020-01-01', end='2025-12-31')['Adj Close']

# 计算收益率
returns = data.pct_change().dropna()

# 计算协方差矩阵（年化）
cov_matrix = returns.cov() * 252
```

### Step 2: 计算市场权重（以市值加权为例）

```python
# 假设市场权重（实际应用中应从指数成分股权重获取）
market_caps = {
    'SPY': 40,   # 40%
    'EFA': 20,   # 20%
    'EEM': 15,   # 15%
    'AGG': 15,   # 15%
    'GSG': 5,    # 5%
    'VNQ': 5,    # 5%
}

w_mkt = np.array([market_caps[t] / 100 for t in tickers])

# 风险厌恶系数（通常取2.5-3）
risk_aversion = 2.5

# 逆向优化：计算隐含预期收益率
pi = risk_aversion * cov_matrix.values @ w_mkt
```

### Step 3: 定义投资者观点

Black-Litterman模型支持**绝对观点**和**相对观点**：

```python
# 示例观点：
# 1. 美股（SPY）未来一年将上涨12%（绝对观点）
# 2. 新兴市场（EEM）将跑赢发达市场（EFA）5%（相对观点）

# 观点矩阵 P (K×N)
P = np.zeros((2, len(tickers)))
P[0, 0] = 1  # 观点1：SPY绝对收益
P[1, 2] = 1  # 观点2：EEM
P[1, 1] = -1 #       减去EFA

# 观点收益向量 Q (K×1)
Q = np.array([0.12, 0.05])

# 观点不确定性矩阵 Ω (K×K)
# 对角线元素表示观点的置信度（越小越自信）
tau = 0.05
omega = np.diag([0.1**2, 0.15**2])  # 观点1不确定性10%，观点2不确定性15%
```

### Step 4: 计算后验预期收益率

```python
# 计算后验预期收益率
tau_sigma = tau * cov_matrix.values

# 中间矩阵
M_inv = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
M = np.linalg.inv(M_inv)

# 后验预期收益率
posterior_ret = M @ (np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q)

# 后验协方差矩阵
posterior_cov = cov_matrix.values + M
```

### Step 5: 均值方差优化

```python
def portfolio_vol(weights, cov_matrix):
    """计算组合波动率"""
    return np.sqrt(weights.T @ cov_matrix @ weights)

def optimize_portfolio(expected_ret, cov_matrix, risk_aversion=2.5):
    """
    均值方差优化
    """
    n_assets = len(expected_ret)
    
    # 目标函数：最大化效用 = 预期收益 - 0.5 * 风险厌恶 * 方差
    def utility(weights):
        port_ret = expected_ret @ weights
        port_vol = portfolio_vol(weights, cov_matrix)
        return -(port_ret - 0.5 * risk_aversion * port_vol**2)
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
    ]
    bounds = [(0, 1) for _ in range(n_assets)]  # 不允许卖空
    
    # 优化
    result = minimize(
        utility,
        x0=np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

# 优化组合
optimal_weights = optimize_portfolio(posterior_ret, posterior_cov, risk_aversion)

print("优化后权重：")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")
```

## 实战案例：A股资产配置

### 场景设定

假设我们要配置以下A股资产：
- 沪深300（代表大盘蓝筹）
- 中证500（代表中盘成长）
- 创业板指（代表小盘成长）
- 中债国债（无风险资产）

### 投资者观点

1. **绝对观点**：沪深300未来一年上涨15%
2. **相对观点**：中证500将跑赢沪深300约8%
3. **相对观点**：创业板指将跑赢沪深300约12%

### 完整代码

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. 数据准备（假设已有收益率数据）
tickers = ['HS300', 'ZZ500', 'CYB', 'Bond']
returns = load_returns_data()  # 自定义函数
cov_matrix = returns.cov() * 252

# 2. 市场权重（市值加权）
market_caps = {'HS300': 50, 'ZZ500': 25, 'CYB': 15, 'Bond': 10}
w_mkt = np.array([market_caps[t] / 100 for t in tickers])

# 3. 逆向优化
risk_aversion = 2.5
pi = risk_aversion * cov_matrix.values @ w_mkt

# 4. 定义观点
P = np.array([
    [1, 0, 0, 0],        # 观点1：HS300绝对收益
    [-1, 1, 0, 0],       # 观点2：ZZ500 - HS300
    [-1, 0, 1, 0],       # 观点3：CYB - HS300
])

Q = np.array([0.15, 0.08, 0.12])

# 观点不确定性
tau = 0.05
omega = np.diag([0.05**2, 0.08**2, 0.10**2])  # 观点1最自信

# 5. 计算后验
tau_sigma = tau * cov_matrix.values
M_inv = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
M = np.linalg.inv(M_inv)

posterior_ret = M @ (np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q)
posterior_cov = cov_matrix.values + M

# 6. 优化
optimal_weights = optimize_portfolio(posterior_ret, posterior_cov, risk_aversion)

print("=== Black-Litterman配置结果 ===")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# 对比：传统均值方差优化（使用历史收益率）
historical_mean = returns.mean() * 252
traditional_weights = optimize_portfolio(historical_mean.values, cov_matrix.values, risk_aversion)

print("\n=== 传统均值方差优化（对比）===")
for ticker, weight in zip(tickers, traditional_weights):
    print(f"{ticker}: {weight:.2%}")
```

![Black-Litterman vs 传统优化](/images/2026-06-15-black-litterman-model/bl_vs_traditional.jpg)

## Black-Litterman模型的优势

### 1. 稳定性

相比传统均值方差优化，Black-Litterman模型的结果更加稳定：

| 方法 | 权重变化（预期收益±1%） |
|------|------------------------|
| 传统均值方差 | 15-30% |
| Black-Litterman | 3-8% |

### 2. 可解释性

优化结果可以解释为：
- **市场均衡部分**：反映市场共识
- **观点部分**：反映投资者独特见解

### 3. 灵活性

可以轻松纳入各种观点：
- 绝对收益预测
- 相对表现预测
- 资产组合预测

## 常见陷阱与注意事项

### 1. 观点不确定性设置

**错误做法**：所有观点使用相同的Ω
**正确做法**：根据观点置信度分别设置

```python
# 高置信度观点（例：基于深度研究）
omega_high = 0.02**2

# 低置信度观点（例：基于直觉）
omega_low = 0.10**2
```

### 2. τ参数选择

τ决定了市场均衡与观点的权重：
- τ → 0：完全相信市场均衡
- τ → ∞：完全相信投资者观点

**建议**：τ取0.05-0.1

### 3. 协方差矩阵估计

协方差矩阵对结果影响巨大，建议使用：
- **收缩估计量**（Ledoit-Wolf）
- **指数加权移动平均**（EWMA）

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
cov_matrix = lw.fit(returns).covariance_ * 252
```

## 总结

Black-Litterman模型是资产配置领域的里程碑式成果，它：
1. 优雅地融合了市场共识与投资者观点
2. 显著提升了均值方差优化的稳定性
3. 提供了灵活的主观观点表达框架

**实战建议**：
- 从少量高质量观点开始
- 定期回顾观点准确性
- 结合其他资产配置方法（风险平价、因子配置）

---

**参考资料：**
1. Black, F., & Litterman, R. (1992). Global portfolio optimization.
2. He, G., & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios.
3. Idzorek, T. M. (2005). A step-by-step guide to the Black-Litterman model.
