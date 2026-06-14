---
title: "Black-Litterman模型：超越Markowitz的投资组合优化"
publishDate: '2026-06-14'
description: "Black-Litterman模型 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

# Black-Litterman模型：超越Markowitz的投资组合优化

## 为什么需要Black-Litterman？

如果你用过Markowitz均值方差模型，一定遇到过这些头疼的问题：

1. **输入敏感**：预期收益率稍微改一点，最优组合就天差地别
2. **集中化**：优化结果往往集中在少数资产（如某只股票占90%）
3. **估计误差**：历史收益率不是未来的好预测器
4. **缺乏观点**：纯依赖历史数据，无法融入分析师判断

**Black-Litterman模型（BL模型）** 正是为了解决这些痛点而生。它由Fischer Black和Robert Litterman在1992年提出，结合了：

- **CAPM均衡组合**（市场中性，稳定）
- **投资者主观观点**（灵活，个性化）
- **贝叶斯框架**（严谨，可量化不确定性）

高盛在1990年代就将BL模型用于资产配置，现在已成为机构标配。

## Black-Litterman模型的理论框架

### 核心思想

BL模型的逻辑非常优雅：

1. **起点**：以市场均衡组合（如沪深300成分股权重）作为"先验分布"
2. **输入观点**：投资者可以对某些资产表达看多/看空观点
3. **贝叶斯更新**：将观点与先验结合，得到"后验分布"
4. **优化**：用后验预期收益率和协方差矩阵做Mean-Variance优化

**关键优势**：
- 均衡组合提供了稳定的基准（不会极端集中）
- 观点可以是不确定的（用置信度表示）
- 输出结果更直观（偏离均衡的幅度反映观点强度）

### 数学模型

BL模型的核心公式：

$$
E[R] = [( \tau \Sigma )^{-1} + P^T \Omega^{-1} P]^{-1} [ ( \tau \Sigma )^{-1} \Pi + P^T \Omega^{-1} Q ]
$$

其中：
- $E[R]$：后验预期收益率（我们要的结果）
- $\Pi$：先验预期收益率（均衡组合隐含的收益率）
- $\Sigma$：收益率协方差矩阵
- $\tau$：缩放因子（通常取0.05-0.1）
- $P$：观点矩阵（哪些资产有观点）
- $Q$：观点收益向量（看多/看空多少）
- $\Omega$：观点不确定性矩阵（置信度）

**不用被公式吓到**！接下来我们用Python一步步实现。

## Black-Litterman模型的Python实现

### Step 1: 计算均衡组合隐含收益率

均衡组合（Equilibrium Portfolio）通常指市场组合（如沪深300指数）。我们可以用 **逆向优化（Reverse Optimization）** 计算均衡隐含收益率：

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_equilibrium_returns(weights, cov_matrix, risk_aversion=2.5):
    """
    逆向优化：从权重反推预期收益率
    weights: 市场组合权重（如沪深300成分股权重）
    cov_matrix: 收益率协方差矩阵
    risk_aversion: 风险厌恶系数（通常2.5-3.0）
    """
    # Pi = risk_aversion * Sigma * w
    pi = risk_aversion * np.dot(cov_matrix, weights)
    
    return pi

# 示例：假设我们有5只股票
stocks = ['股票A', '股票B', '股票C', '股票D', '股票E']
n_assets = len(stocks)

# 市场权重（模拟沪深300成分股权重）
market_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

# 协方差矩阵（模拟）
cov_matrix = np.array([
    [0.04, 0.02, 0.01, 0.005, 0.002],
    [0.02, 0.05, 0.015, 0.008, 0.003],
    [0.01, 0.015, 0.03, 0.01, 0.004],
    [0.005, 0.008, 0.01, 0.035, 0.006],
    [0.002, 0.003, 0.004, 0.006, 0.025]
])

# 计算均衡隐含收益率
risk_aversion = 2.5
pi = calculate_equilibrium_returns(market_weights, cov_matrix, risk_aversion)

print("均衡隐含收益率：")
for i, stock in enumerate(stocks):
    print(f"{stock}: {pi[i]:.4f} ({pi[i]*100:.2f}%)")
```

输出示例：
```
股票A: 0.0425 (4.25%)
股票B: 0.0475 (4.75%)
...
```

### Step 2: 定义投资者观点

BL模型最强大的地方是 **允许不确定的观点**。我们可以表达：

- **绝对观点**：我认为股票A的年化收益是15%
- **相对观点**：我认为股票A会比股票B多涨5%

用矩阵表示：

```python
# 观点矩阵 P (n_views x n_assets)
# 每一行代表一个观点，每一列代表一只股票

# 示例：两个观点
# 观点1：股票A年化收益15%
# 观点2：股票A比股票B多涨5%

P = np.array([
    [1, 0, 0, 0, 0],   # 观点1：100%配置在股票A
    [1, -1, 0, 0, 0]   # 观点2：股票A - 股票B = 5%
])

# 观点收益向量 Q (n_views x 1)
Q = np.array([0.15, 0.05])  # 15% 和 5%

# 观点不确定性 Omega (n_views x n_views)
# 对角线元素表示每个观点的置信度（越小越自信）
tau = 0.05
omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

print("观点矩阵 P：")
print(P)
print("\n观点收益 Q：")
print(Q)
print("\n观点不确定性 Omega：")
print(omega)
```

### Step 3: 计算后验预期收益率

这是BL模型的核心公式：

```python
def black_litterman(pi, cov_matrix, P, Q, tau=0.05):
    """
    Black-Litterman模型主函数
    pi: 均衡隐含收益率
    cov_matrix: 协方差矩阵
    P: 观点矩阵
    Q: 观点收益向量
    tau: 缩放因子
    """
    n = len(pi)
    
    # 计算观点不确定性 Omega
    omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))
    omega_inv = np.linalg.inv(omega)
    
    # BL公式
    M_inv = np.linalg.inv(tau * cov_matrix) + P.T @ omega_inv @ P
    M = np.linalg.inv(M_inv)
    
    posterior_return = M @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ omega_inv @ Q)
    
    # 后验协方差矩阵
    posterior_cov = M @ (np.linalg.inv(tau * cov_matrix) @ cov_matrix @ np.linalg.inv(tau * cov_matrix) + P.T @ omega_inv @ P) @ M
    
    return posterior_return, posterior_cov

# 计算后验收益率
posterior_return, posterior_cov = black_litterman(pi, cov_matrix, P, Q, tau)

print("\n后验预期收益率（BL调整后）：")
for i, stock in enumerate(stocks):
    print(f"{stock}: {posterior_return[i]:.4f} ({posterior_return[i]*100:.2f}%)")

print("\n后验协方差矩阵：")
print(posterior_cov)
```

**关键观察**：
- 股票A的后验收益率会 **上调**（因为我们有看多观点）
- 股票B的后验收益率会 **下调**（因为相对观点中A>B）
- 调整幅度取决于观点的置信度（Omega）

### Step 4: 投资组合优化

现在用后验收益率做Mean-Variance优化：

```python
def portfolio_optimization(expected_returns, cov_matrix, risk_aversion=2.5):
    """
    均值方差优化
    """
    n = len(expected_returns)
    
    # 目标函数：最大化效用 = 预期收益 - 0.5 * risk_aversion * 风险
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.dot(weights, np.dot(cov_matrix, weights))
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk
        return -utility  # scipy最小化，所以加负号
    
    # 约束：权重和为1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 边界：不允许做空（可选）
    bounds = tuple((0, 1) for _ in range(n))
    
    # 初始权重：等权重
    initial_weights = np.array([1/n] * n)
    
    # 优化
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

# 用BL后验收益率优化
optimal_weights_bl = portfolio_optimization(posterior_return, posterior_cov, risk_aversion)

print("\nBL模型最优权重：")
for i, stock in enumerate(stocks):
    print(f"{stock}: {optimal_weights_bl[i]:.4f} ({optimal_weights_bl[i]*100:.2f}%)")

# 对比：用历史收益率优化（传统Markowitz）
historical_returns = np.array([0.12, 0.10, 0.08, 0.06, 0.04])  # 模拟
optimal_weights_mv = portfolio_optimization(historical_returns, cov_matrix, risk_aversion)

print("\n传统Markowitz最优权重：")
for i, stock in enumerate(stocks):
    print(f"{stock}: {optimal_weights_mv[i]:.4f} ({optimal_weights_mv[i]*100:.2f}%)")
```

**典型结果**：
- **BL模型**：权重相对均衡（如30%, 25%, 20%, 15%, 10%）
- **Markowitz**：权重极端（如90%, 0%, 0%, 5%, 5%）

这就是BL模型的魅力！

## Black-Litterman模型的中国实战

### 案例：沪深300成分股配置

假设我们要配置沪深300前10大成分股，步骤如下：

#### 1. 数据准备

```python
import tushare as ts
import pandas as pd

# 获取沪深300成分股
pro = ts.pro_api('your_token')
index_weight = pro.index_weight(index_code='000300.SH', start_date='20260101', end_date='20260131')
top10_stocks = index_weight.groupby('con_code')['weight'].mean().nlargest(10)

print("沪深300前10大成分股权重：")
print(top10_stocks)
```

#### 2. 计算协方差矩阵

```python
# 获取过去252个交易日收益率
returns_df = pd.DataFrame()

for stock in top10_stocks.index:
    df = pro.daily(ts_code=stock, start_date='20250101', end_date='20260101')
    returns_df[stock] = df.set_index('trade_date')['close'].pct_change()

# 年化协方差矩阵
cov_matrix = returns_df.cov() * 252
```

#### 3. 输入观点

假设我们有这些观点：
- **观点1**：新能源板块（宁德时代、比亚迪）未来半年跑赢大盘10%
- **观点2**：金融板块（中国平安、招商银行）相对表现持平
- **观点3**：茅台未来半年收益15%

```python
# 构建观点矩阵
stocks = top10_stocks.index.tolist()
n = len(stocks)

P = []
Q = []

# 观点1：新能源跑赢大盘10%
ev_stocks = ['300750.SZ', '002594.SZ']  # 宁德时代、比亚迪
ev_idx = [stocks.index(s) for s in ev_stocks]
p1 = np.zeros(n)
p1[ev_idx] = 1 / len(ev_idx)  # 等权重平均
P.append(p1)
Q.append(0.10)

# 观点2：金融板块持平（相对观点，略）
# 观点3：茅台15%
moutai_idx = stocks.index('600519.SH')
p3 = np.zeros(n)
p3[moutai_idx] = 1
P.append(p3)
Q.append(0.15)

P = np.array(P)
Q = np.array(Q)
```

#### 4. 运行BL模型

```python
# 市场权重
market_weights = top10_stocks.values

# 均衡隐含收益率
pi = calculate_equilibrium_returns(market_weights, cov_matrix, risk_aversion=2.5)

# BL后验收益率
posterior_return, posterior_cov = black_litterman(pi, cov_matrix, P, Q, tau=0.05)

# 优化
optimal_weights = portfolio_optimization(posterior_return, posterior_cov, risk_aversion=2.5)

# 对比市场权重
comparison = pd.DataFrame({
    '市场权重': market_weights,
    'BL最优权重': optimal_weights
}, index=stocks)

print("\n市场权重 vs BL最优权重：")
print(comparison)
print(f"\n权重变化（BL - 市场）：")
print(comparison['BL最优权重'] - comparison['市场权重'])
```

**输出示例**：
```
            市场权重  BL最优权重
600519.SH   0.15     0.22  # 茅台上调
300750.SZ   0.08     0.12  # 宁德时代上调
002594.SZ   0.06     0.09  # 比亚迪上调
601318.SH   0.12     0.10  # 平安下调
...
```

### 实战要点

1. **观点要少而精**：不要对所有股票都表达观点，聚焦最确定的2-3个
2. **置信度要合理**：不确定的观点用大Omega（低置信度），避免过度偏离市场
3. **定期更新**：观点不是永恒的，每月/季度重新评估
4. **结合风控**：BL输出也要检查集中度、行业偏离等

## Black-Litterman模型的进阶话题

### 1. 相对观点 vs 绝对观点

BL模型支持两种观点：

- **绝对观点**：股票A收益15%（P是一行向量）
- **相对观点**：股票A比股票B多涨5%（P是两列差分）

**实践建议**：
- 相对观点更稳健（不受市场整体影响）
- 绝对观点需要更强的置信度

### 2. 观点不确定性的设定

Omega的设定有几种方法：

```python
# 方法1：与预测误差成正比
omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

# 方法2：手动设定（观点1置信度高，观点2置信度低）
omega = np.diag([0.01, 0.05])  # 越小越自信

# 方法3：基于历史准确率
# 如果过去类似观点的准确率是80%，可以设定omega = 0.2 * P @ Sigma @ P.T
```

### 3. 结合宏观因子

BL模型可以扩展到 **宏观因子**（如利率、通胀、GDP）：

```python
# 宏观因子观点
macro_factors = ['利率', '通胀', 'GDP']
F = np.array([
    [0.5, 0.3, 0.2],  # 利率对股票收益的影响
    [-0.3, 0.6, 0.1], # 通胀的影响
    [0.4, 0.2, 0.7]   # GDP的影响
])

# 宏观观点：我认为利率会下降1%
macro_view = np.array([-0.01, 0, 0])  # 利率-1%，通胀和GDP不变

# 将宏观观点转换为股票收益观点
Q_macro = F @ macro_view
P_macro = np.eye(n_assets)  # 对每个股票都有影响

# 结合微观和宏观观点
P_combined = np.vstack([P_micro, P_macro])
Q_combined = np.hstack([Q_micro, Q_macro])
```

### 4. 动态BL模型

传统BL是静态的（每月更新一次观点）。可以改进为 **动态BL**：

- 用卡尔曼滤波跟踪观点准确性的变化
- 用GARCH模型预测时变协方差矩阵
- 用机器学习预测观点收益Q

```python
# 动态BL伪代码
for t in range(T):
    # 更新协方差矩阵（GARCH）
    cov_matrix_t = garch_forecast(returns[:t])
    
    # 更新观点置信度（卡尔曼滤波）
    omega_t = kalman_update(omega_0, historical_accuracy[:t])
    
    # 运行BL
    posterior_return_t, _ = black_litterman(pi, cov_matrix_t, P, Q, tau, omega_t)
    
    # 优化
    weights_t = portfolio_optimization(posterior_return_t, cov_matrix_t)
```

## 常见陷阱与解决方案

### 陷阱1：观点过多

**问题**：对20只股票都表达观点，导致过度拟合

**解决**：
- 只表达3-5个高置信度观点
- 用PCA降维，只对主成分表达观点

### 陷阱2：Omega设定不当

**问题**：Omega太小（过度自信），导致组合极端集中

**解决**：
- 用交叉验证选择最优Omega
- 设定Omega的下限（如不小于0.01）

### 陷阱3：忽略交易成本

**问题**：BL模型建议大幅调仓，但交易成本吃掉收益

**解决**：
```python
# 在目标函数中加入交易成本
def objective_with_cost(weights, prev_weights):
    turnover = np.sum(np.abs(weights - prev_weights))
    transaction_cost = turnover * 0.001  # 假设单边成本0.1%
    
    portfolio_return = np.dot(weights, expected_returns) - transaction_cost
    portfolio_risk = np.dot(weights, np.dot(cov_matrix, weights))
    
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk
    return -utility
```

## 总结

Black-Litterman模型是投资组合优化的里程碑，核心要点：

1. **理论基础**：贝叶斯框架，结合市场均衡与投资者观点
2. **核心优势**：稳定、直观、灵活
3. **实施步骤**：计算均衡收益 → 输入观点 → BL公式 → 优化
4. **实战建议**：观点要少而精，置信度要合理，定期更新
5. **进阶方向**：宏观因子、动态模型、机器学习结合

**对比Markowitz**：
- Markowitz：纯依赖历史数据，结果不稳定
- Black-Litterman：结合主观判断，结果更稳健

**下期预告**：我们将介绍 **风险平价策略（Risk Parity）**，探讨如何放弃收益预测，纯粹从风险角度构建稳健组合。

---

*本文代码仅供参考，实盘使用前请充分回测和风控。BL模型对观点质量依赖较高，垃圾进垃圾出（GIGO）。*
