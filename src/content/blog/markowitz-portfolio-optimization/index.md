---
title: 马科维茨均值方差优化实战：用Python构建高效投资组合
publishDate: '2026-06-02'
description: 马科维茨均值方差优化实战：用Python构建高效投资组合 - halo的技术博客
tags:
  - 量化交易
language: Chinese
difficulty: intermediate
---

## 现代投资组合理论的基石

1952年，哈里·马科维茨（Harry Markowitz）在《金融杂志》发表了题为《投资组合选择》的论文，标志着现代投资组合理论（Modern Portfolio Theory, MPT）的诞生。这一理论彻底改变了投资者对风险和收益的理解，从单一资产分析转向资产配置的系统化方法。

### 核心思想：分散化降低风险

马科维茨理论的核心洞察是：**投资者应该关注投资组合的整体风险，而非单个资产的风险**。通过组合相关性较低甚至负相关的资产，可以在不降低预期收益的情况下降低投资组合的整体风险。

![马科维茨均值方差模型示意图](/images/markowitz-portfolio-optimization/markowitz_efficient_frontier.svg)

## 均值方差优化的数学框架

### 1. 基本假设

马科维茨模型基于以下关键假设：

- **投资者是风险厌恶的**：在相同的预期收益下，投资者偏好风险更小的投资组合
- **收益率服从正态分布**：可以用均值和方差完全描述
- **投资者只关注单一持有期**：通常以期望收益率和标准差作为决策依据
- **市场是有效的**：所有投资者都能免费获得相同的信息

### 2. 数学模型

对于由 $n$ 个资产组成的投资组合，定义：

- $w_i$ : 资产 $i$ 的权重，$\sum_{i=1}^n w_i = 1$
- $\mu_i$ : 资产 $i$ 的预期收益率
- $\sigma_i$ : 资产 $i$ 收益率的标准差
- $\sigma_{ij}$ : 资产 $i$ 和 $j$ 收益率的协方差

**投资组合的预期收益率**：

$$
E(R_p) = \sum_{i=1}^n w_i \mu_i
$$

**投资组合的方差**：

$$
\sigma_p^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij} = \mathbf{w}^T \Sigma \mathbf{w}
$$

其中 $\Sigma$ 是 $n \times n$ 的协方差矩阵。

### 3. 优化问题

均值方差优化可以表示为以下数学规划问题：

**最小化风险（方差）给定收益目标**：

$$
\begin{aligned}
\min_{\mathbf{w}} \quad & \mathbf{w}^T \Sigma \mathbf{w} \\
\text{s.t.} \quad & \mathbf{w}^T \boldsymbol{\mu} = \mu_p \\
& \sum_{i=1}^n w_i = 1 \\
& w_i \geq 0 \quad (\text{不允许卖空})
\end{aligned}
$$

## Python实战：构建均值方差优化模型

下面用Python完整实现一个均值方差优化系统，使用真实股票数据。

### 1. 数据准备

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 选择股票池（沪深300成分股示例）
tickers = ['600519.SS', '000858.SZ', '601318.SS', '600036.SS', '000333.SZ',
           '601288.SS', '600900.SS', '000002.SZ', '601166.SS', '600016.SS']

# 下载历史数据
start_date = '2023-01-01'
end_date = '2025-12-31'

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 计算日收益率
returns = data.pct_change().dropna()

print(f"数据形状: {returns.shape}")
print(f"时间范围: {returns.index[0]} 到 {returns.index[-1]}")
print(f"股票数量: {len(tickers)}")
```

### 2. 计算期望收益和协方差矩阵

```python
# 计算年化收益率（假设252个交易日）
annual_factor = 252
mean_returns = returns.mean() * annual_factor
cov_matrix = returns.cov() * annual_factor

# 计算相关系数矩阵
corr_matrix = returns.corr()

print("年化预期收益率:")
print(mean_returns.round(4))
print("\n年化波动率:")
print(np.sqrt(np.diag(cov_matrix)).round(4))
```

### 3. 投资组合优化函数

```python
def portfolio_performance(weights, mean_returns, cov_matrix):
    """计算投资组合的收益和风险"""
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    """负夏普比率（用于优化）"""
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

def portfolio_variance(weights, cov_matrix):
    """投资组合方差"""
    return np.dot(weights.T, np.dot(cov_matrix, weights))
```

### 4. 约束条件和边界

```python
# 约束条件：权重之和为1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# 边界条件：不允许卖空（权重在0到1之间）
bounds = tuple((0, 1) for _ in range(len(tickers)))

# 初始猜测：等权重
initial_weights = np.array([1/len(tickers)] * len(tickers))
```

### 5. 三种优化策略

#### (1) 最小化风险（全局最小方差组合）

```python
def minimize_risk(cov_matrix, constraints, bounds, initial_weights):
    """最小化投资组合方差"""
    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

# 执行优化
min_risk_result = minimize_risk(cov_matrix, constraints, bounds, initial_weights)

if min_risk_result.success:
    min_risk_weights = min_risk_result.x
    min_risk_return, min_risk_std = portfolio_performance(
        min_risk_weights, mean_returns, cov_matrix
    )
    print("全局最小方差组合:")
    print(f"预期收益率: {min_risk_return:.4f}")
    print(f"波动率: {min_risk_std:.4f}")
    print(f"夏普比率: {(min_risk_return - 0.03) / min_risk_std:.4f}")
```

#### (2) 最大化夏普比率

```python
def maximize_sharpe_ratio(mean_returns, cov_matrix, constraints, bounds, initial_weights):
    """最大化夏普比率"""
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

# 执行优化
max_sharpe_result = maximize_sharpe_ratio(
    mean_returns, cov_matrix, constraints, bounds, initial_weights
)

if max_sharpe_result.success:
    max_sharpe_weights = max_sharpe_result.x
    max_sharpe_return, max_sharpe_std = portfolio_performance(
        max_sharpe_weights, mean_returns, cov_matrix
    )
    print("\n最大夏普比率组合:")
    print(f"预期收益率: {max_sharpe_return:.4f}")
    print(f"波动率: {max_sharpe_std:.4f}")
    print(f"夏普比率: {(max_sharpe_return - 0.03) / max_sharpe_std:.4f}")
```

#### (3) 给定收益目标下最小化风险

```python
def minimize_risk_for_target_return(target_return, mean_returns, cov_matrix, 
                                   constraints, bounds, initial_weights):
    """给定收益目标下最小化风险"""
    # 添加收益约束
    cons = list(constraints)
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return})
    
    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )
    return result

# 示例：目标年化收益率15%
target_return = 0.15
target_result = minimize_risk_for_target_return(
    target_return, mean_returns, cov_matrix, 
    constraints, bounds, initial_weights
)

if target_result.success:
    target_weights = target_result.x
    target_return_actual, target_std = portfolio_performance(
        target_weights, mean_returns, cov_matrix
    )
    print(f"\n目标收益{target_return:.2%}下的最小风险组合:")
    print(f"实际预期收益率: {target_return_actual:.4f}")
    print(f"波动率: {target_std:.4f}")
```

## 构建有效前沿

有效前沿（Efficient Frontier）是均值方差优化的核心概念，表示在给定风险水平下能够获得最大预期收益的所有投资组合。

```python
def generate_efficient_frontier(mean_returns, cov_matrix, constraints, bounds, 
                               initial_weights, num_portfolios=100):
    """生成有效前沿"""
    results = []
    weights_list = []
    
    # 确定收益范围
    min_ret = min(mean_returns)
    max_ret = max(mean_returns)
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    for target in target_returns:
        result = minimize_risk_for_target_return(
            target, mean_returns, cov_matrix, 
            constraints, bounds, initial_weights
        )
        if result.success:
            weights = result.x
            ret, std = portfolio_performance(weights, mean_returns, cov_matrix)
            results.append([ret, std])
            weights_list.append(weights)
    
    return np.array(results), weights_list

# 生成有效前沿
ef_results, ef_weights = generate_efficient_frontier(
    mean_returns, cov_matrix, constraints, bounds, initial_weights
)

# 绘制有效前沿
plt.figure(figsize=(12, 8))

# 绘制随机投资组合（用于对比）
num_random = 5000
random_returns = []
random_stds = []

for _ in range(num_random):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    ret, std = portfolio_performance(weights, mean_returns, cov_matrix)
    random_returns.append(ret)
    random_stds.append(std)

plt.scatter(random_stds, random_returns, c='gray', alpha=0.3, s=10, label='随机组合')

# 绘制有效前沿
plt.plot(ef_results[:, 1], ef_results[:, 0], 'r-', linewidth=3, label='有效前沿')

# 标记特殊点
plt.scatter(min_risk_std, min_risk_return, c='green', s=100, marker='*', 
            label='全局最小方差', zorder=5)
plt.scatter(max_sharpe_std, max_sharpe_return, c='blue', s=100, marker='*', 
            label='最大夏普比率', zorder=5)

plt.xlabel('波动率 (风险)', fontsize=12)
plt.ylabel('预期收益率', fontsize=12)
plt.title('投资组合有效前沿', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/markowitz-portfolio-optimization/markowitz_efficient_frontier.svg', 
            format='svg', bbox_inches='tight')
plt.show()
```

![投资组合有效前沿可视化](/images/markowitz-portfolio-optimization/markowitz_efficient_frontier.svg)

## 实战中的关键问题

### 1. 估计误差问题

均值方差优化对输入参数（预期收益率和协方差矩阵）非常敏感。历史数据估计的参数可能存在较大误差，导致优化结果不稳定。

**解决方案**：
- 使用收缩估计量（Shrinkage Estimator）改进协方差矩阵估计
- 使用黑-利特曼（Black-Litterman）模型结合投资者观点
- 采用稳健优化方法

```python
from sklearn.covariance import LedoitWolf

# 使用Ledoit-Wolf收缩估计
lw = LedoitWolf()
lw.fit(returns)
shrunk_cov = lw.covariance_ * annual_factor

print("原始协方差矩阵条件数:", np.linalg.cond(cov_matrix))
print("收缩后协方差矩阵条件数:", np.linalg.cond(shrunk_cov))
```

### 2. 卖空限制和权重约束

实际应用中通常不允许卖空，且可能对单个资产权重设限。

```python
# 带权重上限的优化
def optimize_with_weight_limit(max_weight, mean_returns, cov_matrix):
    """限制单个资产最大权重"""
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    )
    bounds = tuple((0, max_weight) for _ in range(len(tickers)))
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

# 限制单个资产不超过30%
result_limit = optimize_with_weight_limit(0.3, mean_returns, cov_matrix)

if result_limit.success:
    weights_limit = result_limit.x
    print("带权重限制（最大30%）的最优组合:")
    for i, (ticker, weight) in enumerate(zip(tickers, weights_limit)):
        if weight > 0.01:  # 只显示权重超过1%的资产
            print(f"{ticker}: {weight:.2%}")
```

### 3. 交易成本考虑

频繁调仓会产生交易成本，应在优化中考虑。

```python
def portfolio_turnover_cost(old_weights, new_weights, transaction_cost_rate=0.001):
    """计算调仓成本"""
    turnover = np.sum(np.abs(new_weights - old_weights))
    cost = turnover * transaction_cost_rate
    return cost

# 示例：从等权重组合调整到最优组合
old_weights = np.array([1/len(tickers)] * len(tickers))
transaction_cost = portfolio_turnover_cost(old_weights, max_sharpe_weights)

print(f"调仓交易成本: {transaction_cost:.4f} ({transaction_cost:.2%})")
print(f"净预期收益率: {max_sharpe_return - transaction_cost:.4f}")
```

## 绩效评估

构建投资组合后，需要进行严格的绩效评估。

```python
def backtest_portfolio(weights, returns, mean_returns, cov_matrix):
    """回测投资组合表现"""
    portfolio_returns = returns.dot(weights)
    
    # 计算累计收益
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 计算风险指标
    annual_return = portfolio_returns.mean() * annual_factor
    annual_vol = portfolio_returns.std() * np.sqrt(annual_factor)
    sharpe = (annual_return - 0.03) / annual_vol
    
    # 计算最大回撤
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'cumulative_returns': cumulative_returns,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_returns': portfolio_returns
    }

# 回测最大夏普比率组合
backtest_results = backtest_portfolio(max_sharpe_weights, returns, mean_returns, cov_matrix)

print("=== 投资组合回测结果 ===")
print(f"年化收益率: {backtest_results['annual_return']:.2%}")
print(f"年化波动率: {backtest_results['annual_volatility']:.2%}")
print(f"夏普比率: {backtest_results['sharpe_ratio']:.4f}")
print(f"最大回撤: {backtest_results['max_drawdown']:.2%}")

# 绘制累计收益曲线
plt.figure(figsize=(12, 6))
plt.plot(backtest_results['cumulative_returns'], linewidth=2, label='最大夏普比率组合')

# 绘制基准（等权重组合）
benchmark_weights = np.array([1/len(tickers)] * len(tickers))
benchmark_returns = returns.dot(benchmark_weights)
benchmark_cumulative = (1 + benchmark_returns).cumprod()
plt.plot(benchmark_cumulative, linewidth=2, label='等权重组合', linestyle='--')

plt.xlabel('日期', fontsize=12)
plt.ylabel('累计收益', fontsize=12)
plt.title('投资组合累计收益对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/markowitz-portfolio-optimization/cumulative_returns.svg', 
            format='svg', bbox_inches='tight')
plt.show()
```

![投资组合累计收益对比](/images/markowitz-portfolio-optimization/cumulative_returns.svg)

## 结论与最佳实践

### 主要发现

1. **分散化价值**：通过组合相关性较低的资产，可以显著降低投资组合风险
2. **风险-收益权衡**：有效前沿直观展示了不同风险水平下的最优收益
3. **参数敏感性**：优化结果对预期收益率和协方差估计非常敏感

### 实战建议

1. **使用稳健的估计方法**：采用收缩估计量或指数加权协方差矩阵
2. **考虑实际约束**：加入卖空限制、权重上限、交易成本等实际约束
3. **定期再平衡**：设定再平衡频率（如季度或半年），避免过度交易
4. **结合定性判断**：量化模型应作为决策工具，而非完全替代人工判断
5. **压力测试**：使用历史危机时期数据测试投资组合稳健性

### 局限性

1. **正态分布假设**：实际收益率往往具有肥尾特征
2. **静态优化**：传统均值方差优化是单期模型，不考虑动态调整
3. **参数不确定性**：历史估计的参数在未来可能失效

## 完整代码资源

本文完整Python代码已上传至GitHub，包含：
- 数据获取模块
- 均值方差优化核心函数
- 有效前沿可视化
- 回测框架
- 绩效评估报告

通过系统掌握均值方差优化，投资者可以构建更科学、更稳健的投资组合，在控制风险的同时追求长期稳定收益。

---

**参考文献**：
1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
2. Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *The Journal of Portfolio Management*, 30(4), 110-119.
3. Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.
