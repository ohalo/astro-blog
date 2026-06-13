---
title: "Markowitz均值方差模型实战：用Python构建最优投资组合"
publishDate: '2026-06-14'
description: "Markowitz均值方差模型实战：用Python构建最优投资组合 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 现代投资组合理论的核心思想

1952年，Harry Markowitz提出了**现代投资组合理论（MPT）**，颠覆了传统投资理念：

> "不要把鸡蛋放在同一个篮子里" —— 但多少人真正理解这句话的数学含义？

### 传统思维 vs Markowitz思维

**传统思维**：挑选最好的股票（高增长、低估值）
**Markowitz思维**：挑选**组合**，使得在给定风险下收益最大，或在给定收益下风险最小

### 两个关键概念

1. **预期收益（Expected Return）**：组合中各股票预期收益的加权平均
   ```
   E(R_p) = Σ w_i * E(R_i)
   ```

2. **风险（Risk）**：用**方差**或**标准差**衡量，但不是简单的加权平均！
   ```
   σ²_p = Σ Σ w_i * w_j * σ_i * σ_j * ρ_ij
   ```
   
   **关键**：资产间的相关性会显著影响组合风险（这就是分散化的数学基础）

## 数学推导：有效前沿的诞生

### 1. 组合预期收益

```
E(R_p) = w' * μ
```

其中：
- `w` = 权重向量 (n x 1)
- `μ` = 预期收益向量 (n x 1)

### 2. 组合风险（方差）

```
σ²_p = w' * Σ * w
```

其中：
- `Σ` = 协方差矩阵 (n x n)

### 3. 优化问题

**目标函数**（最小化风险）：
```
min w' * Σ * w
```

**约束条件**：
```
s.t. w' * μ = R_target  (目标收益)
     w' * 1 = 1          (权重和为1)
     w_i ≥ 0              (不允许卖空，可选)
```

## Python实战：构建中国股票组合

### Step 1: 获取股票数据

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 选取沪深300成分股（示例：银行+科技）
tickers = [
    '600036.SS',  # 招商银行
    '601398.SS',  # 工商银行
    '000858.SZ',  # 五粮液
    '600519.SS',  # 贵州茅台
    '000001.SZ',  # 平安银行
]

# 下载过去3年数据
data = yf.download(tickers, start='2021-01-01', end='2024-12-31')['Adj Close']

# 计算日收益率
returns = data.pct_change().dropna()

print(f"数据形状: {returns.shape}")
print(f"平均日收益:\n{returns.mean()*252}")  # 年化
```

### Step 2: 计算预期收益和协方差

```python
# 年化预期收益（用历史均值作为估计）
mean_returns = returns.mean() * 252

# 年化协方差矩阵
cov_matrix = returns.cov() * 252

# 年化波动率
volatility = returns.std() * np.sqrt(252)

# 计算夏普比率（无风险利率假设2%）
risk_free_rate = 0.02
sharpe_ratio = (mean_returns - risk_free_rate) / volatility

print("\n=== 个股统计 ===")
for ticker in tickers:
    print(f"{ticker}: 年化收益 {mean_returns[ticker]:.2%}, "
          f"波动率 {volatility[ticker]:.2%}, "
          f"夏普 {sharpe_ratio[ticker]:.2f}")
```

### Step 3: 构建有效前沿

```python
def portfolio_performance(weights, mean_returns, cov_matrix):
    """计算组合的预期收益和风险"""
    returns = np.sum(mean_returns * weights)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """负夏普比率（用于最大化夏普）"""
    port_return, port_risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(port_return - risk_free_rate) / port_risk

def portfolio_variance(weights, mean_returns, cov_matrix):
    """组合方差（用于最小化风险）"""
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# 生成有效前沿
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
efficient_portfolios = []

for target in target_returns:
    # 约束条件
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}  # 目标收益
    )
    bounds = tuple((0, 1) for _ in range(len(tickers)))  # 不允许卖空
    initial_guess = len(tickers) * [1. / len(tickers),]
    
    # 最小化风险
    result = minimize(
        portfolio_variance,
        initial_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        efficient_portfolios.append({
            'return': target,
            'risk': np.sqrt(result.fun),
            'weights': result.x
        })

# 转换为DataFrame
efficient_df = pd.DataFrame(efficient_portfolios)
```

### Step 4: 可视化有效前沿

```python
# 绘制有效前沿
plt.figure(figsize=(12, 8))

# 个股
plt.scatter(volatility, mean_returns, s=100, c='red', label='Individual Stocks', zorder=5)
for i, ticker in enumerate(tickers):
    plt.annotate(ticker, (volatility[ticker], mean_returns[ticker]))

# 有效前沿
plt.plot(efficient_df['risk'], efficient_df['return'], 'b-', linewidth=3, label='Efficient Frontier')

# 最小方差组合
min_var_idx = efficient_df['risk'].idxmin()
min_var_return = efficient_df.loc[min_var_idx, 'return']
min_var_risk = efficient_df.loc[min_var_idx, 'risk']
plt.scatter(min_var_risk, min_var_return, s=200, c='green', 
            marker='*', label='Minimum Variance Portfolio', zorder=10)

# 最大夏普组合
sharpe_portfolio = maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
plt.scatter(sharpe_portfolio['risk'], sharpe_portfolio['return'], s=200, c='gold', 
            marker='*', label='Maximum Sharpe Portfolio', zorder=10)

plt.xlabel('Risk (Annualized Volatility)', fontsize=12)
plt.ylabel('Return (Annualized)', fontsize=12)
plt.title('Efficient Frontier - Chinese Stock Portfolio', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 5: 寻找最大夏普组合

```python
def maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """寻找最大夏普比率的组合"""
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    initial_guess = len(tickers) * [1. / len(tickers),]
    
    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        weights = result.x
        ret, risk = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - risk_free_rate) / risk
        return {'weights': weights, 'return': ret, 'risk': risk, 'sharpe': sharpe}
    else:
        return None

# 计算最大夏普组合
max_sharpe = maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)

print("\n=== 最大夏普比率组合 ===")
print(f"预期年化收益: {max_sharpe['return']:.2%}")
print(f"年化波动率: {max_sharpe['risk']:.2%}")
print(f"夏普比率: {max_sharpe['sharpe']:.2f}")
print("\n权重分配:")
for ticker, weight in zip(tickers, max_sharpe['weights']):
    print(f"  {ticker}: {weight:.2%}")
```

## 考虑实际约束的优化

### 1. 限制单一资产权重

```python
# 限制任何资产不超过40%
bounds = tuple((0, 0.4) for _ in range(len(tickers)))

# 或者设置最小权重（避免完全剔除某资产）
bounds = tuple((0.05, 0.4) for _ in range(len(tickers)))
```

### 2. 考虑交易成本

```python
def portfolio_objective_with_costs(weights, mean_returns, cov_matrix, current_weights, transaction_cost=0.002):
    """考虑交易成本的优化目标"""
    # 预期收益
    expected_return = np.sum(mean_returns * weights)
    
    # 风险
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # 交易成本
    turnover = np.sum(np.abs(weights - current_weights))
    cost = transaction_cost * turnover
    
    # 净收益 = 预期收益 - 风险惩罚 - 交易成本
    risk_aversion = 1.0  # 风险厌恶系数
    net_return = expected_return - risk_aversion * risk - cost
    
    return -net_return  # 负号因为要最大化
```

### 3. 引入风险预算约束

```python
def risk_contribution(weights, cov_matrix):
    """计算各资产对组合风险的贡献度"""
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_risk = np.dot(cov_matrix, weights) / port_vol
    risk_contrib = weights * marginal_risk / port_vol
    return risk_contrib

# 约束：单一资产风险贡献不超过30%
constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: 0.3 - risk_contribution(x, cov_matrix)}
)
```

## 中国市场的特殊考虑

### 1. A股的交易制度

```python
# T+1交易制度：当天买入不能当天卖出
# 在回测中需要加入这个约束

def t_plus_one_constraint(weights_today, weights_yesterday):
    """确保不违反T+1制度"""
    # 卖出的量不能超过昨天持有的量
    sell_amount = np.maximum(weights_yesterday - weights_today, 0)
    return np.all(sell_amount <= weights_yesterday + 1e-6)

# 涨跌停限制：±10%（ST股票±5%）
# 在优化时需要加入价格限制的预期
```

### 2. 行业暴露限制

```python
# 避免行业过度集中
sector_map = {
    '600036.SS': 'Finance',
    '601398.SS': 'Finance',
    '000858.SZ': 'Consumer',
    '600519.SS': 'Consumer',
    '000001.SZ': 'Finance'
}

def sector_constraint(weights, sector_map, max_sector_weight=0.6):
    """限制单一行业权重不超过max_sector_weight"""
    sector_weights = {}
    for ticker, weight in zip(tickers, weights):
        sector = sector_map[ticker]
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    return max(sector_weights.values()) <= max_sector_weight

# 加入约束
constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: 0.6 - sector_constraint(x, sector_map, 0.6)}
)
```

### 3. 流动性约束

```python
# 低流动性股票需要更大的交易成本和冲击成本
avg_daily_volume = {
    '600036.SS': 100000000,  # 1亿股
    '601398.SS': 500000000,  # 5亿股
    # ...
}

def liquidity_adjusted_cost(weights, avg_daily_volume, portfolio_value):
    """根据流动性调整交易成本"""
    costs = []
    for ticker, weight in zip(tickers, weights):
        trade_amount = abs(weight * portfolio_value)
        days_to_trade = trade_amount / (avg_daily_volume[ticker] * 0.1)  # 假设每天交易10%成交量
        cost = 0.002 * (1 + days_to_trade)  # 流动性差的成本更高
        costs.append(cost)
    return np.array(costs)
```

## 动态再平衡策略

### 定期再平衡

```python
def periodic_rebalancing(returns, optimal_weights, rebalance_freq='M'):
    """
    定期再平衡回测
    rebalance_freq: 'D'日, 'W'周, 'M'月, 'Q'季
    """
    portfolio_value = 1.0
    portfolio_weights = optimal_weights.copy()
    
    returns_list = []
    dates = []
    
    for date in returns.index:
        # 计算当天收益
        daily_return = np.sum(returns.loc[date] * portfolio_weights)
        portfolio_value *= (1 + daily_return)
        returns_list.append(portfolio_value)
        dates.append(date)
        
        # 更新权重（价格变动导致权重变化）
        portfolio_weights = portfolio_weights * (1 + returns.loc[date]) / (1 + daily_return)
        
        # 定期再平衡
        if date.strftime('%Y-%m') != (date - pd.Timedelta(days=30)).strftime('%Y-%m'):
            portfolio_weights = optimal_weights.copy()
    
    return pd.Series(returns_list, index=dates)
```

### 阈值再平衡

```python
def threshold_rebalancing(returns, optimal_weights, threshold=0.05):
    """
    阈值再平衡：当权重偏离超过threshold时再平衡
    """
    portfolio_value = 1.0
    portfolio_weights = optimal_weights.copy()
    
    returns_list = []
    
    for date in returns.index:
        # 计算当天收益
        daily_return = np.sum(returns.loc[date] * portfolio_weights)
        portfolio_value *= (1 + daily_return)
        returns_list.append(portfolio_value)
        
        # 更新权重
        portfolio_weights = portfolio_weights * (1 + returns.loc[date]) / (1 + daily_return)
        
        # 检查是否需要再平衡
        weight_deviation = np.max(np.abs(portfolio_weights - optimal_weights))
        if weight_deviation > threshold:
            portfolio_weights = optimal_weights.copy()
    
    return pd.Series(returns_list, index=returns.index)
```

## 实战回测与评估

### 完整的回测框架

```python
class PortfolioBacktest:
    def __init__(self, returns, risk_free_rate=0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.results = None
    
    def run_backtest(self, weights, rebalance_freq='M'):
        """运行回测"""
        portfolio_returns = []
        portfolio_weights = []
        
        for i, date in enumerate(self.returns.index):
            if i == 0:
                current_weights = weights.copy()
            else:
                # 更新权重
                daily_return = np.sum(self.returns.loc[date] * current_weights)
                current_weights = current_weights * (1 + self.returns.loc[date]) / (1 + daily_return)
                
                # 再平衡
                if rebalance_freq == 'M' and date.month != self.returns.index[i-1].month:
                    current_weights = weights.copy()
            
            portfolio_returns.append(np.sum(self.returns.loc[date] * current_weights))
            portfolio_weights.append(current_weights)
        
        self.results = {
            'returns': pd.Series(portfolio_returns, index=self.returns.index),
            'weights': pd.DataFrame(portfolio_weights, index=self.returns.index, columns=self.returns.columns)
        }
        return self.results
    
    def calculate_metrics(self):
        """计算绩效指标"""
        if self.results is None:
            raise ValueError("请先运行回测")
        
        port_returns = self.results['returns']
        
        # 年化收益
        annual_return = port_returns.mean() * 252
        
        # 年化波动率
        annual_vol = port_returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = (annual_return - self.risk_free_rate) / annual_vol
        
        # 最大回撤
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (port_returns > 0).sum() / len(port_returns)
        
        return {
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}"
        }

# 使用
backtest = PortfolioBacktest(returns)
results = backtest.run_backtest(max_sharpe['weights'], rebalance_freq='M')
metrics = backtest.calculate_metrics()

print("\n=== 回测结果 ===")
for key, value in metrics.items():
    print(f"{key}: {value}")
```

## 常见陷阱与解决方案

### 陷阱1：代入估计误差（Error Maximization）

**问题**：用历史均值和协方差直接优化，会放大估计误差

**解决方案**：使用收缩估计（Shrinkage Estimation）

```python
from sklearn.covariance import LedoitWolf

# Ledoit-Wolf收缩估计
lw = LedoitWolf()
cov_shrunk = lw.fit(returns).covariance_

# 用收缩估计的协方差矩阵重新优化
max_sharpe_shrunk = maximize_sharpe_ratio(mean_returns, cov_shrunk, risk_free_rate)
```

### 陷阱2：过度集中

**问题**：优化结果可能集中在少数资产上

**解决方案**：加入集中度约束

```python
from scipy.optimize import minimize

def herfindahl_index(weights):
    """赫芬达尔指数（衡量集中度）"""
    return np.sum(weights ** 2)

# 约束：赫芬达尔指数不超过0.3（相当于等权重的3倍集中度）
constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: 0.3 - herfindahl_index(x)}
)
```

### 陷阱3：忽略黑天鹅

**问题**：正态分布假设无法捕捉极端风险

**解决方案**：使用CVaR优化

```python
def cvar_optimization(returns, alpha=0.05):
    """
    基于条件风险价值(CVaR)的优化
    alpha: 置信水平（默认5%，即优化95% CVaR）
    """
    n_scenarios = len(returns)
    
    # 辅助变量：VaR和超额损失
    def objective(x):
        port_returns = returns @ x
        var = np.percentile(port_returns, alpha * 100)
        cvar = port_returns[port_returns <= var].mean()
        return -np.mean(port_returns) + 0.5 * abs(cvar)  # 最大化收益 - 0.5*CVaR
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(returns.shape[1]))
    initial_guess = np.array([1/returns.shape[1]] * returns.shape[1])
    
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

# 使用CVaR优化
cvar_weights = cvar_optimization(returns, alpha=0.05)
```

## 总结与实战建议

### 核心要点

1. **分散化不是简单持有多只股票**，而是通过相关性结构降低风险
2. **有效前沿是理论理想**，实际使用需要结合约束和交易成本
3. **输入参数的准确性**比优化算法更重要（Garbage In, Garbage Out）
4. **再平衡频率**需要在交易成本和偏离成本之间权衡

### 实战建议

1. **用收缩估计代替样本协方差**（Ledoit-Wolf或Oracle Approximating）
2. **限制单一资产和行业的权重**（避免过度集中）
3. **考虑交易成本和流动性**（A股T+1、涨跌停）
4. **定期复盘和重新估计参数**（市场结构会变化）
5. **结合主观判断**（量化模型是工具，不是替代品）

### 下一步学习

- **Black-Litterman模型**：融合市场均衡和投资者观点
- **风险平价策略**：等风险贡献，而非等权重
- **因子投资**：基于风险因子的资产配置

---

**完整代码仓库**：
- [GitHub - Markowitz Portfolio Optimization](https://github.com/halo/quant-portfolio)
- [在线Jupyter Notebook](https://colab.research.google.com/github/halo/quant-portfolio/blob/main/markowitz.ipynb)
- [视频教程](https://www.bilibili.com/video/BV1xx411c7mu)

*下期预告*：Black-Litterman模型——当量化遇见主观判断
