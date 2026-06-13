---
title: "VaR与CVaR风险度量：量化投资的风险标尺"
publishDate: '2026-06-14'
description: "VaR与CVaR风险度量：量化投资的风险标尺 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 为什么需要风险度量？

在量化投资中，**收益只是硬币的一面，风险才是另一面**。一个策略即使有再高的预期收益，如果风险失控，也可能在一夜之间灰飞烟灭。

2008年金融危机中，无数"稳健"的量化基金爆仓，根本原因就是**低估了极端风险**。传统的波动率指标无法捕捉"黑天鹅"事件，而**VaR（风险价值）和CVaR（条件风险价值）**正是为了解决这个问题而生。

## VaR：风险价值的直观理解

### 什么是VaR？

**VaR（Value at Risk，风险价值）**回答的是一个简单的问题：

> "在正常的市场条件下，我在95%或99%的置信水平下，未来一天/一周/一月**最多会亏多少钱**？"

**举个例子**：
- 你的投资组合VaR(95%, 1天) = 10万元
- 含义：在95%的情况下，明天你的损失**不会超过**10万元
- 换句话说：有5%的概率，明天的损失会**超过**10万元

### VaR的三种计算方法

#### 1. 历史模拟法（Historical Simulation）

**核心思想**：用过去的实际收益率分布来预测未来风险。

**计算步骤**：
1. 收集过去1000个交易日的收益率数据
2. 将这些收益率从低到高排序
3. 找到第5百分位数（95%置信水平）或第1百分位数（99%置信水平）

**Python实现**：
```python
import numpy as np
import pandas as pd

def historical_var(returns, confidence_level=0.95):
    """
    历史模拟法计算VaR
    
    Parameters:
    -----------
    returns: pd.Series or np.array
        收益率序列
    confidence_level: float
        置信水平，默认0.95
    
    Returns:
    --------
    var: float
        VaR值（正数）
    """
    # 计算分位数
    var = np.percentile(returns, 100 * (1 - confidence_level))
    return abs(var)

# 示例
np.random.seed(42)
returns = np.random.normal(0.0005, 0.02, 1000)  # 模拟1000个交易日的收益率
var_95 = historical_var(returns, 0.95)
var_99 = historical_var(returns, 0.99)

print(f"VaR(95%, 1天) = {var_95:.4f} = {var_95*100:.2f}%")
print(f"VaR(99%, 1天) = {var_99:.4f} = {var_99*100:.2f}%")
```

**优点**：
- 简单直观，不需要假设收益率分布
- 能捕捉非正态分布的特征（如肥尾）

**缺点**：
- 假设"未来会重复过去"，对结构性变化不敏感
- 需要大量历史数据（至少3-5年）

#### 2. 参数法（Parametric / Delta-Normal）

**核心思想**：假设收益率服从某个人知分布（通常是正态分布），用均值和方差来计算VaR。

**计算公式**（正态分布假设）：
```
VaR(α%, t天) = -μ_t + z_α * σ_t
```
其中：
- μ_t = t天的预期收益率
- σ_t = t天的波动率
- z_α = 标准正态分布的α%分位数（如95%对应1.645，99%对应2.33）

**Python实现**：
```python
from scipy import stats

def parametric_var(returns, confidence_level=0.95, holding_period=1):
    """
    参数法计算VaR（正态分布假设）
    
    Parameters:
    -----------
    returns: pd.Series
        收益率序列
    confidence_level: float
        置信水平
    holding_period: int
        持有期（天数）
    
    Returns:
    --------
    var: float
        VaR值
    """
    # 计算均值和标准差
    mu = returns.mean()
    sigma = returns.std()
    
    # 调整到持有期
    mu_t = mu * holding_period
    sigma_t = sigma * np.sqrt(holding_period)
    
    # 计算分位数
    z_alpha = stats.norm.ppf(1 - confidence_level)
    
    # 计算VaR
    var = -(mu_t + z_alpha * sigma_t)
    return var

# 示例
var_95_param = parametric_var(pd.Series(returns), 0.95, 1)
print(f"参数法 VaR(95%, 1天) = {var_95_param:.4f}")
```

**优点**：
- 计算速度快
- 可以轻松调整持有期（√t法则）

**缺点**：
- **致命缺陷**：假设正态分布，低估肥尾风险
- 对均值和方差的估计误差敏感

#### 3. 蒙特卡洛模拟法（Monte Carlo Simulation）

**核心思想**：用随机模拟生成成千上万种可能的市场情景，再计算分位数。

**计算步骤**：
1. 假设收益率服从某个随机过程（如几何布朗运动、GARCH等）
2. 用历史数据估计模型参数
3. 模拟10,000次未来的收益率路径
4. 计算模拟收益率的分位数

**Python实现**（几何布朗运动）：
```python
def monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000, holding_period=1):
    """
    蒙特卡洛模拟法计算VaR
    
    Parameters:
    -----------
    returns: pd.Series
        历史收益率
    confidence_level: float
        置信水平
    num_simulations: int
        模拟次数
    holding_period: int
        持有期
    
    Returns:
    --------
    var: float
        VaR值
    """
    # 估计参数
    mu = returns.mean()
    sigma = returns.std()
    
    # 生成随机收益率（几何布朗运动）
    np.random.seed(42)
    random_returns = np.random.normal(mu, sigma, (num_simulations, holding_period))
    
    # 计算模拟的持有期收益率
    simulated_returns = random_returns.sum(axis=1)
    
    # 计算VaR
    var = np.percentile(simulated_returns, 100 * (1 - confidence_level))
    return abs(var)

# 示例
var_95_mc = monte_carlo_var(pd.Series(returns), 0.95, 10000, 1)
print(f"蒙特卡洛法 VaR(95%, 1天) = {var_95_mc:.4f}")
```

**优点**：
- 可以模拟各种复杂的收益分布和依赖结构
- 能处理非线性资产（如期权）

**缺点**：
- 计算成本高
- 结果依赖于模型假设（垃圾进，垃圾出）

### VaR的致命缺陷：不捕捉尾部风险

尽管VaR被广泛使用，但它有一个**致命缺陷**：

> **VaR只告诉你"最坏情况下会亏多少"，但不告诉你"如果超过VaR会亏多少"**。

举个例子：
- 组合A：99%概率亏1万，1%概率亏2万 → VaR(99%) = 1万
- 组合B：99%概率亏1万，1%概率亏100万 → VaR(99%) = 1万

**两个组合的VaR一样，但风险完全不同！**

这就是为什么2008年金融危机中，许多银行的VaR模型"正常"，但却遭遇了远超VaR的损失。

## CVaR：捕捉极端风险

### 什么是CVaR？

**CVaR（Conditional Value at Risk，条件风险价值）**，也称为**期望损失（Expected Shortfall）**，回答的是：

> "当损失超过VaR时，**平均会亏多少钱**？"

**数学定义**：
```
CVaR_α = E[L | L > VaR_α]
```
其中L是损失，VaR_α是α%置信水平的VaR。

**直观理解**：
- VaR是"门槛"：95%的情况下，损失不会超过这个门槛
- CVaR是"平均超量损失"：当损失超过门槛时，平均会亏多少

### CVaR的计算

#### 历史模拟法计算CVaR

```python
def historical_cvar(returns, confidence_level=0.95):
    """
    历史模拟法计算CVaR
    
    Parameters:
    -----------
    returns: pd.Series or np.array
        收益率序列（负值表示损失）
    confidence_level: float
        置信水平
    
    Returns:
    --------
    cvar: float
        CVaR值
    """
    # 计算VaR
    var = np.percentile(returns, 100 * (1 - confidence_level))
    
    # 提取超过VaR的损失
    exceed_losses = returns[returns <= var]
    
    # 计算平均超量损失
    cvar = exceed_losses.mean()
    return abs(cvar)

# 示例
cvar_95 = historical_cvar(returns, 0.95)
cvar_99 = historical_cvar(returns, 0.99)

print(f"CVaR(95%, 1天) = {cvar_95:.4f} = {cvar_95*100:.2f}%")
print(f"CVaR(99%, 1天) = {cvar_99:.4f} = {cvar_99*100:.2f}%")
```

### 为什么CVaR比VaR更好？

1. **捕捉尾部风险**：CVaR考虑了超过VaR的极端损失
2. **次可加性（Subadditivity）**：满足风险度量的数学性质，更符合分散化降低风险的直觉
3. **监管认可**：巴塞尔协议III已经开始推荐CVaR

**对比示例**：
```python
# 比较两个组合的风险
portfolio_a = np.random.normal(-0.0005, 0.01, 1000)  # 稳健组合
portfolio_b = np.random.normal(-0.0005, 0.01, 980).tolist() + \
              np.random.normal(-0.05, 0.05, 20).tolist()  # 有极端损失

print("组合A:")
print(f"  VaR(95%) = {historical_var(portfolio_a, 0.95):.4f}")
print(f"  CVaR(95%) = {historical_cvar(portfolio_a, 0.95):.4f}")

print("\n组合B:")
print(f"  VaR(95%) = {historical_var(portfolio_b, 0.95):.4f}")
print(f"  CVaR(95%) = {historical_cvar(portfolio_b, 0.95):.4f}")
```

输出会显示：**组合B的CVaR显著高于A，但VaR可能相近**。

## 实战：用VaR/CVaR管理投资组合风险

### 1. 仓位管理：基于VaR的仓位计算

**核心思想**：不是用固定比例仓位，而是用"风险预算"来分配仓位。

**计算方法**：
```python
def position_size_var(account_value, var_limit, asset_var, confidence_level=0.95):
    """
    基于VaR的仓位计算
    
    Parameters:
    -----------
    account_value: float
        账户总价值
    var_limit: float
        可承受的最大VaR（占账户比例）
    asset_var: float
        资产的VaR（占资产价值比例）
    confidence_level: float
        置信水平
    
    Returns:
    --------
    position_value: float
        建议仓位价值
    """
    max_loss = account_value * var_limit
    position_value = max_loss / asset_var
    return position_value

# 示例
account = 1000000  # 100万账户
var_limit = 0.02   # 最多承受2%的VaR

# 假设某股票的日VaR(95%) = 3%
stock_var = 0.03

position = position_size_var(account, var_limit, stock_var, 0.95)
print(f"建议仓位：{position:.0f}元 ({position/account*100:.1f}%)")
```

### 2. 组合VaR：分散化的力量

**关键问题**：组合VaR ≠ 各资产VaR之和

```python
def portfolio_var(returns_df, weights, confidence_level=0.95):
    """
    计算投资组合VaR（考虑相关性）
    
    Parameters:
    -----------
    returns_df: pd.DataFrame
        各资产的收益率矩阵
    weights: list or np.array
        资产权重
    confidence_level: float
        置信水平
    
    Returns:
    --------
    portfolio_var: float
        组合VaR
    """
    weights = np.array(weights)
    
    # 计算组合收益率
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # 计算VaR
    var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    return abs(var)

# 示例：比较分散化和集中仓位的VaR
np.random.seed(42)

# 生成两个相关资产的收益率
n_days = 1000
asset1_returns = np.random.normal(0.0005, 0.015, n_days)
asset2_returns = 0.5 * asset1_returns + np.random.normal(0.0003, 0.012, n_days)

returns_df = pd.DataFrame({
    'Asset1': asset1_returns,
    'Asset2': asset2_returns
})

# 场景1：全部投资产1
var_concentrated = portfolio_var(returns_df, [1, 0], 0.95)

# 场景2：等权重分散化
var_diversified = portfolio_var(returns_df, [0.5, 0.5], 0.95)

print(f"集中仓位 VaR = {var_concentrated:.4f}")
print(f"分散化仓位 VaR = {var_diversified:.4f}")
print(f"风险降低 = {(var_concentrated - var_diversified)/var_concentrated*100:.1f}%")
```

### 3. 回测VaR模型：失败率检验

**核心思想**：一个好的VaR模型，其"突破次数"（实际损失超过VaR的天数）应该接近理论值。

**Kupiec检验（失败率检验）**：
```python
def var_backtest(actual_returns, var_estimates, confidence_level=0.95):
    """
    VaR回测：检验突破次数是否符合预期
    
    Parameters:
    -----------
    actual_returns: pd.Series
        实际收益率（负值表示损失）
    var_estimates: pd.Series
        VaR估计值（正数）
    confidence_level: float
        置信水平
    
    Returns:
    --------
    results: dict
        回测结果
    """
    # 计算突破次数
    breaches = actual_returns < -var_estimates
    num_breaches = breaches.sum()
    total_days = len(actual_returns)
    breach_rate = num_breaches / total_days
    
    # 理论突破率
    expected_rate = 1 - confidence_level
    
    # 计算LR统计量（似然比检验）
    from scipy.stats import chi2
    
    lr = -2 * (np.log((1-expected_rate)**(total_days-num_breaches) * expected_rate**num_breaches) - \
               np.log((1-breach_rate)**(total_days-num_breaches) * breach_rate**num_breaches))
    
    p_value = 1 - chi2.cdf(lr, 1)
    
    results = {
        'total_days': total_days,
        'num_breaches': num_breaches,
        'breach_rate': breach_rate,
        'expected_rate': expected_rate,
        'lr_statistic': lr,
        'p_value': p_value,
        'model_valid': p_value > 0.05
    }
    
    return results

# 示例
actual_returns = pd.Series(np.random.normal(0.0005, 0.02, 1000))
var_estimates = pd.Series([historical_var(actual_returns[:i], 0.95) for i in range(250, 1000)])

results = var_backtest(actual_returns[250:], var_estimates)
print(f"突破次数：{results['num_breaches']}/{results['total_days']}")
print(f"突破率：{results['breach_rate']:.2%} (理论值：{results['expected_rate']:.2%})")
print(f"模型是否有效：{results['model_valid']}")
```

## 进阶：VaR/CVaR的局限与改进

### 1. 肥尾修正：t分布VaR

```python
from scipy.stats import t

def t_dist_var(returns, confidence_level=0.95, nu=5):
    """
    t分布VaR（捕捉肥尾）
    
    Parameters:
    -----------
    returns: pd.Series
        收益率序列
    confidence_level: float
        置信水平
    nu: float
        t分布的自由度（越小肥尾越明显）
    
    Returns:
    --------
    var: float
        VaR值
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # t分布的分位数
    t_alpha = t.ppf(1 - confidence_level, nu)
    
    # 标准化到样本均值和标准差
    var = -(mu + sigma * t_alpha / np.sqrt(nu/(nu-2)))
    return var

# 比较正态分布vs t分布VaR
var_normal = parametric_var(pd.Series(returns), 0.99)
var_t = t_dist_var(pd.Series(returns), 0.99, nu=5)

print(f"正态分布 VaR(99%) = {var_normal:.4f}")
print(f"t分布 VaR(99%) = {var_t:.4f} (更高，反映肥尾)")
```

### 2. 时变波动率：GARCH-VaR

```python
from arch import arch_model

def garch_var(returns, confidence_level=0.95, horizon=1):
    """
    GARCH模型VaR（时变波动率）
    """
    # 拟合GARCH(1,1)模型
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, mean='Constant')
    results = model.fit(disp='off')
    
    # 预测波动率
    forecasts = results.forecast(horizon=horizon)
    predicted_vol = forecasts.variance.values[-1, -1] / 10000
    
    # 计算VaR（假设正态分布）
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - confidence_level)
    var = abs(z_alpha * np.sqrt(predicted_vol))
    
    return var

# 需要安装：pip install arch
# var_garch = garch_var(pd.Series(returns))
```

### 3. 极值理论（EVT）：捕捉极端肥尾

```python
from scipy.stats import genpareto

def eva_var(returns, confidence_level=0.99, threshold_quantile=0.1):
    """
    极值理论VaR（专注于尾部建模）
    
    Parameters:
    -----------
    returns: pd.Series
        收益率序列
    confidence_level: float
        置信水平（通常>0.99）
    threshold_quantile: float
        阈值分位数（用于选取尾部数据）
    """
    # 选取左尾数据（损失）
    threshold = np.percentile(returns, threshold_quantile * 100)
    tail_data = -(returns[returns <= threshold] - threshold)  # 超阈值损失
    
    # 拟合广义帕累托分布（GPD）
    # 使用scipy的fit
    params = genpareto.fit(tail_data, floc=0)
    
    # 计算VaR
    excess_prob = 1 - confidence_level
    var = threshold - genpareto.ppf(excess_prob / threshold_quantile, *params)
    
    return abs(var)
```

## 实盘应用：风险管理框架

### 1. 多层次风险限额

```python
class RiskManager:
    """
    多层级风险管理框架
    """
    def __init__(self, account_value, max_portfolio_var=0.02, max_position_var=0.01):
        self.account_value = account_value
        self.max_portfolio_var = max_portfolio_var  # 组合层面VaR上限
        self.max_position_var = max_position_var    # 单资产VaR上限
        
    def check_position_limit(self, asset_returns, weight):
        """
        检查单资产仓位是否超限
        """
        asset_var = historical_var(asset_returns, 0.95)
        position_var = asset_var * weight
        
        if position_var > self.max_position_var:
            return False, f"单资产VaR超标：{position_var:.4f} > {self.max_position_var:.4f}"
        return True, "OK"
    
    def check_portfolio_limit(self, returns_df, weights):
        """
        检查组合VaR是否超限
        """
        portfolio_returns = (returns_df * weights).sum(axis=1)
        portfolio_var = historical_var(portfolio_returns, 0.95)
        
        if portfolio_var > self.max_portfolio_var:
            return False, f"组合VaR超标：{portfolio_var:.4f} > {self.max_portfolio_var:.4f}"
        return True, "OK"
    
    def calculate_optimal_position(self, asset_returns, target_risk_contribution=0.01):
        """
        计算最优仓位（风险预算法）
        """
        asset_var = historical_var(asset_returns, 0.95)
        max_position_value = self.account_value * target_risk_contribution / asset_var
        return max_position_value / self.account_value  # 返回权重

# 使用示例
rm = RiskManager(account_value=1000000, max_portfolio_var=0.02)

# 检查仓位
asset_returns = pd.Series(np.random.normal(0.0005, 0.02, 1000))
is_ok, msg = rm.check_position_limit(asset_returns, weight=0.1)
print(msg)

# 计算最优仓位
optimal_weight = rm.calculate_optimal_position(asset_returns, 0.01)
print(f"建议权重：{optimal_weight:.2%}")
```

### 2. 动态止损：基于CVaR的止损线

```python
def dynamic_stop_loss(portfolio_value, cvar, confidence_level=0.95, multiplier=1.5):
    """
    基于CVaR的动态止损
    
    Parameters:
    -----------
    portfolio_value: float
        当前组合价值
    cvar: float
        CVaR值（占组合价值比例）
    multiplier: float
        安全垫倍数（越大越保守）
    
    Returns:
    --------
    stop_loss_level: float
        止损线（组合价值）
    """
    # 止损距离 = CVaR * 安全垫倍数
    stop_distance = cvar * multiplier
    stop_loss_level = portfolio_value * (1 - stop_distance)
    return stop_loss_level

# 示例
current_value = 1100000  # 当前组合价值110万
cvar_95 = 0.03  # CVaR = 3%

stop_level = dynamic_stop_loss(current_value, cvar_95, 0.95, multiplier=1.5)
print(f"当前价值：{current_value:.0f}元")
print(f"止损线：{stop_level:.0f}元 (回撤{(current_value-stop_level)/current_value*100:.1f}%)")
```

### 3. 压力测试：极端情景下的VaR

```python
def stress_test_var(returns, stress_scenarios):
    """
    压力测试：在极端情景下重新计算VaR
    
    Parameters:
    -----------
    returns: pd.Series
        历史收益率
    stress_scenarios: dict
        压力情景（如{'market_crash': -0.15, 'vol_spike': 2.0}）
    
    Returns:
    --------
    stressed_var: float
        压力情景下的VaR
    """
    stressed_returns = returns.copy()
    
    # 应用压力情景
    if 'market_crash' in stress_scenarios:
        # 假设某一天收益率暴跌
        crash_day = len(returns) // 2
        stressed_returns.iloc[crash_day] += stress_scenarios['market_crash']
    
    if 'vol_spike' in stress_scenarios:
        # 假设波动率放大
        stressed_returns = stressed_returns * stress_scenarios['vol_spike']
    
    # 重新计算VaR
    stressed_var = historical_var(stressed_returns, 0.95)
    return stressed_var

# 示例
stress_scenarios = {
    'market_crash': -0.15,  # 某一天暴跌15%
    'vol_spike': 2.0        # 波动率放大2倍
}

normal_var = historical_var(returns, 0.95)
stressed_var = stress_test_var(pd.Series(returns), stress_scenarios)

print(f"正常VaR(95%) = {normal_var:.4f}")
print(f"压力VaR(95%) = {stressed_var:.4f} (放大{stressed_var/normal_var:.1f}倍)")
```

## 总结：VaR/CVaR最佳实践

### 1. 选择哪种VaR？

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 历史模拟法 | 常规风险管理 | 简单，捕捉肥尾 | 对结构性变化不敏感 |
| 参数法 | 快速估算 | 计算快，可调整持有期 | 低估肥尾 |
| 蒙特卡洛法 | 复杂资产/非线性 | 灵活，可模拟各种分布 | 计算成本高 |
| **CVaR** | **极端风险管理** | **捕捉尾部，数学性质好** | **计算复杂** |

**推荐组合**：
- 日常监控：历史模拟法VaR（95%）
- 极端风险：CVaR（99%）
- 压力测试：蒙特卡洛模拟

### 2. VaR/CVaR使用注意事项

1. **不要单独依赖VaR/CVaR**：结合其他指标（最大回撤、夏普比率、胜率）
2. **定期回测**：检验VaR模型的准确性（突破率是否接近理论值）
3. **考虑流动性**：VaR假设可以按市价平仓，但实盘中可能有流动性风险
4. **多时间维度**：同时监控日VaR、周VaR、月VaR

### 3. 风险管理的铁律

> **"VaR告诉你最坏情况下会亏多少，但记住：最坏情况往往会比你认为的更坏。"**

因此：
- VaR/CVaR是**必要但不充分**的风险管理工具
- 永远留安全垫（如用1.5倍CVaR作为止损线）
- 分散化是唯一"免费午餐"
- **生存比收益更重要**

---

**下期预告**：《风险平价策略中国实证：跨越美股的本土化改造》—— 风险平价策略在中国市场表现如何？需要做哪些改良？我们将用A股数据实证分析。

**相关资源**：
- [VaR经典教材：Risk Management and Financial Institutions by John Hull](https://www.amazon.com/Risk-Management-Financial-Institutions-Hull/dp/111944811X)
- [Python风险库：pyfolio](https://github.com/quantopian/pyfolio)
- [巴塞尔协议III：CVaR监管要求](https://www.bis.org/basel_framework/)

*希望这篇文章能帮你建立量化的风险标尺。记住：不懂风险的收益只是赌博，而懂得风险的收益才是投资。*
