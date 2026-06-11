---
title: "Black-Litterman 模型：桥接市场均衡与投资者观点的资产配置框架"
publishDate: '2026-06-12'
description: "Black-Litterman 模型：桥接市场均衡与投资者观点的资产配置框架 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言：传统均值方差优化的困境

1990年，Harry Markowitz 因现代投资组合理论（MPT）获得诺贝尔经济学奖，其核心思想"分散投资降低非系统性风险"深刻影响了资产管理行业。然而，实践中的均值方差优化（Mean-Variance Optimization, MVO）却面临三大痛点：

![Black-Litterman模型框架](/images/black-litterman-model/bl_model.svg)

1. **输入敏感性**：预期收益率的微小变化导致权重剧烈波动
2. **集中化倾向**：优化器往往给出极端集中的投资组合
3. **估计误差放大**：历史收益率的估计误差在优化过程中被放大

1992年，Fischer Black 和 Robert Litterman 在 Goldman Sachs 工作期间提出了 Black-Litterman (BL) 模型，巧妙地将市场均衡配置与投资者主观观点结合，开创了资产配置的新范式。

## Black-Litterman 模型的核心思想

### 逆向思维：从均衡到预期

传统 MVO 直接从历史数据估计预期收益率 $\mu$，而 BL 模型采用逆向路径：

1. **市场均衡收益率**：从市场权重反推隐含的预期收益率（逆优化）
2. **观点调整**：将投资者对特定资产的看法融入预期收益率
3. **贝叶斯框架**：用贝叶斯方法将观点与均衡收益率结合

![投资组合优化框架](/images/black-litterman-model/portfolio_opt.svg)

数学表达为：

$$
E[R] = [( \tau\Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau\Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]
$$

其中：
- $\Pi$：市场均衡收益率（隐含收益率）
- $Q$：投资者观点向量
- $P$：观点映射矩阵
- $\Omega$：观点不确定性矩阵
- $\Sigma$：收益率协方差矩阵
- $\tau$：缩放因子

### 市场均衡收益率的计算

BL 模型假设市场处于均衡状态，通过逆向优化（Reverse Optimization）计算市场隐含收益率：

$$
\Pi = \lambda \Sigma w_{mkt}
$$

其中：
- $\lambda$ 是风险厌恶系数（通常取 2.5-3.0）
- $w_{mkt}$ 是市场市值权重
- $\Sigma$ 是收益率协方差矩阵

这一步骤是该模型的精髓：**不直接估计预期收益率，而是从当前市场价格反推市场共识**。

## 投资者观点的数学表达

BL 模型允许投资者表达两类观点：

### 1. 绝对观点（Absolute Views）

对某个资产的预期收益率直接给出看法：

$$
E[R_i] = q_i \pm \omega_i
$$

例如："我认为贵州茅台未来一年的年化收益率将达到 15% ± 5%"

### 2. 相对观点（Relative Views）

对资产之间的相对表现给出看法：

$$
E[R_i] - E[R_j] = q_k \pm \omega_k
$$

例如："我认为宁德时代未来半年的表现将跑赢沪深300指数 8% ± 3%"

### 3. 观点矩阵 P 和 Q

将所有观点整合为矩阵形式：

$$
P \cdot E[R] = Q + \epsilon, \quad \epsilon \sim N(0, \Omega)
$$

示例：
- 观点1：A股消费板块收益率 = 12%
- 观点2：科技板块收益率 - 金融板块收益率 = 5%

$$
P = \begin{bmatrix}
1 & 0 & 0 & \cdots \\
0 & 1 & -1 & \cdots
\end{bmatrix}, \quad
Q = \begin{bmatrix}
12\% \\ 5\%
\end{bmatrix}
$$

### 4. 观点不确定性 Ω

BL 模型用 $\Omega$ 矩阵表示对每个观点的置信度：

$$
\Omega = \text{diag}( \omega_1^2, \omega_2^2, \ldots, \omega_k^2 )
$$

观点越不确定，对应 $\omega_i$ 越大，该观点对最终预期收益率的影响越小。

## 参数选择与实践经验

### 1. 风险厌恶系数 λ

- **学术建议**：$\lambda = 2.5 \sim 3.0$（对应 60/40 股债组合的夏普比率）
- **实践经验**：A股市场波动率高，可取 $\lambda = 2.0 \sim 2.5$
- **校准方法**：用历史数据拟合市场权重对应的隐含收益率

### 2. 缩放因子 τ

- **理论建议**：$\tau = 1 / T$（T 为历史数据期数）
- **实践取值**：$\tau = 0.05 \sim 0.10$
- **含义**：控制市场均衡收益率的相对权重

### 3. 观点置信度 ω

- **主观设定**：根据对观点的信心程度手动设置
- **自适应方法**：$\omega_i^2 = p_i^T \Sigma p_i \cdot \tau$（与资产波动性挂钩）
- **经验法则**：对强观点取 $\omega = 0.01 \sim 0.05$，弱观点取 $\omega = 0.10 \sim 0.20$

## Python 实现示例

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BlackLitterman:
    """
    Black-Litterman 模型实现
    
    参数：
    - returns: 资产历史收益率 DataFrame (T x N)
    - market_caps: 资产市值权重 Series (N,)
    - risk_aversion: 风险厌恶系数 λ
    - tau: 缩放因子
    """
    
    def __init__(self, returns, market_caps, risk_aversion=2.5, tau=0.05):
        self.returns = returns
        self.market_caps = market_caps
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # 计算协方差矩阵
        self.Sigma = returns.cov() * 252  # 年化
        self.N = len(market_caps)
        
    def compute_implied_returns(self):
        """
        逆向优化：从市场权重计算均衡收益率
        Π = λ * Σ * w_mkt
        """
        Pi = self.risk_aversion * self.Sigma @ self.market_caps
        return Pi
    
    def bl_formula(self, P, Q, Omega):
        """
        Black-Litterman 公式：计算后验预期收益率
        
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        """
        Pi = self.compute_implied_returns()
        
        # 转换 to matrix
        tau_Sigma = self.tau * self.Sigma.values
        P_mat = P.values
        Q_vec = Q.values
        Omega_mat = Omega.values
        
        # 后验预期收益率
        M_inv = np.linalg.inv( np.linalg.inv(tau_Sigma) + P_mat.T @ np.linalg.inv(Omega_mat) @ P_mat )
        posterior_returns = M_inv @ ( np.linalg.inv(tau_Sigma) @ Pi.values + P_mat.T @ np.linalg.inv(Omega_mat) @ Q_vec )
        
        return pd.Series(posterior_returns, index=self.Sigma.columns)
    
    def optimize_portfolio(self, expected_returns, risk_aversion=None):
        """
        给定预期收益率，用均值方差优化计算权重
        """
        if risk_aversion is None:
            risk_aversion = self.risk_aversion
        
        N = len(expected_returns)
        Sigma = self.Sigma.values
        
        # 目标函数：最大化效用 = μ'w - (λ/2) * w'Σw
        def objective(w):
            port_return = expected_returns.values @ w
            port_risk = w @ Sigma @ w
            utility = port_return - (risk_aversion / 2) * port_risk
            return -utility  # 最小化负效用
        
        # 约束：权重和为1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bounds = tuple((0, 1) for _ in range(N))  # 只允许做多
        
        result = minimize(objective, np.ones(N)/N, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=expected_returns.index)

# 使用示例
# 假设有5个资产：消费、科技、金融、医药、新能源
assets = ['Consumer', 'Tech', 'Finance', 'Healthcare', 'NewEnergy']

# 模拟历史收益率（均值和协方差）
np.random.seed(42)
returns = pd.DataFrame(
    np.random.randn(1000, 5) * 0.15 / np.sqrt(252) + 0.0005,
    columns=assets,
    index=pd.date_range('2020-01-01', periods=1000)
)

# 市场市值权重（假设）
market_caps = pd.Series([0.25, 0.30, 0.20, 0.15, 0.10], index=assets)

# 初始化 BL 模型
bl = BlackLitterman(returns, market_caps, risk_aversion=2.5, tau=0.05)

# 计算市场均衡收益率
Pi = bl.compute_implied_returns()
print("市场均衡收益率（年化）:")
print(Pi)

# 定义投资者观点
# 观点1：科技板块收益率 = 18%
# 观点2：新能源 - 金融 = 10%
P = pd.DataFrame([
    [0, 1, 0, 0, 0],   # 科技
    [0, 0, -1, 0, 1]   # 新能源 - 金融
], columns=assets, index=['View1', 'View2'])

Q = pd.Series([0.18, 0.10], index=['View1', 'View2'])

# 观点不确定性（对角线矩阵）
Omega = np.diag([0.05**2, 0.08**2])  # 观点1不确定性5%，观点2不确定性8%
Omega = pd.DataFrame(Omega, index=['View1', 'View2'], columns=['View1', 'View2'])

# BL 后验预期收益率
posterior_returns = bl.bl_formula(P, Q, Omega)
print("\nBL后验预期收益率（年化）:")
print(posterior_returns)

# 优化投资组合
optimal_weights = bl.optimize_portfolio(posterior_returns)
print("\nBL最优投资组合权重:")
print(optimal_weights)
```

## 在中国A股市场的应用实践

### 1. 行业配置视角

A股市场行业轮动特征明显，BL 模型适合表达行业观点：

| 观点类型 | 示例 | 置信度 |
|---------|------|--------|
| 绝对观点 | 新能源行业未来6个月收益率 20% | ±8% |
| 相对观点 | 半导体相对银行跑赢 15% | ±5% |
| 市场观点 | 沪深300未来1年收益率 8% | ±3% |

### 2. 因子整合

将因子观点融入 BL 框架：

```python
# 因子暴露矩阵 F (N assets x K factors)
# 因子预期收益率 f (K x 1)
# 观点：资产 i 的收益率 = 因子暴露 × 因子收益率

P = F  # 观点矩阵 = 因子暴露
Q = f  # 观点向量 = 因子预期收益率
```

### 3. 风险模型校准

A股波动率较高，建议调整参数：

- **风险厌恶系数**：$\lambda = 2.0 \sim 2.5$（低于美股）
- **协方差估计**：使用高频数据 + 指数加权移动平均（EWMA）
- **市值权重**：使用自由流通市值（剔除限售股）

## 与风险平价模型的对比

| 维度 | Black-Litterman | 风险平价 |
|------|----------------|---------|
| **收益假设** | 有（均衡+观点） | 无（隐含平等收益） |
| **输入参数** | 预期收益率+协方差+观点 | 仅协方差 |
| **主观判断** | 需要（观点+置信度） | 不需要 |
| **适用场景** | 有主动观点的配置 | 保守的长期配置 |
| **集中度** | 中等（取决于观点） | 分散（风险均等） |

实践建议：
- **战略性配置**：使用风险平价确定基准组合
- **战术性调整**：用 BL 模型表达短期观点，在基准上调整

## 模型的局限性与改进方向

### 1. 局限性

- **观点设定主观**：不同投资者的观点差异导致结果不可比
- **协方差估计敏感**：$\Sigma$ 的估计误差影响均衡收益率
- **正态分布假设**：极端事件下正态分布假设失效

### 2. 改进方向

- **熵池方法（Entropy Pooling）**：放松正态分布假设，处理非对称观点
- **动态观点更新**：用卡尔曼滤波动态更新观点置信度
- **鲁棒优化**：对 $\Sigma$ 和 $Q$ 进行鲁棒性处理

## 总结

Black-Litterman 模型通过桥接市场均衡与投资者主观观点，提供了严谨的资产配置框架。其核心优势在于：

1. **稳定性**：相比传统 MVO，权重对输入更稳健
2. **灵活性**：允许表达多样化、结构化的观点
3. **可解释性**：权重变化可追溯至具体观点

在A股市场应用时，需注意参数校准、因子整合、风险模型选择等本土化调整。未来可结合机器学习方法自动提取观点，进一步提升模型实用性。

---

**参考文献**：
1. Black, F., & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*.
2. He, G., & Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios. *Goldman Sachs Working Paper*.
3. Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model. *Ibbotson Associates*.
