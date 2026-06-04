---
title: "凯利公式在量化投资中的实战应用：从赌博到投资组合的仓位管理"
publishDate: '2026-06-05'
description: "凯利公式在量化投资中的实战应用：从赌博到投资组合的仓位管理 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 凯利公式：从信息论到量化投资

1956年，贝尔实验室的约翰·凯利（John Kelly）发表了《A New Interpretation of Information Rate》论文，原本用于解决电话线路中的噪声问题。没想到，这个公式后来成为量化投资和赌博界最著名的仓位管理工具。

**核心思想**：在已知期望收益和胜率的情况下，存在一个最优下注比例，使得长期资金的**指数增长率最大化**。

### 经典凯利公式

对于二元结果的赌博游戏（赢或输）：

$$
f^* = \frac{p \cdot b - (1-p)}{b} = \frac{p(b+1) - 1}{b}
$$

其中：
- $f^*$：最优下注比例（占总资金比例）
- $p$：胜率（Win Probability）
- $b$：赔率（净收益/亏损金额，即盈亏比）

**示例计算**：
- 抛硬币游戏：胜率 $p=0.5$，赔率 $b=1$（输赢都是1倍）
- 凯利公式：$f^* = \frac{0.5 \times 1 - 0.5}{1} = 0$
- 结论：公平游戏中，长期不亏不赚，不应下注

- 优势游戏：胜率 $p=0.6$，赔率 $b=1$
- 凯利公式：$f^* = \frac{0.6 \times 1 - 0.4}{1} = 0.2$
- 结论：每次下注20%资金，长期增长率最大化

## 量化投资中的凯利公式变种

### 1. 连续型凯利公式（Continuous Kelly）

在量化交易中，收益率是连续分布而非二元结果。此时凯利公式为：

$$
f^* = \frac{\mu}{\sigma^2}
$$

其中：
- $\mu$：期望收益率（均值）
- $\sigma^2$：收益率方差

```python
import numpy as np
import pandas as pd

def continuous_kelly(returns):
    """
    连续型凯利公式：计算最优仓位比例
    
    Parameters:
    -----------
    returns : pd.Series
        资产收益率序列
        
    Returns:
    --------
    kelly_fraction : float
        最优凯利仓位（可能为负数，表示做空）
    """
    mu = returns.mean() * 252  # 年化收益率
    sigma2 = returns.var() * 252  # 年化方差
    
    if sigma2 == 0:
        return 0
    
    kelly_fraction = mu / sigma2
    
    # 凯利公式无上限，实际中需要截断
    return np.clip(kelly_fraction, -1, 1)

# 示例：计算某股票的凯利仓位
stock_returns = pd.read_csv('stock_returns.csv', index_col=0, parse_dates=True)['return']
kelly_frac = continuous_kelly(stock_returns)
print(f"最优凯利仓位：{kelly_frac:.2%}")
```

**问题**：连续型凯利公式对收益率估计误差极其敏感，容易导致极端仓位。

### 2. 多资产凯利公式（Multivariate Kelly）

当投资组合包含多个资产时，需要扩展到 multivariate 情形：

$$
\mathbf{f}^* = \mathbf{\Sigma}^{-1} \mathbf{\mu}
$$

其中：
- $\mathbf{f}^*$：各资产的最优仓位向量
- $\mathbf{\mu}$：各资产的期望收益率向量
- $\mathbf{\Sigma}^{-1}$：收益率协方差矩阵的逆矩阵

```python
def multivariate_kelly(returns_df):
    """
    多资产凯利公式：计算最优投资组合权重
    
    Parameters:
    -----------
    returns_df : pd.DataFrame, shape (T, N)
        N个资产的收益率序列
        
    Returns:
    --------
    kelly_weights : np.ndarray, shape (N,)
        凯利最优权重（可能未归一化）
    """
    # 年化收益率和协方差
    mu = returns_df.mean() * 252
    Sigma = returns_df.cov() * 252
    
    # 凯利公式：f = Sigma^{-1} * mu
    try:
        inv_Sigma = np.linalg.inv(Sigma.values)
        kelly_weights = inv_Sigma @ mu.values
    except np.linalg.LinAlgError:
        # 协方差矩阵奇异，使用伪逆
        inv_Sigma = np.linalg.pinv(Sigma.values)
        kelly_weights = inv_Sigma @ mu.values
    
    return kelly_weights

# 示例：3资产投资组合
returns_3assets = pd.DataFrame({
    'Stock_A': np.random.normal(0.0005, 0.02, 1000),
    'Stock_B': np.random.normal(0.0003, 0.015, 1000),
    'Stock_C': np.random.normal(0.0008, 0.025, 1000)
})

kelly_w = multivariate_kelly(returns_3assets)
print("凯利最优权重：", kelly_w)
print("归一化权重：", kelly_w / np.sum(np.abs(kelly_w)))
```

**问题**：协方差矩阵估计误差会被逆矩阵放大，导致权重不稳定。

## 凯利公式的实战陷阱

### 陷阱1：过度杠杆（Over-leverage）

凯利公式给出的理论最优仓位往往非常激进。例如：
- 如果 $\mu=0.1$, $\sigma=0.2$，则 $f^* = 0.1/0.04 = 2.5$
- 意味着理论上应该2.5倍杠杆！

**实战解决方案**：**分数凯利（Fractional Kelly）**

```python
def fractional_kelly(returns, fraction=0.5):
    """
    分数凯利：降低实操风险
    
    Parameters:
    -----------
    returns : pd.Series
        资产收益率
    fraction : float
        分数系数（0-1之间，0.5表示半凯利）
        
    Returns:
    --------
    frac_kelly : float
        分数凯利仓位
    """
    full_kelly = continuous_kelly(returns)
    frac_kelly = full_kelly * fraction
    
    return np.clip(frac_kelly, -1, 1)

# 对比：全凯利 vs 半凯利
full = continuous_kelly(stock_returns)
half = fractional_kelly(stock_returns, fraction=0.5)
print(f"全凯利仓位：{full:.2%}")
print(f"半凯利仓位：{half:.2%}")
```

**经验法则**：
- 全凯利（1x）：理论最优，但波动极大，容易破产
- 半凯利（0.5x）：实践中常用的折中方案
- 四分凯利（0.25x）：极度风险厌恶者的选择

### 陷阱2：估计误差敏感（Estimation Error）

凯利公式严重依赖 $\mu$ 和 $\sigma$ 的估计。在样本外，估计误差会导致：
- 过高的仓位（如果高估 $\mu$ 或低估 $\sigma$）
- 过低仓位（如果低估 $\mu$ 或高估 $\sigma$）

**解决方案**：**贝叶斯凯利（Bayesian Kelly）**

```python
from scipy.stats import norm

def bayesian_kelly(returns, prior_mu=0, prior_sigma=0.1, alpha=0.05):
    """
    贝叶斯凯利：引入先验分布，降低估计误差
    
    Parameters:
    -----------
    returns : pd.Series
        资产收益率
    prior_mu : float
        先验期望收益率（通常设为0或无风险利率）
    prior_sigma : float
        先验方差（表示对先验的不确定性）
    alpha : float
        置信水平（用于计算稳健凯利）
        
    Returns:
    --------
    bayes_kelly : float
        贝叶斯凯利仓位（保守版本）
    """
    T = len(returns)
    sample_mu = returns.mean() * 252
    sample_sigma2 = returns.var() * 252
    
    # 贝叶斯更新：后验均值和方差
    post_mu = (prior_mu / prior_sigma**2 + sample_mu * T / sample_sigma2) / \
              (1 / prior_sigma**2 + T / sample_sigma2)
    post_sigma2 = 1 / (1 / prior_sigma**2 + T / sample_sigma2)
    
    # 使用后验均值的分位数（保守估计）
    conservative_mu = norm.ppf(alpha/2, loc=post_mu, scale=np.sqrt(post_sigma2))
    
    # 贝叶斯凯利仓位
    bayes_kelly = conservative_mu / sample_sigma2
    
    return np.clip(bayes_kelly, -1, 1)

# 示例：贝叶斯凯利 vs 经典凯利
classic = continuous_kelly(stock_returns)
bayes = bayesian_kelly(stock_returns, prior_mu=0.05, prior_sigma=0.15)
print(f"经典凯利：{classic:.2%}")
print(f"贝叶斯凯利（保守）：{bayes:.2%}")
```

### 陷阱3：非正态分布（Non-Normal Returns）

凯利公式假设收益率服从正态分布。但实际股票收益率有：
- **厚尾（Fat Tails）**：极端事件概率高于正态分布预期
- **偏度（Skewness）**：收益率分布不对称
- **异方差（Heteroskedasticity）**：波动率聚类

**解决方案**：**修正凯利公式（Modified Kelly）**

```python
from scipy.stats import skew, kurtosis

def modified_kelly(returns):
    """
    修正凯利公式：考虑偏度和峰度
    
    Parameters:
    -----------
    returns : pd.Series
        资产收益率（日度）
        
    Returns:
    --------
    mod_kelly : float
        修正凯利仓位
    """
    # 样本统计量
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    S = skew(returns)  # 偏度
    K = kurtosis(returns) + 3  # 峰度（注意scipy返回值）
    
    # Cornish-Fisher展开：修正正态分布分位数
    z_alpha = norm.ppf(0.05)  # 5% VaR对应的分位数
    
    # 修正分位数
    z_mod = z_alpha + \
            (z_alpha**2 - 1) * S / 6 + \
            (z_alpha**3 - 3*z_alpha) * (K-3) / 24 - \
            (2*z_alpha**3 - 5*z_alpha) * S**2 / 36
    
    # 修正后的VaR
    var_mod = mu + sigma * z_mod / np.sqrt(252)
    
    # 修正凯利：使用修正后的下侧风险
    mod_kelly = mu / (sigma**2 * np.exp(z_mod**2 / 2))
    
    return np.clip(mod_kelly, -1, 1)

# 对比：经典凯利 vs 修正凯利
classic = continuous_kelly(stock_returns)
modified = modified_kelly(stock_returns)
print(f"经典凯利（假设正态）：{classic:.2%}")
print(f"修正凯利（考虑厚尾）：{modified:.2%}")
```

## 凯利公式 vs 其他仓位管理方法

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **凯利公式** | 理论最优，长期增长率最大 | 波动大，对估计误差敏感 | 高频交易，优势明显的策略 |
| **半凯利** | 降低波动，实操性强 | 牺牲部分长期收益 | 大多数量化策略的默认选择 |
| **固定比例** | 简单，低风险 | 未充分利用优势 | 风险厌恶者，新手 |
| **波动率倒数** | 自动降仓高波动资产 | 忽略收益率差异 | 风险平价策略 |
| **马科维茨均值方差** | 考虑组合整体风险 | 需要估计协方差矩阵 | 多资产投资组合 |

## 实战案例：配对交易的凯利仓位

以统计套利中的配对交易为例，展示凯利公式的应用：

```python
def pairs_trading_kelly(spread_returns, entry_zscore=2.0, exit_zscore=0.5):
    """
    配对交易的凯利仓位管理
    
    Parameters:
    -----------
    spread_returns : pd.Series
        配对价差收益率
    entry_zscore : float
        入场Z分数阈值
    exit_zscore : float
        出场Z分数阈值
        
    Returns:
    --------
    positions : pd.Series
        每日仓位（0-1之间）
    kelly_fractions : pd.Series
        每日凯利仓位
    """
    # 计算Z分数
    z_score = (spread_returns - spread_returns.rolling(63).mean()) / \
               spread_returns.rolling(63).std()
    
    # 信号：价差偏离均值时入场
    signal = pd.Series(0, index=spread_returns.index)
    signal[z_score > entry_zscore] = -1  # 做空价差
    signal[z_score < -entry_zscore] = 1   # 做多价差
    signal[np.abs(z_score) < exit_zscore] = 0  # 平仓
    
    # 计算凯利仓位（滚动窗口）
    kelly_fractions = pd.Series(index=spread_returns.index)
    for t in range(63, len(spread_returns)):
        window_returns = spread_returns.iloc[t-63:t]
        kelly = continuous_kelly(window_returns)
        kelly_fractions.iloc[t] = kelly
    
    # 应用信号和凯利仓位
    positions = signal.shift(1) * np.clip(kelly_fractions, 0, 1)
    
    return positions, kelly_fractions

# 回测结果示例
positions, kelly_frac = pairs_trading_kelly(spread_returns)

# 计算策略收益
strategy_returns = positions.shift(1) * spread_returns
cum_returns = (1 + strategy_returns).cumprod()

print(f"策略年化收益率：{strategy_returns.mean()*252:.2%}")
print(f"策略夏普比率：{strategy_returns.mean()/strategy_returns.std()*np.sqrt(252):.2f}")
print(f"最大回撤：{((cum_returns.cummax() - cum_returns) / cum_returns.cummax()).max():.2%}")
```

## 总结与建议

1. **从半凯利开始**：全凯利理论完美，但实操风险过高，建议从0.25x-0.5x凯利开始
2. **结合止损**：凯利公式不包含止损规则，必须与止损结合使用
3. **滚动估计**：定期重新估计 $\mu$ 和 $\sigma$，适应市场变化
4. **多策略组合**：单一策略使用凯利可能导致过度集中，多策略组合更安全
5. **压力测试**：用历史极端行情测试凯利仓位的表现

凯利公式是量化投资中强大的仓位管理工具，但必须**谨慎使用、结合实战约束**。记住：生存比最优更重要。

![凯利公式增长曲线](/images/kelly-criterion-portfolio/growth_curve.png)

*图1：不同凯利比例下的资金增长曲线（模拟）*

![凯利 vs 固定比例](/images/kelly-criterion-portfolio/kelly_vs_fixed.png)

*图2：凯利公式与固定比例仓位的回撤对比*
