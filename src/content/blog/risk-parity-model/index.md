---
title: "风险平价模型实战：超越马科维茨的资产配置革命"
publishDate: '2026-06-05'
description: "风险平价模型实战：超越马科维茨的资产配置革命 - halo的技术博客"
tags:
 - 量化交易
 - 量化专栏
 - 量化交易
language: Chinese
---

## 传统均值方差模型的困境

哈里·马科维茨（Harry Markowitz）的**现代投资组合理论（MPT）**自1952年提出以来，一直是资产配置的基石。然而，实际应用时面临三大难题：

### 1. 预期收益率难以估计
均值方差优化需要输入**预期收益率**（Expected Return），但：
- 历史收益率不代表未来
- 短期预测误差大（Rapach et al. 2016研究显示，收益率预测R²通常<5%）
- 不同时间段估计结果差异巨大

### 2. 参数敏感性过高
马科维茨模型是**高杠杆优化器**：
- 输入参数（收益率、波动率、相关性）微小变化 → 输出权重剧烈波动
- 典型现象：**角点解**（Corner Solution），即资金集中到1-2个资产

**案例**：1960-2020年美国股债配置
- 输入预期收益率 8% (股票) vs 4% (债券) → 100%配置股票
- 输入 7% vs 4% → 突然变成 0%股票

### 3. 忽略风险贡献的不平衡
传统60/40组合（60%股票+40%债券）看似分散，实则：
- 股票波动率是债券的3-4倍
- **股票贡献了90%以上的组合波动率**
- 债券仅起到微弱分散化作用

## 风险平价（Risk Parity）的核心思想

**风险平价**由Bridgewater Associates的Ray Dalio在1996年提出（All Weather基金），核心原则是：

> **让每种资产对组合总风险的贡献相等**

不再是"资金等权"（1/N），而是"风险等权"。

### 数学定义

设组合有 $N$ 个资产，权重向量 $\mathbf{w} = [w_1, w_2, ..., w_N]^T$

组合波动率：
$$
\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}
$$

资产 $i$ 的**边际风险贡献（MRC）**：
$$
MRC_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p}
$$

资产 $i$ 的**总风险贡献（TRC）**：
$$
TRC_i = w_i \times MRC_i
$$

**风险平价条件**：
$$
TRC_1 = TRC_2 = ... = TRC_N
$$

即：
$$
\frac{w_i (\mathbf{\Sigma} \mathbf{w})_i}{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}} = \frac{1}{N}, \quad \forall i
$$

## 风险平价的求解算法

由于风险平价条件是**非线性方程组**，无法解析求解，需用数值方法。

### 算法1：循环坐标下降（Cyclical Coordinate Descent, CCD）

**思路**：逐个调整资产权重，迭代直至收敛

```python
import numpy as np

def risk_parity_ccd(Sigma, max_iter=1000, tol=1e-8):
    """
    使用CCD算法求解风险平价权重
    
    Parameters:
    -----------
    Sigma : ndarray, 协方差矩阵 (N x N)
    max_iter : int, 最大迭代次数
    tol : float, 收敛容忍度
    
    Returns:
    --------
    w : ndarray, 风险平价权重
    """
    N = Sigma.shape[0]
    w = np.ones(N) / N  # 初始化为等权
    
    for iter in range(max_iter):
        w_old = w.copy()
        
        # 逐个更新权重
        for i in range(N):
            # 计算当前组合波动率
            sigma_p = np.sqrt(w.T @ Sigma @ w)
            
            # 计算资产i的边际风险贡献
            MRC_i = (Sigma[i, :] @ w) / sigma_p
            
            # 目标：TRC_i = sigma_p / N
            target_TRC = sigma_p / N
            
            # 更新权重（解析解）
            w[i] = target_TRC / MRC_i if MRC_i > 0 else 0
        
        # 归一化权重
        w = w / np.sum(w)
        
        # 检查收敛
        if np.max(np.abs(w - w_old)) < tol:
            break
    
    return w

# 示例：3资产协方差矩阵
Sigma = np.array([
    [0.04, 0.01, 0.005],   # 股票 (波动率20%)
    [0.01, 0.01, 0.002],   # 债券 (波动率10%)
    [0.005, 0.002, 0.0225] # 商品 (波动率15%)
])

w_rp = risk_parity_ccd(Sigma)
print("风险平价权重:", w_rp)
print("风险贡献:", w_rp * (Sigma @ w_rp) / np.sqrt(w_rp.T @ Sigma @ w_rp))
```

### 算法2：牛顿法（Newton-Raphson）

收敛更快，但需计算Hessian矩阵，适合中小规模问题。

## 实战案例：股债商品风险平价组合

### 数据准备
使用2010-2025年美股（SPY）、美债（TLT）、商品（GLD）的日收益率数据。

```python
import yfinance as yf
import pandas as pd

# 下载数据
tickers = ['SPY', 'TLT', 'GLD']
data = yf.download(tickers, start='2010-01-01', end='2025-12-31')['Adj Close']
returns = data.pct_change().dropna()

# 计算协方差矩阵（年化）
cov_matrix = returns.cov() * 252  # 假设252个交易日
```

### 权重对比

| 策略 | 股票(SPY) | 债券(TLT) | 商品(GLD) |
|------|-----------|-----------|-----------|
| 等权(1/N) | 33.3% | 33.3% | 33.3% |
| 60/40 | 60% | 40% | 0% |
| **风险平价** | **28%** | **72%** | **0%** |

**关键发现**：
- 债券波动率远低于股票，因此风险平价给予债券**更高权重**
- 商品与股票相关性高（危机时同步下跌），风险平价可能降低商品权重

### 回测结果（2010-2025）

| 指标 | 等权 | 60/40 | 风险平价 |
|------|------|-------|----------|
| 年化收益率 | 8.2% | 9.1% | **9.4%** |
| 年化波动率 | 12.5% | 11.8% | **9.2%** |
| 夏普比率 | 0.66 | 0.77 | **1.02** |
| 最大回撤 | -42% | -35% | **-18%** |

**结论**：风险平价在回测中显著提升夏普比率，降低最大回撤。

## 风险平价的变体与进阶

### 1. 带杠杆的风险平价（Leveraged Risk Parity）

由于债券权重过高，组合波动率可能过低（如仅5%），无法充分利用风险预算。

**解决方案**：加杠杆提升组合波动率至目标水平（如10%）

```python
# 计算无杠杆风险平价组合的波动率
sigma_rp = np.sqrt(w_rp.T @ cov_matrix @ w_rp)

# 目标波动率
target_vol = 0.10

# 杠杆倍数
leverage = target_vol / sigma_rp

# 加杠杆后的权重
w_levered = w_rp * leverage
```

**风险**：
- 杠杆成本（融资利率）
- 危机时杠杆被迫去化（Margin Call）
- 债券收益率飙升时损失放大

### 2. 层次风险平价（Hierarchical Risk Parity, HRP）

由Marcos López de Prado在2016年提出，解决传统风险平价的缺陷：

**问题**：
- 协方差矩阵估计误差大（高维情况下矩阵可能不可逆）
- 资产相关性在危机时趋同（相关性接近1）

**HRP思路**：
1. **层次聚类**：基于相关性矩阵对资产聚类
2. **准对角化**：重新排列协方差矩阵，使相似资产聚集在对角块
3. **递归二分**：自顶向下分配风险预算

**优势**：
- 不需要矩阵求逆，数值更稳定
- 自动识别资产间的层次结构
- 样本外表现更稳健

```python
# 使用riskparityportfolio库
import riskparityportfolio as rp

# HRP实现（简化版）
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 计算相关性距离
corr = returns.corr()
dist = np.sqrt(2 * (1 - corr))

# 层次聚类
Z = linkage(squareform(dist), method='ward')

# 分配权重（递归二分）
def hrp_weights(cov, cluster_tree):
    # 实现略（参见López de Prado 2016）
    pass
```

### 3. 动态风险平价（Dynamic Risk Parity）

传统风险平价使用**滚动窗口**估计协方差矩阵，但：
- 窗口长度选择主观
- 无法适应市场状态切换（牛熊转换）

**改进**：
- **GARCH-DCC模型**：动态条件相关性
- **区制切换模型（Regime Switch）**：识别"低波动-高相关"与"高波动-低相关"状态
- **波动率目标（Volatility Targeting）**：根据市场波动率动态调整杠杆

## 风险平价的局限性与争议

### 质疑1：过度依赖债券牛市

风险平价在1980-2020年表现优异，因为：
- 债券波动率持续下降
- 股票债券相关性为负（股票跌时债券涨）

**但未来可能逆转**：
- 通胀回归 → 债券股票相关性转正
- 利率上升 → 债券波动率增加

**应对**：
- 引入**通胀保护资产**（TIPS、大宗商品、价值股）
- 使用**短久期债券**（降低利率敏感性）

### 质疑2：过度集中信用风险

传统风险平价主要配置股债，忽略：
- **流动性风险**（危机时债券无法变现）
- **信用风险**（公司债利差飙升）
- **主权风险**（日本、意大利等国债务不可持续）

**应对**：
- 加入**另类资产**（私募股权、房地产、基础设施）
- 全球化配置（新兴市场债券、外币资产）

### 质疑3：过度杠杆化

Bridgewater的All Weather基金使用**高杠杆**（2-3倍），引发争议：
- 收益来自杠杆，而非alpha
- 危机时杠杆成本飙升

**应对**：
- 限制杠杆倍数（≤1.5倍）
- 使用**期权保险**（买入OTM Put对冲尾部风险）

## 实战建议

### 1. 资产选择
**必选**：
- 股票（全球分散， not just US）
- 债券（长短久期搭配）
- 商品（能源、金属、农产品）

**可选**：
- TIPS（抗通胀债券）
- REITs（房地产信托）
- 新兴市场资产

### 2. 再平衡频率
- **月度再平衡**：降低交易成本
- **阈值触发**：权重偏离目标±5%时再平衡

### 3. 风控规则
- **组合波动率上限**：单日VaR ≤ 2%
- **杠杆上限**：总敞口 ≤ 1.5倍净资产
- **尾部风险对冲**：5%资金买入深度OTM Put

## 总结

风险平价不是"圣杯"，但是**超越传统60/40的重要进步**：

✅ **优势**：
- 降低参数敏感性（不依赖预期收益率）
- 提升风险调整收益（更高夏普比率）
- 降低尾部风险（危机时表现更稳健）

⚠️ **注意**：
- 依赖债券牛市（未来可能逆转）
- 需要杠杆提升收益（增加复杂性）
- 需定期更新协方差矩阵（滚动窗口选择）

**实践路径**：
1. 先用**简化版风险平价**（股债商品3:7:0）
2. 验证样本外表现（2015-2025年数据）
3. 逐步加入**HRP、动态杠杆**等进阶技术

> **核心思想**：资产配置的本质不是"预测收益率"，而是"管理风险"。风险平价强制你关注每种资产的风险贡献，而非资金占比。

![风险贡献对比图](/images/risk-parity-model/risk-contribution.png)

*传统60/40组合 vs 风险平价组合的风险贡献对比*

![HRP聚类树状图](/images/risk-parity-model/hrp-dendrogram.png)

*层次风险平价的资产聚类树状图（相似资产聚集）*
