---
title: "评级迁移矩阵建模：信用组合的多期损失分布"
description: "从评级迁移矩阵出发，构建信用组合的多期损失分布。详解矩阵幂次法、条件独立假设、阈值模型与蒙特卡洛模拟，并用 Python 完整实现 CreditMetrics 框架——从单期迁移矩阵到多期损失分布、VaR 和经济资本计算。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 信用风险
  - CreditMetrics
categories: ["量化交易"]
slug: "rating-migration-matrix-modeling"
image: "/images/rating-migration-matrix-modeling/migration_matrix_heatmap.png"
---

在信用风险管理中，违约不是一次性事件——它是一个过程。一家 BBB 级的公司今天可能是健康的，但经过5年的经营恶化，它可能先从 BBB 降到 BB，再从 BB 降到 B，最终走向违约。评级迁移矩阵（Rating Migration Matrix）正是刻画这个过程的数学工具，它是 CreditMetrics 框架的基石，也是银行经济资本计算和 Basel IRB 方法背后的核心假设。

![一年期评级迁移矩阵热力图](/images/rating-migration-matrix-modeling/migration_matrix_heatmap.png)

这篇文章从迁移矩阵的基本结构出发，推导多期损失分布的构建方法，并用 Python 完整实现一个 CreditMetrics 风格的组合损失模拟器。

## 一、迁移矩阵：信用风险的马尔可夫链

评级迁移矩阵是一个 $N \times N$ 的概率矩阵，其中 $N$ 是评级等级数量。矩阵的每个元素 $M_{ij}$ 表示一个当前评级为 $i$ 的发行人在一年后评级变为 $j$ 的概率。

$$M_{ij} = P(\text{rating at } t+1 = j \mid \text{rating at } t = i)$$

关键假设是**马尔可夫性**：下一期的评级分布只依赖于当前评级，与历史路径无关。这个假设在学术文献中被广泛讨论和质疑，但在实务中是标准做法。

迁移矩阵有几个重要性质：
- 每行求和为1：$\sum_j M_{ij} = 1$
- 对角线元素是"维持评级"的概率，通常最大
- 最后一列是各评级到违约（D）的迁移概率
- 矩阵的最后一行全是0除了最后一个元素为1（违约是不可逆的吸收态）

```python
import numpy as np
import pandas as pd

# 一年期迁移矩阵（基于 S&P 历史均值的简化版本）
ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

M = np.array([
    # AAA   AA    A    BBB   BB    B    CCC   D
    [0.90, 0.08, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00],  # AAA
    [0.02, 0.88, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00],  # AA
    [0.00, 0.02, 0.89, 0.07, 0.01, 0.00, 0.00, 0.00],  # A
    [0.00, 0.00, 0.03, 0.86, 0.08, 0.02, 0.00, 0.01],  # BBB
    [0.00, 0.00, 0.01, 0.05, 0.78, 0.10, 0.03, 0.03],  # BB
    [0.00, 0.00, 0.00, 0.02, 0.06, 0.72, 0.08, 0.12],  # B
    [0.00, 0.00, 0.00, 0.01, 0.02, 0.06, 0.60, 0.31],  # CCC
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # D (吸收态)
])

M_df = pd.DataFrame(M, index=ratings, columns=ratings)
print("一年期迁移矩阵：")
print(M_df.round(2))

# 验证每行求和为1
print(f"\n行和验证: {M_df.sum(axis=1).values}")
```

## 二、多期迁移：矩阵幂次法

从一年期迁移矩阵推到 $T$ 年期，最简单的假设是**时间齐次**——每一年的迁移矩阵相同且独立。在这个假设下，$T$ 年期迁移矩阵就是一年期矩阵的 $T$ 次幂：

$$M^{(T)} = M^T$$

这是马尔可夫链的标准性质。矩阵幂次法的优点是计算极其简单；缺点是它隐含了一个强假设：各期之间的迁移是条件独立的，且迁移概率不随经济周期变化。

![各评级累计违约概率曲线](/images/rating-migration-matrix-modeling/cumulative_default_curves.png)

上图展示了各评级在不同时间跨度下的累计违约概率。注意 CCC 的累计违约概率呈指数式增长——5年内超过80%的概率违约。而 AAA 即使在10年后累计违约概率仍极低。这种巨大的异质性正是信用风险建模的核心挑战。

```python
def multi_period_matrix(M, T):
    """
    计算 T 年期迁移矩阵（矩阵幂次法）。
    假设：时间齐次 + 条件独立。
    """
    return np.linalg.matrix_power(M, T)

# 计算5年期迁移矩阵
M5 = multi_period_matrix(M, 5)
print("5年期迁移矩阵：")
print(pd.DataFrame(M5, index=ratings, columns=ratings).round(3))

# 对比：矩阵幂次 vs 假设无迁移（仅看违约概率）
print("\n累计违约概率对比（5年期）：")
for i, rating in enumerate(ratings[:-1]):
    m_power = M5[i, -1]  # 矩阵幂次法
    pd_annual = M[i, -1]
    m_simple = 1 - (1 - pd_annual) ** 5  # 简单累乘
    print(f"  {rating}: 矩阵幂次={m_power:.3%}, 简单累乘={m_simple:.3%}")
```

矩阵幂次法和简单累乘的区别在于：幂次法考虑了"评级迁移后再违约"的路径（比如 BBB → BB → B → D），而简单累乘只考虑"始终维持当前评级然后违约"。在实务中，这两种方法的差异在高风险评级上可以非常显著。

## 三、CreditMetrics 框架：从评级到损益

CreditMetrics 的核心思想是：信用风险不仅来自违约，还来自评级迁移带来的利差变化。一只 BBB 级债券即使不违约，如果被降级到 BB，其利差扩大、价格下跌，也是一种损失。

完整的损益分两步计算：

**第一步：评级迁移** — 根据迁移矩阵决定期末评级。

**第二步：重新定价** — 用新评级对应的远期利差曲线对债券重新估值。

```python
def price_bond_with_spread(face_value, coupon_rate, remaining_years, 
                           risk_free_rate, credit_spread):
    """
    用信用利差对债券重新定价。
    discount_rate = risk_free_rate + credit_spread
    """
    discount_rate = risk_free_rate + credit_spread
    cash_flows = [coupon_rate * face_value] * int(remaining_years)
    cash_flows[-1] += face_value  # 最后一期加本金
    
    pv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, 1))
    return pv

# 各评级对应的远期利差（bps）
credit_spreads = {
    'AAA': 15, 'AA': 25, 'A': 50, 'BBB': 100,
    'BB': 250, 'B': 400, 'CCC': 800, 'D': 2000  # D = 违约回收
}

# 债券参数
face_value = 100
coupon_rate = 0.05  # 5%票息
remaining_years = 5
risk_free_rate = 0.03

# 计算各评级下的债券价值
bond_values = {}
for rating in ratings:
    if rating == 'D':
        # 违约：按回收率定价
        lgd_rate = 0.40  # 违约损失率40%
        recovery_rate = 1 - lgd_rate
        bond_values[rating] = face_value * recovery_rate
    else:
        spread = credit_spreads[rating] / 10000  # bps -> decimal
        bond_values[rating] = price_bond_with_spread(
            face_value, coupon_rate, remaining_years, risk_free_rate, spread
        )

print("各评级下的债券价值：")
for r, v in bond_values.items():
    print(f"  {r}: ${v:.2f}")

# BBB 级债券的迁移损益分布
bbb_idx = ratings.index('BBB')
print(f"\nBBB 级债券（当前价值 ${bond_values['BBB']:.2f}）的迁移损益：")
for j, rating in enumerate(ratings):
    prob = M[bbb_idx, j]
    value = bond_values[rating]
    pnl = value - bond_values['BBB']
    print(f"  → {rating}: P={prob:.2%}, Value=${value:.2f}, P&L=${pnl:+.2f}")
```

## 四、组合损失分布：蒙特卡洛模拟

单只债券的迁移损益比较简单——它有8种结果（对应7个评级+违约），每种有确定的概率和价值。但当你把1000只债券放在一起，组合损失分布变得极其复杂，因为：

1. **相关性**：经济衰退期，多只债券同时降级和违约的概率上升。
2. **维度爆炸**：1000只债券各8种结果，总共有 $8^{1000}$ 种组合。

CreditMetrics 用蒙特卡洛模拟来解决这个问题。核心步骤：

1. 生成一个 $N \times T$ 的相关标准正态随机数矩阵（用相关矩阵建模依赖性）
2. 对每个债券每个时间点，把标准正态值映射到评级（用迁移概率作为阈值）
3. 根据最终评级计算损益
4. 重复10000次，得到组合损失分布

```python
def creditmetrics_portfolio_simulation(
    M,                    # 迁移矩阵
    exposures,            # 每只债券的敞口列表 [(rating_idx, face_value, lgd), ...]
    n_simulations=10000,
    asset_correlation=0.15,  # 资产相关性
):
    """
    CreditMetrics 风格的组合损失蒙特卡洛模拟。
    使用阈值模型将相关正态变量映射到评级。
    """
    n_bonds = len(exposures)
    
    # 计算每个评级的累积概率（用于阈值映射）
    def rating_thresholds(migration_row):
        """从迁移概率行向量计算累积阈值"""
        cum_probs = np.cumsum(migration_row)
        # 标准正态分位数
        thresholds = stats.norm.ppf(cum_probs[:-1])
        return thresholds
    
    from scipy import stats
    
    # 为每个初始评级计算阈值
    threshold_cache = {}
    for i in range(len(ratings)):
        threshold_cache[i] = rating_thresholds(M[i, :])
    
    # 生成相关正态随机数
    # 使用单因子模型：Z_i = sqrt(rho) * Y + sqrt(1-rho) * eps_i
    rho = asset_correlation
    
    portfolio_losses = np.zeros(n_simulations)
    
    for sim in range(n_simulations):
        # 系统性因子
        Y = np.random.standard_normal()
        
        total_loss = 0
        for bond_idx, (rating_idx, face_value, lgd) in enumerate(exposures):
            # 债券特有因子
            eps = np.random.standard_normal()
            Z = np.sqrt(rho) * Y + np.sqrt(1 - rho) * eps
            
            # 映射到期末评级
            thresholds = threshold_cache[rating_idx]
            final_rating = len(thresholds)  # 默认违约
            for j, t in enumerate(thresholds):
                if Z <= t:
                    final_rating = j
                    break
            
            # 计算损益
            if final_rating == len(ratings) - 1:  # 违约
                loss = face_value * lgd
            else:
                # 评级迁移导致的盯市损益（简化：只用利差变化）
                new_spread = credit_spreads[ratings[final_rating]] / 10000
                new_value = price_bond_with_spread(
                    face_value, 0.05, 5, 0.03, new_spread
                )
                old_value = price_bond_with_spread(
                    face_value, 0.05, 5, 0.03, 
                    credit_spreads[ratings[rating_idx]] / 10000
                )
                loss = max(0, old_value - new_value)
            
            total_loss += loss
        
        portfolio_losses[sim] = total_loss
    
    return portfolio_losses

# 构建测试组合
portfolio = []
for i, (r, pct) in enumerate(zip(range(7), [0.10, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03])):
    n = int(pct * 1000)
    for _ in range(n):
        portfolio.append((r, 100, 0.60))  # 每只面值100，LGD 60%

# 运行模拟
losses = creditmetrics_portfolio_simulation(
    M, portfolio, n_simulations=5000, asset_correlation=0.20
)

# 计算风险指标
from scipy import stats

print(f"模拟次数: {len(losses)}")
print(f"平均损失: ${losses.mean():,.0f}")
print(f"损失标准差: ${losses.std():,.0f}")
print(f"95% VaR: ${np.percentile(losses, 95):,.0f}")
print(f"99% VaR: ${np.percentile(losses, 99):,.0f}")
print(f"99.9% VaR: ${np.percentile(losses, 99.9):,.0f}")

# 经济资本 = 99% VaR - 预期损失
economic_capital = np.percentile(losses, 99) - losses.mean()
print(f"\n经济资本 (99%): ${economic_capital:,.0f}")
```

![多期组合损失分布](/images/rating-migration-matrix-modeling/loss_distribution.png)

上图展示了在不同时间跨度（1年、3年、5年）下的组合损失分布。随着时间拉长，损失分布的右尾明显变厚——这反映了多期迁移矩阵中违约概率的指数式累积效应。注意5年期的分布尾部明显比1年期更长，但峰值也更低，反映了更大的不确定性。

## 五、相关性的角色：从独立到系统性风险

上面的模拟中有一个关键参数：`asset_correlation`。这个参数决定了不同债券的违约/降级之间的依赖程度，也是 Basel IRB 框架中的核心监管参数。

- **相关性 = 0**：各债券独立违约，组合损失分布服从二项分布的卷积，尾部很薄。
- **相关性 = 1**：所有债券同时违约或不违约，组合损失呈双峰分布，尾部极厚。
- **中间值**：最符合现实。Basel 给出的公式是 $\rho = 0.12 \times \frac{1 - e^{-50 \cdot PD}}{1 - e^{-50}} + 0.24 \times (1 - \frac{1 - e^{-50 \cdot PD}}{1 - e^{-50}})$

```python
def basel_correlation(pd_annual):
    """
    Basel IRB 使用的资产相关性公式。
    PD 越低，相关性越高（因为系统性因素主导）。
    PD 越高，特异性因素更重要。
    """
    rho = 0.12 * (1 - np.exp(-50 * pd_annual)) / (1 - np.exp(-50)) + \
          0.24 * (1 - (1 - np.exp(-50 * pd_annual)) / (1 - np.exp(-50)))
    return rho

print("Basel 相关性 vs 违约概率：")
for pd_val in [0.0001, 0.001, 0.01, 0.05, 0.10, 0.20]:
    rho = basel_correlation(pd_val)
    print(f"  PD={pd_val:.4f} → ρ={rho:.4f}")

# 不同相关性下的组合损失尾部
np.random.seed(42)
for rho_val in [0.05, 0.15, 0.30, 0.50]:
    losses_r = creditmetrics_portfolio_simulation(
        M, portfolio[:200], n_simulations=3000, asset_correlation=rho_val
    )
    var99 = np.percentile(losses_r, 99)
    print(f"ρ={rho_val:.2f}: 99% VaR = ${var99:,.0f}, "
          f"预期损失 = ${losses_r.mean():,.0f}")
```

相关性对尾部风险的影响是决定性的。$\rho$ 从5%升至50%，预期损失几乎不变，但99% VaR 可能翻倍。这就是为什么 Basel 对低 PD 资产（如高评级零售贷款）反而赋予更高的相关性——因为这些资产的违约主要由系统性因素驱动，在经济衰退时更容易批量违约。

## 六、模型局限与现实修正

迁移矩阵模型虽然在银行业广泛应用，但有几个重要局限：

**1. 马尔可夫假设的偏差**：真实的评级迁移有记忆性——一只已经被降级一次的债券，再次被降级的概率高于"维持在当前评级"所暗示的水平。这与马尔可夫假设矛盾。修正方法是使用非时齐马尔可夫模型或加入"评级动量"项。

**2. 周期效应**：迁移矩阵在经济扩张期和衰退期差异巨大。使用全样本均值矩阵会平滑掉这种周期性。修正方法是使用"条件迁移矩阵"——把迁移概率建模为宏观经济因子的函数。

**3. 评级惰性**：评级机构的反应往往滞后于市场。当 CDS 利差已经开始大幅扩大时，评级可能还没动。这意味着迁移矩阵模型可能低估了实际的信用风险变化速度。

**4. 迁移矩阵的稳定性**：不同时间窗口估计的迁移矩阵差异很大。20年的长期均值可能与未来1年的实际迁移概率相去甚远。

## 七、总结

评级迁移矩阵是信用风险建模的基础工具，它的价值在于把"评级是一个动态过程"这一直觉形式化为可计算的数学框架。矩阵幂次法提供了从单期到多期的简洁推导路径，CreditMetrics 框架进一步把迁移和重新定价结合，生成了组合层面的损失分布。

但这个框架的核心假设——马尔可夫性、时间齐次、条件独立——每一条都有明确的现实偏差。理解这些假设在哪里失效，比会用公式更重要。在实际应用中，迁移矩阵模型往往只是起点：在此基础上加入评级动量、宏观条件化、市场信号融合等修正，才能得到足够贴近现实的风险画像。

最终，信用风险建模的挑战不在于数学有多精巧——而在于你能在多大程度上捕捉到"公司从健康到违约"这条路径上那些模型无法完全描述的非线性跳跃。
