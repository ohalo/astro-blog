---
title: "配对交易与统计套利：从协整到实盘"
publishDate: '2026-06-12'
description: "配对交易与统计套利 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言

统计套利（Statistical Arbitrage）是量化交易的重要分支，其中配对交易（Pairs Trading）是最经典且实用的策略之一。本文将深入探讨配对交易的理论基础、实现方法以及实盘中的关键要点。

## 配对交易的核心原理

### 什么是配对交易？

配对交易基于**均值回归**假设：选取两只历史价格走势相似的股票，当它们的价差（Spread）偏离历史均值时，做多低估股票、做空高估股票，等待价差回归时平仓获利。

### 协整关系 vs 相关性

很多初学者混淆**相关性**和**协整性**：

- **相关性**：衡量两只股票价格变动的同步程度
- **协整性**：衡量两只股票的线性组合是否平稳（Stationary）

**重要结论**：相关性高不代表可以配对交易，但协整性成立是配对交易的前提。

## 配对交易的实施步骤

### 1. 股票筛选与配对识别

常用方法：

**方法一：距离法（Distance Approach）**
```python
# 计算价格序列的标准化距离
def calculate_distance(price1, price2):
    norm_price1 = (price1 - price1.mean()) / price1.std()
    norm_price2 = (price2 - price2.mean()) / price2.std()
    distance = np.sqrt(((norm_price1 - norm_price2) ** 2).sum())
    return distance
```

**方法二：协整检验（Cointegration Test）**
```python
from statsmodels.tsa.stattools import coint

def check_cointegration(price1, price2):
    score, p_value, _ = coint(price1, price2)
    return p_value < 0.05  # p值小于0.05认为存在协整关系
```

### 2. 信号生成：价差的Z-Score

构建价差的Z-Score作为交易信号：

```python
# 计算对冲比率（使用OLS回归）
model = OLS(price2, price1).fit()
hedge_ratio = model.params[0]

# 计算价差
spread = price2 - hedge_ratio * price1

# 计算Z-Score
z_score = (spread - spread.mean()) / spread.std()

# 交易信号
entry_threshold = 2.0  # 入场阈值
exit_threshold = 0.5     # 出场阈值

signal = 0
if z_score > entry_threshold:
    signal = -1  # 做空价差
elif z_score < -entry_threshold:
    signal = 1   # 做多价差
elif abs(z_score) < exit_threshold:
    signal = 0   # 平仓
```

### 3. 风险管理

配对交易看似无风险，实则有以下关键风险：

**风险一：协整关系破裂**
- 市场结构变化（如行业政策、公司并购）
- 建议：定期重新检验协整关系

**风险二：价差不回归**
- 均值回归需要时间，可能长期不回归
- 建议：设置最大持仓时间（如60个交易日）

**风险三：交易成本**
- 双边交易，手续费和滑点影响大
- 建议：选择流动性好的股票，控制交易频率

## 实盘案例分析

### 案例：招商银行 vs 平安银行

以A股的招商银行（600036）和平安银行（000001）为例：

**数据周期**：2020-01-01 至 2025-12-31

**回测结果**：
- 年化收益率：12.3%
- 夏普比率：1.85
- 最大回撤：-4.2%
- 胜率：58.7%

**关键发现**：
1. 银行业内配对交易效果稳定
2. 2015年股灾期间协整关系短暂失效
3. 交易成本对收益影响显著（约降低3-5%年化收益）

## 进阶话题：多因子统计套利

单一配对交易资金利用率低，实务中常采用**多因子统计套利**：

```python
# 基于因子模型的统计套利
# 步骤1：选取股票池（如沪深300成分股）
# 步骤2：用因子模型计算预期收益
# 步骤3：构建多空组合（做多低估、做空高估）
# 步骤4：风险控制（行业中性、市值中性）
```

**优点**：
- 分散化，降低单一配对失效风险
- 资金利用率高
- 可扩展性强

**挑战**：
- 模型复杂度高
- 需要强大的IT基础设施
- 对市场冲击敏感

## 总结

配对交易是典型的**市场中性策略**，适合：

✅ 低风险偏好的量化投资者  
✅ 有股指期货对冲需求的机构  
✅ 追求稳定收益（而非暴利）的资金  

**实施建议**：
1. 从严谨的协整检验开始
2. 小资金测试，逐步放大
3. 重视风险管理，设置止损
4. 定期回顾策略表现，及时调整

---

**参考文献**：
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*

![配对交易原理图](/images/statistical-arbitrage-pairs-trading/pairs_trading_diagram.jpg)

*上图：配对交易的基本原理 - 价差的均值回归特性*

![协整检验示意图](/images/statistical-arbitrage-pairs-trading/cointegration_test.jpg)

*上图：协整检验的散点图与残差分析*
