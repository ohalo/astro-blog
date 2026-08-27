---
title: "主权 CDS 基差：违约保护为什么比债券便宜"
description: "同样一个主权信用风险，为什么 CDS 保护经常比现金债券更便宜？拆解 CDS-Bond Basis 的来源、符号与套利边界，附 Python 模拟与实务风险清单。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 信用债
  - CDS
  - 基差套利
categories: ["量化交易"]
slug: "sovereign-cds-basis"
image: "/images/sovereign-cds-basis/cds_bond_basis_bar.png"
---

如果你同时关注主权债券市场和 CDS 市场，会经常看到一个奇怪现象：**同一个国家的违约风险，现金债券要求的利差可能比 CDS 保护更高**。例如意大利 10 年期国债比德国国债高 120bp，而 5 年期主权 CDS 却只有 105bp。两者之间的差值，就是**CDS-Bond Basis**。

理论上，这个 basis 不应该长期存在。买债券 + 买 CDS 保护，可以把信用风险剥离成近似无风险组合。如果债券利差明显高于 CDS 溢价，套利者应该涌入买入债券、买入保护，把 basis 压平。可现实中，basis 不仅存在，还经常以**正值**持续数月甚至数年。这篇文章解释它为什么存在、为什么不能简单套利，以及如何用量化方式跟踪它。

![主权 CDS 与债券利差：多数国家存在正 basis](/images/sovereign-cds-basis/cds_bond_basis_bar.png)

## 一、CDS-Bond Basis 的定义与计算

最常用的定义是：

```
Basis = 债券利差（Asset Swap Spread 或 Z-Spread） − CDS 利差
```

- **Basis > 0**：债券更贵（利差更高），CDS 保护相对便宜。
- **Basis < 0**：债券更便宜，CDS 更贵。

对于主权市场，债券利差常用该国国债与德国国债的收益率差（Bund spread）近似，CDS 利差则用 5 年期主权 CDS 报价。严格来说应使用 asset-swap spread，但 Bund spread 已经足以揭示方向。

```python
import pandas as pd
import numpy as np

countries = ['Germany', 'France', 'Italy', 'Spain', 'Portugal', 'Greece', 'Brazil', 'Turkey', 'South Africa']
bond_spread = np.array([0.25, 0.35, 1.20, 0.90, 1.80, 3.50, 2.80, 4.20, 3.00])
cds_spread = np.array([0.15, 0.25, 1.05, 0.75, 1.55, 3.10, 2.20, 3.60, 2.40])

df = pd.DataFrame({
    'country': countries,
    'bond_spread_%': bond_spread,
    'cds_spread_%': cds_spread,
    'basis_%': bond_spread - cds_spread
})
print(df.round(2))
print(f"\n平均 basis: {df['basis_%'].mean():.2f}%，中位数: {df['basis_%'].median():.2f}%")
```

## 二、为什么会产生正 Basis？四个结构性原因

### 1. CDS 不是债券的完美复制

CDS 和债券在**违约触发、回收流程、交割方式、期限结构**上都有差异：
- 债券违约由政府破产或重组事件触发，CDS 违约由 ISDA 委员会裁定「信用事件」触发。
- CDS 现金结算与债券实物交割存在基差。
- CDS 是 5 年期标准合约，债券可能有 10 年、30 年等不同期限。

这些「不完全复制」意味着「买债 + 买 CDS」不是无风险套利，而是**近似对冲**。

### 2. 做空债券的摩擦

正 basis 套利需要「做多债券、买 CDS 保护」。但债券端的「做空」对很多人来说是反向操作：
- 做空国债需要回购市场（repo）借券，成本不低。
- 主权债券 often on special，借券困难。
- 裸卖空在许多市场受监管限制。

因此，把债券利差压低的套利力量受到约束，basis 可以长期为正。

### 3. 市场分割与投资者结构差异

- **债券市场** dominated by real money（养老金、保险公司、央行），他们对收益率和久期有刚性需求，愿意持有「利差高一点」的债券。
- **CDS 市场** dominated by hedge funds、交易商、宏观基金，他们对价格更敏感、杠杆更高。

两个市场的供需曲线不同，定价自然分离。

### 4. 融资成本和流动性差异

CDS 是衍生品，通常只需要缴纳变动保证金；债券是现金资产，需要全额资金或 repo。当整体融资紧张时（如 2020 年 3 月），债券端成本上升会推高 basis。危机中 basis 往往先**扩大**再收敛，因为套利资本在流动性冲击下被迫退出。

![CDS-Bond Basis 时序：危机中扩大，长期围绕正值波动](/images/sovereign-cds-basis/cds_basis_timeseries.png)

## 三、Python 模拟：CDS-Bond Basis 的驱动因子

下面做一个受控实验，分别看融资成本、流动性溢价和违约概率如何影响 basis。

```python
import numpy as np
import pandas as pd

def cds_bond_basis(pd_annual, recovery, bond_liquidity_premium, cds_liquidity_premium,
                   bond_funding_cost, cds_funding_cost):
    """
    简化模型：
    - 债券利差 ≈ 风险中性违约损失 + 债券流动性溢价 + 债券融资成本溢价
    - CDS 利差  ≈ 风险中性违约损失 + CDS 流动性溢价 + CDS 融资成本溢价
    """
    risk_neutral_loss = pd_annual * (1 - recovery)
    bond_spread = risk_neutral_loss + bond_liquidity_premium + bond_funding_cost
    cds_spread  = risk_neutral_loss + cds_liquidity_premium + cds_funding_cost
    return bond_spread - cds_spread, bond_spread, cds_spread

scenarios = {
    '基准':    {'pd': 0.03, 'rec': 0.40, 'bond_liq': 0.0010, 'cds_liq': 0.0005,
               'bond_fund': 0.0005, 'cds_fund': 0.0002},
    '高流动性溢价': {'pd': 0.03, 'rec': 0.40, 'bond_liq': 0.0020, 'cds_liq': 0.0005,
                  'bond_fund': 0.0005, 'cds_fund': 0.0002},
    '危机融资紧张': {'pd': 0.05, 'rec': 0.35, 'bond_liq': 0.0030, 'cds_liq': 0.0010,
                 'bond_fund': 0.0020, 'cds_fund': 0.0005},
    'CDS 更贵':   {'pd': 0.03, 'rec': 0.40, 'bond_liq': 0.0005, 'cds_liq': 0.0015,
                 'bond_fund': 0.0005, 'cds_fund': 0.0002},
}

rows = []
for name, p in scenarios.items():
    basis, b_spread, c_spread = cds_bond_basis(**p)
    rows.append([name, b_spread*100, c_spread*100, basis*100])

res = pd.DataFrame(rows, columns=['情景', '债券利差%', 'CDS利差%', 'Basis%'])
print(res.round(2))
```

这个模型清楚显示：**basis 的符号和大小由多个摩擦共同决定**，而不是单纯由违约概率驱动。即使真实违约风险不变，债券流动性变差或融资变贵，basis 就会扩大。

## 四、套利边界：Basis 不会无限扩大，但也不能轻易消除

理论上，正 basis 的套利收益上限是：

```
套利收益 ≈ Basis − 做空债券成本 − CDS 溢价 − 资本占用成本 − 违约结算摩擦
```

一旦 basis 超过这些成本之和，套利资金就会进入。但在实务中：
- **做空主权债券成本高**：尤其在压力时期，repo 利率飙升。
- **CDS 保护有对手方风险**：需要缴纳变动保证金，压力期 margin 要求上升。
- **信用事件定义不确定**：CDS 委员会是否裁定违约存在主观性。
- **资本占用**：银行做这笔套利需要占用风险资本，回报被稀释。

因此，basis 的「无套利区间」很宽，市场可以在区间内长期偏离。

![买入债券 + 买入 CDS 的近似无风险套利净值路径](/images/sovereign-cds-basis/bond_cds_arbitrage.png)

## 五、量化跟踪：一个可运行的 basis 监控脚本

下面给出一个简化但实用的监控框架：读入债券利差和 CDS 利差序列，计算 rolling basis、z-score 和偏离阈值。

```python
import numpy as np
import pandas as pd

np.random.seed(11)
dates = pd.date_range('2020-01-01', periods=500, freq='B')

# 模拟意大利 BTP-Bund spread 和 5Y CDS
btp_bund = 1.20 + 0.30 * np.sin(np.linspace(0, 6*np.pi, len(dates))) \
           + np.random.normal(0, 0.10, len(dates))
cds_italy = btp_bund - 0.15 + np.random.normal(0, 0.05, len(dates))

# 加入危机爆发
crisis = slice(300, 380)
btp_bund[crisis] += np.linspace(0, 1.50, len(dates[crisis]))
cds_italy[crisis] += np.linspace(0, 1.20, len(dates[crisis]))

basis = btp_bund - cds_italy

# 计算 rolling z-score
window = 60
basis_ma = pd.Series(basis).rolling(window).mean()
basis_std = pd.Series(basis).rolling(window).std()
zscore = (basis - basis_ma) / basis_std

# 信号：zscore > 2 视为 basis 显著走阔（债券相对 CDS 偏贵）
signal = np.where(zscore > 2, '偏贵', np.where(zscore < -2, '偏便宜', '中性'))

df = pd.DataFrame({
    'date': dates,
    'bond_spread': btp_bund,
    'cds_spread': cds_italy,
    'basis': basis,
    'zscore': zscore,
    'signal': signal
})
print(df.tail(20).round(3))

print(f"\nZ-score 最大: {zscore.max():.2f}，最小: {zscore.min():.2f}")
print(f"Basis 为正的比例: {(basis > 0).mean():.1%}")
```

这个框架的用途不是直接交易，而是：
1. **识别异常**：当 basis 突破 2-sigma，说明债券与 CDS 市场出现明显分歧。
2. **寻找解释**：结合 repo 利率、CDS 流动性、市场新闻判断是「真套利机会」还是「假信号」。
3. **风险管理**：若你持有债券多头，而 CDS 显著便宜，可考虑买保护对冲。

## 六、实务中的三个陷阱

### 1. 把 Bund spread 直接当债券利差

 Bund spread 包含了德国国债的「避风港溢价」，在危机中德国利率下行会人为拉大 spread。更准确的做法是用 asset-swap spread 或 matched-maturity spread。

### 2. 忽视 CDS 的「最便宜可交割债券」选项

CDS 保护买方在信用事件发生后可以交割最便宜的可交割债券，这意味着 CDS 利差隐含的是「最便宜债券」的回收率，而不是你手里的那只债券。

### 3. 把 basis 当作「无风险套利」

2020 年 3 月，许多 basis 套利基金同时亏损：债券端流动性枯竭，CDS 端 margin call。_basis 套利不是无风险，而是低夏普、高左尾的「流动性风险套利」。_

## 七、结语

主权 CDS-Bond Basis 之所以迷人，是因为它直指一个核心问题：**同样一种信用风险，在不同市场、不同投资者、不同工具中，价格为什么会系统性地不一样？**

答案不是市场「错了」，而是市场被不同约束切割。债券市场有 real money 的久期需求、做空摩擦和融资约束；CDS 市场有杠杆、保证金和对手方风险。两个市场的供需曲线交汇在不同位置，basis 就此诞生。

对于量化交易者，basis 不是一棵摇钱树，而是一个**持续监测、偶尔出击、严格风控**的相对价值信号。它的价值在于告诉你：当债券和 CDS 出现明显分歧时，市场正在对「流动性、融资、违约定义」给出不同定价——而分歧本身，就是 alpha 的温床。

*本文代码均在 Python 3 环境下可直接运行，数据为模拟生成，仅用于教学演示，不构成任何投资建议。*
