---
title: "Amihud 非流动性溢价：用交易摩擦挖出稳健的选股因子"
description: "流动性不是免费的。Amihud(2002)用 ILLIQ=|r|/成交额 度量「砸动价格需要多少钱」，发现过去成交清淡的股票未来平均多赚约 12% 年化——这是少数经得起样本外拷打的异象。本文从定义、Python 计算、五分组多空检验讲到 A股实务，并指出 ILLIQ 极右偏、幸存者偏差、流动性时变三大真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 因子研究
  - 流动性
  - Amihud
  - 选股
  - 市场微结构
  - Python
language: Chinese
difficulty: advanced
---

一句话结论：**流动性是一种被定价的风险。** Amihud(2002)用极简指标 $\text{ILLIQ}=|r|/(\text{价格}\times\text{成交量})$ 度量「每花一美元成交额，价格会被撼动多少」，发现过去一个月最不流动（ILLIQ 最高）的股票组合，未来相对最流动组合能多赚约 **12% 年化**，且 t 统计高度显著。这个「非流动性溢价」之所以在因子圈长青，是因为它不靠复杂模型、不依赖特定样本区间，本质上是市场对「交易摩擦」的补偿——而交易摩擦，是真实存在的。

## 一、ILLIQ 到底在量什么

日度非流动性定义为：

$$\text{ILLIQ}_d = \frac{|r_d|}{\text{Price}_d \times \text{Volume}_d}$$

分母 $\text{Price}\times\text{Volume}$ 是当日美元成交额（A股用元）。分子是当日绝对收益。直觉：**同样涨 1%，如果只用了很少的成交额就撬动，说明这只股票「很轻、易被撼动」→ 流动性差**。把一个月的日度 ILLIQ 取中位数（不是均值！见图3）再 $\times 10^9$ 方便读数，得到月度 $\text{ILLIQ}_m$。

为什么用绝对值 + 中位数？(1) 收益绝对值抹掉方向，纯粹度量「价格对成交的弹性」；(2) ILLIQ 极度右偏（少数僵尸股把均值拉爆），必须用截面排序或中位数，直接对 ILLIQ 求均值毫无意义。

## 二、Python：从原始行情到 ILLIQ

```python
import numpy as np

rng = np.random.default_rng(20260711)
N, T = 150, 60                       # 150 只股票 × 60 个月

# 构造真实流动性档位 + 时变噪声（危机期流动性骤降）
log_illiq = rng.normal(2.0, 1.0, N)
illiq = np.exp(log_illiq[:, None] + rng.normal(0, 0.30, (N, T)))   # (N,T) 月度 ILLIQ×1e9

# 让“不流动性”携带正向溢价：不流动性越高，预期月收益越高
signal = (log_illiq - log_illiq.mean()) / log_illiq.std()
ret = 0.006 + 0.004 * signal[:, None] + rng.normal(0, 0.045, (N, T))
```

真实数据里，你该用 `df.groupby('stock')` 后逐月算 `abs(close.pct_change()).sum() / dollar_volume.sum() * 1e9`。注意是**月度聚合后再排序**，不是用日度 ILLIQ 直接排序——日度噪声太大。

## 三、五分组多空检验：溢价到底显不显著

经典做法：每个月末，按当月 ILLIQ 把股票分 5 组（Q1 最流动 → Q5 最不流动），持有**下个月**收益，统计多空组合（Q5 − Q1）的月均与 t 统计：

```python
illiq_lag, ret_fwd = illiq[:, :-1], ret[:, 1:]      # t 月 ILLIQ 预测 t+1 月收益
n = illiq_lag.shape[1]
q_ret = np.zeros((5, n)); ls = np.zeros(n)
for t in range(n):
    order = np.argsort(illiq_lag[:, t])
    q = np.empty(N, int); q[order] = np.minimum(4, np.arange(N) * 5 // N)
    for k in range(5):
        q_ret[k, t] = ret_fwd[q == k, t].mean()
    ls[t] = q_ret[4, t] - q_ret[0, t]               # Q5(最不流动) - Q1(最流动)

ls_ann = ls.mean() * 12                              # 年化多空收益
ls_t = ls.mean() / (ls.std(ddof=1) / np.sqrt(n))    # 朴素 t；真实用 Newey-West 修正自相关
```

![按非流动性分5组：收益单调向上，多空组合显著为正](/images/amihud-illiquidity-premium/amihud_quintile.png)

仿真里五组月均收益分别是 **0.12% / 0.27% / 0.61% / 0.78% / 1.14%**，严格单调递增——越不流动，未来收益越高。多空组合月均 1.02%、年化约 12.3%。散点图更直观：

![不流动性越高，下月收益越高：Amihud 溢价的第一眼证据](/images/amihud-illiquidity-premium/amihud_scatter.png)

OLS 斜率约 **+0.78%**（每多一个 log10 单位 ILLIQ，下月收益高约 0.78%），正相关清晰。

## 四、它会不会只是 size / value 的影子？

Amihud 溢价最常被质疑的一点：小盘股、价值股天然不流动，溢价是不是「size/value 因子的马甲」？实证结论是**部分吸收、并未消失**：控制市值和账面市值比后，ILLIQ 仍保留显著残差溢价（文献约 1%–7%/年，比未控制时小但仍显著）。所以正确用法不是「裸多空 ILLIQ」，而是把它当作**正交化后的残差因子**叠进多因子模型——先对 size、value 回归取残差，再排序。

## 五、A股实务要点

- **涨跌停板**让成交量在极端日被迫截断，ILLIQ 会失真；用过去 3–6 个月**中位数**而非均值，且剔除停牌日。
- **微盘股流动性黑洞**：成交集中于少数交易日，月度 ILLIQ 跳动极大，必须做截面 winsorize（如 1%/99%）。
- **流动性是时变的**：2015 股灾、2018 去杠杆、2024 微盘踩踏，全市场 ILLIQ 同步飙升，那时「分组」会塌掉——要把 ILLIQ 相对自身历史分位标准化，而非用绝对水平。
- **交易成本的双刃剑**：不流动股票 bid-ask 宽、冲击成本高，12% 毛溢价里相当一部分会被摩擦吃掉，实盘要扣完成本再算 net alpha。

## 真实陷阱（必须说清楚）

1. **极右偏**：ILLIQ 中位数 6.6、均值 13.7、偏度 4.4（见下图）。直接求均值、或直接回归原始 ILLIQ，会被长尾带偏；务必排序 / log / winsorize。

![ILLIQ 极度右偏：均值被极少数僵尸股拉爆，必须用截面排序](/images/amihud-illiquidity-premium/amihud_distribution.png)

2. **幸存者偏差**：退市股往往正是「不流动 + 暴跌」，把它们从历史列表删掉，会系统性高估溢价。回测必须含已退市标的。
3. **流动性时变 + 全市场同步**：危机期所有股票一起变不流动，横截面区分度下降，多空组合在极端月可能不赚反亏。
4. **风险补偿 vs 错误定价难分**：高 ILLIQ 常伴随高 idio 波动，多赚的钱究竟是为「承受流动性风险」还是「市场定价错误」，没有干净答案——这意味着溢价里有一部分是该拿的补偿，不是免费午餐。
5. **仿真量级 ≠ 真实量级**：本例为结构化仿真，年化 12% 与极高 t 是为了演示方法的单调性；真实 Amihud 溢价文献区间约 1%–7%/年、t 统计 2–6。方法可照搬，别直接套数字。

---

**收尾**：Amihud 的价值不在「暴利因子」，而在它是一个**简洁、稳健、可解释**的流动性风险度量。它提醒你，任何「看起来能赚钱」的信号，第一反应都该问一句：这是 alpha，还是市场对我承担的流动性风险付的费？
