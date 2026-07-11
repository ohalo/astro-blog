---
title: "日历价差与跨期套利：赚期限结构曲线的钱"
description: "买远月、卖近月、同行权价——日历价差赚的是近月比远月衰减更快的 Theta 差，以及波动率期限结构维持 contango 时远月腿的保值 carry。本文用 Black-Scholes 真实定价 + 蒙特卡洛，拆解它的盈亏剖面、Greeks 与三个真实陷阱，并区分它和商品期货跨期套利的不同。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 期权
  - 日历价差
  - 跨期套利
  - 期限结构
  - Python
language: Chinese
difficulty: advanced
---

一句话结论：**日历价差（Calendar Spread）是「卖时间衰减快的那只、买时间衰减慢的那只」的价差——近月腿衰减快、远月腿衰减慢，中间的差额就是你的利润；而波动率期限结构的 contango（远月 IV 高于近月）让远月腿到期时更值钱，构成第二道引擎。** 但我们的蒙特卡洛（S0=K=100，近月 30 天 / 远月 90 天，近月 IV 19%、远月 IV 21%）给出冷冰冰的数字：无管理地持有到近月到期，期望盈亏 **−0.20**、胜率仅 **44.8%**、最大亏损 −2.38、最大盈利 +1.70——它是一张「标的守住行权价附近才赚钱」的短期票，不是印钞机。

## 一、日历价差是什么：同行权价、不同到期

买入一张**远月**看涨（或看跌）、同时卖出一张**近月**、同**行权价 K**、同标的的合约。因为远月更贵，建仓是净支出（debit spread）：

```
持仓 = +1 张 远月 CALL(K, T₂)  −1 张 近月 CALL(K, T₁)   其中 T₂ > T₁
```

它和垂直价差（同到期、不同行权价）最大的区别：两张合约到期日不同，所以它们的**时间价值衰减速度不同**——这正是利润来源。近月只剩 30 天，远月还有 90 天；近月每天掉的肉，远月只掉一点点，差额被你净收。

## 二、Theta 衰减差：利润的真正引擎

先写一段无依赖的 Black-Scholes 定价与 Theta：

```python
import math
import numpy as np

def _ndf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ndf(d1) - K * math.exp(-r * T) * _ndf(d2)

def bs_call_theta(S, K, T, r, sigma):
    if T <= 1e-6:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    return -(S * pdf * sigma) / (2.0 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ndf(d1 - sigma * math.sqrt(T))
```

近月 ATM 期权的 Theta（每日时间价值流失速度）是远月的数倍。我们画的衰减曲线很直白：近月价值在最后两周「跳水」，远月则平缓下坡。日历价差每天净赚的，就是「近月 Theta 的绝对值 − 远月 Theta 的绝对值」。

![Theta 衰减对比：近月（红）比远月（蓝）掉血快得多，差额即日历价差的每日收入](/images/calendar-spread-arbitrage/calendar_theta.png)

## 三、波动率期限结构：第二道引擎是 contango 的 carry

光有 Theta 还不够——如果两腿 IV 相同，日历价差在「标的停留不动」时只是把时间价值差搬来搬去，并不自带正期望。真正的增厚来自**波动率期限结构**：当远月 IV 高于近月 IV（contango），你买入的远月腿更「贵」也更「抗跌」——近月到期后，远月腿剩 60 天、还顶着较高的 IV，价值更厚。换言之，日历价差是**做多期限结构斜率**：你赌 contango 维持，远月 IV 相对近月不塌。

![波动率期限结构：向上倾斜（contango）时，卖便宜的近月、买贵的远月，净吃正向 carry](/images/calendar-spread-arbitrage/calendar_iv_term.png)

注意方向相反的情形：若期限结构**倒挂**（近月 IV 高于远月，常见于事件前），远月腿反而不值钱，日历价差吃亏。所以「赚期限结构曲线」的前提是 contango 不塌——一旦市场进入恐慌、近月 IV 飙升，你的多斜率头寸会被反噬。

## 四、盈亏剖面：最大利润在 ATM

建仓净支出 = 远月权利金 − 近月权利金。在近月到期那一刻，近月腿归零为内在价值，远月腿还剩 T₂−T₁ 的时间：

```python
S0 = K = 100.0
r = 0.0
sigma_near, sigma_far = 0.19, 0.21   # contango：远月 IV 更高
T1, T2 = 30/252, 90/252
T_rem = T2 - T1
net_debit = bs_call(S0, K, T2, r, sigma_far) - bs_call(S0, K, T1, r, sigma_near)

def pnl_at_front_expiry(s):
    back = bs_call(s, K, T_rem, r, sigma_far)   # 远月腿剩余价值
    front = max(s - K, 0.0)                       # 近月腿到期内在价值
    return back - front - net_debit
```

![盈亏剖面：钟形，最大利润出现在 S≈K（远月时间价值最大、近月已归零）](/images/calendar-spread-arbitrage/calendar_payoff.png)

曲线是标准的钟形：标的停在本金 K 附近时，远月腿时间价值最大、近月腿刚好归零，利润最高（本例约 +1.70）；标的单边跑远，远月腿退化为内在价值、时间溢价消失，利润塌回 −net_debit 附近。这正是**短期权（short gamma）**的面孔：靠横盘吃 Theta，怕单边突破。

## 五、蒙特卡洛：不同走势下谁赚谁亏

用 GBM 模拟近月到期时的标的价格，看整张头寸的盈亏分布：

```python
n = 6000
rng = np.random.default_rng(2024)
Z = rng.normal(size=n)
S_T1 = S0 * np.exp((r - 0.5*0.20**2)*T1 + 0.20*math.sqrt(T1)*Z)
pnl = np.array([pnl_at_front_expiry(s) for s in S_T1])
print(pnl.mean(), (pnl > 0).mean(), pnl.min(), pnl.max())
```

![盈亏分布：标的小幅徘徊(≈K)时赚钱，大幅偏离则亏损——典型的中性头寸](/images/calendar-spread-arbitrage/calendar_pnl_dist.png)

分布告诉我们三件事：(1) 盈利核心集中在 S 贴近 K 的窄带；(2) 亏损尾巴拖得很长（最大 −2.38，等于把净支出全亏掉）；(3) 无管理持有的期望 **−0.20**、胜率 **44.8%**——这不是「买了就赚」，而是「赌横盘 + 管得住」的结构。实务里它靠**主动管理**把期望翻正：近月到期前若标的还在 K 附近，平仓落袋（吃到整段钟形顶部）；若标的突破，提前止损（砍掉短 gamma 的亏损尾巴）。

## 六、和期货跨期套利不是一回事

期货日历价差（买远月、卖近月）赚的是**展期收益（roll yield）**：contango 下近月贵、远月便宜，做空近月做多远月，每次移仓赚贴水；它吃的是商品库存/持有成本决定的期限结构，和标的方向弱相关。期权日历价差赚的是 **Theta 衰减差 + 波动率期限结构**，是短期权、怕单边。两者同名「跨期」，但一个赚持有成本的 carry、一个赚时间价值的衰减差，风险来源完全不同——别把商品那套直接搬到期权上。

## 七、三个必须直说的真实陷阱

1. **IV crush 会咬人。** 日历价差是**净多头 Vega**（远月 Vega > 近月 Vega）。财报、议息、非农前你建仓，事件兑现后 IV 整体塌方（vol crush），远月腿大幅贬值，Theta 赚的那点不够 Vega 亏的。事件前做日历，等于用 Vega 去赌「事件后 IV 不塌」。

2. ** early assignment 与分红。** 卖出的是美式近月看涨，若标的跳涨并带分红，卖方可能被提前指派（损失时间价值、还要处理标的交割）。用看跌日历（put calendar）或避开临近除息日的近月，能缓解但不能根除。

3. **期限结构翻转。** contango 是你利润的第二来源，但它会塌。市场恐慌时近月 IV 飙升、曲线倒挂，远月腿相对变便宜，日历价差两头吃瘪。务必监控 IV 期限结构斜率，斜率转负就减仓或离场——别把「赚曲线钱」当成永远成立的免费午餐。

最后一句收口：日历价差是「卖快衰减、买慢衰减 + 赌波动率曲线不塌」的精细价差，它的 alpha 来自时间价值的衰减差和期限结构的 contango carry，但它的脾气是短期权——横盘吃肉、突破挨打。把它当需要主动管理的中性头寸，而不是「挂上去等收钱」的躺赢策略；后者在事件和恐慌里会被教育得很惨。

> 本文所有图表均由 Python（NumPy + Matplotlib）基于 Black-Scholes 定价与 GBM 蒙特卡洛模拟生成，数据为合成但结构贴近真实期权，仅用于方法演示，不构成任何交易建议。
