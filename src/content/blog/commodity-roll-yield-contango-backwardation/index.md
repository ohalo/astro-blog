---
title: "展期收益与期货结构：Contango 和 Backwardation 的真实成本"
description: "拆解商品期货展期收益（Roll Yield）的来源，辨析 Contango 与 Backwardation 两种期限结构如何通过展期侵蚀或增厚收益。附 Python 计算与历史数据回测代码。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 商品期货
  - 期限结构
categories: ["量化交易"]
slug: "commodity-roll-yield-contango-backwardation"
image: "/images/commodity-roll-yield-contango-backwardation/contango_backwardation.png"
---

如果你曾经"长期持有"一只原油或黄金的期货 ETF，却发现自己明明看对了方向却没赚到钱——问题很可能不在你的判断，而在**展期收益（Roll Yield）**。这是商品投资里最容易被忽视、却最致命的一条隐性成本。

![Contango（升水）与 Backwardation（贴水）两种期货曲线结构](/images/commodity-roll-yield-contango-backwardation/contango_backwardation.png)

这篇文章从期限结构出发，用 Python 把展期收益算清楚，并告诉你为什么"升水结构里做多期货等于慢性自杀"。

## 一、什么是期货期限结构？

同一标的、不同交割月份的合约，价格连起来就是一条**期限结构曲线（Term Structure）**。它只有两种基本形态：

- **Contango（升水/正向市场）**：远期合约价格 **高于** 近月。曲线向上倾斜。
- **Backwardation（贴水/反向市场）**：近月价格 **高于** 远期。曲线向下倾斜。

决定形态的是**持有成本理论（Cost of Carry）**：远月价格 ≈ 近月价格 + 仓储 + 资金利息 − 便利收益（Convenience Yield）。当持有成本高（如原油库存爆满、资金贵），曲线就升水；当现货紧俏、持有实物有隐性红利（如停产、地缘风险），曲线就贴水。

## 二、展期收益从哪来？

期货合约有到期日。如果你想"长期持有"某一商品，必须在近月到期前把它**展期（Roll）**到下一个月份。这个动作会产生收益或亏损，就是**展期收益**。

假设你在 Contango 结构里**做多**：

```
近月价 = 80，远月价 = 82（升水 2 元）
你平掉近月（80 卖出），以 82 买入远月
→ 每次展期你都"高买"，白白损失 2 元
```

这就是**负展期收益（Negative Roll Yield）**。反过来，在 Backwardation 里做多，近月（82）换到远月（80），每次"低买"，赚到**正展期收益**。

一个反直觉的真相：**你持有商品期货的总收益 = 现货价格变动 + 展期收益**。展期这一项，在长期里常常比方向本身更决定成败。

## 三、Python 计算展期收益

下面用模拟的近月/远月价格序列，直接计算多空展期收益。

```python
import numpy as np
import pandas as pd

def compute_roll_yield(near_prices, far_prices, position="long"):
    """
    计算期货展期收益。
    near_prices, far_prices : 近月、远月价格序列（等长）
    position : "long" 做多 / "short" 做空
    """
    near = np.asarray(near_prices, dtype=float)
    far = np.asarray(far_prices, dtype=float)
    # 每次展期：平掉近月，换入远月
    # 做多时：卖出近月(near)，买入远月(far)；收益 = 近月 - 远月
    roll = near - far if position == "long" else far - near
    cum_roll = np.cumsum(roll)
    return roll, cum_roll

# 构造一个典型 Contango 场景：远月持续高于近月
np.random.seed(2026)
n = 120
near = 80 + np.cumsum(np.random.normal(0.05, 0.8, n))   # 近月缓慢上行
spread = 1.5 + 0.5 * np.sin(np.arange(n) / 10)          # 稳定的升水价差
far = near + spread

roll, cum = compute_roll_yield(near, far, position="long")
print(f"Contango 做多 累计展期收益: {cum[-1]:.2f} (负=亏损)")
print(f"平均每次展期: {roll.mean():.3f}")
```

运行后你会看到：在稳定的升水下，做多的累计展期收益是**负值**，且随时间线性恶化。这正是 USO（原油 ETF）在 2020 年之前长期跑输油价的根因——它每个月都在"高买低卖"地展期。

## 四、把现货变动和展期加总

真正衡量一只商品期货持仓的损益，必须把两件事叠加：

```python
def total_return(near_prices, far_prices, position="long"):
    """总收益 = 现货(近月)变动 + 展期收益"""
    near = np.asarray(near_prices, dtype=float)
    roll, _ = compute_roll_yield(near, far_prices, position)
    spot_change = np.diff(near)              # 近月自身变动
    # 用近端变动近似"现货方向收益"
    total = np.concatenate([[0], spot_change]) + roll
    return total.cumsum()

total_long = total_return(near, far, "long")
total_short = total_return(near, far, "short")

print(f"做多总收益(含展期): {total_long[-1]:.2f}")
print(f"做空总收益(含展期): {total_short[-1]:.2f}")
```

![不同期限结构下展期收益的路径分化](/images/commodity-roll-yield-contango-backwardation/roll_yield_paths.png)

注意一个关键现象：即便近月价格（现货方向）是上涨的，做多的**总收益仍可能为负**——因为负展期收益把方向收益吃光了。这就是为什么"看多原油却亏钱"在升水结构里是常态。

## 五、Contango 与 Backwardation 的真实成本

我们用一个对比实验，量化两种结构对长期持有者的"税收"。

```python
def structure_cost_demo(n=252, base=80, vol=0.015, seed=1):
    """对比 Contango / Backwardation 下做多的年度化展期损耗"""
    rng = np.random.default_rng(seed)
    near = base * np.cumprod(1 + rng.normal(0.0003, vol, n))
    
    # Contango：远月持续升水 1.5%
    far_cont = near * 1.015
    # Backwardation：远月持续贴水 1.5%
    far_back = near * 0.985
    
    _, cum_cont = compute_roll_yield(near, far_cont, "long")
    _, cum_back = compute_roll_yield(near, far_back, "long")
    
    ann_cont = cum_cont[-1] / base / (n / 252)
    ann_back = cum_back[-1] / base / (n / 252)
    return ann_cont, ann_back

ann_cont, ann_back = structure_cost_demo()
print(f"Contango 做多 年化展期成本: {ann_cont:.2%}")
print(f"Backwardation 做多 年化展期收益: {ann_back:.2%}")
```

输出会显示：1.5% 的稳定升水，折算到全年会吃掉可观的百分比收益；而同样幅度的贴水，则每年白送你一笔"展期红利"。历史上，**原油、天然气长期处于 Contango**，而**贵金属、部分农产品常处于 Backwardation**——这也是为什么商品指数（如标普 GSCI）长期表现分化的深层原因。

![展期收益对结构利差的敏感性分析](/images/commodity-roll-yield-contango-backwardation/roll_yield_sensitivity.png)

## 六、实务中的三个坑

### 1. 升水幅度会突变

升水不是常数。库存周期、交割逼仓、央行政策都会在几周内翻转结构。2020 年 4 月 WTI 近月一度跌到 **−37 美元**，远月仍是正价——极端 Contango 让展期成本瞬间爆炸。任何基于"历史平均升水"的静态假设都可能在拐点失效。

### 2. 展期时点有讲究

不是所有合约都平滑过渡。临近交割时流动性骤降、价差跳变，聪明的做法是**提前展期**或**跨多个月份分散展期**，而不是挤在最后一天。

```python
def roll_schedule_diversified(near, far, window=5):
    """
    分散展期：不是一次性平仓，而是 window 天内均匀换月
    返回每天的实际持仓调整
    """
    adj = np.zeros_like(near, dtype=float)
    step = 1.0 / window
    for i in range(window):
        idx = int(i * len(near) / window)
        adj[idx] -= step * (far[idx] - near[idx])   # 每天平摊展期损益
    return adj.cumsum()
```

### 3. ETF 的隐形损耗

很多投资者以为买商品 ETF 就等于"买商品"。实际上，绝大多数商品 ETF 是用期货滚动复制的，**展期成本直接体现在净值里**。选 ETF 时，必须看它的展期结构：在 Contango 严重的品种上，主动管理型（优化展期、跨月套利）往往长期跑赢被动型。

## 七、如何利用展期而非被它收割？

聪明的商品策略不回避展期，而是**交易展期本身**：

- ** roll 收益策略**：系统性做多 Backwardation 品种、做空 Contango 品种，赚取结构红利（所谓"展期增强"）。
- **日历价差（Calendar Spread）**：直接押注近远月价差收敛/扩大，而非方向。
- **库存-价差套利**：用库存周期预测结构翻转，在 Contango 转 Backwardation 的拐点建仓。

```python
def roll_enhanced_signal(spread_series, threshold=0.005):
    """
    展期增强信号：价差(远-近)/近 为负(贴水)做多，为正(升水)做空
    """
    signal = np.where(spread_series < -threshold, 1,
              np.where(spread_series > threshold, -1, 0))
    return signal

# 示例：根据远-近价差生成多空信号
spread_pct = (far - near) / near
sig = roll_enhanced_signal(spread_pct, threshold=0.005)
print("信号样例 (1=多, -1=空, 0=空仓):", sig[:10])
```

## 八、结语

展期收益是商品投资里那条"看不见的腿"。教科书教你算方向、算波动率，却很少强调：在 Contango 结构里长期做多期货，你是在和一条缓慢但确定的下行曲线对赌——你赢的概率不低，但每赢一次都被抽走一层"过路费"。

理解 Contango 与 Backwardation，不是让你远离商品，而是让你在进场前先回答一个问题：**我赚的这笔钱，是来自方向判断，还是先被展期扣掉了一道？** 把这道账算清，商品期货才从"赌方向"变成"算结构"。

*本文代码均在 Python 3 环境下可直接运行，价格数据为模拟生成，仅用于教学演示，不构成任何投资建议。*
