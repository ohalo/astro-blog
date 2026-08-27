---
title: "CPPI 组合保险：固定比例机制与 1987 年的教训"
description: "深入剖析 CPPI（固定比例投资组合保险）的数学机制、动态再平衡逻辑，以及它在 1987 年黑色星期一中的系统性放大效应。附完整 Python 实现与回测代码。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 风险管理
  - 组合保险
categories: ["量化交易"]
slug: "portfolio-insurance-cppi"
image: "/images/portfolio-insurance-cppi/cppi_mechanism.png"
---

组合保险（Portfolio Insurance）曾经是华尔街最优雅的想法之一：在不牺牲上涨空间的前提下，为投资组合锁定一个下跌底线。CPPI（Constant Proportion Portfolio Insurance，固定比例投资组合保险）作为其中最易实现的变体，在 1980 年代风靡一时。然而 1987 年 10 月 19 日，这个本应"保护"投资者的机制，反而成了踩踏的推手。

![CPPI 机制示意：底线、缓冲垫与风险乘数](/images/portfolio-insurance-cppi/cppi_mechanism.png)

这篇文章带你从数学公式出发，用 Python 完整实现 CPPI，并复现它在极端行情下"越跌越卖"的死亡螺旋。

## 一、CPPI 的核心公式

CPPI 的思想极其简洁：把资产分成"安全资产"（通常是零息债券）和"风险资产"（股票、指数）两部分。核心变量有三个：

- **Floor（底线）**：未来必须保住的最低价值，随时间按无风险利率增长。
- **Cushion（缓冲垫）**：当前组合价值超出底线的部分，`Cushion = Value − Floor`。
- **Multiplier（风险乘数 m）**：放大系数，通常取 2~5。

风险资产的配置金额由下面这个式子决定：

```
Exposure = m × Cushion = m × (Value − Floor)
```

剩余部分投入安全资产：

```
Safe = Value − Exposure
```

乘数 m 决定了"激进程度"。m 越大，缓冲垫被放大得越狠，上涨时赚得多，但下跌时缓冲垫蒸发得也越快。

### 为什么是"固定比例"？

注意式子里的乘数是**常数 m**，不随时间变化——这就是 Constant Proportion 的由来。与之相对的是 OBPI（期权基础组合保险），它用真实期权动态复制保险。CPPI 的优势在于不需要期权市场，纯靠买卖股票和债券就能实现类似效果。

## 二、Python 完整实现

下面是一份可直接运行的 CPPI 回测代码，包含参数校验、路径模拟和可视化。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cppi_rebalance(value, floor, m, risky_return):
    """
    单步 CPPI 再平衡。
    
    参数
    ----
    value : float        当前组合总价值
    floor  : float        当前底线价值（随时间增长）
    m      : float        风险乘数
    risky_return : float  风险资产在本期收益率
    
    返回
    ----
    new_value, exposure   再平衡后的组合价值与风险敞口
    """
    cushion = value - floor
    # 防护：敞口不能为负，也不能超过总资产
    exposure = max(0.0, min(m * cushion, value))
    # 安全资产部分
    safe = value - exposure
    # 下一期：风险资产按 risky_return 增长，安全资产假设本步不增值（简化）
    new_value = exposure * (1 + risky_return) + safe
    return new_value, exposure


def simulate_cppi(n_steps, mu, sigma, m=3, floor_init=0.9, seed=42):
    """
    蒙特卡洛模拟 CPPI 路径。
    mu, sigma : 风险资产年化漂移与波动
    floor_init : 初始底线占初始价值比例
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    value = 1.0
    floor = floor_init
    floor_growth = np.exp(0.02 * dt)   # 底线按 2% 无风险利率增长
    values, floors, exposures = [value], [floor], [0.0]
    
    for _ in range(n_steps):
        # 生成日收益（几何布朗运动）
        z = rng.standard_normal()
        risky_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        value, exp = cppi_rebalance(value, floor, m, risky_ret)
        floor *= floor_growth
        values.append(value)
        floors.append(floor)
        exposures.append(exp)
    
    return pd.DataFrame({
        "value": values,
        "floor": floors,
        "exposure": exposures,
    })


# 运行一次模拟
df = simulate_cppi(n_steps=252, mu=0.08, sigma=0.20, m=3)
print(f"期末价值: {df['value'].iloc[-1]:.4f}")
print(f"最低价值: {df['value'].min():.4f}  (底线: {df['floor'].min():.4f})")
```

这段代码的关键是 `cppi_rebalance`：每一步先算出缓冲垫，乘以乘数得到风险敞口，剩下的放进安全资产。注意 `min(m * cushion, value)` 这个截断——当缓冲垫为负（击穿底线）时，敞口被压到 0，组合全部转为安全资产，此时保险"失效"，投资者已跌破保护线。

## 三、路径分化：牛市捡钱，暴跌裸奔

把多条路径画出来，CPPI 的"不对称"暴露无遗。

![不同市场环境下的 CPPI 组合价值路径分化](/images/portfolio-insurance-cppi/cppi_paths.png)

```python
def multi_path(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    for seed in range(8):
        df = simulate_cppi(252, mu=0.08, sigma=0.20, m=3, seed=seed)
        ax.plot(df["value"], lw=1, alpha=0.7)
        ax.plot(df["floor"], color="red", lw=0.6, alpha=0.3)
    ax.set_title("CPPI 多路径模拟 (m=3, μ=8%, σ=20%)")
    ax.set_xlabel("交易日"); ax.set_ylabel("组合价值")
    plt.show()

multi_path()
```

观察这 8 条路径你会发现两件事：

1. **大多数时候**组合价值贴着或高于底线运行，红色的底线像一张安全网。
2. 一旦某条路径出现连续大跌，价值会急速逼近底线，敞口被清零，之后只能在极低水位"裸奔"——因为已经没有缓冲垫可以放大了。

这就是 CPPI 的本质缺陷：**它在平静期给你超额收益，在崩溃期却让你失去翻本的本钱。**

## 四、1987 年的死亡螺旋

1987 年 10 月 19 日（黑色星期一），道琼斯指数单日暴跌 22.6%。当时市场上大量的组合保险策略（包括 CPPI 和基于期权的变体）在同一时刻做同一件事：**卖出股票、降低风险敞口**。

问题在于，CPPI 的卖出是**路径依赖且顺周期的**：

```
价格下跌 → 缓冲垫缩小 → 风险敞口 = m × 缓冲垫 被迫下降 → 卖出股票 → 价格进一步下跌
```

这不是个别机构的行为，而是无数采用类似机制的账户在同一规则下的**集体共振**。当足够多的资金同时执行"跌了就卖"，卖压本身成为下跌的原因，形成正反馈。

```python
def crash_spiral_demo():
    """
    演示顺周期卖出如何自我强化。
    假设市场有 N 个 CPPI 账户，每个在下跌时按比例减仓。
    """
    n_accounts = 1000
    m = 3
    # 初始组合价值均为 1，底线 0.9
    values = np.ones(n_accounts)
    floors = np.full(n_accounts, 0.9)
    
    price_index = [1.0]
    for step in range(10):
        # 本步市场冲击：由所有账户的卖出行为共同决定
        # 简化：卖出量越大，价格跌得越多
        total_sell = 0.0
        for i in range(n_accounts):
            cushion = values[i] - floors[i]
            target_exp = max(0, m * cushion)
            current_exp = values[i]  # 假设全仓风险资产
            sell = max(0, current_exp - target_exp)
            total_sell += sell
            values[i] = target_exp * 0.97 + (values[i] - target_exp)  # 留安全部分
        # 卖压转化为价格下跌
        price_drop = total_sell / n_accounts * 0.5
        new_price = price_index[-1] * (1 - price_drop)
        price_index.append(new_price)
        # 价格下跌进一步压低所有组合价值
        values *= new_price / price_index[-2]
    
    return price_index

spiral = crash_spiral_demo()
print("价格指数序列:", [f"{p:.3f}" for p in spiral])
```

这个极简模型省略了无数细节，但它抓住了核心：**当"动态对冲"成为市场的主流策略，对冲行为本身就是波动的来源。** 1987 年之后，监管与学界才真正意识到，组合保险在流动性枯竭时不是缓冲垫，而是加速器。

![1987 年崩盘后 CPPI 账户的恢复困境](/images/portfolio-insurance-cppi/cppi_crash_recovery.png)

## 五、实务中的三个致命细节

### 1. 乘数的双刃剑

乘数 m 是最容易被误用的参数。教科书常说 m < 1 / 最大可能回撤 才能保证不击穿底线。例如你假设股票单日最多跌 10%，那 m ≤ 10。但"历史最大回撤"在极端事件面前毫无意义——1987 年一天就跌了 22.6%，任何基于历史波动设定的 m 都会瞬间失效。

```python
def max_drawdown_to_floor(df):
    """计算是否击穿底线，以及最大回撤"""
    breached = (df["value"] < df["floor"]).any()
    peak = df["value"].cummax()
    mdd = (df["value"] / peak - 1).min()
    return breached, mdd

for m in [2, 3, 5, 8]:
    df = simulate_cppi(252, mu=0.08, sigma=0.20, m=m, seed=7)
    breached, mdd = max_drawdown_to_floor(df)
    print(f"m={m:>2}  击穿底线={breached}  最大回撤={mdd:.1%}")
```

### 2. 交易成本被忽略

上面的模拟都没有计入交易成本。CPPI 是高频再平衡策略——只要价格动，敞口就要调。在实际组合中，每天几次买卖的手续费、滑点、印花税会持续侵蚀缓冲垫，尤其在震荡市里，反复"高买低卖"的换手损耗可能吃掉全部保险价值。

### 3. 底线漂移与利率假设

底线按无风险利率增长，这个假设在加息/降息周期里会系统性偏离。若实际利率低于假设，底线增长慢，缓冲垫被人为放大，风险敞口虚高；反之则过早耗尽保护。

## 六、CPPI 还值得用吗？

答案是：**作为理解风险预算的工具很有价值，作为实盘核心策略需极度谨慎。**

现代资管中，CPPI 的思想被拆解后融入了更稳健的框架：

- **波动率目标（Vol Targeting）**：用波动率而非固定乘数调节敞口，对跳空更鲁棒。
- **风险预算（Risk Budgeting）**：直接控制组合 VaR，而非间接控制敞口。
- **期权保护**：用真实看跌期权锁定尾部，避免顺周期卖出的共振。

如果你要在自己的量化系统里用 CPPI，请至少加上三道防火墙：

1. **乘数上限** + **无交易带**（敞口偏离不足 X% 不动，降低换手）。
2. **流动性约束**：在成交量萎缩时主动降速减仓，而非机械执行。
3. **压力测试**：用 1987 年级别的单日跳空校准，而不是用历史波动率。

```python
def cppi_with_band(value, floor, m, risky_return, prev_exposure, band=0.15):
    """加入无交易带（band）的 CPPI：偏离不足 15% 不调仓"""
    cushion = value - floor
    target = max(0.0, min(m * cushion, value))
    if abs(target - prev_exposure) / max(prev_exposure, 1e-9) < band:
        exposure = prev_exposure
    else:
        exposure = target
    safe = value - exposure
    new_value = exposure * (1 + risky_return) + safe
    return new_value, exposure
```

## 七、结语

CPPI 用一个漂亮的数学式子，把一个朴素的风险预算思想工程化了：用确定的安全资产守住底线，用风险资产博取弹性。它的失败从不在公式本身，而在**群体行为的一致性**——当所有人都用同一个规则保护自己的时候，规则本身就变成了风险。

1987 年的教训不是"组合保险是错的"，而是"任何依赖流动性、且在危机中集体行动的防御机制，都必须为流动性枯竭预留冗余"。理解 CPPI，本质是在理解：你的风控假设里，哪一部分依赖于'别人不会同时做同样的事'。

*本文代码均在 Python 3 环境下可直接运行，数据为模拟生成，仅用于教学演示，不构成任何投资建议。*
