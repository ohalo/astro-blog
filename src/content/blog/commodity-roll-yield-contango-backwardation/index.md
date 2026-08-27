---
title: "展期收益与期货结构：Contango 和 Backwardation 的真实成本"
description: "为什么原油现货涨了你期货却亏钱？拆解展期收益（Roll Yield）：Contango 如何悄悄吃掉你的本金，Backwardation 又如何白送你收益。附完整 Python 曲线carry回测。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 期货
  - 商品
  - 展期收益
categories: ["量化交易"]
slug: "commodity-roll-yield-contango-backwardation"
image: "/images/commodity-roll-yield-contango-backwardation/contango_backwardation.png"
---

很多人以为"买原油期货 = 赌油价上涨"。只要你判断对方向，就能赚钱。但 2009—2020 年这十一年里，WTI 油价从 40 美元涨回 60 美元，追踪它的旗舰 ETF（USO）却几乎原地踏步甚至亏损。原因不在你的方向判断，而在**展期收益（Roll Yield）**——一个藏在期货期限结构里、每天都要扣你钱的隐性成本。

这篇文章从期限结构讲起，用 Python 把展期收益拆干净，并给出一个可运行的"曲线 carry"策略回测。

![期限结构：Contango 与 Backwardation 的两种形态](/images/commodity-roll-yield-contango-backwardation/contango_backwardation.png)

## 一、期限结构：远月比近月贵还是便宜？

任何有到期日的合约，都会形成一条**期限结构曲线（Term Structure）**：把不同到期月份的期货价格连起来。

- **Contango（升水）**：远月价格 > 近月价格。曲线向上倾斜。
- **Backwardation（贴水）**：近月价格 > 远月价格。曲线向下倾斜。

直觉上，Contango 意味着"市场愿意为更晚交割付更高价格"。这通常发生在**囤货有成本**的商品上：原油、天然气要仓储、要保险、要占用资金，远月包含了这些持有成本（cost of carry），所以更贵。

```python
import numpy as np
import pandas as pd

def term_structure(months, near_price, slope):
    """
    构造一条简单期限结构。
    slope > 0  -> Contango（远月更贵）
    slope < 0  -> Backwardation（远月更便宜）
    """
    return pd.Series(
        near_price * (1 + slope) ** np.arange(len(months)),
        index=months
    )

months = ["M1", "M2", "M3", "M6", "M12"]
contango   = term_structure(months, near_price=80, slope=0.025)   # 远月逐级更贵
backward  = term_structure(months, near_price=95, slope=-0.020)  # 远月逐级更便宜
print("Contango:\n", contango.round(2))
print("Backwardation:\n", backward.round(2))
```

## 二、展期收益：你每天在"卖低买高"还是"卖高买低"

期货不能永久持有。要维持头寸，你必须在近月合约到期前**卖出近月、买入远月**——这叫"展期（Roll）"。展期那一下的价格差，就是你这笔投资的**展期收益（或成本）**。

- 在 **Contango** 里，远月比近月贵。你卖出便宜的近月、买入昂贵的远月 → **每展期一次就亏一笔** → 展期收益为负。
- 在 **Backwardation** 里，远月比近月便宜。你卖出贵的近月、买入便宜的远月 → **每展期一次就赚一笔** → 展期收益为正。

这正是 2009—2020 年原油投资者的噩梦：WTI 长期处于 Contango，期货曲线每月都在"吸血"。

```python
def roll_yield(near, far):
    """单次展期的年化收益（近似）。
    卖出近月、买入远月，持有到远月变为近月这段时间的收益。"""
    return (near - far) / far   # 远月更贵(far>near)时为负

# 例：近月 80，下月 82（Contango）
print(f"Contango 展期收益: {roll_yield(80, 82):+.2%}")   # 负
# 例：近月 95，下月 93（Backwardation）
print(f"Backwardation 展期收益: {roll_yield(95, 93):+.2%}")  # 正
```

**关键认知**：展期收益和"油价涨跌"是两个独立变量。油价可以涨，但如果你处在深 Contango，展期成本可能比油价涨幅还大——结果就是你亏钱。

## 三、把总收益拆成三块：现货 + 展期 + 基差

一个持有期货的多头，长期总收益可以拆解为：

```
总收益 ≈ 现货价格变动  +  展期收益  +  基差收敛（到期前远月向现货靠拢）
```

多数时候，基差收敛在到期时归零，真正长期持续贡献（或拖累）的是**展期收益**。这就是商品指数长期跑输现货的核心机制。

![展期收益累计：原油(Contango) 把涨幅吃光，黄金(Backwardation) 白送收益](/images/commodity-roll-yield-contango-backwardation/roll_yield_paths.png)

下面用模拟数据复现这条曲线：现货缓慢上涨，但原油处于 Contango（年化展期 −8%），黄金处于 Backwardation（年化展期 +3%）。

```python
def futures_total_return(spot_daily, roll_annual, n_days):
    """给定现货日收益序列与年化展期收益，合成期货净值。"""
    roll_daily = (1 + roll_annual) ** (1 / 252) - 1
    daily = spot_daily + roll_daily
    return np.cumprod(1 + daily)

rng = np.random.default_rng(3)
spot_oil  = rng.normal(0.10/252, 0.016, 252*4)   # 现货年化 +10%，低波动
spot_gold = rng.normal(0.04/252, 0.010, 252*4)

oil  = futures_total_return(spot_oil,  -0.08, 252*4)
gold = futures_total_return(spot_gold, +0.03, 252*4)

print(f"4年后  原油现货隐含: {np.cumprod(1+spot_oil)[-1]:.3f}  "
      f"原油期货: {oil[-1]:.3f}   <- 被展期吃光")
print(f"4年后  黄金现货隐含: {np.cumprod(1+spot_gold)[-1]:.3f}  "
      f"黄金期货: {gold[-1]:.3f}  <- 展期额外加成")
```

## 四、Contango 为什么长期存在？持有成本与便利收益

Contango 不是"市场错了"，它有坚实的微观基础。期货定价的基本关系是：

```
F = S × e^((r + storage − convenience_yield) × T)
```

- `r`：无风险利率
- `storage`：仓储、保险、损耗成本
- `convenience_yield`（便利收益）：持有实物能立刻应对短缺的隐性价值

当 `r + storage > convenience_yield`（典型如原油过剩、库容紧张），远月更贵 → Contango。
当实物极度紧缺、谁手握库存谁说了算（典型如 2000 年代黄金、供给侧冲击中的金属），便利收益很高 → Backwardation。

所以**期限结构的形态，本质上是市场对"现在稀缺还是未来稀缺"的定价**。

## 五、一个可运行的曲线 Carry 策略

既然展期收益长期可正可负，最朴素的玩法就是：**只做 Backwardation 的品种（吃正展期），回避/做空 Contango 的品种（赚负展期收敛）**。下面用一段简化回测演示"曲线 carry"：每个调仓日，根据近月与远月的价差符号决定多空。

```python
def curve_carry_backtest(near_series, far_series, hold_days=21):
    """
    曲线 carry 简化回测：
      near < far (Contango)  -> 做空远月(等价于吃负展期收敛)
      near > far (Backwardation) -> 做多近月(吃正展期)
    这里用 far/near 的相对变化近似盈亏。
    """
    log_near = np.log(near_series)
    log_far  = np.log(far_series)
    # 每 hold_days 调仓一次，持有期间赚 far 向 near 收敛的钱
    pnl = []
    for start in range(0, len(near_series) - hold_days, hold_days):
        # 做多 near / 做空 far：赚 (near涨 - far涨)
        ret = (log_near[start+hold_days] - log_near[start]) \
            - (log_far[start+hold_days]  - log_far[start])
        pnl.append(ret)
    pnl = np.array(pnl)
    equity = np.cumprod(1 + pnl)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252 / hold_days) if pnl.std() > 0 else 0
    win = (pnl > 0).mean()
    return equity, sharpe, win

# 构造一段交替出现的期限结构（演示用）
T = 500
rng = np.random.default_rng(1)
spread = np.concatenate([
    -np.abs(rng.normal(0.01, 0.005, 250)),   # 前段 Backwardation（near>far, 负价差）
     np.abs(rng.normal(0.01, 0.005, 250)),    # 后段 Contango（near<far, 正价差）
])
near = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, T)))
far  = near * (1 + spread)
eq, sh, win = curve_carry_backtest(near, far)
print(f"曲线carry: 净值终值 {eq[-1]:.3f}  Sharpe {sh:.2f}  胜率 {win:.1%}")
```

回测结论不是"这个策略能暴富"，而是揭示一个事实：**期限结构的形态本身就是一个可被定价的因子**。商品指数长期跑输现货，恰恰是因为它被动地、无条件地承担了所有品种的展期成本——而你完全可以挑着吃。

![展期收益（年化）随持有年限的累计拖累/贡献：时间站在 Contango 的对立面](/images/commodity-roll-yield-contango-backwardation/roll_yield_sensitivity.png)

## 六、实务中最容易被忽视的三件事

### 1. ETF 的展期不是"无成本搬家"

以 USO 为代表的商品 ETF，每月固定把近月展期到远月。当它持有的是深 Contango 品种，这个动作每天都在侵蚀净值。2009—2020 年 WTI 多数时间 Contango，USO 的长期回报因此系统性落后于油价本身。买这类 ETF 前，先问一句：**它的底层处在 Contango 还是 Backwardation？**

### 2. 展期成本随持有时间指数增长

持有 1 年 Contango −8%，看起来不多；但持有 10 年，累计拖累是 `(1−0.08)^10 − 1 ≈ −56%`。**越长的持有期，结构成本越致命**。这也是为什么商品不适合"无脑长拿"。

### 3. 结构是动态的，会翻转

Contango 和 Backwardation 会随供需反转。2020 年 4 月 WTI 甚至出现**负油价**——近月崩到负值、远月仍为正，是极端 Contango。做曲线 carry 必须动态跟踪价差，而不是死记"原油永远 contango"。

## 七、结语

展期收益是期货世界里最被散户低估、却被专业商品交易员天天计量的东西。它告诉我们一个朴素的真理：**你持有的是什么结构，有时比你看对方向更重要。**

Contango 不是错误，它是持有成本的诚实体现；Backwardation 不是免费午餐，它是实物稀缺的定价。真正的 alpha，往往不在"油价涨不涨"的判断里，而在"我是不是在为一个糟糕的期限结构买单"的警惕里。

*本文代码均在 Python 3 环境下可直接运行，数据为模拟生成，仅用于教学演示，不构成任何投资建议。*
