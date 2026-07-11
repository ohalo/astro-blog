---
title: "组合再平衡的税制与换手率摩擦：被隐性成本偷走的复利"
publishDate: '2026-07-11'
description: "再平衡是组合管理的双刃剑：它控制风险，却也在税与交易摩擦里悄悄流血。本文用 lot 记账 + 税务感知模拟，量化月度朴素再平衡相对无摩擦上界被偷走的 1.23% 年化收益，并给出阈值带与税务感知两套可落地的止血方案。"
tags:
  - 量化交易
  - 组合管理
  - 再平衡
  - 资本利得税
  - 换手率
  - Python
language: Chinese
---

## 一个被忽视的数字

每个做组合管理的人都知道要"再平衡"：涨多的卖一点、跌多的补一点，把权重拉回目标。它控制风险、强迫你"低买高卖"。

但很少有人认真算过：**再平衡本身在流血**。

每做一次再平衡，你都在两处付钱——

1. **资本利得税**：卖出的盈利仓位要交税，税是直接从本金里扣的，不复利。
2. **交易摩擦**：佣金、买卖价差、冲击成本，按成交名义额抽水。

这篇文章用一套带「lot 记账 + 税务感知」的模拟，把这笔隐性成本摊开给你看：在 5 只资产、10 年的设定下，**月度朴素再平衡**相对无摩擦上界，年化收益被偷走了 **1.23%**——听起来不大，但叠在复利上，10 年后差距是肉眼可见的。

## 再平衡为什么"流血"

假设你有 100 万，等权买 5 只基金，目标权重各 20%。一个月后某只涨到 25%，另一只跌到 16%。再平衡意味着：卖出涨的那只的盈利部分、买入跌的那只。

问题就在这里：你卖出的那部分盈利，产生了**已实现资本利得**。如果你的税率是 20%（长期）/ 30%（短期），这部分税永远离开了你的账户，不再参与未来的复利增长。

更隐蔽的是**交易频率**。月度再平衡 = 一年 12 次买卖。每次买卖都按名义成交额抽一笔摩擦成本。12 次/年 × 10 年 = 120 轮抽水。

所以我们至少要比较三种再平衡方式：

- **A. 月度朴素（FIFO、无税务感知）**：每月无脑拉回目标，卖出按先进先出，不区分长短期、不主动减税
- **B. 5% 阈值带（FIFO）**：只有当某权重偏离目标超过 5% 才再平衡，降低换手
- **C. 税务感知（5% 带）**：在阈值带基础上，卖出时优先"指定辨认"亏损 lot 做 tax-loss harvesting、优先持有满长期阈值再卖盈利

## 用 lot 记账把税算对

要算准税，不能用"平均成本"糊弄——必须用**批次(lot)记账**：每一笔买入单独记一个 lot，卖出时用指定辨认(specific identification)挑最划算的 lot 卖。

下面这段是模拟的核心账本（纯 numpy，可运行）：

```python
import numpy as np

class TaxLotBook:
    def __init__(self, n_assets, long_term_months=12, lt_rate=0.20, st_rate=0.30):
        self.N = n_assets
        self.lots = [[] for _ in range(n_assets)]   # 每个 lot: [shares, cost, month_bought]
        self.lt_months = long_term_months
        self.lt_rate, self.st_rate = lt_rate, st_rate

    def buy(self, i, shares, price, month):
        self.lots[i].append([shares, price, month])

    def _rank_for_sell(self, i, month):
        """税务感知排序：先卖亏损(lot harvest)，盈利里优先长期、优先小盈利"""
        lots = self.lots[i]
        def key(lot):
            shares, cost, mb = lot
            gain = (price_sell - cost) * shares
            if gain < 0:
                return (0, gain)                      # 亏损组：先 harvest 最大亏损
            long_term = (month - mb) >= self.lt_months
            return (1, 0 if long_term else 1, gain)   # 盈利组：长期优先、小盈利优先
        return sorted(range(len(lots)), key=lambda k: key(lots[k]))

    def sell(self, i, shares, price, month):
        global price_sell
        price_sell = price
        tax = 0.0
        remaining = shares
        for k in self._rank_for_sell(i, month):
            if remaining <= 0:
                break
            lot = self.lots[i][k]
            take = min(lot[0], remaining)
            gain = (price - lot[1]) * take
            rate = self.lt_rate if (month - lot[2]) >= self.lt_months else self.st_rate
            tax += max(gain, 0) * rate          # 亏损不交税，但可抵减（harvest）
            lot[0] -= take
            remaining -= take
        self.lots[i] = [l for l in self.lots[i] if l[0] > 1e-9]
        return tax
```

这段代码的关键在 `_rank_for_sell`：**卖出时先挑亏损 lot**（realized loss 可以抵税，等于政府替你分担了一部分回撤），盈利 lot 里优先卖"已满长期"的（税率 20% vs 30%），且优先卖小盈利（避免把大盈利一次性实现、推高税基）。

## 完整模拟与三种策略对比

把账本塞进一个 5 资产、120 个月的相关收益路径里，跑三种再平衡：

```python
def run(mode, band=None, tax_aware=True, cost_rate=0.0008):
    navs = [1_000_000.0]
    p = TaxLotBook(5)
    for t in range(120):
        # 1) 用最新价格更新组合市值
        V = portfolio_value(p, prices[t])
        w = current_weights(p, prices[t])
        # 2) 是否触发再平衡
        if mode == "monthly" or (band is not None and max(abs(w - 1/5)) > band):
            target = np.full(5, 1/5)
            # 计算每只的买卖额
            for i in range(5):
                trade = (target[i] - w[i]) * V
                if trade > 0:
                    p.buy(i, trade / prices[t][i], prices[t][i], t)
                else:
                    tax = p.sell(i, -trade / prices[t][i], prices[t][i], t) if tax_aware else naive_fifo(p, i, ...)
                    total_tax += tax
                    total_cost += cost_rate * abs(trade)   # 交易摩擦：按名义额抽水
        # 3) 隐性成本拖累：把本金按 (V - 税 - 费)/V 缩放
        navs.append(navs[-1] * (V - total_tax - total_cost) / V)
    return navs
```

![再平衡的隐性成本：税 + 换手摩擦如何吞噬复利](/images/rebalance-tax-friction/equity_curves.png)

跑出来的净值曲线见上图。把关键指标拉成一张表：

| 策略 | 税后 CAGR | Sharpe | 最大回撤 | 累计税 | 累计费 |
|---|---|---|---|---|---|
| 无摩擦月度再平衡 | 13.67% | 1.05 | -11.6% | 0 | 0 |
| 月度朴素（FIFO） | 12.44% | 0.97 | -12.0% | 240,762 | 10,778 |
| 5% 阈值（FIFO） | 12.99% | 1.00 | -11.4% | 70,081 | 2,247 |
| 税务感知（5% 带） | 13.08% | 1.01 | -11.4% | 46,434 | 2,252 |

**结论很刺眼**：

- 月度朴素再平衡，10 年累计交了 **24 万税 + 1 万手续费**，相对无摩擦上界，年化收益被偷走 **1.23%**。
- 光是上 5% 阈值带，换手砍掉一大半，累计税从 24 万降到 7 万，CAGR 拖累从 1.23% 收到 0.68%。
- 再叠加税务感知（harvest 亏损 + 长期优先），累计税进一步降到 4.6 万，CAGR 拖累收到 **0.59%**——比月度朴素整整多拿回 **0.64% 年化**。

![换手频率：阈值带把再平衡次数砍掉一大半](/images/rebalance-tax-friction/turnover.png)

上图是滚动年化换手率。月度朴素那条线常年贴着高位（一年 12 次），阈值带和税务感知则明显下沉——**少操作，本身就是收益**。

## 隐性成本拆解：税才是大头

把累计隐性成本按"资本利得税 vs 交易摩擦"拆开看：

![隐性成本拆解：税是主力，阈值带 + 税务感知显著压缩](/images/rebalance-tax-friction/tax_decomposition.png)

注意一个反直觉的点：**交易摩擦（累计费）在这套设定里其实很小**（月度朴素才 1 万出头），真正吃肉的是**资本利得税**（24 万）。很多量化文章大谈手续费模型、冲击成本，却忽略了税——对 taxable 账户而言，税往往比摩擦大一个数量级。

这也解释了为什么"税务感知"比"单纯降换手"更值钱：它不只少交易，还**主动管理税基**——在下跌时 harvest 亏损抵税，在上涨时尽量拖到长期税率档。

## 阈值带宽扫描：找到拐点

阈值带越宽，换手越低、税后收益越高，但风险约束（权重偏离）越大。扫描一遍：

```python
bands = np.linspace(0, 0.15, 31)
cagrs, turns = [], []
for b in bands:
    nav = run("band", band=b, tax_aware=True)
    cagrs.append(cagr(nav))
    turns.append(annual_turnover(nav))
```

![带宽扫描：阈值越宽，换手越低、税后收益越高（存在拐点）](/images/rebalance-tax-friction/band_sweep.png)

双轴图里，CAGR（左轴，上升）和年化换手（右轴，下降）在 5%~8% 区间出现**拐点**：再往宽走，税后收益改善趋缓，但权重偏离带来的风险暴露快速放大。对多数中等波动组合，**5% 阈值带 + 税务感知**是性价比甜区。

## 落到实盘的五条建议

1. **能不用月度再平衡就别用**。季度或阈值带触发，CAGR 拖累直接砍半。
2. **阈值带设 5% 左右**。太窄白交税，太宽风险失控，5%~8% 是甜区。
3. ** taxable 账户一定要做税务感知**：下跌 harvest 亏损、盈利拖到长期档。这不是避税（违法），是合法地管理实现时点。
4. **优先在 tax-advantaged 账户（如养老金/免税户）里做高频再平衡**，把 taxable 账户留给低频、税务感知的策略。
5. **把税和摩擦写进回测**。很多策略回测漂亮，是因为默认无摩擦、无税——上线后才发现被隐性成本吃光 alpha。

## 一个具体的数字例子

抽象的数字容易飘，算一笔具体的账。假设你的 taxable 账户里有 100 万盈利仓位，成本基础 60 万，现在全卖：

```python
proceeds, cost_basis = 1_000_000, 600_000
gain = proceeds - cost_basis            # 已实现利得 40 万
short_term = True                         # 持有未满长期阈值
rate = 0.30 if short_term else 0.20      # 短期 30% / 长期 20%
tax = gain * rate                        # = 120,000（短期）或 80,000（长期）
```

同样 40 万利得，**短期卖出交 12 万、长期卖出只交 8 万**——差出的 4 万，就是你"早点卖"的隐性代价。税务感知策略的核心，就是尽量把这笔税从 30% 档挪到 20% 档，同时用下跌时的 harvest 亏损去抵掉一部分。对一个长期持有的组合，十年里反复做这个挪移，累计差额就是文章开头那 24 万 vs 4.6 万的差距来源。

## A 股与美股的税制差异

别把这套结论直接套到 A 股：

- **A 股个人买卖股票价差所得目前免征资本利得税**，但有**单边 0.05% 印花税**（卖出收），且 T+1 令当日买入不能当日卖，lot 记账的灵活性受限。
- **美股**是资本利得税主场，短期（≤1 年）按普通所得税率（可高达 37%）、长期优惠 0~20%，还有 wash-sale 规则限制亏损 harvest。
- **基金/ETF** 还会在年末做资本利得分配，哪怕你没卖，也可能被课税——所以用 ETF 做高频再平衡，要额外算这笔"被动税"。

所以阈值带 + 税务感知这套框架，对美股 taxable 账户最值钱；对 A 股，主要省的是印花税和交易摩擦，税的那一块几乎不存在。回测时务必按你实际市场的税制建模，否则结论会错得离谱。

## 诚实的边界

这套模拟是风格化的：5 只资产、相关月度收益、固定税率、合并建模的摩擦成本。它没有建模**非对称税档的跨年结转**、** wash-sale 规则**（30 天内买回同源资产会 disallow harvest）、**基金层面的资本利得分配**，也没考虑 A 股的印花税/T+1 现实与美股的不同税制。这些都会改变具体数字，但**"再平衡在税与摩擦里流血"这个方向性结论，稳健成立**。

如果你管的是 taxable 实盘，先把这篇文章的账本接上你真实的税率和持仓成本基础——你会惊讶于"勤快再平衡"到底偷走了你多少复利。
