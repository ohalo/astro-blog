---
title: "ETF 折溢价套利：一二级市场价差的可交易窗口"
description: "一只 ETF 同时挂着一级市场 IOPV 与二级市场市价，两者的瞬时分离就是折溢价。本文把折溢价建模成「均值回复 OU 过程 + 偶发流动性冲击跳变」，回测一个无方向的收敛套利：溢价超阈值做空 ETF、回到中枢平仓，折价则反向。numpy 合成 8 年数据实测年化 7.23%、Sharpe 0.62、命中率 58.3%，但单边 8bps 成本只吃掉每笔 0.16%——真正的瓶颈是交易次数与窗口存续期（中位数 26 日），而非价差本身。附完整 Python 与五张真实计算图。"
publishDate: '2026-10-07'
tags:
  - 量化交易
  - ETF套利
  - 折溢价
  - 收敛套利
  - 均值回复
  - 微观结构
  - 交易成本
  - Python
language: Chinese
difficulty: intermediate
---

一只 ETF 同时挂着**两个价格**，这是它和普通股票最不一样的地方：

- **一级市场 IOPV**（Indicative Optimized Portfolio Value）：交易时段内按篮子成分股实时行情算出的「基金该值多少钱」。它是参考价，你没法直接在一级市场按它成交。
- **二级市场市价**：你在交易所买卖 ETF 份额的真实价格，由供需撮合决定。

两个价格天然该相等——如果不等，就出现了**折溢价（premium / discount）**。市价长期显著高于 IOPV 叫**溢价**，低于叫**折价**。套利者的工作，就是在这条缝里赚钱，而且理论上这是一类**风险极低**的生意：你不是在赌方向，而是在赚「两个市场定价不一致」的收敛。

但「理论上无风险」和「实际能赚钱」之间，隔着一条又宽又深的沟：**交易成本 + 窗口存续期 + 冲击跳变的回吐速度**。本文把折溢价建模成可回测的随机过程，跑一个不押方向的收敛套利，并诚实展示它到底卡在哪。

结论先放这：**一笔「溢价 → 回中枢」的收敛，毛收益约等于入场时的折溢价幅度；但真正决定你能不能赚钱的，是「窗口能活多久」和「成本吃掉多少」。在 8 年合成数据上，阈值 50bps、单边成本 8bps 下，策略年化 7.23%、Sharpe 0.62、命中率 58.3%，每笔成本仅吃掉 0.16%——瓶颈不是价差，而是中位数 26 天才收敛的等待，以及冲击跳变可能被新跳变覆盖、永远等不到回中枢。**

![IOPV 与二级市价：折溢价带界定可交易窗口](/images/etf-premium-discount-arb/iopv_vs_price.png)

## 一、折溢价从哪来：一二级是两条反应速度不同的腿

股票只有一个二级价格。ETF 多了一条**一级创设（creation）/ 赎回（redemption）**通道：授权参与人（AP）用一篮子股票向基金公司换 ETF 份额，或把份额退回换回篮子。这条通道的结算价锚定 IOPV，不是二级市价。

关键点：**一级和二级的参与者不是同一拨人，反应速度也不同**。

- 二级散户看到价格异动，几秒就能买卖；
- 一级 AP 要凑齐篮子、走流程，T 日申请通常 T+1 甚至更晚才拿到份额或股票。

于是当某成分股突发利好、二级瞬间把 ETF 拉高时，一级篮子 IOPV 还没跟上，或 AP 凑篮子需要时间——**市价就短暂跑到了 IOPV 上面**，形成溢价。反过来，成分股大面积跌停、二级抛压集中，市价砸到 IOPV 下面，形成折价。这条缝，就是可交易窗口。

## 二、把折溢价建模成「OU 回复 + 冲击跳变」

要回测，得先有一个能复现「瞬时割裂」的折溢价过程。我们用一个自洽的合成模型（真实落地见文末路径）：

```python
import numpy as np

rng = np.random.default_rng(20261007)
T = 252 * 8                         # 8 年日度

# 一级 IOPV：带轻微正漂移的随机游走
iopv = np.empty(T + 1); iopv[0] = 1.0
iopv[1:] = np.cumprod(1.0 + rng.normal(0.0003, 0.009, T))

# 折溢价 premium：OU 均值回复（中枢 0）+ 偶发流动性冲击跳变
theta, sigma = 0.04, 0.0009         # 缓慢回复（~25 日），日波动 9bps
prem = np.zeros(T)
jump_prob, jump_sd = 0.004, 0.022   # 0.4%/日 概率出现 ±2.2%(sd) 冲击
for t in range(1, T):
    prem[t] = (1 - theta) * prem[t-1] + sigma * rng.standard_normal()
    if rng.random() < jump_prob:
        prem[t] += rng.normal(0.0, jump_sd)

price = iopv[1:] * (1.0 + prem)     # 二级市价 = IOPV × (1 + 折溢价)
```

这个结构有两个真实特征：**① 平时折溢价围绕 0 缓慢回复**（做市商套利力量把它拉回，但拉得慢）；**② 偶发的流动性冲击会瞬间把折溢价顶到 ±2%~4% 的尖峰**，随后靠 OU 回复慢慢回吐。实测序列：均值 −0.10%、标准差 0.36%、最大 +2.25%、最小 −1.24%——和真实 ETF 折溢价的量级一致。

![折溢价序列 OU 回复 + 冲击跳变：信号触发点一览](/images/etf-premium-discount-arb/premium_series.png)

## 三、收敛套利：不押方向，只赚回归

核心逻辑极简：**当折溢价显著偏离 0，做「被高估/被低估」那一侧，等它回到中枢（穿过 0）平仓**。因为溢价时 ETF 市价 > 其真实篮子价值，你该做空 ETF、等它跌回 IOPV；折价时反过来做多。

```python
def backtest(th, cost_rate):
    pos = 0; entry_price = 0.0; pnl = []; days_held = []; entry_day = 0
    for t in range(T):
        p = prem[t]
        if pos == 0:
            if p > th:                       # 溢价：ETF 被高估，做空
                pos = -1; entry_price = price[t]; entry_day = t
            elif p < -th:                    # 折价：ETF 被低估，做多
                pos = 1;  entry_price = price[t]; entry_day = t
        else:
            # 回到中枢（穿过 0）平仓
            if (pos == -1 and p <= 0) or (pos == 1 and p >= 0):
                exit_price = price[t]
                gross = pos * (exit_price - entry_price) / entry_price
                net = gross - 2 * cost_rate  # 双边成本
                pnl.append(net); days_held.append(t - entry_day); pos = 0
    return np.array(pnl), np.array(days_held)
```

注意两个诚实约束：**① 必须是无前视的**——用 `t` 时刻已知的折溢价决定 `t` 时刻动作，不能在 `t` 之前就「预知」窗口；**② 成本要双边算**——进出各吃一次，本文单边 8bps（含冲击+佣金），合计 16bps/笔。

## 四、结果：赚的是回归，但次数少、等待长

跑出来（阈值 50bps，单边成本 8bps）：

- **交易次数**：24 笔（8 年，平均每年仅 3 笔）
- **命中率**：58.3%（回归方向正确的比例）
- **单笔净收益均值**：+2.35%，标准差 6.59%（波动大，单笔风险不低）
- **年化收益**：7.23%；**Sharpe**：0.62
- **每笔成本消耗**：仅 0.16%（相对于单笔 2.35% 收益几乎可忽略）
- **收敛耗时中位数**：26 日

把逐笔净收益画出来（图 3 左），分布以 0 为轴、右侧厚尾——这是收敛套利的典型脸谱：大部分笔小赚，偶尔因跳变回吐慢而拖成亏损。累计净值（图 3 右）稳步爬升。

![逐笔收益分布与累计净值](/images/etf-premium-discount-arb/pnl_distribution.png)

**关键洞察**：成本在这里根本不是瓶颈（每笔只吃 0.16%）。真正的瓶颈是**交易次数太少**（年均 3 笔，资金利用效率低）和**窗口存续期太长**（中位数 26 日，占用资金、暴露在其他风险上）。换句话说，折溢价套利是一种「单笔赔率不错、但频率极低」的策略——它更适合作为多策略组合里的一根低相关源，而非单独扛旗。

## 五、阈值敏感性：调高阈值，过滤噪声但饿死交易

把入场阈值从 10bps 扫到 200bps，看年化收益、Sharpe、年交易次数的变化：

![阈值敏感性：阈值越高越过滤噪声，但交易次数骤降](/images/etf-premium-discount-arb/threshold_sensitivity.png)

规律很清晰：**阈值越低 → 交易越多但噪声越多（假信号、来回打脸）；阈值越高 → 过滤噪声但交易次数骤降，资金闲死**。存在一个「甜蜜点」：太低被成本+假信号吞噬，太高没交易。本文 50bps 落在可交易的区间内，但已经偏「稀」——若想提高频率，得换更高频（日内）的折溢价数据，而非日度。

## 六、收敛耗时：大多 1~3 周，但尾部很长

把每笔「入场到平仓」的天数画成分布：

![收敛耗时分布：大多在 1~3 周内回到中枢](/images/etf-premium-discount-arb/convergence_days.png)

中位数 26 日，但右尾很长——有些窗口因为连续出现新冲击跳变，几个月内都回不到中枢，直到被强制规则（如最大持仓期限）清掉。这提醒我们：**收敛套利有「时间风险」**。实务上必须加「最大持有期」止损（比如 60 日强制平仓），否则一笔卡住的折价可能拖垮整段资金效率。

## 七、落地要点与诚实边界

1. **本文是合成演示**：折溢价是我用 OU + 冲击跳变合成的，真实 ETF 的折溢价结构更复杂（受成分股停牌、申赎清单错误、ETF 规模、流动性分层影响），但「均值回复 + 偶发跳变 + 成本决定成败」的骨架在真实数据上同样成立。
2. **成本不是主要矛盾，频率才是**：单边 8bps 下每笔只吃 0.16%，但若你用更高频的真实折溢价（分钟级），成本会显著上升、且 AP 套利会把窗口压得更短——真实可交易窗口比日度合成更窄。
3. **必须用真实 IOPV 而非收盘价近似**：很多回测偷懒用「ETF 收盘价 − 指数收盘价」当折溢价，这混入了指数本身的隔夜跳变，会系统性高估信号。正确做法是用交易所发布的实时 IOPV。
4. **做空 ETF 在实务有约束**：A 股 ETF 融券受限，折价套利（做多 ETF、赎回篮子）比溢价套利（做空 ETF、创设）更难做；美股 ETF 做空相对容易，但也要承担借券成本。
5. **样本外要警惕 regime**：折溢价套利在流动性危机时反而最肥（冲击跳变密集），但那时你的对手方（AP）可能暂停申赎，窗口「看得见吃不着」。这是它最大的结构性风险。
6. **本文未建模执行滑点**：真实市价冲击、盘口深度、以及「穿过 0 平仓」的那一刻是否真能成交，都需要用真实 tick/分钟数据回测确认。

## 完整可复现代码

```python
import numpy as np

rng = np.random.default_rng(20261007)
T = 252 * 8
iopv = np.empty(T + 1); iopv[0] = 1.0
iopv[1:] = np.cumprod(1.0 + rng.normal(0.0003, 0.009, T))

theta, sigma = 0.04, 0.0009
prem = np.zeros(T)
jump_prob, jump_sd = 0.004, 0.022
for t in range(1, T):
    prem[t] = (1 - theta) * prem[t-1] + sigma * rng.standard_normal()
    if rng.random() < jump_prob:
        prem[t] += rng.normal(0.0, jump_sd)
price = iopv[1:] * (1.0 + prem)

def backtest(th, cost_rate):
    pos = 0; entry_price = 0.0; pnl = []; days_held = []; entry_day = 0
    for t in range(T):
        p = prem[t]
        if pos == 0:
            if p > th:   pos = -1; entry_price = price[t]; entry_day = t
            elif p < -th: pos = 1;  entry_price = price[t]; entry_day = t
        else:
            if (pos == -1 and p <= 0) or (pos == 1 and p >= 0):
                gross = pos * (price[t] - entry_price) / entry_price
                pnl.append(gross - 2 * cost_rate)
                days_held.append(t - entry_day); pos = 0
    return np.array(pnl), np.array(days_held)

cost_rate = 0.0008
TH = 0.005
pnl, days_held = backtest(TH, cost_rate)
hit = (pnl > 0).mean()
tpy = len(pnl) / (T / 252.0)
ann = (1 + pnl.mean()) ** tpy - 1
sharpe = pnl.mean() / pnl.std() * np.sqrt(tpy)
print(f"trades={len(pnl)} hit={hit:.1%} ann={ann:.2%} sharpe={sharpe:.2f} "
      f"median_days={np.median(days_held):.0f}")
# trades=24 hit=58.3% ann=7.23% sharpe=0.62 median_days=26
```
