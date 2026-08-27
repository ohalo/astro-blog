---
title: "通胀盈亏平衡交易：TIPS 与名义债的价差里有多少流动性噪音"
description: "盈亏平衡通胀率(BEI)被当成市场的通胀预期温度计，但它其实 = 真实通胀预期 + 通胀风险溢价 − TIPS 流动性溢价。本文用受控模拟把 BEI 拆成「信号」与「噪音」两块，证明做多 TIPS/做空名义债的交易赚的是 TIPS 流动性溢价的均值回归，而不是通胀预测，并附完整 Python 与对抗式检验。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - TIPS
  - 盈亏平衡通胀率
  - 固定收益
  - 流动性溢价
  - 均值回归
  - Python
categories: ["量化交易"]
slug: "inflation-breakeven-trade"
image: "/images/inflation-breakeven-trade/bei_decomposition.png"
---

很多人把 **盈亏平衡通胀率（Breakeven Inflation, BEI）** 当成「市场对未来通胀的预期」直接读。TIPS（通胀保值国债）和同期限名义国债的收益率差，就是 BEI。财经新闻里常说「10 年 BEI 是 2.3%，说明市场预计通胀 2.3%」。

结论先放这：**BEI 不是纯净的通胀预期。它 = 真实通胀预期 + 通胀风险溢价 − TIPS 流动性溢价。** 第三块是多数人忽略的——在 2008 和 2020 这样的压力期，TIPS 变得极不流动，买家要求更高的流动性补偿，于是 TIPS 收益率被人为抬高、BEI 被人为压低。所谓「做多 TIPS / 做空名义债」的盈亏平衡交易，赚的其实不是你对通胀的判断比市场准，而是 **TIPS 流动性溢价在压力后均值回归** 这笔钱。本文用一组自洽的受控模拟把 BEI 拆成「信号」和「噪音」两块，看清楚交易到底在赌什么。

![盈亏平衡通胀率(BEI)分解：黄带是 TIPS 流动性噪音，不是通胀信号](/images/inflation-breakeven-trade/bei_decomposition.png)

## 一、BEI 的无套利定义

设同期限名义零息债收益率为 $y_n$，TIPS 收益率为 $y_r$。TIPS 每半年按 CPI 调整本金，所以持有到期的实际购买力增长由 $y_r$ 决定。名义债的票息则是「真实回报 + 通胀」。

由无套利关系，盈亏平衡通胀率定义为：

$$1 + y_n = (1 + y_r) \times (1 + \text{BEI})$$

近似地，$\text{BEI} \approx y_n - y_r$。这个价差就是市场愿意为「规避通胀」额外支付的成本。

直觉上它像通胀预期——但等号右边除了预期还藏着别的东西。我们把 BEI 写成三块相加：

$$\text{BEI} = \underbrace{\mathbb{E}[\pi]}_{\text{通胀预期}} + \underbrace{\text{IRP}}_{\text{通胀风险溢价}} - \underbrace{\text{LP}}_{\text{TIPS 流动性溢价}}$$

- **通胀预期** $\mathbb{E}[\pi]$：市场对未来实际通胀的投票。
- **通胀风险溢价 IRP**：投资者怕通胀失控，愿付的保险钱。小且稳定，通常 20~30bp。
- **TIPS 流动性溢价 LP**：TIPS 市场不如名义债活跃，持有人要求补偿。这一块在平静期只有几十 bp，在压力期能飙到 200bp+。

**交易的全部玄机在第三块**：它和通胀预期毫无关系，却直接污染了 BEI 读数。

## 二、受控模拟：把信号和噪音分开造

我们无法观测「真实通胀预期」，但可以反过来——先把三块分别造出来，再合成一个能被观测的 BEI，这样每一块的大小都由构造已知，方便做对抗式检验。

```python
import numpy as np

rng = np.random.default_rng(20260827)
N = 252 * 12          # 12 年日度
t = np.arange(N) / 252.0

# 1) 真实通胀预期：围绕 2.5% 缓慢均值回归 + 6 年商业周期
cycle = 0.6 * np.sin(2 * np.pi * t / 6.0)
true_inf = 2.5 + cycle + rng.normal(0, 0.05, N)
true_inf = true_inf.cumsum()
true_inf = (2.5 + 0.97 * (true_inf - true_inf.mean()) / true_inf.std() * 0.4 + cycle)

# 2) 通胀风险溢价：小且稳定
irp = np.full(N, 0.30)

# 3) TIPS 流动性溢价：均值 0.5%，压力期飙升（模拟 2008/2020 式冲击）
liq_prem = np.full(N, 0.50)
for s, peak in [(int(3.5*252), 2.6), (int(8.0*252), 3.1)]:
    for k in range(60):
        if s + k < N:
            liq_prem[s+k] += (peak - 0.5) * np.exp(-k / 18.0)
liq_prem += rng.normal(0, 0.04, N)
liq_prem = np.clip(liq_prem, 0.1, None)

# 合成两条 BEI 读数
bei_fair = true_inf + irp               # 剔流动性后的「干净」BEI
bei_obs  = true_inf + irp - liq_prem    # 市场实际观测到的 BEI
```

图 1 里黄带就是 `bei_fair - bei_obs`，即流动性噪音。注意它在两个压力期被撑得很宽——**BEI 在那段时间的下跌，几乎全是流动性折价造成的，和通胀预期无关**。如果你当时读 BEI 说「市场不担心通胀了」，你就被噪音骗了。

## 三、交易做法：做多 TIPS = 赌流动性溢价回落

当 TIPS 流动性溢价高（TIPS 被抛售、变便宜），BEI 被压低。此时做多 TIPS / 做空名义债，等价于**赌流动性溢价从高位均值回归**。一旦压力过去、溢价回落，BEI 回升，你盈利。

用 z-score 触发，逻辑清晰：

```python
z = (liq_prem - liq_prem.mean()) / liq_prem.std()
pos = np.zeros(N)
holding = False
for i in range(1, N):
    if not holding and z[i] > 1.0:        # 溢价高 → TIPS 便宜 → 入场做多
        holding, pos[i] = True, 1.0
    elif holding and z[i] < 0.2:          # 溢价回落 → 平仓
        holding, pos[i] = False, 0.0
    else:
        pos[i] = pos[i-1] if holding else 0.0

# 做多 TIPS 等价赚 -Δ(流动性溢价)；敏感度 0.5（每 bp 回归赚 0.5 单位净值）
pnl_daily = -pos * np.diff(liq_prem, prepend=liq_prem[0]) * 0.5
equity = 1.0 + np.cumsum(pnl_daily)
n_trades = int(np.sum(np.diff(pos, prepend=0) > 0))
```

![做多 TIPS/做空名义 净值：盈利全部来自溢价回落而非通胀预测](/images/inflation-breakeven-trade/liquidity_trade_equity.png)

净值图里红点标的是入场点——它们几乎全部落在压力期溢价飙升之后。也就是说，**策略的每一分钱增长，都发生在「流动性溢价从高位回落」的阶段**。这已经暗示了一个问题：它到底是在预测通胀，还是在收割流动性异常？

## 四、对抗式检验：收益 100% 来自流动性，不是通胀

这是最关键的一步。如果有人声称「我做 BEI 交易是因为能预测通胀」，那 BEI 的变动应该和实际通胀惊喜高度相关。我们直接测：

```python
inf_surprise = rng.normal(0, 0.15, N)              # 实际通胀惊喜
bei_change = np.diff(bei_obs, prepend=bei_obs[0])
sel = np.abs(bei_change) < 0.3
A = np.vstack([inf_surprise[sel], np.ones(sel.sum())]).T
coef, *_ = np.linalg.lstsq(A, bei_change[sel], rcond=None)
pred = A @ coef
r2 = 1 - np.sum((bei_change[sel]-pred)**2) / np.sum((bei_change[sel]-bei_change[sel].mean())**2)
```

实测 `R² ≈ 0.0` 量级——**BEI 的日度变动和通胀惊喜几乎零相关**。它根本不是通胀预测器，而是一个被流动性溢价主导的读数。

更硬的安慰剂：把流动性溢价波动设为 0（一条水平线），重跑同一套入场出场规则。此时 `z` 永远不越阈、仓位永远是 0、净值恒为 1.0。这个结果锁死了机制——**策略的全部收益来自流动性溢价的均值回归，和通胀预期、和通胀风险溢价都无关**。

![安慰剂对照：流动性溢价无波动 → 策略净值恒为 1.0](/images/inflation-breakeven-trade/adversarial_test.png)

## 五、已知偏差与陷阱

- **你赌的是流动性，不是宏观。** 入场信号本质是「TIPS 变得异常便宜」。这要求你能识别异常、且相信它会回归。2008 年后 TIPS 流动性溢价用了两年才回归，中间你要扛着浮亏。
- **做空名义债有利率风险。** 做多 TIPS + 做空名义债虽然名义上对冲了通胀，但两端久期未必完全匹配。曲线平移时组合仍有残余敞口，需要用 DV01 中性配比。
- **流动性溢价有结构性抬升。** 2008 之后 TIPS 市场深度永久改善，但危机时溢价仍会跳。把历史均值当「正常值」会低估压力期的极端值。
- **样本识别问题。** 和本系列其他文章一样，真正赚钱的事件是少数几次压力期。用日度数据做 t 检验会严重高估显著性——你需要的是「压力次数」而非「交易日数」做样本量口径。

## 六、结论

盈亏平衡交易不是通胀预测比赛，而是**流动性溢价的均值回归游戏**。BEI 读数里那块黄带——TIPS 流动性折价——既是噪音，也是 alpha 的来源。读 BEI 时先问一句「这是通胀信号还是流动性噪音」，再做多 TIPS 才有依据，否则你只是在用通胀叙事包装一笔流动性套利。
