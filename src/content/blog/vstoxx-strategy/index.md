---
title: "VSTOXX 策略：欧洲恐慌指数的期限结构与交易"
publishDate: '2026-07-16'
description: "VSTOXX 是 EURO STOXX 50 的'欧洲 VIX'。它和 VIX 同构却更小众：期限结构 contango 为常态、曲线 carry 稳赚但危机翻转日爆亏，而 VIX-VSTOXX 的背离窗口还能做跨大西洋配对相对价值。本文从零构造 VSTOXX 期限结构、跑通 carry 与配对策略，附完整 Python 与六类真实陷阱。"
tags:
  - 量化交易
  - VSTOXX
  - 波动率风险溢价
  - 期限结构
  - 期货曲线
  - 波动率套利
  - 跨市场配对
language: Chinese
difficulty: advanced
---

美国有 VIX，欧洲有 **VSTOXX**——EURO STOXX 50 指数的 30 日隐含波动率，常被叫做「欧洲恐慌指数」。机制和 VIX 完全同构：都是用一篮子期权隐含波动反推的「恐慌温度计」。但因为它更小众、流动性更薄、和美国市场高度联动又偶尔背离，反而藏着几条美国玩家容易忽略的交易线索。

本文用一套自洽的合成模型把 VSTOXX 造出来，讲清三件事：

1. VSTOXX 期货期限结构如何在 **contango（升水）与 backwardation（贴水）** 间切换；
2. 「卖近月买远月」的**曲线 carry** 怎么稳赚、又在哪天一把吐回去；
3. **VIX 与 VSTOXX 的背离窗口**——两洲恐慌的联动里，藏着跨大西洋配对相对价值。

## 一、VSTOXX 期限结构：两副面孔

VSTOXX 本身衡量 EURO STOXX 50 未来 30 天隐含波动，而 VSTOXX 期货在不同期限上定价「未来那个时点的预期 VSTOXX」。把各期限期货 IV 按到期排开，就是期限结构。

我们用和 VIX 篇同构的模型生成 VSTOXX 近端路径 $V0_t$（带危机跳的均值回复），再叠加上期限衰减的 VRP 项：

$$\text{IV}(M,t) = LV + (V0_t - LV)\,e^{-M/\tau} + \text{VRP}_0\,e^{-M/\tau_{\text{vrp}}}$$

- **Contango（升水）**：近月 IV < 远月 IV，曲线向上。这是**常态**——欧洲投资者也为近端「保险」付更多，且近端波动均值回复到长端中枢。
- **Backwardation（贴水）**：近月 IV > 远月 IV，曲线倒挂。**只在危机时出现**——近端 V0 暴冲，远月相对平静。

![VSTOXX 期货期限结构：两种面孔](/images/vstoxx-strategy/vstoxx_term_structure.png)

左图平静期 contango：近月 IV 约 14、远月约 21，斜率 +0.18；右图危机 snapshot：近端 V0 飙到 80+，近月 IV 跳到 57、远月仍 23，曲线整体倒挂（斜率 −0.30）。同一条曲线，两种相反形态。

## 二、Python：合成 VSTOXX 期限结构

下面这段代码与生成图表的是同一套逻辑。VSTOXX 的 V0 均值低于长端中枢（所以平时是 contango），危机时向上跳。

```python
import numpy as np

# ---- 1) 生成 VSTOXX 近端波动（带危机跳的 AR(1)，去均值）----
def gen_vstoxx(T=2520, theta=7.0, xi=2.2,
               p_jump=0.012, jump_mean=24.0, jump_sd=9.0, seed=31415):
    rng = np.random.default_rng(seed)
    s = np.empty(T); s[0] = 0.0
    for t in range(1, T):
        s[t] = 0.97 * s[t-1] + xi * rng.normal()          # 强均值回复
        if rng.random() < p_jump:
            s[t] += max(0.0, rng.normal(jump_mean, jump_sd))  # 波动率危机跳
    return np.clip(theta + (s - s.mean()), 9.0, 150.0)

# ---- 2) VSTOXX 期货期限结构 ----
LV = 22.0                         # 长端中枢
V0 = gen_vstoxx(seed=31415)
def IV_at(M, t):
    return LV + (V0[t] - LV) * np.exp(-M / 45.0) + 3.2 * np.exp(-M / 32.0)

mats = np.array([21, 42, 63, 84, 105, 126])
IV = {M: np.array([IV_at(M, t) for t in range(len(V0))]) for M in mats}
slope = (IV[42] - IV[21]) / IV[21]
contango = slope > 0
print("contango 占比=%.1f%%  backwardation 占比=%.1f%%"
      % (100*contango.mean(), 100*(1-contango.mean())))
```

跑出：**contango 占比 83%，backwardation 占比 17%**——contango 是常态，backwardation 是稀有的危机片段，与真实 VSTOXX 经验一致。

![VSTOXX 期限结构斜率：绿色 contango 为常态、红色 backwardation 为危机片段](/images/vstoxx-strategy/vstoxx_slope.png)

## 三、曲线 carry：卖近买远，稳赚但有牙

既然平时 contango（近低远高），经典 carry 就是**卖出近月、买入远月**。盈利来自 roll-down：近月合约临近到期，价格沿向上的曲线往下滑，空头获利；远月基本不动，微亏。只在 contango 持有，backwardation 立刻平仓。

用「次日期限下滑一个档」重新估值，收益写为：

$$\text{PnL}_t = \big[\text{IV}_{21}(t)-\text{IV}_{20}(t+1)\big] + \big[\text{IV}_{125}(t+1)-\text{IV}_{126}(t)\big]$$

第一项（短近月）在 contango 里稳定为正，第二项（长远月）轻微为负，净额即 carry。

```python
# ---- 3) 做多曲线 carry: sell near / buy far，仅 contango 持有 ----
daily = np.zeros(len(V0))
for t in range(1, len(V0) - 1):
    if contango[t]:
        short_leg = IV_at(21, t) - IV_at(20, t + 1)   # 卖近月：随到期下滑赚价差
        long_leg  = IV_at(125, t + 1) - IV_at(126, t)  # 买远月：微损
        daily[t]  = short_leg + long_leg
eq = np.cumsum(daily)
peak = np.maximum.accumulate(eq)
mdd = (eq - peak).min()
print("carry 累计=%.1f vol点  最大回撤=%.1f" % (eq[-1], mdd))
```

跑出：**累计 +61.5 vol 点，最大回撤 −42.4**。

![VSTOXX 曲线 carry：sell near / buy far，仅 contango 持有](/images/vstoxx-strategy/vstoxx_carry_equity.png)

曲线大部分时间斜向上爬——contango carry 的「免费午餐」。但那次深坑（回撤 −42.4）是**危机翻转日**：近端 V0 一天从平静跳到 80，近月 IV 跟着从十几跳到五十多，而你正空头近月——近端跳涨，空头单日巨亏。carry 的甜，全靠「危机别让我在场」；而危机恰恰在你最放松（长期 contango）时降临。

## 四、跨大西洋配对：VIX 与 VSTOXX 的背离窗口

VIX（美国）和 VSTOXX（欧洲）高度联动——它们都度量「发达市场股票恐慌」，且美股与欧股相关系数很高。但二者**并非完美同步**：有时美国先恐慌、欧洲滞后，有时欧洲独自动荡（如欧债危机、英国脱欧）。这种背离就是配对相对价值的来源。

我们生成一条与 VSTOXX 同步但带独立跳的 VIX 路径，算两者价差 $z = (VIX - VSTOXX - \bar\mu)/\sigma$，当 $|z|>1$ 时做均值回复：z 高（VIX 相对 VSTOXX 过贵）→ 空 VIX / 多 VSTOXX；反向同理。

```python
# ---- 4) VIX vs VSTOXX 跨市场配对相对价值 ----
rng = np.random.default_rng(999)
def gen_vix(T=2520, theta=16.0, seed=7):
    v = np.empty(T); v[0] = theta
    for t in range(1, T):
        common = 0.6 * (V0[t] - V0[t-1])          # 跟随 VSTOXX 的共同冲击
        v[t] = v[t-1] + 0.02*(theta - v[t-1]) + common + 1.4*rng.normal()
        if rng.random() < 0.006:
            v[t] += max(0.0, rng.normal(18.0, 7.0))
    return np.clip(v, 9.0, 150.0)
VIX = gen_vix(seed=7)
spread = VIX - V0
z = (spread - np.nanmean(spread)) / np.nanstd(spread)

pair = np.zeros(len(VIX))
for t in range(1, len(VIX)):
    if z[t-1] > 1.0:
        pair[t] = -(VIX[t]-VIX[t-1]) + (V0[t]-V0[t-1])   # 空VIX/多VSTOXX
    elif z[t-1] < -1.0:
        pair[t] = (VIX[t]-VIX[t-1]) - (V0[t]-V0[t-1])    # 多VIX/空VSTOXX
peq = np.cumsum(pair)
print("VIX-VSTOXX 配对 Sharpe=%.2f  相关性=%.3f"
      % (np.sqrt(252)*np.diff(peq).mean()/np.diff(peq).std(), np.corrcoef(VIX, V0)[0,1]))
```

跑出：**配对 Sharpe ≈ 1.82，两序列相关性仅 0.47**（因为我们特意注入了独立跳）。背离窗口给了稳定的均值回复收益。

![VIX vs VSTOXX：两洲恐慌联动，背离窗口可做配对相对价值](/images/vstoxx-strategy/vix_vstoxx_spread.png)

图上 VIX（紫）与 VSTOXX（蓝）大趋势同步，但 z-分数（灰）经常突破 ±1 阈值——每一次突破都是一次配对入场信号。注意：真实世界里二者相关性远高于 0.47（常达 0.8+），配对空间更小，但欧债、英国脱欧、瑞郎黑天鹅等**欧洲特有事件**仍会制造可观背离。

## 五、真实陷阱（别把欧洲当美国的简单复制）

**1. VSTOXX 流动性远薄于 VIX。** VIX 期货是全球最活跃的波动率衍生品，VSTOXX 期货日均成交量小一个数量级。本文合成模型没计买卖价差与冲击成本，**实盘 carry 会比 +61.5 vol 点薄得多**，翻转日还可能平不掉。

**2. Contango 常态 ≠ 无风险。** 本文 contango 占 83%，但那 17% 的 backwardation 片段（及翻转当天）制造了 −42.4 的回撤。「卖 VSTOXX」长期正收益，是用「偶尔巨亏」换的——它是**卖保险**，不是套利。

**3. 跨市场配对有「共同因子」风险。** VIX 与 VSTOXX 共享全球股票风险，真正可交易的是**差值**里的 idiosyncratic 部分。当两洲同步暴跌（如 2020 新冠），背离会先扩大再一起跳，配对两头都亏——背离策略在系统性危机里失效。

**4. 时区与定价错位。** 欧洲市场比美国早收盘，VSTOXX 在美盘时段靠 VIX 拉动定价，套利窗口可能是伪信号（只是时区导致的瞬时错位），实盘无法无风险捕获。

**5. 期限结构会翻转，别假设永远 contango。** 用「斜率>0 才持有」是对翻转的保护，但翻转当天你仍在场（本例当天吃跳空），因为真实近端是**跳**上去的，平不掉在最高点。

**6. 这是合成模型，不是实盘数据。** 真实 VSTOXX 要拉 VSTOXX 期货实盘序列（或按各期限期权隐含波动反推），并做股息/持有成本/到期效应调整。本文用自洽模型只为**可复现地演示机制**——机制是真的，数字量级需以实盘校准。

## 六、小结

- VSTOXX 是「欧洲 VIX」，与 VIX 同构却更小众：期限结构平时 contango（本文 83%）、危机时 backwardation（17%）；
- **曲线 carry**（卖近买远）在 contango 稳赚（累计 +61.5 vol 点），但翻转日近端跳涨让空头单日巨亏（回撤 −42.4）——它是卖保险，不是无风险；
- **VIX-VSTOXX 背离窗口**可做跨大西洋配对相对价值（本文配对 Sharpe ≈ 1.82），但真实相关性更高、系统性危机里失效；
- 实盘记住六条陷阱：欧洲流动性薄、contango 非无风险、共同因子风险、时区伪信号、曲线会翻转、本文是合成演示需实盘校准。

> 附：本文所有图表与数值均来自上方可运行 Python（带危机跳 AR(1) 的 VSTOXX 路径 + 三组件期限结构 + 滚动下滑 carry + VIX 配对），参数与结果一致，可直接复现。
