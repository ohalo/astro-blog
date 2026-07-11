---
title: "VIX 期限结构与波动率风险溢价：跨期结构"
publishDate: '2026-07-12'
description: "VIX 期货不是一条线而是一条期限曲线：contango 与 backwardation 如何切换、波动率风险溢价(VRP)为何短端最肥、卖近买远的 carry 如何稳赚又在恐慌日爆亏。附完整 Python。"
tags:
  - 量化交易
  - VIX
  - 波动率风险溢价
  - 期限结构
  - 期货曲线
  - 波动率套利
language: Chinese
difficulty: advanced
---

很多人把 VIX 当成一个数字——「恐慌指数」今天 15 还是 30。但做波动率交易的人看的不是 VIX 一个点，而是**一整条曲线**：VIX 期货在每个到期期限上都有一个价格，把 1 个月、2 个月……6 个月的隐含波动连起来，就是**波动率期限结构（volatility term structure）**。这条曲线的形状（升水 contango 还是贴水 backwardation）、以及它在每个期限上相对「未来真实波动」的溢价（波动率风险溢价 VRP），才是波动率策略真正下注的地方。

本文用一套自洽的合成模型把这条曲线造出来，讲清三件事：曲线为何会在 contango / backwardation 之间翻转、VRP 为什么**短端最肥、随期限递减**、以及「卖近月买远月」的 carry 策略如何稳赚却在恐慌日一把吐回去。

## 一、期限结构的两副面孔

VIX 本身是衡量标普 500 未来 30 天隐含波动的前瞻指标。VIX 期货则在不同期限上定价「未来那个时点的预期 VIX」。把各期限的期货隐含波动（IV）按到期日排开：

- **Contango（升水）**：近月 IV < 远月 IV，曲线向上倾斜。这是**常态**——投资者愿意为近端的「保险」付更多钱，且近端波动会均值回复到长端中枢，所以远端更贵。经验上 VIX 期货约 70–80% 的时间处于 contango。
- **Backwardation（贴水）**：近月 IV > 远月 IV，曲线倒挂。这只在**恐慌**时出现——近端波动暴冲，远月还相对平静，曲线瞬间翻过来。

我们用一个极简但自洽的模型生成这条曲线。设近端瞬时波动 $V0_t$ 围绕一个低于长端中枢 $LV$ 的水平起伏（所以平时是 contango），危机时向上跳：

$$\text{IV}(M,t) = LV + (V0_t - LV)\,e^{-M/\tau} + \text{VRP}_0\,e^{-M/\tau_{\text{vrp}}}$$

第一项 $LV$ 是长端中枢；第二项让近端随瞬时波动偏离中枢、$e^{-M/\tau}$ 使其随期限衰减（危机主要砸近端）；第三项就是**跨期限的波动率风险溢价**，集中在短端。

![同一标的在 contango 与 backwardation 两种状态下的期限结构](/images/vix-term-structure/vix_term_structure.png)

左图是典型的平静期 contango：近月 IV 11.2、远月 19.2，斜率 +0.30；右图是恐慌 snapshot，近端 V0 飙到 83，近月 IV 跳到 59.8、远月仍只有 23.2，曲线**整体倒挂**（斜率 −0.27）。同一条曲线，两种完全相反的形态。

## 二、Python：合成 VIX 期限结构 + 危机跳

下面这段代码与生成图表的是同一套逻辑。要点是：近端波动用带**向上跳（Poisson 危机）**的 AR(1) 生成，所以 contango 是常态、backwardation 是偶发的危机片段——和真实市场一致。

```python
import numpy as np

# ---- 1) 生成近端波动冲击(含危机跳) ----
def gen_shock(T=2500, phi=0.97, sigma=2.2, p_jump=0.010, seed=21):
    rng = np.random.default_rng(seed)
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + sigma * rng.normal()
        if rng.random() < p_jump:
            s[t] += max(0.0, rng.normal(22.0, 8.0))   # 波动率危机跳
    return s - s.mean()                               # 去均值，稳定 contango 占比

# ---- 2) 合成 VIX 期限结构 ----
LV, V0_MEAN = 20.0, 5.0
def iv_curve(V0, M):
    return LV + (V0 - LV) * np.exp(-M / 42.0) + 3.0 * np.exp(-M / 30.0)

s = gen_shock()
V0 = np.clip(V0_MEAN + s, 3.0, 200.0)
mats = np.array([21, 42, 63, 84, 105, 126])
IV = {M: iv_curve(V0, M) for M in mats}

slope = (IV[42] - IV[21]) / IV[21]                  # 期限结构斜率
contango = slope > 0
print("contango 占比=%.1f%%, backwardation 占比=%.1f%%"
      % (100 * contango.mean(), 100 * (1 - contango.mean())))
```

在这组参数下跑出：**contango 占比 81.5%，backwardation 占比 18.5%**——contango 绝对是常态，backwardation 是稀有的危机片段，符合经验。

![期限结构斜率在 contango 与 backwardation 之间反复切换](/images/vix-term-structure/vix_term_slope.png)

## 三、波动率风险溢价（VRP）的期限结构：短端最肥

波动率风险溢价定义为**隐含波动减预期未来实现波动**：$\text{VRP}(M)= \text{IV}(M)-\mathbb E[\text{RV}(M)]$。在我们这个（DGP 隐含预期实现波动随期限向中枢回复）的模型里，VRP 干净地等于第三项的衰减：

$$\text{VRP}(M) = \text{VRP}_0\,e^{-M/\tau_{\text{vrp}}}$$

也就是说，**VRP 是期限结构的函数，短端最大、随期限递减**。这背后的直觉很直：投资者为「近期保护性保险」付的恐慌溢价最贵，越往远月，恐慌被时间稀释，溢价越薄。

![VRP 期限结构：短端溢价最大、随期限递减](/images/vix-term-structure/vix_vrp_term.png)

把各期限的平均 VRP 画出来（1M→6M）：

```
VRP(1M→6M): [1.49, 0.74, 0.37, 0.18, 0.09, 0.04] %
```

从近月的约 1.5% 一路衰减到远月的 0.04%——**短端 VRP 是整条曲线上最肥的一块肉**。这也是为什么「做空 VRP」的策略几乎都集中在前端（卖近月 VIX 期货 / 卖跨式），因为只有短端这块溢价够厚、值得下嘴。

## 四、做多曲线 carry：卖近买远，稳赚但有牙

既然曲线平时是 contango（近低远高），一个经典 carry 就是：**卖出近月、买入远月**。盈利来自「滚动下滑（roll-down）」——随着近月合约临近到期，它的价格沿向上的曲线往下滑，空头因此获利；远月基本不动，微亏。只在 contango 时持有，backwardation 立刻平仓。

用「次日期限下滑一个档」给头寸重新估值，收益可写为：

$$\text{PnL}_t = \big[\text{IV}_{21}(t)-\text{IV}_{20}(t+1)\big] + \big[\text{IV}_{125}(t+1)-\text{IV}_{126}(t)\big]$$

第一项（短近月）在 contango 里稳定为正，第二项（长远月）轻微为负，净额即 carry。

```python
# ---- 3) VRP 期限结构 ----
E_RV = {M: LV + (V0 - LV) * np.exp(-M / 42.0) for M in mats}
VRP = {M: IV[M] - E_RV[M] for M in mats}          # = 3.0·e^{-M/30}，随期限递减
print("VRP(1M→6M):", ["%.2f" % VRP[M].mean() for M in mats])

# ---- 4) 做多曲线 carry: 卖近买远，仅 contango 持有 ----
def IV_at(M, t):
    return LV + (V0[t] - LV) * np.exp(-M / 42.0) + 3.0 * np.exp(-M / 30.0)
daily = np.zeros(len(s))
for t in range(1, len(s) - 1):
    if contango[t]:
        short_leg = IV_at(21, t) - IV_at(20, t + 1)   # 短近月：随到期下滑赚价差
        long_leg  = IV_at(125, t + 1) - IV_at(126, t) # 长远月：微损
        daily[t]  = short_leg + long_leg
eq = np.cumsum(daily)
mdd = eq.min() - np.maximum.accumulate(eq).min()
print("carry 累计=%.1f vol点, 最大回撤=%.1f" % (eq[-1], mdd))
```

跑出：**累计 +123.6 vol 点，最大回撤 −60.2**。

![做多曲线 carry：contango 稳赚，恐慌翻转日吃掉利润](/images/vix-term-structure/vix_strategy.png)

曲线大部分时间斜向上爬——这就是 contango carry 的「免费午餐」。但那次深坑（回撤 −60.2）是什么？是**危机翻转日**：近端 V0 一天从平静跳到 83，近月 IV 跟着从十几跳到六十，而你正空头近月——近端跳涨，空头当天就亏掉几十个 vol 点。carry 的甜，全靠「危机别让我在场」；而危机恰恰在你最放松（长期 contango）时降临。

## 五、真实陷阱（别把 contango 当提款机）

**1. VIX 不是近月期货。** VIX 现货是 30 天前向 IV，VIX 期货是「未来那个时点到期的 VIX 期货」，二者期限不同、不要混用。算期限结构斜率要用同口径的期货 IV。

**2. Contango 常态 ≠ 无风险。** 本文 contango 占 81.5%，但那 18.5% 的 backwardation 片段（以及翻转当天）足以吞掉大量累计收益（回撤 −60.2）。「卖 VIX」类策略的长期正收益，是用「偶尔巨亏」换来的——它是**卖保险**，不是套利。

**3. VRP 是风险溢价，不是免费钱。** VRP 为正，是因为投资者愿意为下行保护付费；做空 VRP = 卖保险，危机时赔。短端 VRP 最肥（1.49%），也最危险——它赌的就是「近期别出事」。

**4. 期限结构会翻转，别假设永远 contango。** 用「斜率>0 才持有」已经是对翻转的保护，但翻转当天你仍在场（本例当天就吃了跳空），因为真实市场里近端是**跳**上去的，你平不掉在最高点。

**5. 远月流动性与基差。** 实盘里远月 VIX 期货合约流动性远差于近月，买卖价差大、滚动摩擦高；本文合成模型没计这部分成本，实盘 carry 会更薄。

**6. 这是合成模型，不是实盘数据。** 真实 VIX 期限结构要拉 VIX 期货实盘序列（或按各期限期权隐含波动反推），并做股息/持有成本/到期效应调整。本文用一套自洽模型只为**可复现地演示机制**——机制是真的，数字量级需以实盘校准。

## 六、小结

- VIX 期货是一条**期限曲线**而非一个点：平时 contango（近低远高，本文占 81.5%），危机时 backwardation（倒挂，占 18.5%）；
- **VRP 有期限结构**，且短端最肥（1M≈1.49% → 6M≈0.04%），所以「做空 VRP」几乎都集中在前端；
- **卖近买远的 carry** 在 contango 里稳赚（本文累计 +123.6 vol 点），但翻转日近端跳涨让空头单日巨亏（回撤 −60.2）——它是卖保险，不是无风险；
- 实盘记住六条陷阱：别混 VIX 与近月期货、别把 contango 当提款机、VRP 是风险溢价、曲线会翻转、远月流动性差、本文是合成演示需实盘校准。

> 附：本文所有图表与数值均来自上方可运行代码（带危机跳的 AR(1) 近端波动 + 三组件期限结构 + 滚动下滑 carry），参数与结果一致，可直接复现。
