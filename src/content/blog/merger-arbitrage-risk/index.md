---
title: "并购套利风险：用价差与完成概率给 deal spread 定价"
publishDate: '2026-07-15'
description: "并购套利赚的是『收购报价与目标现价之差』，但它本质是给交易失败概率定价。本文用 deal spread、隐含完成概率、双态收益和净年化收益，把这笔风险溢价算到每股、算到年化，并给出 Python 全流程。"
tags:
  - 事件驱动
  - 并购套利
  - 风险定价
  - Python
language: Chinese
---

![并购套利：报价与目标价的价差就是风险定价](/images/merger-arbitrage-risk/cover.jpg)

并购套利（merger arbitrage，也叫 risk arbitrage）是事件驱动策略里最「像生意」的一种：一笔收购宣布后，目标公司股价通常会从 40 跳到 47，但离收购方报的 50 还差 3 块。这 3 块就是 **deal spread（并购价差）**。

很多人以为「买入目标、等收购完成，白捡这 3 块」——大错。这 3 块不是免费午餐，而是市场对你说：**我有概率这笔交易会黄，黄了你不仅吃不到 3 块，还可能倒亏 7 块（股价跌回 40）**。所以并购套利本质上是在**给交易失败概率定价**。本文把它拆成可算的模型，并附完整 Python。

## 一、价差不是利润，是风险补偿

收购宣布后，目标价 P_tar 与报价 P_bid 之间天然存在价差：

```
spread = P_bid − P_tar
```

如果 spread = 0，说明市场 100% 相信交易会完成；spread 越大，市场越担心会黄。套利者的收益不是 spread 全拿，而是**带条件地拿**：

- 交易完成（概率 p）：你以 P_tar 买入，最终按 P_bid 换股/现金，赚 spread
- 交易失败（概率 1−p）：目标价跌回 P_pre（公告前价）附近，你亏 (P_tar − P_pre)

所以价差本身就是「失败概率 × 失败损失」的资本化。理解这一点，才不会在 spread 很大时兴奋地冲进去——那往往意味着市场在定价一个你不掌握的坏消息。

![价差越窄 → 市场认为完成概率越高](/images/merger-arbitrage-risk/implied_prob.png)

上图蓝线是「隐含完成概率」随价差的变化：价差从 0.5 拉到 12，隐含概率从接近 100% 跌到 40% 出头。橙虚线是毛价差收益率（spread / P_tar），它随价差单调上升。两者反向 —— 价差越大、账面收益越高，但你要承担的概率风险也同步放大。套利不是「选收益最高的」，而是「选风险调整后最值的」。

## 二、隐含完成概率：从价差反推市场怎么想

假设市场无套利，且失败时目标价完全跌回 P_pre（先忽略分手费），则：

```
期望收益 = p · spread + (1 − p) · (P_pre − P_tar) = 0
```

解出隐含完成概率 p*：

```
p* = (P_tar − P_pre) / (spread + P_tar − P_pre)
```

代入 P_pre=40、P_bid=50、spread=3（即 P_tar=47）：

```
p* = (47 − 40) / (3 + 47 − 40) = 7 / 10 = 0.70
```

意思是：当前价差定价了 **30% 的交易失败概率**。如果你的研究判断真实失败概率只有 15%，那这就是一笔正向期望的赌注。

```python
import numpy as np

def implied_prob(P_pre, P_bid, spread):
    P_tar = P_bid - spread
    p_star = (P_tar - P_pre) / (spread + P_tar - P_pre)
    return P_tar, p_star

P_pre, P_bid = 40.0, 50.0
for s in [0.5, 3.0, 6.0, 10.0]:
    P_tar, p = implied_prob(P_pre, P_bid, s)
    print(f"spread={s:4.1f}  目标价={P_tar:5.1f}  隐含完成概率={p*100:5.1f}%")
```

跑出来：spread=0.5 → 隐含概率 95.2%；spread=3 → 70.0%；spread=10 → 47.4%。价差翻倍再多，市场对完成的信心掉得很快。

## 三、双态收益：把每股盈亏画清楚

给定 spread、P_pre、P_tar，还要算失败时到底亏多少。现实里目标价很少完全跌回 P_pre，监管否决后常停在 P_pre 之上（因为有别的买家或残余价值），且**分手费（break fee）**会补偿一部分。设分手费等价于每股 1.5 元：

```python
def payoff_table(spread, P_pre=40.0, fee_recover=1.5):
    P_tar = 50.0 - spread
    downside = P_tar - P_pre            # 全回撤损失
    loss_if_break = max(downside - fee_recover, 1.0)
    return P_tar, loss_if_break

spread = 3.0
P_tar, loss = payoff_table(spread)
print(f"目标价={P_tar}, 失败每股亏={loss:.1f}")
```

![并购套利双态收益：期望每股 +2.15 元](/images/merger-arbitrage-risk/outcome_payoff.png)

图上左边「完成」柱赚 +3.0（=spread），右边「失败」柱亏 −5.5（被分手费从 7.0 软化为 5.5）。在 p=0.90 完成 / 0.10 失败时：

```
期望每股 = 0.90×3.0 + 0.10×(−5.5) = 2.70 − 0.55 = +2.15 元
```

这笔交易期望为正，但风险不对称：小概率亏大钱。所以并购套利从不是「稳赚」，而是「大概率小赚、小概率大亏」——它的风险管理核心，是**控制单笔仓位，让任何一笔失败都不至于伤筋动骨**。

## 四、算到年化：把每股期望变成可比较的收益率

事件驱动策略不能只看每股绝对收益，要年化才能和别的策略比。设持有期 T 天（从宣布到完成/终止），投入成本≈P_tar：

```python
def annualized_return(spread, P_pre, fee_recover, p_complete, T_days):
    P_tar = 50.0 - spread
    downside = P_tar - P_pre
    loss_if_break = max(downside - fee_recover, 1.0)
    exp_per_share = p_complete * spread + (1 - p_complete) * (-loss_if_break)
    gross_ret = exp_per_share / P_tar
    ann = (1 + gross_ret) ** (365.0 / T_days) - 1
    return exp_per_share, gross_ret, ann

cases = [
    ("低价差/高概率", 3.0, 0.90, 60),
    ("中价差/中概率", 6.0, 0.75, 90),
    ("高价差/低概率", 10.0, 0.55, 120),
]
for name, s, p, T in cases:
    eps, gr, ann = annualized_return(s, 40.0, 1.5, p, T)
    print(f"{name}: 期望每股={eps:+.2f}  毛收益={gr*100:+.1f}%  年化={ann*100:+.1f}%")
```

注意一个反直觉的结果：**价差越大、年化收益率往往越高**（因为分母 P_tar 没涨多少，而 spread 直接进分子，且持有期更长摊薄）。这正是诱惑所在——但那是用「更高的失败概率」换来的。真要做组合，必须叠加夏普、最大回撤、相关性，而不是挑年化最高的单笔。

## 五、净收益还要扣三道成本

上面都是毛收益。实盘里要扣：

1. **融资成本**：并购套利占用资金数周甚至数月，按融资利率 r_f 计。净收益 ≈ 毛收益 − r_f × (T/365) × 仓位成本。
2. **交易摩擦与做空对冲**：若你同时卖空收购方做 market-neutral（对冲大盘 beta），做空的借券费、分红派息、滑点都要算。
3. **机会成本与多交易并发**：一笔钱不能同时押两笔。组合层面要按「占用资金 × 持有期」做资本配置，而不是按笔数。

```python
def net_return(spread, P_pre, fee_recover, p_complete, T_days,
               financing=0.05, trade_cost=0.001):
    eps, gr, ann = annualized_return(spread, P_pre, fee_recover, p_complete, T_days)
    P_tar = 50.0 - spread
    fin_cost = financing * (T_days / 365.0) * (P_tar / P_tar)  # 占成本比例
    net_gr = gr - fin_cost - trade_cost
    return net_gr

print("毛 vs 净 (低价差案例):",
      round(annualized_return(3.0, 40.0, 1.5, 0.90, 60)[2]*100, 1), "% ->",
      round(net_return(3.0, 40.0, 1.5, 0.90, 60)*100, 1), "%")
```

融资 + 交易成本看似小，但并购套利毛收益本就不高（个位数百分比），成本一扣可能把一半利润吃掉。这正是散户很难做这个策略的原因——资金成本结构不占优。

## 六、五个真实陷阱（高阶）

1. **把 spread 当无风险利润。** 永远记住 spread = 失败概率的定价。大 spread 不是便宜货，是市场在警告你。先问「为什么市场比我悲观」，再决定下不下注。

2. **低估「尾部相关性」。** 并购失败往往不是孤立事件：监管收紧、信用冻结、系统性下跌时，一批交易会**同时**黄。你觉得持了 10 笔不相关并购，实则危机里相关性飙到 0.8+，分散化在最需要时失效。

3. **忽略监管与时间不确定性。** 反垄断审查（尤其跨国外资、科技巨头）可能拖过原定完成日，持有期 T 被拉长，融资成本和不确定性同步上升。估概率时必须分情景：批准 / 附条件批准 / 否决 / 无限期拖延。

4. **分手费不是兜底。** 分手费（占交易额 1%~4%）确实软化下跌，但只在「被更优报价截胡」或「监管否决」等特定情形触发；若目标自身暴雷（财务造假），分手费不赔你，股价可能腰斩。

5. **内幕信息红线。** 并购套利离重大非公开信息极近。只用公开信息（公告、 filing、媒体报道）做判断，绝不碰任何疑似未公开知情交易——这是合规底线，不是策略细节。

## 七、什么时候这笔生意值得做

并购套利适合**有低成本资金、能做中性对冲、且能持续跟踪监管与交易进展**的玩家。对个人而言，更务实的入口是：**只参与高概率、大分手的现金收购**（失败概率低、下行有分手费托底），单笔仓位压到组合的 1%~2%，并接受「小赚多次、偶尔挨一记」的长期正期望。

它的精髓不在「预测哪笔成」，而在**把失败概率量化进价格、用仓位管理把小概率大亏关进笼子**。当你学会用 spread 反推市场怎么想，再用自己的研究去赚「认知差」——并购套利才从赌大小，变成一门可复利的生意。
