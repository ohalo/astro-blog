---
title: "美股隔夜异象：收盘到开盘跳变能否预测次日方向"
description: "股市的钱到底白天赚的还是晚上赚的？实证显示几乎所有长期收益发生在「隔夜」(close→open)，日内(intraday)几乎不贡献甚至略亏。但「隔夜跳变能预测次日方向吗」是另一个问题：本文把日内收益建模成对「前一日隔夜跳变」的过度反应反转(AR(1) 隔夜自相关使信号可预测)，400 股票×20 年合成面板实测：指数层隔夜年化 +11.5% vs 日内 −2.5%(隔夜贡献 128% 收益)；十档分组同日日内收益从 D1 −0.33% 单调到 D10 +0.31%(反转成立)；但原始 rank-IC 仅 −0.0011(t=−1.6)偏弱。诚实结论：反转是组合层真实存在、但日频换手成本大概率吞噬它——隔夜异象更适合做因子暴露而非高频交易。附完整 Python 与五张真实计算图。"
publishDate: '2026-10-07'
tags:
  - 量化交易
  - 隔夜收益
  - 市场微观结构
  - 均值回归
  - 横截面
  - 跳空
  - 过度反应
  - Python
language: Chinese
difficulty: intermediate
---

你有没有想过一个问题：**股市的钱，到底是在白天交易时段赚的，还是晚上闭市时赚的？**

直觉告诉你当然是白天——毕竟价格跳动、成交、博弈都发生在开盘到收盘。但一堆论文（Berkman, Jacobsen & Lee 2012 的 *"Stocks Go Up by Night"*；Liu, Luo & Zhang 2019）用美股数据给出反直觉结论：**长期来看，几乎所有收益都发生在「隔夜」(close→open)，而「日内」(open→close)几乎不赚钱，甚至略亏**。

更妙的是，这个异象还能变成**可交易的信号**：某只股票今天隔夜跳得越凶（尤其向上），今天日内越倾向于反向——也就是「隔夜承载情绪/过度反应，日内被信息交易者纠正」。

但「隔夜收益整体存在」和「隔夜跳变能预测次日方向」是**两件事**。本文把后者跑通，并诚实展示它到底能不能真的拿来赚钱。结论先放这：

**① 收益确实由隔夜贡献**：合成指数 20 年，隔夜年化 +11.5%、日内年化 −2.5%，隔夜吃掉 128% 的总收益（日内是负的）。
**② 反转在组合层成立**：按前一日隔夜跳变十档分组，同日日内收益从 D1（前夜涨最多）的 −0.33% 单调滑到 D10（前夜跌最多）的 +0.31%——典型过度反应反转。
**③ 但原始 rank-IC 偏弱**（−0.0011，t=−1.6）：说明这层信号被大量个股噪声淹没，日频直接做多空几乎肯定被交易成本吞噬。隔夜异象更适合当「因子暴露」持有，而非高频博弈。**

![指数隔夜 vs 日内累计：隔夜是发动机，日内几乎不贡献](/images/us-overnight-anomaly/overnight_vs_intraday.png)

## 一、收益的两段：隔夜 vs 日内

任意一天的对数收益都能干净拆成两段：

$$r_{day} = \underbrace{(close_{t-1} \to open_t)}_{\text{隔夜}} + \underbrace{(open_t \to close_t)}_{\text{日内}}$$

- **隔夜**：收盘后到第二天开盘前。这段时间你无法交易，但信息（财报、宏观、海外）在累积，散户「想了一晚上」后开盘挂单。实证上隔夜有**正漂移 + 高波动**——它是收益的发动机。
- **日内**：开盘到收盘的真实博弈。对指数而言，这段漂移≈0 甚至略负——开盘信息已被部分 price-in，剩下的是零和博弈。

我们用一个 400 股票 × 20 年（日度）的合成面板把这件事演示出来：隔夜给正漂移 `μ=0.00045`、波动 `σ=0.0065`；日内给轻微负漂移 `μ=−0.00010`、波动 `σ=0.0072`。

```python
import numpy as np
rng = np.random.default_rng(20261007)
N, T = 400, 252 * 20
mu_on, sd_on   = 0.00045, 0.0065     # 隔夜：正漂移
mu_in, sd_in   = -0.00010, 0.0072    # 日内：略负漂移

# 隔夜加入 AR(1) 自相关(phi=0.30)：过度反应的状态会延续到次日，
# 于是「前一日隔夜跳变」能携带可预测信息
phi = 0.30
on = np.empty((N, T))
for j in range(N):
    o = np.empty(T); o[0] = mu_on
    for t in range(1, T):
        o[t] = (1-phi)*mu_on + phi*o[t-1] + sd_on*rng.standard_normal()
    on[j] = o

# 日内 = 自身噪声 + 对「当日隔夜跳变」的反转(beta<0) + 市场因子
beta = -0.9
inr = mu_in + 0.0025*rng.standard_normal((N, T)) \
      + beta*(on - on.mean(axis=1, keepdims=True)) \
      + 0.5*rng.normal(0, 0.0012, T)
```

注意一个建模要点：**隔夜必须是自相关的（AR(1), φ=0.30）**。如果隔夜是纯 iid 噪声，那「今天的隔夜」和「明天的隔夜」毫无关系，自然也无法预测「明天的日内」。真实的过度反应往往延续几日，AR(1) 让「前一日隔夜跳变」携带了对次日的预测力——这是异象可交易的前提。

## 二、指数层面：隔夜贡献了 128% 的收益

把截面平均后画累计净值（图 1）：隔夜那条线稳步向上，日内那条几乎平走甚至微跌。拆成年化（图 2）：

- 隔夜年化：**+11.5%**（波动 1.4%）
- 日内年化：**−2.5%**（波动 1.6%）
- 隔夜占总收益比例：**128%**（因为日内是负的，隔夜不仅贡献全部，还「补回」了日内的亏损）

![隔夜 vs 日内年化收益占比](/images/us-overnight-anomaly/return_share.png)

这个结果和文献一致：**长期股权溢价几乎全在隔夜实现**。「buy and hold 赚钱」这件事，精确地说，是「buy at close、sell at next open」在赚钱。

## 三、横截面：隔夜跳变预测次日日内（反转）

现在问真正的交易问题：**某只股票前一天隔夜跳得越狠，它当天的日内会怎样？**

直觉是「过度反应反转」：隔夜的大涨很多是散户情绪/收盘后消息驱动的，开盘后信息交易者进场纠正，于是日内倾向于回吐。我们在日内收益里显式写了一个 `beta × 隔夜跳变` 的反转项（beta=−0.9），然后检验它是否真能在横截面上被检测到。

用 `t−1` 隔夜跳变对 `t` 日内收益做每日 rank-IC：

```python
prev_on = on[:, :-1]          # t-1 隔夜
cur_in  = inr[:, 1:]          # t   日内
ic_list = []
for t in range(prev_on.shape[1]):
    ic = np.corrcoef(np.argsort(prev_on[:, t]),
                     np.argsort(cur_in[:, t]))[0, 1]
    ic_list.append(ic)
ic_mean = np.mean(ic_list)
ic_t = ic_mean / (np.std(ic_list) / np.sqrt(len(ic_list)))
```

![横截面 rank-IC 时序：前夜跳得越狠，同日日内越倾向反转](/images/us-overnight-anomaly/ic_timeseries.png)

实测均值 IC = **−0.0011**，t-stat = **−1.6**。负号正确（反转），但**统计上偏弱**——这意味着反转信号在横截面上被大量个股噪声稀释，单看「隔夜跳变 → 次日日内」的 pairwise 关系并不显著。这恰恰是真实市场的样子：异象存在，但噪声极大。

## 四、十档分组：反转在组合层干净成立

rank-IC 偏弱不代表信号没用——组合层（把股票按信号分十档、看档内平均）往往比 pairwise 更敏感。我们用 Fama-MacBeth 式逐日分组：

```python
# 每天按前一日隔夜跳变排名，分十档（D1=涨最多 .. D10=跌最多）
ranks = np.argsort(np.argsort(prev_on, axis=0), axis=0)
dec = (9 - ranks * 10 // N).astype(int)
grp_mean = np.array([cur_in[dec[:, t] == g, t].mean()
                     for t in range(prev_on.shape[1])
                     for g in range(10)]).reshape(-1, 10).mean(axis=0)
```

![十档分组：D1（前夜涨最多）日内 −0.33% → D10（前夜跌最多）日内 +0.31%](/images/us-overnight-anomaly/decile_intraday.png)

结果非常干净：**D1（前夜涨最多）的同日日内平均收益 −0.33%，D10（前夜跌最多）+0.31%**，中间档位单调过渡。这是教科书式的过度反应反转——最高组最低组差约 0.64%/日。组合层把个股噪声平均掉了，信号比 pairwise IC 清晰得多。

## 五、可交易多空：信号有，但日频成本大概率吞噬它

把 D10 多日内、D1 空日内拼成多空组合，逐日净值（图 5）确实单调向上。但**这里必须泼冷水**：这个多空 Sharpe 在合成数据上被「横截面平均」放大了——它衡量的是「如果你能无成本地每日等权做多/做空一篮子」，现实里你做不到：

1. **日频换手成本极高**：每天重新分档意味着每天买卖整组，佣金+买卖价差+做空借券成本会把 0.64%/日的档差啃掉一大块（A 股甚至无法顺畅做空 D1 组）。
2. **做空约束**：美股做空需借券、有成本且可能召回；A 股融券极受限，D1（涨最多的要空）组根本做不出。
3. **幸存者偏差**：合成面板没有退市，真实里跌最狠的隔夜跳变股可能次日直接停牌/退市，反转收益拿不到。

所以诚实结论：**隔夜反转是真实存在的因子暴露，适合作为「持有型多空因子」或「信号叠加在别的策略上」，不适合直接日频高频交易**。把它和动量、规模等因子一起做横截面排序，比单挑出来日频博弈稳健得多。

![可交易多空组合净值（理想无成本情景，仅作信号存在性演示）](/images/us-overnight-anomaly/longshort_equity.png)

## 六、落地要点与诚实边界

1. **本文是合成演示**：隔夜的 AR(1) 自相关、日内反转 beta 都是我注入的结构，真实数据要用美股 CRSP/Compustat 的 close-to-open 与 open-to-close 收益直接算，但文献（Berkman et al. 2012；Liu et al. 2019）已实证隔夜效应真实存在且 robust。
2. **两段拆分必须用同一标的的 close/open**：常见错误是用「指数收盘价」近似，但指数成分股调整、分红处理会让 overnight 与 intraday 定义错位，必须用个股自身的 (close_t−1, open_t, close_t)。
3. **预测的是「日内」不是「隔夜」**：异象的「可交易」部分是「前夜跳变 → 当日日内反转」，你要在开盘后做日内方向，收盘平掉——不是赌下一个隔夜。
4. **成本是这个策略的命门**：档差 0.6%/日听起来香，但日频全组换手的交易成本（尤其做空端）在真实市场常在 10~30bps/笔，吃掉了大部分。降低了换手（如周频、月频重排）才能留住信号。
5. **regime 依赖**：隔夜效应在散户主导、情绪化时段（如 2000 科网泡沫、2020 疫情恐慌）最强；在机构主导的低波动期会变弱甚至反转。样本外要分 period 检验。
6. **不要把「整体隔夜收益」和「隔夜预测次日」混为一谈**：前者是 buy-and-hold 级别的长期事实，后者是短窗口交易信号，两者的持仓周期、成本结构、可实现性完全不同。本文同时验证了前者（强）与后者（组合层成立但偏弱、需控成本）。

## 完整可复现代码

```python
import numpy as np

rng = np.random.default_rng(20261007)
N, T = 400, 252 * 20
mu_on, sd_on = 0.00045, 0.0065
mu_in, sd_in = -0.00010, 0.0072
phi, beta = 0.30, -0.9

on = np.empty((N, T))
for j in range(N):
    o = np.empty(T); o[0] = mu_on
    for t in range(1, T):
        o[t] = (1-phi)*mu_on + phi*o[t-1] + sd_on*rng.standard_normal()
    on[j] = o

inr = mu_in + 0.0025*rng.standard_normal((N, T)) \
      + beta*(on - on.mean(axis=1, keepdims=True)) \
      + 0.5*rng.normal(0, 0.0012, T)

# ① 指数层面
idx_on, idx_in = on.mean(axis=0), inr.mean(axis=0)
ann_on = (1+idx_on.mean())**252 - 1
ann_in = (1+idx_in.mean())**252 - 1
print(f"overnight ann={ann_on:.1%}  intraday ann={ann_in:.1%}  "
      f"share={ann_on/(ann_on+ann_in):.0%}")

# ② 横截面 rank-IC (t-1 隔夜 -> t 日内)
prev_on, cur_in = on[:, :-1], inr[:, 1:]
ics = [np.corrcoef(np.argsort(prev_on[:, t]),
                   np.argsort(cur_in[:, t]))[0, 1]
       for t in range(prev_on.shape[1])]
ic_mean = np.mean(ics); ic_t = ic_mean/(np.std(ics)/np.sqrt(len(ics)))
print(f"rank-IC={ic_mean:.4f}  t={ic_t:.1f}")

# ③ 逐日十档分组
ranks = np.argsort(np.argsort(prev_on, axis=0), axis=0)
dec = (9 - ranks*10//N).astype(int)
grp = np.array([cur_in[dec[:, t] == g, t].mean()
                for t in range(prev_on.shape[1])
                for g in range(10)]).reshape(-1, 10).mean(axis=0)
print(f"D1={grp[0]:.3%}  D10={grp[9]:.3%}  reversal={grp[0]<grp[9]}")
```
