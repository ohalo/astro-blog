---
title: "做空 Beta(BaB)：押注高 Beta 股票的脆弱性"
description: "CAPM 说高 beta 高收益，杠杆约束却让高 beta 被高估、低 beta 被低估。300 股 20 年模拟：SML 实际斜率 2.1% 远低于理论 5.8%，杠杆中性的 BaB 组合年化 1.6%/Sharpe 0.33/beta≈0 截获正 alpha，附完整 Python（高阶）。"
publishDate: '2026-07-13'
tags:
  - 量化交易
  - 做空Beta
  - 因子投资
  - Betting Against Beta
  - 证券市场线
  - 杠杆约束
  - 资产定价
  - Python
language: Chinese
difficulty: advanced
---

资本资产定价模型(CAPM)给我们一个干净的交易指令：**想要高收益，就去买高 β 的股票**。证券市场线(SML)的斜率是正的——β 越高，预期超额收益越高。这几乎是所有组合管理的起点。

但 Frazzini & Pedersen(2014) 在 *Betting Against Beta* 里指出一个反直觉的事实：**真实世界里，高 β 股票并没有拿到 CAPM 许诺的那么高补偿，低 β 股票反而被系统性低估**。根因是「杠杆约束」——大多数投资者(共同基金、机构、散户)无法或不愿用杠杆，于是为了追逐风险敞口，他们把高 β 股票买贵了；而真正想要低波动的人，被迫只能买低 β 股票，把它买便宜了。结果是一条「过平的 SML」。

这一篇，我们用 300 只股票、20 年月度数据亲手把这条过平的 SML 揪出来，并用 Frazzini-Pedersen 的「杠杆中性」方法构造一个 **做空 Beta(BaB)** 组合：做多低 β、做空高 β，两条腿各自加杠杆到 β=1，让组合 β≈0，却截获低 β 正 alpha 与高 β 负 alpha 之间的落差。

结论先放：在我们的模拟里，**理论 SML 斜率 = 市场溢价 5.77%/年，而实际拟合斜率只有 2.12%/年**(过平)；低 β 十分位(β≈0.39)被赋予约 +1.2%/年的截距 alpha，高 β 十分位(β≈2.12)则是负的。杠杆中性的 BaB 组合 **年化 1.57%、波动 4.83%、Sharpe 0.33、最大回撤 −12.1%、组合 β≈0**，把纯 alpha 干净地剥离出来；相比之下高 β 十分位虽然裸收益 6.17%，但 Sharpe 只有 0.20——那点收益几乎全是市场 β 的暴露，风险调整后并不好看。

![β vs 年化超额收益：CAPM 理论 SML(虚线正斜率 5.77%) vs 实际拟合 SML(红线 2.12%，过平)](/images/bet-against-beta/bab_sml.png)

## 一、数据：怎么合成（并揪出）这个异象

为演示机制，我们合成 300 只股票、240 个月(20 年)的月度超额收益。每只股票服从「市场 + 特质」结构，并**显式嵌入 BaB 异常**：`α_i = C·(1 − β_i)`——β 越低 alpha 越高，β 越高 alpha 越低。这正是「杠杆约束下 SML 过平」的数学表达。

```python
import numpy as np

rng = np.random.default_rng(20260713)
N, T = 300, 240
Rf = 0.002            # 月度无风险
mu_m, sig_m = 0.005, 0.040   # 市场月度超额收益均值、波动
C_ANOM = 0.0030       # BaB 异常强度：alpha = C·(1 - beta)

beta = rng.uniform(0.3, 2.2, N)        # 个股 beta，跨 0.3~2.2
idio = rng.uniform(0.015, 0.045, N)    # 特质月度波动
mkt_ex = rng.normal(mu_m, sig_m, T)    # 市场月度超额收益
eps = rng.standard_normal((T, N)) * idio
alpha = C_ANOM * (1.0 - beta)          # 低 beta -> 正, 高 beta -> 负
ret = alpha[None, :] + beta[None, :] * mkt_ex[:, None] + eps
excess = ret - Rf
```

注意 `alpha` 的构造：当 β=1 时 alpha=0(恰好不偏不倚)；β<1 时 alpha>0(低 β 被低估)；β>1 时 alpha<0(高 β 被高估)。这就是「高 β 没拿到足额补偿」的来源。

## 二、证据一：证券市场线(SML)过平

把每只股票的年化超额收益对 β 回归，得到实际 SML；再画出 CAPM 理论 SML(斜率 = 市场溢价)。两者一比，异象就显形了。

```python
ann_ex = excess.mean(0) * 12
A = np.vstack([np.ones(N), beta]).T
coef = np.linalg.lstsq(A, ann_ex, rcond=None)[0]
capm_int, capm_slope = coef[0], coef[1]
mkt_prem_ann = mkt_ex.mean() * 12      # 理论 SML 斜率 = 已实现市场溢价
print(capm_slope * 100, mkt_prem_ann * 100)
# 实际斜率 2.12%  理论 5.77%
```

**实际拟合斜率仅 2.12%/年，而 CAPM 理论斜率是 5.77%/年**——差了将近三分之二。也就是说，一只 β=2 的股票，CAPM 承诺它比 β=1 多拿 5.77%/年，实际只多拿约 2.12%/年；β=0.4 的低 β 股票，CAPM 说它该比 β=1 少拿 3.46%/年，实际只少拿约 1.27%/年，甚至还带一点正截距(回归截距 +1.17%/年)。**高 β 的溢价被「挤掉」了，低 β 悄悄占了便宜**。图上红实线(实际)明显比灰虚线(理论)平，高 β 区域整片落在理论线下方。

## 三、证据二：BaB 组合的杠杆中性构造

光看 SML 还不够。Frazzini-Pedersen 的精髓是：**既然低 β 被低估、高 β 被高估，就做多低 β、做空高 β，而且两条腿都按 1/β 加杠杆，使组合 β 精确归零**。这样组合不赚市场的钱，只赚「低 β 正 alpha − 高 β 负 alpha」的落差。

```python
order = np.argsort(beta)
dec = np.array_split(order, 10)
D1, D10 = dec[0], dec[9]            # 低 beta / 高 beta 十分位
beta_D1, beta_D10 = beta[D1].mean(), beta[D10].mean()
ex_D1 = excess[:, D1].mean(1)       # 低 beta 腿月度超额(等权)
ex_D10 = excess[:, D10].mean(1)     # 高 beta 腿月度超额(等权)

lev_long = 1.0 / beta_D1            # 做多腿杠杆：把 β 拉到 1
lev_short = 1.0 / beta_D10          # 做空腿杠杆：把 |β| 拉到 1
bab_ex = lev_long * ex_D1 - lev_short * ex_D10
bab_beta = lev_long * beta_D1 - lev_short * beta_D10   # ≈ 0
print(beta_D1, beta_D10, lev_long, lev_short, bab_beta)
# 0.390  2.122  2.564  0.471  0.0000
```

本例低 β 腿平均 β=0.39，需要 2.56 倍杠杆把它放大到 β=1；高 β 腿平均 β=2.12，做空时只需 0.47 倍名义就把 |β| 压到 1。两条腿一多一空，组合 β = 1 − 1 = **0.0000**，完全市场中立。

![BaB 杠杆中性构造：两条腿各按 1/β 加杠杆，组合 β≈0(净敞口=0)](/images/bet-against-beta/bab_leverage.png)

两条腿各自贡献的 alpha 落差也很干净：做多低 β 腿的月度 alpha 贡献约 **+0.469%/月**，做空高 β 腿贡献约 **+0.159%/月**(高 β 自身 alpha 为负，做空它等于做多这份负 alpha 的反面)。两者相加，就是 BaB 的超额收益来源。

## 四、回测：BaB 组合 vs 市场 vs 高 β 腿

把 BaB 组合、等权全市场、纯高 β 十分位三条净值放在一起比：

```python
def netvalue(r):
    return np.cumprod(1.0 + r)

eq_bab = netvalue(bab_ex)
eq_mkt = netvalue(excess.mean(1))
eq_hb = netvalue(ex_D10)

bab_ann = bab_ex.mean() * 12
bab_vol = bab_ex.std(ddof=1) * np.sqrt(12)
bab_shp = bab_ann / bab_vol
print(bab_ann, bab_vol, bab_shp)
# 0.0157  0.0483  0.33
```

结果：

| 组合 | 年化超额 | 波动 | Sharpe | 最大回撤 |
|---|---|---|---|---|
| **BaB(杠杆中性)** | **1.57%** | 4.83% | **0.33** | **−12.1%** |
| 等权市场 | 3.84% | — | 0.21 | — |
| 高 β 十分位 | 6.17% | — | 0.20 | −82.1% |

乍看高 β 十分位裸收益最高(6.17%)，但那是用 2.12 倍市场 β 换来的——它的 Sharpe 只有 0.20，和等权市场(0.21)几乎一样，**风险调整后毫无优势**，还背着 −82% 的恐怖回撤。BaB 虽然年化只有 1.57%，却是 **β≈0 下的纯 alpha**，回撤只有 −12%，Sharpe 反而最高。

![BaB 用近零市场 beta 跑赢：高 beta 腿风险调整后明显最弱](/images/bet-against-beta/bab_cumulative.png)

## 五、CAPM 归因：β≈0，alpha 为正

最后用 CAPM 给 BaB 组合本身做一次回归，确认它的收益不是来自市场 β：

```python
A2 = np.vstack([np.ones(T), mkt_ex]).T
bcoef = np.linalg.lstsq(A2, bab_ex, rcond=None)[0]
bab_alpha_capm = bcoef[0] * 12
bab_beta_capm = bcoef[1]
print(bab_alpha_capm, bab_beta_capm)
# 0.0157  0.001
```

BaB 组合的市场 β 仅 **0.001**(统计上就是 0)，而截距 alpha 高达 **1.57%/年**。一句话总结：**BaB 赚的不是市场的钱，是「低 β 被低估、高 β 被高估」这个定价偏差的钱**。

![BaB 组合的 CAPM 归因：beta≈0(市场中立)，截获显著正 alpha](/images/bet-against-beta/bab_capm.png)

## 六、真实陷阱：结论要打哪些折扣

- **这是合成数据**：我们直接把 `α=C·(1−β)` 写进了生成过程，所以必然能复现过平 SML。真实世界的 BaB 异象来自杠杆约束 + 投资者偏好，幅度更小、更 noisy，但 Frazzini-Pedersen 用美股、环球股票、国债、外汇 20+ 年的实盘数据都验证过它的存在(且多空组合确实接近市场中性)。
- **融资成本与做空成本**：BaB 做空高 β 腿要付融券费、借券费，做多低 β 腿要加杠杆付利息。我们的模拟没扣这些，实盘会把 1.57%/年的 alpha 啃掉一部分(尤其在高利率环境)。
- **硬难借(hard-to-borrow)**：真正高 β 的小盘股往往借不到券，做空腿无法完全建仓，组合会残留正 β，异象被稀释。
- **危机时刻的反转**：崩盘时高 β 股票暴跌，做空高 β 本该大赚；但同一时刻杠杆被强平、流动性枯竭，BaB 在 2008、2020 这类极端日反而可能短期失血——它不是「危机保险」，而是「日常定价偏差」策略。
- **样本外衰减**：BaB 被写进论文、做成 ETF 后，拥挤度上升，溢价会收窄。任何因子策略都要警惕「被发现后变小」。

## 小结

CAPM 告诉我们「高风险高收益」，但杠杆约束让市场做不到：高 β 股票被买贵、低 β 被买便宜，SML 因此过平。用「做多低 β + 做空高 β + 各按 1/β 加杠杆」构造的 BaB 组合，把这条过平 SML 的落差变成可交易的纯 alpha——在我们的 300×240 模拟里，它做到了 β≈0、年化 1.57%、Sharpe 0.33、回撤仅 −12%。数字不大，但干净：它证明了一件事，**跑赢市场不一定需要承担更多市场风险，有时候只要纠正别人的杠杆约束就够了**。
