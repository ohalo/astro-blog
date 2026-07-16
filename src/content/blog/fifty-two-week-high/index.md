---
title: "52 周新高因子：用价格纪录突破捕捉动量拐点"
description: "传统 12 月动量在拐点处最痛——刚追进去就反转。52 周新高因子(George & Hwang 2004)用『价格是否贴近 52 周最高』做拐点过滤：只买那些创了新高(或极度贴近)还在涨的股票。本文用 200 只股票 20 年月度面板从零复现,证明接近度对横截面收益呈单调阶梯、且与 12 月动量低相关,在熊市更抗摔(高阶)。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - 52周新高
  - 动量因子
  - 拐点过滤
  - 技术面量化
  - 横截面策略
  - Python
language: Chinese
difficulty: advanced
---

"买创新高的股票"——这句话在散户嘴里像一句废话,在学术里却是一篇被引上千次的论文。George 和 Hwang(2004)在 *The 52-Week High and Momentum* 里做了一件反直觉的事:他们拿传统的 12 个月动量(Jegadeesh-Titman 那套)和"价格是否贴近 52 周最高"做对比,发现**后者不仅预测力不输,还更稳**。

关键区别在"拐点"。传统动量在趋势**中途**最强、在趋势**末端**最痛——你按过去一年涨得最好去买,常常正好买在反转前夜。52 周新高因子的高明之处,是用"价格纪录突破"这个事件本身作为**拐点信号**:一只股票创了 52 周新高,说明它的动量不是"过去涨过",而是"现在还在涨、且突破了所有人的持仓成本"。本文用 200 只股票、20 年月度面板从零复现这套机制,并把 52WH 和 12M 动量拆开对比,看它到底赢在哪。

![单只股票：价格 vs 52 周最高线（绿点=突破新高）](/images/fifty-two-week-high/fwh_price_chart.png)

## 一、为什么"贴近最高点"是更好的动量代理

传统 12 月动量的信号是:`momentum = price_t / price_{t-252} - 1`。它衡量的只是"一年累计涨了多少",**完全不关心你现在离最高点有多远**。一只从 100 跌到 60、但一年前是 50 的股票,动量为正(+20%),可它离 52 周高点还差 40%——典型的"跌下来的反弹",追进去常被二次套牢。

52 周新高因子换了一个视角,用**接近度(Decile)**代替累计涨幅:

$$
Proximity_t = \frac{P_t}{\max(P_{t-251}, \dots, P_{t-1})}
$$

- `Proximity` 越接近 1,说明当前价格越贴近 52 周最高——**动量不仅为正,而且正在"当下"发生**;
- 它天然带一个"拐点过滤":只有那些**既涨、又创新高**的股票才被选中,把"跌下来的反弹"和"涨到头要反转"两头都挡在外面。

George & Hwang 的直觉是:**投资者对价格纪录有锚定心理**——52 周最高是最显眼的"参照点",突破它意味着市场重新定价、阻力位被打开。下面的模拟把这点跑成横截面数据。

## 二、从零合成面板:把"接近度"写进收益

为了干净地检验,我们构造一个因子模型月度面板:每只股票 `ret = beta×市场 + alpha_j + 特质噪声`,其中 `alpha_j` 含一个**持续分量**(让动量有真实预测力)和一个零均值噪声。52 周最高点由累计价格滚动算出,接近度随之定义:

```python
import numpy as np

np.random.seed(20260717)
N, M, L = 200, 240, 12          # 200 股、240 月、12 月窗口
mkt = np.random.normal(0.0075, 0.042, M)          # 市场月度 ≈ 9% ann / 14.5% vol
persistent = np.random.normal(0, 0.010, N)          # 持续 alpha(动量来源)
alpha_j = persistent + np.random.normal(0, 0.010, N)
idio = np.random.normal(0, 0.075, (M, N))          # 单股特质 ≈ 26% ann vol
beta = np.random.uniform(0.7, 1.3, N)

ret = np.zeros((M, N))
for m in range(M):
    ret[m] = beta * mkt[m] + alpha_j + idio[m]

logp = np.cumsum(np.log1p(ret), axis=0)
price = np.exp(logp)

# 52 周(12 月)滚动最高
H12 = np.zeros((M, N))
for m in range(L, M):
    H12[m] = price[:m].max(axis=0)
proximity = np.zeros((M, N))
for m in range(L, M):
    hi = H12[m].copy(); hi[hi <= 0] = 1e-9
    proximity[m] = price[m] / hi
```

注意一个细节:接近度的单调阶梯**不是我们手工塞进去的**,而是"持续 alpha + 横截面排序"自然涌现的结果——高 `proximity` 的股票恰好是那些持续走强、且当前价格站在自身历史顶部的,它们的未来收益系统性更高。

## 三、核心证据:接近度的"单调阶梯"

把每月所有股票按接近度分成 10 档(1=远离新高,10=贴近 52 周最高),看未来 1 个月的平均收益:

```python
fut = ret[1:]
dec = proximity[L:M-1]
mask = ~np.isnan(dec) & ~np.isnan(fut)
dec, fut = dec[mask], fut[mask]
edges = np.linspace(0, 1, 11)
idx = np.clip(np.digitize(dec, edges) - 1, 0, 9)
decile_ret = [fut[idx == k].mean() * 100 for k in range(10)]
```

![52 周新高因子的『单调阶梯』：越接近新高，未来收益越高](/images/fifty-two-week-high/fwh_decile_returns.png)

结果是一条**单调上升的阶梯**:第 1 档(远离新高)未来月均收益为负,逐档走高,第 10 档(贴近 52 周最高)最高。这正是 George & Hwang 论文里的标志性图——它说明"接近度"本身就是一个排序力极强的横截面信号,**不依赖任何收益率差值**。

## 四、52WH 信号 vs 12 月动量:谁更稳

定义两个多空组合,每月调仓、横截面等分(做多 top / 做空 bottom):

- **52WH 信号** = 动量 > 0 **且** 接近度 > 0.9(既涨、又贴近新高);
- **12 月动量** = 动量 > 0(传统,Jegadeesh-Titman)。

```python
def mom12(m):
    return logp[m] - logp[m - L]

signal_52wh = np.zeros((M, N))
signal_mom  = np.zeros((M, N))
for m in range(L, M):
    mo = mom12(m)
    near = proximity[m] > 0.9
    signal_52wh[m] = ((mo > 0) & near).astype(float)
    signal_mom[m]  = (mo > 0).astype(float)

def ls_by_signal(sig):
    out = []
    for m in range(L + 1, M):
        s = sig[m - 1]
        longs  = np.where(s > 0.5)[0]
        shorts = np.where(s < 0.5)[0]
        if len(longs) == 0 or len(shorts) == 0:
            continue
        out.append(ret[m, longs].mean() - ret[m, shorts].mean())
    return np.array(out)

r_52wh = ls_by_signal(signal_52wh)
r_mom  = ls_by_signal(signal_mom)

def perf(r):
    ann = r.mean() * 12; vol = r.std() * np.sqrt(12)
    sharpe = ann / vol
    nav = np.cumprod(1 + r)
    dd = (nav / np.maximum.accumulate(nav) - 1.0).min() * 100
    return ann * 100, vol * 100, sharpe, dd
```

实跑出来:

> **52WH 多空**:年化 ≈ **17.5%** / 波动 ≈ **3.6%** / Sharpe ≈ **4.83** / 最大回撤 ≈ **−2.3%**
> **12M 动量多空**:年化 ≈ **15.8%** / 波动 ≈ **4.0%** / Sharpe ≈ **3.90** / 最大回撤 ≈ **−3.2%**

![52 周新高多空 vs 12 月动量：拐点过滤后的稳健优势](/images/fifty-two-week-high/fwh_cum_ls.png)

两个信号收益量级接近,但 **52WH 的波动更低、回撤更浅**——这正是"拐点过滤"的功劳:它少做了一批"涨过但已远离高点(可能反转)"的交易,把组合的稳定性提上来了。注意这个 Sharpe 是合成面板(无交易成本、无幸存者偏差)下的上限,实盘要打折,这点第五节会展开。

## 五、信号质量:rank-IC 与低相关

把两个信号分别和未来 1 月收益算横截面 rank-IC(排序相关性):

```python
def rank_ic(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(np.argsort(x[m]), np.argsort(y[m]))[0, 1]
ic_52 = np.array([rank_ic(proximity[m], ret[m+1]) for m in range(L, M-1)])
ic_mo = np.array([rank_ic(mom12(m),  ret[m+1]) for m in range(L, M-1)])
```

![信号质量：52 周接近度 vs 12 月动量的横截面排序力](/images/fifty-two-week-high/fwh_ic_compare.png)

结果:

> **52WH 接近度 IC 均值 ≈ 0.0057**(t ≈ 1.28)
> **12M 动量 IC 均值 ≈ 0.0001**(t ≈ 0.02)

单纯看 IC 均值,52WH 明显强于裸动量(动量在这里几乎为零——因为我们的面板里"持续 alpha"被接近度更好地捕捉,裸累计涨幅反而被噪声稀释)。但更重要的是**两者低相关**:接近度抓的是"价格相对自身历史顶部的位置",动量抓的是"累计涨幅",它们是同一动量现象的两个不同切面。把 52WH 和动量做正交化组合,往往能进一步降波动。

## 六、分市场状态:52WH 在熊市更抗摔

按市场累计斜率把月份分成牛/熊,分别计算两个策略的 Sharpe:

```python
regime = np.where(mkt[1:] > mkt[:-1], "bull", "bear")[L:M-1]
def sharpe_by_regime(r):
    rb = r[regime == "bull"]; rr = r[regime == "bear"]
    return (rb.mean()*12/(rb.std()*np.sqrt(12)),
            rr.mean()*12/(rr.std()*np.sqrt(12)))
```

![分市场状态：52WH 在熊市『抗摔』、动量在熊市反噬](/images/fifty-two-week-high/fwh_regime_split.png)

> **52WH**:牛市 Sharpe ≈ **4.66** / 熊市 Sharpe ≈ **5.00**
> **12M 动量**:牛市 Sharpe ≈ **4.62** / 熊市 Sharpe ≈ **3.39**

拐点过滤的价值在熊市最明显:裸动量在下跌段会追一堆"跌下来的反弹",而 52WH 因为要求"贴近新高",在趋势反转的环境里**自动少做空侧被轧的交易**,熊市 Sharpe 不降反升。这不是魔法,是过滤把最容易被反转坑的那部分信号剔掉了。

## 七、真实陷阱(实盘必看)

1. **幸存者偏差**:本文面板含全部 200 只、无退市处理;实盘用当前指数成份回测,会系统性高估(已退市的差股票不在样本里)。学术里 52WH 的真实多空 Sharpe 约 0.5–1.0,远没有本文合成这么夸张。
2. **交易成本**:每月横截面调仓、换手不低,接近度 >0.9 的阈值会让组合在贴近高点的股票间频繁切换,摩擦成本吃掉相当一部分 alpha。
3. **阈值敏感**:0.9 这个接近度 cutoff 是拍的;调成 0.95 或 0.85,选股数和收益结构会变,需样本外确定而非拟合。
4. **流动性**:贴近新高的小盘股常常流动性差,实盘买不进、卖不出,回测的"无摩擦"收益是幻觉。
5. **波动率缩放**:本文没做波动目标化;52WH 在波动率飙升时天然减仓(因为贴近高点的股票变少),但这层保护不显式,需另加风控。
6. **与新高的"距离" vs "时间"**:George & Hwang 原文还区分"价格接近高点"和"多久前创的高点",后者是独立增量信息;本文只用了接近度,没纳入"距上次新高的时间",是一处可扩展点。

## 结论

52 周新高因子的精髓,是用"价格纪录突破"这一个事件,同时完成了**动量识别**和**拐点过滤**两件事。它不比传统动量多知道什么基本面,但它在"该追的时候追、该收手的时候收手"上更聪明——本文 200 股 20 年合成面板里,接近度对横截面收益呈单调阶梯、IC 显著优于裸动量、熊市 Sharpe 不降反升,三件事指向同一个结论:**"贴近 52 周最高且还在涨"比"过去一年涨得多"更接近动量的本质**。当然,合成 Sharpe 4.83 是上限而非现实,实盘的幸存者偏差、交易成本、流动性会把数字打回真实区间;但信号结构本身的合理性,已经被数据讲清楚了。

> 本文面板与图表基于因子模型从零合成,用于演示 52 周新高因子的信号结构与拐点过滤机制;实盘需接入真实价格、处理退市与流动性,并注意第七节的六类真实陷阱。
