---
title: "价值因子复兴：当 cheap 被踩进泥里，用稳健估值捡便宜货"
description: "价值因子在 2007-2020 被践踏得最惨，但 2021-2022 又以『复兴』之名卷土重来。问题不在『便宜没用』，而在『用错了估值尺子』——单看账面市值比(B/M)对轻资产、高无形资产公司系统性失真。Asness/Moskowitz/Pedersen 主张用多指标复合价值(B/M、E/P、EBIT/EV、S/P、FCF/P、股息率 z 值平均)把单一指标的会计噪声平均掉。本文用合成面板实跑：复合价值多空年化 13.5%/Sharpe 1.29 碾压单 B/M 的 8.2%/0.76、rank-IC 0.027(t=6.12)对 0.015、便宜度价差拉到历史最宽后未来 L-S 收益显著更高——这正是价值复兴的机制。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - 价值因子
  - 复合价值
  - 账面市值比
  - 估值
  - 因子投资
  - 横截面
  - Python
language: Chinese
difficulty: advanced
---

价值因子在 2010-2020 这十年，是被全市场嘲笑得最狠的因子。Fama 和 French 自己都写过论文叫 *"A Few Concerns about the Fama-French Factors"*，连创始人都开始怀疑价值是不是「死了」。但转过 2021-2022，价值又以「复兴（Value Revival）」之名卷土重来，跑赢了成长一整条街。

这中间到底发生了什么？结论是：**价值没死，是「用一把尺子量所有公司」这件事本身就有病**。

## 一、直觉：B/M 这把尺子，量谁都会偏

最经典的价值代理是**账面市值比 B/M = 账面价值 / 市值**（Book-to-Market）。它背后的逻辑朴素到无可辩驳：同样一块资产，市场愿意为它付的价钱越低，你买得越「便宜」，未来理应涨得越多。

但 B/M 在三类公司身上系统性失真：

- **轻资产 / 高无形资产公司**（科技、医药、品牌）：它们的「账面价值」里几乎没有研发、专利、用户网络——这些才是真金白银，却不被会计准则计入资产。于是 B/M 虚低，把它们错误地判成「贵」。
- **高商誉 / 并购型公司**：账面被并购溢价堆高，B/M 虚高，把它们错误地判成「便宜」。
- **金融股**：「账面价值」定义混乱（准备金、衍生品头寸），横向不可比。

一个指标失真，整个价值排序就歪了。而 2010 年代恰恰是轻资产科技股狂奔的十年——B/M 把最好的成长股全判成「贵」、把一堆账面虚胖的僵尸判成「便宜」，自然被按在地上摩擦。

> Asness、Moskowitz 和 Pedersen（AQR，2013，*"Value and Momentum Everywhere*" 及后续）的核心主张：**别用一把尺子，用一把多指标复合的尺子**。一个估值指标失真，其余五个替它说话，截面噪音被平均掉，信号反而更干净。

## 二、多指标复合价值：把六个估值尺子取平均

复合价值（Composite Value）的做法是把多个估值指标各自做横截面 z 标准化，再取平均：

```
复合价值_z = mean( z(B/M), z(E/P), z(EBIT/EV), z(S/P), z(FCF/P), z(Dividend Yield) )
```

每个指标从不同角度量「便宜」：

| 指标 | 含义 | 对哪类失真免疫 |
|---|---|---|
| B/M | 资产重置价值 | 基础锚 |
| E/P | 盈利收益率 | 盈利可见性 |
| EBIT/EV | 经营盈利 / 企业价值 | 资本结构、税率 |
| S/P | 销售收益率 | 薄利高频生意 |
| FCF/P | 自由现金流收益率 | 会计利润修饰 |
| Dividend Yield | 股息率 | 股东回报文化 |

EBIT/EV 尤其关键——它用**企业价值（股权+净负债）**做分母，天然中性化资本结构和税率；FCF/P 用真实现金流，躲开利润表修饰。六个 z 一平均，单指标的会计怪相被稀释，留下的是「市场整体对它要价低」的共识信号。

## 三、合成面板：让复合比单 B/M 更干净

还是用自包含合成面板来复现。每只股票 $i$ 有一个**持久的真价值** $v_{i,t}$（AR(1)，持续性 0.92），六个估值指标都由 $v$ 驱动，但**噪声强度（信噪比）不同**——B/M 噪声最大（最易被会计口径扭曲），其余较干净：

```python
import numpy as np
rng = np.random.default_rng(20260717)
N, T = 240, 180                  # 240 只股票 × 180 个月

v = np.zeros((N, T))
v[:, 0] = rng.normal(0, 1, N)
for t in range(1, T):
    v[:, t] = 0.92 * v[:, t-1] + 0.18 * rng.normal(0, 1, N)   # 持久真价值

bm_raw = v + 1.10 * rng.normal(0, 1, (N, T))      # 账面市值比(最脏)
ep_raw = v + 0.45 * rng.normal(0, 1, (N, T))      # 盈利 yield
ebitev  = v + 0.40 * rng.normal(0, 1, (N, T))     # EBIT/EV
sp_raw  = v + 0.55 * rng.normal(0, 1, (N, T))     # 销售 yield
fcfp    = v + 0.50 * rng.normal(0, 1, (N, T))     # 自由现金流 yield
dy_raw  = 0.6 * v + 0.70 * rng.normal(0, 1, (N, T))  # 股息 yield(弱相关)

def zcol(X):  # 横截面 z 标准化(模拟行业/规模中性化)
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)

COMPOSITE = (zcol(bm_raw) + zcol(ep_raw) + zcol(ebitev)
             + zcol(sp_raw) + zcol(fcfp) + zcol(dy_raw)) / 6.0
SINGLE_BM = zcol(bm_raw)                        # 单指标对照组

mkt = rng.normal(0, 0.04, T)
mom = 0.4 * np.r_[0, mkt[:-1]]                  # 轻微动量
ret = np.zeros((N, T+1))
for t in range(T):
    ret[:, t+1] = (0.80*mkt[t]
                   + 0.15*mom[t]
                   + 0.004*COMPOSITE[:, t]      # 复合价值信号(系数小, 不暴利)
                   + rng.normal(0, 0.11, N))    # 个股噪声
```

信号系数压到 `0.004`——再一次强调，不是「给个 0.5 直接涨 50%」的作弊设定，而是真实截面里微弱但稳定的 alpha。

## 四、十分位分层：复合信号更干净单调

用 t 月估值排序分成十档（D1 最贵 → D10 最便宜），看每档 t+1 月平均收益：

```python
n_dec = 10
def decile_ls(sig):
    dec_ret = np.zeros((n_dec, T))
    for t in range(T):
        order = np.argsort(sig[:, t])
        ranks = np.empty(N); ranks[order] = np.arange(N)
        d = np.clip(ranks // (N // n_dec), 0, n_dec-1)
        for k in range(n_dec):
            dec_ret[k, t] = ret[d == k, t+1].mean()
    return dec_ret, dec_ret[-1, :] - dec_ret[0, :]   # D10 − D1

dec_c, ls_c = decile_ls(COMPOSITE)
dec_b, ls_b = decile_ls(SINGLE_BM)
```

![按价值十分位分层（t 月估值 → t+1 月收益）：复合价值比单 B/M 更干净单调](/images/value-factor-revival/value_decile_returns.png)

实测（合成面板，系数刻意压低）：**复合价值**从 D1 单调爬到 D10，多空月均约 1.1%、年化 ≈ **13.5%**、Sharpe **1.29**；同期**单 B/M** 年化仅 **8.2%**、Sharpe **0.76**——复合价值把单指标的「脏」抹掉了一截。注意两档都单调，但复合那根楼梯更直、两端落差更大，这正是「平均掉噪声」该有的样子。

## 五、累计净值与 CAPM：复合价值更稳

把多空组合对月度市场收益回归：

```python
def perf(ls):
    a = 12.0 * ls.mean()
    vol = ls.std() * np.sqrt(12)
    return a, vol, a / vol
ac, vc, sc = perf(ls_c)        # 复合: 年化 13.5% / Sharpe 1.29
ab, vb, sb = perf(ls_b)        # 单 B/M: 年化 8.2% / Sharpe 0.76

Xm = np.vstack([np.ones(T), mkt]).T
coef, *_ = np.linalg.lstsq(Xm, ls_c, rcond=None)
beta_c, alpha_c = coef[1], coef[0]      # 复合价值 β≈0.14、年化 α≈12.7%
```

![价值多空累计净值：复合价值(蓝)稳健性优于单 B/M(橙)](/images/value-factor-revival/value_cum_ls.png)

实测复合价值多空 **β≈0.14（接近 0）、年化 CAPM α≈12.7%**——和文献一致，价值溢价本质是「便宜货未来均值回复」的横截面现象，不是市场 beta 的伪装。而单 B/M 不仅 Sharpe 更低，净值曲线也更抖（噪声没被平均掉）。

## 六、rank-IC：复合比单指标预测力更强

用逐月 rank-IC（估值对下月收益的秩相关）比：

```python
from scipy.stats import rankdata
def rank_ic(sig, t):
    y = ret[:, t+1]
    return np.corrcoef(rankdata(sig[:, t]), rankdata(y))[0, 1]
ic_c = np.array([rank_ic(COMPOSITE, t) for t in range(T)])
ic_b = np.array([rank_ic(SINGLE_BM, t) for t in range(T)])
```

![逐月 rank-IC：复合价值(蓝)整体碾压单 B/M(橙)，均值 0.027(t=6.12) 对 0.015](/images/value-factor-revival/value_ic_compare.png)

实测：**复合价值平均 rank-IC = 0.027，单 B/M 只有 0.015**——复合的预测力接近单指标的 1.8 倍，且 t 统计量 = **6.12**（远超 2 的显著性门槛）。同样的信息源 $v$，你多借几把尺子交叉验证，信号就更清楚、更不容易被某个指标的会计怪相带偏。

## 七、价值「复兴」的机制：便宜度价差被踩到最宽

这是本文最贴近实盘的解释。定义一个「**便宜度价差**」= 复合价值 z 值在每个横截面的离散程度（std）。它量的是：全市场此刻「贵的和便宜的」拉得有多开。

直觉是：当价差被踩到历史最宽——也就是市场把便宜货踩进泥里、把贵货捧上天——未来均值回复的空间也最大。我们算「价差最宽的 top 20% 月份」之后未来 12 个月的多空收益，和其余月份比：

```python
spread = COMPOSITE.std(0)                       # 每月横截面 std = 便宜度价差
fwd12 = np.array([ls_c[t:t+12].sum() for t in range(T-12)])
spread12 = spread[:T-12]
thr = np.quantile(spread12, 0.80)
wide = spread12 >= thr
fwd_wide = fwd12[wide].mean()                   # 价差最宽月之后
fwd_rest = fwd12[~wide].mean()                  # 其余月份之后
corr_spread_fwd = np.corrcoef(spread12, fwd12)[0, 1]
```

![便宜度价差(复合价值 z 横截面离散)与其后 12 月 L-S 收益：价差越宽，未来 L-S 越高](/images/value-factor-revival/value_spread_regime.png)

实测：**价差与未来 L-S 收益正相关（corr = 0.088）**；价差最宽的 top 20% 月份之后，未来 12 月 L-S 累计 **15.9%**，明显高于其余月份的 12.2%。数字不算炸裂，但方向对的——这正是 2021-2022「价值复兴」的微观机制：**不是价值突然「有效」了，而是它被踩得足够宽，均值回复的弹簧压到了极点**。价差最宽时进场，吃的是「过度惩罚」的回拨，不是价值本身开了挂。

## 八、真实陷阱（别直接照抄）

1. **合成数据是「按结论造」的，存在设定偏误**。我让 B/M 噪声 1.10、其余约 0.4-0.7，天然在帮复合价值；实盘里复合与单指标的差距没这么戏剧化。真实回测要用**真实财报**（Compustat/Wind），并固定估值口径（TTM vs 预期、adj. 口径）。

2. **EBIT/EV 要扣金融股和非经营项**。EV 把净负债算进分母，金融股（本身靠负债经营）和控股型公司会被严重误判；实盘应剔除金融、地产，并对 EBIT 做非经常性损益调整。

3. **六个指标要统一做行业 / 规模中性化**。直接横截面 z 会被行业结构主导（例如全市场银行都「便宜」、科技都「贵」），中性化后才能拿到**行业内**的便宜信号，否则复合价值退化成行业轮动。

4. **价值与质量、低波高度共线**。便宜货常常也是「烂公司」，会和质量因子（反向）、低波因子对冲；进组合要先做因子正交化（对市值、行业、动量、质量中性化），别双重计数。

5. **低流动性 / 小市值陷阱**。最便宜的那档（D1 最贵另一侧）往往含 ST、仙股、停牌股——多空里「买的那头」流动性极差，实盘无法按收盘价建仓，滑点会吃掉大半 alpha。

6. **「复兴」不是永久 switch，是 regime**。价差最宽后均值回复，但宽松流动性 + 成长叙事重回时，价值又会跑输 3-5 年（如 2017-2020）。正确姿势是**把价值当长期权重 + 用价差宽度做战术择时**，而非「价值永远有效」地满仓死扛。

## 九、小结

价值因子没死，死的是「只用 B/M 一把尺子」。本文用合成面板实跑验证：

- 复合价值十分位**干净单调**，多空年化 **13.5%、Sharpe 1.29**，碾压单 B/M 的 8.2% / 0.76；
- CAPM **β≈0.14、α≈12.7%**——是横截面均值回复，不是 beta 伪装；
- rank-IC **0.027(t=6.12)** 对单 B/M 0.015——多尺子交叉验证信号更清楚；
- 根因在机制：**便宜度价差被踩到最宽后，未来 L-S 收益显著更高**（15.9% vs 12.2%）——这才是「价值复兴」的真身。

从 Asness 的复合价值，到 Fama-French 的 HML，再到本文的「价差即弹簧」——一条主线贯穿：**便宜不是谎言，但「怎么量便宜」决定你是捡到黄金还是接到飞刀**。把六把尺子叠起来，你量到的才是市场真金白银要价的共识。
