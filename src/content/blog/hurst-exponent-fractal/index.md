---
title: "Hurst 指数与分形市场假说：用重标极差判断均值复归还是趋势"
publishDate: '2026-07-11'
description: "有效市场认为价格是随机游走，分形市场假说指出市场有记忆。Hurst 指数把持久性量化成 H>0.5 趋势 / H<0.5 均值回复 / H=0.5 随机游走。本文用可复现代码讲清 R/S 重标极差与滚动 Hurst 检测机制切换，并点明经典 R/S 的有限样本偏差。"
tags:
  - 量化交易
  - Hurst指数
  - 分形市场
  - 均值回复
  - 趋势跟踪
  - Python
language: Chinese
difficulty: advanced
---

如果你相信有效市场假说（EMH），那么价格序列就是随机游走——过去涨不涨，和明天涨不涨没关系。但做交易的人都有一种直觉：**有些行情"涨了还会涨"（趋势），有些"涨多了就跌回来"（均值回复）**。这两种走势背后的数学结构完全不同，而把它们区分开来的，是一个叫 **Hurst 指数**（赫斯特指数）的数。

分形市场假说（Peters, 1994）正是用 Hurst 指数来挑战 EMH 的：它认为资本市场不是简单的随机游走，而是一个有"记忆"、有分形结构的系统。本文不堆公式，而是用一套**可复现的模拟 + 真实计算的图表**，把三件事讲清楚：

1. Hurst 指数到底在量什么；
2. 怎么用 **R/S 重标极差法**算出来（附完整代码）；
3. 落地到策略时，哪些坑会让你"算出来是趋势，实盘却是均值回复"。

## 一、Hurst 指数：把"市场记忆"量化成一个数

Hurst 指数 $H$ 衡量一个时间序列的**持久性（persistence）**，取值范围 $(0,1)$：

| $H$ | 含义 | 直观 |
|---|---|---|
| $H = 0.5$ | 随机游走（无记忆） | 今天涨跌和明天无关，纯噪声 |
| $H > 0.5$ | **趋势型 / 持久**（persistent） | 涨了更容易接着涨，跌了更容易接着跌，序列有"惯性" |
| $H < 0.5$ | **均值回复型 / 反持久**（anti-persistent） | 涨了更容易跌回来，跌了更容易弹回去，序列"不爱出远门" |

注意一个关键但常被误解的点：**Hurst 指数描述的是"增量（收益）"的持久性，不是价格水平本身**。一个 $H=0.5$ 的随机游走，它的价格曲线看起来也"弯弯曲曲像有趋势"，但那只是因为你看了价格水平——它的逐日收益是无记忆的。所以估计 Hurst 时，我们要对**收益/差分序列**做，而不是对价格水平做。这一点后面用代码验证时会非常关键。

一个冷知识：自然界里 $H$ 到处都是。尼罗河流量（$H\approx0.9$，强持久）、气温（$H\approx0.6$）、以及一堆金融价格（多在 0.5 附近游走）。Peters 的著名主张是：**股票市场长期看 $H>0.5$**，即存在温和的趋势惯性——这和"动量因子能赚钱"在直觉上是一致的。

## 二、怎么算：R/S 重标极差法（Rescaled Range）

最经典的 Hurst 估计方法是 **R/S 重标极差法**，逻辑非常朴素：

1. 把序列切成长度为 $w$ 的若干个窗口；
2. 在每个窗口内，算累积偏差 $Z_t = \sum_{i=1}^{t}(X_i - \bar X)$，取其极差 $R = \max Z - \min Z$；
3. 再除以该窗口的标准差 $S$，得到重标极差 $R/S$；
4. 对所有窗口平均，得到该窗口长度下的 $(R/S)_w$；
5. 理论上 $(R/S)_w \sim w^H$，于是对 $\log w$ 和 $\log(R/S)$ 做线性回归，**斜率就是 Hurst 指数**。

下面是从零实现的完整代码。注意我们用 Davies–Harte 算法生成"指定 Hurst"的分数高斯噪声（fGn），再积分成价格序列，用来验证算法对不对。

```python
import numpy as np

# ---------- 1. Davies-Harte 生成指定 Hurst 的分数高斯噪声 fGn(H) ----------
def fgn(H, n, seed=0):
    rng = np.random.default_rng(seed)
    def acf(k):
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))
    M = 1
    while M < 2 * n:
        M *= 2
    g = np.zeros(M)
    g[0] = acf(0)
    for k in range(1, n + 1):
        g[k] = acf(k)
        g[M - k] = acf(k)
    lam = np.real(np.fft.rfft(g))
    lam = np.maximum(lam, 1e-12)
    half = M // 2 + 1
    W = np.sqrt(lam) * (rng.standard_normal(half) + 1j * rng.standard_normal(half))
    W[0] = np.sqrt(lam[0]) * rng.standard_normal() / np.sqrt(2)
    if M % 2 == 0:
        W[M // 2] = np.sqrt(lam[M // 2]) * rng.standard_normal() / np.sqrt(2)
    return np.fft.irfft(W, n=M)[:n]

def price(H, n, seed=0):
    p = np.cumsum(fgn(H, n, seed))
    return p - p[0]

# ---------- 2. 经典 R/S 重标极差 Hurst 估计（对增量/收益序列）----------
def hurst_rs(x, min_w=20, max_w=None):
    x = np.asarray(x, float)
    N = len(x)
    if max_w is None:
        max_w = N // 2
    ws = np.unique(np.linspace(min_w, max_w, 22).astype(int))
    rs = []
    for w in ws:
        n_w = N // w
        if n_w < 1:
            continue
        vals = []
        for i in range(n_w):
            seg = x[i * w:(i + 1) * w]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append(np.mean(vals))
    rs = np.array(rs)
    H, _ = np.polyfit(np.log(ws), np.log(rs), 1)
    return H

# ---------- 3. 三类合成序列验证 ----------
N = 1200
rw = price(0.50, N, seed=11)   # 随机游走
tr = price(0.75, N, seed=22)   # 趋势型
mr = price(0.25, N, seed=33)   # 均值回复型

print("随机游走 H =", round(hurst_rs(np.diff(rw)), 3))
print("趋势型   H =", round(hurst_rs(np.diff(tr)), 3))
print("均值回复 H =", round(hurst_rs(np.diff(mr)), 3))
```

跑出来的结果：

- 随机游走 $H \approx 0.43$（单次估计有偏差，后文解释）
- **趋势型 $H \approx 0.79$**
- **均值回复型 $H \approx 0.35$**

三类序列被干净地分开了。下面两张图把这个过程画出来——图 1 是价格曲线本身（肉眼其实很难区分），图 2 才是关键：把 $\log(R/S)$ 对 $\log(w)$ 画出来，**斜率就是 Hurst 指数**。

![三类价格序列：随机游走、趋势与均值回复的肉眼对比](/images/hurst-exponent-fractal/hurst_series.png)

![R/S 重标极差法：对收益序列估计，log(R/S) 对 log(w) 的斜率即 Hurst](/images/hurst-exponent-fractal/hurst_rs_scaling.png)

## 三、用滚动 Hurst 检测"市场状态切换"

单看一整段历史的 Hurst，只能给你一个"平均记忆强度"。但市场是时变的：**牛市早期可能有强趋势（H 高），震荡市里均值回复占主导（H 低）**。把整段历史当成一个固定的 $H$ 来用，等于假设市场机制不变——这显然不现实。

做法是做**滚动窗口**估计：每 200 个交易日（约一年）重新算一次 Hurst，得到一条随时间变化的 $H(t)$ 曲线。下面我们构造一段"趋势市 → 震荡市 → 趋势市"切换的序列来演示：

```python
seg = 600
s1 = price(0.72, seg, seed=101)   # 趋势市
s2 = price(0.30, seg, seed=102)   # 震荡/均值回复市
s3 = price(0.68, seg, seed=103)   # 又转趋势
regime = np.concatenate([s1, s2, s3])
regime -= regime[0]
ret = np.diff(regime)                # 对收益序列估计

win, step = 200, 20
roll, xr = [], []
for end in range(win, len(ret) + 1, step):
    roll.append(hurst_rs(ret[end - win:end]))
    xr.append(end)
roll = np.array(roll)
```

结果非常直观：Hurst 曲线在 $0.5$ 上下波动，当机制进入趋势市时爬到 $0.6$ 以上、进入震荡市时掉到 $0.4$ 以下。**这就是 Hurst 最实用的地方——当一个 regime 过滤器**：

- $H > 0.6$：市场有记忆、趋势可乘，给趋势跟踪策略"开绿灯"；
- $H < 0.4$：市场爱回头，均值回复策略更合适；
- $H \approx 0.5$：接近随机游走，干脆别猜方向，做做市/套利更稳。

![滚动 Hurst 检测市场状态切换：趋势市>0.6，均值回复市<0.4](/images/hurst-exponent-fractal/hurst_rolling.png)

## 四、别迷信点估计：经典 R/S 的有限样本偏差

到这里你可能会想："那我每天算个 Hurst 不就行了？"——**不行，而且这是新手最容易栽的坑**。

经典 R/S 估计在小样本、短窗口上**系统性偏高**：它会把本来是随机游走的东西估成略大于 0.5。我们做一个实验——生成 500 条**真随机游走**，各自用 R/S 估一次 Hurst：

```python
H_vals = []
for k in range(500):
    x = np.cumsum(np.random.default_rng(1000 + k).standard_normal(1000))
    H_vals.append(hurst_rs(np.diff(x)))
H_vals = np.array(H_vals)
print("500 条随机游走 Hurst 均值 =", round(H_vals.mean(), 3))
print("95% 区间 =", np.percentile(H_vals, [2.5, 97.5]).round(3))
```

结果：500 条**理论 H 严格等于 0.5** 的随机游走，估计均值约 **0.546**，95% 区间是 **[0.435, 0.669]**。也就是说，**你算出一个 0.55 的 Hurst，很可能什么都不代表**——它完全可以来自纯噪声。

这带来两个实践要点：

1. **永远配置信带**：别报单点 $H=0.57$ 就下结论，先看它是否显著落在 0.5 的置信区间之外。
2. **交叉验证估计量**：经典 R/S 有偏差，可以用 **Anis–Lloyd 修正 R/S**（扣掉小样本期望）、**方差时间法（Variance-Time）**、**周期图/Whittle 估计**互相印证。多个估计量一致说"趋势"，你才敢信。

![随机游走的 Hurst 估计分布：经典 R/S 存在有限样本上偏，需置信带而非点估计](/images/hurst-exponent-fractal/hurst_histogram.png)

## 五、落地到 A 股与你的策略

真实数据怎么接？以 A 股日线为例（用 `westock-data` 取复权收盘价，或 `tushare`）：

```python
# 伪代码：把真实行情接进上面的 hurst_rs
import pandas as pd
# close = 复权收盘价（务必用复权，否则分红除权会人为制造“下跌”伪信号）
# ret = np.diff(np.log(close.values))   # 对数收益
# H = hurst_rs(ret)
```

更实用的，是把 Hurst 当成**策略的开关/权重**，而不是一个孤立指标：

- **趋势 + 均值回复双轨**：用滚动 Hurst 决定当前主策略。H 高时上趋势跟踪（如突破、动量），H 低时切均值回复（如布林带回归、配对）。
- **和均线/布林带结合**：Hurst 告诉你"现在该用哪类策略"，具体进出场仍交给经典信号，避免 Hurst 本身的噪声。
- **多周期对齐**：日线、周线、月线的 Hurst 常常不一致。日线 H<0.5（日内噪音大、易回复）但月线 H>0.5（长期趋势）是很常见的，别拿一个周期的结论套另一个周期。

## 六、五大真实陷阱

1. **前视偏差（最致命）**：用"包含明天"的整段数据算出的 Hurst 去决定"今天"的交易，等于偷看了未来。Hurst 只能用在**截至昨日**的数据窗口，且要 Rolling，不能全局一次性算完再回测。
2. **对价格水平直接算**：正如前文强调，对价格水平（而非收益）算 R/S 会得到系统性偏高的 $H$（接近 1.0），错误地"什么都像趋势"。务必 `np.diff()` 之后再算。
3. **窗口长度选择偏差**：短窗口噪声大、长窗口样本少。建议用对数等距的多个窗口做 log-log 拟合，并和修正 R/S 交叉验证，而不是拍脑袋选一个 $w$。
4. **复权与停牌**：A 股长期停牌后复牌、不复权价格里的除权缺口，都会在收益序列里制造人造的"跳跃"，污染 Hurst。必须用复权价，并清理异常跳空。
5. **平稳性幻觉**：Hurst 假设增量结构在时间上稳定。但市场会 regime 切换（见第三节），所以**滚动 Hurst 比全局 Hurst 靠谱得多**，而且窗口不宜过长，否则会把不同机制搅成一锅粥。

## 结语

Hurst 指数不是"预测涨跌的魔法数字"，而是一个**帮你判断当前市场性格的诊断工具**：它量化了序列到底有没有记忆、是趋势还是均值回复。它的价值在于——**在正确的市场状态里，用正确的策略**。

但记住那张 500 条随机游走的分布图：点估计充满噪声，置信带和多个估计量的交叉验证缺一不可。把 Hurst 当 regime 过滤器、配合严格的前视隔离和滚动窗口，它才真正有用；把它当"算个数就能梭哈"的信号，它只会给你一个穿了随机游走外衣的伪趋势。
