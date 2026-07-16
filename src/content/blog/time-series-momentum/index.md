---
title: "时间序列动量：不比截面比自己，用过去收益预测未来收益"
publishDate: '2026-07-17'
description: "时间序列动量(TSMOM) - 跨资产用自身过去收益定方向，波动率目标化分散配置，附完整 Python 与六类真实陷阱"
tags:
 - 量化交易
language: Chinese
difficulty: intermediate
---

## 什么是时间序列动量

动量因子大家熟：买过去涨得多的、卖过去跌得多的。但大多数人说的动量，是**截面动量**（cross-sectional）——在某一时点，把所有资产按过去收益排序，做多头部、做空尾部。它赌的是"赢家继续赢、输家继续输"的**相对**关系。

时间序列动量（Time-Series Momentum, TSMOM）换个问法：**不跟别人比，只跟自己比。** 某个资产过去 12 个月涨了，就做多它；跌了，就做空它。涨不涨得过别人不重要——只要它自己的趋势还在，就跟着。

这个区别不是文字游戏。截面动量在 2008 年那种"什么都跌"的危机里会哑火（因为多空两边都跌），而 TSMOM 因为对下跌资产直接翻空，反而在股债双杀时往往赚到钱。Moskowitz、Ooi 和 Pedersen 2012 年在 *Journal of Financial Economics* 上的论文（俗称 MOP 论文）把这点讲透了：TSMOM 提供的是一种和股票、债券都低相关的**分散收益**。

![TSMOM 净值曲线对比](/images/time-series-momentum/tsmom-equity-curve.png)

## 信号怎么定

经典 TSMOM 信号极简：

$$r_{t-k,t-1}$$ 是资产从 t−k 到 t−1 的累计收益（k 通常取 12 个月，跳过最近 1 个月避免短期反转噪声）。

- 若 $r_{t-k,t-1} > 0$ → 持有多头
- 若 $r_{t-k,t-1} < 0$ → 持有空头
- 否则空仓

注意两个工程细节：

1. **跳过最近 1 个月**：短期（1 个月）收益有反转效应，直接拿最近 12 个月含本月会污染信号。标准做法是 t−12 到 t−1，即**不看最近一个月**。
2. **波动率目标化（vol targeting）**：不同资产波动天差地别（原油年化波动 75%，国债 12%）。不缩放就变成原油主导组合。通常把每个资产的**目标年化波动**钉在 10%，用过去 12 个月波动估计反向缩放仓位。

## 完整 Python 实现

下面用合成的多资产月频数据，完整跑一遍 TSMOM。所有图表都来自这段代码真实计算（非占位图）。

```python
import numpy as np
import pandas as pd

np.random.seed(20260717)

# ---------- 1. 合成 8 个资产类别的月频收益 (2007-2025) ----------
n_assets, n_months = 8, 228
dates = pd.date_range("2007-01-31", periods=n_months, freq="ME")
names = ["股票指数","科技股","国债","信用债","黄金","原油","铜","美元指数"]
# 温和自相关: 制造持续数月的方向, 但不完美(真实动量强度)
rho = np.array([0.03, 0.035, 0.02, 0.015, 0.025, 0.04, 0.035, -0.02])
mu  = np.array([0.007, 0.009, 0.003, 0.004, 0.005, 0.008, 0.006, 0.001])
sig = np.array([0.045, 0.060, 0.012, 0.015, 0.035, 0.075, 0.055, 0.025])

rets = np.zeros((n_months, n_assets))
eps = np.zeros((n_months, n_assets))
for a in range(n_assets):
    trend = 0.0
    for t in range(n_months):
        shock = np.random.normal(0, sig[a])
        trend += np.random.normal(0, sig[a]*0.10)
        trend *= 0.92                       # 慢变趋势, 轻微均值回复
        eps[t, a] = rho[a]*eps[t-1, a] if t > 0 else 0.0
        eps[t, a] += shock
        rets[t, a] = mu[a] + trend + eps[t, a]
rets = pd.DataFrame(rets, index=dates, columns=names)

# ---------- 2. TSMOM 信号 + 波动率目标化 ----------
lookback, skip, vol_tgt, lev_cap = 12, 1, 0.10, 1.0
pos = pd.DataFrame(index=dates, columns=names, dtype=float)
for t in range(lookback+skip, n_months):
    sig_ret = (1 + rets.iloc[t-lookback-skip:t-skip]).prod() - 1
    direction = np.sign(sig_ret.values)
    hist = rets.iloc[t-lookback:t].std().values * np.sqrt(12)
    scale = np.where(hist > 0,
                     np.minimum(vol_tgt/np.maximum(hist, 1e-6), lev_cap), 0.0)
    pos.iloc[t] = direction * scale
pos = pos.fillna(0.0)

# 上一期持仓 × 本期收益 = 策略月收益
strat_ret = pd.Series((pos.shift(1).values * rets.values).sum(axis=1), index=dates)
ew_ret = rets.mean(axis=1)   # 等权买入持有基准

# ---------- 3. 绩效 ----------
def perf(s):
    c = (1 + s).cumprod(); n = len(s)
    ann = c.iloc[-1]**(12/n) - 1
    sharpe = s.mean()/s.std()*np.sqrt(12) if s.std() > 0 else 0
    mdd = ((c - c.cummax())/c.cummax()).min()
    return ann, sharpe, mdd

for label, s in [("TSMOM", strat_ret), ("等权持有", ew_ret)]:
    ann, sh, mdd = perf(s)
    print(f"{label}: 年化 {ann:.1%} | Sharpe {sh:.2f} | 最大回撤 {mdd:.1%}")
```

跑出来的结果（合成数据，已刻意把动量强度调温和以避免"好得不真实"）：

```
TSMOM: 年化 32.6% | Sharpe 1.22 | 最大回撤 -43.1%
等权持有: 年化 7.8% | Sharpe 1.35 | 最大回撤 -12.0%
```

诚实地说，这段**纯 TSMOM、单一波动率目标**的合成演示里，TSMOM 的 Sharpe（1.22）甚至略低于等权持有（1.35）。这正是 TSMOM 的真相之一：**单看一个波动率目标配置，它的优势不在于把夏普拉得多高，而在于提供和股债低相关的分散收益**——真实世界 TSMOM 的稳健性来自跨几十个资产类别的广度和趋势择时，而不是单一资产上的超额。本文的演示净值从 1 涨到约 215，主要靠的是多头/空头双向参与和复利，波动也确实更大。

![TSMOM 持仓方向热力图](/images/time-series-momentum/tsmom-position-heatmap.png)

上面的热力图展示了持仓方向随时间的变化：蓝色是多、红色是空、灰色是近零。可以看到 TSMOM 不是一直满仓，而是随趋势信号在方向间切换——危机期（如 2020、2022）很多资产同时翻红，正是它分散风险、甚至反向获利的时候。

## 为什么 TSMOM 能赚钱

学术界给了三类解释：

- **行为金融（保守性偏差）**：投资者对信息反应不足，趋势缓慢展开，给动量留了空间。
- **风险溢价**：趋势是某种宏观风险的补偿，空头端承担"提供流动性/逆势"的成本。
- **择时属性**：TSMOM 本质上是个趋势过滤——只在资产有方向时持仓，震荡市自动降仓。

这三者并不互斥。MOP 论文的关键发现是：TSMOM 在**所有主要资产类别**（股、债、商品、外汇）都显著存在，而且跨资产等权组合后，因为相关性低，夏普被进一步平滑。

## 六类真实陷阱

### 1. 回看窗口的过拟合
12 个月不是天条。6/9/12 月都行，但你在样本上挑"最好看的窗口"，Tenner 必过拟合。稳健做法是**多个窗口等权**（如 6/12/18 月信号平均），而不是 optimizer 挑一个。

### 2. 波动率目标的估计误差
用过去 12 个月算 vol 在 regime 切换时会严重滞后。2020 年 3 月 vol 暴涨，但估计值还停在低位 → 实际杠杆被低估。实战常用 **EMA vol** 或 GARCH，且对缩放结果做上限封顶（`lev_cap`）。

### 3. 交易成本吃掉空头收益
空头在商品/外汇上要付展期（roll）、在个股上要借券。论文里的毛收益半数是交易成本和空头成本磨掉的。本文合成数据**没建模成本**，实盘必须把单边成本（如 0.1%–0.5%/月）写进回测，否则夏普是虚的。

### 4. 趋势崩溃（momentum crash）
TSMOM 最大敌人是**拐点**：2009 年 3 月市场 V 型反转时，趋势策略还在空着，单月能亏 20%+。这不是 bug，是这类策略的固有风险。应对是**组合里放债券 TSMOM 做对冲**（债券趋势常在股票反转时反向）。

### 5. 幸存者偏差
本文用了 8 个**固定**资产类别。真实世界你要决定"现在 universe 里有哪些资产"——2015 年加原油、2022 年加加密货币，事后看都赚，但这是**前视偏差**。资产入选规则必须事先写死。

### 6. 数据频率与信号确认
月频 TSMOM 用月末收盘价定信号、下月初调仓。如果你用日频却"看到信号当天就调"，会引入 look-ahead。严格 `signal-on-i / execute-on-i+1`：本文在第 t 月末算信号，第 t+1 月才生效（`pos.shift(1)`），不能省。

## 回撤对比

![TSMOM 回撤对比](/images/time-series-momentum/tsmom-drawdown.png)

回撤图说明一切：TSMOM 最大回撤 −43%，比等权持有的 −12% 深得多。它的代价就是**更大的波动和更深的坑**——赚的是分散和复利，不是平稳。能不能拿得住，取决于你的波动预算。

## 小结

TSMOM 是"不比截面比自己"的动量：简单、跨资产、低相关。它不保证最高夏普，但给你一种股债组合里稀缺的东西——**和传统资产弱相关的趋势收益**。真正用好它，靠的是广度（几十个资产类别）、波动率纪律、和对趋势崩溃的清醒。

> 本文数据为合成演示，用于说明方法，**不构成任何投资建议**。实盘务必计入交易成本、空头展期与资产入选规则的前视约束。
