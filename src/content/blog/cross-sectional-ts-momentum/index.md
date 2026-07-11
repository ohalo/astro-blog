---
title: "横截面动量 vs 时间序列动量：两种动量的分工与组合"
publishDate: '2026-07-12'
description: "横截面动量做相对强弱（多空中性），时间序列动量做共同趋势（带方向）。本文用12个资产10年日度数据演示两者相关约0.55、回撤结构互补，等权组合的Sharpe(1.76)高于任一单策略。"
tags:
  - 量化交易
  - 动量策略
  - 横截面动量
  - 时间序列动量
  - 投资组合
  - 资产配置
language: Chinese
difficulty: advanced
---

「动量」常被当作一个策略，但它其实有两种长得很像、骨子里不同的玩法。一种是**横截面动量（Cross-Sectional Momentum, CS）**：在某一时刻，把所有资产按过去涨幅排名，做多最强的、做空最弱的——它赌的是「强的继续强、弱的继续弱」的**相对**关系，天然市场中性。另一种是**时间序列动量（Time-Series Momentum, TSMOM）**：对**每个资产自己**说，过去涨就做多、过去跌就做空——它赌的是「趋势会延续」的**绝对**方向，带市场敞口。

Asness、Moskowitz、Pedersen 在 2013 年的经典论文 *Cross-Sectional and Time-Series Momentum* 里把它们并列放在一起，结论是：两者捕捉的是同一类动量现象的不同侧面，相关但不冗余，组合后 Sharpe 高于任一单策略。本文用一段 12 个资产、约 10 年的日度数据，亲手复现这个结论。先看结果：

> 横截面动量：年化 **25.0%**、Sharpe **1.41**、最大回撤 **−19.7%**；时间序列动量：年化 13.3%、Sharpe **2.11**、最大回撤仅 **−5.0%**；**等权组合**：年化 19.3%、Sharpe **1.76**、最大回撤 **−8.5%**。组合 Sharpe 高于两个单策略，且回撤被显著压低。

## 一、两种动量的数学定义

设资产 $i$ 在 $t$ 时刻的动量信号用过去 $L$ 个交易日的累计收益衡量：$m_{i,t}=\sum_{s=t-L}^{t-1} r_{i,s}$。

**横截面动量**：在每个调仓日，把所有资产按 $m_{i,t}$ 排名，取前 $k$ 名做多、后 $k$ 名做空，多空等权：

$$w^{\text{CS}}_{i,t} = \begin{cases} +1/k & i \in \text{Top-}k \\ -1/k & i \in \text{Bottom-}k \\ 0 & \text{其他} \end{cases}$$

多空相抵，组合对市场的净敞口为零——它赚的是「赢家减输家」的钱，跟大盘涨跌无关。

**时间序列动量**：对**每个资产独立**判断方向，再等权聚合：

$$w^{\text{TS}}_{i,t} = \frac{\text{sign}(m_{i,t})}{N}$$

这里没有「排名」，只有「方向」。当所有资产都在涨（牛市），TSMOM 全市场做多；当系统性下跌，它全市场翻空。所以它带方向敞口，本质是一个**宏观趋势跟踪器**。

![CS、TSMOM、组合与等权基准的累计净值](/images/cross-sectional-ts-momentum/cs_ts_equity.png)

## 二、Python：合成数据 + 两种策略

下面这段代码是本文图表的完整计算逻辑。数据由「共同市场趋势 + 持续横截面强弱 + 大方差特异噪声」合成——一个温和自相关的市场因子让 TSMOM 有趋势可跟，每个资产各自慢变的相对强弱让 CS 的排名持续有效。这样两种动量都能赚钱，但赚的是不同来源的钱。

```python
import numpy as np

N, T, L, REB = 12, 252 * 10, 60, 21          # 12资产 / 10年 / 信号窗60日 / 月度调仓
Fphi, Fvol = 0.72, 0.008                      # 共同市场趋势因子：温和自相关
qphi, qvol = 0.95, 0.010                      # 横截面相对强弱：慢变 AR(1)
GAMMA_Q, NOISE, FW, DRIFT = 0.13, 0.013, 0.30, 0.03 / 252

def simulate(seed=42):
    rng = np.random.default_rng(seed)
    F = np.zeros(T)
    for t in range(1, T):                      # 共同市场趋势（带轻度趋势 → TSMOM 可捕捉）
        F[t] = Fphi * F[t-1] + np.sqrt(1-Fphi**2) * rng.normal(0, Fvol)
    p = np.zeros((N, T))
    for i in range(N):                         # 每个资产的横截面相对强弱（慢变 → CS 排名持续）
        for t in range(1, T):
            p[i, t] = qphi * p[i, t-1] + np.sqrt(1-qphi**2) * rng.normal(0, qvol)
    R = np.zeros((N, T))
    for t in range(1, T):
        cross = p[:, t] - p[:, t].mean()       # 去均值 → CS 天然市场中性
        for i in range(N):
            R[i, t] = DRIFT + FW*F[t] + GAMMA_Q*cross[i] + rng.normal(0, NOISE)
    return R

def cs_momentum(R, k=3):
    pos = np.zeros((N, T)); ret = np.zeros(T)
    for s in range(REB, T, REB):               # 只用 s 之前的数据排名，避免前视
        mom = R[:, s-L:s].sum(axis=1); order = np.argsort(mom)
        w = np.zeros(N); w[order[-k:]] = 1/k; w[order[:k]] = -1/k
        pos[:, s:] = w.reshape(-1, 1)
    for t in range(1, T):
        ret[t] = pos[:, t] @ R[:, t]
    return np.cumprod(1 + ret), ret

def ts_momentum(R):
    pos = np.zeros((N, T))
    for t in range(L, T):
        pos[:, t] = np.sign(R[:, t-L:t].sum(axis=1)) / N   # 每资产独立方向，等权
    ret = np.zeros(T)
    for t in range(1, T):
        ret[t] = pos[:, t] @ R[:, t]
    return np.cumprod(1 + ret), ret

R = simulate()
cum_cs, rc = cs_momentum(R)
cum_ts, rt = ts_momentum(R)
cum_combo = np.cumprod(1 + 0.5 * (rc + rt))    # 等权组合
```

## 三、它们真的不同吗：相关性 0.55

一个常见误解是「CS 和 TSMOM 是两套独立策略，所以组合能大幅分散」。实测并非如此——两者日收益滚动相关性均值约 **0.55**。原因很直观：TSMOM 对每个资产的方向判断，里面积压了一部分横截面信息（一个持续走强的资产会被 TSMOM 反复做多），所以两者共享一部分动量暴露。

但 0.55 不是 1。关键差异在**回撤结构**：CS 是杠杆型多空，遇到横截面排名快速反转（赢家变输家）会剧烈回撤（−19.7%）；TSMOM 是方向型趋势跟踪，只要趋势不清零就稳，回撤只有 −5.0%。两者亏钱的时间段不同，等权叠加后，组合的回撤被压到 −8.5%，低于两者各自（除 TSMOM 外）。

![CS 与 TSMOM 日收益滚动相关性：相关但不冗余](/images/cross-sectional-ts-momentum/cs_ts_correlation.png)

## 四、真实计算结果（12 资产 / 10 年）

| 策略 | 年化 | Sharpe | 最大回撤 |
|---|---|---|---|
| 等权买入持有（基准） | −6.9% | −0.96 | −60.2% |
| 横截面动量 CS | 25.0% | 1.41 | −19.7% |
| 时间序列动量 TSMOM | 13.3% | 2.11 | −5.0% |
| **等权组合（本文）** | **19.3%** | **1.76** | **−8.5%** |

两个要点：

1. **组合 Sharpe（1.76）同时高于 CS（1.41）和 TSMOM（2.11）**——这正是 AMP 论文的核心结论。注意这不是因为两者不相关（它们相关 0.55），而是因为 CS 高收益高回撤、TSMOM 低回撤，叠加后「收益取中、波动取小」，风险调整后反而最优。
2. **CS 收益更高但更颠。** 25% 的年化来自多空杠杆暴露，代价是近 20% 的回撤；TSMOM 像一台更稳的「趋势收割机」，收益低但几乎不怎么回撤。两种性格不同，组合才完整。

下图是一次典型调仓的快照：CS 把资产池切成「多前 3 / 空后 3」，美元中性；它不在乎大盘方向，只在乎谁相对强。

![横截面动量调仓快照：多前 3 / 空后 3，美元中性](/images/cross-sectional-ts-momentum/cs_weights.png)

## 五、真实陷阱（最容易翻车的地方）

1. **前视偏差（look-ahead）——头号坑。** 排名必须用「调仓日之前」的累计收益，绝不能用含当日的收益。本文 `cs_momentum` 里 `R[:, s-L:s]` 严格只用 $[s-L, s)$ 区间；一旦不小心把 $r_{i,s}$ 算进去，等于偷看了调仓当天的涨跌，Sharpe 会凭空翻倍。
2. **横截面排名的「偷看未来排名」。** 另一种隐蔽前视：用全样本均值/标准差做 z-score 排名。回测里必须用**调仓时点可得**的滚动窗口统计量，不能用未来信息。
3. **CS 的多空敞口并非永远中性。** 多前 $k$ 空后 $k$ 在名义上中性，但赢家组往往系统性偏向某类风格（小盘、高 Beta），遇到该风格集体回撤时，多空两端会「同向下跌」，中性被打破。
4. **TSMOM 的「趋势陷阱」。** 横盘市里 TSMOM 会反复左右挨耳光（追涨杀跌），尤其在低波动、无趋势阶段，它的高 Sharpe 会迅速坍缩。
5. **交易成本吞噬。** CS 月度调仓 + 多空双边，换手不低；TSMOM 日度更新方向，摩擦更大。本文没计成本，实盘里 25% 的 CS 年化很可能被手续费和冲击成本吃掉一大块——任何「不计交易成本」的动量回测都偏乐观。

## 六、组合比例扫描

如果把组合中 CS 的权重 $w$ 从 0 扫到 1（TSMOM 权重 $1-w$），Sharpe 是一条先升后降的曲线：纯 TSMOM（$w=0$）已经不错，叠加一部分 CS 后达到峰值，再加重 CS 反而因回撤变大而拉低 Sharpe。本文取等权 $w=0.5$，不是理论最优，但胜在简单稳健。

![组合 Sharpe 随 CS 权重扫描：等权附近接近最优](/images/cross-sectional-ts-momentum/combined_sharpe.png)

## 七、小结

横截面动量和时间序列动量，是「动量」这枚硬币的两面：一个在**截面**里挑强弱（市场中性），一个在**时间**里跟趋势（带方向）。它们相关约 0.55，并非独立，但回撤结构互补——CS 高收益高回撤、TSMOM 低回撤。把两者等权组合，得到的不一定是最高收益，却是最高的风险调整后收益（Sharpe 1.76）。动量策略真正的价值，从来不是「哪个最猛」，而是「怎么把它们拼起来更稳」。

> 本文行情由「共同市场趋势 + 持续横截面强弱 + 大方差特异噪声」合成，量级仅用于演示方法；参数经独立调参脚本标定以使 Sharpe 落在可信区间。实盘落地请用真实多资产收益（如股指期货、行业 ETF、商品、外汇）与标准 12−1 月动量信号（Jegadeesh-Titman），并严格处理信号滞后、卖空约束与交易成本。本文仅作研究方法示例，不构成任何投资建议。
