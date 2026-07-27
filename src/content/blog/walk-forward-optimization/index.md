---
title: "走向前优化 Walk-Forward：用滚动样本外把参数拟合的水分挤掉"
publishDate: '2026-07-27'
description: "走向前优化 Walk-Forward：用滚动样本外把参数拟合的水分挤掉 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

先说结论：**一次性在全部历史上调出来的"最优参数"，其回测收益里混着无法分离的拟合水分；Walk-Forward 优化（WFO）用「滚动调参 → 冻结参数 → 样本外实测 → 拼接净值」的流水线，把水分直接挤出来变成一个数**。本文的动量策略模拟里，样本内调参后的平均年化夏普是 1.13，冻结参数后的样本外平均只剩 0.39——Walk-Forward 效率 WFE = 0.35，意味着**样本内业绩里约三分之二是拟合出来的幻觉**。这个数字，全样本一次性优化永远不会告诉你。

## 一次性优化的原罪：参数见过答案

标准流程的问题出在时序上：你在 2016-2025 的完整数据上扫参数网格，挑出夏普最高的组合，然后报告这个组合在 2016-2025 的表现。**参数是看着全部十年的答案选出来的，再用同一个十年打分**——这不是预测能力的度量，是拟合能力的度量。

它和[数据窥探](/blog/white-reality-check/)是同一枚硬币的两面，但角度不同：SPA/WRC 这类检验问"挑出来的最优统计上是否显著"，Walk-Forward 问的是更工程化的问题——"**如果我历史上真的按这套流程操作，每个时点只用当时可得的数据调参，实际能赚到多少**"。前者是统计学答案，后者是可交易性答案。

## Walk-Forward 的机械结构

WFO 把时间轴切成一串重叠的"训练-验证"对：

![Walk-Forward 滚动窗口：样本内调参、样本外冻结实测](/images/walk-forward-optimization/wfo-rolling-scheme.jpg)

每一轮做三件事：

1. **样本内（IS）窗口**：跑参数网格，按目标函数（夏普、Calmar、净利润……）选出最优参数；
2. **样本外（OOS）窗口**：冻结这组参数，在紧接着的从未参与调参的时段实测；
3. **滚动**：窗口整体前移一个 OOS 长度，重复。

所有 OOS 段首尾相接，拼成一条完整的净值曲线。这条曲线的每一天，用的都是"当时可得数据调出的参数"——它是对真实操作流程的历史仿真，而不是对策略规则的历史仿真。这是 WFO 与普通回测最本质的区别：**普通回测检验规则，WFO 检验"规则+调参流程"这个完整系统**。

窗口有两种滚法：滚动窗（rolling，IS 长度固定，旧数据丢弃）适合认为市场结构会漂移的场景；锚定窗（anchored，IS 起点固定、越滚越长）适合认为数据越多估计越准的场景。没有普适答案，但对参数敏感型策略，滚动窗更诚实——它强迫你面对"旧 regime 的数据可能有毒"这个现实。

## 代码：完整的 Walk-Forward 流水线

策略是最朴素的时序动量：过去 `lb` 日累计收益为正则持有，信号次日生效。参数只有一个回看窗口，网格 7 个取值——故意选这么简单，是为了让"水分"无处可藏。

```python
import numpy as np

LOOKBACKS = [5, 10, 20, 40, 60, 90, 120]
IS_LEN, OOS_LEN = 504, 126        # 2 年样本内 + 半年样本外

def momo_returns(ret, lb):
    """过去 lb 日累计收益>0 则持有；signal-on-i, execute-on-i+1"""
    cum = np.convolve(ret, np.ones(lb), "full")[:len(ret)]
    pos = (cum > 0).astype(float)
    sig = np.zeros_like(ret)
    sig[1:] = pos[:-1]            # 次日生效，杜绝 look-ahead
    sig[:lb + 1] = 0              # warmup 段不交易
    return sig * ret

def sharpe(x):
    return 0.0 if x.std() == 0 else float(x.mean() / x.std() * np.sqrt(252))

# 滚动切窗
folds, start = [], 0
while start + IS_LEN + OOS_LEN <= len(ret):
    folds.append((start, start + IS_LEN, start + IS_LEN + OOS_LEN))
    start += OOS_LEN

is_sharpes, oos_sharpes, best_lbs, oos_concat = [], [], [], []
for a, b, c in folds:
    # 1) 样本内：网格选参
    scores = [sharpe(momo_returns(ret[a:b], lb)) for lb in LOOKBACKS]
    k = int(np.argmax(scores))
    best_lbs.append(LOOKBACKS[k]); is_sharpes.append(scores[k])
    # 2) 样本外：冻结参数实测（信号计算带 lookback 缓冲，收益只取 OOS 段）
    buf = LOOKBACKS[k] + 2
    seg = ret[max(0, b - buf):c]
    strat = momo_returns(seg, LOOKBACKS[k])[-(c - b):]
    oos_sharpes.append(sharpe(strat)); oos_concat.append(strat)

oos_all = np.concatenate(oos_concat)
WFE = np.mean(oos_sharpes) / np.mean(is_sharpes)   # Walk-Forward 效率
```

两个容易写错的细节。**OOS 段的信号缓冲**：计算 OOS 第一天的信号需要之前 `lb` 天的收益，这些天来自 IS 段末尾——用它们算信号不是泄漏（信号只用历史），但收益统计必须严格从 OOS 第一天开始，`[-(c-b):]` 切片保证了这一点。**warmup 不计入**：每段前 `lb+1` 天强制空仓，避免半截信号污染统计——这与我们回测框架里 warmup 切片的铁律一脉相承。

## 实验：WFE = 0.35，三分之二是水

数据是 10 年模拟日频收益，故意埋了一个 regime 切换：前 5 年趋势市（动量自相关强），后 5 年震荡市（动量几乎消失）。这不是刁难策略，而是复刻真实市场的常态——动量因子在 2010 年代的衰减就是这个剧本。

16 轮 Walk-Forward 的结果：

![每轮样本内 vs 样本外夏普：WFE = 0.35](/images/walk-forward-optimization/wfo-is-vs-oos.jpg)

三个观察。**第一，IS 夏普系统性虚高**：16 轮平均 1.13，每一轮都是从 7 个候选里挑最大——哪怕动量毫无预测力，"7 选 1"的极值效应也会把 IS 夏普抬高一截。**第二，OOS 平均只剩 0.39**：WFE = 0.39/1.13 = 0.35，即样本内业绩只有约三分之一能活到样本外。经验上 WFE > 0.5 算健康，0.3-0.5 灰色地带，< 0.3 基本可以断定过拟合主导。**第三，OOS 夏普的逐轮波动巨大**：从 -1 到 +2.5 都有，后半段（震荡市 regime）明显恶化——单轮 OOS 说明不了任何事，必须看全部轮次的分布。

### 参数漂移：不稳定本身就是判决

每轮 IS 选出的最优回看窗口序列是：10, 10, 10, 10, 5, 5, 5, 40, 5, 5, 5, 5, 5, 5, 10, 40。

![每轮最优参数跳来跳去：参数不稳定的直接证据](/images/walk-forward-optimization/wfo-param-drift.jpg)

前四轮稳定在 10 日，之后在 5 和 40 之间反复横跳。参数漂移图是 WFO 送的免费诊断：**如果"最优参数"每滚一轮就换一个数量级，说明目标函数曲面平坦、极值位置由噪声决定，任何单点参数都不值得信任**。稳定的参数序列（比如始终落在 20-40 区间内）才是策略结构稳健的信号。这个诊断和[贝叶斯优化调参](/blog/bayesian-optimization-hyperparam/)末尾提的"walk-forward 看参数稳定性"是同一件事的完整展开。

### 三条净值曲线：哪条能交易？

![全样本优化 vs Walk-Forward OOS vs 买入持有](/images/walk-forward-optimization/wfo-equity-compare.jpg)

蓝线是全样本一次性优化（lb=5，全样本夏普 0.78）——这条曲线最常出现在策略报告里，但它不可交易：2016 年的你不可能知道 lb=5 会是未来十年的全样本最优。红线是 Walk-Forward 拼接 OOS（整体夏普 0.52）——这才是"历史上真按这套流程操作"能得到的净值。灰线是买入持有。蓝红之间的差距，就是报告里看不见的拟合溢价。

值得注意的是拼接 OOS 夏普（0.52）高于逐轮 OOS 平均（0.39）：逐轮夏普对短窗口的波动估计不稳定，拼接后样本变长、估计更平滑。报告时两个口径都给，但以拼接口径为主。

## WFO 与其他防过拟合工具的关系

本站已经写过一族"回测可信度"工具，它们和 WFO 的分工：

- **WFO（本文）**：仿真完整的"调参+交易"流程，输出可交易口径的净值和 WFE。它是**工程视角**：不问显著性，问"按这个流程操作历史上能赚多少"。
- [CPCV](/blog/combinatorial-purged-cv/)：组合分块生成多条训练-测试路径，路径数远多于 WFO 的单条时间线，统计功效更高；但组合拼接打破了时间顺序，不再是"可操作流程"的仿真。两者互补：CPCV 评估策略族，WFO 仿真最终流程。
- [PBO](/blog/pbo-overfitting-probability/)：量化"IS 最优在 OOS 沦为平庸"的概率，和 WFE 讲同一件事但口径不同——PBO 是排名视角的概率，WFE 是业绩视角的比例。
- [Hansen SPA](/blog/hansen-spa-test/) / [WRC](/blog/white-reality-check/)：统计检验视角，回答"这堆参数里有没有真信号"。SPA 说"有"，WFO 再告诉你"按可操作流程能兑现多少"。

一条务实的流水线：网格扫描 → SPA 确认策略族里有真东西 → WFO 仿真可交易净值并检查 WFE 与参数稳定性 → [DSR](/blog/deflated-sharpe-ratio/) 报告最终显著性。四步全过再谈仓位。

## 使用清单

1. **OOS 长度先定，IS 长度次之**。OOS 太短则单轮夏普全是噪声，太长则参数陈旧。日频策略 OOS 取 3-6 个月、IS 取 OOS 的 3-5 倍是常见起点；关键是 OOS 段拼起来要覆盖至少一个完整牛熊。
2. **WFE 和参数漂移一起看**。WFE 高但参数每轮乱跳，可能只是运气好；WFE 中等但参数稳定，反而更可信。
3. **别用 WFO 的 OOS 结果再去调 WFO 的超参**。IS/OOS 长度、滚动步长、目标函数——这些"元参数"如果按 OOS 表现反复调整，OOS 就退化成了另一个 IS。元参数应当按先验定死，或者最多在一个完全隔离的早期样本上定。
4. **每轮 OOS 都要过成本**。滚动调参意味着参数切换，切换意味着换仓成本。参数漂移越剧烈，这笔隐性成本越高——这也是偏好稳定参数的又一个理由。

## 局限

WFO 不是免罪金牌。第一，**它只有一条时间路径**：16 轮 OOS 彼此不独立（滚动窗口重叠），有效样本远少于表面轮次，WFE 本身的估计误差不小——这正是 CPCV 用组合路径要解决的问题。第二，**regime 依赖无法消除**：滚动调参默认"近期最优参数在近未来仍近似最优"，在 regime 突变点上这个假设恰好失效，我们实验里震荡市段的 OOS 恶化就是证据；[HMM](/blog/hmm-market-regime/) 一类的 regime 识别可以部分缓解但引入新参数。第三，**元参数的隐性搜索**：文献和实务里 IS/OOS 长度的"惯例值"本身就是几十年集体数据窥探的产物，对此保持清醒。第四，WFE 是描述性指标，没有显著性——0.35 和 0.45 的差异可能纯属噪声，要下统计结论还得回到 SPA/DSR 的框架。

把话说穿：WFO 的价值不在于它给出的数字更准，而在于它强迫你的回测流程和你的交易流程**同构**——你报告的每一分钱，都来自"当时不知道未来"的决策。这是回测能做到的诚实上限。

## 参考文献

1. Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.
2. Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39-69.
3. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Ch. 11-12)
4. White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126.
