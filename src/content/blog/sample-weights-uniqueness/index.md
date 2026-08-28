---
title: "样本唯一性加权：重叠标签下的正确样本权重"
description: "上一篇 CUSUM 把采样点从 1430 砍到 236，但事件标签的区间重叠依然存在：1500 日模拟中活跃时点平均被 2.9 个标签同时覆盖，平均唯一性仅 0.339——每个样本携带的独立信息只有名义值的三分之一。AFML 第四章给出三件配套工具：concurrency 计数把重叠精确量化到每个时点；唯一性权重 + 收益归因权重让'独立且赚大钱'的样本主导训练；序列 Bootstrap 用动态更新的抽样概率替代均匀抽样，实测抽样集合唯一性 0.312→0.331。关键洞见：这不是可选的调参项，而是让 bagging 的方差缩减公式在金融数据上重新成立的前提——标准 bootstrap 在重叠标签下抽出的是'伪独立'样本，OOB 分数系统性虚高。A 股日频 + T+1 下持有期普遍更长，重叠更severe，加权更必要（中高阶）"
publishDate: '2026-07-30'
tags:
  - 量化交易
  - 金融机器学习
  - 样本加权
  - AFML
  - Python
language: Chinese
difficulty: "intermediate"
---

## 一句话版本

金融标签的生命周期互相重叠，重叠的部分不是新信息而是复读。样本唯一性加权做三件事：数清楚每个时点被几个标签覆盖（concurrency）、按「独立程度 × 收益贡献」给样本发言权、用序列 Bootstrap 让抽样天然偏向独立样本——三件事共同修复的是同一个地基：**IID 假设在金融训练集上从来就不成立**。

---

## CUSUM 之后，还剩一个没修完的洞

这个系列的上一站，CUSUM 过滤器把「每天都开样本」改成「偏离够大才开样本」，1430 个名义样本压缩到 236 个事件样本，平均唯一性从 0.051 抬到 0.314。

问题解决了吗？只解决了一半。

CUSUM 管的是**入口**：什么时候开样本。但每个样本开出来之后，还有一段**生命周期**——三重障碍标注下，标签要等到止盈、止损或超时三者先触发其一才结算。这段区间短则两三天，长则二十天。只要两个事件的间隔小于前一个标签的持有期，它们的生命周期就会重叠。

重叠意味着什么？两个标签共享同一段收益路径。市场在那几天的涨跌，同时进入了两个样本的结算。模型看到的是两个「不同」的训练样本，但其中一部分信息是同一段行情的复印件。

我在 1500 日三区制模拟（中段波动率 2.5 倍于两端）上跑了完整的 CUSUM + 三重障碍管线，量化这个重叠：

![标签区间重叠与并发数](/images/sample-weights-uniqueness/swu-concurrency.png)

上图每个红色阴影是一个标签从开仓到结算的生命周期——肉眼可见大量区间叠在一起。下图是精确计量：**活跃时点平均被 2.9 个标签同时覆盖**，高波动段（事件密集 + 持有期未同步缩短）局部并发数冲到 6 以上。

## Concurrency：把重叠量化到每个时点

López de Prado 在 AFML 第四章给的工具链从一个极简的计数开始。对每个时点 $t$，数一数有多少个标签的区间 $[t_{i,0}, t_{i,1}]$ 覆盖它：

$$c_t = \sum_{i} \mathbf{1}\{t_{i,0} \le t \le t_{i,1}\}$$

这个数字叫 concurrency（并发数）。$c_t = 1$ 表示时点 $t$ 的行情只被一个标签独享；$c_t = 5$ 表示这段行情被五个标签瓜分——每个标签从这个时点获得的「独立信息份额」只有 1/5。

代码只有几行：

```python
import numpy as np
import pandas as pd

def get_concurrency(close_index, t_events, t1):
    """每个时点被多少个标签区间覆盖
    close_index: 全部交易日索引
    t_events:    事件开始时点（CUSUM 触发日）
    t1:          每个事件的结算时点（三重障碍出场日）
    """
    conc = pd.Series(0, index=close_index)
    for t0, t_end in zip(t_events, t1):
        conc.loc[t0:t_end] += 1
    return conc
```

有了 $c_t$，每个标签 $i$ 的**平均唯一性**就是它生命周期内独立份额的均值：

$$\bar{u}_i = \frac{1}{t_{i,1} - t_{i,0} + 1} \sum_{t=t_{i,0}}^{t_{i,1}} \frac{1}{c_t}$$

$\bar{u}_i = 1$ 表示这个标签的整段行情没有任何人分享，是完全独立的观测；$\bar{u}_i = 0.2$ 表示平均每个时点有 5 个标签在瓜分信息。

```python
def avg_uniqueness(t_events, t1, conc):
    out = []
    for t0, t_end in zip(t_events, t1):
        out.append((1.0 / conc.loc[t0:t_end]).mean())
    return pd.Series(out, index=t_events)
```

模拟中 236 个事件标签的唯一性分布：

![唯一性分布与收益归因权重](/images/sample-weights-uniqueness/swu-uniqueness-weights.png)

左图：**平均唯一性 0.339**，离「完全独立 = 1.0」的理想状态差着两倍。换算成有效独立样本量 $\sum_i \bar{u}_i \approx 76.6$——236 个名义样本，独立信息只值 77 个。注意这和上一篇 CUSUM 文章里「1430 个逐日样本有效样本量 72.5」形成了一个有教育意义的对照：**CUSUM 没有增加独立信息（市场里就那么多），它做的是用 1/6 的样本量把同样的独立信息装进更干净的容器**。而唯一性加权处理的是容器内残余的重叠。

## 两种权重：独立性 × 收益贡献

知道了每个标签的唯一性，最直接的用法是把它当作样本权重喂给模型（`sklearn` 的 `fit(X, y, sample_weight=w)`）。但 AFML 建议再乘上一层**收益归因**：

$$w_i = \left| \sum_{t=t_{i,0}}^{t_{i,1}} \frac{r_t}{c_t} \right|$$

分子是标签期间的对数收益，除以 $c_t$ 表示只认领属于自己的那一份。直觉是双重的：

1. **独立的样本多发言**——它带来的信息没有被复读；
2. **大行情的样本多发言**——一个在 ±0.5% 内磨出来的标签和一个 ±8% 大趋势结算的标签，后者携带的经济信号强度完全不同，标签符号相同不代表信息量相同。

上面右图的散点显示两种权重信息互补：唯一性高的样本收益归因权重不一定高（独立的小行情），收益归因权重高的样本唯一性可能一般（大行情往往事件密集）。乘起来之后，训练集的发言权集中到「独立且赚大钱/亏大钱」的样本——这正是你希望模型认真学习的那部分。

```python
def return_attribution_weights(t_events, t1, ret, conc):
    """收益归因权重：|sum(r_t / c_t)|"""
    out = []
    for t0, t_end in zip(t_events, t1):
        r_slice = ret.loc[t0:t_end]
        c_slice = conc.loc[t0:t_end]
        out.append(abs((r_slice / c_slice).sum()))
    w = pd.Series(out, index=t_events)
    return w * len(w) / w.sum()   # 归一化到均值 1
```

## 为什么这不是可选项：bagging 的方差公式坏了

到这里可能有人觉得：加权嘛，锦上添花的调参项。不是的。**样本重叠直接破坏了随机森林 / bagging 的理论根基**。

Bagging 的方差缩减公式：

$$\text{Var}\left[\frac{1}{B}\sum_b \hat{f}_b(x)\right] = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

其中 $\rho$ 是基学习器之间的相关性。这个公式起效的前提是 bootstrap 抽样能产生「足够不同」的训练子集。但在重叠标签下，标准 bootstrap 均匀地有放回抽样，抽出来的样本大概率彼此重叠——**每棵树看到的都是那几段被复印过的行情**，$\rho$ 被推高，方差缩减名存实亡。更隐蔽的后果：out-of-bag 样本与 in-bag 样本共享收益路径，**OOB 分数系统性虚高**，你以为的「免费验证集」在给你报喜。

## 序列 Bootstrap：让抽样自己偏向独立样本

AFML 的第三件工具直接改抽样机制。序列 Bootstrap（Sequential Bootstrap）的规则：

1. 第一次抽样，所有样本等概率;
2. 从第二次开始，每个候选样本的抽样概率正比于它**相对于已抽中集合的条件唯一性**——如果候选样本与已抽中的样本重叠严重，概率被压低；与已抽集合完全不重叠的样本概率最高;
3. 重复直到抽满。

```python
def seq_bootstrap(ind_matrix, size, rng):
    """ind_matrix: (T, N) 指示矩阵, ind[t, i]=1 表示标签 i 覆盖时点 t"""
    T, N = ind_matrix.shape
    phi = []
    for _ in range(size):
        c_prev = ind_matrix[:, phi].sum(axis=1) if phi else np.zeros(T)
        avg_u = np.zeros(N)
        for k in range(N):
            span = ind_matrix[:, k] > 0
            # 相对已抽集合的条件唯一性
            avg_u[k] = (1.0 / (1.0 + c_prev[span])).mean()
        prob = avg_u / avg_u.sum()
        phi.append(rng.choice(N, p=prob))
    return phi
```

在 236 个事件标签上对比两种抽样（各抽 236 个、有放回）：

![标准 vs 序列 Bootstrap](/images/sample-weights-uniqueness/swu-seq-bootstrap.png)

抽样集合的平均唯一性从 **0.312 提到 0.331（+6%）**。提升看起来不大？两个原因：其一，CUSUM 已经在上游干掉了最严重的重叠，剩余重叠本来就低（这是好事，说明管线各层分工正确）；其二，AFML 自己的蒙特卡洛实验也是同一量级——序列 Bootstrap 的价值随重叠程度上升，**在逐日开样本、长持有期的旧式管线上提升可以到 2-3 倍**。它是保险，不是魔法。

代价要诚实说：朴素实现是 $O(\text{size} \times N \times T)$，几百个样本没问题，几万个样本必须用 AFML 附录里的稀疏矩阵加速或直接预计算指示矩阵的累计和。

## 三件工具在管线里的位置

到这一篇，AFML 前四章的数据端管线已经完整：

```
信息驱动 bar（成交量/美元/失衡）      ← 数据结构层：让收益分布回归正态
    → CUSUM 事件过滤                 ← 采样层：偏离够大才开样本
    → 三重障碍标注                    ← 标注层：路径依赖的诚实标签
    → concurrency + 唯一性/归因权重    ← 加权层：残余重叠的校正（本篇）
    → 序列 Bootstrap + bagging        ← 训练层：抽样机制适配非 IID
```

每一层修一个 IID 假设的破口。跳过任何一层，下游的交叉验证分数、特征重要性、OOB 估计都会带着对应的偏差——而且是**偏乐观**的偏差，这正是金融 ML「回测惊艳、实盘拉胯」的经典成因之一。

## A 股实操注记

- **T+1 放大重叠**。T+1 制度下当日买入不能当日卖出，实际持有期天然被拉长至少一天；叠加 A 股策略常见的周频调仓，标签持有期普遍长于美股日内/短线场景——**重叠更严重，加权更不是可选项**。
- **涨跌停顺延出场**。三重障碍的止损日若撞上跌停无法成交，出场顺延，标签区间被动拉长，concurrency 进一步上升。计算 $c_t$ 时必须用**实际结算日**而非理论触障日。
- **权重与流通市值加权区分开**。这里的样本权重是训练集内部的信息权重，与组合构建层的市值加权是两回事，不要混在一个变量里。

## 你现在拥有的

一套完整的重叠校正工具：concurrency 计数（几行代码）、唯一性 + 收益归因双重权重（喂给任何支持 `sample_weight` 的模型）、序列 Bootstrap（bagging 场景的抽样替换）。

以及一个更重要的心智模型：**金融训练集的样本量是虚的，有效独立样本量 $\sum \bar{u}_i$ 才是真的**。任何时候看到「我有 5000 个训练样本」，先问一句：唯一性加权后还剩多少？

下一站自然的问题：数据和权重都修好了，模型训练出来之后，**哪些特征真的在起作用**？MDI 和 MDA 特征重要性——以及它们各自会在金融数据上以什么方式骗你——是下一篇的主题。

---

*参考文献：*
- *López de Prado, M. (2018). Advances in Financial Machine Learning, Chapter 4: Sample Weights. Wiley.*
- *Breiman, L. (1996). Bagging Predictors. Machine Learning, 24(2).*
- *本文模拟代码：1500 日三区制几何布朗运动 + EWMA 自适应 CUSUM + 三重障碍标注，全部结果可复现。*
