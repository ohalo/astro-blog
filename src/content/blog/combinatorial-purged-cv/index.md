---
title: "组合式净化交叉验证 CPCV：用组合分块把回测过拟合概率算出来"
publishDate: '2026-07-27'
description: "组合式净化交叉验证 CPCV：用组合分块把回测过拟合概率算出来 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

一次回测给你一个夏普 1.8，你敢上吗？大多数人会说「再跑一次样本外看看」。但样本外只有一段，它本身也可能是运气。真正的问题是：能不能从一次回测里，同时估计出「这个策略的样本外表现分布」和「它其实是过拟合噪声的概率」？López de Prado 的组合式净化交叉验证（Combinatorial Purged Cross-Validation，CPCV）就是干这个的——它用组合数学榨出几十条互不相同的回测路径，再用 purge 和 embargo 堵死金融数据特有的信息泄漏，最后算出一个叫 PBO（Probability of Backtest Overfitting，回测过拟合概率）的数字。

## 为什么普通交叉验证在金融上会骗你

机器学习里 K 折交叉验证是标配：把数据切成 K 份，轮流拿一份做测试、其余做训练，得到 K 个样本外分数，取平均。这套在图片、文本上工作得很好，但直接搬到金融时间序列上会出两个致命问题。

**第一个问题：信息泄漏。** 金融标签几乎总是「跨越时间」的。比如你用「未来 5 天的收益」当标签，那么 t 时刻的样本，它的标签依赖 t 到 t+5 的价格。如果训练集里有 t+3 的样本、测试集里有 t 的样本，训练集就「偷看」了测试期的信息——两个样本的标签在时间上重叠了。普通 K 折随机打乱，这种重叠遍地都是，导致样本外分数被系统性高估。

**第二个问题：只有一条路径。** K 折给你 K 个分数，但它们高度相关（训练集大量重叠），本质上只是对「同一个模型」的 K 次略有差异的测量，不能刻画「如果历史稍有不同，策略会怎样」。你想知道的是样本外表现的**分布**，K 折给不了。

CPCV 用两招分别解决：**组合分块**解决路径太少，**purge + embargo**解决信息泄漏。

## 组合分块：从 N 块里榨出 C(N,k) 条路径

CPCV 的第一个想法：不要每次只留一块做测试，而是留 k 块。把时间序列按顺序切成 N 个等长分块，每次从中选 k 块当测试集、其余 N-k 块当训练集。这样一共能组合出 C(N, k) 种不同的训练/测试划分。

举个具体例子：N=6，k=2，那么 C(6,2)=15 条路径。相比传统 6 折只有 6 条，CPCV 直接给了 15 条。而且更妙的是，每一块数据都会在不同组合里被用作测试若干次，把这些测试段按时间拼起来，能重构出多条完整的「样本外净值曲线」——每条对应历史的一种可能排列。

```python
from itertools import combinations
import numpy as np

def cpcv_splits(n_samples, n_groups=6, n_test_groups=2):
    """
    生成 CPCV 的所有训练/测试划分。
    返回: [(train_idx, test_idx), ...]，共 C(n_groups, n_test_groups) 条
    """
    # 按时间顺序把样本索引切成 n_groups 个连续分块
    groups = np.array_split(np.arange(n_samples), n_groups)
    splits = []
    for test_grp in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_grp])
        train_grp = [g for g in range(n_groups) if g not in test_grp]
        train_idx = np.concatenate([groups[g] for g in train_grp])
        splits.append((np.sort(train_idx), np.sort(test_idx)))
    return splits

splits = cpcv_splits(1000, n_groups=6, n_test_groups=2)
print(f"总划分数: {len(splits)}")   # 15
```

下面这张图把 N=6、k=2 的全部 15 种组合画成一个矩阵，每行一种划分，红色是测试块、蓝色是训练块：

![CPCV 组合分块矩阵](/images/combinatorial-purged-cv/cpcv-matrix.jpg)

关键在于：**每个分块（每一列）在整个矩阵里被标成「测试」的次数是相同的**。C(6,2)=15 种组合中，每块恰好出现在 C(5,1)=5 种组合的测试集里。这保证了每段历史都被公平地当作样本外测试，不存在某段数据从没被验证过的死角。

## Purge 与 Embargo：堵死时间泄漏

组合分块解决了路径数量，但没解决信息泄漏——训练块和测试块相邻的边界处，标签仍会重叠。这就是 purge 和 embargo 登场的地方。

**Purge（净化）**：删除训练集中那些「标签时间窗与测试集重叠」的样本。如果标签是未来 5 天收益，那么测试块开始前 5 天内的训练样本，其标签会伸进测试期，必须剔除。同理测试块结束后紧邻的训练样本也可能因反向依赖被污染，一并 purge 掉。

**Embargo（禁运）**：在测试块之后，再额外空出一小段缓冲区（比如总样本的 1%），这段时间的训练样本也不要。原因是金融数据有序列相关性（波动率聚集、动量），即使标签窗不重叠，测试期刚结束的那点数据仍可能因自相关而泄漏信息。Embargo 是一道保险。

![Purge 与 Embargo 示意](/images/combinatorial-purged-cv/purge-embargo.jpg)

橙色是 purge 区（剔除标签重叠的训练样本），紫色是 embargo 区（额外的禁运缓冲），红色是测试集。只有清干净这两道区域，剩下的蓝色训练样本才真正与测试集「信息隔离」。

代码实现的核心是根据每个样本的标签持续期（label span）做时间剔除：

```python
def purge_embargo(train_idx, test_idx, label_span=5, embargo_pct=0.01, n_total=1000):
    """
    对一个 train/test 划分执行 purge + embargo。
    label_span: 标签的时间跨度（如未来 5 天收益 -> 5）
    embargo_pct: 禁运区占总样本比例
    """
    test_start, test_end = test_idx.min(), test_idx.max()
    embargo = int(n_total * embargo_pct)

    # purge: 剔除标签窗与测试集重叠的训练样本
    # 训练样本 i 的标签覆盖 [i, i+label_span]，若与测试区间相交则删
    purge_lo = test_start - label_span
    purge_hi = test_end + label_span
    # embargo: 测试结束后再多禁运一段
    embargo_hi = test_end + embargo

    keep = (train_idx < purge_lo) | (train_idx > max(purge_hi, embargo_hi))
    return train_idx[keep]

# 用在每条划分上
clean_splits = []
for tr, te in splits:
    tr_clean = purge_embargo(tr, te, label_span=5, embargo_pct=0.01, n_total=1000)
    clean_splits.append((tr_clean, te))
```

这个实现是简化版（假设索引即时间序）。实战中标签跨度往往不等长（比如三重障碍法的持仓期各不相同），需要按每个样本各自的标签结束时间逐一判断重叠，逻辑更细，但原理完全一致：**只要训练样本的标签时间与测试期有任何交集，就 purge**。

## PBO：把过拟合概率算成一个数

有了几十条干净的样本外路径，就能做 CPCV 最有价值的事——估计 PBO，回测过拟合概率。

思路是这样的：假设你不是只测一个策略，而是在做参数搜索，有 S 个候选配置。对每条 CPCV 路径，你在「样本内」（训练段）挑出表现最好的那个配置，然后看它在对应「样本外」（测试段）的表现排名。如果一个策略是真有 alpha，那它样本内最好、样本外也该不错；如果它只是过拟合了噪声，样本内最好的那个到样本外往往泯然众人、甚至垫底。

PBO 就定义为：**样本内选出的最优策略，在样本外表现低于中位数的概率**。PBO 越接近 0.5，说明你的「最优选择」和扔硬币没区别——纯粹过拟合；PBO 接近 0，说明样本内的优势能稳定延续到样本外。

```python
def compute_pbo(is_perf, oos_perf):
    """
    is_perf, oos_perf: shape (n_strategies, n_paths)
    每列是一条 CPCV 路径下所有候选策略的样本内/样本外表现（如夏普）
    返回: PBO
    """
    n_paths = is_perf.shape[1]
    logits = []
    for j in range(n_paths):
        best = np.argmax(is_perf[:, j])          # 样本内最优策略
        # 该策略在样本外的相对排名（0=最差，1=最好）
        rank = (oos_perf[:, j] < oos_perf[best, j]).mean()
        w = np.clip(rank, 1e-3, 1 - 1e-3)
        logits.append(np.log(w / (1 - w)))       # logit 变换
    logits = np.array(logits)
    return float((logits < 0).mean())            # logit<0 即排名低于中位数
```

下面用模拟数据演示。设 200 个候选策略，它们的真实 alpha 都很弱（样本外夏普基本收缩到 0 附近），看 CPCV 能不能识破：

```python
np.random.seed(7)
S, n_paths = 200, 15
is_sharpe = np.random.normal(1.0, 0.5, (S, n_paths))       # 样本内看着都不错
oos_sharpe = 0.1 * is_sharpe + np.random.normal(0, 0.6, (S, n_paths))  # 样本外收缩

pbo = compute_pbo(is_sharpe, oos_sharpe)
print(f"PBO = {pbo:.2f}")   # PBO ≈ 0.40
```

![样本内外夏普关系与 PBO 分布](/images/combinatorial-purged-cv/pbo-distribution.jpg)

左图是每个策略的样本内夏普对样本外夏普的散点：如果样本内能预测样本外，点应该沿 45 度线（灰色虚线）分布；但实际它们贴着那条几乎水平的红线（收缩线）——样本内夏普再高，样本外也被打回 0 附近。右图是 15 条路径的 logit 分布，红线左边（logit<0，即样本外低于中位数）占了 40%，于是 PBO ≈ 0.40。这个数字在告诉你：这批策略里，你辛辛苦苦挑出来的「最优」，有四成概率在样本外还不如中位数——过拟合风险相当高，不该轻易上仓位。

## 为什么这比「留一段样本外」强

传统做法是切出最后 20% 当样本外，跑一次看结果。CPCV 相比它有三个实质提升。

**样本利用率高。** 单一样本外浪费了 20% 的数据从不参与训练，而 CPCV 里每块数据既当过训练也当过测试，信息利用充分——这对本就稀缺的金融历史尤其关键。

**给的是分布不是点。** 单一样本外只有一个数字，你不知道它是运气好还是真本事。CPCV 给你 15（或更多）条样本外路径，能画出样本外夏普的整个分布，看它的均值、方差、最差情形。

**PBO 直接量化过拟合。** 这是最独特的。单一样本外无法回答「我做了这么多参数搜索，选出的最优有多大概率是蒙的」，而 PBO 把这个问题变成一个可计算的概率。做参数扫描、因子挖掘时，PBO 应该和夏普一起报告——一个夏普 2.0 但 PBO 0.5 的策略，远不如夏普 1.2 但 PBO 0.1 的策略可信。

## 小结

CPCV 的三件套——组合分块、purge/embargo、PBO——分别对应回测验证的三个痛点：路径太少、信息泄漏、过拟合无法量化。它不能让一个坏策略变好，但能让你在上真金白银之前，诚实地看清「这个漂亮的回测有多大概率是自欺欺人」。在参数越调越多、因子越挖越深的今天，PBO 这种把过拟合概率算成数字的工具，比又一个高夏普回测有价值得多。

**诚实边界**：CPCV 计算成本随 C(N,k) 组合爆炸增长，N 稍大就要训练成百上千次模型，对重模型（深度学习）可能不现实，需在分块数和算力间权衡；purge/embargo 的正确性完全依赖对标签时间跨度的准确刻画，标签窗估错则泄漏照样发生；PBO 假设候选策略集合能代表你真实的搜索空间，若隐性尝试（那些没记录下来的调参）没被计入，PBO 仍会低估真实过拟合程度；此外 CPCV 假设各分块间机制大致平稳，遇到强 regime 切换时组合出来的路径可能包含现实中不会发生的时间拼接。（中阶）
