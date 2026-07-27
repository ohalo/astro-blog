---
title: "嵌套聚类优化 NCO：先聚类再优化的两层组合构建"
description: "Markowitz 优化器的误差会在相关资产之间共振：40 只资产、4 个板块的块状相关结构下，T=90 天样本协方差直接优化的 OOS 波动率 9.7%、总杠杆 3.1 倍。López de Prado 的 NCO 把问题切成两层：先按相关距离聚类把 40×40 矩阵切成 4 个小块，簇内各自优化、再在 4 个簇组合之间优化——T=90 时 OOS 波动率降到 8.1%、杠杆压到 2.0 倍，Bootstrap 权重标准差从 3.07pp 砍到 1.44pp（稳定性翻倍）。代价同样实测得到：T=1008 数据充足时 NCO（7.68%）反而略输给直接优化（7.52%），两层结构禁止跨簇对冲，是用偏差换方差的又一笔交易。与 HRP 的区别：HRP 全程不求逆，NCO 只是把求逆的规模变小——簇内维度低到样本够用时，二次优化的信息还能保留（中阶）。"
publishDate: '2026-07-28'
tags:
  - 量化交易
  - NCO
  - 聚类
  - 组合优化
  - 机器学习
  - Python
language: Chinese
difficulty: intermediate
---

这个协方差系列已经讲了三种给优化器"喂更好输入"的办法：[Ledoit-Wolf 收缩](/blog/ledoit-wolf-shrinkage/)、[Marchenko-Pastur 去噪](/blog/random-matrix-marchenko-pastur/)、[Bayes-Stein 收缩均值](/blog/bayes-stein-shrinkage-returns/)。今天这篇换个思路：**不改输入，改优化问题本身的结构**。López de Prado（2019）的嵌套聚类优化（Nested Clustered Optimization, NCO）主张：与其求解一个 40 维的病态问题，不如切成 4 个 10 维的良态小问题再拼起来。

先给判决数字。40 只资产、4 个板块的块状真实相关（板块内 0.6、板块间 0.15）、最小方差目标、60 次 Monte Carlo 的 OOS 真实波动率中位数：

| 估计窗 T | 样本协方差直接优化 | NCO | 总杠杆（直接 vs NCO） |
|---|---|---|---|
| 90 天 | 9.71% | **8.11%** | 3.08 vs 1.97 |
| 126 天 | 8.86% | **7.92%** | 2.74 vs 1.93 |
| 252 天 | 7.98% | **7.81%** | 2.49 vs 1.89 |
| 504 天 | 7.69% | 7.72% | 2.40 vs 1.86 |
| 1008 天 | **7.52%** | 7.68% | 2.32 vs 1.86 |

理论最优 7.38%，等权 12.87%。规律一目了然：**数据越少 NCO 优势越大，数据充足时 NCO 反而略输**。这条交叉曲线就是本文要讲清楚的全部内容。

![不同估计窗下的 OOS 波动率：NCO 在数据稀缺区间领先，数据充足时被反超](/images/nested-clustered-optimization/oos-vol.png)

## 优化器的误差为什么在相关资产间共振

[Michaud 说优化器是误差最大化器](/blog/michaud-resampled-frontier/)，但误差最大化有一个具体的作案模式：**高相关资产之间的对冲对赌**。两只相关系数 0.9 的股票，样本协方差稍微估歪一点，优化器就会做多"看起来"波动低的那只、做空另一只，赌这 0.1 的差异是真实的——而它几乎总是噪声。相关性越高，这种对赌的杠杆越大：权重之差正比于 $1/(1-\rho)$。

块状相关结构（现实市场的常态：板块内高相关、板块间低相关）正是这种误差的温床。看 T=252 的单次实验权重图：直接优化总杠杆 2.71、21 个负权重，同板块内多空互搏清晰可见；理论最优解的杠杆只有 2.28。

![三组权重对比：直接优化在板块内多空对赌，NCO 权重结构接近理论最优](/images/nested-clustered-optimization/weights.png)

NCO 的洞察：**板块内的对冲对赌发生在簇内，板块间的配置是另一个独立问题**。把两层拆开，每层的维度都变小、每层的病态程度都下降。

## 算法：两层各求各的逆

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def min_var_w(S):
    Si = np.linalg.pinv(S)
    ones = np.ones(len(S))
    return Si @ ones / (ones @ Si @ ones)

def nco_min_var(S, k):
    """嵌套聚类优化（最小方差版本）"""
    # 1) 相关距离矩阵 d_ij = sqrt((1-rho_ij)/2)
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    dist = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    np.fill_diagonal(dist, 0.0)

    # 2) 层次聚类切成 k 簇
    Z = linkage(squareform(dist, checks=False), method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")

    # 3) 簇内优化：每簇内部解一个小规模最小方差
    N = len(S)
    W_intra = np.zeros((N, k))
    for j in range(1, k + 1):
        idx = np.where(labels == j)[0]
        W_intra[idx, j-1] = min_var_w(S[np.ix_(idx, idx)])

    # 4) 簇间优化：把每个簇当成一个合成资产，在 k×k 缩减协方差上再解一次
    S_reduced = W_intra.T @ S @ W_intra
    w_inter = min_var_w(S_reduced)

    return W_intra @ w_inter   # 最终权重 = 簇内权重 × 簇间权重
```

四步全是标准件：相关距离聚类 → 簇内小优化 → 缩减协方差 → 簇间小优化。关键的维度算术：直接优化要求逆一个 40×40 矩阵（820 个自由参数）；NCO 只需要求逆四个约 10×10 矩阵（各 55 个参数）加一个 4×4 矩阵（10 个参数）。**每个子问题的 T/N 比都翻了 4 倍**——T=90 喂 40 维是灾难（T/N=2.25），喂 10 维是及格（T/N=9）。

聚类这一步在 T=252 时就能把 4 个板块干净切开——相关矩阵的块状结构比单个协方差元素稳定得多，这是整个方法的统计根基：**先估容易估的（谁和谁一伙），再估难估的（组内怎么配权）**。

![Ward 聚类树与重排后的相关矩阵：块状结构在 T=252 时清晰可辨](/images/nested-clustered-optimization/dendrogram.png)

## 稳定性：Bootstrap 权重标准差砍半

对同一份 T=252 数据做 80 次 Bootstrap 重抽样，看权重晃动幅度：

- 直接优化：权重标准差均值 **3.07 个百分点**
- NCO：**1.44 个百分点**

![Bootstrap 权重稳定性：NCO 的权重标准差是直接优化的一半以下](/images/nested-clustered-optimization/weight-stability.png)

权重稳定性直接决定换手率和交易成本，这一点上 NCO 和 [Michaud 重抽样](/blog/michaud-resampled-frontier/)殊途同归，但机制不同：Michaud 靠平均抹掉噪声，NCO 靠结构约束让噪声没有表达渠道——簇的划分不变，噪声只能在低维子问题内部晃，晃不出跨簇的大动作。

## 代价：禁止跨簇对冲的结构偏差

T=1008 时 NCO（7.68%）输给直接优化（7.52%）不是实验噪声，是方法的固有代价。NCO 的解空间是受限的：最终权重必须能写成"簇内权重 × 簇间权重"的乘积形式，**跨簇的精细对冲被结构性禁止**。真实最优解如果需要"做多板块 A 的第 3 只、做空板块 B 的第 7 只"这种跨簇搭配，NCO 表达不出来。数据充足时直接优化能把这些精细结构估准，NCO 的约束就从保护变成了枷锁。

这和收缩估计的逻辑完全同构：**用偏差换方差**。收缩的偏差是"往目标矩阵拉"，NCO 的偏差是"锁死两层结构"；收缩强度 δ 随 T 增大自动退场，NCO 的结构约束却不会自动松开——这是它相对收缩的一个真实劣势，实践中的对策是数据充足时直接切回全局优化，或者两者都算再按 T/N 加权。

**与 HRP 的分工**也值得说清楚。López de Prado 更出名的 HRP（层次风险平价）全程不求任何逆矩阵，靠树状结构递归二分配权——更稳但更粗糙，簇内信息只用到对角线。NCO 保留了二次优化（簇内簇间都正经求逆），只是把求逆规模压到样本量能支撑的水平。粗略的选择规则：T/N < 2 用 HRP，2~10 用 NCO，> 10 直接优化加收缩。

## 工程细节

**簇数 k 怎么定**：本文实验直接用了真实值 4（上帝视角）。实践中用轮廓系数（silhouette）或 gap statistic 扫 k=2..10 选最优；López de Prado 原文建议对聚类质量做进一步的 Monte Carlo 检验。k 选错的代价不对称——切多了（把一个板块切成两半）几乎无害，切少了（把两个板块并成一簇）会把跨板块对赌重新放进簇内。

**先去噪再聚类**：原版 NCO 在聚类前先做 [Marchenko-Pastur 去噪](/blog/random-matrix-marchenko-pastur/)，两个方法是串联关系不是二选一。去噪让相关矩阵的块状结构更清晰，聚类更稳。

**均值-方差版本**：把 `min_var_w` 换成带期望收益的效用最大化即可，但簇内期望收益请先过 [Bayes-Stein 收缩](/blog/bayes-stein-shrinkage-returns/)——NCO 修不了均值的估计误差。

**簇的时变性**：真实市场的板块结构会漂移（尤其危机时相关性趋同、簇结构塌缩成一大块）。滚动重估簇划分时加惯性（新旧划分差异小于阈值则不换），否则簇边界的抖动本身就是换手来源。

## 结语

NCO 是这个系列里第一个不动输入、动问题结构的方法：把一个大病态问题拆成几个小良态问题，每一层的求逆都发生在样本量撑得住的维度上。它在数据稀缺区间的优势真实且可观（OOS 波动率降 16%、杠杆降 36%、权重稳定性翻倍），代价是数据充足时约 2% 的结构偏差。下一篇我们会把这个系列收个尾：把收缩、去噪、重抽样、聚类这几件工具怎么串成一条生产线讲清楚。

## 参考文献

1. López de Prado, M. (2019). A Robust Estimator of the Efficient Frontier. *SSRN Working Paper*.
2. López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
3. Raffinot, T. (2017). Hierarchical Clustering-Based Asset Allocation. *Journal of Portfolio Management*, 44(2).
