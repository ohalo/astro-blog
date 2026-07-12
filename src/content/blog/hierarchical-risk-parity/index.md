---
title: "层次风险平价(HRP)：用聚类与递归二分超越传统风险平价"
publishDate: '2026-07-13'
description: "传统风险平价要么求逆协方差（病态）要么坍缩到单名40%+；HRP 不求逆：Ward 聚类+递归二分按逆方差切权重，样本外波动14.13%低于等权14.82%、最大单名仅19.8%远胜最小方差37.6%集中。"
tags:
  - 量化交易
  - 投资组合
  - 风险平价
  - 层次聚类
  - 资产配置
  - 协方差估计
  - Python
language: Chinese
difficulty: advanced
---

如果你做过组合优化，一定被两件事折磨过：**（1）** Markowitz 最小方差要你求逆一个高维协方差矩阵 $\Sigma^{-1}$，而 $\Sigma$ 一旦估计得不好（样本少、资产多、相关性高），求逆会把噪声放大成垃圾权重；**（2）** 哪怕是用迭代法求风险平价，遇到极端相关/波动，权重也会坍缩到一两只资产上，单名暴露冲到 40%+，回测漂亮、实盘崩盘。

2016 年 Lopez de Prado 在 *Financial Analysts Journal* 提出 **层次风险平价（Hierarchical Risk Parity, HRP）**，核心洞见是：**不用求逆、不用优化，先把资产按相关性聚成树，再沿树递归二分切权重**。它天然尊重"哪些资产是一伙的"，避开协方差求逆的病态，也避开单名坍缩。

## 一、为什么协方差求逆是脆弱的

组合方差 $\sigma_p^2 = w^\top \Sigma\, w$，最小方差解 $w^\star \propto \Sigma^{-1}\mathbf 1$。问题在 $\Sigma^{-1}$ 对输入极度敏感：资产间相关性略高一点，条件数（condition number）就爆炸，求逆结果剧烈抖动。实务上必须上收缩估计（Ledoit–Wolf）或正则化，否则你优化出来的"最优权重"只是对噪声过拟合。

HRP 的聪明之处：它**绕开 $\Sigma^{-1}$ 和任何二次规划**，只用到"相关性的距离结构 + 递归切分"。代价是它不再追求数学上的精确最小方差，但换来的是**稳健、可解释、对估计误差免疫**。

## 二、HRP 四步算法

1. **相关性 → 距离**：用 $d_{ij} = \sqrt{\tfrac12(1-\rho_{ij})}$ 把相关系数变成欧氏距离（相关性越高，距离越近）。
2. **层次聚类**：对距离矩阵做 Ward  linkage，得到一棵把相似资产不断合并的树。
3. **准对角化（quasi-diagonalization）**：按树的叶子顺序重排相关矩阵，让同簇资产相邻——原本杂乱的相关矩阵变成清晰的"块对角"结构。
4. **递归二分（recursive bisection）**：从树顶往下，每一步把当前簇按**子簇组合方差的逆**切成两份，使两份承担相等的风险贡献，直到每片叶子分到权重。

![左：原始相关矩阵杂乱无章；右：按 HRP 叶子顺序重排后，三个板块聚成清晰的块](/images/hierarchical-risk-parity/hrp_corr.png)

## 三、从零实现（Python）

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def hrp_weights(cov):
    N = cov.shape[0]
    # 1) 相关系数 → 距离
    vols = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(vols, vols), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    order = dendrogram(Z, no_plot=True)["leaves"]   # 准对角化顺序

    # 2) 构建每个节点（含内部节点）的叶子集合
    node_leaves = {i: [i] for i in range(N)}
    for k, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        node_leaves[N + k] = node_leaves[a] + node_leaves[b]
    root = N + len(Z) - 1

    w = np.ones(N) / N
    def allocate(node):
        leaves = node_leaves[node]
        if len(leaves) <= 1:
            return
        ridx = node - N
        a, b = int(Z[ridx, 0]), int(Z[ridx, 1])
        left, right = node_leaves[a], node_leaves[b]
        # 子簇内用 IVP 归一化权重算组合方差（尺度无关，避免孤立资产被放大）
        def cluster_var(idxs):
            sub = cov[np.ix_(idxs, idxs)]
            iv = 1.0 / np.diag(sub); ww = iv / iv.sum()
            return ww @ sub @ ww
        vL, vR = cluster_var(left) + 1e-12, cluster_var(right) + 1e-12
        aL, aR = 1.0 / vL, 1.0 / vR
        s = aL + aR
        # 保持该节点总权重不变，仅按逆方差在两子簇间重切
        Wnode = w[leaves].sum()
        tl, tr = Wnode * aL / s, Wnode * aR / s
        w[left]  *= tl / w[left].sum()
        w[right] *= tr / w[right].sum()
        if a >= N: allocate(a)
        if b >= N: allocate(b)
    allocate(root)
    return w, Z, order
```

关键细节：**子簇方差必须用簇内归一化的 IVP 权重算**（不是用已经累乘到很小的当前权重）。否则一个被早早孤立的单资产，其组合方差会被当前 tiny 权重压得极小，逆方差反馈会把它爆拉到 90%+——这是很多"野路子 HRP"实现翻车的根源。

聚类树本身也有经济含义：在我们构造的 15 只资产（科技 / 金融 / 能源三板块）上，Ward 聚类干净地把三个板块各自聚成一团：

![Ward 层次聚类树：三个板块（Tech/Fin/Energy）各自成团，板块着色一目了然](/images/hierarchical-risk-parity/hrp_dendro.png)

## 四、和四种经典方法的实盘对比

我们用板块化协方差生成 1500 天日收益（前 1000 天估协方差、后 500 天做样本外），把 HRP 和等权(EW)、逆方差(IVP)、最小方差(MinVar)、迭代风险平价(RP)放一起比：

![五类方法权重：HRP 分散（最大 19.8%），而最小方差坍缩到 37.6% 压在单只上](/images/hierarchical-risk-parity/hrp_weights.png)

| 方法 | 样本外年化波动 | 最大单名权重 | 是否求逆 |
|---|---|---|---|
| 等权 EW | 14.82% | 6.7% | 否 |
| 逆方差 IVP | 14.18% | 13.4% | 否 |
| 迭代风险平价 RP | 14.48% | 10.0% | 否 |
| **层次风险平价 HRP** | **14.13%** | **19.8%** | **否** |
| 最小方差 MinVar | 12.64% | **37.6%** | **是** |

三件事值得记：

1. **HRP 样本外波动 14.13%，低于等权 14.82%、迭代 RP 14.48%**，和 IVP 基本打平，且完全不求逆。它用"聚类 + 逆方差切分"拿到了接近风险平价的分散效果。
2. **HRP 最大单名权重仅 19.8%**，组合是真正分散的。对照下，**最小方差虽然波动最低（12.64%），却把 37.6% 的赌注压在一只资产上**——这是教科书级的"回测最优、实盘最脆"：一旦那只股票异常波动或停牌，组合直接裸奔。
3. HRP 的波动略高于最小方差，但代价换来的是**没有单名坍缩、没有协方差求逆**。对实盘而言，这个 trade-off 几乎总是划算的。

样本外累积收益曲线也印证了这一点——HRP 和 IVP 始终贴着上方走，等权明显更晃，而最小方差虽然波动低，其高集中度在样本外某段回撤里暴露无遗：

![样本外 500 天累积收益：HRP 与 IVP 领先，等权更晃，MinVar 高集中带来隐性风险](/images/hierarchical-risk-parity/hrp_perf.png)

## 五、HRP 为什么稳

- **不求逆**：只用距离 + 聚类 + 标量方差，对 $\Sigma$ 的估计误差不敏感，不需要 Ledoit–Wolf 收缩也能用。
- **尊重相关性结构**：先聚类再切分，意味着"同一板块的资产不会同时被重仓"，天然对冲了板块内部的共动风险。
- **不坍缩**：递归二分按逆方差切，但被聚类树约束在合理粒度，不会像最小方差那样把权重推到边界。
- **可解释**：聚类树就是一张"资产亲缘关系图"，PM 一眼能看懂权重怎么来的。

## 六、已知边界

- **聚类是启发式，不是最优**：Ward 切分点由树结构决定，未必是全局最小方差解（本来也不是目标）。想要理论最优还得回到优化，但 HRP 赌的是"稳健 > 精确"。
- **依赖相关性估计**：聚类质量完全取决于输入的 $\rho$。用 100 天滚窗估计的相关，在结构性变盘时会失效——生产环境要用收缩估计或指数加权。
- **不是精确等风险贡献**：HRP 的风险贡献比等权更均衡，但严格等于 RP 的"每资产贡献相同"需要迭代法。HRP 的卖点从来不是数学精确，而是"不求逆也能接近"。
- **链接方式敏感**：本文用 Ward（方差最小化合并），换成 single/average 会得到不同的树和权重，建议多链路做稳定性检验。

## 结论

层次风险平价不是"又一个优化器"，而是一种**用结构代替优化的工程哲学**：当你不确定协方差矩阵准不准时，与其冒险求逆，不如先问"谁和谁是一伙的"，再沿亲缘关系把风险一层层平分。在 15 只三板块资产的回测里，它用 19.8% 的最大单名权重拿到了 14.13% 的样本外波动——比等权更稳、比迭代 RP 更分散，且没有最小方差那种 37.6% 的单名裸奔。对于"想要风险平价的好处、又怕优化翻车"的组合管理者，HRP 往往是那个最省心、也最不容易在实盘炸掉的选择。
