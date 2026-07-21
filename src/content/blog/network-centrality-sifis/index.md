---
title: "网络中心性与系统性重要机构识别：用图论给「谁不能倒」定个价"
description: "单家机构的 VaR 只看自己亏多少，看不见「你倒了连累谁」。网络中心性（度/特征向量/介数）把机构间的敞口纠缠量化成节点重要性，直接回答监管与风险管理的核心问题：哪些机构是系统重要性最高、最不能倒的 SIFI。本文用纯 numpy 从零构造银行间敞口网络，计算三种中心性并对比排名，并诚实拆穿网络不完整/阈值主观/时变断裂/多层隐藏/中心性权重五类真实陷阱（中阶）。"
publishDate: '2026-07-21'
tags:
  - 量化交易
  - 系统性风险
  - 网络中心性
  - SIFI
  - 图论
  - 复杂网络
  - 风险管理
  - Python
language: Chinese
difficulty: intermediate
---

2008 年雷曼倒下，远在亚洲的股指也跟着崩——风险会**传染**。但传统风险管理只看单家机构的 VaR：「我自己最多亏多少」。VaR 看不见「你倒了会拖垮多少同行」。系统性风险研究的全部努力，就是把这种看不见的外部性**量化成数字**。

结论先放这：

**金融机构不是孤立的点，而是银行间敞口网络里的节点。一个机构「重不重要」，不取决于它自己的体量，而取决于它连着谁、连了多少、以及它是不是别人绕不开的「桥」。网络中心性用度/特征向量/介数三类指标，把这种「网络连接重要性」变成可排序、可监管、可定价的分数——排名最高的那些节点，就是系统重要性最高、最不能倒的 SIFI（Systemically Important Financial Institution）。**

---

## 一、为什么需要「网络视角」

### 1. 单点风险度量的盲区

VaR、ES 这些工具回答的是：「给定我的持仓，我自己最坏亏多少」。它们默认损失是**边际独立**的。但真实金融系统里，机构之间通过三类通道互相牵连：

- **同业拆借**：A 借钱给 B，B 倒了 A 的债权就烂了；
- **衍生品对手方**：CDS/利率互换的对手方违约，会触发连锁赔付；
- **共同持仓**：大家都持有某类资产，抛售会形成「去杠杆螺旋」。

只看自己，等于在火灾连片的公寓里只检查自己房间的灭火器。

### 2. 网络能回答什么

把机构当节点、敞口当边，整个系统变成一个**图（graph）**。图论提供了一整套「谁更中心」的度量：

- **度中心性（Degree）**：你直接连着多少机构、敞口多大；
- **特征向量中心性（Eigenvector）**：你连着的人本身重不重要；
- **介数中心性（Betweenness）**：你是不是别人之间绕不开的桥。

这三把尺子，恰好对应监管关心的三个问题：敞口规模、关联质量、网络枢纽性。

---

## 二、从零构造一张银行间敞口网络

我们用 numpy 手工造一张「核心—外围」结构的网络：少数大型银行互相密集拆借（核心），大量 regional 银行主要挂在核心行上（外围）。这正是真实银行体系的典型结构。

```python
import numpy as np

np.random.seed(42)
names = ["ICBC","CCB","BOC","ABC","BoCom","CMB","SPDB","MSB",
         "CIB","PingAn","CITIC","Huaxia","Everbright","Minsheng",
         "BOS","JSB-A","JSB-B","JSB-C"]
n = len(names)
core = list(range(7))      # 0-6: 系统重要性核心行
peri = list(range(7, n))   # regional / 小行

rng = np.random.default_rng(7)
A = np.zeros((n, n))
# 核心-核心：密集、大敞口
for i in core:
    for j in core:
        if i < j:
            w = rng.uniform(0.6, 1.0)
            A[i, j] = A[j, i] = w
# 外围优先挂核心：小行从大行借钱
for i in peri:
    k = rng.integers(2, 4)
    for t in rng.choice(core, size=k, replace=False):
        w = rng.uniform(0.15, 0.5)
        A[i, t] = A[t, i] = w
# 少量外围-外围：本地聚簇
for i in peri:
    if rng.random() < 0.4:
        j = rng.choice(peri)
        if j != i and A[i, j] == 0:
            A[i, j] = A[j, i] = rng.uniform(0.1, 0.3)
```

`A[i, j]` 就是机构 i 对 j 的（标准化）敞口权重。邻接矩阵 `A` 就是这张网络的完整描述。看它的热力图，右上角那块密集的红色就是核心行之间的互相纠缠——这正是风险最容易堆积的地方。

![敞口邻接矩阵 A：右上角密集块即核心行的相互纠缠](/images/network-centrality-sifis/adjacency_heatmap.png)

---

## 三、三种中心性，三种「重要」的定义

### 1. 度中心性（Degree Centrality）

最朴素的版本：一个节点的邻居越多、边越重，它越中心。对加权网络，直接用「强度」：

```python
# 度（强度）中心性：邻居敞口之和
deg = A.sum(axis=1)
deg_c = deg / (n - 1)   # 标准化到 [0,1]
```

它回答「这家机构总敞口有多大」。局限：连着 100 家小行，可能还不如连着 1 家巨型行重要。

### 2. 特征向量中心性（Eigenvector Centrality）

这是 PageRank 的同款思想：**重要的邻居才让你重要**。它解一个方程：

$$c_i = \frac{1}{\lambda} \sum_{j} A_{ij} c_j$$

把所有机构摆成一个向量，中心性就是邻接矩阵最大特征值对应的特征向量。用幂迭代几行就能解：

```python
def eigenvector_centrality(A, iters=2000):
    x = np.ones(A.shape[0])
    for _ in range(iters):
        x = A @ x
        x = x / np.linalg.norm(x)
    return x

eig_c = eigenvector_centrality(A)
eig_c = eig_c / eig_c.max()   # 相对标准化
```

特征向量中心性天然识别出「核心行」——它们互相连接、彼此都重要，于是分数一起被推高。在我们的网络里，ICBC / CCB / ABC / BoCom / SPDB 排在最前。

### 3. 介数中心性（Betweenness Centrality）

前两个看「连接多少」，介数看「你是不是桥」。它数：在所有两两最短路径里，有多少条必须经过你。一个 regional 行如果只有一条路连到核心，那它头顶上那家核心行就是它的唯一桥梁，介数极高。

```python
import scipy.sparse.csgraph as cg
# 用阈值把加权图变成无权图，再算最短路径
B = (A > 0.2).astype(float)
np.fill_diagonal(B, 0)
dist = cg.shortest_path(cg.csgraph_from_dense(B), directed=False, unweighted=True)

def betweenness(B):
    n = B.shape[0]; bet = np.zeros(n)
    for s in range(n):
        # Brandes 算法：栈 S、前驱 P、最短路径计数 sigma
        S = []; P = [[] for _ in range(n)]
        sigma = np.zeros(n); sigma[s] = 1
        D = np.full(n, np.inf); D[s] = 0; Q = [s]; qh = 0
        while qh < len(Q):
            v = Q[qh]; qh += 1; S.append(v)
            for w in np.where(B[v] > 0)[0]:
                if D[w] == np.inf:
                    D[w] = D[v] + 1; Q.append(w)
                if D[w] == D[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]
    return bet / bet.max() if bet.max() > 0 else bet

bet_c = betweenness(B)
```

介数高 = 系统的「咽喉」。一旦咽喉节点出问题，大片外围节点会被「隔离」，这正是传染链最先断裂的地方。

---

## 四、把三种尺子摆到一起看

把度、特征向量、介数画成并列柱状图，核心行在三项上**全面领先**——这说明三种定义「英雄所见略同」，结论稳健；而某些 regional 行可能在度上不低、介数却很高，提示它是局部枢纽。

![三种中心性指标对比：核心行全面领先](/images/network-centrality-sifis/centrality_bars.png)

更直观的是网络图：节点大小 = 特征向量中心性。红点是核心行、蓝点是 regional，一眼能看出中心那团密密麻麻的大红点就是系统核心。

![银行间敞口网络：节点大小 = 特征向量中心性（红=核心行，蓝=regional）](/images/network-centrality-sifis/network_graph.png)

---

## 五、从「中心性」到「系统重要性排名」

监管要的是**一个可操作的名单**，而不是三张图。常见做法是合成一个综合分：

```python
composite = 0.4 * eig_c + 0.3 * bet_c + 0.3 * deg_c
order = np.argsort(composite)[::-1]
print("SIFI 名单(前5):", [names[i] for i in order[:5]])
```

在我们的网络里，综合排名与特征向量排名高度一致——核心行稳居前列。用「特征向量 vs 介数」画散点，右上角那批点（既高度互联、又是桥梁）就是**系统重要性最高**的机构，监管应当对其附加额外资本缓冲。

![双维排序：右上角 = 既高度互联又掌握最短路径 → 系统重要性最高](/images/network-centrality-sifis/sifi_scatter.png)

![系统重要性综合排名（前高后低）](/images/network-centrality-sifis/sifi_ranking.png)

这套方法的政策含义很直接：识别出的 SIFI，应当被要求持有**更高资本充足率**、接受更严的压力测试、并限制其关联集中度——因为「它倒了」的社会成本，远大于「它自己亏的钱」。

---

## 六、五个不能忽略的真实陷阱

别把网络排名当成真理。以下每一条都可能让结论翻车：

1. **网络不完整（最大陷阱）**：真实同业敞口大多不公开，你的 `A` 往往是抽样或估算。缺一条关键边，介数排名可能整个重排。监管用的双边敞口申报（如全球系统重要性银行 G-SIB 申报）才是金标准。
2. **阈值主观**：把加权图转无权图时，「多重的边才算连接」是拍的。阈值一变，最短路径和介数全变。应当对多个阈值做**敏感性扫描**，看排名是否稳定。
3. **时变与断裂**：危机时相关性飙升、连接骤增，平时的网络结构在崩盘日完全失效。静态快照会低估危机期的系统性关联——应当用滚动窗口看网络演化。
4. **多层隐藏**：我们只建了「同业拆借」一层，真实还有衍生品、共同持仓、支付清算系统多层叠加。单层网络会**低估**跨层枢纽的重要性。
5. **中心性权重拍脑袋**：综合分里 0.4/0.3/0.3 的权重是我定的。不同权重下排名会变，应当用 bootstrap 或 Shapley 值检验稳健性，而非当成客观事实。

---

## 七、小结

网络中心性把「谁不能倒」从一个直觉问题，变成了可计算、可排序、可监管的量化问题。度中心性看规模、特征向量看关联质量、介数看咽喉位置——三者结合，就能在崩盘发生前，先画出一张「系统重要性地图」。

但记住它的边界：排名的质量上限，就是你的网络数据的质量上限。网络不完整，再漂亮的中心性算法也只是**在噪声上做精密测量**。把它当作「风险地图的草稿」，而非监管判决的最终依据——这才是工程上诚实的用法。
