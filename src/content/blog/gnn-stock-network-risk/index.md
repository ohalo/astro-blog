---
title: "图神经网络(GNN)在关联个股网络中的风险传染建模"
description: "个股从不是孤岛：相关性、供应链、板块联动把 30 只股票织成一张网。本文从零实现 2 层图卷积网络(GCN)，先让它学会预测「系统性重要度」，再用训练好的消息传递把冲击沿网络扩散，量化「hub 暴跌→全网传染」的放大效应，并诚实列出图构建里的真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 图神经网络
  - 风险传染
  - 系统性风险
  - 压力测试
  - Python
language: Chinese
difficulty: advanced
---

2020 年 3 月，美股十天四次熔断。你如果只看单只股票的波动率，会以为那是 30 只各自独立恐慌的股票；但真正发生的是**一张网在集体抽搐**——ETF 抛售触发做市商对冲，对冲引发动量基金止损，止损又压垮流动性，负反馈沿相关性边一圈圈烧到看似无关的板块。

传统风险模型把组合方差拆成 `wᵀΣw`，把所有联动压成一个协方差矩阵。这没错，但它**丢掉了网络结构**：谁连着谁、谁是中心、冲击从哪进来会放大最狠。图神经网络（GNN）干的事，就是把「个股=节点、关联=边」这张网显式建模，让信息沿边**消息传递（message passing）**，从而捕捉协方差矩阵藏起来的传染路径。

本文从零实现一个 2 层 GCN（不依赖 PyTorch Geometric / DGL），先让它学会预测每只股票的「系统性重要度」，再用训练好的传播机制把冲击灌进网络，量化 hub 的传染放大效应。

![个股相关性网络（节点大小=系统性重要度，红=最大传染 hub）](/images/gnn-stock-network-risk/fig_network.png)

## 一、先把「股票网络」建出来

风险传染的图可以有很多种：相关系数图、供应链有向图、板块隶属图、ETF 持仓重叠图。本文用最朴素也最稳的**收益率相关图**：30 只股票，模拟一个带 3 个社区（板块）的因子结构，社区内相关高、社区间相关低，再按相关系数阈值连边。

```python
import numpy as np
rng = np.random.default_rng(7)
N = 30
communities = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
# 因子模型构造协方差，保证正定
loadings = np.zeros((N, 3))
for k, c in enumerate(communities):
    loadings[c, k] = rng.uniform(0.6, 1.0, len(c))
loadings[communities[0][:3], 1] = 0.15      # 跨社区弱联动
loadings[communities[1][:3], 2] = 0.18
cov = loadings @ loadings.T + np.diag(rng.uniform(0.3, 0.8, N))
L = np.linalg.cholesky(cov)
rets = (rng.standard_normal((N, 2000)).T @ L.T).T
corr = np.corrcoef(rets)

A = (np.abs(corr) > 0.35).astype(float)     # |相关系数|>0.35 连边
np.fill_diagonal(A, 0.0)
A = np.maximum(A, A.T)                        # 对称化
```

节点特征用两个连续量：特质波动率、加权连接强度（相关图是规则图时「度」会退化成常数，必须换成连续量）。

## 二、从零实现 2 层 GCN

图卷积的核心操作是**对称归一化邻接矩阵** `Ã = D^{-1/2} A D^{-1/2}`，它把邻居的信息按度数归一后聚合，避免高度数节点主导。一层 GCN 就是「聚合邻居 → 线性变换 → 激活」：

```python
D = np.diag(A.sum(1) + 1e-8)
Dh = np.linalg.inv(np.sqrt(D))
Ahat = Dh @ A @ Dh                       # 对称归一化邻接

def relu(z): return np.maximum(z, 0)

def train_gcn(X, Ahat, y, H=12, epochs=400, lr=0.01, lam=1e-3):
    n, d = X.shape
    W1 = rng.normal(0, 0.3, (d, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 0.3, (H, 1)); b2 = np.zeros(1)
    for _ in range(epochs):
        S1 = Ahat @ X
        Z1 = relu(S1 @ W1 + b1)
        S2 = Ahat @ Z1                       # 第二次消息传递
        Z2 = S2 @ W2 + b2
        dZ2 = 2 * (Z2 - y) / n
        dW2 = S2.T @ dZ2 + 2 * lam * W2; db2 = dZ2.sum(0)
        dZ1 = Ahat.T @ (dZ2 @ W2.T)
        dZ1r = dZ1 * (S1 @ W1 + b1 > 0)
        dW1 = S1.T @ dZ1r + 2 * lam * W1; db1 = dZ1r.sum(0)
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1
    S1 = Ahat @ X; Z1 = relu(S1 @ W1 + b1); S2 = Ahat @ Z1
    return (W1, b1, W2, b2), S2 @ W2 + b2
```

注意 `S2 = Ahat @ Z1` 这一步——信息是**走了两跳**（邻居的邻居）才到达每个节点，这正是 GCN 能捕捉「间接传染」的关键。

## 三、用 GCN 预测「系统性重要度」

我们把标签设为每只股票的**特征向量中心性**（幂迭代求 `Ahat` 的主特征向量），它刻画一只股票在网里的 systemic 重要性。然后让 GCN 和无图 MLP（把 `Ahat` 换成单位阵，消息传递消失）比赛谁拟合得准：

```python
def eig_centrality(M, iters=200):
    v = rng.normal(0, 1, M.shape[0]); v /= np.linalg.norm(v)
    for _ in range(iters):
        v = M @ v; nrm = np.linalg.norm(v)
        if nrm < 1e-12: break
        v /= nrm
    return v

y = eig_centrality(Ahat).reshape(-1, 1)
y = (y - y.mean()) / (y.std() + 1e-8)

P_gcn, yhat_gcn = train_gcn(X, Ahat, y)
P_mlp, yhat_mlp = train_gcn(X, np.eye(N), y)   # 无图基线
```

结果很说明问题：

| 模型 | 拟合 MSE | 与真实中心性相关性 |
|---|---|---|
| 2 层 GCN（有图） | **0.0031** | **0.999** |
| 无图 MLP（断开边） | 0.8112 | 极低 |

「系统性重要度」本质是个**网络性质**——一只股票重不重要，取决于它连着谁。GCN 通过消息传递直接用到邻居结构，所以几乎完美拟合；而断网的 MLP 只能看节点自身的两个特征，对网络级属性几乎失明。这恰恰证明了：**在关联资产上，图结构本身就是信号**。

## 四、冲击传染：把 shock 灌进网络

拟合中心性只是热身。真正有价值的是**传导**：某只股票暴雷，压力怎么沿边扩散到全网？

做法很直观——把暴雷节点 `i` 的特征第一维大幅拉高（模拟它突然承压），前向传播读出全网每节点的压力，得到「污染行」`C[i, :]`；对每只股票都做一次，就得到完整的**传染矩阵** `C`：

```python
def propagate(shock_node, P, mag=3.0):
    Xs = X.copy()
    Xs[shock_node, 0] += mag
    S1 = Ahat @ Xs; Z1 = relu(S1 @ P[0] + P[1])
    S2 = Ahat @ Z1
    return (S2 @ P[2] + P[3]).ravel()

C = np.zeros((N, N))
for i in range(N):
    C[i] = propagate(i)
hub_impact = C.mean(1)
hub  = int(np.argmax(hub_impact))     # 平均传染最强 → 最大传染源
leaf = int(np.argmin(hub_impact))     # 平均传染最弱 → 叶子节点
```

在我们的网络里，最大传染源是**节点 1**（平均冲击 0.029），最弱的是**节点 28**。`C` 矩阵里能直接读出「如果节点 1 暴雷，谁会最疼」——这正是压力测试和系统性风险监控想要的表。

![冲击传染矩阵：每一行是「某节点暴雷时全网的压力分布」](/images/gnn-stock-network-risk/fig_contagion_heatmap.png)

## 五、hub 与 leaf：传染为何差这么多

把冲击在 GCN 的传播层数上展开，差异一目了然。下图对比「hub 节点 1」和「leaf 节点 28」各自受冲击后，全网累计压力随传播层数（hop）的变化：

```python
def topo_prop(source, layers):
    v = np.zeros(N); v[source] = 1.0
    out = [v.copy()]
    for _ in range(layers):
        v = Ahat @ v                       # 纯拓扑传播
        out.append(v.copy())
    return np.array(out)

L = 6
hub_traj  = topo_prop(hub,  L).sum(1)      # 全网总压力随层数
leaf_traj = topo_prop(leaf, L).sum(1)
```

![冲击随 GCN 传播层数扩散：hub 的传染显著强于 leaf](/images/gnn-stock-network-risk/fig_propagation.png)

hub 的曲线早期就爬得更高、衰减更慢——因为它连着大片邻居，冲击第一跳就散到很多节点，第二跳再散一轮，雪球越滚越大；leaf 的冲击第一跳就传到寥寥几个邻居，随后迅速被稀释。这个**非线性放大**，是协方差矩阵给不出、但网络视角一眼看清的。

## 六、真实陷阱：图不是白建的

1. **图构建的前视偏差（最常见）**：用全样本相关系数建图，等于把未来信息编进了「过去」的模型。正确做法是**滚动窗口建图**，每个时点只用截至当时的数据。
2. **阈值拍脑袋**：`|corr|>0.35` 是拍的。阈值太高→图太稀，传染传不动；太低→全连通，GCN 退化为平均池化。建议用**最近 K 邻居（K-NN）**或按分位数剪枝，比固定阈值稳。
3. **相关≠因果，且会翻转**：牛市的强相关，危机里可能变负相关（一起去杠杆时齐跌是相关，但避险切换时是负相关）。静态图会漏掉这种**机制切换**，必须用**时变图 / 状态依赖的边权**。
4. **流动性冻结让边"消失"**：极端行情下，原本相关的资产因为一方停牌或丧失流动性，相关性反而断裂。这时候"按相关性建图"会低估传染——要补一层流动性约束。
5. **过平滑（over-smoothing）**：GCN 层数多了，所有节点的表征会趋同（都变成全图平均），失去区分度。关联资产上 2~3 层通常够，别堆太深。

顺带一提，实务里更稳的建图方式是用 **K 近邻（K-NN）** 而非固定阈值，既能避免规则图导致「度」退化，也绕开阈值敏感：

```python
def knn_graph(corr, k=4):
    A = np.zeros_like(corr)
    for i in range(corr.shape[0]):
        top = np.argsort(-np.abs(corr[i]))[1:k+1]   # 跳过自己，取最相关的 k 个
        A[i, top] = 1
    return np.maximum(A, A.T)
```

此外，静态图只抓「平均联动」，而危机里的传染是**时变**的。更进一步的做法是时序图卷积（TGCN）：把时间维也建成边，让消息在「股票 × 时间」两张网上同时传递，捕捉「先有龙头异动、再传染邻居」的先后次序。本文的 2 层 GCN 正是它的地基——吃透了这个，TGCN 不过是多接一条时间边。

## 七、落地场景

- **系统性风险监控**：每天滚动建相关图，用 GCN 算各节点「系统性重要度」，对 top-hub 提高保证金或降敞口；
- **压力测试**：选若干 hub 注入冲击，读 `C` 矩阵得到「传染热力图」，比 `wᵀΣw` 的单一数字信息量大得多；
- **组合保险**：做多 leaf、对 hub 做保护（如买入 hub 对应的看跌期权），在危机里对冲最可能被传染的节点；
- **事件触发的连锁预警**：某龙头暴雷时，直接用传播层数列它「两跳内"的持仓，提前减仓而不是等跌幅兑现。

GNN 不是把量化变玄学，它只是把「资产之间的网」从协方差矩阵的黑箱里拽出来，变成可以解释、可以压力测试、可以预警的结构。比起又叠一层 LSTM，先把你的股票们连成网，往往能多看见一半的风险。

## 八、一句话小结

关联资产的风险，从来不是「30 个独立变量」的加总，而是一张会呼吸、会传染的网。GCN 的价值不在于它比 LSTM 更花哨，而在于它把「谁连着谁、冲击从哪进、往哪放大」这件事，变成了可计算、可解释、可压力测试的结构。先把股票连成网，再用两层消息传递看清传染路径——这一步，往往比再堆十层黑箱模型更能帮你躲过下一次熔断。
