---
title: "贝叶斯因果发现 PC 算法：用条件独立性找出变量间的真实因果链"
description: "相关不等于因果。本文用纯 numpy 从零实现约束式因果发现算法 PC：以偏相关 + Fisher z 做条件独立性检验，先学骨架再定碰撞方向，在 4 节点线性高斯 SCM（X1→X2、X1→X3、X2→X4←X3）上还原出真实碰撞结构；并给出骨架恢复 F1 随样本量从 0.925 升到 0.99 的实证。文末诚实讨论 PC 只能恢复 CPDAG、哪些方向天然不可识别。"
publishDate: '2026-10-29'
tags:
  - 量化交易
  - 因果推断
  - 因果发现
  - PC 算法
  - 条件独立性
  - 机器学习
  - Python
language: Chinese
difficulty: advanced
---

做因子研究的人迟早会撞上同一堵墙：**两个量一起动，不等于其中一个导致了另一个**。因子 A 和收益 B 高度相关，可能是 A 驱动了 B，可能是 B 驱动了 A，也可能两者都是某个隐藏变量 C 的共同结果。在把因子写进策略之前，搞清"谁导致谁"直接决定了你能不能做干预、能不能外推到没见过的市场状态。

因果发现（causal discovery）要做的，就是从一堆观测数据里反推出背后的因果图（DAG，有向无环图）。其中 **PC 算法**（Peter–Clark，以 Peter Spirtes 和 Clark Glymour 命名）是最经典、也最容易被从头实现的约束式（constraint-based）方法。它的核心信念只有一句话：**图上被"d-分离"的变量，在统计上应当条件独立；反过来，统计上的条件独立性，会约束因果图的结构**。

下面所有数字都来自一份可复现的 numpy 代码，没有任何黑盒。

## 一、怎么判断"在给定 S 的条件下，X 和 Y 独立"

PC 假设变量服从联合高斯分布（线性高斯 SCM），于是"条件独立"可以翻译成"偏相关为零"。给定条件集 S，X 与 Y 的偏相关用**精度矩阵（相关矩阵的逆）**直接读出：

$$\rho_{XY\cdot S} = -\frac{\Omega_{XY}}{\sqrt{\Omega_{XX}\,\Omega_{YY}}},\qquad \Omega = \Sigma^{-1}$$

拿到偏相关后，用 **Fisher z 变换**把它变成近似正态的量做显著性检验：

$$z = \frac{1}{2}\ln\frac{1+r}{1-r},\qquad |z|\sqrt{n-|S|-3} \sim \mathcal{N}(0,1)$$

当这个统计量的绝对值小于临界值（α=0.05 时是 1.96）时，我们就判定"在 S 条件下 X 与 Y 独立"，并删掉它们之间那条边。整套逻辑如下：

```python
import numpy as np
import itertools

def partial_corr(data, i, j, S):
    cols = list(set([i, j] + list(S)))
    C = np.corrcoef(data[:, cols], rowvar=False)
    P = np.linalg.inv(C)                      # 精度矩阵
    ii, jj = cols.index(i), cols.index(j)
    r = -P[ii, jj] / np.sqrt(P[ii, ii] * P[jj, jj])
    return float(np.clip(r, -0.999999, 0.999999))

def indep(data, i, j, S, alpha=0.05):
    n = data.shape[0]
    r = partial_corr(data, i, j, S)
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = abs(z) * np.sqrt(n - len(S) - 3)   # 渐近标准正态
    return stat, (stat <= 1.959963985)        # <= 临界值 => 判定独立
```

## 二、骨架学习：从完全图开始，层层"瘦身"

PC 的第一步是学**骨架**（skeleton）——也就是"哪些变量之间还有一条直接边"。初始时假设所有变量两两相连，然后从小到大枚举条件集 S：

1. 设阶数 `l = 0`；
2. 对每一对还连着的变量 (a, b)，尝试用 |S|=l 的条件集去检验独立性；
3. 一旦找到某个 S 使 a ⟂ b | S，就删掉边 a—b，并把 S 记为它们的"分隔集"；
4. `l` 加一，重复，直到没有任何一对还能被进一步条件化分离。

```python
def pc_skeleton(data, nodes=4, alpha=0.05):
    adj = {i: set(range(nodes)) - {i} for i in range(nodes)}  # 完全图
    sep = {}
    l = 0
    while True:
        edges = {(min(a, b), max(a, b)) for a in range(nodes)
                 for b in adj[a] if a < b}
        if not edges:
            break
        tested_any = False
        remove = []
        for (a, b) in edges:
            neigh = adj[a] & adj[b]
            if len(neigh) < l:
                continue
            tested_any = True
            for S in itertools.combinations(sorted(neigh), l):
                _, independent = indep(data, a, b, S, alpha)
                if independent:
                    remove.append((a, b))
                    sep[(a, b)] = set(S); sep[(b, a)] = set(S)
                    break
        for (a, b) in remove:
            adj[a].discard(b); adj[b].discard(a)
        if not tested_any or l >= nodes - 2:
            break
        l += 1
    return adj, sep
```

关键直觉：**若 a 和 b 在一个共同原因 Z 的作用下相关，那么一旦控制住 Z，两者就会独立**。所以"加上某个条件集后相关性消失"正是我们要找的信号。

## 三、定向：碰撞结构（collider）是唯一的"箭头方向锚点"

骨架只告诉我们"谁和谁相连"，不告诉方向。PC 第二步靠**碰撞结构**钉死一部分箭头。所谓碰撞（v-structure）是 `a → c ← b`：a 和 b 之间**没有**边，而 c 恰好是它们各自指向的那个共同子节点。判定规则极其优雅：

> 对三元组 a—c—b（a、b 不相邻），如果 c **不在** (a,b) 的分隔集里，则 c 一定是碰撞节点，方向定为 a→c←b。

背后的因果语义是：如果 c 是 a、b 的共同**原因**，那么控制 c 反而会"制造"出 a、b 之间的虚假相关（伯克森悖论），所以 c 必定出现在分隔集里；反之若 c 不在分隔集里，c 就不可能是原因，只能是**结果**——这就强制性地把两条箭头都指向 c。

```python
def pc_orient(adj, sep, nodes=4):
    directed = set()
    for c in range(nodes):
        for a in range(nodes):
            if a == c or c not in adj[a]:
                continue
            for b in range(nodes):
                if b == c or b == a or c not in adj[b] or b in adj[a]:
                    continue
                if c not in sep.get((a, b), set()):
                    directed.add((a, c)); directed.add((b, c))
    # Meek R1：i→j--k 且 i、k 不相邻 => j→k，把箭头顺势传播
    changed = True
    while changed:
        changed = False
        for j in range(nodes):
            incoming = [i for (i, k) in directed if k == j]
            for i in incoming:
                for k in list(adj[j]):
                    if k in incoming or (j, k) in directed or k in adj[i]:
                        continue
                    directed.add((j, k)); changed = True
    return directed
```

## 四、实证：一个 4 节点碰撞宇宙

我造了一个干净的线性高斯 SCM，真实因果图是 `X1→X2`、`X1→X3`、`X2→X4`、`X3→X4`——也就是说 **X4 是 X2 与 X3 的碰撞节点**，而 X1 是 X2、X3 的共同原因：

```python
def sample_scm(n, rng):
    e1, e2, e3, e4 = (rng.normal(0, 1, n) for _ in range(4))
    X1 = e1
    X2 = 0.8 * X1 + e2
    X3 = -0.6 * X1 + e3
    X4 = 0.7 * X2 + 0.5 * X3 + e4
    return np.column_stack([X1, X2, X3, X4])
```

把 4000 条样本喂给 PC，恢复的图如下。它会正确识别出 X4 是碰撞节点（两条红箭头都射入 X4），骨架与真实图完全吻合：

![真实因果图（左）与 PC 恢复的偏有向图 PDAG（右），碰撞方向 X2→X4←X3 被正确还原](/images/bayesian-causal-discovery-pc/fig1_dag_vs_pdag.png)

第二张图把"PC 凭什么删边"摊开给你看。对 `X2 ⟂ X3 | S`，无条件时两者相关（|z| 远超 1.96，保留边）；**一旦控制 X1，统计量立刻塌到临界值以下**（变绿、删边）——因为 X1 是它们的共同原因。对 `X1 ⟂ X4 | S` 则相反：控制 X1 没用，必须同时控制 X2 和 X3 才能让 X1 与 X4 独立，于是 X1—X4 这条边被正确删除。红=依赖(留边)，绿=独立(删边)：

![X2 与 X3、X1 与 X4 的条件独立性检验：Fisher z 统计量随条件集变化，红绿区分保留/删除](/images/bayesian-causal-discovery-pc/fig2_ci_tests.png)

第三张图回答一个现实问题：**数据不够时 PC 会犯什么错**。横轴是样本量，纵轴是骨架恢复的 F1。N=80 时 F1 只有 0.925（小样本下偶发的伪独立/伪相关会多删或漏删边），样本量升到 1280 以上稳定在 0.99。因果发现不是免费的，它要的是**足够多、且确实 i.i.d. 的样本**：

![PC 骨架恢复的精确率、召回率、F1 随样本量从 0.925 提升到约 0.99](/images/bayesian-causal-discovery-pc/fig3_skeleton_recovery.png)

## 五、必须说清的三个边界

1. **PC 恢复的是 CPDAG，不是 DAG。** 在上面的例子里，X1—X2 和 X1—X3 这两条边在恢复的图里保持无向。原因是：观测数据 + 碰撞结构只能确定 X4 是碰撞节点，却无法区分"X1 是 X2 的原因"还是"X2 是 X1 的原因"——两者在纯观测下统计等价。要彻底定方向，必须上**干预（do-算子）/ 随机实验**数据，这正是后续 ICP、GIES 等算法的舞台。
2. **隐变量会伪造碰撞。** 如果存在一个未被观测的 Z 同时影响 a 和 b，PC 可能把 `a → c ← b` 误判为碰撞，进而反转一大片箭头方向。有无隐变量的可识别性是因果发现里最硬的开放问题之一。
3. **高斯/线性假设是"免训练"的代价。** 本文用偏相关 + Fisher z，等价于对线性高斯世界做检验。遇到非线性或非高斯，需要换成基于 HSIC、回归误差或熵的**独立检验**（如 KCIP、RESIT），PC 的骨架-定向框架本身不变，只是把"独立性检验"这个零件换掉。

一句话总结：**PC 算法用"条件独立性 ↔ d-分离"这把尺子，从观测数据里量出因果图的骨架，再用碰撞结构这唯一可靠的箭头锚点把方向钉死**。它是因果推断通往"可干预、可外推"策略的必经第一站——而认清它只能恢复 CPDAG，恰恰是诚实使用它的前提。
