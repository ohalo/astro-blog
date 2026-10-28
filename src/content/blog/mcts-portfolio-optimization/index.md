---
title: "蒙特卡洛树搜索做组合优化：UCB1 在指数级空间里剪枝选股"
description: "组合优化本质是组合爆炸：15 只里选 5 只 = C(15,5)=3003 个候选。穷举在玩具尺度可行，真实市场几千只根本算不动。本文手写 UCB1 蒙特卡洛树搜索（选择/扩展/模拟/回传）在 2000 次目标评估预算内搜索：精确命中穷举最优 Sharpe 0.2321（差距 0%），而同预算随机搜索只到 0.218（差 6.07%）；MCTS 只评估 1483 个不同子集就逼近最优（省 50.6% 穷举量），并把搜索重心压到高 Sharpe 右尾（好资产所在子集平均 Sharpe 0.129 vs 差资产 0.117）。附完整代码与五张真实计算图。"
publishDate: '2026-10-28'
tags:
  - 量化交易
  - 蒙特卡洛树搜索
  - MCTS
  - UCB1
  - 组合优化
  - 投资组合
  - Python
language: Chinese
difficulty: advanced
---

选股组合优化是个让人头秃的问题：从 N 只资产里挑 K 只做等权组合，候选空间是组合数 C(N,K)。N=15、K=5 时就有 3003 种；N=50、K=10 时直接爆到 10²¹，穷举的物理极限。**蒙特卡洛树搜索（MCTS）正是为"指数级、不可穷举、能局部评估"的空间设计的**——它用 UCB1 平衡"探索未知"与"利用已知好区域"，在有限预算内把搜索集中到高价值角落。

本文**不调用任何搜索库**，手写一套标准 MCTS（选择→扩展→模拟→回传），在已知最优解的玩具市场上判卷。结论先放这（全部数字来自本次 `scripts/gen_mcts_portfolio.py`，seed=20261028）：

**① MCTS 精确命中穷举最优，随机搜索差 6%。** 穷举 3003 个组合的真实最优 Sharpe* = 0.2321（组合 {0,3,4,5,14}）。MCTS 用 2000 次目标评估预算就找到**完全相同的 0.2321（差距 0.0%）**；同预算的随机搜索只摸到 0.218（差距 **6.07%**）。MCTS 的收敛曲线早早贴到红虚线，随机搜索迟迟够不着。

**② 剪枝：只评估不到一半的子集就够。** MCTS 实际评估的**不同** K 子集只有 1483 个（占 3003 的 49.4%，省 **50.6%**），却已复现最优。注意：在 15 选 5 这种"能穷举"的尺度，剪枝收益有限；真正价值在 N、K 放大到不可穷举时，UCB1 的导向性让评估次数随空间规模**次线性**增长。

**③ 搜索重心真的压到了高 Sharpe 右尾。** 把所有被 MCTS 评估过的子集的目标值画成分布，整条曲线相对穷举全体的均匀直方明显右移——它不撒网，专挑高收益组合的邻居。含注入"好资产"的子集平均 Sharpe 0.129，含"差资产"的 0.117（高约 11%）。

## 一、考场：15 只资产，5 只高 alpha 的"好资产"

合成 15 只资产的 250 日收益：5 只注入日 alpha 10bp（绿），10 只只有 1bp（灰），都带市场因子 + 特异波动。目标函数是等权组合 Sharpe。先穷举拿 ground truth：

```python
import numpy as np
from itertools import combinations
rng = np.random.default_rng(20261028)
N, K, T = 15, 5, 250
market = rng.normal(0, 0.005, T)
alpha = np.array([0.0010] * 5 + [0.0001] * 10)
idio  = np.array([0.008] * 5 + [0.010] * 10)
eps   = rng.normal(0, 1, (N, T))
Rmat  = alpha[:, None] + market[None, :] + idio[:, None] * eps

def obj(sel):
    p = Rmat[list(sel)].mean(0)
    m, s = p.mean(), p.std(ddof=1)
    return m / s if s > 0 else -1e9

best_e, best_sel_e = -1e18, None
for comb in combinations(range(N), K):
    v = obj(comb)
    if v > best_e: best_e, best_sel_e = v, comb
# best_e = 0.2321，best_sel_e = (0,3,4,5,14)
```

注意一个诚实的细节：最优组合里混进了两只"差资产"(5,14)——因为**实现的**特异噪声让它们当期 Sharpe 反超。MCTS 找的是"实现最优"，不是"标签最优"，这恰恰是它的本职。

![资产散点与选中组合](/images/mcts-portfolio-optimization/fig1_asset_scatter.png)

## 二、手写 UCB1 MCTS

四个标准步骤。节点存"已选集合 + 可用集合 + 子节点 + 访问/价值"。

```python
import math
C = math.sqrt(2.0)
def make_node(selected):
    return {'sel': tuple(sorted(selected)),
            'avail': [a for a in range(N) if a not in selected],
            'children': {}, 'visits': 0, 'val': 0.0,
            'terminal': len(selected) == K}

def rollout(node):                       # 模拟：随机补全到 K 只
    if node['terminal']: return obj(set(node['sel'])), set(node['sel'])
    sel = set(node['sel']); avail = list(node['avail'])
    rng.shuffle(avail)
    for a in avail[: K - len(sel)]: sel.add(a)
    return obj(sel), sel

def select_expand(root):                 # 选择 + 扩展
    path, cur = [root], root
    while not cur['terminal']:
        if len(cur['children']) < len(cur['avail']):
            a = rng.choice([x for x in cur['avail'] if x not in cur['children']])
            child = make_node(set(cur['sel']) | {a})
            cur['children'][a] = child; path.append(child); return path
        best, bs = None, -1e18
        for a, ch in cur['children'].items():
            sc = ch['val'] / ch['visits'] + C * math.sqrt(math.log(cur['visits']) / ch['visits'])
            if sc > bs: bs, best = sc, ch
        cur = best; path.append(cur)
    return path
```

主循环：扩展出路径 → 模拟打分 → 沿路径**回传**累加访问与价值。UCB1 的 `val/visits + C·√(log N / n)` 让"被低估的节点"逐渐被多探。

```python
root = make_node(set())
best_m, best_sel_m = -1e18, None
for _ in range(2000):
    path = select_expand(root)
    val, completed = rollout(path[-1])
    for n in path: n['visits'] += 1; n['val'] += val
    if val > best_m: best_m, best_sel_m = val, tuple(sorted(completed))
```

![收敛曲线对比](/images/mcts-portfolio-optimization/fig2_convergence.png)

## 三、判卷：MCTS vs 随机 vs 穷举

| 方法 | 评估预算 | 找到的最优 Sharpe | 与最优差距 |
|---|---|---|---|
| 穷举 | 3003 子集 | 0.2321 | 0%（基准） |
| MCTS (UCB1) | 2000 次 | 0.2321 | **0.0%** |
| 随机搜索 | 2000 次 | 0.218 | 6.07% |

MCTS 只评估了 1483 个**不同**子集（其余是重复走到的），就复现了需要 3003 次才能穷举到的最优。

![剪枝比](/images/mcts-portfolio-optimization/fig3_pruning.png)

## 四、搜索偏好可视化

把每个资产在"被 MCTS 评估过的子集"里的平均组合 Sharpe 画出来：注入的好资产整体站在更高处（平均 0.129 vs 差资产 0.117）。有趣的是，**边际入选频率却近乎均匀**（好 0.329 / 差 0.335）——因为 rollout 是随机补全的，每只资产被捞进某子集的概率差不多；MCTS 的"聪明"不在挑选单只资产，而在**引导整条路径走向高价值组合**。想让单资产偏好也凸显，得用带偏置（按个体 Sharpe 加权）的 rollout。

![每资产平均 Sharpe](/images/mcts-portfolio-optimization/fig4_inclusion.png)

![目标分布对比](/images/mcts-portfolio-optimization/fig5_objective_dist.png)

## 五、冷水与落地建议

1. **本实验叶节点访问为 0**：2000 预算下树只展开到浅层，价值估计几乎全来自 rollout。放大 N、K 时更要靠"导向性模拟"，纯树展开不现实。
2. **rollout 质量是天花板**。随机 rollout 给的只是"还行"的导向；换成按个体/历史 Sharpe 加权的有偏 rollout，或用轻量代理模型当模拟器，收敛与剪枝会再上一个台阶。
3. **真实市场要加约束**：流动性、行业中性、单票上限、交易成本——这些让 obj 变贵，但也正好是 MCTS"局部评估"比穷举省钱的舞台。把约束写进 `obj`，框架一行不用改。

一句话：**MCTS 不是穷举的替代品，而是"当穷举不可能时，用导向性搜索逼近它"的那把刀。**
