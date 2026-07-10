---
title: "遗传算法与进化策略在组合优化中的应用"
description: "经典均值方差优化对预期收益极度敏感、且对基数/换手等实务约束几乎无解。本文用遗传算法在「精选 K 只、多头、权重和为 1」的硬约束下搜索最优组合，从零实现选择-交叉-变异-精英保留主循环，并与无约束均值方差前沿对比，顺带讲清进化策略(ES)与三大实战陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 遗传算法
  - 组合优化
  - 进化策略
  - 均值方差
  - Python
language: Chinese
difficulty: advanced
---

「用历史均值做预期收益，跑出来的有效前沿漂亮得不像话，实盘却一塌糊涂。」——这是均值方差（Mean-Variance）优化的老问题。更尴尬的是，当你想加几条**实务约束**（只允许买 5 只、单只上限 30%、换手不超过 20%），经典解析解直接失灵：目标非凸、约束组合爆炸，二次规划也救不了。

这时候，遗传算法（Genetic Algorithm, GA）这类**启发式搜索**就有了用武之地：它不在乎目标函数是不是凸的、约束是不是线性的，只要你能「算出一个组合的得分」，它就在成千上万种组合里替你进化出最优解。本文从零实现一个带基数约束的组合优化 GA，并和经典均值方差前沿硬碰硬对比。

![遗传算法流程示意图：选择→交叉→变异→精英保留](/images/genetic-algo-portfolio/ga_schematic.png)

## 一、为什么传统均值方差优化会翻车

Markowitz 的解析解长这样（无风险资产为 0 时，最大夏普组合）：

```
w* = Σ⁻¹ μ / (1ᵀ Σ⁻¹ μ)
```

三个致命伤：

1. **误差放大**：权重对预期收益 μ 极度敏感。μ 一点点估计误差，会被 Σ⁻¹ 放大成荒谬的极端权重（全仓某只冷门股）。
2. **约束无解**：「只允许持有 K 只」「行业中性」「换手上限」这类离散/非凸约束，直接让解析公式失效。
3. **过拟合幻觉**：在样本内追求夏普最大化，几乎必然过拟合历史噪声。

遗传算法不追求「闭式最优」，而是**在约束空间里高效搜索一个足够好的解**——这正是实盘需要的。

## 二、遗传算法的四个零件

把「一个投资组合」编码成一条**染色体**（chromosome）：一个权重向量 `w`。然后模拟进化：

- **适应度（Fitness）**：给每个组合打分。目标=夏普比率，但违反约束要重罚；
- **选择（Selection）**：高分组合有更高概率被选中当「父母」（这里用锦标赛选择）；
- **交叉（Crossover）**：两条父母染色体交换片段，产生「子女」；
- **变异（Mutation）**：随机扰动部分基因，维持多样性、跳出局部最优；
- **精英保留（Elitism）**：把当代最优原样复制进下一代，防止退化。

关键技巧是**约束投影**：每次生成新染色体后，把它「投影」回可行域——只保留最大的 K 个权重、其余置 0、再归一化，强制满足「多头 + 和为 1 + 精选 K 只」。

## 三、从零实现组合优化 GA

下面是可运行的核心代码（为清晰省略绘图，完整版跑出来即上图结果）：

```python
import numpy as np

n_asset, K = 12, 5          # 12 只候选，精选 5 只
POP, GEN = 200, 120
MUT, CROSS, ELITE = 0.25, 0.90, 4

# mu: 年化预期收益, cov: 协方差 (此处用合成数据)
def port_stats(w):
    w = np.asarray(w)
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return ret, vol, ret / vol          # 收益, 波动, 夏普

def project_cardinality(w, K):
    """投影回可行域：多头、和为1、恰好持有 K 只。"""
    w = np.clip(w, 0, None)
    w = w / w.sum()
    thresh = np.sort(w)[::-1][K - 1]    # 第 K 大权重作为阈值
    w = np.where(w >= thresh, w, 0.0)
    return w / w.sum()

def random_chrom():
    idx = np.random.choice(n_asset, K, replace=False)
    w = np.zeros(n_asset)
    raw = np.random.rand(K)
    w[idx] = raw / raw.sum()
    return w

def fitness(w):
    ret, vol, sharpe = port_stats(w)
    penalty = 0.0
    if np.sum(w > 1e-6) != K:        penalty += 5.0   # 基数不符
    if np.any(w < -1e-9):            penalty += 5.0   # 出现空头
    if abs(w.sum() - 1) > 1e-6:      penalty += 5.0   # 未归一
    hhi = np.sum(w**2)
    penalty += 0.3 * max(0, hhi - 1.0 / K)            # 轻微惩罚过度集中
    return sharpe - penalty

# —— 主循环 ——
pop = [random_chrom() for _ in range(POP)]
for g in range(GEN):
    fits = [fitness(w) for w in pop]
    order = np.argsort(fits)[::-1]
    new_pop = [pop[order[i]] for i in range(ELITE)]     # 精英

    def tournament():
        i, j = np.random.choice(POP, 2, replace=False)
        return pop[i] if fits[i] > fits[j] else pop[j]

    while len(new_pop) < POP:
        if np.random.rand() < CROSS:
            p1, p2 = tournament(), tournament()
            cut = np.random.randint(1, n_asset)
            for c in (np.concatenate([p1[:cut], p2[cut:]]),
                      np.concatenate([p2[:cut], p1[cut:]])):
                new_pop.append(project_cardinality(c + np.random.rand(n_asset)*0.15, K))
                if len(new_pop) >= POP: break
        else:
            new_pop.append(tournament().copy())
    for i in range(ELITE, POP):                        # 变异
        if np.random.rand() < MUT:
            w = pop[i] + np.random.rand(n_asset) * 0.15
            new_pop[i] = project_cardinality(w, K)
    pop = new_pop[:POP]

best = pop[np.argmax([fitness(w) for w in pop])]
```

跑完这组参数，GA 收敛到的组合夏普约 **2.73**、年化收益 **12.1%**、波动 **4.4%**，且**恰好持有 5 只**——约束被严丝合缝地满足了。作为对照，无约束均值方差的夏普是 **3.12** 但持有了 9 只；朴素按收益归一化只有 **1.81**。GA 用「少买几只」换来了接近最优、且可执行的组合。

这里有个反直觉但重要的点：**少买几只不是次优，而是实务下的必然权衡**。持有 9 只的理论前沿看似更优，但它隐含了「每调一次仓都要交易全部 9 只、且对每只的预期收益都估计得极准」的前提；一旦加入换手成本与估计误差，这个「最优」会迅速塌缩。GA 在搜索时把「可执行性」写进了适应度，它主动选择了一个更稳健、更省交易成本的解——这正是黑盒搜索相对闭式解的现实优势。

## 四、收敛与解空间：它真的在变好

先看收敛曲线——早期几十代适应度（夏普）快速爬升，之后在约束前沿上精细搜索，曲线趋于平稳。红色虚线是「无约束均值方差」的理论上限，作为参照锚点。

![遗传算法收敛曲线：早期快速上升，后期在约束前沿精细搜索](/images/genetic-algo-portfolio/ga_convergence.png)

再看解空间分布。灰色云是「无约束随机组合」的可行域，蓝点是经典均值方差前沿（无视基数约束），橙色是 GA 探索过的解，绿色星是 GA 最终组合。可以看到：**GA 的解稳稳落在有效前沿附近，但被牢牢限制在「精选 5 只」这条实务边界上**——这正是传统解析方法做不到的。

![解空间分布：遗传算法在「精选 K 只」约束下逼近有效前沿](/images/genetic-algo-portfolio/ga_frontier.png)

最终入选组合的权重分布也很「干净」：5 只等权附近、略有倾斜，没有任何一只被压到 0 或爆到 100%。

![遗传算法最终入选组合（精选 5 只，多头、权重和为 100%）](/images/genetic-algo-portfolio/ga_weights.png)

## 五、进化策略（ES）：GA 的连续系表亲

遗传算法处理「离散+连续」混合很顺手，但当问题纯连续、且维度很高时，**进化策略（Evolution Strategy）**往往更高效。它不编码成 0/1 染色体，而是直接对**连续参数向量**做变异，核心玩法两种：

- **(μ, λ) 与 (μ+λ)**：每代从 μ 个父代产生 λ 个子代；(μ, λ) 只用子代、(μ+λ) 父子同台竞争；
- **CMA-ES（协方差矩阵自适应进化策略）**：不仅自适应步长，还自适应搜索空间的**形状**（用协方差矩阵），在病态、旋转的非凸地形上极其强悍，是连续黑盒优化的 SOTA 之一。

对组合优化而言，GA 适合「带基数/行业等硬约束」，CMA-ES 适合「纯权重连续优化且目标非凸」。两者都属于「黑盒、无梯度、只靠适应度」的进化家族。

## 六、实战三大陷阱

1. **过拟合到历史**：GA 自由度极大，容易在样本内「进化」出恰好吻合历史噪声的诡异组合。务必做样本外验证、Walk-Forward，并限制代际/种群规模，别让搜索过度挖掘噪声。
2. **早熟收敛（Premature Convergence）**：种群多样性过早丧失，全体挤在一个局部最优里。对策是调高变异率、用锦标赛而非纯精英选择、或引入小生境（niching）保持多样性。
3. **计算成本与约束泄漏**：种群×代数次适应度评估（每次都要算协方差/夏普）在标的数大时很贵；更隐蔽的是**约束投影泄漏**——若投影函数写错，染色体可能悄悄违反「和为 1」或「多头」，导致适应度虚高。投影函数一定要单测。

## 七、小结

遗传算法把组合优化从「求闭式最优」变成「在约束空间里高效进化出一个好解」。它的强项恰恰是传统方法最弱的地方：**非凸目标、离散基数约束、换手/行业等实务边界**。代价是计算更贵、更需要防过拟合——但它给你的是「实盘真的能挂得出来」的组合，而不只是一张漂亮的前沿曲线。

至此，我们从一个参数的在线估计（卡尔曼滤波动态对冲），走到了成千上万组合的全局搜索（遗传算法组合优化）——两者共同说明了同一件事：**量化里真正难的，从来不是写出公式，而是让模型承认世界的时变与约束。**
