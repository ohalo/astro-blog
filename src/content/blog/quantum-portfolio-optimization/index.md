---
title: "量子计算在组合优化中的早期应用与 NISQ 现实"
description: "把 Markowitz 选股写成 QUBO，用 numpy 手写 statevector QAOA 在 10 只股票的小问题上命中精确解，再戳破 NISQ 现实：量子比特数、噪声、嵌入惩罚与 barren plateau，讲清它今天能干什么、不能干什么。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 量子计算
  - 组合优化
  - QUBO
  - QAOA
  - NISQ
  - Python
language: Chinese
difficulty: advanced
---

一句话结论：**组合优化（选股、再平衡、套利路径）本质是 NP-hard 的组合问题，量子近似优化算法（QAOA）确实能在小问题上命中精确解——我们用 numpy 手写的 statevector 模拟器在 10 只股票选 4 只的 QUBO 上把近似比跑到了 1.0；但今天的含噪中等规模（NISQ）硬件被「量子比特数」和「噪声」两道墙死死卡住：你能可靠优化的资产规模大约只有几十只，远不够覆盖一个真实组合。量子计算在金融里不是「明天就替代经典求解器」，而是「为特定结构化 QUBO 准备的长线期权」。

## 一、为什么是组合优化，而不是预测

量化里绝大多数活儿（收益率预测、因子拟合）是连续回归，GPU/CPU 上随随便便就解了，量子计算机在这上面没有任何优势。量子真正被寄予厚望的，是**组合优化**：在 2^N 个候选里挑一个最优的离散组合。

经典例子就是 Markowitz 选股：**从 N 只股票里选出 k 只，让 μᵀx − λ·xᵀΣx 最大**。枚举 C(N, k) 个组合，N=50、k=10 时就有约 10²¹ 种，经典暴力法直接爆。这就是量子计算的切入口——它把问题编码成**哈密顿量**，让基态自然对应最优组合。

## 二、把选股写成 QUBO

量子退火与 QAOA 都吃同一种输入：**二次无约束二值优化（QUBO）**。变量 xᵢ ∈ {0,1} 表示「选不选第 i 只」，目标写成 xᵀQx + qᵀx。带约束的 Markowitz 用「惩罚项」塞进同一个目标：

```python
import numpy as np, itertools

def build_qubo(mu, Sigma, lam, k, P):
    """选 k 只、最大化 μᵀx − λ·xᵀΣx；预算约束用惩罚项吸收。"""
    N = len(mu)
    def cost_full(x):
        x = np.asarray(x, float)
        obj  = lam * (x @ Sigma @ x) - (mu @ x)     # 风险 − 收益
        pen  = P * (x.sum() - k) ** 2                # 选够 k 只的惩罚
        return obj + pen
    return cost_full

# 精确解：暴力枚举（只在 N 很小可行，用来当 QAOA 的基准）
def exact_portfolio(cost_full, N, k):
    best = None
    for combo in itertools.combinations(range(N), k):
        xv = np.zeros(N); xv[list(combo)] = 1.0
        c = cost_full(xv)
        if best is None or c < best[0]:
            best = (c, xv.copy())
    return best   # (最优成本, 最优组合)
```

**一个关键工程细节**：惩罚系数 P 必须远大于可行解的目标跨度。我们第一轮跑了 P=5，结果 QAOA「聪明地」选了个不满足预算约束的组合——因为那个不可行解的目标值更低，惩罚压不住它。把 P 设成可行成本跨度的 100 倍后，QUBO 的全局最优才真正等于「选满 k 只」的可行最优。这类坑在真实量子编码里会被放大一百倍：**QUBO 惩罚项设错，量子计算机算得越快，错得越离谱。**

## 三、从零实现 QAOA 的 statevector 模拟器

QAOA 不在真机上跑（我们也没 10 个干净量子比特），但可以用 2^N 维 statevector 精确模拟它在理想噪声下的行为。核心就两个酉：

- **代价酉** exp(−iγ H_C)：H_C 是对角矩阵，对角元就是每个 bitstring 的 QUBO 成本，直接逐元素相位旋转；
- **混合酉** exp(−iβ H_B)：H_B = Σ Xᵢ，即每个量子比特上一个 R_x(2β)，把振幅在相邻基态间搅动，负责「探索」。

```python
import math

def apply_mixer(state, beta, N):
    """exp(−iβ·ΣXᵢ)：每个比特一个 R_x(2β)，逐位在基态间翻转。"""
    s, c = -1j * math.sin(beta), math.cos(beta)
    for i in range(N):
        mask = (np.arange(1 << N) >> i) & 1
        even, odd = state[mask == 0].copy(), state[mask == 1].copy()
        out = state.copy()
        out[mask == 0] = c * even + s * odd
        out[mask == 1] = s * even + c * odd
        state = out
    return state

def run_qaoa(gammas, betas, c_norm, N):
    dim = 1 << N
    psi = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)   # |+>^⊗N
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * c_norm)     # 代价酉
        psi = apply_mixer(psi, b, N)             # 混合酉
    return np.abs(psi) ** 2                       # 测量概率
```

p 层电路 = 交替 p 次（γ₁,β₁,…,γ_p,β_p）。参数用随机重启 + 局部坐标下降优化，目标是让最终概率质量压到成本最低的基态上。

## 四、结果：小问题上它确实命中了精确解

在 N=10、k=4、因子模型造的数据上，我们的 statevector QAOA 表现如下：

![QAOA 近似比随层数 p 变化：小问题上直接命中精确解，随机参数采样远不及](/images/quantum-portfolio-optimization/fig_qaoa_ratio.png)

近似比定义为 AR = (C_随机 − C_QAOA) / (C_随机 − C_精确)（1.0 = 命中精确解，0 = 和随机采样一样差）。结果很直白：

- **QAOA 在 p=1 就已经把近似比拉到 1.0**——10 量子比特的 QUBO 对它还太「小」，代价酉很容易把概率压到最优组合上；
- 作为对照的**随机参数采样**近似比只有 ~0.21，说明「随便转几圈」不行，参数优化才是 QAOA 的灵魂。

把 QAOA 选出的子集画到经典有效前沿上，它也确实贴在前沿附近：

![QAOA 选出的子集（红星）落在经典 Markowitz 有效前沿附近，远优于随机子集云](/images/quantum-portfolio-optimization/fig_frontier.png)

这给了我们一个**诚实的结论**：在「量子比特够用、噪声可忽略」的理想设定下，QAOA 求解组合 QUBO 是有效的。问题从来不是「能不能解」，而是「能解多大」。

这里必须说清「和谁比才公平」。在 N 小到能暴力枚举的问题上，QAOA 没有速度优势——你的笔记本 1 毫秒就算完，量子计算机从提交任务到拿到结果反而更慢；在 N 大到经典需要启发式（模拟退火、Gurobi、branch-and-bound）的问题上，QAOA 又受 NISQ 规模墙限制跑不了。所以 QAOA 的「潜在优势区间」是一个很窄的缝隙：中等规模、且问题结构能被 QUBO 干净表达的特定组合问题。把它和「经典暴力法」比速度（必输），或和「无限大容错量子机」比能力（尚早），都是不公平的——公平的标尺是「当前含噪硬件 + 同规模 QUBO 求解器」。

## 五、NISQ 现实：两道墙

NISQ = Noisy Intermediate-Scale Quantum，含噪中等规模。今天（2026 年）最先进的超导量子芯片约 1000+ 物理量子比特，但**错误率高、相干时间短**。对我们来说真正的约束是：

1. **量子比特数 = 资产数**：每只股票占 1 个逻辑比特，带预算约束还得再翻倍。你想优化 100 只股票？理想情况要 ~200 个逻辑比特，而当前**有实用价值的 QAOA 规模大约在几十一只以内**——再大电路深度暴涨、噪声把答案淹没了。
2. **噪声墙**：真实门有错误，p 层越深、纠缠越多，误差累积越快。statevector 模拟是**无噪声**的，所以它表现得像神；真机上那张 ratio=1.0 的图会迅速塌陷。

![问题规模 vs 硬件能力：组合越大越超出当前含噪硬件的可靠区间](/images/quantum-portfolio-optimization/fig_nisq.png)

图上那条绿色带，就是「今天 QAOA 还玩得转」的尺度。一个真实 A 股/美股组合动辄几百上千只标的——**现阶段量子求解器连把问题完整编码进去都做不到，更别说解出好答案。**

## 六、别被这三个叙事骗了

1. **「量子退火（D-Wave）已经能跑组合优化」**：能跑，但要把 QUBO **嵌入**到芯片的物理拓扑（Chimera/Pegasus）里，链长（chain strength）又是一个像 QUBO 惩罚一样的调参深坑；嵌入开销还会吃掉大量物理比特，进一步压低可用规模。
2. **「QAOA 比经典求解器快」**：在 N 小到经典能暴力枚举的问题上，QAOA 没有任何速度优势；在 N 大到经典吃力的地方，QAOA 又受 NISQ 限制跑不了。目前它在「中等规模 + 特定结构」的缝隙里才可能占优，且尚无严格量子加速证明。
3. **「barren plateau（ barren plateau 」**：变分量子电路（QAOA/VQE 都属此类）在参数空间里常出现梯度整体趋近于 0 的平坦区，优化器根本学不动。层数越深、比特越多，这个坑越容易出现——这恰好和「加深电路提升精度」的愿望相反。

## 七、那它今天到底能干什么

- **量子退火做精确组合再平衡**：在约束少、标的 ≤ 几十的战术场景（比如从 30 只备选里挑 8 只、带行业上限），D-Wave 的实时求解有真实落地案例，和经典混合求解器（带量子回退）搭配用比纯量子靠谱。
- **VAR 压缩与风险聚合**：组合风险度量里很多「在约束下求极值」的子问题（如最小方差、最大分散化、CVaR 约束下的配置）同样能 QUBO 化，属于同一类可提前准备的资产。
- **QUBO 化一切结构化约束**：最重要的一类「量子准备」其实是**把业务问题漂亮地写成 QUBO**——交易成本控制、套利环、清算路径、最优执行切片，凡是「在约束里挑组合」都能先翻译成 QUBO 存着，等硬件成熟直接喂。
- **作为长线期权持有**：把组合优化的 QUBO 库、嵌入与惩罚的最佳实践先沉淀下来。等容错（fault-tolerant）时代到来，这些资产立刻变现。

## 八、一句话小结

量子计算对量化最有价值的部分，不是「预测更准」，而是「把组合优化这个 NP-hard 硬骨头用哈密顿量编码、让基态自己长出最优解」。今天它在 NISQ 下能解的规模太小（几十只资产），离替代经典求解器还差得远；但**先把问题写成 QUBO、把惩罚与嵌入的工程坑踩遍**，才是现在最该做的事——等硬件跨过噪声墙，你手里的 QUBO 资产会直接变成 alpha。
