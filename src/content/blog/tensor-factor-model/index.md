---
title: "张量因子模型：用 HOSVD/Tucker 分解把「股票×特征×时间」三维面板一次抽因子，为什么它比逐期矩阵化 PCA 更省样本、又在异质载荷下露馅"
description: "40 股票×15 特征×120 天真值已知实验。真秩 (3,2,3) 时 HOSVD 只用 528 个参数就把相对误差压到 0.052，逐期矩阵化 PCA 花 5520 参数最优也只能到 0.166，噪声放大到 σ=1.5 时更崩到 1.065（HOSVD 仅 0.136）。缺失 80% 数据时 HOSVD 填补 RMSE 0.104，矩阵化 1.02。但当四个时期各有独立股票载荷时，全局 (3,2,3) 误差飙到 0.748，分段建模 0.033；硬把秩拉到 12 需 2238 参数才回到 0.050。附完整 numpy 实现的 HOSVD 模式积、四张真实计算图与适用边界。"
publishDate: '2026-11-25T10:55:00Z'
tags:
  - 量化
  - 因子模型
  - 张量分解
  - 降维
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

做因子的人手里几乎都是同一块数据：一张股票对特征的二维表，每个交易日来一张。最常见的做法是把它沿时间摊平成一个大矩阵，再做一次 PCA。这个动作看起来自然，代价却很少被算清楚——摊平之后参数量是"秩 ×（股票数 × 时间 × 特征）"，几乎是三维面板本身的大小，样本根本不够填。

张量因子模型换了个思路：不摊平，直接把三维面板做 Tucker 分解，三个模式各抽一组低维因子，中间留一个小核心张量。本文用一个真值已知的合成实验把收益和边界都算出来。

先说收益。40×15×120 的面板，真实秩只有 (3,2,3)，HOSVD 用 **528** 个参数（3×40 + 2×15 + 3×120 + 3×2×3）就能把相对重构误差压到 **0.052**。同样数据摊成矩阵做 PCA，最优秩 3 已经要用 **5520** 个参数，误差却只到 **0.166**；噪声从 σ=0.2 加到 1.5，HOSVD 误差从 0.017 温和爬到 **0.136**，矩阵化 PCA 从 0.143 崩到 **1.065**——噪声越大，摊平方法的参数劣势越是灾难。

再说最实用的场景：缺失数据。A 股面板里停牌、退市、财报缺列是常态。随机挖掉 80% 的条目再做低秩填补，HOSVD 在留出条目上的 RMSE 是 **0.104**，矩阵化 PCA 是 **1.02**，差了近十倍。少参数在这里变成了正则化：模型被迫只学三个模式上的共同结构，没法去 memorize 稀疏观测里的噪声。

最后说边界。这套方法的命门是"全局低秩"这个假设。我构造了一个四个时期各有独立股票载荷的市场——每一段内部仍是干净的秩 (3,2,3)，但跨段拼起来后股票的全局秩变成 12。这时全局 HOSVD 的相对误差飙到 **0.748**，几乎什么都没学到；按已知分段点分别做 HOSVD 只有 **0.033**。不想分段、只想靠加大秩救回来？要把秩拉到 12，参数从 528 涨到 **2238**，误差才回到 0.050。

**一句话：数据真是全局低秩时，张量分解以一成的参数打赢矩阵化 PCA；一旦载荷随制度漂移，全局因子就是个昂贵的幻觉。**

![参数效率对比](/images/tensor-factor-model/fig1_param_efficiency.png)

## 二、Tucker 分解与 HOSVD 到底在做什么

把收益面板记成三阶张量 $\mathcal X \in \mathbb R^{N\times P\times T}$，N 只股票、P 个特征、T 个时间。Tucker 分解写成：

$$\mathcal X \approx \mathcal G \times_1 A \times_2 B \times_3 C$$

$A\in\mathbb R^{N\times r_1}$、$B\in\mathbb R^{P\times r_2}$、$C\in\mathbb R^{T\times r_3}$ 是三个模式的因子矩阵，$\mathcal G\in\mathbb R^{r_1\times r_2\times r_3}$ 是核心张量，$\times_k$ 表示第 k 模式的乘积。参数量是 $Nr_1+Pr_2+Tr_3+r_1r_2r_3$——注意是三个模式的**和**，而不是摊平后的**乘积**。

HOSVD（高阶奇异值分解，De Lathauwer 2000）是最常用的求法：把张量沿每个模式展开成矩阵，做 SVD，取前 $r_k$ 个左奇异向量作为该模式的因子矩阵，再把原张量依次投影到这三个子空间上得到核心张量。它不像矩阵 SVD 那样给出全局最优截断（最优要交替最小二乘 ALS 迭代），但一次展开三次 SVD 就够用，且误差有不超过 $\sqrt{3}$ 倍最优的理论界。

实现的关键是两个小函数。模式展开把第 k 维挪到最前面再拉平：

```python
def unfold(t, mode):
    return np.moveaxis(t, mode, 0).reshape(t.shape[mode], -1)

def mode_product(t, mat, mode):
    # mat: (new_dim, old_dim)
    shp = list(t.shape)
    res = mat @ unfold(t, mode)
    newshp = [mat.shape[0]] + [shp[i] for i in range(len(shp)) if i != mode]
    return np.moveaxis(res.reshape(newshp), 0, mode)
```

这里最容易踩的坑是 `mode_product` 之后的轴顺序。`tensordot` 会把新维放到最前面，如果不 `moveaxis` 放回原位，后面几个模式的乘积全部错维，而且往往不报错、只是结果悄悄失真。我第一版就是这么把核心张量算歪的，检查重构误差才发现。

构造真值张量时，让 $A$、$B$、$C$ 都取正交列（对随机矩阵做 QR），核心 $\mathcal G$ 取标准正态，再依次做三次模式积，最后整体除以标准差。这样信号方差为 1，后面加噪声时 σ 直接就是信噪比的倒数平方根，便于解读。

## 三、参数效率：为什么"不摊平"能省一个数量级

实验一固定真秩 (3,2,3)，加不同强度的高斯噪声，比较两种方法的重构误差（都是对无噪真值算相对 Frobenius 误差，不是对观测值——后者会被噪声本身迷惑）。

摊平方法沿股票模式展开成 $N\times(P\cdot T)=40\times1800$ 的矩阵，做秩 R 的 SVD。它的问题一眼可见：一个奇异向量就有 1800 个载入，秩 18（等于三个模式秩的乘积，名义上"容量相当"）需要 **33120** 个参数，是 HOSVD 的 63 倍。

```python
def hosvd(X, ranks):
    factors = []
    for k, r in enumerate(ranks):
        U, s, _ = np.linalg.svd(unfold(X, k), full_matrices=False)
        factors.append(U[:, :r].copy())
    core = X.copy()
    for k, U in enumerate(factors):
        core = mode_product(core, U.T, k)
    return core, factors

def matricized_pca(X, rank, mode=0):
    M = unfold(X, mode)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return np.moveaxis(((U[:, :rank]*s[:rank]) @ Vt[:rank])
                       .reshape([X.shape[0]] + list(X.shape[1:])), 0, mode)
```

结果很干脆。σ=0.2 时 HOSVD 误差 0.017、矩阵化 PCA 已是 0.143；σ=1.0 时是 0.088 对 0.709；σ=1.5 时 HOSVD 0.136、矩阵化 1.065——后者误差超过信号本身，说明它把大量噪声拟进了那 33120 个自由参数。这不是算法优劣，是自由度的问题：样本只有 40×15×120=72000 个数，用 33120 个参数去拟合一个只有 528 个自由度的真结构，过拟合是必然。

参数-误差前沿（图 1 右）更直观。HOSVD 在秩网格上扫描，最优点落在 (3,2,4)：528 到 654 个参数之间误差 0.052；矩阵化 PCA 的最优秩是 3，5520 参数、误差 0.166。把矩阵化秩降到 1（1840 参数）误差反而升到 0.638——它没法在"少参数"和"低误差"之间找到一个像样的折中，因为沿一个模式摊平本身就浪费了另外两个模式的结构。

三个模式的奇异值谱（图 4）也印证了结构：股票模式前 3 个奇异值和后面有明显断崖，特征模式是 2 个，时间模式是 3 个。真实金融面板未必这么干净，但这个"看谱定秩"的流程是通用的。

![模式奇异值谱](/images/tensor-factor-model/fig4_spectra.png)

## 四、缺失数据填补：少参数就是正则化

实验二更贴近日常。观测 $X_{\text{obs}}=\mathcal X + 0.5\varepsilon$，再随机挖掉 20%–90% 的条目。填补用交替迭代：先缺失填 0，重复"低秩重构 → 只在观测位置写回真观测值"40 次，最后在留出（被挖掉）的条目上算 RMSE，参照真值而非观测值。

```python
def impute_hosvd(obs, mask, ranks, n_iter=40):
    X = np.where(mask, obs, 0.0)
    for _ in range(n_iter):
        core, fac = hosvd(X, ranks)
        Xr = reconstruct(core, fac)
        X = np.where(mask, obs, Xr)
    return X
```

矩阵化 PCA 用完全相同的迭代框架，只是把 `hosvd` 换成 `unfold + SVD`，秩取 18。这个对比是公平的：同样的观测、同样的迭代次数，唯一差别是低秩结构怎么设。

结果：缺失 20% 时 HOSVD RMSE **0.048**、矩阵化 **0.797**；缺失 60% 时 0.069 对 1.059；缺失 80% 时 **0.104** 对 **1.020**；缺失 90% 的极端情况下 HOSVD 也撑不住，跳到 0.333，但矩阵化仍停在 0.997。

![缺失填补对比](/images/tensor-factor-model/fig2_imputation.png)

差距来自两个地方。第一，HOSVD 只估 528 个参数，观测足够约束它们；矩阵化方法要估 33120 个，稀疏观测下大量自由度没有任何数据支撑。第二，迭代填补本质上是把"未观测位置"交给模型的低秩假设去猜，假设越强（参数越少），猜的方差越低。当缺失率到 90%，每个模式展开矩阵只剩极少数观测，连三个模式的子空间都估不准，HOSVD 的优势自然收敛。

实践里这个结论可以直译：面板越稀疏、历史越短，越该用张量结构而不是摊平。只有当你有海量完整观测、且确信因子结构是全局的时候，摊平方法的额外灵活度才有意义。

## 五、异质载荷：全局低秩的致命假设

实验三把市场改成四个时期，每段 30 天，**股票载荷 $A_q$ 彼此正交且独立**，特征载荷 $B$ 共享。每段内部仍是干净的秩 (3,2,3)，加 0.3 的观测噪声。

这时全局张量的股票模式真实秩变成 $4\times3=12$——四个子空间正交叠加。全局 (3,2,3) 的 HOSVD 只能取其中一个方向，相对误差 **0.748**，几乎等于什么都没提取。而如果已知分段点，对每段单独做 (3,2,3) 的 HOSVD，误差只有 **0.033**。

![异质载荷下的失效](/images/tensor-factor-model/fig3_heterogeneous.png)

有人会想：那我不分段，直接把秩加大不就行了？图 3 右回答了这个问题。同时把股票和时间两个模式的秩从 3 拉到 12，误差才从 0.748 降到 **0.050**，参数量从 528 涨到 **2238**，翻了四倍多。秩 9 时误差还有 0.226——三个正交子空间里只装下两个半，结构性欠拟合不会因为你多给几个自由度就消失。秩 15 相比 12 几乎没有改善（0.052 对 0.050），说明 12 确实是真值。

这里的诚实前提要说清楚：分段方法假设你知道断点在哪，这是上界对比，不是免费午餐。真实市场里制度切换点要先用变点检测去估，误差会高于 0.033。但如果载荷真的随制度漂移，承认它、检测它、分段建模，仍比假装全局低秩、然后花四倍参数去硬拟合要划算得多。

## 六、把这套方法用到真数据前该问的三个问题

第一，**你的面板真的共享同一组载荷吗？** 把时间切成几段，分别看股票模式的奇异值谱。如果各段的主方向长得完全不一样、全局谱又异常平缓，全局低秩假设就值得怀疑。跨行业、跨市场的大型面板尤其危险——不同板块的驱动因子本就不同。

第二，**秩怎么定？** 本文用真值，实盘没有。常用做法是对模式谱做"肘部"判断，或用样本外重构/预测误差做交叉验证。注意张量补全的交叉验证要随机掩码条目，而不是整块切时间，否则会把制度漂移误当成噪声。

第三，**你要的是重构还是要的是收益？** 重构误差低不等于组合赚钱。张量因子更适合当协方差估计和信号去噪的前置步骤：先用 HOSVD 把 $N\times P\times T$ 的面板压成低维表示，再在核心张量上学预测关系。本文没有直接回测，就是因为从"重构准"到"收益高"中间还隔着一层预测假设，混在一起讲会显得讨巧。

## 七、结论

张量因子模型不是矩阵 PCA 的花哨改写，而是一次参数预算的重新分配。当数据真是全局低秩，它用三个模式的和代替摊平后的乘积，在本实验中以 **528 对 5520** 的参数、**0.052 对 0.166** 的误差赢了矩阵化 PCA；噪声放大和条目缺失时优势进一步拉开。代价是它对"全局共享载荷"这个假设极为敏感：载荷一随制度漂移，全局模型误差 **0.748**，必须分段或把秩翻四倍才救得回来。

把它当成一根杠杆：面板越宽、越稀疏、结构越稳定，撬动的效果越大；市场越善变，越要先把"结构稳不稳"这件事检验清楚，再决定要不要上这根杠杆。

## 附：完整可运行代码

```python
import numpy as np

def unfold(t, mode):
    return np.moveaxis(t, mode, 0).reshape(t.shape[mode], -1)

def mode_product(t, mat, mode):
    shp = list(t.shape)
    res = mat @ unfold(t, mode)
    newshp = [mat.shape[0]] + [shp[i] for i in range(len(shp)) if i != mode]
    return np.moveaxis(res.reshape(newshp), 0, mode)

def hosvd(X, ranks):
    factors = []
    for k, r in enumerate(ranks):
        U, _, _ = np.linalg.svd(unfold(X, k), full_matrices=False)
        factors.append(U[:, :r].copy())
    core = X.copy()
    for k, U in enumerate(factors):
        core = mode_product(core, U.T, k)
    return core, factors

def reconstruct(core, factors):
    X = core.copy()
    for k, U in enumerate(factors):
        X = mode_product(X, U, k)
    return X

# 1) 造真值: N=40, P=15, T=120, 真秩 (3,2,3)
rng = np.random.default_rng(20261125)
N, P, T, r1, r2, r3 = 40, 15, 120, 3, 2, 3
orth = lambda n, r: np.linalg.qr(rng.standard_normal((n, r)))[0]
A, B, C = orth(N, r1), orth(P, r2), orth(T, r3)
G = rng.standard_normal((r1, r2, r3))
Xc = mode_product(mode_product(mode_product(G, A, 0), B, 1), C, 2)
Xc /= Xc.std()

# 2) 加噪 -> HOSVD 重构 -> 相对误差
sigma = 0.6
Xo = Xc + sigma * rng.standard_normal(Xc.shape)
core, fac = hosvd(Xo, (r1, r2, r3))
err = np.linalg.norm(reconstruct(core, fac) - Xc) / np.linalg.norm(Xc)
print("HOSVD relative error:", round(err, 4))
# HOSVD relative error: 0.0495（σ=0.6；不同随机流数值略有浮动）
```
