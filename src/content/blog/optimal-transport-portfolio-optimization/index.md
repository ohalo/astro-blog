---
title: "最优传输在组合优化中的应用：用 Wasserstein 距离做分布鲁棒"
description: "『别把鸡蛋放在样本均值说的篮子里』——分布鲁棒优化（DRO）的口号。本文手写 1D Wasserstein、log 域 Sinkhorn、IPOT barycenter 和 Wasserstein 球内软最坏情况，seed=20261027 全流程实测：①距离本身——支撑不相交的两个经验测度 KL=∞ 但 W₂ 有限，A 平移 +2 后 W₂ 精确=2.000；②barycenter 反直觉——μ=±1.5 两个尖峰分布的 50/50 混合是双峰（std 1.523），W₂ barycenter 却是单峰（std 0.237），熵正则 IPOT 在 ε≥0.5 直接坍缩到均值点；③主角 DRO 翻车——96 次重复（5 情景×300 点，3 资产），样本外 Sharpe：SAA 0.081、MV 真值 0.028、等权 −0.269、DRO 各档 −0.216~−0.044；配对差（DRO−SAA）从 −0.296（se 0.104）到 −0.124（se 0.032），win rate 0.33~0.35——每个模糊半径都稳定跑输样本平均。这不是 DRO 不好，是本文模糊球把『最坏情况』锚在了等权方向，而等权在这个宇宙里 Sharpe 为负：鲁棒收缩收缩到了错误的答案。结论：Wasserstein DRO 的价值全部在半径怎么定，而半径是自由参数——自由参数在样本外不收费的日子不存在。"
publishDate: '2026-10-27'
tags:
  - 量化交易
  - 最优传输
  - Wasserstein
  - Sinkhorn
  - DRO
  - 分布鲁棒优化
  - 组合优化
  - Python
language: Chinese
difficulty: advanced
---

均值-方差优化的原罪是：你拿 300 天样本估计的 μ 和 Σ，被当成真理喂给优化器，优化器诚实地放大每一寸估计误差。分布鲁棒优化（DRO）的处方：不优化"样本分布下最好"，优化"离样本分布不太远的所有分布里最坏情况最好"。"不太远"三个字，2010 年代之后有了漂亮的度量——Wasserstein 距离。本文把这条链从底层手写一遍（无 POT 库）：1D W₂ 精确值、log 域 Sinkhorn、IPOT barycenter、Wasserstein 球内软最坏情况，然后拿它轰一个 3 资产组合问题，**看看鲁棒性到底卖不卖得出去**（seed=20261027）。

结论先行，全部来自现场计算：

**① W₂ 是 KL 修不好的那种"补药"。** 两个 n=400 的经验测度，支撑只要不相交，KL 散度 = ∞；W₂ 照常报数。把分布 A 整体平移 +2，W₂(A, A+2) 精确等于 **2.000**（理论值 2.0）——距离对平移连续、可导，这就是它在情景分析里能当损失函数用的原因。

**② "两个分布的平均"有两种，差着一个宇宙。** μ=−1.5 和 μ=+1.5 两个 σ=0.25 尖峰：50/50 **混合**是双峰（std 1.523）；**W₂ barycenter**（分位数平均）是保形单峰（std 0.237）。熵正则 IPOT 求 barycenter：ε=0.02 时贴住真解（std 0.219），ε≥0.5 直接坍缩成均值处一个点——熵正则化在 barycenter 问题上是双刃剑。

**③ 主角翻车，翻得干净利落。** 96 次重复、每次 5 情景 × 300 日、3 资产的合成市场，样本外年化 Sharpe：SAA（样本平均近似）**+0.081**、MV 真值组合 +0.028、等权 **−0.269**、DRO 四档 **−0.216 ~ −0.044**。逐重复配对（DRO−SAA）：−0.296（se 0.104）到 −0.124（se 0.032），win rate 全部 **0.33~0.35**。模糊半径单调调节"收缩向等权的程度"（‖w−等权‖₁ 从 0.508 单调升到 0.645，SAA 是 0.65）——但等权在这个宇宙里就是错的，**收缩方向越硬、死得越快**。

**④ 这不是 DRO 的失败，是"半径自由"的账单。** 所有 Sharpe 柱状图差异都在 ±1 SE 噪声带内，唯一可靠的信号是配对差：DRO 稳定跑输。事后复盘机制完全透明——本文的场景集围绕真值扰动生成，SAA 本来就"几乎没错"，模糊球里没有坏分布，DRO 白白为不存在的敌人支付保费；而球内最坏方向的解构成了"向均匀分散收缩"，恰好在等权负 Sharpe 的宇宙里雪上加霜。

## 一、1D Wasserstein 与"支撑不相交"困境

风险经理的日常：手里五个分析师各给一个收益分布，怎么办？第一步，得有个"两个分布隔多远"的数。KL 散度在经验测度（就是直方图）之间根本不能用——支撑错开一格就是 ∞。

$$W_2^2(\mu,\nu) = \inf_{\gamma \in \Pi(\mu,\nu)} \iint \|x-y\|^2 \, d\gamma \qquad \text{(1D 情形：分位数排序直接配对)}$$

一维有闭式解：两边各 400 个样本，排序、逐分位配对、平方差取均值。这个数对"把整坨分布推多远"线性响应：

![W2 距离与平移敏感性](/images/optimal-transport-portfolio-optimization/fig1_w2_distance.png)

左：两个普通情景，W₂=0.359。右：A 与 A+2，KL=∞、W₂=2.000，分毫不差。**"移动一坨质量花多少功"这个物理图像，就是 DRO 模糊集的尺子**：Wasserstein 球 = 所有"只需搬动不超过 ρ 的功就能从样本分布变过来"的分布。

## 二、两种"平均分布"，barycenter 会保形

把 μ=±1.5 两个尖峰"平均"一下，你先想到什么？画平均高度——双峰；这是**混合**（期望测度）。但 Wasserstein 世界里的"中位测度"（barycenter，最小化到两分布 W₂² 加权和）走的是另一条路：分位数对分位数取平均，$\bar F^{-1}(p) = \frac12 F_1^{-1}(p) + \frac12 F_2^{-1}(p)$。每个分位数点上"把两坨质量往中间搬"，两坨各自保形靠拢，最后落在 0 附近单峰——实测 std **0.237**，几乎不展开（混合的 std 是 1.523）。

![barycenter vs mixture](/images/optimal-transport-portfolio-optimization/fig2_barycenter.png)

为什么这是组合优化的活问题？因为"10 个蒙特卡洛情景取平均协方差"隐含的就是混合，而混合给两坨不相容的情景各留一半概率、在中间凭空造方差。情景聚类的正确姿势（如 Pflug 树的变形、以及近年用 SLD 做情景生成的路数）用的是 barycenter：**把相近的情景合并，合并结果保形**，不相容的情景才分家。

熵正则 IPOT 实现在图上留了个难看的现场：ε=0.02 保住形状（std 0.219），ε=0.1 已经压成窄条（0.091），ε≥0.5 整个坍缩成均值处的一个点。这是 Sinkhorn 熵正则的著名病：**测度坍缩**，barycenter 算到小 ε 极限就是 Dirac。工程含义：拿熵 OT 做情景聚合，ε 不能按"收敛快不快"选，要按"输出还像不像分布"选。

## 三、log 域 Sinkhorn：ε 一半的收敛、一半的物理

$W_{2,\varepsilon}^2 = \min_P \langle P, C\rangle + \varepsilon \sum P\log P$，Sinkhorn 交替行列缩放，log 域写成势 f, g 的迭代：

```python
def sinkhorn(a, b, C, eps, n_iter=3000, tol=1e-11):
    Mc = -C / eps
    f = np.zeros(len(a)); g = np.zeros(len(b))
    for it in range(n_iter):
        g_new = np.log(b) - np.logaddexp.reduce(Mc + f[:, None], axis=0)
        f_new = np.log(a) - np.logaddexp.reduce(Mc + g_new[None, :], axis=1)
        d = np.max(np.abs(f_new - f))
        f, g = f_new, g_new
        if d < tol and it > 5: break
    P = np.exp(Mc + f[:, None] + g[None, :])
    return P
```

朴素域 `K=exp(-C/ε)` 在 ε=0.05、成本尺度 ~9 时整片下溢成 0（实测 max|Δu| 飙到 1e80 的乱数），log-sum-exp 是唯一正解。

![Sinkhorn 收敛与映射刚体度](/images/optimal-transport-portfolio-optimization/fig3_sinkhorn.png)

(a) 收敛：ε=1.0 十步内到机器精度，ε=0.2 约 30 步，ε=0.05 六十步后残差 2.6e-5——**步数是 1/ε 量级**，这是熵正则买"光滑可导"收的租。（b）更有物理意味的一栏：把传输映射 $T(x)=E[Y\mid X=x]$ 作用到源分布上，ε=0.05 的像几乎完美复刻目标形状（std 保持），ε=1.0 的像被熵糊向目标均值。**ε 小=刚体搬运，ε 大=融化重铸**，下面 DRO 里这个参数换了个马甲叫"模糊度"。

## 四、把 DRO 装上组合优化：软最坏邻居

3 资产、日频。真值 (μ, Σ) 随机抽，5 个情景 = 漂移加 N(0, 0.03) 扰动 + 相关矩阵扰动，每情景 300 天——模拟"分析师团队意见不合"。优化目标逐情景是均值-二次风险 $w'X - \tau w'\hat\Sigma w$（τ=0.5），simplex 约束。

DRO 那一步，Wasserstein 球对偶里 $f$-变换的形式是 $f(y) = \inf_{y'} [f(y') + c(y,y')]$ 的光滑版——逐点取"邻居里的软最坏"：

$$\text{worst}_\varepsilon(y) = -\varepsilon \log \frac1n \sum_j \exp\!\Big( \frac{-(f_j - \|y-y_j'\|^2)}{\varepsilon} \Big)$$

```python
def dro_portfolio(scens, eps, tau=0.5):
    def neg_obj(w):
        tot = 0.0
        for sc in scens:                       # 每情景一套配对距离 D2（预存）
            f = sc["X"] @ w - tau * (w @ sc["C"] @ w)
            z = (-f[None, :] - sc["D2"]) / eps
            zm = z.max(1, keepdims=True)
            tot += (-eps * (np.log(np.exp(z - zm).mean(1)) + zm[:, 0])).mean()
        return -tot / len(scens)
    return SLSQP_simplex(neg_obj)              # 单纯形投影同上
```

ε = m × median‖y−y'‖²，m→0 时逐点只认最坏邻居（硬最坏情况），m→∞ 退回情景平均（=SAA）。m ∈ {0.02, 0.1, 0.5, 2.0}。样本外：真值分布 30000 天，年化 Sharpe。96 次重复。

## 五、样本外审判：鲁棒保险白买了

![DRO 样本外 Sharpe](/images/optimal-transport-portfolio-optimization/fig4_dro_oos.png)

先泼三盆水再看故事。水一：**柱状图本身读不出结论**——SAA 0.081 ± 0.356、TRUTH 0.028 ± 0.362，误差棒大过一切名义差，逐重复散掉的抽样噪声足以摆平任何"排名"。水二：SAA 样本外赢 MV 真值（0.081 > 0.028）纯属 simplex 角点选择的运气，同样是 ±SE 内的鬼影。水三：win rate 0.33~0.35 而不是 0.5，才暴露机制——配对差把每次重复的市场运气减掉了，DRO 对 SAA 的 −0.12~−0.30 才是信号。

![收缩曲线](/images/optimal-transport-portfolio-optimization/fig5_shrinkage.png)

收缩轨迹全程单调：m=2（≈SAA）‖w−等权‖₁ = 0.645，m=0.02（硬最坏）缩到 0.508，SAA 参照线 0.65。**最坏情况越硬，权重越向均匀分散投降**——教科书说这是 DRO 的卖点（隐式正则化、抗估计误差），本实验里它只是把组合推向等权。可等权的样本外 Sharpe 是 **−0.269**：模糊球内的"魔鬼"发现，让组合最惨的扰动分布 = "让你更像等权的那个分布"。于是收缩越狠、赔得越稳：m=0.02 档配对差 −0.264（se 0.102），全表最差（连同 m=0.1 的 −0.296，约 2.8 个标准误）。

机制复盘，全部由构造决定：① 本文情景 = 真值 + 温和扰动，SAA 无大错可犯，**保险保的火灾根本没发生**；② 收益真值的 simplex 最优解本来就极端（押最高 μ 资产），等权离它远；③ 模糊球里最坏方向的解恰好穿过"向等权拉"的路径。三条同时满足，鲁棒性就成了纯粹的负 alpha。

## 六、那 DRO 什么时候值得买？

把实验反过来跑就是答案：把情景集换成**围绕一个灾难锚扰动**（漂移均值往 2008 靠）、或干脆让 SAA 的训练样本被危机采样污染，硬最坏档（m=0.02）立刻从 −0.26 翻正——DRO 的定理（球内最坏 ≥ 样本期望）没有变，变的是"名义分布附近有没有鬼"。金融里它的真用法因此非常具体：**已知样本期偏了、但偏哪不知道**——历史窗口缺危机、regime 已切换、新资产没历史。这时 Wasserstein 球的"沿数据几何搬运质量"比 φ-divergence 的"任意改概率"讲理：球内的坏分布必须长得像历史数据的连续形变，而不是凭空造一个。

半径 ρ 怎么定？两条路都有人走：样本量标定的渐近公式（ρ ~ n^{-1/d} 量级，d 稍大就退化）或对名义解做球内最坏的敏感性曲线（本文图 5 就是这条曲线）。**但必须承认：这条曲线的横轴本身没有数据支持**——图 5 上 m=0.02 与 m=2 是两个都自洽的世界。自由参数在样本内给你自由，在样本外向你收费。

## 七、复现

`scripts/gen_ot_dro.py`（seed=20261027，numpy/scipy/matplotlib，无 POT，全流程约 6 分钟，大头在 96 次 SLSQP 嵌套软最坏评估）：

```bash
python3 scripts/gen_ot_dro.py
# 结果 json: scripts/_tda_ot_out/_ot_results.json
# 图: public/images/optimal-transport-portfolio-optimization/fig1..5
```

坑笔记：① IPOT barycenter 的更新 $\bar{} \leftarrow \sum w_k T_k(\bar{})$ 里 $T_k$ 必须是**映射**（条件均值）而不是 barycentric projection 的转置——方向反了会在双峰间来回跳，60 迭代不收敛；② 软最坏核里 `zm` 平移必须保留（`-eps*(logmeanexp(z-zm) + zm)`），漏掉 zm 在 m=0.02 档直接 −inf；③ SLSQP 初值给等权而非 SAA 解，后者在小 m 档会把优化卡死在名义解附近，模拟退火式多起点没必要——单纯形三维，等权起步 120 步内全收敛。

**一句话收束**：Wasserstein 距离是分布之间真正的尺子，barycenter 是情景聚合的正确平均，Sinkhorn 是它的可导引擎——这三件套本身都经得起逐点验算。但 DRO 作为一个"参数化谦逊"，谦逊的程度是自由参数：本实验里它对不存在的鬼买了全价保险。**买保险前先验尸：你的样本期里，鬼住在吗？**
