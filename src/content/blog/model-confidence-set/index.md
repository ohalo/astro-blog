---
title: "模型置信集 MCS：别再从回测里挑冠军，先剪掉确定输的那批"
description: "九个波动率模型跑完样本外，损失最低的那个就是最好的吗？几乎不是。损失差里有抽样噪声，而且各模型的损失高度相关、时间上还自相关，两两 t 检验会集体失灵。模型置信集（Model Confidence Set, Hansen-Lunde-Nason 2003）换个问法：不找冠军，只把『统计上确定更差』的模型一个个剪掉，剩下的那一批就是『以这份数据还分不出高下』的集合。本文用 GARCH+杠杆的合成市场（RV 带测量噪声）从零实现：平稳 bootstrap 重抽时间块、max|t| 等价检验、序贯剔除。2000 天样本外 QLIKE 下 9 个候选被砍到只剩 HAR-RV（首轮等价检验 T=11.33、p=0.000）；换成 MSE 却留下 GARCH/GJR/HAR 三个——结论依赖损失函数，这一点 MCS 不替你决定。噪声扫描显示冠军本身会换人：RV 测量噪声 s=0.15 时 GARCH 胜、-8.3048 对 -8.3028，s≥0.35 后 HAR-RV 一路领先。真正体现 MCS 价值的是势均力敌设定：三个模型挤在 0.00075 的损失差里，30 次独立重复中『单次最优』在 GARCH(4 次)/GJR(11 次)/HAR(15 次) 之间乱跳，而 MCS 平均保留 2.27 个、把六个真输家稳定剪掉。附完整 numpy 实现与五张真实计算图。"
publishDate: '2026-10-26'
tags:
  - 量化交易
  - 波动率
  - 统计检验
  - 模型选择
  - Bootstrap
  - Python
language: Chinese
difficulty: advanced
---

你跑完九个波动率模型的样本外回测，把平均损失排个序，选最低的那个上线。这个动作里藏着两个没人替你检查的假设：一、损失最低=真实最优；二、第二名和第三名的差距是真的。

第一个假设在样本外一样会骗人（损失是随机变量）；第二个更危险——九个模型的损失序列高度相关（它们预测的是同一个东西），并且各自时间上强烈自相关。拿两个模型的损失差去做朴素 t 检验，标准误会严重低估，于是你总能"显著"地选出冠军。

**模型置信集（Model Confidence Set, MCS）**换了个问法：不宣布谁赢，只回答"这批模型里，哪些是统计上确定更差的"。剩下的集合叫置信集——含义是"以这份数据，我还分不出它们的高下"。

## 一、MCS 在做什么

设第 $i$ 个模型在第 $t$ 天的损失为 $L_{i,t}$（我们用 QLIKE：$L=\log f_{i,t}+y_t/f_{i,t}$，$y_t$ 是已实现波动 $RV_t$）。对任意两个模型定义损失差

$$d_{ij,t}=L_{i,t}-L_{j,t},\qquad \bar d_{ij}=\frac1T\sum_t d_{ij,t}$$

MCS 的原假设是**集合 $M$ 内所有模型等价**：$H_0: \mathbb E[d_{ij,t}]=0,\ \forall i,j\in M$。检验统计量取"最难看的一对"：

$$T_M=\max_{i,j\in M}\left|\frac{\bar d_{ij}}{\widehat{\mathrm{se}}(\bar d_{ij})}\right|$$

关键是标准误怎么算。损失差有强时间依赖，不能用 $s/\sqrt T$，得上**平稳 bootstrap**（Politis-Romano 1994）：随机起点、块长服从几何分布、结尾回绕，重抽出的时间索引同时施加给所有模型——这一点必须做到，否则横截面的相关性被破坏，检验会过度拒绝。bootstrap 给出 $T_M$ 的零分布，$p$ 值就是 $\Pr(T^\ast_M>T_M)$。

$p<\alpha$ 就拒绝"全等价"，然后剔除**最差**的那个：$e_i=\max_{j\neq i} t_{ij}$ 最大者（它显著地比集合里某个成员差）。剩下的集合重跑一遍，直到 $p\ge\alpha$ 停手。每个模型拿到一个 $p$ 值——被剔除的记剔除那一步的 $p$，留在最后的记为最终 $p$。

几个实操细节：$\alpha$ 通常取 0.10 而不是 0.05，因为这个检验天生保守（它要的是"集合里所有模型等价"，集合越小越难拒绝）；统计量也可以用半二次型 $T_{SQ}$ 代替 $\max|t|$，前者对"多个模型一起略差"更敏感，后者对"某一个明显拉胯"更敏感。方法谱系上，Diebold-Mariano 只比两个模型、White's Reality Check 与 SPA 是"一堆模型对标一个基准"，MCS 则是"一堆模型互相比、且不预设基准"——当你没有公认的基准模型时，MCS 才是合适的那一档。

## 二、实验设置

数据生成过程是带杠杆项的 GARCH：$\omega=4\times10^{-6},\alpha=0.06,\beta=0.88,\gamma=0.04$，日收益标准差 0.99%，长期日方差 $10^{-4}$。已实现波动按计量文献的常见设定造：$RV_t=h_t e^{u_t},\ u_t\sim N(-s^2/2,s^2)$，保证 $\mathbb E[RV_t|h_t]=h_t$，$s$ 就是测量噪声强度。共 4200 天，最后 2000 天做样本外。

九个候选：GARCH(1,1)、GJR-GARCH（都滚动 1250 日窗口重估 QMLE）、HAR-RV（日/周/月三个已实现波动分量）、EWMA-RV(0.94)、EWMA-RV(0.97)、EWMA-$r^2$(0.94)、MA22-RV、AR(1)-logRV、常数方差。全部严格只用 $t$ 之前的信息预测 $RV_t$。

![一步预测 vs 已实现波动](/images/model-confidence-set/forecast-path.png)

## 三、2000 天样本外：砍到只剩一个

QLIKE 排序（越低越好，括号内是 MCS $p$ 值）：HAR-RV −8.3119（1.00）、GJR-GARCH −8.3058（0.000）、GARCH(1,1) −8.3046（0.000）、EWMA-$r^2$ −8.2901（0.000）、AR(1)-logRV −8.2719（0.000）、MA22-RV −8.2715（0.000）、EWMA-RV(0.94) −8.2764（0.000）、EWMA-RV(0.97) −8.2659（0.000）、常数方差 −8.2362（0.000）。

首轮 $H_0$"九个全等价"：$T_M=11.33$，2000 次 bootstrap 的 $T^\ast$ 分布里没有一个超过它，$p=0.000$，直接拒绝。剔除顺序是 AR(1)-logRV → MA22-RV → EWMA-RV(0.94) → EWMA-RV(0.97) → EWMA-$r^2$ → 常数方差 → GARCH(1,1) → GJR-GARCH，最终集合只剩 HAR-RV。

![平均损失与 MCS p 值](/images/model-confidence-set/loss-and-pvalues.png)

![|t| 矩阵与 bootstrap 零分布](/images/model-confidence-set/tmatrix-bootstrap.png)

**换损失函数，结论就变了。** 改用 MSE（$\times10^8$）：HAR-RV 0.1593、GJR 0.1639、GARCH 0.1698、EWMA-$r^2$ 0.2011、AR(1) 0.2658、EWMA-RV 0.2638、MA22 0.2813、EWMA-RV(0.97) 0.2944、常数 0.3624。MCS 这次留下 **三个**：{GARCH(1,1), GJR-GARCH, HAR-RV}。QLIKE 对低估惩罚重、MSE 对极端值敏感，两者给出的"可分性"不同。MCS 只负责在你选定的损失口径下做剔除，选什么损失是你的责任，别指望它替你兜住。

块长在这一轮实验中不敏感：期望块长 $L_b$ 从 1 到 100 全部砍到只剩 HAR-RV。原因是损失差足够大、结论在边界之外，bootstrap 的依赖结构怎么设都翻不了案。稳健性是好事，但别把它读成"块长不重要"——下一节你会看到它什么时候重要。

## 四、三个"看起来很像证据"的陷阱

**陷阱一：单日胜负几乎全是噪声。** 统计每个模型在多少个交易日拿到当日最低损失，结果是 HAR-RV 19.5%、EWMA-$r^2$ 18.4%、GJR 12.7%、常数方差 11.3%、AR(1) 11.0%、GARCH 8.7%、MA22 8.2%、EWMA-RV(0.97) 5.9%、EWMA-RV(0.94) 4.3%。就算平均损失最低的 HAR-RV，也有八成的日子输给别人；连常数方差模型都有 11.3% 的日子取胜。任何"今天谁准"的看板都是在读噪声。

**陷阱二：朴素 t 检验的偏差方向不固定。** 拿损失差直接做 iid t 检验，和 bootstrap 长程标准误比一比：HAR-RV 对 EWMA-$r^2$ 是 9.34 vs 6.31（朴素放大 1.48 倍），EWMA-$r^2$ 对常数方差是 9.66 vs 4.30（放大 **2.25 倍**）；可 GJR 对 GARCH 却几乎没差（1.63 vs 1.64），HAR 对 GJR 甚至反过来（−3.68 vs −4.01）。问题不在于它总偏大，而在于**你不知道它偏多少**——这正是不能用的理由。

**陷阱三：排名一致 ≠ 可分性一致。** 本实验中 QLIKE 与 MSE 给出的九个模型排名完全相同（Spearman 相关系数 1.000，连顺序都一模一样），但 MCS 集合一个是 1 个、一个是 3 个。排序只告诉你谁在前，MCS 告诉你"前几名是不是真的分得开"——后者才是上线决策需要的。

![累积损失优势曲线](/images/model-confidence-set/cumulative-loss.png)

累积优势曲线（相对常数方差基准）也值得看：GARCH 相对 HAR-RV 的累积优势符号只变化 3 次、GJR 7 次、EWMA-$r^2$ 5 次，而 HAR-RV 在 **96.6%–97.1%** 的交易日里保持累积领先。也就是说，早期的领先归属确实有偶然性（交叉都发生在样本前段），但一旦样本够长，优势就稳定下来不再回头——这和长度扫描里"样本越长砍得越狠"是同一件事的两面。

## 五、冠军是会换人的：噪声扫描与势均力敌设定

把 RV 测量噪声 $s$ 从 0.15 拉到 0.90：$s=0.15$ 时 GARCH −8.3048 略胜 HAR-RV −8.3028，MCS 只剩 GARCH；$s=0.35$ 起 HAR-RV 反超（−8.3197 对 −8.3111），$s=0.90$ 时拉开到 −8.4387 对 −8.3381，MCS 每次都只剩 HAR-RV。**谁"最好"取决于数据的测量质量，不是模型的固有属性**——RV 越脏，越要靠 RV 序列自身的结构（HAR 的日/周/月分量），干净的时候收益数据本身就够了。

真正能体现 MCS 价值的是**势均力敌**的情形。取 $s=0.19$，三个模型挤在一起：GJR-GARCH −8.307458、GARCH(1,1) −8.306214、HAR-RV −8.305461，GARCH 与 HAR 的平均损失差只有 −0.00075。这时候：

- **样本长度扫描**：250 天外样本 → MCS 保留 3 个 {GJR, HAR, EWMA-$r^2$}；500/1000 天 → 2 个 {GJR, HAR}；1500/2000 天 → 1 个 {GJR}。功效是拿样本长度换来的，这点非常直观。
- **30 次独立重复（每次 500 天样本外）**：按"损失最低"选冠军，结果是 GARCH(1,1) 4 次、GJR-GARCH 11 次、HAR-RV 15 次——**冠军在这三个之间乱跳，等于抽奖**；而 MCS 平均保留 2.27 个，入选率 HAR-RV 27/30、GJR 25/30、GARCH 14/30，另外六个模型最多只入选 1 次。

![样本长度扫描与 30 次重复实验](/images/model-confidence-set/robustness.png)

这就是 MCS 的真正用法：它不告诉你"选 HAR"，它告诉你"EWMA、MA22、AR(1)、常数方差这六个是真输家，可以放心扔；前三个以你现在的数据量分不出来——要么继续攒样本，要么承认它们等价，用实现成本、鲁棒性这些损失以外的标准去选"。把 2.27 个候选交给人做最后决定，比假装冠军是确定的要诚实得多。

## 六、代码：MCS 的 numpy 实现

```python
import numpy as np

def stationary_bootstrap_idx(T, p, rng):
    """Politis-Romano 平稳 bootstrap：随机起点 + 几何块长 + 回绕"""
    idx = np.empty(T, dtype=np.int64); t = 0
    while t < T:
        start = rng.integers(0, T)
        blk = rng.geometric(p)            # p = 1 / 期望块长
        end = min(t + blk, T)
        idx[t:end] = (start + np.arange(end - t)) % T
        t = end
    return idx

def mcs(L, B=2000, Lb=20, alpha=0.10, seed=7):
    """L: (T, K) 损失矩阵。返回 (剔除顺序, p 值字典, 存活集合)"""
    T, K = L.shape
    mu = L.mean(0)
    rng = np.random.default_rng(seed)
    # 关键：一次重抽的时间索引同时施加给所有模型，保留横截面相关
    M = np.empty((B, K))
    for b in range(B):
        M[b] = L[stationary_bootstrap_idx(T, 1.0 / Lb, rng)].mean(0)
    D = M - mu                                        # (B, K) 中心化 bootstrap 均值
    alive = list(range(K)); order = []; pvals = {}
    while len(alive) > 1:
        A = np.array(alive); m = len(A)
        iu = np.triu_indices(m, 1)
        d_obs = mu[A][:, None] - mu[A][None, :]       # 观测到的两两损失差
        db = D[:, A]
        vb = db[:, :, None] - db[:, None, :]          # bootstrap 版本
        var = (vb ** 2).mean(0) + 1e-18               # 长程方差，来自 bootstrap
        t_ij = d_obs / np.sqrt(var)
        Tstat = np.abs(t_ij[iu]).max()                # max|t| 统计量
        Tb = np.abs(vb[:, iu[0], iu[1]] / np.sqrt(var[iu])).max(1)
        pv = float((Tb > Tstat).mean())
        if pv >= alpha:                               # 剩下的分不出高下，收工
            for k in A:
                pvals[int(k)] = pv
            break
        e = np.array([t_ij[i][np.arange(m) != i].max() for i in range(m)])
        worst = int(A[int(np.argmax(e))])             # 剔除"最显著地更差"的那个
        order.append(worst); pvals[worst] = pv; alive.remove(worst)
    return order, pvals, alive

# 用法：L = np.column_stack([qlike(rv_oos, f) for f in forecasts])
# order, pvals, mcs_set = mcs(L, B=2000, Lb=20, alpha=0.10)
```

## 七、落地清单与诚实边界

1. **损失函数先定死再跑**。QLIKE 与 MSE 在本实验中给出完全不同的集合（1 个 vs 3 个），事后挑一个"看起来合理"的损失，MCS 就退化成数据窥探。
2. **候选集必须事先固定**。MCS 的覆盖率是对"给定候选集"成立的，跑完再加一个模型进来，等于又一次多重比较。
3. **别再用两两 t 检验**。序贯剔除 + max 统计量已经把多重比较和横截面相关算进去了，朴素检验只会让你过度自信。
4. **块长要覆盖依赖长度**。波动率损失的自相关能拖几十天，$L_b$ 取 1 等于假设iid。本次因为差异太大看不出影响，势均力敌时它会直接决定结论。
5. **MCS 不输出冠军**。它的输出是"还不能区分的一批"——集合大小为 1 只是说明这份数据够强，不是说明那个模型真的唯一最优（换损失函数就破了）。
6. **损失必须平稳**。结构性断裂会把 MCS 变成"剔除旧制度下表现差的模型"，先用断点检测划段再跑，或者干脆对"每个制度各跑一次"分别给集合。
7. **把 MCS 的输出当成决策输入，不是决策本身**。它给出的是"数据能支撑到哪一步"的边界；最后从边界里挑哪个，要结合实现成本、数据依赖、运维复杂度和你对模型结构的先验——这些统计量一个都说不了。

---
本内容仅为信息整理与分析参考，不构成投资建议，投资有风险，决策需谨慎。
