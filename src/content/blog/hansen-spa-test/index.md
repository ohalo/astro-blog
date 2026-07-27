---
title: "Hansen SPA 检验：在一堆策略里判断有没有真正跑赢基准的"
publishDate: '2026-07-27'
description: "Hansen SPA 检验：在一堆策略里判断有没有真正跑赢基准的 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

先说结论：**White Reality Check（WRC）有一个致命软肋——池子里塞进越多明显亏钱的垃圾策略，它对真信号越迟钝；Hansen 在 2005 年提出的 SPA（Superior Predictive Ability）检验用「学生化统计量 + 差策略剔除」补上了这个洞**。本文的模拟给出直接证据：一个夏普 0.9 的真 alpha 混在 20 个噪声策略里，WRC p 值 0.034 尚能识别；往池子里再倒进 150 个年化 -25% 的垃圾策略后，WRC p 值恶化到 0.224，真信号被淹没——而 SPA 的 p 值全程稳定在 0.03-0.04，几乎不受垃圾策略污染。

## WRC 留下的两个洞

在[上一篇 White Reality Check](/blog/white-reality-check/) 里我们讲过：当你从 M 个策略里挑出样本内最优者时，正确的问题不是"它是否显著"，而是"**它是否好到超越了 M 次搜索本身能带来的运气**"。WRC 的答案是为"最大统计量"构造自助法零分布。

但 Hansen（2005, *Journal of Business & Economic Statistics*）指出，WRC 的构造在两处不够精细：

**第一个洞：统计量没有学生化。** WRC 原版用的是 $\sqrt{T}\,\bar{d}_k$（平均超额收益直接放大），不除以各策略自身的波动。后果是高波动策略在 max 运算里天然占便宜——一个波动 40% 的策略随便晃两下就能产生很大的绝对均值，把零分布的右尾撑宽，反过来稀释了低波动真信号的显著性。

**第二个洞：所有策略都被重心化到零。** WRC 构造零分布时，把全部 M 个策略的均值都拉回 0（"最不利的零假设"配置）。这意味着一个年化 -25%、显然毫无希望的垃圾策略，在零分布里也被当成"期望恰好为零"的候选者参与 max 竞争。垃圾策略越多，零分布的右尾被推得越远，检验越保守——**你往池子里扔垃圾，反而帮所有策略"洗白"了**。

第二个洞在实务里格外致命。真实的参数扫描池从来不是精心筛选过的：网格搜描出来的大部分参数组合是明显亏钱的。按 WRC 的逻辑，这些注定失败的组合会系统性地钝化检验。

## SPA 的两个修补

### 修补一：学生化

SPA 把统计量换成每个策略各自标准化后的形式：

$$T_n^{SPA} = \max_k \frac{\sqrt{T}\,\bar{d}_k}{\hat{\omega}_k}$$

其中 $\hat{\omega}_k$ 是第 $k$ 个策略超额收益的长程标准差估计（需容忍序列相关，实务用自助法样本直接估计）。学生化之后，每个策略在 max 竞争里以"t 统计量"而非"绝对均值"参赛，高波动策略的天然优势被消除。

### 修补二：差策略不参与零分布抬升

这是 SPA 的核心创新。构造零分布时，不再把所有策略都重心化到零，而是用一个数据驱动的阈值判断哪些策略"明显差"：

$$\hat{\mu}_k^c = \bar{d}_k \cdot \mathbf{1}\left\{ \frac{\sqrt{T}\,\bar{d}_k}{\hat{\omega}_k} \le -\sqrt{2\ln\ln T} \right\}$$

翻译成人话：如果一个策略的 t 统计量低于 $-\sqrt{2\ln\ln T}$（T=1250 时约 -1.97），就认定它"显著差于基准"，在零分布里保留它的负均值——它的自助法样本会以深负值为中心波动，对 max 几乎没有贡献；其余策略（好的和分不清的）照旧重心化到零。

阈值 $\sqrt{2\ln\ln T}$ 来自重对数律（law of the iterated logarithm），保证渐近下差策略被正确剔除、边界策略被正确保留。这个构造让零分布只由"真正有竞争力的策略"撑起来，垃圾策略再多也推不动右尾。

## 代码：WRC 与 SPA 的并排实现

```python
import numpy as np

T, B, BLOCK = 1250, 500, 20          # 5 年日频，500 次自助，块长 20
SIG, ANN = 0.01, np.sqrt(252)

def block_bootstrap_idx(rng, T, block):
    """circular block bootstrap 的索引"""
    n_blocks = T // block + 1
    starts = rng.integers(0, T, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
    return idx[:T]

def wrc_spa(X, B=B, seed=1):
    """X: (T, M) 各策略相对基准的超额收益。返回 (wrc_p, spa_p)"""
    rng = np.random.default_rng(seed)
    Tn, M = X.shape
    mu, sd = X.mean(0), X.std(0, ddof=1)
    stat_obs = np.sqrt(Tn) * mu / sd              # 学生化观测统计量
    V_obs = max(stat_obs.max(), 0.0)

    # SPA 的差策略判定：t 统计量 <= -sqrt(2 lnln T)
    thresh = -np.sqrt(2 * np.log(np.log(Tn)))
    is_junk = stat_obs <= thresh

    V_wrc, V_spa = np.empty(B), np.empty(B)
    for b in range(B):
        Xb = X[block_bootstrap_idx(rng, Tn, BLOCK)]
        # 以观测均值为中心的自助波动（= 重心化到 0 的零分布样本）
        z = np.sqrt(Tn) * (Xb.mean(0) - mu) / Xb.std(0, ddof=1)
        V_wrc[b] = max(z.max(), 0.0)              # WRC：全部重心化
        z_spa = z + np.where(is_junk, stat_obs, 0.0)  # SPA：差策略移回负中心
        V_spa[b] = max(z_spa.max(), 0.0)
    return np.mean(V_wrc >= V_obs), np.mean(V_spa >= V_obs)
```

三点实现说明。其一，这里两套检验都用了学生化统计量，所以对比隔离出来的是**第二个修补（差策略剔除）单独的贡献**——原版 WRC 不学生化，实务表现比这里演示的更差。其二，`z + is_junk * stat_obs` 是"差策略以其观测负统计量为中心波动"的直接实现：它们的自助样本大部分深埋在负值区，抢不到 max。其三，块自助（block bootstrap）保留收益的序列相关结构，块长 20 对日频数据是常用起点。

## 实验一：垃圾策略如何毒化 WRC

构造一个池子：20 个纯噪声策略（期望为零）+ 80 个垃圾策略（年化约 -25%）+ 1 个夏普 0.9 的真 alpha，5 年日频。

真 alpha 的观测 t 统计量是 2.98。两套检验对同一个观测值给出的判决：**WRC p = 0.130（不显著，真信号被判死刑）；SPA p = 0.030（显著）**。

![80 个垃圾策略把 WRC 零分布推向右侧，SPA 几乎不受影响](/images/hansen-spa-test/spa-null-dist.jpg)

图中两个零分布的差异就是全部答案：红色的 WRC 零分布因为 80 个垃圾策略全部被重心化到零参与 max 竞争，右尾明显更肥——101 个"期望为零"的策略里挑最大，运气本身就能碰到 2.98；蓝色的 SPA 零分布把 80 个垃圾策略按其真实的深负中心处理，实际参与竞争的只有 21 个策略，2.98 在这个零分布里就是显著的。

## 实验二：p 值随垃圾数量的变化曲线

固定 20 个噪声策略和同一个夏普 0.9 的真 alpha（随机种子不变，观测统计量恒为 2.98），只改变垃圾策略数量：

| 垃圾策略数 | WRC p 值 | SPA p 值 |
|---|---|---|
| 0 | 0.034 | 0.028 |
| 10 | 0.046 | 0.028 |
| 20 | 0.066 | 0.030 |
| 40 | 0.088 | 0.034 |
| 80 | 0.130 | 0.030 |
| 150 | 0.224 | 0.044 |

![垃圾策略越多 WRC 越迟钝，SPA 稳定显著](/images/hansen-spa-test/spa-pvalue-vs-junk.jpg)

WRC 的 p 值随垃圾数量单调恶化：从 0.034 一路爬到 0.224，在垃圾数超过 20 后就跌破了 0.05 显著性线——**同一个真 alpha，同样的观测数据，仅仅因为池子里多了些注定亏钱的参数组合，就从"显著"变成"不显著"**。SPA 全程压在 0.05 以下，最大波动不过 0.028→0.044。

这张图对实务的含义很直接：如果你用 WRC 检验参数网格扫描的结果，你实际上在惩罚"扫得宽"——网格越大、包含的差参数越多，检验越难通过。SPA 把这个惩罚精确限制在"有竞争力的参数"范围内。

## SPA 的三个 p 值：自带的稳健性区间

Hansen 还设计了一个聪明的副产品。差策略判定阈值毕竟是个渐近构造，有限样本下总有策略处于"分不清是差还是零"的灰色地带。SPA 于是同时报告三个 p 值：

- **SPA_l（lower）**：所有均值为负的策略都按其观测负中心处理——最激进的剔除，p 值下界；
- **SPA_c（consistent）**：按 $\sqrt{2\ln\ln T}$ 阈值剔除——推荐的主报告值；
- **SPA_u（upper）**：不剔除任何策略，全部重心化到零——退化回（学生化版的）WRC，p 值上界。

![SPA 的三个 p 值：consistent 夹在 lower 与 upper 之间](/images/hansen-spa-test/spa-three-pvalues.jpg)

三种场景下的表现印证了设计意图：纯噪声池（50 个零期望策略）三个 p 值都在 0.34-0.55，一致不显著，没有假阳性；"噪声+垃圾"池里 upper 被垃圾推到 1.00 而 lower/consistent 保持诚实；含真 alpha 的污染池里 lower=0.020、consistent=0.048、upper=0.132——**结论敏感性一目了然：如果连 upper 都显著，结果无可争议；如果只有 lower 显著，你该怀疑结论依赖剔除规则**。

三个 p 值的间距本身就是诊断信息。间距大，说明池子里灰色地带策略多、结论对剔除阈值敏感；间距小，说明池子结构清晰、结论稳健。

## 与其他多重检验工具的分工

这是本站回测统计检验系列的收官定位：

- [White Reality Check](/blog/white-reality-check/)：回答"最优者是否显著"，历史起点，但受垃圾策略污染；
- **Hansen SPA（本文）**：同样回答"有没有至少一个跑赢基准"，学生化 + 差策略剔除，功效更高，是 WRC 的直接升级替代；
- [多重检验夏普折扣](/blog/haircut-sharpe-multiple-testing/)：不做完整检验，直接给最优夏普打折，快速粗筛；
- [Deflated Sharpe Ratio](/blog/deflated-sharpe-ratio/)：把搜索次数、偏度、峰度一起折进单个策略的夏普显著性；
- [PBO](/blog/pbo-overfitting-probability/) 与 [CPCV](/blog/combinatorial-purged-cv/)：换一个角度，问"样本内最优在样本外沦为平庸的概率"。

工作流上的建议分工：参数扫描完成后先跑 SPA 回答"这堆里到底有没有真东西"（没有就整族放弃，别浪费时间挑）；有真东西再用 PBO/CPCV 评估选择过程的过拟合程度；最后对选定的那一个用 DSR 报告校正后的显著性。

## 使用清单

实务落地的几个要点：

1. **基准要明确**。SPA 检验的是"相对基准的超额收益"，$d_{k,t} = r_{k,t} - r_{0,t}$。基准可以是买入持有、零收益（纯多空策略）或已有的生产策略。基准选错，整个检验答非所问。
2. **完整申报策略池**。和 WRC 一样，SPA 只能校正你告诉它的搜索。试过 500 组参数只喂给它 50 组，p 值照样被低估。差策略剔除减轻的是"垃圾稀释功效"问题，不是"隐瞒搜索"问题——后者无药可医。
3. **块长要做敏感性检查**。块自助的块长影响零分布的形状，日频数据从 10-40 各试一遍，结论应当稳定；不稳定说明序列相关结构复杂，考虑 Politis-Romano 平稳自助。
4. **报告三个 p 值**。只报 consistent 是合规的，但三个一起报能让读者自己判断结论对剔除规则的敏感度。

## 局限

SPA 不是终点。第一，它只回答"**有没有**至少一个策略跑赢基准"，不回答"**哪些**"——后者需要 Romano-Wolf StepM 这类逐步多重检验程序。第二，差策略剔除阈值是渐近论证，样本短（比如两年以内日频）时灰色地带很宽，三个 p 值会散得很开，此时结论本质上不可靠。第三，块自助假设平稳性，跨越重大结构断裂（政策换轨、交易机制改革）的样本会让零分布失真。第四，和所有频率派检验一样，"显著"只说明不太可能纯靠运气，不说明未来能赚钱——regime 变了，真 alpha 也会死。

顺序永远是：SPA 告诉你这堆里值不值得挑，[样本外验证](/blog/purged-kfold-cv/)告诉你挑出来的那个值不值得上钱。

## 参考文献

1. Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*, 23(4), 365-380.
2. White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126.
3. Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. *Econometrica*, 79(2), 453-497.
4. Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica*, 73(4), 1237-1282.
