---
title: "PC 算法的功效边界：样本量、维度与显著性水平如何决定骨架还原"
publishDate: 2026-11-01
language: zh
tags: [因果推断, 因子模型, 统计检验, PC算法, 量化]
difficulty: advanced
description: "前篇在六因子 DAG 上做了一次单点还原，本篇系统追问：PC 算法在什么条件下会失效？修正出处与偏相关公式后，用 Fisher-z 加 PC-stable 骨架，在随机线性 SEM 上扫样本量、维度与显著性水平，共 264 次独立还原。实测：n 从 100 增到 5000 时召回率 0.7186 升到 0.9179，但 SHD 始终不归零；α=0.005 优于惯用的 0.05；p 从 6 增到 16 检验次数涨 36 倍、SHD 从 0 涨到 32.3。collider 处边际相关 0.0366（p=0.247）而条件偏相关 −0.8308（p 下溢）。诚实边界：只评估骨架、不定向、无隐变量与线性高斯假设、全部为模拟数据。附完整可运行代码、结果 JSON 与五张真实计算图。"
image: "images/pc-algorithm-causal-factor/cover.png"
---

## 一、结论先行：PC 的瓶颈是样本量，不是调参

本站 10 月 26 日那篇《因果发现在因子选择中的 PC 算法》用六因子 DAG 演示了 collider 如何骗过相关矩阵，并在单次设定下还原出 CPDAG、验证了下游选因子的差异。那篇是「一次成功的案例」。本篇要问的是一个更该先问的问题：**这个成功有多脆弱？**

答案是——脆弱在样本量和维度上，而几乎不脆弱在 α 上。把随机 DAG 重复跑 264 次独立还原后：样本量从 100 涨到 5000，边召回率从 0.7186 升到 0.9179，结构汉明距离（SHD）从 3.25 降到 1.15，但**始终没有归零**；显著性水平从 0.001 扫到 0.2，最优值是 0.005 而不是教科书常用的 0.05，且相邻档位的 SHD 差异很小；变量数从 6 涨到 16，条件独立检验次数从 83 次涨到 3027 次（36 倍），SHD 从 0 暴涨到 32.3。

所以工程排序是：**先把样本量堆够、把维度压下来，最后才轮到调 α**。反过来的顺序，是绝大多数因子研究里 PC 算法跑出「一堆说不通的边」的根因。

## 二、先把出处和数学摆正

PC 算法由 Peter Spirtes 与 Clark Glymour 在 1990 年前后提出（系统化表述见 Spirtes、Glymour 与 Scheines 的《Causation, Prediction, and Search》，1993/2000），**PC 就是两人姓氏的首字母**。它基于因果充分性、马尔可夫条件与忠诚性假设，通过逐层加深的条件独立检验，把全连接图削成骨架（skeleton），再用 v-structure 与 Meek 规则定向为 CPDAG。

条件独立检验是整套算法的地基。对连续变量，正确的是**基于精度矩阵的偏相关**：

$$\Omega = \Sigma_S^{-1},\qquad \rho_{ij\cdot Z} = -\frac{\Omega_{ij}}{\sqrt{\Omega_{ii}\Omega_{jj}}}$$

其中 $S=\{i,j\}\cup Z$，$\Sigma_S$ 是这些变量的协方差矩阵。常见的等价做法是先把 $i,j$ 分别对 $Z$ 做线性回归，再对两个残差求相关。而把它写成 $r_{XY\cdot Z}=\rho_{XY\cdot Z}/\sqrt{\rho_{XX\cdot Z}\rho_{YY\cdot Z}}$ 是错的——那是把普通相关系数的定义硬套在偏相关上，$\rho_{XX\cdot Z}$ 恒等于 1，式子退化成同义反复，没有任何信息。

显著性判断走 Fisher-z 变换：

$$z = \frac{1}{2}\ln\frac{1+\rho}{1-\rho}\,\sqrt{n-|Z|-3},\qquad p = 2\Phi(-|z|)$$

$p>\alpha$ 判为条件独立，删除该边。这里也不能用「$|r|<t_\alpha$」这类写法：$r$ 落在 $[-1,1]$，$t_\alpha$ 通常是 1.96 量级，量纲都对不上，照此判定几乎会把所有边一次删光。另外必须用**双尾** p 值，单尾会把负相关整片放过。本篇代码用 `np.linalg.pinv` 求精度矩阵以规避小样本下的奇异，并用 `2*stats.norm.sf(abs(z))` 保证双尾。

**范围说明**：本篇只评估**骨架还原**（哪些边存在），不做 v-structure 定向与 CPDAG——那部分请见 10 月 26 日前篇。因此下面的 precision / recall / SHD 都只就无向边计算，SHD 在此等于假阳边数加漏检边数。

![真值骨架与 PC-stable 还原结果，橙色虚线为唯一漏检边](/images/pc-algorithm-causal-factor/pc_1_dag.png)

## 三、实验设计：让每次还原都有已知真值

真值 DAG 按拓扑序随机生成：变量 $j$ 从 $0,\dots,j-1$ 中按密度约 0.4 抽父母，系数取 $\pm[0.35,0.7]$，再按线性 SEM 逐列生成 $X_j=\sum_i a_{ij}X_i+\sqrt{1-\sum a_{ij}^2}\,\epsilon_j$，最后整体标准化。这样每条真实边、每个 collider 都是已知答案。

算法实现的是 **PC-stable**：在第 $k$ 层，用本层开始时的邻接快照枚举条件集（而不是边删边更新邻接），避免删除顺序影响结果；条件集只从 $i$ 的当前邻居里挑（不含 $j$），大小从 0 逐层加到 `max_k`。评估指标为 precision（找到的边中真边占比）、recall（真边被找回的比例）与 SHD。

四组实验：样本量扫描 $n\in\{100,200,500,1000,2000,5000\}$（$p=8$、$\alpha=0.05$、每档 20 次重复）；α 扫描 $\{0.001,\dots,0.2\}$（$n=1000$、每档 20 次）；维度扫描 $p\in\{6,8,12,16\}$（$n=1000$、$\max_k=2$、每档 6 次）；外加一次 $n=2000$ 的具体还原案例与 collider 反转演示。合计 264 次独立还原，全部纯 numpy/scipy 现场计算，结果写进同目录 `results.json`。

## 四、先看清 collider 为什么必须做条件检验

构造 $X_1\to X_3\leftarrow X_2$，$X_3=0.7X_1+0.7X_2+0.3\epsilon$，而 $X_1,X_2$ 独立。实测：

| 样本量 | 边际相关 $r$ | 边际 p | 条件偏相关 $\rho_{12\cdot3}$ | 条件 p |
|---|---|---|---|---|
| 200 | 0.112349 | 0.113290 | −0.851820 | 下溢为 0 |
| 1000 | 0.036624 | 0.247305 | **−0.830760** | 下溢为 0 |
| 5000 | −0.009266 | 0.512443 | −0.844581 | 下溢为 0 |

![collider 反转：边际独立，给定共同结果后强相关](/images/pc-algorithm-causal-factor/pc_5_collider.png)

边际上 $X_1,X_2$ 干净得像两张不相干的资产负债表（$n=1000$ 时 $r=0.0366$、$p=0.247$，判定独立），可一旦把共同结果 $X_3$ 放进条件集，偏相关立刻跳到 −0.8308，p 值小到浮点下溢。这就是「解释掉共同结果后，两个原因此消彼长」的碰撞器效应。只看相关矩阵选因子，会系统性漏掉这类结构；而条件独立检验正是 PC 唯一在做的事。

## 五、结果一：样本量是唯一真正管用的旋钮

| $n$ | 100 | 200 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| precision | 0.9682 | 0.9537 | 0.9685 | 0.9830 | 0.9846 | 0.9813 |
| recall | 0.7186 | 0.8023 | 0.8159 | 0.8844 | **0.9287** | 0.9179 |
| SHD | 3.25 | 2.60 | 2.35 | 1.45 | **0.90** | 1.15 |

![样本量功效曲线：召回率随 n 上升，SHD 不归零](/images/pc-algorithm-causal-factor/pc_2_n_power.png)

三条规律值得记住。第一，**precision 从一开始就很稳**（0.95 以上），PC 很少凭空造边，错误主要来自漏检——小样本下 Fisher-z 的方差大，弱边的偏相关被当成噪声。第二，recall 随 $n$ 单调改善，$n=100$ 时只找回 71.86% 的边，$n=2000$ 时到 92.87%。第三，**SHD 在 $n=5000$ 反而比 $n=2000$ 略高**（1.15 对 0.90），这不是算法退化，是 20 次重复的蒙特卡洛波动（$n=5000$ 的 SHD 标准差为 1.014，大于两档均值之差）。换句话说，在 $n\ge 2000$ 之后，剩余误差已经进入「靠重复次数也难以稳定区分」的区域，继续堆样本的边际收益很低。

给因子的直接含义：拿 100 条月度样本去跑几十个因子，漏掉近三成真实依赖是**统计上的必然**，不是代码写错。

## 六、结果二：α 主要换 precision 与 recall，不是换正确率

| $\alpha$ | 0.001 | 0.005 | 0.01 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|
| precision | 0.9817 | **1.0000** | 0.9616 | 0.9756 | 0.9714 | 0.9156 |
| recall | 0.8354 | 0.8628 | 0.8439 | 0.8596 | 0.8897 | **0.9014** |
| SHD | 2.00 | **1.40** | 2.15 | 1.85 | 1.55 | 2.05 |

![α 敏感性：保守的 0.005 给出最低 SHD，宽松 α 用精度换召回](/images/pc-algorithm-causal-factor/pc_3_alpha.png)

$\alpha=0.005$ 拿到最低 SHD 1.40（precision 恰好 1.0），惯用的 0.05 是 1.85，$\alpha=0.2$ 则把 precision 压到 0.9156、把 recall 抬到 0.9014——典型的「宁可多连、不可漏连」。但整张表的 SHD 都挤在 1.4–2.15 之间，相邻档差异远小于把 $n$ 从 100 提到 2000 带来的 2.35 改善。**结论是：α 是个二阶旋钮**，把时间花在调 α 上不如去多找样本；若一定要选，在本设定下偏保守（0.005–0.01）更划算，因为它能守住 precision，而漏检可以靠样本量补。

## 七、结果三：维度才是真正的悬崖

| 变量数 $p$ | 6 | 8 | 12 | 16 |
|---|---|---|---|---|
| 平均检验次数 | 83 | 257 | 1201 | 3027 |
| SHD | 0.000 | 1.000 | 10.833 | **32.333** |

![维度代价：检验次数与 SHD 随 p 爆炸式增长](/images/pc-algorithm-causal-factor/pc_4_dimension.png)

从 6 维到 16 维，检验次数涨了 36 倍，SHD 从 0 涨到 32.3。这里有两个叠加因素：一是候选边数按 $p(p-1)/2$ 增长、条件集组合数按组合数增长，$\max_k=2$ 在 12 维以上已经不够用，很多本该被更高阶条件集解释掉的边留了下来；二是多重检验问题——3000 次检验里，即使全是零假设，$\alpha=0.05$ 也期望产生上百次假阳性，precision 必然下滑。

$p=8,n=2000,\alpha=0.05$ 的单次案例中，真值 12 条边找回 11 条，precision 1.0、recall 0.9167、SHD 1，共执行 288 次条件独立检验（上图中橙色虚线就是那条被漏掉的边）。这个「漂亮成绩」的前提是 $n/p=250$ 且 `max_k=3`。把同样的配置搬到 100 个因子、200 条样本上（$n/p=2$），结果不会只是变差一点，而是整张图失去意义。

## 八、给因子研究的四条操作结论

1. **先算 $n/p$，再决定要不要跑。** 本实验里 $n/p\ge 250$ 才能把 SHD 压到 1 附近；$n/p=125$（$p=16,n=1000$）时 SHD 已到 32.3。高维场景务必先做变量预筛或分组检验，别把 100+ 因子一次性丢进去。
2. **提高 `max_k` 要跟着维度走，但代价是检验次数爆炸。** $p\ge 12$ 时 $\max_k=2$ 明显不足；扩到 3 会让检验次数再上一个数量级，需要配合并行与多重检验校正。
3. **报告 precision/recall/SHD，别只报「发现了几条边」。** 边数多可能只是 α 太松；本篇 $\alpha=0.2$ 的 recall 最高，但 precision 最低。
4. **α 建议从 0.005–0.01 起步**，并且认识到它是二阶参数。若样本量不足，调 α 只是在 precision 与 recall 之间搬椅子，不会凭空造出统计功效。

## 九、诚实边界

本篇全部为线性高斯 SEM 模拟，假设因果充分性（无隐变量）、无选择偏差、变量间无同期反馈——真实因子面板这三条几乎都不满足，尤其行业/风格共同因子本身就是典型的潜在混淆。评估只覆盖无向骨架，不含定向准确率与 Markov 等价类的固有不可辨识性，那部分见前篇。SHD 在 $n=2000$ 与 $n=5000$ 之间的非单调来自重复次数有限，不应解读为「样本越多越差」。$\max_k$ 在不同扫描里取 2 或 3，维度扫描的 SHD 因此偏悲观。所有结论都是关于**算法在已知真值下的统计功效**，不构成任何真实 A 股回测结果，也不支持任何「PC 选因子能多赚几个百分点」的收益断言。

想看定向、CPDAG 与下游选因子的完整链路，请读前篇《因果发现在因子选择中的 PC 算法》（10 月 26 日）。两篇合起来是一个完整判断：**PC 能对的地方是结构，容易错的地方是功效；先确认样本量撑得起维度，再谈因果图长什么样。**

## 十、完整可运行代码

以下代码即本篇实验源码，只依赖 numpy 与 scipy；直接运行会打印出与 `results.json` 完全一致的 12 项关键数字。

```python
import numpy as np
from scipy import stats
from itertools import combinations

SEED_IB, SEED_PC = 20261101, 20261102
L, HID, K, ITERS, BS, LR = 10, 32, 6, 600, 128, 0.008
BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
DGPS = ['ar1', 'ar2', 'lorenz']
NUIS = [0, 2, 4, 8, 16]


def gen_dag(p, n, rng, density=0.4):
    coef = np.zeros((p, p))
    for j in range(1, p):
        cand = list(range(j)); rng.shuffle(cand)
        npar = min(len(cand), max(1, int(round(density * j))))
        parents = cand[:npar] if rng.random() < 0.9 else []
        for i in parents:
            coef[i, j] = rng.uniform(0.35, 0.7) * rng.choice([-1.0, 1.0])
    X = np.empty((n, p))
    for j in range(p):
        var = max(0.10, 1.0 - float(np.sum(coef[:j, j] ** 2)))
        X[:, j] = X[:, :j] @ coef[:j, j] + np.sqrt(var) * rng.normal(size=n)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    truth = {frozenset((i, j)) for i in range(p) for j in range(p) if coef[i, j] != 0}
    return X, truth


def ci_test(data, i, j, Z, alpha):
    """Fisher-z 偏相关检验。偏相关来自精度矩阵（不是文章 v1 的错误公式）。"""
    cols = [i, j] + list(Z)
    C = np.corrcoef(data[:, cols], rowvar=False)
    Pm = np.linalg.pinv(C)
    r = float(np.clip(-Pm[0, 1] / np.sqrt(Pm[0, 0] * Pm[1, 1]), -0.999999, 0.999999))
    k, n = len(Z), len(data)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(1, n - k - 3))
    pval = float(2 * stats.norm.sf(abs(z)))
    return bool(pval > alpha), pval, r


def pc_skeleton(data, alpha=0.05, max_k=3):
    """PC-stable 骨架（不定向；定向/CPDAG 见 10-26 前篇）。"""
    n, p = data.shape
    adj = [set(range(p)) - {i} for i in range(p)]
    ntests, sepset = 0, {}
    for k in range(max_k + 1):
        snap = [set(a) for a in adj]          # stable: 用本层开始时的邻接快照
        for i in range(p):
            for j in sorted(snap[i]):
                if j not in adj[i]:
                    continue
                cand = sorted(snap[i] - {j})
                if len(cand) < k:
                    continue
                for Z in combinations(cand, k):
                    ntests += 1
                    indep, _, _ = ci_test(data, i, j, Z, alpha)
                    if indep:
                        adj[i].discard(j); adj[j].discard(i)
                        sepset[frozenset((i, j))] = Z
                        break
    found = {frozenset((i, j)) for i in range(p) for j in adj[i] if i < j}
    return found, ntests, sepset


def score(found, truth):
    tp = len(found & truth); fp = len(found - truth); fn = len(truth - found)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return float(prec), float(rec), int(fp + fn)


def run_pc(seed=SEED_PC, reps_n=20, reps_a=20, reps_p=6):
    out, plots = {}, {}
    NS = [100, 200, 500, 1000, 2000, 5000]
    ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    PS = [6, 8, 12, 16]

    # 1) 样本量功效
    nres = {}
    for n in NS:
        pr, rc, sh = [], [], []
        for r in range(reps_n):
            X, truth = gen_dag(8, n, np.random.default_rng(seed + 1000 + 97 * n + r))
            found, _, _ = pc_skeleton(X, 0.05, max_k=3)
            a, b, c = score(found, truth); pr.append(a); rc.append(b); sh.append(c)
        nres[str(n)] = {'precision': round(float(np.mean(pr)), 4),
                        'recall': round(float(np.mean(rc)), 4),
                        'shd': round(float(np.mean(sh)), 3),
                        'shd_std': round(float(np.std(sh)), 3),
                        'recall_std': round(float(np.std(rc)), 4)}
    out['n_scan'] = nres

    # 2) α 敏感性
    ares = {}
    for al in ALPHAS:
        pr, rc, sh = [], [], []
        for r in range(reps_a):
            X, truth = gen_dag(8, 1000, np.random.default_rng(seed + 2000 + 89 * int(al * 10000) + r))
            found, _, _ = pc_skeleton(X, al, max_k=3)
            a, b, c = score(found, truth); pr.append(a); rc.append(b); sh.append(c)
        ares[str(al)] = {'precision': round(float(np.mean(pr)), 4),
                         'recall': round(float(np.mean(rc)), 4),
                         'shd': round(float(np.mean(sh)), 3)}
    out['alpha_scan'] = ares
    out['best_alpha'] = float(min(ares, key=lambda a: ares[a]['shd']))

    # 3) 维度灾难
    pres = {}
    for p in PS:
        sh, nt = [], []
        for r in range(reps_p):
            X, truth = gen_dag(p, 1000, np.random.default_rng(seed + 3000 + 53 * p + r))
            found, t, _ = pc_skeleton(X, 0.05, max_k=2)
            sh.append(score(found, truth)[2]); nt.append(t)
        pres[str(p)] = {'shd': round(float(np.mean(sh)), 3),
                        'n_tests': int(round(float(np.mean(nt))))}
    out['p_scan'] = pres

    # 4) 一次具体还原（画图 + 报告）
    X, truth = gen_dag(8, 2000, np.random.default_rng(seed + 4000))
    found, ntests, sepset = pc_skeleton(X, 0.05, max_k=3)
    pr, rc, shd = score(found, truth)
    out['demo'] = {'n_edges_truth': len(truth), 'n_edges_found': len(found),
                   'precision': round(pr, 4), 'recall': round(rc, 4),
                   'shd': shd, 'n_tests': ntests}
    plots['demo_truth'], plots['demo_found'] = truth, found

    # 5) collider 反转：构造 i->k<-j，边际无关但给定 k 后相关
    coll = {}
    for n in [200, 1000, 5000]:
        g = np.random.default_rng(seed + 5000 + n)
        a, b = g.normal(size=n), g.normal(size=n)
        c = 0.7 * a + 0.7 * b + 0.3 * g.normal(size=n)
        M = (np.column_stack([a, b, c]) - np.column_stack([a, b, c]).mean(0)) / np.column_stack([a, b, c]).std(0)
        r_marg = float(np.corrcoef(M[:, 0], M[:, 1])[0, 1])
        _, p_marg, _ = ci_test(M, 0, 1, (), 0.05)
        ind, p_cond, r_part = ci_test(M, 0, 1, (2,), 0.05)
        coll[str(n)] = {'marginal_r': round(r_marg, 6), 'marginal_p': round(p_marg, 6),
                        'partial_r': round(r_part, 6), 'partial_p': round(p_cond, 8),
                        'declared_independent': bool(ind)}
    out['collider'] = coll
    return out, plots


if __name__ == '__main__':
    s, _ = run_pc()
    print('recall_n100       ', s['n_scan']['100']['recall'])
    print('recall_n5000      ', s['n_scan']['5000']['recall'])
    print('shd_n100          ', s['n_scan']['100']['shd'])
    print('shd_n5000         ', s['n_scan']['5000']['shd'])
    print('best_alpha        ', s['best_alpha'])
    print('shd_at_alpha01    ', s['alpha_scan']['0.1']['shd'])
    print('ntests_p16        ', s['p_scan']['16']['n_tests'])
    print('demo_precision    ', s['demo']['precision'])
    print('demo_recall       ', s['demo']['recall'])
    print('demo_n_tests      ', s['demo']['n_tests'])
    print('collider_marg_r   ', s['collider']['1000']['marginal_r'])
    print('collider_partial_r', s['collider']['1000']['partial_r'])
```
