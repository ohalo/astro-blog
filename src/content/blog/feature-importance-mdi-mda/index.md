---
title: "MDI 与 MDA 特征重要性：识别真正有用的因子特征"
description: "回测好看不等于特征有用——不打开黑箱，你不知道模型赚的是 alpha 还是巧合。受控实验（3 信息特征 + 3 冗余复制品 + 6 纯噪声）暴露两套主流工具各自的说谎方式：MDI（树内不纯度减少）给纯噪声也发 0.04+ 的非零重要性且永不为负，替代效应让 I_1 与它的两个复制品 R_1/R_2 三分天下（0.179/0.179/0.167），单看排名会以为有三个独立因子；MDA（OOS 置换退化）把噪声干净归零（≤0.0006）但冗余特征互相'顶班'——置换 I_1 时 R_1/R_2 还在场，损失退化被系统性低估。修复方案是聚类 MDA：相关特征整组同排置换，G1 组重要性 0.247 一举显形，是单列置换的 5 倍。三条铁律：MDI 只可比较不可检验、MDA 必须配 Purged CV 否则泄漏伪装成重要性、特征重要性分析先于回测而非之后。A 股因子普遍多重共线（规模/流动性/波动率纠缠），聚类版几乎是必选项（中高阶）"
publishDate: '2026-07-30'
tags:
  - 量化交易
  - 金融机器学习
  - 特征重要性
  - AFML
  - Python
language: Chinese
difficulty: "intermediate"
---

## 一句话版本

MDI 说「这个特征在树里被用了多少」，MDA 说「拿掉这个特征模型坏多少」——两个都会在相关特征面前说谎，方式不同：MDI 给噪声发钱、让冗余稀释排名；MDA 让冗余互相顶班、低估整个特征组。解法不是二选一，是**聚类之后整组置换**。

---

## 为什么特征重要性排在回测前面

López de Prado 在 AFML 里有一句被引用很多的狠话，大意是：**回测不是研究工具，特征重要性才是**。

逻辑链条：回测输出的是一条净值曲线，它只告诉你「这套规则在这段历史上赚了多少」，不告诉你**为什么**。如果你不知道模型依赖哪些特征、这些特征为什么应该有预测力，你无法区分三种情形——真 alpha、对历史噪声的过拟合、以及碰巧押中了某个未来会消失的 regime。而反复调参数直到回测好看，本质是在第二种情形上做梯度下降。

特征重要性是打开黑箱的第一把钥匙。但这把钥匙本身有两个主流版本，各自有系统性的说谎方式。不搞清楚它们怎么骗你，钥匙会把你带进另一间错误的房间。

## 受控实验：12 个特征，只有 3 个是真的

延续这个系列的方法论：不用真实市场数据做演示（无法知道 ground truth），用受控模拟把因果钉死。

构造 4000 个样本、12 个特征的分类数据集：

- **信息特征 I_1, I_2, I_3**：标签直接由它们生成，`logit = 1.2·I_1 + 0.9·I_2 − 0.7·I_3 + 噪声`；
- **冗余特征 R_1, R_2, R_3**：I_1 和 I_2 的高相关复制品（R_1 = I_1 + 小噪声，相关系数 0.989）——模拟量化实践中「同一信号的不同包装」：动量的 20 日版和 22 日版、PE 和 EP、规模因子的对数版和排名版；
- **纯噪声 N_1 … N_6**：与标签零关系。

![特征相关矩阵](/images/feature-importance-mdi-mda/fi-corr-matrix.png)

理想的特征重要性工具应该输出：I_1 > I_2 > I_3 显著非零，R 组与对应 I 特征共享重要性（毕竟携带同样的信息），N 组严格为零。看看两套工具各自交出什么答卷。

## MDI：树内记账，噪声也能分到钱

MDI（Mean Decrease Impurity）是随机森林自带的 `feature_importances_`：每次分裂带来的不纯度下降，按特征累加，全森林平均。它是**训练时的副产品**，零额外计算成本——这也是它被滥用最广的原因。

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

clf = RandomForestClassifier(
    n_estimators=400,
    max_features=1,        # AFML 建议：每次分裂只看 1 个特征，
                           # 强制所有特征都有机会入选，缓解掩蔽效应
    min_samples_leaf=20,
    n_jobs=-1, random_state=0,
)
clf.fit(X, y)

# 不要只看均值——用各树的分布算标准误
imp_all = np.array([t.feature_importances_ for t in clf.estimators_])
mdi_mean = imp_all.mean(axis=0)
mdi_se   = imp_all.std(axis=0) / np.sqrt(len(clf.estimators_))
```

![MDI 特征重要性](/images/feature-importance-mdi-mda/fi-mdi.png)

结果暴露 MDI 的两个结构性缺陷：

**缺陷一：噪声特征拿到显著非零的重要性。** 六个纯噪声特征每个都分到 0.04 上下——不是 bug，是数学必然。MDI 全森林归一化到和为 1，且每次分裂的不纯度下降非负，所以**MDI 永远不会给出零或负值**。树足够深时，噪声特征总会在某些节点上「碰巧」切出不纯度下降。你无法从 MDI 数值本身判断一个特征是否真的有用，只能做特征间的**相对比较**——而且比较基准线是 1/N 而不是 0。

**缺陷二：替代效应稀释排名。** I_1 的重要性 0.179，它的两个复制品 R_1、R_2 分别拿到 0.179 和 0.167——三者几乎均分。原因：树在某个节点选了 I_1 之后，下个节点用 R_1 能提供的增量信息就近似为零，但随机森林的列抽样让三者轮流被选中，把**本属于一份信息的功劳记在三个账户上**。如果你按「重要性 > 0.15」筛选特征，会以为自己找到了三个独立因子——实际上只有一个。

## MDA：样本外置换，噪声归零但冗余顶班

MDA（Mean Decrease Accuracy，即置换重要性）换了一个问法：模型训练好之后，在**样本外**数据上把某一列打乱重排，看性能退化多少。退化大 = 模型真的依赖它。

```python
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import pandas as pd

def mda(X, y, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    delta = pd.DataFrame(index=range(n_splits), columns=X.columns, dtype=float)
    for i, (tr, te) in enumerate(cv.split(X)):
        m = RandomForestClassifier(n_estimators=300, max_features=1,
                                   min_samples_leaf=20, n_jobs=-1, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        base = -log_loss(y[te], m.predict_proba(X.iloc[te]))
        for c in X.columns:
            Xp = X.iloc[te].copy()
            Xp[c] = np.random.default_rng(i).permutation(Xp[c].values)
            delta.loc[i, c] = base - (-log_loss(y[te], m.predict_proba(Xp)))
    return delta.mean(), delta.std() / np.sqrt(n_splits)
```

注意三个实现细节：性能指标用 **log-loss 而不是准确率**（金融场景下概率质量比分类对错重要，后续仓位函数直接消费概率）；置换发生在**测试折**上（训练集置换回答的是另一个问题）；每折重新训练，输出跨折均值 ± 标准误。

![MDA 特征重要性](/images/feature-importance-mdi-mda/fi-mda.png)

好消息：**六个噪声特征全部干净归零**（最大 0.0006，置信区间横跨零）。MDA 有天然的零点——置换一个模型根本不依赖的特征，OOS 性能不动。这让 MDA 具备了 MDI 没有的**检验属性**：可以回答「这个特征是否显著有用」。

坏消息藏在数值里：I_1 的退化只有 0.048。一个以 1.2 的系数直接生成标签的特征，置换后损失只坏了这么点？原因就是**顶班效应**：置换 I_1 时，R_1 和 R_2 还原封不动地留在特征矩阵里，模型（尤其 `max_features=1` 的随机森林，每棵树本来就分散依赖三个副本）转身就用复制品把信息补了回来。单列置换测的不是「这份信息值多少」，是「这一列不可替代的部分值多少」——在冗余普遍存在的金融特征集上，后者系统性低估前者。

推到极端：如果 R_1 与 I_1 相关性是 1.0，两列的 MDA 都会是**零**——尽管它们携带的信息是标签的直接成因。这是 MDA 最危险的失效模式：**你可能因为「重要性为零」扔掉一组实际上至关重要的特征**。

## 聚类 MDA：把顶班通道堵死

修复思路直白：既然相关特征互相顶班，就**把它们打包，整组一起置换**。组内所有列用同一个排列索引重排（保留组内相关结构），组外不动——顶班者和被置换者一起下场，退化才反映整组信息的真实价值。

```python
def clustered_mda(X, y, groups, n_splits=5):
    """groups: {组名: [列名, ...]}，由相关矩阵聚类得到"""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    res = pd.DataFrame(index=range(n_splits), columns=list(groups), dtype=float)
    for i, (tr, te) in enumerate(cv.split(X)):
        m = RandomForestClassifier(n_estimators=300, max_features=1,
                                   min_samples_leaf=20, n_jobs=-1, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        base = -log_loss(y[te], m.predict_proba(X.iloc[te]))
        for g, cols in groups.items():
            Xp = X.iloc[te].copy()
            perm = np.random.default_rng(i).permutation(len(Xp))
            Xp[cols] = Xp[cols].values[perm]   # 整组同排置换
            res.loc[i, g] = base - (-log_loss(y[te], m.predict_proba(Xp)))
    return res.mean(), res.std() / np.sqrt(n_splits)
```

分组本身可以用相关矩阵的层次聚类自动完成（AFML 第二版把这套流程称为 clustered feature importance）。本实验按已知结构分组：G1 = {I_1, R_1, R_2}，G2 = {I_2, R_3}，G3 = {I_3}，噪声各自成组：

![聚类 MDA](/images/feature-importance-mdi-mda/fi-clustered-mda.png)

结果一举归位：**G1 组重要性 0.247，是单列置换 I_1（0.048）的 5 倍**；G2 组 0.109；G3 组 0.041（I_3 没有复制品，单列与聚类结果一致，交叉验证了方法本身没有引入偏移）；噪声组依然干净归零。三个组的相对量级 0.247 : 0.109 : 0.041 也大致还原了生成系数 1.2 : 0.9 : 0.7 的信息贡献排序（经 logit 非线性和噪声衰减后的合理变形）。

## 三条铁律与工具选型

**铁律一：MDI 只做组内相对比较，永远不做显著性判断。** 它没有零点、不会为负、训练集内计算。用途限于「同一棵森林里谁被用得多」的快速侦察。任何「MDI > 阈值所以特征有效」的推断都是错的。

**铁律二：MDA 必须配 Purged K-Fold。** 本文演示用普通 KFold 是因为模拟样本 IID。真实金融标签带持有期（上一篇讲的区间重叠），普通 KFold 的测试折与训练折共享收益路径，**信息泄漏会伪装成特征重要性**——泄漏特征在被污染的测试折上表现优异，置换它退化巨大，MDA 给出漂亮的高分。Purging（清除与测试折标签区间重叠的训练样本）+ embargo 是前置条件，不是可选优化。

**铁律三：相关特征先聚类，再谈重要性。** 单列 MDA 在冗余面前系统性低估，单列 MDI 在冗余面前系统性稀释。真实因子库里「同一信息的不同包装」是常态而非例外——不聚类的特征重要性报告，参考价值要打对折。

选型速查：

| 场景 | 工具 |
|---|---|
| 训练时免费侦察、特征粗筛 | MDI（max_features=1，看分布不只看均值） |
| 「这个特征/特征组是否真的有用」 | 聚类 MDA + Purged CV |
| 特征间高度相关（金融常态） | 先层次聚类，再整组置换 |
| 需要单样本归因、非树模型 | SHAP（本文范围外，注意它同样受冗余干扰） |

## A 股实操注记

- **A 股因子天然重共线**。规模、流动性、波动率、反转在 A 股横截面上纠缠极深（小市值股票同时低流动性、高波动、强反转），单列置换会把这一整团信息的重要性摊薄到近乎不可见。聚类 MDA 在 A 股因子研究里几乎是必选项而非增强项。
- **分组不要用全样本相关矩阵**。相关结构本身有 regime 依赖（危机期相关性普遍抬升），用训练窗口内的相关矩阵聚类，且检查分组在不同子窗口的稳定性——分组本身不稳定，是特征工程质量的警报。
- **每折重训的成本**。聚类 MDA 的计算量是「折数 × 组数」次预测加「折数」次训练，几百特征的日频因子库在单机上完全可行，别用「太慢」当借口跳过 Purged CV。

## 你现在拥有的

一套分层的特征审计流程：MDI 快速侦察 → 相关矩阵聚类 → 聚类 MDA + Purged CV 做显著性判断。以及两个反直觉的教训：**重要性为零的特征可能至关重要（冗余顶班），重要性非零的特征可能纯属噪声（MDI 无零点）**——工具输出的数字，必须先经过「它会以什么方式说谎」的过滤器。

到这一篇，AFML 的核心管线走完了数据、标注、加权、审计四层。下一个自然的问题是横截面的：这些经过审计的特征，怎么组合成对全市场股票的预测并处理好中性化与拥挤度？留给后面的文章。

---

*参考文献：*
- *López de Prado, M. (2018). Advances in Financial Machine Learning, Chapter 8: Feature Importance. Wiley.*
- *Breiman, L. (2001). Random Forests. Machine Learning, 45(1).*
- *Strobl, C. et al. (2008). Conditional Variable Importance for Random Forests. BMC Bioinformatics, 9(307).*
- *本文实验：4000 样本 × 12 特征受控数据集（3 信息 + 3 冗余 + 6 噪声），全部结果可复现。*
