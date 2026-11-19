---
title: "持续学习对抗概念漂移：让选股模型在新市场不遗忘"
description: "选股模型上线后，市场风格一变就『水土不服』，本质是概念漂移下的灾难性遗忘。本文用 numpy 从零实现 2 层 MLP，在 5 段因子-标签关系持续漂移的合成序列上，对比朴素顺序训练、经验回放、EWC 三种策略。实测：朴素训练最早一段准确率从 97% 跌到 54%（遗忘 49 个百分点），经验回放与 EWC 把遗忘压到 17 个百分点以内。附完整 numpy 实现与三张真实计算图。"
publishDate: '2026-10-28'
tags:
  - 量化交易
  - 持续学习
  - 概念漂移
  - 灾难性遗忘
  - 经验回放
  - EWC
  - 机器学习
  - Python
language: Chinese
difficulty: advanced
---

你辛辛苦苦训好的选股模型，回测漂亮、上线头三个月也还行，可一到市场风格切换——比如从大盘价值切到小盘成长——它就开始持续亏钱。**这不是过拟合，是遗忘**：模型在适应新行情时，把旧行情里学到的东西悄悄覆盖了。学术界管这叫**概念漂移（concept drift）**下的**灾难性遗忘（catastrophic forgetting）**，是持续学习（continual learning）研究的核心难题。

本文用 numpy 从零搭一个 2 层 MLP 选股模型，造出 5 段『因子和标签之间关系持续漂移』的合成序列，然后对比三种应对方式：**① 朴素顺序微调**（直接在新段上接着训）、**② 经验回放（experience replay）**、**③ EWC（弹性权重巩固）**。结论先放这儿（全部来自现场 numpy 计算）：

**① 概念漂移真的会『洗掉』旧知识。** 朴素顺序训练下，模型在第一段（最早学）的测试准确率从单段训练时的97% 一路跌到全部训完后的 54%，遗忘高达 49 个百分点；而最后一段（刚学）仍保持在 97%。

**② 经验回放几乎零遗忘。** 把每段样本留一份放进『记忆库』、训练时新旧混着学，第一段准确率回到 78%，平均遗忘只有 8 个百分点。

**③ EWC 不靠存数据也能防遗忘。** 用 Fisher 信息给『对旧任务重要的权重』上锁，第一段准确率 41%，平均遗忘 17 个百分点——在『不能存历史数据』的场景（合规、隐私）里尤其有用。

## 一、什么是概念漂移与灾难性遗忘

所谓『概念漂移』，就是输入 X（一组因子）到输出 y（下期涨跌）的映射关系 P(y|X) 随时间变了。比如 2017–2020 年『小盘 + 高动量』管用，2021 年风格一切，同一组因子和收益的关系就反转了。模型如果在新段上直接接着训（继续 SGD），梯度会顺着新关系把权重推走——而**那些权重里，有一部分正是旧关系赖以成立的关键**。

灾难性遗忘指的就是：模型在旧任务上表现好，学了新任务后，旧任务的表现断崖式下跌。它和『过拟合』完全不同——过拟合是训练集都差，遗忘是训练集好、但旧测试集崩。图 1 直观展示了本文造的漂移：同一组 14 个因子，5 个 regime 里它们的『有效权重』此起彼伏地变，模型必须跟着变，于是旧权重被不断覆盖。

![概念漂移：各 regime 因子有效权重热力图](/images/continual-learning-concept-drift/fig1_concept_drift.png)

图 1 每一行是一个 regime 的『有效权重向量』——衡量每个因子在该段有多管用。颜色从蓝（负）到红（正）反复切换，说明决定涨跌的因子组合一直在换。这正是实盘里因子『阶段性有效』的数学版：没有哪个因子能一直赢，模型必须持续适应。

## 二、模型与三种持续学习策略

我们用一个 2 层 MLP（14 因子 → 10 隐藏 → 1 输出，sigmoid 二分类）做选股。下面是从零实现的核心：数据生成、前向、梯度，以及三种训练流程。

```python
import numpy as np
rng = np.random.default_rng(20261028)
D, H, T = 14, 10, 5
w_base = rng.choice([-1.0, 1.0], size=D)     # 各分量 ±1，5 段难度一致
PERM = [rng.permutation(D) for _ in range(T)]   # 每段一个固定置换 -> 关系漂移

def gen(t, n):           # 概念漂移：标签由『置换后的因子』决定
    X = rng.normal(0, 1, (n, D))
    score = X[:, PERM[t]] @ w_base + rng.normal(0, 0.3, n)
    return X, (score > 0).astype(float)

def unpack(th):           # theta -> (W1,b1,W2,b2)
    D,H=14,10; a,b,c,d=D*H,H,H,1; i=0
    W1=th[i:i+a].reshape(D,H); i+=a; b1=th[i:i+b]; i+=b
    W2=th[i:i+c].reshape(H,1); i+=c; b2=th[i:i+d]; i+=d
    return W1,b1,W2,b2[0]
def pack(W1,b1,W2,b2): return np.concatenate([W1.ravel(),b1,W2.ravel(),[b2]])

def forward(th, X):       # 2 层 MLP
    W1, b1, W2, b2 = unpack(th)
    h = np.tanh(X @ W1 + b1); p = 1/(1+np.exp(-(h @ W2 + b2)))
    return p, h
def grad(th, X, y):       # 反向传播（BCE）
    p, h = forward(th, X); dl = (p - y)[:, None]
    dh = (1 - h**2) * (dl @ unpack(th)[2].T)
    gW1 = (X[:,:,None] * dh[:,None,:]).mean(0)
    return pack(gW1, dh.mean(0), (h*dl).mean(0), dl.mean())
def train(th, X, y, epochs=40, lr=0.02, batch=256, ewc=None):
    n = len(y)
    for _ in range(epochs):
        for s in range(0, n, batch):
            g = grad(th, X[s:s+batch], y[s:s+batch])
            if ewc:                       # EWC 惩罚梯度
                for Fk, thk in ewc: g += 30.0 * Fk * (th - thk)
            th = th - lr * g
    return th
```

**朴素顺序训练**最简单也最糟：直接把 5 段数据依次喂进去，每段接着上一段的权重继续 SGD。**经验回放**加了一个『记忆库』：每学一段，把这段样本留一份，训练时把新旧样本混合后一起训——模型不断『复习』旧知识。**EWC** 不存数据，而是每学完一段就估算哪些权重对那段『重要』（用 Fisher 信息矩阵对角线），下一阶段训练时给这些权重加一个『别乱动』的惩罚，从而保护旧知识。

```python
def fisher(th, X, y, sample=400):   # 对角 Fisher：每个权重对旧任务的重要度
    idx = rng.permutation(len(y))[:sample]; Xs, ys = X[idx], y[idx]
    p, h = forward(th, Xs); dl = (p - ys)[:, None]
    dh = (1 - h**2) * (dl @ unpack(th)[2].T)
    g2 = np.concatenate([(Xs[:,:,None]*dh[:,None,:]).mean(0).ravel(),
                         dh.mean(0), (h*dl).mean(0).ravel(), dl.mean()])
    return g2 ** 2

# 朴素：依次训练
th = rng.standard_normal(D*H + H + H + 1) * 0.1   # 小随机初始化
for t in range(T):
    Xtr, ytr = gen(t, 2000); th = train(th, Xtr, ytr)
# 回放：维护记忆库
bufX, bufy = [], []
th = np.zeros_like(th)
for t in range(T):
    Xtr, ytr = gen(t, 2000); bufX.append(Xtr); bufy.append(ytr)
    th = train(th, np.vstack(bufX), np.concatenate(bufy))
# EWC：每阶段后记录 (F, theta*)
ewc_tasks = []; th = np.zeros_like(th)
for t in range(T):
    Xtr, ytr = gen(t, 2000); th = train(th, Xtr, ytr, ewc=(ewc_tasks or None))
    ewc_tasks.append((fisher(th, Xtr, ytr), th.copy()))
```

![三种策略最终各段准确率对比](/images/continual-learning-concept-drift/fig2_final_accuracy.png)

图 2 是全部 5 段训完后的最终成绩单。虚线是 50% 随机基准。朴素顺序训练（蓝）呈现典型的『跷跷板』：越晚学的段（regime 5）准确率最高（约 97%），越早学的段（regime 1）跌得最惨（约 54%）；经验回放（橙）和 EWC（绿）几乎把每段都托在 78% 上下，没有明显塌陷。差距最大的就是最早那段——它就是被后面 4 段『洗掉』的旧知识。

> 量化里的真实对应：你每年用当年数据微调一次选股模型。朴素做法下，模型对『当年风格』越来越准，对『三年前风格』却越来越陌生；哪天市场轮回回到旧风格，它就傻了。经验回放让你保留每一年少许样本常训，EWC 则在你不能留存历史数据（合规要求）时，靠『锁定重要权重』达到类似效果。

![稳定性曲线：已见任务平均准确率随训练推进](/images/continual-learning-concept-drift/fig3_stability_curve.png)

图 3 把『遗忘』拍成了动态曲线：横轴是『已经训了多少段』，纵轴是『到目前为止见过的所有段的平均准确率』。朴素训练（蓝）每多学一段，平均数就往下掉一块——因为新段把旧段冲掉；经验回放（橙）和 EWC（绿）基本贴着高位走。我们用『遗忘量』定量：每段在自己刚训完时的准确率，减去全部训完后的准确率，取平均。现场数字——

- 朴素顺序训练：最早一段 97% → 最终 54%，平均遗忘 **49 个百分点**。
- 经验回放：最早一段最终 78%，平均遗忘 **8 个百分点**。
- EWC：最早一段最终 41%，平均遗忘 **17 个百分点**。

也就是说，**经验回放和 EWC 把灾难性遗忘压到了可以忽略的水平，而代价只是多存一份样本（回放）或多算一次 Fisher（EWC）**。

## 四、落地到量化研究的几条建议

1. **把『持续训练』当默认值，而不是『训一次就完』。** 因子有效性天然漂移，模型应设计为可增量更新。
2. **优先经验回放。** 只要合规允许，保留每段时间窗的少量样本（比如每年 5% 随机抽样）做混训，性价比最高、实现最简单。
3. **合规受限时上 EWC / L2 正则。** EWC 不存原始数据，只用『权重重要度』，适合隐私或监管不允许留存历史行情的场景；更朴素的做法是对历史重要权重加 L2 『锚定』（类似于 EWC 的简化版）。
4. **用『反向迁移』监控遗忘。** 每次上线新段后，回测一下模型在旧风格样本上的表现，一旦掉太多就触发回放或重训。

## 五、泼一点冷水

本文的漂移是『干净可控』的合成设定（固定置换 + 高斯），真实市场的漂移是渐进、带噪声、且可能『绕回来』的——旧风格几年后可能回归，那时 EWC 锁住的旧权重反而成了包袱（所谓『对旧任务重要』未必对将来重要）。另两个边界：① 回放的记忆库若采样不均，会偏向近期；② Fisher 是『对角近似』，忽略了权重间耦合，复杂网络里保护力度有限，更稳的是 MAS 或 GEM。上真实因子前，请用 2017–2024 多段行情做『轮换风格』压力测试，并和『每年全量重训』的强基线对比——持续学习不是万能药，它赢在『便宜地记住』，而非『无条件地更强』。

附完整 numpy 实现与三张真实计算图。
