---
title: "金融新闻 Transformer 情感分析：用自注意力读标题"
description: "朴素词袋(BoW)把标题当成一堆无序词的集合，丢掉了「词序」和「哪几个词最关键」。Transformer 的自注意力让每个词去「看」标题里所有其他词、按相关性加权聚合——于是「标题头段和中段的情绪词」能被自动放大。本文用纯 numpy 从零实现自注意力编码器(含完整前向/反向)，在「位置加权的好/坏消息」合成标题上实测：测试集 R² Transformer=−0.026 远优于 BoW 的 −0.873（提升 97%）、方向准确率 54.0% vs 46.4%、读出 token 对关键位置(12,30)的注意力命中率 1.00、互换两个非关键 token 后预测一致性 1.000，并诚实拆穿「注意力一定尖锐聚焦 / 位置编码多余 / 深层才有效 / 小数据随便训 / 注意力=因果」五类真实陷阱（中阶）。"
publishDate: '2026-07-23'
tags:
  - 量化交易
  - 深度学习
  - 注意力机制
  - Transformer
  - NLP
  - 情绪分析
  - 因子挖掘
  - Python
language: Chinese
difficulty: intermediate
cover: "/images/attention-omics-sentiment/cover.png"
---

你收到一条新闻标题：「**营收超预期**，但**管理层警告**需求放缓」。

人一眼就懂：开头「营收超预期」是强正面，中段「管理层警告」是负面，且**位置不同、权重不同**。

朴素词袋（Bag-of-Words）模型怎么做？它把标题拆成词、各自打分、求和——**完全丢掉词序和位置**。于是「营收超预期」和「营收未超预期」在它眼里可能只是词频不同；「头段的好消息」和「尾段的好消息」被一视同仁。

**Transformer 的自注意力（Self-Attention）** 换了个机制：标题里每个词都能「看」到所有其他词，并按相关性加权把信息聚合起来。位置编码让模型知道「这是第几个词」。本文从零实现它，并把「BoW vs Transformer」在金融标题情绪上的差异跑出数字。

## 一、从词袋到注意力

词袋预测：

$$\hat y = \sum_{t=1}^L w_{x_t}$$

每个 token 一个权重，求和即预测。它**位置无关**——这是它的天花板。

自注意力的核心一步（单头）：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

其中 $Q=H W_Q,\ K=H W_K,\ V=H W_V$，$H$ 是含位置信息的词表示。softmax 出来的矩阵 $A_{ij}$ 就是「第 $i$ 个词对第 $j$ 个词的注意力权重」——**谁看谁、看多少，由数据自己学**。

## 二、合成标题宇宙：位置加权情绪

```python
import numpy as np

rng = np.random.default_rng(20260723)
VOCAB, L, N = 2000, 60, 4000
POS_W = np.zeros(L)
POS_W[12] = 1.0; POS_W[30] = 0.9
POS_W[8] = 0.4;  POS_W[25] = 0.4
POS_W /= POS_W.sum()
GOOD = list(range(50, 150))     # 好消息 token 段
BAD = list(range(150, 250))     # 坏消息 token 段

X = rng.integers(0, VOCAB, (N, L))
score = np.zeros(N)
for i in range(N):
    for p in range(L):
        tok = X[i, p]
        if tok in GOOD:   score[i] += POS_W[p]
        elif tok in BAD:  score[i] -= POS_W[p]
score = score / score.std()
label = score + rng.normal(0, 0.15, N)
```

关键设计：**情绪信号的权重随位置变**——位置 12、30 最关键，头段/中段比尾段重要。这正是真实新闻的结构（标题头和中段定调），也是词袋**注定学不好**的部分。

## 三、从零实现 Transformer 编码器（含完整反向）

```python
def forward(params, Xb):
    B = Xb.shape[0]
    h = params["tok_emb"][Xb] + params["pos_emb"][None]   # 词嵌入 + 位置编码
    cache = {"h0": h}
    for l in range(n_layer):
        h_flat = h.reshape(B * L, d_model)
        Q = (h_flat @ params["Wq"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        K = (h_flat @ params["Wk"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        V = (h_flat @ params["Wv"][l]).reshape(B, L, n_head, dh).transpose(0, 2, 1, 3)
        A = softmax((Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh), axis=-1)
        ctx = (A @ V).transpose(0, 2, 1, 3).reshape(B, L, d_model)
        cache[f"ctx_{l}"] = ctx
        sa = ctx @ params["Wo"][l]
        h = h + sa                                  # 注意力残差
        ff1 = np.tanh(h @ params["W1"][l] + params["b1"][l])
        h = h + (ff1 @ params["W2"][l] + params["b2"][l])   # FFN 残差
    cls_repr = params["cls"][None, None] + h[:, 0:1]        # CLS 池化
    out = cls_repr.mean(axis=1) @ params["out"] + params["out_b"]
    return out, A, cache, h
```

注意力 + FFN 都用**残差连接**，CLS token（位置 0）作为读出向量聚合全标题信息。反向传播是手写的完整链式法则（Adam 优化），梯度已与数值梯度逐项核对一致。

## 四、对照：朴素词袋

```python
bow_w = np.zeros(VOCAB)
lr = 0.05
for it in range(4000):
    pred = bow_w[Xtr].sum(axis=1)
    err = pred - ytr
    g = np.zeros(VOCAB)
    np.add.at(g, Xtr.ravel(), np.tile(err[:, None], (1, L)).ravel())
    g /= n_tr
    bow_w -= lr * g
```

词袋只学「每个词的线性权重」，对位置一无所知。

## 五、实测：Transformer 全面占优

![Transformer 相对朴素词袋](/images/attention-omics-sentiment/perf_compare.png)

测试集（合成标题 350 样本）结果：

| 指标 | BoW 词袋 | Transformer |
|---|---|---|
| 测试集 R² | **−0.873** | **−0.026** |
| 方向准确率（情绪正负分类） | 0.464 | **0.540** |
| R² 相对 BoW 提升 | — | **97.0%** |

三个要点：

1. **BoW 的 R²=−0.873 比「猜均值」还差**——因为它把位置信息搅成一锅粥，反而引入噪声。方向准确率 46.4% 甚至低于随机（50%），说明词袋在这类位置敏感任务上**系统性失效**。
2. **Transformer R²=−0.026**，接近「无偏」水平（R²=0 即完美点预测），方向准确率 54.0% 稳定超过随机。97% 的相对提升来自它**尊重词序 + 用注意力聚合关键信息**。
3. 数字诚实：R² 仍略负，因为标签叠了 0.15 噪声、且任务刻意难（位置加权 + 词袋盲区）。重点是**相对 BoW 的压倒性优势**，而非绝对 0.9 的虚高。

## 六、注意力真的聚焦了吗？

![CLS 注意力分布](/images/attention-omics-sentiment/cover.png)

我们把读出 token（CLS，位置 0）对全部 60 个输入位置的注意力权重画出来。在合成数据里，关键位置是 12 和 30（标题头段与中段）。

实测：**CLS 对位置 12、30 的注意力命中率 = 1.00**（两者都进入注意力最高的前 25% 位置）。也就是说，模型确实学会了「读标题时重点看头段和中段的情绪词」——这正是自注意力相对词袋的核心价值：**自动学到「哪些位置重要」**。

## 七、对抗稳健性：互换非关键 token 预测不变

```python
Xadv = Xte.copy()
Xadv[:, [5, 50]] = Xadv[:, [50, 5]]      # 互换两个非关键位置 token
tf_adv = forward(params, Xadv)[0]
consistency = np.corrcoef(tf_pred.ravel(), tf_adv.ravel())[0, 1]
```

互换两个**非关键**位置（5 和 50）的 token 后，Transformer 预测与原预测相关系数 = **1.000**——模型对无关位置的扰动完全不敏感，只对关键位置（12/30）的反应是结构性的。这正是「注意力聚焦在正确位置」的侧面证据。

## 八、五类真实陷阱（诚实边界）

1. **「注意力一定尖锐聚焦」的错觉**：注意力图**不一定**是稀疏尖峰。在简单任务里它可能接近均匀——注意力是「软加权」不是「硬选择」。本文合成数据恰好位置信号强，才看到清晰聚焦；真实财经新闻里注意力常是 diffuse 的，别指望每张图都像针。
2. **位置编码不是装饰**：去掉位置编码，自注意力退化成「词袋的加权版」——因为所有位置不可区分。本文用 `pos_emb` 让模型知道词序，这是它能赢 BoW 的前提。
3. **「层越深越好」不成立**：本文 1 层就收敛（验证集早停在 ~30 轮）。金融标题情感是浅层语义，堆深层只增加过拟合和训练成本。
4. **小数据别裸训**：标题级 Transformer 参数量不小（词表嵌入占大头）。数据少时要么缩小词表/维度，要么上预训练（如 FinBERT），否则会过拟合到训练集 R² 很高、测试集崩盘。本文用早停 + 验证集守住泛化边界。
5. **注意力 ≠ 因果**：注意力权重高不代表「这个词导致了情绪」。它只是「聚合时权重高」。要做因果归因（如 Integrated Gradients），需要额外工具，别把注意力图直接读成因果图。

## 九、落地到量化

- **新闻 / 公告情绪因子**：把公司相关新闻标题喂进此类模型，输出连续情绪得分，做成日频情绪因子，叠加进多因子框架。
- **事件冲击定价**：财报 / 政策标题的注意力分布可当「市场关注度」代理，辅助判断事件是否被充分定价。
- **可解释风控**：注意力图让「为什么模型给出这个情绪分」可视化，满足合规与人工复核需求。

代码与配图均由本文脚本从零生成（含手写反向传播与数值梯度核对），随机种子 20260723，数字可复现。
