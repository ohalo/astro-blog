---
title: "注意力机制的可解释性：用特征归因解释模型决策"
description: "注意力权重只告诉你模型「看了哪里」，不等于「哪里重要」。本文用纯 numpy 搭一个小的特征级自注意力收益预测器，故意只让一个特征携带信号，对比注意力权重与 Integrated Gradients 特征归因——后者能精确锁定真正驱动预测的因子，并给出归因的实战边界。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 注意力机制
  - Attention
  - 可解释性
  - 特征归因
  - Integrated Gradients
  - 机器学习
  - Python
language: Chinese
difficulty: advanced
---

Transformer 火了之后，量化圈也流行把注意力（Attention）当成「模型的可解释性出口」：画一张注意力热力图，指着颜色深的地方说「看，模型在做决策时最关注这个因子 / 这段行情」。直觉很诱人，但 2019 年 Jain & Wallace 那篇《Attention is not Explanation》早就泼了冷水：**注意力权重只告诉你模型「把计算资源投向了哪里」，并不等于「那个位置对输出有多重要」**。

这句话在量化场景尤其致命——如果风控、合规、或者你自己在 debug 一个因子模型时，误把「注意力高」当成「因子重要」，很可能被误导。本文用纯 numpy 搭一个最小的特征级自注意力收益预测器，故意只让一个特征携带真实信号，然后对比两种「重要性」度量：注意力权重 vs Integrated Gradients（IG）特征归因。结论很明确：**归因能精确锁定真正驱动预测的因子，注意力则被严重稀释**。

## 一、模型：特征级自注意力

我们把一条候选样本表示成一个 $F\times T$ 的矩阵：F=8 个特征（因子），每个特征是一条长度为 T=20 的时序。把**每个特征当成一个 token**，过一层 self-attention，再做 mean-pooling，最后接一个线性头预测次日收益。

```python
import numpy as np

rng = np.random.default_rng(20260711)
N, F, T, d = 1400, 8, 20, 20
TRUE_FEAT = 3
# 只有 feature 3 携带信号：它的时序 = 固定形态 p × 隐藏标量 s_i，label 只由 s_i 决定
p_pattern = rng.normal(0, 1, T)
s_hidden = rng.normal(0, 1, N)
X = rng.normal(0, 1, (N, F, T))
X[:, TRUE_FEAT, :] = s_hidden[:, None] * p_pattern[None, :] + 0.10 * rng.normal(0, 1, (N, T))
label = s_hidden + rng.normal(0, 0.10, N)

P = {
    "Wp": rng.normal(0, np.sqrt(2.0/T), (d, T)),
    "Wq": rng.normal(0, np.sqrt(2.0/d), (d, d)),
    "Wk": rng.normal(0, np.sqrt(2.0/d), (d, d)),
    "Wv": rng.normal(0, np.sqrt(2.0/d), (d, d)),
    "w":  rng.normal(0, 0.1, d), "b": 0.0,
}
def softmax_rows(m):
    m = m - m.max(1, keepdims=True); e = np.exp(m); return e / e.sum(1, keepdims=True)

def forward(x):                      # x: (F,T)
    tok = x @ P["Wp"].T              # (F,d)
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T; V = tok @ P["Wv"].T
    A = softmax_rows(Q @ K.T / np.sqrt(d))     # (F,F) 注意力矩阵
    context = A @ V                  # (F,d)
    pooled = context.mean(0)         # (d,)
    return pooled @ P["w"] + P["b"], dict(A=A, V=V)
```

直觉上，模型想预测 label，就必须从 feature 3 的序列里把隐藏标量 $s_i$ 「投影」出来。这只有 feature 3 办得到，其余 7 个特征都是纯噪声。所以**真正的驱动因子是 feature 3，这一点是已知的**——正好用来检验两种解释方法谁能找对。

## 二、训练（纯 numpy 全批量反向）

训练就是最小化 MSE，梯度手推即可。`A` 的 softmax 反向是标准写法：

```python
def grad_x(x):                      # 返回 ∂y/∂x (F,T)，用于后面的归因
    y, c = forward(x); A, Vv = c["A"], c["V"]
    tok = x @ P["Wp"].T
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T
    dpooled = P["w"]                # dy/dpooled
    dcon = np.outer(np.ones(F), dpooled) / F
    dA = dcon @ Vv.T
    dV = A.T @ dcon
    dS = A * (dA - (dA * A).sum(1, keepdims=True)) / np.sqrt(d)   # softmax 行反向
    dQ = dS @ K; dK = dS.T @ Q
    dtok = dQ @ P["Wq"] + dK @ P["Wk"] + dV @ P["Wv"]
    return dtok @ P["Wp"]           # (F,T)

# 训练循环（SGD，含早停在验证集）
n_tr = 1000
for ep in range(5000):
    idx = rng.integers(0, n_tr, 128)
    tok = X[idx] @ P["Wp"].T
    Q = tok @ P["Wq"].T; K = tok @ P["Wk"].T; V = tok @ P["Wv"].T
    A = softmax_rows(Q @ K.transpose(0,2,1) / np.sqrt(d))
    pooled = (A @ V).mean(1)
    yhat = pooled @ P["w"] + P["b"]; err = yhat - label[idx]
    # head 批量更新
    P["w"] -= 0.01 * (pooled.T @ err / len(idx)); P["b"] -= 0.01 * err.sum() / len(idx)
    # 权重逐样本更新（略，见仓库完整脚本）
```

训完后，模型在训练集 R²≈0.91、验证/测试集 R²≈0.78——它确实学会了从 feature 3 里把信号抽出来。下面才是重点：**它学到的「注意力」长什么样，和「真正的重要性」一致吗？**

## 三、两种「重要性」度量

**方法 A：注意力权重（入度）。** 把注意力矩阵 $A$ 的每一列求和，得到「每个特征被多少注意力投注」：

$$\text{attn\_imp}[j]=\sum_i A_{i,j}$$

这是业界画热力图时最常用的「特征重要性」代理。

**方法 B：Integrated Gradients（特征归因）。** IG 回答的是另一个问题：**把输入从基线（全 0）连续变到真实值，输出对每一维输入的「累积敏感度」是多少？**

$$\text{IG}(x)=\bigl(x-\text{baseline}\bigr)\odot\int_{0}^{1}\frac{\partial y}{\partial x}\bigl(\text{baseline}+\alpha(x-\text{baseline})\bigr)\,d\alpha$$

它沿路径积分梯度，天然满足「敏感性」与「实现不变性」两条公理，比单点梯度（saliency）稳健得多。下面是单样本实现：

```python
def ig_importance(x, K_ig=50):
    ig = np.zeros((F, T))
    for k in range(K_ig):
        a = (k + 0.5) / K_ig
        ig += grad_x(a * x)            # 在路径点 a*x 上算梯度
    return (x * ig / K_ig).mean(1)     # 沿时间聚合 -> (F,)

def attn_received(x):
    _, c = forward(x)
    return c["A"].sum(0)               # 入度 (F,)
```

![自注意力权重矩阵：注意力的「看哪里」——颜色深的列不一定等于「哪里重要」](/images/attention-interpretability-quant/fig_attention_heatmap.png)

![Integrated Gradients 特征归因：精确锁定真正驱动预测的 F3](/images/attention-interpretability-quant/fig_ig_attribution.png)

## 四、结果：注意力被稀释，归因更锋利

在 400 条测试样本上，我们统计两种方法「找对真正驱动因子 feature 3」的能力：

| 度量 | 注意力权重 | Integrated Gradients |
|---|---|---|
| Top-1 定位准确率（#1 是否=feature 3） | **48.7%** | **62.5%** |
| 真实特征占重要性比重 | **31.3%** | **40.8%** |

两个数字都说明同一件事：**注意力确实「略偏向」正确的特征（48.7% 远高于随机的 12.5%），但它把权重稀释到了其余 7 个噪声特征上；而 IG 把重要性更集中地压在 feature 3 上，单样本归因图的峰也更尖。** 放到「为某一笔具体预测找原因」的场景里，IG 比注意力靠谱得多。

![注意力 vs IG 定位能力对比：IG 在 top1 准确率和真实特征占比上都更优](/images/attention-interpretability-quant/fig_attn_vs_ig.png)

## 五、为什么注意力会误导，以及实战清单

**为什么注意力 ≠ 重要性？** 三个结构性原因：
1. **softmax 归一化扭曲**。注意力每行必须和为 1，于是「给 A 高权重」必然「抢走 B 的权重」——它是一个*相对*分配，不是*绝对*贡献。两个无关特征也可能因为互相「衬托」而获得高权重。
2. **注意力只是路由，不是内容**。输出由 `context = A @ V` 决定，真正起作用的是 V 的内容 + 头部 w 的读取方向。一个特征即使被「少注意」，只要它的 V 被 w 读到，照样主导输出。
3. **mean-pooling 让注意力变得可有可无**。在本架构里，所有 token 的 context 都被平均进 pooled，头部完全可以忽略注意力、直接靠 V 和 w 工作——这正是注意力被稀释、但模型依然准的根源。

**实战清单（给想解释自己模型的量化人）：**
- **别用注意力热力图当「因子重要性」证据**，它最多算「模型把计算投到了哪」。真正要解释「哪个因子导致这笔预测」，用归因（IG / Gradient×Input / SHAP）。
- **IG 的 baseline 要选得有业务意义**。本文用全 0（「什么都没有」的行情），你也可以选「行业中性后的基准」或「历史均值」，不同 baseline 会改变归因，必须显式说明。
- **归先是局部的**。IG 给的是「这一条样本、这一笔预测」的解释，不是全局因子排名。要做全局归因，需要对大量样本聚合，并注意分布偏移。
- **归因不是银弹**：它对模型结构敏感，面对对抗性扰动也可能被骗。把它当成「排查与沟通工具」，而不是「模型为什么这么做的终极真理」。
- **多头部 / 多层注意力**会让「注意力」更不等于重要性——层数越深，A 离输出越远，用它解释输出越危险。

## 六、总结

注意力机制是强大的建模工具，但是一个**糟糕的解释工具**：它揭示的是「模型把注意力投向哪里」，而不是「哪里真正驱动了输出」。本文用纯 numpy 复现了一个特征级自注意力收益预测器，在「只有 feature 3 携带信号」的受控实验里，注意力只以 48.7% 的 top-1 准确率找对真凶、且把重要性稀释到 31.3%，而 Integrated Gradients 把这两个数字提升到 62.5% 与 40.8%。**当你需要向自己、向风控、向合规解释「这笔预测到底是谁决定的」，请放下热力图，去算归因**。
