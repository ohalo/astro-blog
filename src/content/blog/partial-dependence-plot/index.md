---
title: "局部依赖图 PDP：把单个因子与收益的边际关系画成曲线"
description: "机器学习因子模型跑出了收益，你却说不清『动量到底怎么影响预测』——因为模型是黑箱，特征重要性只给排名不给形状。PDP（Partial Dependence Plot）把某个因子在其取值范围上扫一遍、其余特征原样保留后对全样本预测求平均，画出这个因子与预测收益的边际曲线：是单调、倒U 还是阶梯，一眼看穿。本文用合成因子数据训练梯度提升树，画动量的倒 U 型甜区、四因子 PDP 面板、动量×规模的二维交互热图，并诚实拆穿 PDP 最致命的陷阱——相关因子下它会强行外推到根本不存在的样本组合、把借来的效应算成自己的（附完整 Python，中阶）。"
publishDate: '2026-07-25'
tags:
  - 量化交易
  - PDP
  - 可解释AI
  - 机器学习
  - 因子研究
  - 边际效应
  - Python
language: Chinese
difficulty: intermediate
---

用机器学习做因子选股，最难受的不是模型不赚钱，而是模型赚了钱、你却画不出"这个因子到底怎么起作用"的那条曲线。特征重要性能告诉你"动量排第一"，但排第一是**单调向上**、还是**过犹不及的倒 U**、还是**只在某段区间有效**？这三种形状对应完全不同的交易含义，而重要性排名一个都答不了。

结论先放这：**PDP（Partial Dependence Plot，局部依赖图）把某个因子在它的取值范围上扫一遍，其余特征保持原样，对全样本预测求平均，画出这个因子与预测收益的边际关系曲线。** 它把黑箱模型学到的"因子—收益"关系可视化成一条你能读、能讲、能质疑的线。本文用合成因子数据训练梯度提升树，画动量的倒 U 甜区、四因子面板、动量×规模的二维交互，并诚实拆穿 PDP 在相关因子下会把借来的效应算成自己的致命陷阱（中阶）。

## PDP 到底在算什么

给定训练好的模型 $\hat{f}$，要看第 $j$ 个因子的边际效应，PDP 的定义是：

$$
\text{PDP}_j(v) = \frac{1}{n}\sum_{i=1}^{n}\hat{f}(x_j = v,\; x_{-j}^{(i)})
$$

翻译成人话：把**所有样本**的第 $j$ 个因子**统统强行设为** $v$，其余因子保留每个样本自己的原值，然后让模型预测、对这 $n$ 个预测取平均。$v$ 从因子的低分位扫到高分位，得到一串平均预测，连起来就是 PDP 曲线。

这条曲线回答的是："在其他因子的**平均分布**下，只把这个因子从小调到大，模型的预测会怎么变。" 它是一条**边际**曲线——把其他维度积分掉，只留下你关心的那一维。

关键点：PDP 平均掉了其他因子，所以它给的是**群体层面的平均趋势**，不是任何单个样本的路径。想看单样本的路径要用它的兄弟 ICE（Individual Conditional Expectation，个体条件期望曲线）——每个样本画一条，PDP 就是这些 ICE 的平均。两者叠在一起看信息量最大。

## 合成一份有已知答案的因子数据

要验证 PDP 画得对不对，最干净的办法是**自己造数据**——因为只有合成数据里，真实的边际形状是已知的，可以拿来对答案。

我造四个因子，故意给它们不同性质的效应：

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(42)
N = 4000

momentum   = rng.normal(0, 1, N)   # 动量：倒 U（过犹不及）
value      = rng.normal(0, 1, N)   # 价值：单调向上
size       = rng.normal(0, 1, N)   # 规模：本身弱，但与动量交互
volatility = rng.normal(0, 1, N)   # 波动率：极弱负效应

def true_signal(m, v, s, vol):
    mom_eff = 0.9 * m - 0.35 * m**2          # 倒 U：m 太大反而拖累
    val_eff = 0.5 * v                         # 线性单调
    inter   = 0.4 * m * np.clip(-s, -2, 2)    # 动量在小盘股(s<0)更强
    vol_eff = -0.15 * vol                     # 弱负
    return mom_eff + val_eff + inter + vol_eff

y = true_signal(momentum, value, size, volatility) + rng.normal(0, 1.0, N)
X = np.column_stack([momentum, value, size, volatility])

model = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
)
model.fit(X, y)
```

注意 `true_signal` 里的四种形状：动量是 $0.9m - 0.35m^2$ 的倒 U，价值是纯线性，规模自己没有主效应但通过 `0.4·m·(-s)` 和动量**交互**，波动率是微弱负效应外加大量噪声。这四种形状就是我们要 PDP 帮我们"猜"出来的答案。

## PDP 手写实现：一个循环就够

PDP 的实现简单到令人发指——它就是"改一列、预测、求平均"的循环：

```python
def pdp_1d(model, X, feat_idx, grid):
    vals = []
    Xtmp = X.copy()
    for g in grid:
        Xtmp[:, feat_idx] = g        # 整列强行设为 g
        vals.append(model.predict(Xtmp).mean())  # 全样本预测取平均
    return np.array(vals)

def ice_1d(model, X, feat_idx, grid, sample_idx):
    lines = []
    for i in sample_idx:
        Xt = np.tile(X[i], (len(grid), 1))  # 复制这个样本 len(grid) 份
        Xt[:, feat_idx] = grid              # 只改目标因子
        lines.append(model.predict(Xt))
    return np.array(lines)
```

`pdp_1d` 是全样本平均，`ice_1d` 是逐样本单独画。把 grid 取在因子的 2%~98% 分位之间（避开极端值把曲线拉飞），扫 50 个点即可。

![动量因子的 PDP 与 ICE：倒 U 型甜区被清晰还原](/images/partial-dependence-plot/pdp_momentum_ice.png)

上图淡蓝色细线是 40 条 ICE（每条是一个样本把动量从小调到大的路径），红色粗线是它们的平均——PDP。绿色虚线是我们埋进去的**真实倒 U 形状**。三者几乎重合：模型确实学到了"动量适度为佳、过大反而回落"的倒 U，PDP 把这个甜区画得清清楚楚。这就是 PDP 的价值——它把一个只会输出数字的黑箱，变成了一张能讲给风控听的图。

## 四因子面板：一眼分辨四种形状

把四个因子的 PDP 并排画出来，四种性质立刻分明：

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for idx, ax in enumerate(axes.ravel()):
    g = np.linspace(np.percentile(X[:, idx], 2), np.percentile(X[:, idx], 98), 50)
    p = pdp_1d(model, X, idx, g)
    ax.plot(g, p, lw=2.5)
    ax.plot(X[:200, idx], np.full(200, p.min()), "|", alpha=0.3)  # rug 显示数据密度
```

![四个因子的 PDP：动量倒U、价值单调、规模弱、波动率近平](/images/partial-dependence-plot/pdp_all_factors.png)

读图：动量是漂亮的倒 U，价值几乎是直线向上，规模的曲线幅度很小（因为它的作用主要藏在交互里、被 PDP 平均掉了），波动率近乎水平微降。底部的竖线是 rug plot——显示数据密度，曲线两端数据稀疏的地方要少信几分。**规模那条几乎平的线是个重要警告：PDP 会低估以交互为主的因子**，因为它把交互对象平均掉了。这正好引出下一张图。

## 二维 PDP：把交互作用画成热图

PDP 可以同时扫两个因子，画出它们联合的平均预测热图。这是发现交互作用的利器：

```python
gm = np.linspace(-2, 2, 30); gs = np.linspace(-2, 2, 30)
Z = np.zeros((len(gs), len(gm)))
for i, sv in enumerate(gs):
    for j, mv in enumerate(gm):
        Xt = X.copy(); Xt[:, 0] = mv; Xt[:, 2] = sv
        Z[i, j] = model.predict(Xt).mean()
```

![二维 PDP：动量效应在小盘股中更强（交互作用可见）](/images/partial-dependence-plot/pdp_2d_interaction.png)

如果动量和规模没有交互，等高线会是**平行的竖线**（动量效应与规模无关）。但图里等高线是**倾斜、扭曲**的：在规模轴下方（小盘股，`size<0`）动量的颜色梯度更陡——同样加一个单位动量，小盘股的预测收益涨得更多。这正是我们埋进去的 `0.4·m·(-s)` 交互项。一维 PDP 把规模压平了，二维 PDP 把它救回来了。

## 致命陷阱：相关因子下 PDP 会说谎

PDP 最大的死穴不在于画得难看，而在于**它会理直气壮地画错**。原因藏在它的定义里：$\hat{f}(x_j=v, x_{-j}^{(i)})$ 把目标因子强行设为 $v$、其余因子保留原值——**但如果目标因子和其他因子高度相关，这个组合可能根本不存在**。

举个例子：动量和"动量的 EMA 平滑版"相关系数 0.9。PDP 在算"动量EMA=+2"的效应时，会把**所有样本**的动量EMA 都设成 +2，却保留它们原来的动量值——包括那些动量是 -2 的样本。可现实里动量 -2 而动量EMA +2 的股票几乎不存在。模型在这些**不存在的组合**上是纯外推，输出的是幻觉，PDP 却把这些幻觉平均进了曲线。

```python
# 造一个与动量 0.9 相关的因子，它本身对 y 没有独立贡献
mom2 = 0.9 * momentum + np.sqrt(1 - 0.81) * rng.normal(0, 1, N)
X2 = np.column_stack([momentum, mom2, value])
y2 = 0.8 * momentum + 0.5 * value + rng.normal(0, 1, N)  # 注意 y2 不依赖 mom2
```

![PDP 陷阱：相关因子下 PDP 外推到不真实的样本组合](/images/partial-dependence-plot/pdp_correlated_pitfall.png)

红线是 PDP：它给 `mom2` 画出了一条明显向上的斜坡，仿佛这个因子很有用。但 `y2` 的生成公式里**根本没有 mom2**——它的斜率完全是从相关的动量那里"借"来的，而 PDP 靠外推到不真实组合把这个借来的效应放大了。蓝色的"真实条件均值"（只在数据真实支撑的区间取样）平缓得多。**如果你照着红线的 PDP 去判断"mom2 是个好因子"，你会把一个纯冗余的因子当成 alpha 来用。**

这就是为什么有了 ALE（Accumulated Local Effects，累积局部效应）——它用条件期望和局部差分，只在数据真实存在的邻域里算效应，专门修 PDP 这个外推病。那是另一篇的主题。

## A. 实现细节

- **信号口径**：PDP 与 ICE 的输入是模型对合成因子的**回归预测值**（预期收益），不是真实标签。曲线画的是"模型认为的边际关系"，用来解释模型、不是解释市场。
- **grid 范围**：所有一维 PDP 的扫描网格取因子的 2%~98% 分位，避免极端值把曲线尾部拉飞；二维热图取 ±2 标准差的规整网格。
- **ICE 居中**：40 条 ICE 按起点对齐（减去各自在 grid 起点的预测再加回 PDP 起点值），消除样本间的水平位移，只看形状差异。
- **模型**：梯度提升树（300 棵、深度 3、学习率 0.05、行采样 0.8），足以拟合倒 U 与交互又不至于过拟合噪声。
- **rug plot**：面板图底部竖线显示前 200 个样本的因子分布，提示曲线各段的数据支撑强弱。

## B. 已知偏差

- **相关因子外推**：这是 PDP 的头号问题，正文已用 0.9 相关的构造复现——PDP 会把相关因子借来的效应算成自己的，并在不真实的组合上纯外推。相关性强的因子集上，PDP 的绝对水平不可尽信，趋势方向也可能被污染。
- **交互被平均掉**：一维 PDP 把交互对象积分掉，会**系统性低估**以交互为主的因子（正文里规模那条几乎平的线）。必须配合二维 PDP 或 ICE 的"发散扇形"来补。
- **合成数据的局限**：真实市场的因子相关结构、非平稳性、regime 切换远比这里复杂。合成实验证明的是"PDP 的机制和陷阱"，不是"这套因子能赚钱"。
- **平均掩盖异质性**：PDP 只给一条平均线。若一半样本因子效应为正、另一半为负，PDP 可能画成一条平线——ICE 才能揭露这种分裂。

## C. 结果解读

- **形状 > 排名**：PDP 的核心价值是把"动量重要"升级成"动量呈倒 U、甜区在中等偏上"。倒 U 意味着极端动量股要减配而非加配——这是重要性排名永远给不出的交易含义。
- **交互只能靠二维看**：规模因子在一维 PDP 里几乎无效（幅度最小），但二维 PDP 的倾斜等高线证明它在小盘股里放大动量。**把一维 PDP 当因子筛选唯一依据，会漏掉所有交互型因子。**
- **相关因子上别信绝对水平**：构造实验里 `mom2` 对 y 零贡献，PDP 却给它画出向上斜坡（借自 0.9 相关的动量）。凡因子集内部相关性高，PDP 的效应大小要打折，最好交叉验证 ALE。
- **ICE 是 PDP 的测谎仪**：40 条 ICE 与 PDP 高度平行 → 效应同质、平均可信；若 ICE 扇形发散 → 存在交互或异质，此时 PDP 那条平均线具有误导性。
- **落地建议**：PDP 适合做"模型体检"和"向非技术方讲解"，但**不要**直接拿 PDP 曲线的绝对高度去构造交易权重。因子相关性高时，优先用 ALE 校准；判断交互时，二维 PDP 与 ICE 缺一不可。
