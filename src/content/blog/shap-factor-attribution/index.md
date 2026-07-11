---
title: "用 SHAP 做因子贡献归因与模型监控：把组合 PnL 拆回因子，并盯住生产漂移"
publishDate: '2026-07-11'
description: "选股模型上线后，老板只问一句话：这个月赚的 3% 到底是谁贡献的？本文用 SHAP 把每一条预测（进而把组合 PnL）拆回单个因子与因子组，并进一步把样本按时间切片，追踪各因子 SHAP 贡献的分布漂移——漂移本身就是生产环境的预警信号。附完整 Python 实现与归因完整性自检。"
tags:
  - 量化交易
  - 机器学习
  - 因子归因
  - 模型监控
  - SHAP
  - Python
language: Chinese
difficulty: advanced
---

「回测年化 30%、夏普 2.5，但我不懂它为什么赚钱」是黑箱模型的老问题；而模型**上线之后**还有第二个、更现实的问题：**这个月组合赚了 3%，到底是哪些因子贡献的？哪个因子这周突然不灵了？**

第一问叫**因子贡献归因（attribution）**，第二问叫**生产模型监控（monitoring）**。两者都能用同一个工具干净地解决——SHAP。本文不重复讲「单样本可解释性」（那篇已经写过），而是聚焦这两条线，并给你一套能直接接进生产流水线的代码。

## 一、为什么归因和监控是同一件事

因子模型天生讲究「逻辑可解释」：Fama-French 三因子、质量因子、动量因子，每一个都有经济学叙事。但当你把 11 个因子丢进梯度提升树，模型会学到交互与非线性，于是：

- **单因子 IC 看不见交互**：`动量 × 质量` 的联合效应，用逐个因子的相关系数完全测不出来。
- **组合 PnL 不能简单按因子权重分**：因子之间相关性、非线性、时变暴露，都会让「线性加权」式的归因失真。
- **实盘漂移没人盯**：训练时价值因子贡献 30%，三个月后变成 −10%，往往等到回撤爆了才发现。

SHAP 用博弈论的 Shapley 值，把模型的**每一次预测**公平地分摊到每个因子头上，满足「效率性」：所有因子的 SHAP 之和等于这次预测相对基线的偏离。这一条性质既是归因的基石，也是监控的抓手——**把 SHAP 贡献随时间切片，漂移立刻现形**。

## 二、准备数据并训练因子模型

我们用 11 个因子（覆盖价值 / 动量 / 质量 / 波动 / 其他五大类）合成一个含交互项与非线性的真实收益生成过程，再训练梯度提升树去拟合它。这样 SHAP 归因就有「标准答案」可对照。

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(20260711)

groups = {
    "价值": ["bp", "ep"],
    "动量": ["mom1m", "mom12m"],
    "质量": ["roe", "gross_margin"],
    "波动": ["vol_60d", "beta"],
    "其他": ["ln_mktcap", "sales_growth", "amihud_illiq"],
}
names = [f for g in groups for f in groups[g]]
p = len(names)

cov = 0.6 * np.eye(p) + 0.4 / p          # 温和相关性

def dgp(Z):
    bp, ep, mom1m, mom12m, roe, gm, vol, beta, lnmc, sg, ami = Z.T
    return (0.50*bp + 0.35*ep + 0.40*mom12m + 0.30*roe + 0.25*gm
            - 0.35*vol - 0.20*beta + 0.15*lnmc + 0.10*sg - 0.10*ami
            + 0.32*mom1m*roe                       # 交互项：动量 × 质量
            + 0.10*np.tanh(vol*2.0)*mom12m)        # 轻度非线性

Xtr = rng.multivariate_normal(np.zeros(p), cov, size=3000)
rtr = dgp(Xtr) + rng.normal(0, 0.40, 3000)
rtr = (rtr - rtr.mean()) / rtr.std()

model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                  learning_rate=0.05, subsample=0.8, random_state=7)
model.fit(Xtr, rtr)
bg = Xtr.mean(0, keepdims=True)          # 基线 = 全样本因子均值
base_pred = float(model.predict(bg)[0])
```

## 三、向量化蒙特卡洛 Shapley 估计

模型无关地估计 SHAP，最稳的是蒙特卡洛采样 coalition（特征联盟）。对因子 $j$，随机采样一批「其他因子是否在场」的联盟 $S$，比较「$S$ 加上 $j$」与「只有 $S$」的预测差，取期望即得 $\phi_j$。下面用 `np.where` 把整批联盟向量化，一次预测成千上万行，速度比逐样本循环快两个数量级：

```python
def mc_shap(X, model, bg, K=120, seed=3):
    n = X.shape[0]
    rr = np.random.default_rng(seed)
    phi = np.zeros((p, n))
    for j in range(p):
        C  = rr.random((n, K, p)) < 0.5        # (n, K, p) 联盟掩码
        Cw  = C.copy(); Cw[:, :, j]  = True     # 含因子 j
        Cwo = C.copy(); Cwo[:, :, j] = False    # 不含因子 j
        Xw  = np.where(Cw,  X[:, None, :], bg[None, None, :]).reshape(n*K, p)
        Xwo = np.where(Cwo, X[:, None, :], bg[None, None, :]).reshape(n*K, p)
        pw  = model.predict(Xw).reshape(n, K)
        pwo = model.predict(Xwo).reshape(n, K)
        phi[j] = (pw - pwo).mean(1)             # 边际贡献的期望
    return phi
```

`phi[j, i]` 就是第 $i$ 条样本上因子 $j$ 的 SHAP 值——它可正可负，表示该因子把这次预测往「赚」还是「亏」推了多少。

## 四、单笔归因瀑布：这条买入信号押了什么

取预测最高（最强买入信号）的一条样本，按 $|\text{SHAP}|$ 降序把因子一个个叠加，从基线推到最终预测，就是一张直观的归因瀑布：

![单条最强买入信号的因子级归因瀑布：各因子把预测从基线推到最终值](/images/shap-factor-attribution/factor_attribution_waterfall.png)

红色是正向推动（看多该因子的暴露），绿色是负向拖累。一眼就能看出：这次信号主要押在**价值 + 质量**，而不是你以为的动量。如果研究员坚持「这是个动量策略」，这张图就是最好的对质证据。

## 五、因子组聚合：Alpha 来自哪一大类

单因子归因太碎，汇报时要聚到因子组层面。对每个组，取组内因子 `|SHAP|` 的均值作为「贡献强度」：

```python
group_abs = {}
for g, fs in groups.items():
    idxs = [names.index(f) for f in fs]
    group_abs[g] = np.abs(phi[idxs, :]).mean()   # 全样本平均
```

![因子组层面的 SHAP 贡献强度：Alpha 主要来自哪些大类](/images/shap-factor-attribution/group_attribution.png)

这一步把「11 个因子」压成「5 个大类」，组合经理能立刻回答老板：本月超额收益里，**价值与质量**是主引擎，波动类是负贡献。这就是把模型输出翻译成投资语言的标准动作。

## 六、生产监控：SHAP 贡献漂移就是预警

归因是「事后看」，监控是「持续看」。把样本按时间切成 10 个周期（模拟周度 / 月度），计算每个周期内各因子的平均 SHAP，画成热力图：

```python
mean_phi_period = np.array([phi[:, period == pp].mean(1) for pp in range(n_periods)])
norm = mean_phi_period / (np.abs(mean_phi_period).max(0, keepdims=True) + 1e-9)
```

![生产监控：各因子 SHAP 贡献随时间漂移（按行归一化）](/images/shap-factor-attribution/monitoring_drift.png)

红色是正向贡献、蓝色是负向。**价值因子（bp）的贡献随周期稳步抬升、动量因子（mom12m）逐步衰减**——这正是典型的 regime 漂移。把它接成告警规则非常简单：

```python
rolling = pd.Series(mean_phi_period[:, names.index("mom12m")])
z = (rolling - rolling.mean()) / (rolling.std() + 1e-9)
if abs(z.iloc[-1]) > 2.0:                        # 贡献偏离历史 2σ
    alert(f"动量因子 SHAP 贡献异常漂移，触发复核")
```

核心思想：**你不必等回撤爆了才发现问题，SHAP 贡献分布的漂移会提前几周发出信号**。这正是「模型监控」和「单样本解释」最本质的区别——前者是时间序列，后者是横截面。

## 七、完整性自检：ΣSHAP ≈ 预测 − 基线

归因能不能信，先过一道数学关。Shapley 有「效率性」：对所有因子求和应等于这次预测相对基线的偏离。我们用样本逐个校验：

```python
x = phi.sum(0)            # 各因子 SHAP 之和
y = model.predict(Xte) - base_pred
r = np.corrcoef(x, y)[0, 1]
```

![归因完整性检验：ΣSHAP 与 预测−基线 高度一致（效率性质）](/images/shap-factor-attribution/attribution_efficiency.png)

两条线几乎重合（相关性 0.999+），说明我们的归因是**自洽、可加、不漏项**的。这步自检必须做——如果 ΣSHAP 和预测偏离很大，说明 coalition 采样数 $K$ 不够或基线选错，归因结果直接作废。

## 边界声明

- **SHAP ≈ 真实因果**：SHAP 度量的是「模型在给定数据分布下的边际依赖」，不是经济因果。它告诉你模型**用了**什么，不保证那是一条可交易的真理。
- **相关因子会稀释归因**：`bp` 与 `mom12m` 负相关时，两者的 SHAP 会互相「抢功」，单个因子的绝对贡献被低估。需要时可做分组 SHAP（把一组高度相关因子当一个「超因子」算）。
- **监控 ≠ 可解释**：贡献漂移报警只告诉你「分布变了」，至于变的原因是宏观 regime、流动性还是数据管道 bug，仍需人工下钻。
- **采样数要够**：本文 $K=120$ 已能让效率相关性到 0.999；因子更多或模型更非线性时，应把 $K$ 提到 300+ 并复核效率性。
- **基线选择有讲究**：用训练集均值做基线最稳；用单样本做基线会让归因失去「相对谁」的参照。

把上面四张图接进你的研究面板，你就同时拥有了**归因**（对老板）+ **监控**（对自己）两道防线。
