---
title: "CVaR 优化 Rockafellar：把『最坏情况的平均』写进目标函数"
description: "均值-方差优化只关心平均亏损和方差，对尾部视而不见——于是它会把权重压向『平时最稳、危机最惨』的资产。Rockafellar-Uryasev(2000) 把 CVaR（条件风险价值，即最坏 5% 情形的平均损失）写成一个线性规划，直接把尾部平均塞进目标函数。本文用合成收益(常态高斯 + 5% 灾难跳变)实跑：均值-方差把 100% 砸进肥尾资产4（组合 CVaR 5.93%），而 CVaR 优化主动分散、把资产4 压到 12.7%（组合 CVaR 仅 2.03%）。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - CVaR
  - 风险优化
  - Rockafellar
  - 投资组合
  - 尾部风险
  - 线性规划
  - Python
language: Chinese
difficulty: advanced
---

你做组合优化时，目标函数里写的是什么？

大多数人用的是 Markowitz 均值-方差：最大化 `wᵀμ − λ·wᵀΣw`。它优化的其实是**平均收益**和**方差**——也就是「典型日子」的好坏。但 2008 年和 2020 年告诉我们：毁掉组合的从来不是典型日子，是最坏那 5% 的日子。而方差对尾部是「近视眼」：一个平时很稳、但危机时暴跌 40% 的资产，和一个每天小亏 1% 的资产，方差可能差不多，但尾部风险天差地别。

**CVaR（Conditional Value-at-Risk，条件风险价值）** 直接优化尾部：它问的不是「第 95 百分位亏多少」（那是 VaR），而是「**超过 VaR 阈值的那些最坏情形，平均亏多少**」。Rockafellar & Uryasev (2000) 证明了 CVaR 可以写成一个**线性规划**——凸、可解、还能直接把约束（预算、行业上限、个股权重上限）塞进去。这就意味着你终于能把「最坏情况的平均」写进目标函数，而不是事后拿 VaR 当红绿灯。

本文用一个自洽的合成收益（常态高斯 + 5% 概率灾难跳变），把这件事跑通。

> ⚠️ 全部数字来自**合成数据**，仅用于演示方法学。真实资产的尾部依赖、流动性、相关性在危机中飙升等结构，合成里做了简化。

## 一、Rockafellar-Uryasev 的核心技巧：把 CVaR 变成 LP

CVaR 的定义看着吓人：
$$\text{CVaR}_α = \mathbb{E}[\,L \mid L \ge \text{VaR}_α\,]$$
其中 $L = -wᵀr$ 是组合损失。直接优化它要算分位条件期望，非凸。R&U 的妙招是引入两个辅助变量：

- 一个标量 $z$（它是 VaR 的代理）；
- 一组非负偏差变量 $u_t \ge 0$（捕捉每个样本超出 $z$ 的部分）。

目标函数变成：
$$\min_{w, z, u} \quad z + \frac{1}{(1-\alpha)T}\sum_{t=1}^{T} u_t$$
约束只有一条关键不等式（对每个样本 $t$）：
$$z + u_t \ge wᵀ L_t, \quad u_t \ge 0$$
其余是预算 $wᵀ\mathbf{1}=1$ 和权重上下限。这整个是**线性规划**——`scipy.optimize.linprog` 一行 `method="highs"` 就能解。

直觉：$z$ 被压成损失分布的分位阈值，$u_t$ 是超出部分，目标里那项 $\frac{1}{(1-\alpha)T}\sum u_t$ 正好是「超阈值损失的平均」——也就是 CVaR。最小化它，优化器自然会**躲开肥尾**。

## 二、合成数据：一个藏着灾难跳变的收益矩阵

四个资产，常态是不同波动的高斯，但每个资产都有 5% 概率的灾难跳变（危机时额外暴跌，且资产4 跌得最狠）：

```python
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(20260717)
N_ASSET, N_SAMP, alpha = 4, 2000, 0.95

mu  = np.array([0.0008, 0.0010, 0.0007, 0.0014])
vol = np.array([0.010,  0.015,  0.012,  0.030])
R = np.zeros((N_SAMP, N_ASSET))
for i in range(N_ASSET):
    base = mu[i] + vol[i] * rng.standard_normal(N_SAMP)
    crash = (rng.random(N_SAMP) < 0.05) * rng.exponential(0.04 + 0.03 * i) * (-1)
    R[:, i] = base + crash
R = -R                      # 转成"损失"视角: loss_t = -r_t
```

## 三、把 LP 写出来并求解

```python
def cvar_min_portfolio(losses, alpha):
    T, n = losses.shape
    # 变量: [w(0..n-1), z, u(0..T-1)]  共 n+1+T
    c = np.zeros(n + 1 + T)
    c[n] = 1.0                                   # z
    c[n + 1:] = 1.0 / ((1 - alpha) * T)          # u_t 权重
    A_ub, b_ub = [], []
    for t in range(T):                            # z + u_t >= wᵀ loss_t
        row = np.zeros(n + 1 + T)
        row[:n] = losses[t]; row[n] = -1.0; row[n + 1 + t] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    bounds = [(0, 1)] * n + [(None, None)] + [(0, None)] * T
    A_eq = np.zeros((1, n + 1 + T)); A_eq[0, :n] = 1.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=np.array([1.0]), bounds=bounds, method="highs")
    w = res.x[:n]
    port_loss = losses @ w
    thr = np.quantile(port_loss, alpha)
    cvar = port_loss[port_loss >= thr].mean()
    return w, cvar

w_cvar, cvar_val = cvar_min_portfolio(R, alpha)
```

## 四、对照：均值-方差在重尾下会犯什么错

同一批数据，用带非负约束的均值-方差求解器对照：

```python
from scipy.optimize import minimize
def mv_portfolio(Ret, risk_aversion=8.0):
    mu_e = Ret.mean(axis=0); cov = np.cov(Ret.T); n = len(mu_e)
    obj = lambda w: w @ cov @ w - risk_aversion * (w @ mu_e)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(obj, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n, constraints=cons)
    return res.x / res.x.sum()
```

![CVaR 最小化组合 vs 均值-方差组合的权重分配](/images/cvar-optimization-rockafellar/cvar_weights.png)

结果一目了然：

| 组合 | 权重（资产1~4） | 组合 CVaR (α=0.95) |
|---|---|---|
| **CVaR 最小化** | 1.9% / 15.0% / 70.4% / 12.7% | **2.03%** |
| **均值-方差** | 0% / 0% / 0% / **100%** | **5.93%** |

均值-方差看到资产4 期望收益最高（0.14%/天），把 **100% 砸了进去**——完全无视它危机时暴跌最狠。结果组合 CVaR 高达 5.93%。CVaR 优化则主动把资产4 压到 12.7%，把权重分散到更干净的资产，组合 CVaR 砍到 2.03%——**尾部风险降了约 66%**，而这是在不靠任何主观判断的情况下自动发生的。

> CVaR 与 VaR 的关系：本例中组合 CVaR 2.03% 是 VaR 阈值（1.64%）的 1.24 倍。CVaR 永远 ≥ VaR，多出来的那块就是「最坏情形比阈值还差多少」的尾部厚度。

## 五、危机样本下的损失分布

只看危机日（组合损失落在上 10% 分位）的损失直方图：

![危机样本下两组合的损失分布（CVaR 组合左尾更薄）](/images/cvar-optimization-rockafellar/cvar_crisis_dist.png)

均值-方差组合（红）在危机日的左尾明显更厚、更靠左——它那 100% 的资产4 在灾难跳变日集体暴毙。CVaR 组合（蓝）左尾更薄，因为它根本没把鸡蛋放在那个会碎的篮子里。

## 六、CVaR 有效前沿

和均值-方差一样，CVaR 优化也有有效前沿：在目标里加一个期望收益项 $\lambda·wᵀ\mu$，扫描 $\lambda$ 就能画出「横轴收益、纵轴尾部风险」的前沿。

![CVaR 有效前沿 vs VaR(分位)前沿](/images/cvar-optimization-rockafellar/cvar_frontier.png)

两条曲线都往上走——想多赚，就得多扛尾部风险，这是不可免的。但注意 **CVaR 前沿（圆点）整条在 VaR 前沿（方块）上方**：因为 CVaR 量的是超阈值平均，永远比单一分位 VaR 更悲观、更诚实。用 VaR 做优化，等于盯着分位线就觉得安全了，却对线下的尾巴视而不见。

## 七、尾部对冲：加一个危机对冲资产

放开一个「危机时上涨」的对冲资产（平时微跌、灾难日 +3%），再看 CVaR 优化的反应：

```python
hedge = -0.0002 + 0.006 * rng.standard_normal(N_SAMP)
hedge += (rng.random(N_SAMP) < 0.05) * 0.03      # 危机时 +3%
R5 = np.column_stack([R, -hedge])                # 损失视角
w_cvar_h, cvar_h = cvar_min_portfolio(R5, alpha)
```

![放开危机对冲资产后组合 CVaR 显著下降](/images/cvar-optimization-rockafellar/cvar_hedge.png)

组合 CVaR 从 2.03% 直接掉到 **1.03%**。但注意一个真实且重要的副作用：CVaR 优化把 **75.9% 的权重塞进了这个对冲资产**，资产4 被压到 4.1%。这是 CVaR 优化的典型行为——**它一旦找到完美的危机保险，就会过度集中**。

![不同置信度 α 下 CVaR 与 VaR 的关系（CVaR 永远 ≥ VaR）](/images/cvar-optimization-rockafellar/cvar_alpha.png)

α 越接近 1（越关注极端尾部），CVaR 与 VaR 的差距越大——尾部越肥，只看分位 VaR 越危险。

## 八、六类真实陷阱（高阶必看）

1. **样本估计误差**：LP 是确定性优化，喂进去的是**样本**损失矩阵。样本 CVaR 对极端样本点极敏感——少算一次跳变，最优权重可能大变。实务必须用**收缩估计**或**bootstrap 平均**多组权重，别直接用单组解。
2. **相关性在危机中飙升**：合成里各资产跳变独立，真实里危机会让一切同跌（相关性→1）。这时分散失效，CVaR 优化给出的「分散权重」在和平时期好看、危机里照样陪跌。要喂**危机情景**的损失矩阵，而不是全样本平均。
3. **过度集中对冲资产**：第七节已演示——CVaR 优化会重仓找到的危机保险。必须加**个股权重上限**约束（`bounds=[(0,0.1)]`），否则 76% 单资产是实盘不能接受的集中度风险。
4. **α 的选取是任意的**：α=0.95 还是 0.99，最优权重可能完全不同。它不是被数据决定的，是你拍的。要做**α 敏感性分析**，看结论在 α∈[0.9,0.99] 是否稳健。
5. **流动性与交易约束**：LP 里没建模冲击成本。真实里调权重本身有成本，高频再优化可能把 alpha 全付给券商。要加换手惩罚或限制调仓频率。
6. **CVaR 假设损失分布已知**：R&U 公式成立的前提是损失分布固定。市场体制切换时分布漂移，昨天的 CVaR 解今天可能不再最优——需要**滚动窗口重估**，且对最近样本适当加权。

## 九、小结

均值-方差优化「平均主义」的盲区，在尾部会致命。Rockafellar-Uryasev 把 CVaR 写成线性规划，让你能把「最坏 5% 情形的平均损失」直接塞进目标函数——凸、可解、可加约束。合成里它自动把均值-方差 100% 重仓的肥尾资产4 压到 12.7%，组合 CVaR 从 5.93% 砍到 2.03%。

但它不是免死金牌：样本误差、危机相关性飙升、对冲资产过度集中、α 任意性，每一条都能让漂亮的 CVaR 解在实盘翻车。真正用得好的人，是把 CVaR 优化当**尾部约束器**而非**收益发动机**——先定收益目标，再用 CVaR 把尾部锁死。一个务实的做法是「双层」：外层用均值-方差或风险平价定收益与分散骨架，内层把 CVaR 作为约束（`wᵀL 的 CVaR ≤ 预算`）而非直接优化目标，这样既保留分散、又不让尾部失控。

---

*代码与配图均由本文 Python 片段在合成数据上真实运行生成。所有收益数字仅用于演示方法，不构成任何投资建议。*
