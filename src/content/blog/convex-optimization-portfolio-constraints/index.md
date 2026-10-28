---
title: "凸优化组合约束 CVXPY：把行业上限与止损线直接写进优化器"
description: "朴素均值-方差把 57.4% 压在科技一个行业，单日 95% CVaR 高达 3.87%——这是裸的行业下注。本文用 CVXPY 把个股上限、行业上限、CVaR 止损线三种约束直接写进优化器：硬性上限把科技压到 40%、CVaR 降到 3.76% 却牺牲收益（Sharpe 0.436）；而只写一条 CVaR95≤3.7% 止损线、不设任何权重上限，CVXPY 自动把尾部压到 3.70% 且 Sharpe 反而升到 0.444——直接盯住损失分布，比拍脑袋设上限更聪明。附完整 CVXPY 实现与三张真实计算图。"
publishDate: '2026-10-28'
tags:
  - 量化交易
  - 组合优化
  - 凸优化
  - CVXPY
  - CVaR
  - 风险管理
  - Python
language: Chinese
difficulty: advanced
---

做组合优化的人几乎都经历过同一个尴尬：**用教科书上的均值-方差算出来的权重，往往把 60%、70% 全押在一两个资产上**。这不是你算错了，而是无约束的 Markowitz 在数学上就是这么"激进"——它把所有信念都压在那一个历史夏普最高的方向上，哪怕这个方向只是样本里的一次偶然。

真实的钱不能这么下注。合规有持仓上限，风控有行业敞口约束，基金经理自己也有"单日最多亏多少"的心理止损线。问题在于：**这些约束该写在哪一层？** 大多数人是在优化器跑完之后，再手动去"砍"超限的权重——先把科技砍到 40%，再把超的挪去消费。这种事后修补既不优雅，也保证不了全局最优。

更聪明的做法是把约束**直接写进优化器**，让求解器在可行域里自己找最优解。本文用 `CVXPY` 这一行声明式凸优化框架，把三种最常见的约束——个股上限、行业上限、以及一条 CVaR 尾部止损线——写进同一个问题，并在一个 12 资产 / 3 行业的合成因子模型上实测它们各自的代价与收益。

结论先放这里（全部来自现场的 CVXPY 计算）：

**① 无约束的纯多组合，本质上是裸的行业下注。** 它在科技行业堆了 **57.4%** 的仓位，单日 95% CVaR 高达 **3.87%**；允许做空后更是把科技顶到 57.8%、年化收益虚高到 11%，但那是用杠杆和做空换来的幻觉。

**② 硬性上限能压住集中度，但代价是浪费风险预算。** 把个股上限设 25%、行业上限设 40% 后，科技被强制压到 40%，组合 CVaR 降到 **3.76%**，但年化收益从 9.61% 掉到 **8.63%**，Sharpe 从 0.464 降到 **0.436**——它为了凑满"每个行业都要有仓位"的硬指标，被迫把 25.5% 配到了收益最低的消费上。

**③ 只写一条 CVaR 止损线，不设任何权重上限，结果反而更好。** 告诉优化器"单日 95% 损失不得超过 3.7%"，它自己把科技从 57.4% 削到 46.6%、把风险摊开，最终 CVaR **3.70%**（比硬性上限更低）、年化收益 **8.70%**、Sharpe **0.444**（比硬性上限更高）。**直接盯住损失分布，比拍脑袋设权重天花板更聪明。**

## 一、为什么约束要写进优化器，而不是事后砍

均值-方差优化的标准形式是一个二次规划（QP）：

$$\min_{w}\; w^\top\Sigma w - \lambda\,\mu^\top w \quad \text{s.t.}\quad \mathbf{1}^\top w = 1,\; w \ge 0$$

这里 $\Sigma$ 是协方差矩阵，$\mu$ 是预期收益向量，$\lambda$ 是风险厌恶系数。这个 QP 有闭式解 $w^\star \propto \Sigma^{-1}\mu$，解出来的权重几乎必然极度集中——因为 $\Sigma^{-1}$ 会把历史波动最小的那个方向无限放大。

实务里的约束大致分两类：

- **权重型约束**：`w_i ≤ 个股上限`、`Σ_{i∈行业} w_i ≤ 行业上限`、`w_i ≥ 0`（纯多）。它们管的是"组合长什么样"。
- **风险型约束**：`CVaR_α ≤ 止损线`、`VaR_α ≤ 阈值`、`最大回撤 ≤ 某值`。它们管的是"组合会亏多少"。

事后砍权重的问题在于：你砍掉一个超限资产后，腾出的权重要往哪放？随手挪去别的行业，很可能只是把风险从一种形态换成了另一种形态，**并没有真正降低尾部损失**。而把约束写进凸优化器，求解器会在满足全部硬约束的前提下，自动挑选那个让目标函数最优的分配——这是"全局一致"的，而不是"先优化、再打补丁"。

`CVXPY` 的价值正在于此：你用几行声明式代码描述问题，它自动调用 CLARABEL / OSQP / SCS 等求解器，该转成 SOCP、QP 还是 SDP 它自己判断。下面所有的约束，本质上都是往 `constraints` 列表里加几行。

## 二、把 CVaR 止损线写成一个硬约束

`CVaR`（条件风险价值，也叫 Expected Shortfall）是尾部损失的期望值：给定置信水平 $\alpha$（比如 95%），`CVaR_α` 就是"最差的 $(1-\alpha)$ 那部分情景的平均损失"。它比 VaR 更适合做约束，因为 VaR 不满足次可加性（不是一个 coherence risk measure），而 CVaR 是，且它能写成线性约束。

Rockafellar–Uryasev 给出一个漂亮的等价形式：引入一个辅助变量 $t$ 和一组非负变量 $z_s$，对 $S$ 个情景的损失 $l_s = -r_s^\top w$，令

$$\text{CVaR}_\alpha = t + \frac{1}{(1-\alpha)S}\sum_{s=1}^{S} z_s, \qquad z_s \ge 0,\; z_s \ge l_s - t$$

当 $t$ 取 VaR_α 时，上式正好等于 CVaR_α，而 $z_s$ 只在损失超过 $t$ 的情景里取正值。于是"`CVaR_α ≤ L`"就变成了一组**线性不等式约束**，凸优化器处理起来毫无压力。下面这段代码就是它的完整实现：

```python
import cvxpy as cp
import numpy as np

def build_problem(mu_d, Sigma_d, R, alpha=0.05,
                  long_only=True, name_cap=None, sector_cap=None,
                  cvar_limit=None, gamma=3.0):
    N = len(mu_d)
    w = cp.Variable(N)
    ret = mu_d @ w                       # 日度预期收益
    risk = cp.quad_form(w, cp.psd_wrap(Sigma_d))   # 方差项
    obj = cp.Maximize(ret - 0.5 * gamma * risk)

    cons = [cp.sum(w) == 1]              # 满仓（无杠杆）
    if long_only:
        cons += [w >= 0]                 # 纯多
    if name_cap is not None:
        cons += [w <= name_cap]          # 个股上限
    if sector_cap is not None:           # 行业上限
        for idx in sector_indices:
            cons += [cp.sum(w[idx]) <= sector_cap]
    if cvar_limit is not None:           # CVaR 止损线（核心）
        t = cp.Variable()
        z = cp.Variable(R.shape[0])
        port_loss = -R @ w               # 各情景的组合日度损失
        cons += [z >= 0, z >= port_loss - t]
        cvar = t + (1.0 / (alpha * R.shape[0])) * cp.sum(z)
        cons += [cvar <= cvar_limit]
    return cp.Problem(obj, cons), w

prob, w = build_problem(mu_d, Sigma_d, R, cvar_limit=0.037)
prob.solve(solver=cp.CLARABEL)
print("最优权重:", w.value)
```

注意 `cp.psd_wrap(Sigma_d)` 这个细节：数值上算出来的协方差矩阵有时会有微小的负特征值（浮点误差），`psd_wrap` 告诉 CVXPY"相信它是半正定的，别去校验"，避免求解器报非凸。这是实战里很容易踩的坑。

## 三、实验设置：一个带厚尾的合成市场

为了让约束的代价"看得见"，我构造了一个透明的因子模型，而不是拿一段真实行情糊弄过去——真实行情只有一个样本，你没法判断结果是约束的功劳还是行情的运气。

- **12 个资产，分 3 个行业**（科技 / 金融 / 消费，各 4 个）。
- **协方差**用因子结构生成：市场因子 + 3 个行业因子 + 特质波动。行业因子方差（5%）远高于特质方差（1.5%），意味着**同一个行业里的资产高度联动**——这正是"行业集中度会真实推高风险"的前提。
- **预期收益**：科技最高（~11%）、金融居中（~7%）、消费最低（~5%）。这诱使无约束优化器去重仓科技。
- **情景收益**：8000 条日度路径，从**多元 Student-t（自由度 4）** 抽样，刻意做出比正态更肥的尾部。因为 CVaR 度量的是尾部，只有情景本身有厚尾，集中度对 CVaR 的影响才会显现——用正态分布，所有组合的 CVaR 都差不多，故事就讲不响。

年化用 252 个交易日缩放，风险厌恶 $\lambda = 3$。下面把四种约束档位摆在一起对比（全部由上面的 `build_problem` 实跑得出）：

| 约束档位 | 科技仓位 | 年化收益 | 年化波动 | Sharpe | CVaR95(单日) |
|---|---|---|---|---|---|
| 无约束（可做空） | 57.8% | 11.00% | 20.96% | 0.525 | 3.93% |
| 纯多（仅 w≥0） | 57.4% | 9.61% | 20.72% | 0.464 | 3.87% |
| 个股25%+行业40% | 40.0% | 8.63% | 19.79% | 0.436 | 3.76% |
| CVaR95≤3.7%（无上限） | 46.6% | 8.70% | 19.62% | 0.444 | 3.70% |

![四种约束下的 12 个资产权重分配](/images/convex-optimization-portfolio-constraints/fig1_weights.png)

图 1 把每个资产在四档约束下的权重画了出来。无约束（灰）和纯多（橙）几乎把柱子全堆在科技那 4 根上；硬性上限（蓝）被 40% 的红色虚线死死压住，只能把多出来的仓位匀到金融和消费；而 CVaR 止损线（绿）没有碰任何一根权重天花板，却也自发地把科技从 57.4% 削到了 46.6%——它的收敛方向是"风险"，不是"权重长相"。

## 四、关键发现：止损线比硬性上限更聪明

把表里的尾部指标和收益指标单独拉出来看，结论更刺眼：

![硬性上限 vs CVaR 止损线：尾部更薄，Sharpe 反而更高](/images/convex-optimization-portfolio-constraints/fig2_cvar_sharpe.png)

**硬性上限那一组（蓝），CVaR 是 3.76%、Sharpe 是 0.436；CVaR 止损线那一组（绿），CVaR 反而更低（3.70%）、Sharpe 反而更高（0.444）。** 两者都在管尾部，为什么止损线赢了？

原因藏在行业配置里。硬性上限是个"形状约束"：它强制"每个行业都不能超过 40%"，但完全不关心你往哪个行业多放。结果优化器为了填满"科技被压下去腾出的空间"，把高达 **25.5%** 的仓位塞进了收益最低的消费（见图 3 最右的绿色块），等于用低收益资产去凑上限——既没把尾部压到最低，又白白牺牲了收益。

![各行业配置：止损线自动压低科技集中度](/images/convex-optimization-portfolio-constraints/fig3_sector.png)

而 CVaR 止损线是"目标约束"：它只说"单日 95% 损失不许超过 3.7%"，至于权重怎么分它不管。优化器于是可以自由地挑选那个真正尾部最薄、收益最高的分配——它把科技削到 46.6%、金融略增、消费只留 22.6%，比硬性上限的消费仓位少了 3 个百分点，却拿到了更低的尾部风险和更高的 Sharpe。**约束的语义对了，结果自然更优。**

一句话总结这个发现：**权重上限是尾部风险的代理变量（proxy），而 CVaR 约束是尾部风险本身。用代理去管，总会漏掉代理捕捉不到的跨资产联动；直接管本体，求解器才能给出最优解。**

## 五、什么时候该用哪一种约束

不是要你从此只用 CVaR 约束。两者各有合法战场：

1. **合规 / 合同约束必须用权重上限。** 基金契约里写的"单行业不超过 40%"是硬法律条款，CVaR 约束替代不了它。这时正确做法是权重上限 + CVaR 止损线**叠加**，前者满足合规，后者在合规框定的可行域里进一步压尾部。本文实验里这两者本就可以同时写进 `constraints` 列表。

2. **没有硬合规要求、只是自己想控制风险时，优先用风险型约束。** 它不会为了"形状好看"而强迫你持有一个你本来不想要的低收益资产。

3. **CVaR 约束的质量，取决于你的情景集 $R$。** 本文用 Student-t 造了厚尾，所以集中度能影响 CVaR。如果你的情景来自一段平静期的正态拟合，CVaR 约束和方差约束几乎等价，这时它额外带来的复杂度就不划算了。务必用**能反映极端情形的**情景（历史压力日、蒙特卡洛肥尾、或者把跳空事件显式加进去）去喂它。

4. **求解器选择。** 带 `quad_form` 的是 QP/SOCP，CLARABEL、OSQP 都行；叠加 CVaR 线性约束后仍是凸的，CLARABEL 一般最稳。资产数上千时，注意 `R @ w` 那一项会把问题规模撑大，可以考虑用因子化的协方差或者对情景做降采样。

## 六、完整可运行代码

下面是把全文实验串起来的最小可运行脚本（依赖 `numpy` 和 `cvxpy`，求解器用 CLARABEL）：

```python
import numpy as np
import cvxpy as cp

rng = np.random.default_rng(20261028)
N = 12
sectors = np.array([0,0,0,0,1,1,1,1,2,2,2,2])
sector_idx = [np.where(sectors == s)[0] for s in range(3)]

# 因子模型协方差（年化）
beta = np.clip(rng.normal(1.0, 0.2, N), 0.5, 1.5)
sload = np.clip(rng.normal(0.8, 0.15, N), 0.4, 1.2)
B = np.zeros((N, 4))
B[:, 0] = beta
for i, s in enumerate(sectors):
    B[i, s + 1] = sload[i]
fvar = np.array([0.03, 0.05, 0.05, 0.05])
Sigma = B @ np.diag(fvar) @ B.T + np.diag(np.full(N, 0.015))

mu = np.array([0.11,0.10,0.12,0.09, 0.075,0.07,0.08,0.065, 0.045,0.05,0.04,0.055]) \
     + rng.normal(0, 0.005, N)
Sd, md = Sigma / 252.0, mu / 252.0

# 厚尾情景：多元 Student-t（df=4）
S = 8000
L = np.linalg.cholesky(Sd)
z = rng.standard_normal((S, N)); g = rng.chisquare(4.0, S) / 4.0
R = md + (z / np.sqrt(g)[:, None]) @ L.T

def solve(long_only=True, name_cap=None, sector_cap=None, cvar_limit=None, gamma=3.0):
    w = cp.Variable(N)
    obj = cp.Maximize(md @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(Sd)))
    cons = [cp.sum(w) == 1]
    if long_only: cons += [w >= 0]
    if name_cap is not None: cons += [w <= name_cap]
    if sector_cap is not None:
        for idx in sector_idx: cons += [cp.sum(w[idx]) <= sector_cap]
    if cvar_limit is not None:
        t = cp.Variable(); zz = cp.Variable(S)
        pl = -R @ w
        cons += [zz >= 0, zz >= pl - t]
        cons += [t + (1.0/(0.05*S)) * cp.sum(zz) <= cvar_limit]
    cp.Problem(obj, cons).solve(solver=cp.CLARABEL)
    return w.value

def report(w):
    pl = -R @ w.value if hasattr(w, "value") else -R @ w
    w = np.asarray(w)
    ar, av = mu @ w, np.sqrt(w @ Sigma @ w)
    v = np.percentile(pl, 95); cvar = pl[pl >= v].mean()
    return ar, av, ar/av, cvar

for name, kw in [("无约束", dict(long_only=False)),
                 ("纯多", dict()),
                 ("硬上限", dict(name_cap=0.25, sector_cap=0.40)),
                 ("CVaR止损线", dict(cvar_limit=0.037))]:
    w = solve(**kw)
    ar, av, sh, cv = report(w)
    print(f"{name:8s} 收益={ar:.4f} 波动={av:.4f} Sharpe={sh:.3f} CVaR95={cv:.4f}")
```

跑出来正是正文那张表里的四行数字。把 `cvar_limit` 和 `sector_cap` 同时打开，你还能立刻得到"合规上限 + 尾部保险"的组合解——这就是把约束写进优化器的真正便利：**所有规则在同一帧里一起满足，而不是优化完再去打补丁。**

## 七、收尾

约束不是优化的敌人，而是让优化结果能落地的前提。本文用 CVXPY 证明了三件事：

1. 无约束均值-方差在真实因子结构下会把 57% 的仓位压在一个行业上，这是不可接受的风险暴露。
2. 硬性权重上限能管住形状，但它是尾部风险的代理，会为了凑形状而强迫持有低收益资产，反而牺牲 Sharpe。
3. 把 CVaR 止损线直接写进优化器，让求解器自己权衡收益与尾部，能在更低的尾部风险下拿到更高的 Sharpe——**当你没有硬合规约束时，优先用风险型约束。**

CVXPY 把这一切浓缩成往 `constraints` 列表里加几行。下一次你再想"先优化、再把超的砍掉"时，不妨停一下：那条规则，能不能直接写进优化器？
